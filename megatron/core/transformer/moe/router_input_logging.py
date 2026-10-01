# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Router-input and pre-MLP RMSNorm gain diagnostics for MoE layers.

Splits a router logit blow-up into its input-side causes. For logit_i = W_i . x, where x is the
pre-MLP RMSNorm output (x_j = g_j * h_j / rms(h)), each logged step reports per layer:

- ``tok_norm_mean`` / ``tok_norm_max``: per-token ||x||. About sqrt(d) * rms(g) when nothing is
  unusual; a large max means some tokens carry a much larger input.
- ``x_absmax`` and ``x_absmax_channel``: the largest |x_j| over tokens and channels and its
  channel index (an outlier / massive-activation channel shows up as a stable index).
- ``gain_at_absmax`` / ``prenorm_at_absmax``: the RMSNorm gain g_j at that channel and the
  normalized pre-gain value |x_j| / |g_j|, separating "the gain is large" from "the residual
  stream concentrates in this channel".
- ``conc_max``: max over tokens of max_j |x_j| / ||x||, 1/sqrt(d) for a spread token and 1.0
  when one channel holds the whole input.
- ``lse_mean_sq`` (equals the logged z-loss), ``lse_max``, ``logit_max`` and
  ``frac_lse_gt_{5,10,20,40}``: whether a high z-loss comes from a few tokens or from all.
- ``rms_gain_{mean,std,min,max,absmax}``: the pre-MLP RMSNorm gain itself.

Statistics are accumulated on-device during the forward passes of a logged step (no host sync)
and reduced once over the world when collected: sums with SUM (tokens are disjoint across
DP/CP/SP ranks; ratios are unaffected by duplicated TP copies), extrema with MAX.
"""

from typing import Dict, Iterable, Optional

import torch

_LSE_THRESHOLDS = (5.0, 10.0, 20.0, 40.0)
# Accumulator layout per layer.
_SUM_TOK_NORM, _N_TOK, _SUM_LSE_SQ = 0, 1, 2
_CNT_LSE = 3  # .. 3 + len(_LSE_THRESHOLDS)
_N_SUM = _CNT_LSE + len(_LSE_THRESHOLDS)
_MAX_TOK_NORM, _MAX_ABS, _MAX_CONC, _MAX_LSE, _MAX_LOGIT = range(_N_SUM, _N_SUM + 5)
_ARGMAX_CHANNEL = _N_SUM + 5
_N_FIELDS = _ARGMAX_CHANNEL + 1

_active = False
_layers: Optional[frozenset] = None
_acc: Dict[int, torch.Tensor] = {}


def set_active(active: bool, layers: Optional[Iterable[int]] = None) -> None:
    """Enable or disable recording for the forward passes of the current step."""
    global _active, _layers
    _active = bool(active)
    _layers = frozenset(int(layer) for layer in layers) if layers else None
    if not _active:
        _acc.clear()


def is_active() -> bool:
    return _active


@torch.no_grad()
def record(
    layer: int, x: torch.Tensor, logits: torch.Tensor, padding_mask: Optional[torch.Tensor]
) -> None:
    """Accumulate statistics for one router forward. ``layer`` is the 0-based global index."""
    if not _active or (_layers is not None and layer not in _layers):
        return
    x = x.detach().reshape(-1, x.shape[-1]).float()
    logits = logits.detach().reshape(-1, logits.shape[-1]).float()
    valid = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
    if padding_mask is not None:
        valid = ~padding_mask.reshape(-1).to(device=x.device, dtype=torch.bool)
    validf = valid.float()

    tok_norm = x.norm(dim=1) * validf
    tok_absmax, tok_channel = (x.abs() * validf.unsqueeze(1)).max(dim=1)
    conc = tok_absmax / tok_norm.clamp_min(1e-12)
    lse = torch.logsumexp(logits, dim=-1)
    lse_valid = torch.where(valid, lse, torch.full_like(lse, float("-inf")))
    logit_max = torch.where(valid.unsqueeze(1), logits, torch.full_like(logits, float("-inf"))).max()
    best_token = tok_absmax.argmax()

    acc = _acc.get(layer)
    if acc is None:
        acc = torch.zeros(_N_FIELDS, dtype=torch.float64, device=x.device)
        acc[_N_SUM:_ARGMAX_CHANNEL] = float("-inf")
        acc[_ARGMAX_CHANNEL] = -1.0
        _acc[layer] = acc

    sums = [tok_norm.sum(), validf.sum(), (lse.square() * validf).sum()]
    sums += [((lse > t) & valid).sum() for t in _LSE_THRESHOLDS]
    acc[:_N_SUM] += torch.stack([s.to(torch.float64) for s in sums])

    new_abs = tok_absmax[best_token].to(torch.float64)
    acc[_ARGMAX_CHANNEL] = torch.where(
        new_abs > acc[_MAX_ABS], tok_channel[best_token].to(torch.float64), acc[_ARGMAX_CHANNEL]
    )
    maxima = torch.stack(
        [tok_norm.max(), tok_absmax.max(), (conc * validf).max(), lse_valid.max(), logit_max]
    ).to(torch.float64)
    acc[_MAX_TOK_NORM:_ARGMAX_CHANNEL] = torch.maximum(acc[_MAX_TOK_NORM:_ARGMAX_CHANNEL], maxima)


def _pre_mlp_norm_weights(model, zero_centered_gamma: bool) -> Dict[int, torch.Tensor]:
    """0-based global layer index -> effective pre-MLP RMSNorm gain, for layers on this rank."""
    chunks = model if isinstance(model, (list, tuple)) else [model]
    gains = {}
    for chunk in chunks:
        for module in chunk.modules():
            norm = getattr(module, "pre_mlp_layernorm", None)
            layer_number = getattr(module, "layer_number", None)
            weight = getattr(norm, "weight", None)
            if layer_number is None or not isinstance(weight, torch.Tensor):
                continue
            gain = weight.detach().float()
            if zero_centered_gamma:
                gain = gain + 1.0
            gains[int(layer_number) - 1] = gain
    return gains


@torch.no_grad()
def collect(model, num_layers: int, zero_centered_gamma: bool = False) -> Dict[str, float]:
    """Reduce the step's statistics over the world. Must be called on every rank."""
    global _active
    _active = False
    gains = {
        layer: gain
        for layer, gain in _pre_mlp_norm_weights(model, zero_centered_gamma).items()
        if 0 <= layer < num_layers and (_layers is None or layer in _layers)
    }
    device = (
        next(iter(_acc.values())).device
        if _acc
        else next(iter(gains.values())).device
        if gains
        else torch.device("cuda", torch.cuda.current_device())
    )
    sums = torch.zeros((num_layers, _N_SUM), dtype=torch.float64, device=device)
    maxima = torch.full((num_layers, 5), float("-inf"), dtype=torch.float64, device=device)
    channel = torch.full((num_layers,), -1.0, dtype=torch.float64, device=device)
    for layer, acc in _acc.items():
        if 0 <= layer < num_layers:
            sums[layer] = acc[:_N_SUM]
            maxima[layer] = acc[_MAX_TOK_NORM:_ARGMAX_CHANNEL]
            channel[layer] = acc[_ARGMAX_CHANNEL]
    _acc.clear()

    # Gain stats: [mean, std, -min, max, absmax, present]; copies are identical, so MAX merges.
    gain_stats = torch.full((num_layers, 6), float("-inf"), dtype=torch.float64, device=device)
    for layer, gain in gains.items():
        gain_stats[layer] = torch.stack(
            [gain.mean(), gain.std(), -gain.min(), gain.max(), gain.abs().max(),
             torch.ones((), device=gain.device)]
        ).to(device=device, dtype=torch.float64)

    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    if distributed:
        local_max_abs = maxima[:, 1].clone()
        torch.distributed.all_reduce(sums, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(maxima, op=torch.distributed.ReduceOp.MAX)
        torch.distributed.all_reduce(gain_stats, op=torch.distributed.ReduceOp.MAX)
        # Channel of the global |x| maximum: only the rank(s) that hold it report their index.
        channel = torch.where(local_max_abs == maxima[:, 1], channel, torch.full_like(channel, -1))
        torch.distributed.all_reduce(channel, op=torch.distributed.ReduceOp.MAX)

    # Gain at that channel, read by the ranks that own the layer.
    gain_at = torch.full((num_layers,), float("-inf"), dtype=torch.float64, device=device)
    for layer, gain in gains.items():
        index = int(channel[layer].item())
        if 0 <= index < gain.numel():
            gain_at[layer] = gain[index].to(device=device, dtype=torch.float64)
    if distributed:
        torch.distributed.all_reduce(gain_at, op=torch.distributed.ReduceOp.MAX)

    sums, maxima, channel = sums.cpu(), maxima.cpu(), channel.cpu()
    gain_stats, gain_at = gain_stats.cpu(), gain_at.cpu()
    out: Dict[str, float] = {}
    for layer in range(num_layers):
        if sums[layer, _N_TOK] > 0:
            n = float(sums[layer, _N_TOK])
            prefix = "router_input/"
            out[f"{prefix}tok_norm_mean_layer_{layer}"] = float(sums[layer, _SUM_TOK_NORM]) / n
            out[f"{prefix}lse_mean_sq_layer_{layer}"] = float(sums[layer, _SUM_LSE_SQ]) / n
            for k, t in enumerate(_LSE_THRESHOLDS):
                out[f"{prefix}frac_lse_gt_{int(t)}_layer_{layer}"] = (
                    float(sums[layer, _CNT_LSE + k]) / n
                )
            for k, name in enumerate(("tok_norm_max", "x_absmax", "conc_max", "lse_max", "logit_max")):
                out[f"{prefix}{name}_layer_{layer}"] = float(maxima[layer, k])
            out[f"{prefix}x_absmax_channel_layer_{layer}"] = float(channel[layer])
            g = float(gain_at[layer])
            if g != float("-inf"):
                out[f"{prefix}gain_at_absmax_layer_{layer}"] = g
                if g != 0.0:
                    out[f"{prefix}prenorm_at_absmax_layer_{layer}"] = float(maxima[layer, 1]) / abs(g)
        if gain_stats[layer, 5] > 0:
            mean, std, neg_min, gmax, absmax = (float(v) for v in gain_stats[layer, :5])
            out[f"rms_gain/mean_layer_{layer}"] = mean
            out[f"rms_gain/std_layer_{layer}"] = std
            out[f"rms_gain/min_layer_{layer}"] = -neg_min
            out[f"rms_gain/max_layer_{layer}"] = gmax
            out[f"rms_gain/absmax_layer_{layer}"] = absmax
    return out
