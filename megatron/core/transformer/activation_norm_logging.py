# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Sampled activation RMS diagnostics without retaining activation tensors.

Only the scheduled forward-step callable enables recording. Backward checkpoint
recomputation therefore cannot add observations, including no-grad initial
checkpoint forwards. Each rerun starts with fresh accumulators.
"""

from contextlib import contextmanager
from functools import wraps
from typing import Callable, Dict, Iterator, Optional

import torch

METRICS = ("residual_input", "attention_output", "mlp_output")
# Bound FP32 workspace to about 1 MiB (or one sequence position if larger).
_CHUNK_ELEMENTS = 262144
_active = False
_in_forward = False
_acc: Dict[tuple[int, str], torch.Tensor] = {}


def should_log(step: int, interval: int = 0) -> bool:
    """Sample completed-step numbers, with zero disabling diagnostics."""
    if interval < 0:
        raise ValueError("activation-norm-log-interval must be nonnegative")
    return interval > 0 and step > 0 and step % interval == 0


def set_active(active: bool) -> None:
    """Start a sampled step, or release any stale diagnostics."""
    global _active, _in_forward
    _active = bool(active)
    _in_forward = False
    _acc.clear()


def is_active() -> bool:
    return _active


def reset_attempt() -> None:
    """Discard observations from a rejected forward/backward attempt."""
    _acc.clear()


@contextmanager
def forward_scope() -> Iterator[None]:
    """Enable observation only while executing a scheduled model forward."""
    global _in_forward
    previous = _in_forward
    _in_forward = _active
    try:
        yield
    finally:
        _in_forward = previous


def wrap_forward_step(forward_step: Callable) -> Callable:
    """Wrap the schedule's callback, preserving optional checkpoint arguments."""
    @wraps(forward_step)
    def wrapped(*args, **kwargs):
        with forward_scope():
            return forward_step(*args, **kwargs)

    return wrapped


def record(
    layer: int,
    metric: str,
    x: torch.Tensor,
    padding_mask: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    *,
    training: bool = True,
) -> None:
    """Accumulate squared sum and element count for [sequence, batch, hidden].

    ``padding_mask`` is already CP/SP-local, shaped [batch, sequence], with True
    for padding. Outputs are measured BEFORE sandwich norm/residual scaling/add,
    including the separate linear bias if present. FP32 copies are bounded and
    discarded immediately; neither inputs nor their autograd graphs are stored.
    """
    if not (_active and _in_forward and training):
        return
    if metric not in METRICS:
        raise ValueError(f"Unknown activation norm metric: {metric}")
    if x.ndim != 3:
        raise ValueError(f"Expected [sequence, batch, hidden], got {tuple(x.shape)}")
    if padding_mask is not None and tuple(padding_mask.shape) != (x.shape[1], x.shape[0]):
        raise ValueError("Activation norm padding mask must match the local batch/sequence")
    _accumulate(layer, metric, x, padding_mask, bias)


@torch.no_grad()
def _accumulate(
    layer: int,
    metric: str,
    x: torch.Tensor,
    padding_mask: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
) -> None:
    """Perform tensor work only after the cheap Python recording guards pass."""
    key = (layer, metric)
    acc = _acc.get(key)
    if acc is None:
        acc = torch.zeros(2, device=x.device, dtype=torch.float64)
        _acc[key] = acc
    if x.numel() == 0:
        return
    positions = max(1, _CHUNK_ELEMENTS // (x.shape[1] * x.shape[2]))
    for start in range(0, x.shape[0], positions):
        end = min(start + positions, x.shape[0])
        # copy=True is essential for FP32 inputs: in-place squaring must never
        # modify model activations, even when detach() shares their storage.
        values = x.detach()[start:end].to(dtype=torch.float32, copy=True)
        if bias is not None:
            values.add_(bias.detach().float())
        if padding_mask is not None:
            padding = padding_mask[:, start:end].transpose(0, 1).to(
                device=x.device, dtype=torch.bool
            )
            # Mask before squaring so NaNs in padding do not poison the sum.
            values.masked_fill_(padding.unsqueeze(-1), 0.0)
            count = (~padding).sum(dtype=torch.int64) * x.shape[-1]
        else:
            count = values.numel()
        values.square_()
        acc[0].add_(values.sum(dtype=torch.float64))
        acc[1].add_(count)


@torch.no_grad()
def collect(num_layers: int, device: torch.device) -> Dict[str, float]:
    """Pool scalar sums/counts across PP/DP/CP/TP ranks; call on every rank.

    Sequence shards contribute disjoint elements. Replicated TP observations
    multiply both sums and counts equally, leaving the RMS unchanged. Absent
    PP layers contribute zero. Only scalar diagnostics cross the network.
    """
    global _active, _in_forward
    _active = _in_forward = False
    totals = torch.zeros((num_layers, len(METRICS), 2), device=device, dtype=torch.float64)
    for (layer, metric), acc in _acc.items():
        if 0 <= layer < num_layers:
            totals[layer, METRICS.index(metric)] = acc
    _acc.clear()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
    totals = totals.cpu()
    stats = {}
    for layer in range(num_layers):
        for index, metric in enumerate(METRICS):
            square_sum, count = totals[layer, index].tolist()
            if count > 0:
                stats[f"activation_norm/{metric}_rms_layer_{layer}"] = (square_sum / count) ** 0.5
    return stats
