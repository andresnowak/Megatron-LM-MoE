# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os

import pytest
import torch

from megatron.core.optimizer import HAVE_EMERGING_OPTIMIZERS
from megatron.core.optimizer.md_decoupling import MDDecoupling
from megatron.core.optimizer.md_decoupling import _split_qkv
from megatron.core.process_groups_config import ProcessGroupCollection
from tests.unit_tests.test_utilities import Utils


requires_cuda_and_emerging = pytest.mark.skipif(
    not torch.cuda.is_available() or not HAVE_EMERGING_OPTIMIZERS,
    reason="CUDA and emerging_optimizers are required for MDDecoupling orthogonal updates",
)


class _NoProcessGroups:
    tp = None
    expt_tp = None


def _step_sum_loss(model, input_tensor):
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()


def _record_md_split_output(param, grad, **md_kwargs):
    optimizer = MDDecoupling(
        params=[param],
        lr=0.01,
        split_qkv=True,
        pg_collection=None,
        tp_mode="duplicated",
        **md_kwargs,
    )
    calls = []

    def record_split(split_grad, tp_group, partition_dim, flat_mode=False):
        del tp_group, partition_dim, flat_mode
        calls.append(split_grad.detach().clone())
        return torch.full_like(split_grad, float(len(calls)))

    optimizer._orthogonalize_tensor = record_split
    return optimizer._orthogonalize_param(
        param, grad, is_qkv=getattr(param, "is_qkv", False)
    ), calls


def _gqa_qkv_optimizer(param, **kwargs):
    return MDDecoupling(
        params=[param],
        lr=0.01,
        split_qkv=True,
        is_qkv_fn=lambda p: getattr(p, "is_qkv", False),
        qkv_split_shapes=(4, 2, 2),
        pg_collection=_NoProcessGroups(),
        tp_mode="duplicated",
        **kwargs,
    )


def _assert_qkv_split_flat_norms(optimizer, tensor, expected_norm):
    parts = _split_qkv(tensor, optimizer.qkv_split_shapes)
    expected = torch.full((len(parts),), expected_norm, dtype=tensor.dtype, device=tensor.device)
    actual = torch.stack([torch.linalg.vector_norm(part) for part in parts])
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def _assert_qkv_split_tangent(optimizer, param, grad):
    p_parts = _split_qkv(param, optimizer.qkv_split_shapes)
    g_parts = _split_qkv(grad, optimizer.qkv_split_shapes)
    residuals = torch.stack(
        [
            (p_part * g_part).sum().abs()
            / (torch.linalg.vector_norm(p_part) * torch.linalg.vector_norm(g_part)).clamp_min(1e-12)
            for p_part, g_part in zip(p_parts, g_parts)
        ]
    )
    torch.testing.assert_close(residuals, torch.zeros_like(residuals), rtol=1e-5, atol=1e-6)


def test_md_decoupling_router_gains_mode_override():
    param = torch.nn.Parameter(torch.ones(2, 2))
    param.is_router = True
    optimizer = MDDecoupling(
        params=[param],
        lr=0.01,
        hypersphere_gains_mode="rowcol",
        hypersphere_gains_mode_router="none",
        pg_collection=None,
    )

    assert optimizer._resolve_gains_mode(param) == "none"


def test_md_decoupling_direct_gains_no_clamp_min_round_trip():
    param = torch.nn.Parameter(torch.tensor([[2.0, -4.0], [6.0, -8.0]]))
    original = param.detach().clone()
    param.grad = torch.ones_like(param)
    optimizer = MDDecoupling(
        params=[param],
        lr=0.01,
        hypersphere_gains_mode="flat",
        gain_parametrization="direct",
        gains_no_clamp_min=True,
        pg_collection=None,
    )
    optimizer.state[param]["flat_gain"] = torch.tensor(-2.0)

    gain_grads = optimizer._preprocess_gains(param)

    torch.testing.assert_close(param, original / -2.0)
    torch.testing.assert_close(gain_grads["flat_gain"], torch.tensor(2.0))

    optimizer._apply_gains(param)

    torch.testing.assert_close(param, original)


@requires_cuda_and_emerging
def test_md_decoupling_qkv_split():
    qkv_size = 3 * 8 * 4
    hidden_size = 64
    qkv_split_shapes = (8, 8, 8)

    torch.manual_seed(42)
    input_tensor = torch.randn(8, hidden_size, dtype=torch.float32, device="cuda")

    model_split = torch.nn.Linear(
        hidden_size, qkv_size, bias=False, dtype=torch.float32, device="cuda"
    )
    model_no_split = torch.nn.Linear(
        hidden_size, qkv_size, bias=False, dtype=torch.float32, device="cuda"
    )
    model_split.weight.data.fill_(1.0)
    model_no_split.weight.data.copy_(model_split.weight.data)
    model_split.weight.is_qkv = True

    optimizer_split = MDDecoupling(
        params=[model_split.weight],
        lr=0.01,
        use_orthogonal_updates=True,
        split_qkv=True,
        is_qkv_fn=lambda p: getattr(p, "is_qkv", False),
        qkv_split_shapes=qkv_split_shapes,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )
    optimizer_no_split = MDDecoupling(
        params=[model_no_split.weight],
        lr=0.01,
        use_orthogonal_updates=True,
        split_qkv=False,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    original_weight = model_split.weight.data.clone()
    _step_sum_loss(model_split, input_tensor)
    optimizer_split.step()
    weight_with_split = model_split.weight.data.clone()

    _step_sum_loss(model_no_split, input_tensor)
    optimizer_no_split.step()
    weight_without_split = model_no_split.weight.data.clone()

    assert not torch.equal(weight_with_split, original_weight)
    assert not torch.equal(weight_without_split, original_weight)
    assert not torch.equal(weight_with_split, weight_without_split)


@requires_cuda_and_emerging
@pytest.mark.parametrize("tp_mode", ["duplicated", "blockwise", "distributed"])
def test_md_decoupling_different_tp_modes_single_rank(tp_mode):
    torch.manual_seed(42)
    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device="cuda")
    model.requires_grad_(True)
    model.weight.data.normal_(0, 0.02)

    optimizer = MDDecoupling(
        params=[model.weight],
        lr=0.01,
        weight_decay=0.0,
        use_orthogonal_updates=True,
        momentum_beta=0.95,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode=tp_mode,
    )

    torch.manual_seed(42)
    input_tensor = torch.randn(32, 100, dtype=torch.float32, device="cuda")
    original_weight = model.weight.data.clone()
    _step_sum_loss(model, input_tensor)
    optimizer.step()

    assert not torch.equal(model.weight.data, original_weight)


@requires_cuda_and_emerging
@pytest.mark.skipif(
    int(os.getenv("WORLD_SIZE", "1")) == 1, reason="Multi-rank test requires WORLD_SIZE > 1"
)
class TestMDDecouplingMultiRankTP:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        world = int(os.getenv("WORLD_SIZE", "1"))
        Utils.initialize_model_parallel(tensor_model_parallel_size=min(world, 2))
        yield
        Utils.destroy_model_parallel()

    def create_tp_model_and_optimizer(self, tp_mode):
        rank = int(os.getenv("RANK", "0"))
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        torch.manual_seed(42 + rank)
        model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device="cuda")
        model.requires_grad_(True)
        model.weight.data.normal_(0, 0.02)
        model.weight.partition_dim = 0

        optimizer = MDDecoupling(
            params=[model.weight],
            lr=0.01,
            weight_decay=0.0,
            use_orthogonal_updates=True,
            momentum_beta=0.95,
            num_ns_steps=5,
            pg_collection=pg_collection,
            tp_mode=tp_mode,
        )

        return model, optimizer

    @pytest.mark.parametrize("tp_mode", ["duplicated", "distributed"])
    def test_md_decoupling_modes_multirank_update(self, tp_mode):
        model, optimizer = self.create_tp_model_and_optimizer(tp_mode)

        torch.manual_seed(42)
        input_tensor = torch.randn(32, 100, dtype=torch.float32, device="cuda")
        original_weight = model.weight.data.clone()
        _step_sum_loss(model, input_tensor)
        optimizer.step()

        assert not torch.equal(model.weight.data, original_weight)

    def test_md_decoupling_blockwise_mode_multirank_update(self):
        model, optimizer = self.create_tp_model_and_optimizer("blockwise")

        torch.manual_seed(42)
        input_tensor = torch.randn(32, 100, dtype=torch.float32, device="cuda")
        original_weight = model.weight.data.clone()
        _step_sum_loss(model, input_tensor)
        optimizer.step()

        assert not torch.equal(model.weight.data, original_weight)


def test_md_decoupling_gqa_qkv_split_mechanics():
    param = torch.nn.Parameter(torch.empty(8, 4))
    param.is_qkv = True
    grad = torch.arange(32, dtype=torch.float32).view(8, 4)

    output, calls = _record_md_split_output(
        param,
        grad,
        is_qkv_fn=lambda p: getattr(p, "is_qkv", False),
        qkv_split_shapes=(4, 2, 2),
    )

    assert [call.shape for call in calls] == [
        torch.Size([4, 4]),
        torch.Size([2, 4]),
        torch.Size([2, 4]),
    ]
    expected = torch.tensor([1] * 4 + [2] * 2 + [3] * 2).view(8, 1)
    assert torch.equal(output, expected.expand_as(output))


def test_md_decoupling_gqa_split_flat_normalization_is_block_local():
    param = torch.nn.Parameter(torch.arange(1, 33, dtype=torch.float32).view(8, 4))
    param.is_qkv = True
    optimizer = _gqa_qkv_optimizer(
        param,
        hypersphere_mode="flat",
        hypersphere_preserve_init=True,
    )

    with torch.no_grad():
        optimizer._normalize(param, param, is_qkv=True)

    _assert_qkv_split_flat_norms(optimizer, param, expected_norm=2.0)


def test_md_decoupling_gqa_split_tangential_grad_is_block_local():
    param = torch.nn.Parameter(torch.arange(1, 33, dtype=torch.float32).view(8, 4))
    param.is_qkv = True
    grad = torch.arange(33, 65, dtype=torch.float32).view(8, 4)
    optimizer = _gqa_qkv_optimizer(
        param,
        hypersphere_mode="flat",
        hypersphere_tangential_grad=True,
        hypersphere_preserve_init=True,
    )

    with torch.no_grad():
        optimizer._project_tangent_inplace(param, grad, is_qkv=True)

    _assert_qkv_split_tangent(optimizer, param, grad)


def test_md_decoupling_gqa_split_row_normalization():
    param = torch.nn.Parameter(torch.arange(1, 33, dtype=torch.float32).view(8, 4))
    param.is_qkv = True
    optimizer = _gqa_qkv_optimizer(
        param,
        hypersphere_mode="row",
        hypersphere_preserve_init=True,
    )

    with torch.no_grad():
        optimizer._normalize(param, param, is_qkv=True)

    for part in _split_qkv(param, optimizer.qkv_split_shapes):
        row_norms = torch.linalg.vector_norm(part, dim=1)
        torch.testing.assert_close(row_norms, torch.ones_like(row_norms), rtol=1e-5, atol=1e-6)


@requires_cuda_and_emerging
@pytest.mark.parametrize("num_ns_steps", [5, 15, 25])
def test_md_decoupling_num_ns_steps(num_ns_steps):
    torch.manual_seed(42)
    model = torch.nn.Linear(80, 40, bias=False, dtype=torch.float32, device="cuda")
    model.weight.data.normal_(0, 0.02)

    optimizer = MDDecoupling(
        params=[model.weight],
        lr=0.01,
        use_orthogonal_updates=True,
        num_ns_steps=num_ns_steps,
        pg_collection=None,
        tp_mode="duplicated",
    )

    input_tensor = torch.randn(16, 80, dtype=torch.float32, device="cuda")
    original_weight = model.weight.data.clone()
    _step_sum_loss(model, input_tensor)
    optimizer.step()

    assert not torch.equal(model.weight.data, original_weight)
    assert optimizer.num_ns_steps == num_ns_steps


@requires_cuda_and_emerging
@pytest.mark.parametrize("use_nesterov", [True, False])
def test_md_decoupling_nesterov(use_nesterov):
    torch.manual_seed(42)
    model = torch.nn.Linear(80, 40, bias=False, dtype=torch.float32, device="cuda")
    model.weight.data.normal_(0, 0.02)

    optimizer = MDDecoupling(
        params=[model.weight],
        lr=0.01,
        use_orthogonal_updates=True,
        use_nesterov=use_nesterov,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    input_tensor = torch.randn(16, 80, dtype=torch.float32, device="cuda")
    original_weight = model.weight.data.clone()
    _step_sum_loss(model, input_tensor)
    optimizer.step()

    assert not torch.equal(model.weight.data, original_weight)
    assert optimizer.use_nesterov is use_nesterov
