# Copyright (c) 2026, EPFL / Swiss AI Initiative.

import pytest
import torch

import tools.checkpoint.merge as checkpoint_merge
from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from tests.unit_tests.test_utilities import Utils


def test_resolve_sources_from_one_root():
    assert checkpoint_merge.resolve_sources("checkpoints".split(), [1000, 2000, 3000]) == [
        "checkpoints/iter_0001000",
        "checkpoints/iter_0002000",
        "checkpoints/iter_0003000",
    ]


def test_resolve_sources_rejects_mismatched_roots():
    with pytest.raises(ValueError, match="one checkpoint root or one root per"):
        checkpoint_merge.resolve_sources(["one", "two"], [1000, 2000, 3000])


def test_mean_coefficients():
    assert checkpoint_merge.merge_coefficients(4) == [0.25] * 4


def test_wsm_linear_decay_coefficients_from_stable_schedule():
    coefficients = checkpoint_merge.merge_coefficients(
        5, method="linear-decay", target_end_multiplier=0.2
    )

    assert coefficients == pytest.approx([0.2] * 5)


def test_wsm_cancels_matching_original_linear_decay():
    coefficients = checkpoint_merge.merge_coefficients(
        5,
        method="linear-decay",
        original_schedule="linear-decay",
        original_end_multiplier=0.2,
        target_end_multiplier=0.2,
        checkpoint_steps=[0, 25, 50, 75, 100],
        original_decay_steps=100,
    )

    assert coefficients == pytest.approx([0.0, 0.0, 0.0, 0.0, 1.0])


def test_original_decay_uses_training_steps_not_checkpoint_count():
    coefficients = checkpoint_merge.merge_coefficients(
        3,
        method="linear-decay",
        original_schedule="linear-decay",
        original_end_multiplier=0.5,
        target_end_multiplier=0.5,
        checkpoint_steps=[0, 20, 40],
        original_decay_steps=100,
    )

    assert coefficients == pytest.approx([1 / 6, 5 / 24, 5 / 8])


def test_wsm_rejects_zero_original_multiplier():
    with pytest.raises(ValueError, match="reaches zero"):
        checkpoint_merge.merge_coefficients(
            5,
            method="linear-decay",
            original_schedule="linear-decay",
            original_end_multiplier=0.0,
            checkpoint_steps=[0, 25, 50, 75, 100],
            original_decay_steps=100,
        )


def test_muon_md_optimizer_gains_are_excluded():
    assert not checkpoint_merge._is_training_key("decoder.layers.0.mlp.weight")
    assert checkpoint_merge._is_training_key(
        "optimizer.state.row_gain.decoder.layers.0.mlp.weight"
    )
    assert checkpoint_merge._is_training_key(
        "optimizer.state.row_gain_m.decoder.layers.0.mlp.weight"
    )


def test_empty_workers_still_enter_load_collectives(monkeypatch):
    calls = []

    def fake_load(templates, source, validate_access_integrity):
        calls.append((templates, source, validate_access_integrity))
        return {"common": "state"}

    monkeypatch.setattr(checkpoint_merge.dist_checkpointing, "load", fake_load)

    assert checkpoint_merge._load_assigned("checkpoint", {}) == {}
    assert calls == [({}, "checkpoint", True)]


def test_owner_map_balances_largest_items_first():
    items = [("large", 8), ("medium", 5), ("small", 3)]

    owners = checkpoint_merge._owner_map(items, 2, lambda size: size)

    assert owners == {"large": 0, "medium": 1, "small": 1}


@pytest.fixture
def initialized_model_parallel():
    Utils.initialize_model_parallel(1, 1)
    yield
    Utils.destroy_model_parallel()


def test_merge_torch_dist_checkpoints(initialized_model_parallel, tmp_path_dist_ckpt):
    sources = []
    for name, value in (("merge_one", 1.0), ("merge_two", 5.0)):
        source = tmp_path_dist_ckpt / name
        source.mkdir()
        dist_checkpointing.save(
            {
                "weight": ShardedTensor.from_rank_offsets("weight", torch.full((4,), value)),
                "optimizer.state.row_gain.weight": ShardedTensor.from_rank_offsets(
                    "optimizer.state.row_gain.weight", torch.full((4,), 99.0)
                ),
                "iteration": 123,
                "rng_state": "excluded",
            },
            str(source),
        )
        sources.append(str(source))

    for output_name, coefficients, expected in (
        ("mean_merge", None, 3.0),
        ("linear_decay_merge", [0.75, 0.25], 2.0),
    ):
        output = tmp_path_dist_ckpt / output_name
        checkpoint_merge.merge_checkpoint_directories(sources, output, coefficients)
        release = output / "release"
        loaded = dist_checkpointing.load(
            {"weight": ShardedTensor.from_rank_offsets("weight", torch.empty(4))},
            str(release),
        )
        torch.testing.assert_close(loaded["weight"], torch.full((4,), expected))
        assert "optimizer.state.row_gain.weight" not in dist_checkpointing.load_tensors_metadata(
            str(release)
        )
        common = dist_checkpointing.load_common_state_dict(str(release))
        assert common["iteration"] == 0
        assert "rng_state" not in common
