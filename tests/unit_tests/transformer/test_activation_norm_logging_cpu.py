# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Standalone CPU tests, avoiding Megatron/TE/CUDA imports.

Run: python tests/unit_tests/transformer/test_activation_norm_logging_cpu.py
"""

import importlib.util
import gc
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import weakref

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "activation_norm_logging",
    ROOT / "megatron/core/transformer/activation_norm_logging.py",
)
logging = importlib.util.module_from_spec(spec)
spec.loader.exec_module(logging)


def distributed_worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        logging.set_active(True)
        with logging.forward_scope():
            # Unequal token counts must produce sqrt(7), not mean([1, 5]).
            x = torch.full((3 if rank == 0 else 1, 1, 2), 1.0 if rank == 0 else 5.0)
            logging.record(0, "residual_input", x)
            # Simulate a layer owned only by one PP stage.
            if rank == 1:
                logging.record(1, "mlp_output", torch.full((2, 1, 2), 7.0))
        stats = logging.collect(2, torch.device("cpu"))
        assert abs(stats["activation_norm/residual_input_rms_layer_0"] - 7.0 ** 0.5) < 1e-6
        assert stats["activation_norm/mlp_output_rms_layer_1"] == 7.0
    finally:
        dist.destroy_process_group()


class TestActivationNormLogging(unittest.TestCase):
    def tearDown(self):
        logging.set_active(False)

    def collect(self, layers=2):
        return logging.collect(layers, torch.device("cpu"))

    def test_interval(self):
        self.assertFalse(any(logging.should_log(s) for s in range(1, 601)))
        self.assertEqual([s for s in range(1, 601) if logging.should_log(s, 200)], [200, 400, 600])
        self.assertFalse(logging.should_log(200, 0))
        with self.assertRaises(ValueError):
            logging.should_log(200, -1)

    def test_no_work_when_inactive(self):
        logging.set_active(False)
        # Even an object that cannot be inspected must be ignored off-interval.
        with logging.forward_scope():
            logging.record(0, "residual_input", object())
        self.assertFalse(logging._acc)

    def test_weighted_rms_not_average_of_microbatch_norms(self):
        logging.set_active(True)
        with logging.forward_scope():
            logging.record(0, "residual_input", torch.ones(3, 1, 2))
            logging.record(0, "residual_input", torch.full((1, 1, 2), 5.0))
        self.assertAlmostEqual(self.collect()["activation_norm/residual_input_rms_layer_0"], 7 ** 0.5)

    def test_padding_orientation_bias_and_input_unchanged(self):
        x = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3).requires_grad_()
        original = x.detach().clone()
        padding = torch.tensor([[False, True, False, False], [True, False, False, True]])
        bias = torch.tensor([0.5, -1.0, 2.0])
        expected = (x.detach().double() + bias.double())[~padding.T].square().mean().sqrt().item()
        logging.set_active(True)
        with patch.object(logging, "_CHUNK_ELEMENTS", 6), logging.forward_scope():
            logging.record(0, "attention_output", x, padding, bias)
        self.assertEqual(logging._acc[(0, "attention_output")].numel(), 2)
        self.assertIsNone(logging._acc[(0, "attention_output")].grad_fn)
        torch.testing.assert_close(x.detach(), original, rtol=0, atol=0)
        self.assertAlmostEqual(self.collect()["activation_norm/attention_output_rms_layer_0"], expected)
        x.sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x))

    def test_all_padding_and_empty(self):
        logging.set_active(True)
        with logging.forward_scope():
            logging.record(0, "mlp_output", torch.full((2, 1, 4), float("nan")), torch.ones(1, 2, dtype=torch.bool))
            logging.record(1, "residual_input", torch.empty(0, 1, 4))
        self.assertEqual(self.collect(), {})

    def test_accumulator_does_not_retain_activation_storage(self):
        logging.set_active(True)
        x = torch.ones(2, 1, 4, requires_grad=True)
        ref = weakref.ref(x)
        with logging.forward_scope():
            logging.record(0, "mlp_output", x)
        del x
        gc.collect()
        self.assertIsNone(ref())
        self.assertEqual(self.collect()["activation_norm/mlp_output_rms_layer_0"], 1.0)

    def test_uniform_tp_replication_leaves_rms_unchanged(self):
        x = torch.arange(12, dtype=torch.float32).reshape(3, 1, 4)
        results = []
        for replicas in (1, 2, 4):
            logging.set_active(True)
            with logging.forward_scope():
                for _ in range(replicas):
                    logging.record(0, "residual_input", x)
            results.append(self.collect()["activation_norm/residual_input_rms_layer_0"])
        self.assertEqual(results, [results[0]] * 3)

    def test_checkpoint_forward_records_but_backward_recompute_does_not(self):
        logging.set_active(True)
        x = torch.tensor([[[2.0, 4.0]]], requires_grad=True)

        def block(tensor):
            logging.record(0, "mlp_output", tensor)
            return tensor.square()

        def forward(tensor):
            from torch.utils.checkpoint import checkpoint
            return checkpoint(block, tensor, use_reentrant=True)

        output = logging.wrap_forward_step(forward)(x)
        output.sum().backward()
        self.assertEqual(logging._acc[(0, "mlp_output")][1].item(), 2)
        self.assertAlmostEqual(self.collect()["activation_norm/mlp_output_rms_layer_0"], 10 ** 0.5)
        torch.testing.assert_close(x.grad, 2 * x.detach())

    def test_rerun_discards_rejected_attempt(self):
        logging.set_active(True)
        with logging.forward_scope():
            logging.record(0, "residual_input", torch.full((2, 1, 4), float("nan")))
        logging.reset_attempt()
        with logging.forward_scope():
            logging.record(0, "residual_input", torch.full((2, 1, 4), 3.0))
        self.assertEqual(self.collect()["activation_norm/residual_input_rms_layer_0"], 3.0)

    def test_scope_restored_on_exception_and_eval_is_ignored(self):
        logging.set_active(True)
        with self.assertRaises(RuntimeError), logging.forward_scope():
            raise RuntimeError("test")
        logging.record(0, "residual_input", object())
        with logging.forward_scope():
            logging.record(0, "residual_input", object(), training=False)
        self.assertFalse(logging._acc)

    def test_noncontiguous_input_and_bf16(self):
        x = torch.arange(24, dtype=torch.bfloat16).reshape(2, 4, 3).transpose(0, 1)
        logging.set_active(True)
        with patch.object(logging, "_CHUNK_ELEMENTS", 6), logging.forward_scope():
            logging.record(0, "residual_input", x)
        expected = x.double().square().mean().sqrt().item()
        self.assertAlmostEqual(self.collect()["activation_norm/residual_input_rms_layer_0"], expected)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo unavailable")
    def test_distributed_unequal_shards_and_pp_ownership(self):
        with tempfile.TemporaryDirectory() as tmp:
            rendezvous = Path(tmp, "rendezvous").resolve().as_uri()
            mp.spawn(distributed_worker, args=(rendezvous,), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
