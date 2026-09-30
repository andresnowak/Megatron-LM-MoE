# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import dataclasses
from argparse import ArgumentParser
from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.config import AsyncScheduleMode, InferenceConfig
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.argument_utils import inference_cfg_from_args
from megatron.training.arguments import _add_inference_args
from megatron.training.config.inference_config import InferenceSetupConfig


class TestInferenceConfig:
    def test_mutual_exclusivity_with_transformer_config(self):
        """
        Ensure mutual exclusivity between fields in `InferenceConfig` and
        `TransformerConfig`.
        """
        dynamic_inference_config_fields = set(dataclasses.fields(InferenceConfig))
        transformer_config_fields = set(dataclasses.fields(TransformerConfig))
        assert len(dynamic_inference_config_fields.intersection(transformer_config_fields)) == 0

    def test_verbose_is_init_only_and_not_serialized_or_compared(self):
        quiet = InferenceConfig()
        verbose = InferenceConfig(verbose=True)

        assert verbose._verbose is True
        assert quiet._verbose is False
        field_names = {field.name for field in dataclasses.fields(InferenceConfig)}
        assert "verbose" not in field_names
        assert "_verbose" not in field_names
        assert "verbose" not in dataclasses.asdict(verbose)
        assert quiet == verbose

    def test_setup_config_passes_verbose_to_runtime_config(self):
        model = SimpleNamespace(
            position_embedding_type="rotary",
            max_sequence_length=2560,
            pg_collection=None,
            config=SimpleNamespace(params_dtype=torch.float16),
            decoder=SimpleNamespace(layer_type_list=None, layers=[]),
        )

        runtime_config = InferenceSetupConfig().to_inference_config(model, verbose=True)

        assert runtime_config._verbose is True

    def test_async_sched_mode_default_and_coercion(self):
        assert InferenceConfig().async_sched_mode == AsyncScheduleMode.LEGACY
        assert (
            InferenceConfig(async_sched_mode="serial").async_sched_mode == AsyncScheduleMode.SERIAL
        )
        assert InferenceConfig(async_sched_mode=AsyncScheduleMode.SERIAL).async_sched_mode == (
            AsyncScheduleMode.SERIAL
        )

    def test_async_sched_mode_rejects_invalid_value(self):
        with pytest.raises(ValueError):
            InferenceConfig(async_sched_mode="invalid")

    def test_async_sched_cli_defaults_to_legacy_and_accepts_serial(self):
        parser = _add_inference_args(ArgumentParser())
        assert parser.parse_args([]).inference_dynamic_batching_async_sched_mode == "legacy"
        args = parser.parse_args(["--inference-dynamic-batching-async-sched-mode", "serial"])
        assert args.inference_dynamic_batching_async_sched_mode == "serial"
        assert inference_cfg_from_args(args).inference_dynamic_batching_async_sched_mode == "serial"

    def test_inference_setup_config_maps_async_sched_mode(self):
        model = SimpleNamespace(
            position_embedding_type="rotary",
            max_sequence_length=2560,
            pg_collection=None,
            config=SimpleNamespace(params_dtype=torch.float16),
            decoder=SimpleNamespace(layer_type_list=None, layers=[]),
        )
        setup_config = InferenceSetupConfig(inference_dynamic_batching_async_sched_mode="serial")

        runtime_config = setup_config.to_inference_config(model, verbose=False)

        assert runtime_config.async_sched_mode == AsyncScheduleMode.SERIAL
