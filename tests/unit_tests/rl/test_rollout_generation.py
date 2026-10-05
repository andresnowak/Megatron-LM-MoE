# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from unittest.mock import AsyncMock, MagicMock

import pytest

from megatron.rl.inference import InferenceRequest, LLMChatMessage
from megatron.rl.inference.megatron import MegatronLocal


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "temperature, expected_temperature", [(None, 1.0), (0.0, 0.0)], ids=["default", "greedy"]
)
async def test_megatron_local_preserves_explicit_greedy_temperature(
    monkeypatch, temperature, expected_temperature
):
    monkeypatch.setattr("megatron.rl.inference.megatron.get_args", lambda: MagicMock())
    monkeypatch.setattr("megatron.rl.inference.megatron.get_tokenizer", lambda: MagicMock(bos=None))

    choice = MagicMock(finish_reason="stop")
    choice.message.model_dump.return_value = {"role": "assistant", "content": "response"}
    choice.raw_text = "response"
    choice.prompt_token_ids = [1]
    choice.generation_token_ids = [2]
    choice.generation_log_probs = [0.0]
    choice.policy_epoch = [(0, 0)]
    choice.kv_cache_epoch = [(0, 0)]
    choice.num_evictions = 0

    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=MagicMock(id="completion-id", choices=[choice])
    )
    server = MegatronLocal(host="localhost", port=0)
    server._openai_client = client
    request = InferenceRequest(
        prompt=[LLMChatMessage(role="user", content="prompt")],
        generation_args={"temperature": temperature},
    )

    await server.base_generate(request)

    client.chat.completions.create.assert_awaited_once()
    assert client.chat.completions.create.await_args.kwargs["temperature"] == expected_temperature
