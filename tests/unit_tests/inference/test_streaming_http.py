# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json

import pytest

quart = pytest.importorskip("quart")

from megatron.core.inference.async_stream import AsyncStream
from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints import (
    chat_completions,
    completions,
)
from tests.unit_tests.inference.test_openai_streaming import _make_byte_level_fast_tokenizer


class _Tokenizer:
    def __init__(self):
        self._tokenizer = _make_byte_level_fast_tokenizer()._tokenizer

    def tokenize(self, text):
        return self._tokenizer.tokenizer.encode(text, add_special_tokens=False)

    def detokenize(self, tokens):
        return self._tokenizer.tokenizer.decode(tokens)


class _Client:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.requests = []

    def add_request_streaming(self, prompt, sampling_params):
        self.requests.append((prompt, sampling_params))
        stream = AsyncStream(request_id=len(self.requests), cancel=lambda: None)
        tokens = self.tokenizer.tokenize("ab")
        stream.put({"partial": {"new_tokens": tokens[:1], "new_log_probs": [-0.1]}})
        stream.put(
            {
                "final": {
                    "prompt_tokens": prompt,
                    "generated_tokens": tokens,
                    "generated_log_probs": [-0.1, -0.2],
                    "sampling_params": {"num_tokens_to_generate": 2},
                }
            }
        )
        stream.finish()
        return stream


@pytest.mark.asyncio
@pytest.mark.parametrize("chat", [False, True])
async def test_http_streams_deltas_and_usage_through_real_endpoint(chat):
    tokenizer = _Tokenizer()
    client = _Client(tokenizer)
    app = quart.Quart(__name__)
    app.config.update(client=client, tokenizer=tokenizer, parsers=None, verbose=False)
    app.register_blueprint(chat_completions.bp if chat else completions.bp)
    payload = {
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": 2,
        "streaming_interval": 2,
    }
    if chat:
        payload.update(messages=[{"role": "user", "content": "hi"}], n=2)
    else:
        payload["prompt"] = ["hi", "hello"]

    response = await app.test_client().post(
        "/v1/chat/completions" if chat else "/v1/completions", json=payload
    )
    assert response.status_code == 200
    assert response.content_type == "text/event-stream"
    body = await response.get_data(as_text=True)
    records = [record for record in body.split("\n\n") if record]
    assert records[-1] == "data: [DONE]"
    chunks = [json.loads(record.removeprefix("data: ")) for record in records[:-1]]
    terminal = [
        choice for chunk in chunks for choice in chunk["choices"] if choice.get("finish_reason")
    ]
    assert len(terminal) == 2
    assert [choice["generated_text"] for choice in terminal] == ["ab", "ab"]
    assert all(params.streaming_interval == 2 for _, params in client.requests)
    prompt_tokens = (
        len(client.requests[0][0]) if chat else sum(len(prompt) for prompt, _ in client.requests)
    )
    assert chunks[-1]["usage"]["prompt_tokens"] == prompt_tokens
    assert chunks[-1]["usage"]["completion_tokens"] == 4
    assert chunks[-1]["usage"]["total_tokens"] == prompt_tokens + 4
