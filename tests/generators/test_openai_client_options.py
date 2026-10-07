# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import httpx
import openai
import pytest

from garak.attempt import Conversation, Message, Turn
from garak.generators.openai import (
    OpenAICompatible,
    OpenAIGenerator,
    OpenAIResponsesGenerator,
)

URI = "http://client-options.test/v1"

CHAT_OK = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 0,
    "model": "m",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "done"},
        }
    ],
}


def _build(klass, **settings):
    config = {"api_key": "sk-test-fake-key", **settings}
    if klass is not OpenAIGenerator:
        config["uri"] = URI
    return klass(
        name="gpt-4o" if klass is OpenAIGenerator else "m",
        config_root={"generators": {"openai": {klass.__name__: config}}},
    )


@pytest.fixture
def created(monkeypatch):
    calls = []
    real_client = openai.OpenAI
    monkeypatch.setattr(
        "garak.generators.openai.openai.OpenAI",
        lambda **kwargs: calls.append(kwargs) or real_client(**kwargs),
    )
    return calls


@pytest.mark.parametrize(
    "klass", [OpenAICompatible, OpenAIGenerator, OpenAIResponsesGenerator]
)
def test_client_options_reach_the_sdk_client(klass, created):
    _build(klass, client_options={"timeout": 7.5})

    assert created[0]["timeout"] == 7.5, "client_options are passed to openai.OpenAI"


def test_client_options_override_the_retry_setting_they_share_a_key_with(created):
    _build(OpenAICompatible, max_retries=2, client_options={"max_retries": 4})

    assert created[0]["max_retries"] == 4, "an explicit client option wins"


def test_a_supplied_http_client_carries_the_requests():
    seen = []

    def handler(request):
        seen.append(request.url.path)
        return httpx.Response(200, json=CHAT_OK)

    generator = _build(
        OpenAICompatible,
        client_options={
            "http_client": httpx.Client(transport=httpx.MockTransport(handler))
        },
    )

    output = generator._call_model(Conversation([Turn("user", Message("hi"))]))

    assert output[0].text == "done"
    assert seen == [
        "/v1/chat/completions"
    ], "the request went through the supplied client"
