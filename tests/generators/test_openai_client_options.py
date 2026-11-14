# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``client_options`` reach the SDK client the OpenAI generators build."""

import json

import httpx
import pytest
import respx

from garak import cli
from garak.attempt import Conversation, Message, Turn
from garak.generators.openai import (
    OpenAICompatible,
    OpenAIGenerator,
    OpenAIReasoningGenerator,
    OpenAIResponsesGenerator,
)

URI = "http://client-options.test/v1"
API_KEY = "sentinel-client-options-api-key-must-not-be-kept"
HEADER_CREDENTIAL = "sentinel-client-options-header-must-not-be-kept"

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
KEYED_ONLY = (OpenAIGenerator, OpenAIReasoningGenerator)


def _build(klass, **settings):
    config = {"api_key": API_KEY, **settings}
    if klass not in KEYED_ONLY:
        config["uri"] = URI
    return klass(
        name="gpt-4o" if klass in KEYED_ONLY else "m",
        config_root={"generators": {"openai": {klass.__name__: config}}},
    )


@pytest.mark.parametrize(
    "klass",
    [
        OpenAICompatible,
        OpenAIGenerator,
        OpenAIReasoningGenerator,
        OpenAIResponsesGenerator,
    ],
)
def test_client_options_reach_the_sdk_client(klass):
    generator = _build(klass, client_options={"timeout": 7.5})

    assert generator.client.timeout == 7.5, "client_options are passed to the client"


def test_client_options_override_the_retry_setting_they_share_a_key_with():
    generator = _build(
        OpenAICompatible, max_retries=2, client_options={"max_retries": 4}
    )

    assert generator.client.max_retries == 4, "an explicit client option wins"


def test_a_timeout_and_bounded_retries_set_together_both_reach_the_client():
    generator = _build(
        OpenAIResponsesGenerator,
        max_retries=0,
        client_options={"timeout": 180.0},
    )

    assert generator.client.timeout == 180.0, "the timeout reaches the client"
    assert generator.client.max_retries == 0, "and so does the retry setting"


def test_a_supplied_http_client_carries_the_requests():
    seen = []

    def handler(request):
        seen.append(request.url.path)
        return httpx.Response(200, json=CHAT_OK)

    generator = _build(
        OpenAICompatible,
        max_retries=0,  # a client that goes unused fails once, not for ever
        client_options={
            "http_client": httpx.Client(transport=httpx.MockTransport(handler))
        },
    )

    output = generator._call_model(Conversation([Turn("user", Message("hi"))]))

    assert output[0].text == "done"
    assert seen == [
        "/v1/chat/completions"
    ], "the request went through the supplied client"


def test_a_credential_in_client_options_is_scrubbed_from_a_failed_generation():
    generator = _build(
        OpenAICompatible,
        max_retries=0,
        client_options={"default_headers": {"Authorization": HEADER_CREDENTIAL}},
    )
    echo = {"error": {"message": f"unavailable for {HEADER_CREDENTIAL}"}}

    with respx.mock(base_url=URI) as router:
        router.post("/chat/completions").mock(
            return_value=httpx.Response(503, json=echo)
        )
        output = generator._call_model(Conversation([Turn("user", Message("hi"))]))

    error = output[0].notes["error"]
    assert "unavailable" in error["body"], "the reason is kept"
    assert cli.REDACTED in error["body"], "and the gap is marked"
    assert HEADER_CREDENTIAL not in json.dumps(
        output[0].notes
    ), "the note holds no secret"
