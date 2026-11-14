# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded retries of the OpenAI-compatible generators.

With ``max_retries`` set, a call sends one request per attempt and returns the
last failure as a failed generation once the retries are spent.
"""

import json
import logging

import httpx
import openai
import pytest
import respx

from garak import cli
from garak.attempt import Conversation, Message, Turn
from garak.generators.azure import AzureOpenAIGenerator
from garak.generators.groq import GroqChat
from garak.generators.nim import NVOpenAIChat, NVOpenAICompletion
from garak.generators.openai import OpenAICompatible, OpenAIResponsesGenerator

URI = "http://retry.test/v1"
API_KEY = "sentinel-retry-api-key-must-not-be-kept"
TOOL_AUTHORIZATION = "sentinel-retry-tool-authorization-must-not-be-kept"
SECRETS = (API_KEY, TOOL_AUTHORIZATION)
TOOLS = [
    {
        "type": "mcp",
        "server_label": "bank",
        "server_url": "http://tools.test/sse",
        "authorization": TOOL_AUTHORIZATION,
    }
]

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
RESPONSES_OK = {
    "id": "resp_1",
    "object": "response",
    "created_at": 0,
    "model": "m",
    "status": "completed",
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
    "output": [
        {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "done", "annotations": []}],
        }
    ],
}

# generator class, request path, success body
FLAVOURS = {
    "chat": (OpenAICompatible, "/chat/completions", CHAT_OK),
    "responses": (OpenAIResponsesGenerator, "/responses", RESPONSES_OK),
}


def _server_error(message="unavailable"):
    return httpx.Response(503, json={"error": {"message": message}})


def _build(flavour, **settings):
    klass, _, _ = FLAVOURS[flavour]
    config = {"uri": URI, "api_key": API_KEY, **settings}
    return klass(
        name="m",
        config_root={"generators": {"openai": {klass.__name__: config}}},
    )


def _conversation(notes=None):
    return Conversation([Turn("user", Message("hello"))], notes=notes)


def _route(mock, flavour, effects):
    _, path, ok_body = FLAVOURS[flavour]
    effects = [
        httpx.Response(200, json=ok_body) if effect == "ok" else effect
        for effect in effects
    ]
    return mock.post(path).mock(side_effect=effects)


@pytest.fixture
def sleeps(monkeypatch):
    recorded = []
    monkeypatch.setattr("garak.generators.openai.time.sleep", recorded.append)
    return recorded


@pytest.fixture
def mock():
    with respx.mock(base_url=URI, assert_all_called=False) as router:
        yield router


@pytest.mark.parametrize("flavour", FLAVOURS)
@pytest.mark.parametrize(
    "settings, expected",
    [({}, openai.DEFAULT_MAX_RETRIES), ({"max_retries": 2}, 0)],
    ids=["unset", "set"],
)
def test_sdk_retries_are_off_only_when_the_generator_owns_retrying(
    flavour, settings, expected
):
    generator = _build(flavour, **settings)

    assert generator.client.max_retries == expected, "SDK retries follow max_retries"


@pytest.mark.parametrize(
    "klass, family, settings",
    [
        (NVOpenAIChat, "nim", {}),
        (NVOpenAICompletion, "nim", {}),
        (GroqChat, "groq", {}),
        (
            AzureOpenAIGenerator,
            "azure",
            {"target_name": "gpt-4o", "uri": "https://retry.test/"},
        ),
    ],
)
def test_generators_that_build_their_own_client_switch_sdk_retries_off_too(
    klass, family, settings
):
    config = {"api_key": API_KEY, "max_retries": 2, **settings}
    generator = klass(
        name="m", config_root={"generators": {family: {klass.__name__: config}}}
    )

    assert generator.client.max_retries == 0, "the subclass's client has no SDK retries"


@pytest.mark.parametrize("flavour", FLAVOURS)
def test_max_retries_retries_a_server_error_once_after_the_delay(flavour, mock, sleeps):
    route = _route(mock, flavour, [_server_error(), "ok"])
    generator = _build(flavour, max_retries=1, retry_delay=0.25)

    output = generator._call_model(_conversation())

    assert output[0].text == "done", "the second attempt's reply is returned"
    assert route.call_count == 2, "one failed attempt plus one retry"
    assert sleeps == [0.25], "the retry waits retry_delay once"


@pytest.mark.parametrize("flavour", FLAVOURS)
def test_spent_retries_return_the_last_failure_as_a_failed_generation(
    flavour, mock, sleeps
):
    route = _route(mock, flavour, [_server_error(), _server_error(), "ok"])
    generator = _build(flavour, max_retries=1)

    output = generator._call_model(_conversation())

    assert route.call_count == 2, "attempts are the first request plus max_retries"
    assert len(output) == 1 and output[0].text is None, "the failure has no text"
    error = output[0].notes["error"]
    assert error["type"] == "InternalServerError", "the note names the SDK error"
    assert error["status_code"] == 503, "and the status the API answered"
    assert "unavailable" in error["body"], "and the body it sent"


@pytest.mark.parametrize("flavour", FLAVOURS)
def test_max_retries_zero_sends_exactly_one_request(flavour, mock, sleeps):
    route = _route(mock, flavour, [_server_error(), "ok", "ok"])
    generator = _build(flavour, max_retries=0)

    output = generator._call_model(_conversation())

    assert route.call_count == 1, "neither Garak nor the SDK retries"
    assert sleeps == [], "no retry means no wait"
    assert output[0].notes["error"]["type"] == "InternalServerError"


@pytest.mark.parametrize("flavour", FLAVOURS)
def test_spent_retries_return_one_failed_generation_per_requested_generation(
    flavour, mock, sleeps
):
    _route(mock, flavour, [_server_error()])
    generator = _build(flavour, max_retries=0)

    output = generator._call_model(_conversation(), 3)

    assert [m.notes["error"]["type"] for m in output] == ["InternalServerError"] * 3
    assert output[0] is not output[1], "each generation has its own message"


@pytest.mark.parametrize("flavour", FLAVOURS)
def test_a_connection_error_retries_and_a_timeout_ends_up_as_a_note(
    flavour, mock, sleeps
):
    _, path, _ = FLAVOURS[flavour]
    route = mock.post(path).mock(
        side_effect=[
            httpx.ConnectError("refused"),
            httpx.ReadTimeout("slow"),
            httpx.ReadTimeout("slow"),
        ]
    )
    generator = _build(flavour, max_retries=1)

    output = generator._call_model(_conversation())

    assert route.call_count == 2, "the connect failure was retried once"
    assert output[0].notes["error"] == {
        "type": "APITimeoutError"
    }, "a failure with no response names its type only"


@pytest.mark.parametrize("flavour", FLAVOURS)
def test_bounded_retries_retry_rate_limits(flavour, mock, sleeps):
    route = _route(
        mock,
        flavour,
        [httpx.Response(429, json={"error": {"message": "slow"}}), "ok"],
    )
    generator = _build(flavour, max_retries=1)

    assert generator._call_model(_conversation())[0].text == "done"
    assert route.call_count == 2, "rate limits retry as they do under the backoff"


@pytest.mark.parametrize("flavour", FLAVOURS)
def test_bounded_retries_leave_client_errors_to_the_existing_handling(
    flavour, mock, sleeps
):
    route = _route(
        mock, flavour, [httpx.Response(400, json={"error": {"message": "bad"}}), "ok"]
    )
    generator = _build(flavour, max_retries=3)

    generator._call_model(_conversation())

    assert route.call_count == 1, "a 400 is not retried"
    assert sleeps == []


@pytest.mark.parametrize("flavour", FLAVOURS)
def test_without_max_retries_a_transient_error_still_retries_until_it_clears(
    flavour, mock, sleeps
):
    route = _route(mock, flavour, [_server_error()] * 6 + ["ok"])
    generator = _build(flavour)

    output = generator._call_model(_conversation())

    assert output[0].text == "done", "the backoff keeps asking"
    assert route.call_count > 2, "more attempts than a bound of two would allow"


@pytest.mark.parametrize("flavour", FLAVOURS)
def test_a_failed_generation_after_the_retries_holds_no_credential(
    flavour, mock, sleeps, caplog
):
    # only a Responses request carries tools, so only it knows their credential
    notes = {"tools": TOOLS} if flavour == "responses" else None
    secrets = SECRETS if flavour == "responses" else (API_KEY,)
    echo = _server_error(f"unavailable for {' and '.join(secrets)}")
    _route(mock, flavour, [echo, echo])
    generator = _build(flavour, max_retries=1)

    with caplog.at_level(logging.DEBUG):
        output = generator._call_model(_conversation(notes))

    error = output[0].notes["error"]
    assert "unavailable" in error["body"], "the reason is kept"
    assert cli.REDACTED in error["body"], "and the gap is marked"
    records = {"error note": json.dumps(output[0].notes), "garak.log": caplog.text}
    for name, text in records.items():
        assert not any(secret in text for secret in secrets), f"{name} holds no secret"
