# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import httpx
import openai
import pytest
import respx

from garak.attempt import Conversation, Message, Turn
from garak.generators.openai import OpenAICompatible, OpenAIResponsesGenerator

URI = "http://retry.test/v1"

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

# generator class, namespace key in the config, request path, success body
FLAVOURS = {
    "chat": (OpenAICompatible, "/chat/completions", CHAT_OK),
    "responses": (OpenAIResponsesGenerator, "/responses", RESPONSES_OK),
}


def _server_error():
    return httpx.Response(503, json={"error": {"message": "unavailable"}})


def _build(flavour, **settings):
    klass, _, _ = FLAVOURS[flavour]
    config = {"uri": URI, "api_key": "sk-test-fake-key", **settings}
    return klass(
        name="m",
        config_root={"generators": {"openai": {klass.__name__: config}}},
    )


def _conversation():
    return Conversation([Turn("user", Message("hello"))])


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
    [({}, {}), ({"max_retries": 2}, {"max_retries": 0})],
    ids=["unset", "set"],
)
def test_sdk_retries_are_off_only_when_the_generator_owns_retrying(
    flavour, settings, expected, monkeypatch
):
    created = []
    real_client = openai.OpenAI
    monkeypatch.setattr(
        "garak.generators.openai.openai.OpenAI",
        lambda **kwargs: created.append(kwargs) or real_client(**kwargs),
    )

    _build(flavour, **settings)

    sdk_options = {k: v for k, v in created[0].items() if k == "max_retries"}
    assert sdk_options == expected, "SDK max_retries is set to 0 only with max_retries"


@pytest.mark.parametrize("flavour", FLAVOURS)
def test_max_retries_retries_a_server_error_once_after_the_delay(flavour, mock, sleeps):
    route = _route(mock, flavour, [_server_error(), "ok"])
    generator = _build(flavour, max_retries=1, retry_delay=0.25)

    output = generator._call_model(_conversation())

    assert output[0].text == "done", "the second attempt's reply is returned"
    assert route.call_count == 2, "one failed attempt plus one retry"
    assert sleeps == [0.25], "the retry waits retry_delay once"


@pytest.mark.parametrize("flavour", FLAVOURS)
def test_max_retries_stops_after_the_count_and_raises_the_last_error(
    flavour, mock, sleeps
):
    route = _route(mock, flavour, [_server_error(), _server_error(), "ok"])
    generator = _build(flavour, max_retries=1)

    with pytest.raises(openai.InternalServerError):
        generator._call_model(_conversation())

    assert route.call_count == 2, "attempts are the first request plus max_retries"


@pytest.mark.parametrize("flavour", FLAVOURS)
def test_max_retries_zero_sends_exactly_one_request(flavour, mock, sleeps):
    route = _route(mock, flavour, [_server_error(), "ok", "ok"])
    generator = _build(flavour, max_retries=0)

    with pytest.raises(openai.InternalServerError):
        generator._call_model(_conversation())

    assert route.call_count == 1, "neither Garak nor the SDK retries"
    assert sleeps == [], "no retry means no wait"


@pytest.mark.parametrize("flavour", FLAVOURS)
def test_retry_condition_decides_which_errors_retry(flavour, mock, sleeps):
    seen = []

    def server_errors_only(error):
        seen.append(type(error).__name__)
        return isinstance(error, openai.InternalServerError)

    route = _route(
        mock,
        flavour,
        [
            _server_error(),
            httpx.Response(429, json={"error": {"message": "slow down"}}),
            "ok",
        ],
    )
    generator = _build(flavour, max_retries=3, retry_condition=server_errors_only)

    with pytest.raises(openai.RateLimitError):
        generator._call_model(_conversation())

    assert route.call_count == 2, "a 429 the condition rejects is not retried"
    assert seen == ["InternalServerError", "RateLimitError"]


@pytest.mark.parametrize("flavour", FLAVOURS)
def test_retry_condition_is_not_asked_once_the_count_is_spent(flavour, mock, sleeps):
    seen = []

    def always(error):
        seen.append(error)
        return True

    _route(mock, flavour, [_server_error(), _server_error()])
    generator = _build(flavour, max_retries=1, retry_condition=always)

    with pytest.raises(openai.InternalServerError):
        generator._call_model(_conversation())

    assert len(seen) == 1, "only a failure that can still retry reaches the condition"


@pytest.mark.parametrize("flavour", FLAVOURS)
def test_timeout_and_connection_errors_reach_the_condition(flavour, mock, sleeps):
    kinds = []

    def condition(error):
        kinds.append(type(error))
        return not isinstance(error, openai.APITimeoutError)

    _, path, _ = FLAVOURS[flavour]
    mock.post(path).mock(
        side_effect=[httpx.ConnectError("refused"), httpx.ReadTimeout("slow")]
    )
    generator = _build(flavour, max_retries=3, retry_condition=condition)

    with pytest.raises(openai.APITimeoutError):
        generator._call_model(_conversation())

    assert kinds == [
        openai.APIConnectionError,
        openai.APITimeoutError,
    ], "a connect failure retries and a timeout, which the condition rejects, ends it"


@pytest.mark.parametrize("flavour", FLAVOURS)
def test_bounded_retries_without_a_condition_retry_the_transient_errors(
    flavour, mock, sleeps
):
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

    assert route.call_count == 1, "a 400 is neither retried nor raised"
