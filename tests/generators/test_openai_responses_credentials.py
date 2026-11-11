# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A Responses request keeps the credentials of its tools out of garak's logs and records."""

import json
import logging

import httpx
import pytest
import respx

import garak.exception
from garak import cli
from garak.attempt import Attempt, Conversation, Message, Turn
from garak.generators.openai import OpenAIResponsesGenerator

URI = "http://credentials.test/v1"
API_KEY = "sentinel-generator-api-key-must-not-be-kept"
AUTHORIZATION = "sentinel-tool-authorization-must-not-be-kept"
BEARER_TOKEN = "sentinel-bearer-token-must-not-be-kept"
QUOTED = 'sentinel-"quoted"-secret-must-not-be-kept'
SECRETS = (API_KEY, AUTHORIZATION, BEARER_TOKEN, QUOTED)
TOOLS = [
    {
        "type": "mcp",
        "server_label": "bank",
        "server_url": "http://tools.test/sse",
        "authorization": AUTHORIZATION,
        "headers": {"Authorization": f"Bearer {BEARER_TOKEN}", "X-Api-Key": QUOTED},
    }
]
OK = {
    "id": "resp_1",
    "object": "response",
    "created_at": 0,
    "model": "m",
    "status": "completed",
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
    "output": [],
}


def _generate(response, configured_tools=None, notes=None):
    options = {"uri": URI, "api_key": API_KEY}
    if configured_tools:
        options["tools"] = configured_tools
    generator = OpenAIResponsesGenerator(
        name="m",
        config_root={
            "generators": {"openai": {"OpenAIResponsesGenerator": options}},
        },
    )
    with respx.mock(base_url=URI) as router:
        router.post("/responses").mock(return_value=response)
        return generator._call_model(
            Conversation([Turn("user", Message("hello"))], notes=notes)
        )


@pytest.mark.parametrize("where", ["configured", "conversation"])
def test_the_client_debug_log_holds_no_tool_credential(where, caplog):
    kwargs = {"configured_tools": TOOLS} if where == "configured" else {}
    kwargs |= {"notes": {"tools": TOOLS}} if where == "conversation" else {}

    with caplog.at_level(logging.DEBUG):
        _generate(httpx.Response(200, json=OK), **kwargs)

    assert "Request options" in caplog.text, "the client still logs the request"
    assert "bank" in caplog.text, "and the tool it sends"
    assert not any(secret in caplog.text for secret in SECRETS), "but not a credential"
    assert cli.REDACTED in caplog.text, "the gap is marked"


@pytest.mark.parametrize("status", [400, 404, 422])
def test_a_rejection_that_echoes_the_request_is_not_kept(status, caplog):
    """A validation error can send the request back, tools and credentials included."""
    body = {"detail": [{"msg": "invalid tool", "input": {"tools": TOOLS}}]}

    with caplog.at_level(logging.DEBUG):
        outputs = _generate(httpx.Response(status, json=body), notes={"tools": TOOLS})

    attempt = Attempt(prompt=Message("hello"))
    attempt.outputs = outputs
    error = outputs[0].notes["error"]
    assert error["status_code"] == status
    assert "invalid tool" in error["body"], "the reason is kept"
    assert cli.REDACTED in error["body"], "and the gap is marked"
    records = {
        "garak.log": caplog.text,
        "error note": json.dumps(outputs[0].notes),
        "serialised attempt": json.dumps(attempt.as_dict()),
    }
    for name, text in records.items():
        assert not any(secret in text for secret in SECRETS), f"{name} holds no secret"
    assert "invalid tool" in caplog.text, "the rejection is still logged"


def test_a_rejection_that_echoes_only_a_bearer_token_is_scrubbed():
    body = {"error": {"message": f"unauthorised token {BEARER_TOKEN}"}}

    outputs = _generate(httpx.Response(400, json=body), notes={"tools": TOOLS})

    assert BEARER_TOKEN not in outputs[0].notes["error"]["body"], "the token alone"
    assert "unauthorised token" in outputs[0].notes["error"]["body"]


def test_a_rejection_that_echoes_the_api_key_is_scrubbed():
    body = {"error": {"message": f"bad key {API_KEY}"}}

    outputs = _generate(httpx.Response(400, json=body))

    assert API_KEY not in outputs[0].notes["error"]["body"]


@pytest.mark.parametrize("status", [401, 403])
def test_a_rejected_credential_does_not_repeat_the_api_key(status, caplog):
    body = {"error": {"message": f"Incorrect API key provided: {API_KEY}"}}

    with (
        caplog.at_level(logging.DEBUG),
        pytest.raises(garak.exception.GarakException) as raised,
    ):
        _generate(httpx.Response(status, json=body))

    assert "Incorrect API key provided" in str(raised.value), "the reason is kept"
    assert API_KEY not in str(raised.value), "the exception holds no key"
    assert API_KEY not in caplog.text, "nor does the log"


def test_a_rejection_without_credentials_is_kept_as_received():
    body = b'{"error": {"message": "unsupported tool"}}'

    outputs = _generate(httpx.Response(400, content=body), notes={"tools": TOOLS})

    assert outputs[0].notes["error"]["body"] == body.decode(), "nothing to scrub"


def test_credential_values_lists_secrets_and_the_last_word_of_a_scheme():
    found = cli.credential_values({"openai": {"x": TOOLS, "api_key": API_KEY}})

    assert set(found) == {
        API_KEY,
        AUTHORIZATION,
        f"Bearer {BEARER_TOKEN}",
        BEARER_TOKEN,
        QUOTED,
    }
