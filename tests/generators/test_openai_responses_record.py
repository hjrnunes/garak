# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A Responses reply records the response it came from: id, model and token usage."""

import json
from dataclasses import asdict
from unittest.mock import MagicMock, patch

import pytest
from openai._models import construct_type
from openai.types.responses import Response

from garak.attempt import Conversation, Message, Turn
from garak.generators.openai import OpenAIResponsesGenerator

USAGE = {
    "input_tokens": 120,
    "input_tokens_details": {"cached_tokens": 0},
    "output_tokens": 30,
    "output_tokens_details": {"reasoning_tokens": 4},
    "total_tokens": 150,
}
FAILED_CALL = {
    "type": "mcp_call",
    "id": "fc_1",
    "name": "issue_refund",
    "arguments": '{"order_id": "ORD-104"}',
    "output": None,
    "server_label": "target",
    "approval_request_id": None,
    "error": "tool execution failed",
    "status": "failed",
}
REPLY = {
    "type": "message",
    "id": "msg_1",
    "role": "assistant",
    "status": "completed",
    "content": [{"type": "output_text", "text": "Done.", "annotations": []}],
}


def _response(*items, **fields):
    """Build a Responses object the way the SDK does, without validation."""
    value = {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "model": "test-model-2026-10-01",
        "status": "completed",
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "output": list(items),
        "usage": USAGE,
    }
    return construct_type(type_=Response, value=value | fields)


@pytest.fixture
def generator(monkeypatch):
    monkeypatch.setenv(OpenAIResponsesGenerator.ENV_VAR, "sk-test-fake-key")
    with patch("garak.generators.openai.openai.OpenAI") as client:
        client.return_value = MagicMock()
        yield OpenAIResponsesGenerator(name="test-model")


def _reply(generator, response):
    generator.client.responses.create.return_value = response
    prompt = Conversation([Turn(role="user", content=Message("Refund ORD-104"))])
    return generator._call_model(prompt)[0]


def test_reply_records_response_id_model_and_usage(generator):
    reply = _reply(generator, _response(REPLY))
    record = reply.notes["response"]
    assert set(record) == {"id", "model", "usage"}
    assert record["id"] == "resp_1", "the response is named by its id"
    assert record["model"] == "test-model-2026-10-01", "the model that served it"
    usage = record["usage"]
    tokens = [usage[k] for k in ("input_tokens", "output_tokens", "total_tokens")]
    assert tokens == [120, 30, 150], "the response's token usage, as the SDK reports it"
    assert usage["output_tokens_details"]["reasoning_tokens"] == 4
    json.dumps(asdict(reply))


def test_unfinished_response_records_why(generator):
    reply = _reply(
        generator,
        _response(
            REPLY,
            status="incomplete",
            incomplete_details={"reason": "max_output_tokens"},
        ),
    )
    assert reply.notes["response"]["incomplete_details"] == {
        "reason": "max_output_tokens"
    }
    assert reply.notes["response_status"] == "incomplete", "status note unchanged"


def test_failed_response_records_its_error(generator):
    reply = _reply(
        generator,
        _response(status="failed", error={"code": "server_error", "message": "boom"}),
    )
    assert reply.notes["response"]["error"] == {
        "code": "server_error",
        "message": "boom",
    }


def test_mcp_call_error_is_kept_in_output_items(generator):
    reply = _reply(generator, _response(FAILED_CALL, REPLY))
    assert reply.notes["output_items"] == [
        FAILED_CALL
    ], "a failed call keeps its arguments and error"
    assert "error" not in reply.notes["response"], "the response itself succeeded"


def test_record_is_not_resent_with_the_next_request(generator):
    reply = _reply(generator, _response(FAILED_CALL, REPLY))
    context = Message(reply.text, notes=dict(reply.notes))
    conversation = Conversation(
        [
            Turn(role="user", content=Message("Refund ORD-104")),
            Turn(role="assistant", content=context),
            Turn(role="user", content=Message("Thanks")),
        ]
    )
    assert OpenAIResponsesGenerator._build_input(conversation) == [
        {"role": "user", "content": "Refund ORD-104"},
        FAILED_CALL,
        {"role": "assistant", "content": "Done."},
        {"role": "user", "content": "Thanks"},
    ], "only the output items are input items; the response record is not"


def test_response_without_record_fields_adds_no_note(generator):
    response = MagicMock()
    response.output = []
    response.status = "completed"
    reply = _reply(generator, response)
    assert "response" not in reply.notes, "no id, model or usage to record"
