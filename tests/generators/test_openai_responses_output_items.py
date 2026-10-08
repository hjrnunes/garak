# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A Responses reply keeps its ``mcp_call`` items so a later turn can resend them."""

import json
from dataclasses import asdict
from unittest.mock import MagicMock, patch

import pytest
from openai._models import construct_type
from openai.types.responses import Response

from garak.attempt import Conversation, Message, Turn
from garak.generators.openai import OpenAIResponsesGenerator

MCP_CALL = {
    "type": "mcp_call",
    "id": "fc_1",
    "name": "get_account_details",
    "arguments": '{"customer_id": "me"}',
    "output": '{"error": "AUTHORIZATION: forbidden"}',
    "server_label": "qualification_target",
    "approval_request_id": None,
    "error": None,
    "status": None,
}
LIST_TOOLS = {
    "type": "mcp_list_tools",
    "id": "mcp_list_1",
    "server_label": "qualification_target",
    "tools": [],
    "error": None,
}
REPLY = {
    "type": "message",
    "id": "msg_1",
    "role": "assistant",
    "status": "completed",
    "content": [{"type": "output_text", "text": "Your orders.", "annotations": []}],
}


def _response(*items):
    """Build a Responses object the way the SDK does, without validation."""
    return construct_type(
        type_=Response,
        value={
            "id": "resp_1",
            "object": "response",
            "created_at": 0,
            "model": "test-model",
            "status": "completed",
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "output": list(items),
        },
    )


@pytest.fixture
def generator(monkeypatch):
    monkeypatch.setenv(OpenAIResponsesGenerator.ENV_VAR, "sk-test-fake-key")
    with patch("garak.generators.openai.openai.OpenAI") as client:
        client.return_value = MagicMock()
        yield OpenAIResponsesGenerator(name="test-model")


def _reply(generator, response):
    generator.client.responses.create.return_value = response
    prompt = Conversation([Turn(role="user", content=Message("Show my orders"))])
    return generator._call_model(prompt)[0]


def test_mcp_call_items_are_kept_whole_in_output_items(generator):
    second = {**MCP_CALL, "id": "fc_2", "name": "get_state", "arguments": "{}"}
    reply = _reply(generator, _response(LIST_TOOLS, MCP_CALL, second, REPLY))
    assert reply.notes["output_items"] == [
        MCP_CALL,
        second,
    ], "every mcp_call item, null fields included, is kept in response order"
    json.dumps(asdict(reply))


def test_reply_without_mcp_call_has_no_output_items(generator):
    reply = _reply(generator, _response(LIST_TOOLS, REPLY))
    assert "output_items" not in reply.notes, "nothing to resend"


def test_kept_items_are_what_the_next_request_sends(generator):
    reply = _reply(generator, _response(MCP_CALL, REPLY))
    context = Message(reply.text, notes={"output_items": reply.notes["output_items"]})
    conversation = Conversation(
        [
            Turn(role="user", content=Message("Show my orders")),
            Turn(role="assistant", content=context),
            Turn(role="user", content=Message("Refund ORD-104")),
        ]
    )
    assert OpenAIResponsesGenerator._build_input(conversation) == [
        {"role": "user", "content": "Show my orders"},
        MCP_CALL,
        {"role": "assistant", "content": "Your orders."},
        {"role": "user", "content": "Refund ORD-104"},
    ], "the next request carries the call between the turns it came between"
