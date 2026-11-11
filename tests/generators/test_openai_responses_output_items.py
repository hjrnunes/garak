# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A Responses reply keeps its ``mcp_call`` items whole, results and errors included."""

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
    "output": '{"balance": "100.00"}',
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


def test_server_side_tool_loop_is_kept_in_order(generator):
    """The API runs the loop: list tools, two calls, then the answer."""
    second = {
        **MCP_CALL,
        "id": "fc_2",
        "name": "get_orders",
        "arguments": "{}",
        "output": '{"orders": ["ORD-104"]}',
    }

    reply = _reply(generator, _response(LIST_TOOLS, MCP_CALL, second, REPLY))

    assert reply.text == "Your orders.", "only the final message is the text"
    assert [call["name"] for call in reply.notes["tool_calls"]] == [
        "get_account_details",
        "get_orders",
    ], "tool calls are recorded in the order the API ran them"
    assert reply.notes["output_items"] == [
        MCP_CALL,
        second,
    ], "every mcp_call item, null fields included, is kept in response order"
    json.dumps(asdict(reply))


def test_reply_without_mcp_call_has_no_output_items(generator):
    reply = _reply(generator, _response(LIST_TOOLS, REPLY))
    assert "output_items" not in reply.notes, "nothing to resend"


@pytest.mark.parametrize(
    "failure",
    [
        {"output": None, "error": "tool execution failed", "status": "failed"},
        {
            "output": '{"isError": true, "content": "AUTHORIZATION: forbidden"}',
            "error": None,
            "status": "completed",
        },
    ],
    ids=["error_field", "error_in_output"],
)
def test_failed_mcp_call_keeps_its_error_and_output(generator, failure):
    """An MCP result flagged isError reaches the reply with its evidence intact."""
    call = {**MCP_CALL, "id": "fc_err", "name": "issue_refund", **failure}

    reply = _reply(generator, _response(call, REPLY))

    assert reply.notes["output_items"] == [call], "the item is kept verbatim"
    entry = reply.notes["tool_calls"][0]
    assert entry["name"] == "issue_refund"
    for field in ("output", "error", "status"):
        if failure[field] is None:
            assert field not in entry, f"an absent {field} is not recorded"
        else:
            assert entry[field] == failure[field], f"the call's {field} is recorded"
    assert reply.notes["response_status"] == "completed", "the response itself ended"
    json.dumps(asdict(reply))
