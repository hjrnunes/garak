# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import httpx
import respx

from garak.attempt import Conversation, Message, Turn
from garak.generators.openai import OpenAIResponsesGenerator

URI = "http://responses-input.test/v1"

MCP_CALL = {
    "id": "mcp_1",
    "type": "mcp_call",
    "arguments": '{"order_id": "ORD-104"}',
    "name": "lookup_order",
    "server_label": "qualification_target",
    "error": None,
    "output": '{"status": "delivered"}',
}
OTHER_CALL = {**MCP_CALL, "id": "mcp_2", "name": "retrieve_policy"}


def _reply(text, *items):
    return Turn("assistant", Message(text, notes={"output_items": list(items)}))


def test_output_items_are_sent_before_the_reply_they_came_with():
    conversation = Conversation(
        [
            Turn("user", Message("Where is ORD-104?")),
            _reply("It was delivered.", MCP_CALL),
            Turn("user", Message("Refund it.")),
        ]
    )

    assert OpenAIResponsesGenerator._build_input(conversation) == [
        {"role": "user", "content": "Where is ORD-104?"},
        MCP_CALL,
        {"role": "assistant", "content": "It was delivered."},
        {"role": "user", "content": "Refund it."},
    ], "each reply is preceded by the output items its turn produced, in order"


def test_output_items_of_several_replies_keep_their_turn_order():
    conversation = Conversation(
        [
            Turn("user", Message("one")),
            _reply("a", MCP_CALL, OTHER_CALL),
            Turn("user", Message("two")),
            _reply("b", OTHER_CALL),
            Turn("user", Message("three")),
        ]
    )

    sent = OpenAIResponsesGenerator._build_input(conversation)

    assert [item.get("id") or item["content"] for item in sent] == [
        "one",
        "mcp_1",
        "mcp_2",
        "a",
        "two",
        "mcp_2",
        "b",
        "three",
    ]


def test_system_turn_is_still_left_out_of_the_input():
    conversation = Conversation(
        [
            Turn("system", Message("Be brief.")),
            Turn("user", Message("hi")),
            _reply("hello", MCP_CALL),
            Turn("user", Message("bye")),
        ]
    )

    sent = OpenAIResponsesGenerator._build_input(conversation)

    assert [item.get("role") for item in sent] == ["user", None, "assistant", "user"]


def test_a_single_user_turn_is_still_sent_as_a_string():
    conversation = Conversation([Turn("user", Message("hi"))])

    assert OpenAIResponsesGenerator._build_input(conversation) == "hi"


def test_sent_items_are_copies_of_the_notes():
    conversation = Conversation(
        [
            Turn("user", Message("one")),
            _reply("a", MCP_CALL),
            Turn("user", Message("two")),
        ]
    )

    sent = OpenAIResponsesGenerator._build_input(conversation)
    assert sent[1] == MCP_CALL
    sent[1]["name"] = "changed"

    assert (
        conversation.turns[1].content.notes["output_items"][0]["name"] == "lookup_order"
    ), "editing the request input must not edit the conversation"


def test_output_items_survive_a_conversation_loaded_from_json():
    loaded = Conversation.from_dict(
        {
            "turns": [
                {"role": "user", "content": {"text": "one"}},
                {
                    "role": "assistant",
                    "content": {"text": "a", "notes": {"output_items": [MCP_CALL]}},
                },
                {"role": "user", "content": {"text": "two"}},
            ]
        }
    )

    assert (
        OpenAIResponsesGenerator._build_input(loaded)[1] == MCP_CALL
    ), "a probe that reads conversations from JSON can carry output items"


def test_request_body_carries_the_output_items():
    generator = OpenAIResponsesGenerator(
        name="m",
        config_root={
            "generators": {
                "openai": {
                    "OpenAIResponsesGenerator": {
                        "uri": URI,
                        "api_key": "sk-test-fake-key",
                    }
                }
            }
        },
    )
    conversation = Conversation(
        [
            Turn("user", Message("one")),
            _reply("a", MCP_CALL),
            Turn("user", Message("two")),
        ]
    )
    answer = {
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

    with respx.mock(base_url=URI) as router:
        route = router.post("/responses").mock(
            return_value=httpx.Response(200, json=answer)
        )
        generator._call_model(conversation)

    body = json.loads(route.calls[0].request.content)
    assert body["input"][1] == MCP_CALL, "the wire request holds the item verbatim"
