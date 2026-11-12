# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A later Responses request resends the tool calls of earlier turns, in turn order."""

import json

import httpx
import pytest
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
MCP_TOOLS = [
    {
        "type": "mcp",
        "server_label": "qualification_target",
        "server_url": "http://target.test/mcp",
        "require_approval": "never",
    }
]


def _generator(**options):
    return OpenAIResponsesGenerator(
        name="m",
        config_root={
            "generators": {
                "openai": {
                    "OpenAIResponsesGenerator": {
                        "uri": URI,
                        "api_key": "sk-test-fake-key",
                        **options,
                    }
                }
            }
        },
    )


def _answer(*items):
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "model": "m",
        "status": "completed",
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "output": list(items),
    }


def _message(text):
    return {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


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


def test_a_reply_without_notes_sends_only_its_message():
    conversation = Conversation(
        [
            Turn("user", Message("one")),
            Turn("assistant", Message("a", notes=None)),
            Turn("user", Message("two")),
        ]
    )

    assert [
        item["content"] for item in OpenAIResponsesGenerator._build_input(conversation)
    ] == [
        "one",
        "a",
        "two",
    ]


def test_a_reply_without_text_sends_its_items_and_no_message():
    conversation = Conversation(
        [
            Turn("user", Message("one")),
            _reply(None, MCP_CALL),
            Turn("user", Message("two")),
        ]
    )

    sent = OpenAIResponsesGenerator._build_input(conversation)

    assert sent == [
        {"role": "user", "content": "one"},
        MCP_CALL,
        {"role": "user", "content": "two"},
    ], "a tool-only reply sends the call, and no message with null content"


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


def test_second_request_carries_the_first_turns_tool_calls_in_order():
    """The first request's reply, fed back as a turn, shapes the second request."""
    generator = _generator()
    first = Conversation([Turn("user", Message("Where is ORD-104?"))])
    answers = [
        httpx.Response(
            200, json=_answer(MCP_CALL, OTHER_CALL, _message("It was delivered."))
        ),
        httpx.Response(200, json=_answer(_message("Refunded."))),
    ]

    with respx.mock(base_url=URI) as router:
        route = router.post("/responses").mock(side_effect=answers)
        reply = generator._call_model(first)[0]
        second = Conversation(
            [
                *first.turns,
                Turn("assistant", Message(reply.text, notes=reply.notes)),
                Turn("user", Message("Refund it.")),
            ]
        )
        generator._call_model(second)

    first_body, second_body = (json.loads(call.request.content) for call in route.calls)
    assert first_body["input"] == "Where is ORD-104?", "no history on the first turn"
    assert [item.get("id") or item["content"] for item in second_body["input"]] == [
        "Where is ORD-104?",
        "mcp_1",
        "mcp_2",
        "It was delivered.",
        "Refund it.",
    ], "the second request resends the calls between the turns they came between"
    assert (
        second_body["input"][1]["output"] == '{"status": "delivered"}'
    ), "a resent call keeps its result"
    assert "previous_response_id" not in second_body, "no server-side state is needed"


@pytest.mark.parametrize("source", ["conversation", "generator"])
def test_every_request_of_a_conversation_carries_the_same_tool_block(source):
    """The tool block is built per request, so each turn must send the first turn's."""
    generator = _generator(tools=MCP_TOOLS) if source == "generator" else _generator()
    notes = {"tools": MCP_TOOLS} if source == "conversation" else None
    first = Conversation([Turn("user", Message("Where is ORD-104?"))], notes=notes)
    answers = [
        httpx.Response(200, json=_answer(MCP_CALL, _message("It was delivered."))),
        httpx.Response(200, json=_answer(_message("Refunded."))),
    ]

    with respx.mock(base_url=URI) as router:
        route = router.post("/responses").mock(side_effect=answers)
        reply = generator._call_model(first)[0]
        generator._call_model(
            Conversation(
                [
                    *first.turns,
                    Turn("assistant", Message(reply.text, notes=reply.notes)),
                    Turn("user", Message("Refund it.")),
                ],
                notes=notes,
            )
        )

    first_body, second_body = (json.loads(call.request.content) for call in route.calls)
    assert first_body["tools"] == MCP_TOOLS, "the first request lost its tool block"
    assert second_body["tools"] == first_body["tools"], "the tool block changed by turn"


def test_second_request_after_a_tool_only_reply_holds_no_null_content():
    generator = _generator()
    first = Conversation([Turn("user", Message("Where is ORD-104?"))])
    answers = [
        httpx.Response(200, json=_answer(MCP_CALL)),
        httpx.Response(200, json=_answer(_message("Refunded."))),
    ]

    with respx.mock(base_url=URI) as router:
        route = router.post("/responses").mock(side_effect=answers)
        reply = generator._call_model(first)[0]
        assert reply.text is None, "the reply holds a tool call and no text"
        generator._call_model(
            Conversation(
                [
                    *first.turns,
                    Turn("assistant", Message(reply.text, notes=reply.notes)),
                    Turn("user", Message("Refund it.")),
                ]
            )
        )

    second_body = json.loads(route.calls[1].request.content)
    assert all(
        item.get("content", "") is not None for item in second_body["input"]
    ), "the API takes no message with null content"
    assert [item.get("id") or item["content"] for item in second_body["input"]] == [
        "Where is ORD-104?",
        "mcp_1",
        "Refund it.",
    ]
