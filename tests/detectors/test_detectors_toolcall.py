# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for tool-call condition validation and evaluation.

The vendored contract vectors pin the condition language; these tests cover
what they do not reach: the normalisation of the generator capture shapes and
the evaluation semantics on normalised calls.
"""

import json

import pytest

from garak.attempt import Message
from garak.detectors.toolcall import (
    evaluate_condition,
    normalise_tool_call,
    read_tool_calls,
    validate_condition,
)

UNDECODABLE = "{not json}"


def call(name, arguments, **extra):
    """A Responses API function_call item as the generator captures it."""
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {"type": "function_call", "name": name, "arguments": raw, **extra}


def chat_call(name, arguments):
    """A chat-completions tool call as the chat generator captures it."""
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {
        "id": "call-1",
        "type": "function",
        "function": {"name": name, "arguments": raw},
    }


def mcp_call(name, arguments, output=None, **extra):
    """A Responses API mcp_call item as the generator captures it."""
    return {
        "type": "mcp_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": name,
        "arguments": json.dumps(arguments),
        "output": output,
        "error": None,
        "status": None,
        "server_label": "target",
        **extra,
    }


def output(*calls, complete=True):
    notes = {"tool_calls": list(calls)} if calls else {}
    if not complete:
        notes["response_status"] = "in_progress"
    return Message(None, notes=notes)


def arg(operation, argument):
    return {"source": "argument", "operation": operation, "argument": argument}


def literal(value):
    return {"source": "literal", "value": value}


def value(left, op, right):
    return {"kind": "value", "left": left, "op": op, "right": right}


def not_called(operation, where=None):
    comparison = {"kind": "not_called", "operation": operation}
    if where is not None:
        comparison["where"] = where
    return comparison


_UNSET = object()


def order(operation, prior, same=_UNSET):
    """An order comparison; ``same`` left unset omits ``same_argument``."""
    comparison = {"kind": "order", "operation": operation, "requires_prior": prior}
    if same is not _UNSET:
        comparison["same_argument"] = same
    return comparison


def condition(*comparisons):
    return {"comparisons": list(comparisons)}


# ── normalisation of the capture shapes ──────────────────────────────────────


def test_normalise_reads_the_responses_function_call_shape():
    normalised = normalise_tool_call(call("get_orders", {"order": "ORD-1"}))
    assert normalised == {
        "name": "get_orders",
        "arguments": {"order": "ORD-1"},
        "result": None,
    }, "name and JSON-string arguments are decoded"


def test_normalise_reads_the_responses_mcp_call_shape():
    normalised = normalise_tool_call(
        mcp_call("get_account", {"id": 7}, output='{"balance": "10.00"}')
    )
    assert normalised["name"] == "get_account", "mcp name is read"
    assert normalised["arguments"] == {"id": 7}, "mcp arguments are decoded"
    assert normalised["result"] == {"balance": "10.00"}, "mcp output is decoded"


def test_normalise_reads_the_chat_completions_shape():
    normalised = normalise_tool_call(chat_call("send_email", {"to": "a@b.c"}))
    assert normalised == {
        "name": "send_email",
        "arguments": {"to": "a@b.c"},
        "result": None,
    }, "function.name and function.arguments are read"


def test_normalise_decodes_null_and_broken_arguments():
    assert normalise_tool_call({"type": "function_call", "name": "t"})["arguments"] is (
        None
    ), "absent arguments are null"
    assert (
        normalise_tool_call(call("t", UNDECODABLE))["arguments"] is None
    ), "undecodable JSON is null"
    assert normalise_tool_call(
        {"type": "mcp_call", "name": "t", "arguments": {"a": 2}}
    )["arguments"] == {"a": 2}, "already-decoded arguments pass through"


def test_normalise_marks_a_non_object_call_unknown():
    assert normalise_tool_call("junk") == {
        "name": None,
        "arguments": None,
        "result": None,
    }, "a non-object call has no name or arguments"


def test_normalise_passes_blocked_through():
    blocked = normalise_tool_call(call("t", {}, blocked=True))
    assert blocked["blocked"] is True, "a blocked Responses call stays blocked"
    chat = {**chat_call("t", {}), "blocked": False}
    assert normalise_tool_call(chat)["blocked"] is False, "chat shape too"
    assert "blocked" not in normalise_tool_call(call("t", {})), "absent stays absent"
    assert "blocked" not in normalise_tool_call(
        call("t", {}, blocked="yes")
    ), "only a boolean marks a call blocked"


def test_read_tool_calls_completeness():
    done = Message(None, notes={"tool_calls": [], "response_status": "completed"})
    assert read_tool_calls(done)[1] is True, "completed is complete"
    absent = Message("hello")
    assert read_tool_calls(absent) == ([], True), "no note means no call, complete"
    unfinished = Message(None, notes={"tool_calls": [], "response_status": "failed"})
    assert read_tool_calls(unfinished)[1] is False, "any other status is incomplete"


def test_read_tool_calls_normalises_and_rejects_a_non_list():
    out = Message(None, notes={"tool_calls": [call("a", {}), chat_call("b", {})]})
    calls, _ = read_tool_calls(out)
    assert [c["name"] for c in calls] == ["a", "b"], "both shapes are normalised"
    broken = Message(None, notes={"tool_calls": {"name": "x"}})
    assert read_tool_calls(broken) == (None, False), "a non-list capture is unreadable"


def test_read_tool_calls_without_notes_attribute():
    message = Message("hi")
    message.notes = None
    assert read_tool_calls(message) == ([], True), "a None notes dict is no capture"


@pytest.mark.parametrize("notes", ["broken", [], 3, ["tool_calls"]])
def test_read_tool_calls_rejects_notes_that_are_not_an_object(notes):
    message = Message("hi")
    message.notes = notes
    assert read_tool_calls(message) == (None, False), "notes hold no capture to read"


@pytest.mark.parametrize(
    "entry", [None, "call", 3, []], ids=["null", "text", "number", "list"]
)
def test_read_tool_calls_marks_an_entry_that_is_not_an_object_incomplete(entry):
    out = Message(None, notes={"tool_calls": [call("a", {}), entry]})
    calls, complete = read_tool_calls(out)
    assert len(calls) == 2, "the entry keeps its place"
    assert complete is False, "an unreadable entry may have been any call"


def test_normalise_reads_alias_keys_and_the_input_field():
    aliased = normalise_tool_call(
        {
            "type": "mcp_call",
            "tool_name": "t",
            "raw_arguments": '{"a": 1}',
            "raw_result": "[1]",
        }
    )
    assert aliased == {"name": "t", "arguments": {"a": 1}, "result": [1]}, "aliases"
    held = normalise_tool_call(
        {"type": "mcp_call", "name": "t", "input": {"id": 7}, "status": "completed"}
    )
    assert held["arguments"] == {"id": 7}, "an input-only entry keeps its arguments"
    both = normalise_tool_call(
        {
            "type": "mcp_call",
            "name": "t",
            "arguments": '{"a": 1}',
            "raw_arguments": '{"a": 3}',
            "input": '{"a": 2}',
        }
    )
    assert both["arguments"] == {"a": 1}, "arguments win over the aliases"
    assert normalise_tool_call(
        {"type": "mcp_call", "arguments": None, "raw_arguments": '{"a": 3}'}
    )["arguments"] == {"a": 3}, "raw_arguments come before input"
    assert normalise_tool_call(
        {"type": "mcp_call", "arguments": None, "input": '{"a": 2}'}
    )["arguments"] == {"a": 2}, "input is the last source"


@pytest.mark.parametrize("kind", ["custom_tool_call", "web_search_call", None])
def test_normalise_reads_input_only_for_an_mcp_call(kind):
    item = {"name": "x", "input": '{"id": 7}'}
    if kind is not None:
        item["type"] = kind
    assert normalise_tool_call(item)["arguments"] is None, "freeform text is not read"


@pytest.mark.parametrize("name", [5, ["x"], {"a": 1}], ids=["number", "list", "dict"])
def test_normalise_keeps_only_text_names(name):
    assert normalise_tool_call({"name": name, "arguments": "{}"})["name"] is None
    assert normalise_tool_call({"function": {"name": name}})["name"] is None


@pytest.mark.parametrize(
    "raw", ["9" * 5000, "[" * 100_000 + "]" * 100_000], ids=["digits", "nesting"]
)
def test_normalise_treats_text_python_cannot_parse_as_null(raw):
    assert normalise_tool_call({"name": "x", "arguments": raw})["arguments"] is None


# ── evaluation on normalised calls ───────────────────────────────────────────


def test_evaluate_condition_three_valued_on_normalised_calls():
    cond = condition(
        not_called("escalate"), value(arg("refund", "amount"), "gt", literal(10))
    )
    clean = [normalise_tool_call(call("refund", {"amount": 50}))]
    assert evaluate_condition(cond, clean, True)[0] == "detected", "condition holds"
    dirty = clean + [normalise_tool_call(call("escalate", {}))]
    assert evaluate_condition(cond, dirty, True)[0] == "not_detected", "escalated"
    missing = [normalise_tool_call(call("other", {}))]
    assert (
        evaluate_condition(cond, missing, False)[0] == "inconclusive"
    ), "an incomplete capture may lack the refund"
    assert (
        evaluate_condition(condition(not_called("x")), [], False)[0] == "inconclusive"
    ), "absence is only proven by a complete capture"


def test_evaluate_condition_reads_blocked_calls():
    cond = condition(order("refund", "lookup", "id"))
    calls = [
        normalise_tool_call(call("lookup", {"id": "A"}, blocked=True)),
        normalise_tool_call(call("refund", {"id": "A"})),
    ]
    assert evaluate_condition(cond, calls, True)[:2] == (
        "detected",
        "the condition holds on the captured calls",
    ), "a blocked lookup did not run"


def test_validate_condition_names_the_first_problem():
    assert validate_condition("not a dict") == "condition is not an object"
    assert (
        validate_condition(condition(value(arg("t", "a"), "approx", literal(1))))
        == "comparisons[0] has unknown op 'approx'"
    )
    assert (
        validate_condition(condition(order("pay", "check", None))) is None
    ), "null same_argument is valid"
