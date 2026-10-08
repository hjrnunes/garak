# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the sequential injection probe, which sends scripted user turns one by one."""

import json

import pytest

import garak._plugins
from garak import _config
from garak.attempt import Message
from garak.generators.openai import OpenAIResponsesGenerator
from garak.probes.injection import SequentialInjection

TOOLS = [{"type": "mcp", "server_label": "target", "server_url": "http://mcp"}]
CONDITION = {"any": [{"tool": "issue_refund"}]}
RUBRIC = {"question": "Was a refund promised?", "criteria": []}


def _call(name, call_id):
    return {"type": "mcp_call", "id": call_id, "name": name, "error": None}


class ScriptedTarget:
    """Answer each request with the next scripted reply and keep every prompt."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def generate(self, prompt, generations_this_call=1):
        self.prompts.append(prompt)
        return [self.replies.pop(0)]

    def clear_history(self):
        pass


def _reply(text, calls=(), status="completed"):
    notes = {"response_status": status}
    if calls:
        notes["tool_calls"] = [{"type": c["type"], "name": c["name"]} for c in calls]
        notes["output_items"] = list(calls)
    return Message(text, notes=notes)


def _probe(tmp_path, entry, **params):
    source = tmp_path / "script.json"
    source.write_text(json.dumps([entry]), encoding="utf-8")
    config_root = {
        "probes": {
            "injection": {
                "SequentialInjection": {"conversation_source": str(source), **params}
            }
        }
    }
    return SequentialInjection(config_root=config_root)


def _entry(*texts, **extra):
    return {"messages": [{"role": "user", "content": t} for t in texts], **extra}


@pytest.fixture
def report(tmp_path):
    _config.load_base_config()
    path = tmp_path / "probe.report.jsonl"
    with open(path, "w+", encoding="utf-8") as reportfile:
        _config.transient.reportfile = reportfile
        _config.transient.report_filename = str(path)
        yield reportfile


def _run(probe, replies):
    target = ScriptedTarget(replies)
    return probe.probe(target), target


def test_shipped_script_loads():
    probe = garak._plugins.load_plugin("probes.injection.SequentialInjection")
    assert isinstance(probe, SequentialInjection)
    assert probe.scripts, "the shipped example holds at least one script"


def test_each_turn_is_one_request_carrying_the_earlier_turns(tmp_path, report):
    first = _call("get_orders", "fc_1")
    probe = _probe(tmp_path, _entry("Show my orders", "Refund ORD-104", tools=TOOLS))
    attempts, target = _run(
        probe, [_reply("Your orders.", [first]), _reply("Refund issued.")]
    )

    assert len(target.prompts) == 2, "one request per scripted user turn"
    assert OpenAIResponsesGenerator._build_input(target.prompts[0]) == "Show my orders"
    assert OpenAIResponsesGenerator._build_input(target.prompts[1]) == [
        {"role": "user", "content": "Show my orders"},
        first,
        {"role": "assistant", "content": "Your orders."},
        {"role": "user", "content": "Refund ORD-104"},
    ], "a later request resends earlier turns, their mcp_call items and real replies"
    assert all(
        p.notes["tools"] == TOOLS for p in target.prompts
    ), "every turn sends the entry's tools"
    assert len(attempts) == 2, "one attempt per turn"


def test_attempts_record_turn_index_and_all_calls_so_far(tmp_path, report):
    first, second = _call("get_orders", "fc_1"), _call("issue_refund", "fc_2")
    probe = _probe(tmp_path, _entry("a", "b", "c"))
    attempts, _ = _run(
        probe, [_reply("1", [first]), _reply("2"), _reply("3", [second])]
    )

    sequences = [a.notes["sequence"] for a in attempts]
    assert [s["turn_index"] for s in sequences] == [0, 1, 2]
    assert [s["call_offset"] for s in sequences] == [
        0,
        1,
        1,
    ], "calls made by earlier turns"
    assert {s["turn_count"] for s in sequences} == {3}
    assert [c["name"] for c in attempts[2].outputs[0].notes["tool_calls"]] == [
        "get_orders",
        "issue_refund",
    ], "a turn's output carries every call made so far, for the detectors"


def test_attempts_are_written_to_the_report_in_turn_order(tmp_path, report):
    probe = _probe(tmp_path, _entry("a", "b"))
    _run(probe, [_reply("1"), _reply("2")])
    report.seek(0)
    entries = [json.loads(line) for line in report.read().splitlines()]
    assert [e["notes"]["sequence"]["turn_index"] for e in entries] == [0, 1]
    assert [e["prompt"]["turns"][-1]["content"]["text"] for e in entries] == [
        "a",
        "b",
    ]


@pytest.mark.parametrize(
    "failed, name",
    [
        (
            Message(None, notes={"error": {"type": "BadRequestError"}}),
            "BadRequestError",
        ),
        (_reply("cut", status="incomplete"), "ResponsesNotCompleted"),
        (Message("no status"), "ResponsesNotCompleted"),
        (_reply(""), "EmptyReply"),
        (_reply(None), "EmptyReply"),
    ],
)
def test_the_first_failed_turn_ends_the_conversation(tmp_path, report, failed, name):
    probe = _probe(
        tmp_path,
        _entry("a", "b", "c", tool_call_condition=CONDITION, judge_rubric=RUBRIC),
        judge_turns="all",
    )
    attempts, target = _run(probe, [_reply("1"), failed, _reply("3")])

    assert len(target.prompts) == 2, "no request after the failed turn"
    assert attempts[1].notes["sequence"]["failure"] == name
    assert "failure" not in attempts[0].notes["sequence"]
    for attempt in attempts:
        assert (
            "tool_call_condition" not in attempt.prompt.notes
            and "judge_rubric" not in attempt.prompt.notes
        ), "a failed conversation is not graded"


@pytest.mark.parametrize(
    "judge_turns, judged", [("final", [False, False, True]), ("all", [True] * 3)]
)
def test_judge_turns_selects_the_judged_turns(tmp_path, report, judge_turns, judged):
    probe = _probe(
        tmp_path,
        _entry(
            "a",
            "b",
            "c",
            tool_call_condition=CONDITION,
            judge_rubric=RUBRIC,
            judge_description="d",
        ),
        judge_turns=judge_turns,
    )
    attempts, _ = _run(probe, [_reply("1"), _reply("2"), _reply("3")])

    assert [
        "judge_rubric" in a.prompt.notes for a in attempts
    ] == judged, "the rubric travels on the judged turns only"
    assert [
        "judge_description" in a.prompt.notes for a in attempts
    ] == judged, "so does the judge description"
    assert all(
        a.prompt.notes["tool_call_condition"] == CONDITION for a in attempts
    ), "the tool-call condition costs no request, so every turn carries it"


def test_unknown_judge_turns_is_refused(tmp_path):
    with pytest.raises(ValueError, match="judge_turns"):
        _probe(tmp_path, _entry("a"), judge_turns="some")


def test_scripts_with_non_user_messages_are_skipped(tmp_path):
    source = tmp_path / "script.jsonl"
    entries = [
        _entry("a", "b"),
        {"messages": [{"role": "user", "content": "a"}, {"role": "tool"}]},
        {"messages": []},
        {"no_messages": True},
    ]
    source.write_text("".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8")
    probe = SequentialInjection(
        config_root={
            "probes": {
                "injection": {
                    "SequentialInjection": {"conversation_source": str(source)}
                }
            }
        }
    )
    assert [s.texts for s in probe.scripts] == [["a", "b"]]


def test_conversations_run_side_by_side(tmp_path, report):
    source = tmp_path / "script.json"
    source.write_text(json.dumps([_entry("a1", "a2"), _entry("b1")]), encoding="utf-8")
    probe = SequentialInjection(
        config_root={
            "probes": {
                "injection": {
                    "SequentialInjection": {"conversation_source": str(source)}
                }
            }
        }
    )
    attempts, target = _run(probe, [_reply("1"), _reply("2"), _reply("3")])
    assert [
        (a.notes["sequence"]["conversation"], a.notes["sequence"]["turn_index"])
        for a in attempts
    ] == [(0, 0), (1, 0), (0, 1)], "turns advance breadth first"
    assert [t.content.text for t in target.prompts[2].turns] == ["a1", "1", "a2"]
