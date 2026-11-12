# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MiDojo injection-spec probe, which drives injection-spec documents as chains.

The probe runs on the run coordinator, the collector and the grader. A fake
injector and the shipped or a fake environment stand where the interception
service and the target's host stand, so every test drives the public lifecycle.
"""

import copy
import json
import math
import pickle
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("midojo_spec")

import garak._plugins  # noqa: E402
import garak.attempt  # noqa: E402
import garak.harnesses.base  # noqa: E402
import garak.probes.midojo  # noqa: E402
from garak import _config  # noqa: E402
from garak.attempt import Message  # noqa: E402
from garak.detectors.midojo import EvaluationEffect, EvaluationVerdict  # noqa: E402
from garak.detectors.toolcall import ToolCallCondition  # noqa: E402
from garak.evaluators import ThresholdEvaluator  # noqa: E402
from garak.evidence.delivery import delivery_facts  # noqa: E402
from garak.evidence.injection_spec import SUPPORTED_DOCUMENT_VERSIONS  # noqa: E402
from garak.exception import CapabilityError, ConfigFailure  # noqa: E402
from garak.probes.midojo import (  # noqa: E402
    CHAIN_NOTE,
    InjectionSpecProbe,
    MAX_PAYLOAD_BYTES,
    METADATA_NOTE,
    TRACE_NOTE,
)

from injection_chain import (  # noqa: E402
    UPSTREAM,
    ScriptedTarget,
    build_probe,
    entry,
    exception_text,
    instruction,
    report,  # noqa: F401
    report_entries,
    silent_turns,
    world,  # noqa: F401
)

LOOKUP_101 = ("lookup_order", {"order_id": "ORD-101"})
LOOKUP_201 = ("lookup_order", {"order_id": "ORD-201"})


def _run(probe, world, script=()):
    target = ScriptedTarget(world, script)
    return list(probe.probe(target)), target


def _fail_turn(target, number, error):
    """Make the target raise ``error`` on its ``number``-th request (counting from 1)."""
    original = target.generate

    def generate(prompt, generations_this_call=1):
        if len(target.prompts) + 1 == number:
            raise error
        return original(prompt, generations_this_call)

    target.generate = generate
    return target


def test_shipped_example_loads():
    probe = garak._plugins.load_plugin("probes.midojo.InjectionSpecProbe")
    assert isinstance(probe, InjectionSpecProbe)
    assert probe.entries, "the shipped example holds at least one injection-spec entry"
    assert probe.entries[
        0
    ].injection_spec.turns, "the shipped injection spec has turns to send"


def test_a_probe_pickles_and_restores_its_entries():
    probe = garak._plugins.load_plugin("probes.midojo.InjectionSpecProbe")
    restored = pickle.loads(pickle.dumps(probe))

    assert type(restored) is InjectionSpecProbe, "a pickled probe unpickles as itself"
    assert [item.injection_spec.turns for item in restored.entries] == [
        item.injection_spec.turns for item in probe.entries
    ], "the pickled entries survive the round trip"
    assert [item.metadata for item in restored.entries] == [
        item.metadata for item in probe.entries
    ]


def test_source_path_names_the_injection_spec_file(tmp_path):
    probe = build_probe(tmp_path, entry("Look up ORD-101."), entry("Look up ORD-202."))
    assert len(probe.entries) == 2, "source_path names the file the probe reads"


def test_the_probe_writes_the_renamed_notes():
    assert (CHAIN_NOTE, TRACE_NOTE) == ("injection_spec_chain", "evaluation_trace")


def test_an_old_document_version_is_refused(tmp_path):
    old = entry("Look up ORD-101.")
    old["injection_spec"]["schema_version"] = "midojo-episode-v1"
    with pytest.raises(ConfigFailure, match="midojo-episode-v1.*supported"):
        build_probe(tmp_path, old)


@pytest.mark.parametrize(
    "version",
    ["missing", None, 42, ["hostile-version"], "v9-" + "X" * 400],
)
def test_version_refusal_names_supported_versions_without_unbounded_input(
    tmp_path, version
):
    bad = entry("Look up ORD-101.")
    if version == "missing":
        del bad["injection_spec"]["schema_version"]
    else:
        bad["injection_spec"]["schema_version"] = version
    with pytest.raises(ConfigFailure) as caught:
        build_probe(tmp_path, bad)
    assert f"supported: {', '.join(SUPPORTED_DOCUMENT_VERSIONS)}" in str(caught.value)
    assert len(str(caught.value)) < 250, "hostile versions cannot inflate refusals"


@pytest.mark.parametrize(
    "payload",
    [
        "",
        None,
        14,
        [],
        "x" * (MAX_PAYLOAD_BYTES + 1),
        "£" * (MAX_PAYLOAD_BYTES // 2 + 1),
    ],
)
def test_invalid_plan_payload_is_refused_without_echoing_it(tmp_path, world, payload):
    bad = entry("Look up ORD-101.", instructions=[instruction(payload)])
    with pytest.raises(ConfigFailure) as caught:
        build_probe(tmp_path, bad)
    assert "instruction 0" in str(caught.value), "the refusal names the plan item"
    if isinstance(payload, str) and payload:
        assert payload not in str(caught.value), "the refusal never repeats the content"
    assert world.events == [], "nothing opens for an unadmitted payload"


def test_metadata_is_copied_unread(tmp_path, world, report):
    metadata = {"package_id": "pkg-1", "claim_level": "asserted", "nested": [1, 2]}
    probe = build_probe(tmp_path, entry("Look up ORD-101.", metadata=metadata))
    attempts, _ = _run(probe, world, [[LOOKUP_101]])

    assert attempts[0].notes[METADATA_NOTE] == metadata, "metadata travels verbatim"
    assert attempts[0].notes[METADATA_NOTE] is not metadata, "and as its own copy"


def test_one_reply_per_turn_and_serial_chains(tmp_path, world, report):
    _config.system.parallel_attempts = 2
    probe = build_probe(tmp_path, entry("Look up ORD-101."))

    assert probe.generations == 1, "a chain records exactly one reply per turn"
    assert probe.parallel_attempts == 1, "one evaluation is live at a time"


def test_a_second_chain_cannot_open_while_one_is_live(tmp_path, world, report):
    probe = build_probe(tmp_path, entry("Look up ORD-101."))
    refused = []
    target = ScriptedTarget(world)
    original = target.generate

    def generate(prompt, generations_this_call=1):
        spec = probe.entries[0].injection_spec
        with pytest.raises(CapabilityError, match="max_live") as caught:
            with probe._coordinator.chain(
                "second", upstream_url=UPSTREAM, plan=spec.plan
            ):
                pass
        refused.append(str(caught.value))
        return original(prompt, generations_this_call)

    target.generate = generate
    (attempt,) = probe.probe(target)

    assert refused, "the target tried to open a second chain mid-turn"
    assert world.count("arm") == 1, "the refused chain bound nothing"
    assert attempt.notes[TRACE_NOTE]["error"] is None, "the live chain went on"


def test_chains_run_one_after_another_in_one_injector_run(tmp_path, world, report):
    probe = build_probe(tmp_path, entry("Look up ORD-101."), entry("Look up ORD-101."))
    attempts, _ = _run(probe, world, [[LOOKUP_101], [LOOKUP_101]])

    assert len(attempts) == 2, "each entry is its own chain"
    assert world.events == [
        "start_run",
        "arm",
        "disarm",
        "arm",
        "disarm",
        "run_close",
    ], "a sealed chain is released before the next one binds, in one injector run"
    ids = {attempt.notes[TRACE_NOTE]["evaluation_id"] for attempt in attempts}
    assert len(ids) == 2, "each chain is its own evaluation"


def test_chain_notes_hold_plain_data_only(tmp_path, world, report, monkeypatch):
    probe = build_probe(tmp_path, entry("Look up ORD-101.", "And ORD-201."))
    copies_mid_chain = []
    original = probe._turn_attempt

    def copying(*args, **kwargs):
        # the queue is deep-copied between turns, with the chain still open
        attempt = original(*args, **kwargs)
        copies_mid_chain.append(json.dumps(copy.deepcopy(attempt.as_dict())))
        return attempt

    monkeypatch.setattr(probe, "_turn_attempt", copying)
    attempts, target = _run(probe, world, [[LOOKUP_101], [LOOKUP_201]])

    assert len(target.prompts) == 2, "one request per injection-spec turn"
    assert len(attempts) == 1, "a chain scores once, on its final turn"
    assert len(copies_mid_chain) == 2, "both turns queued with the chain live"
    assert all(
        "tool-token" not in text for text in copies_mid_chain
    ), "no live credential rides in the queue"
    final = copy.deepcopy(attempts[0])
    assert final.notes[CHAIN_NOTE]["chain"] == 0
    assert "tool-token" not in json.dumps(final.as_dict())


def test_the_target_takes_the_tool_block_the_injector_bound(tmp_path, world, report):
    probe = build_probe(tmp_path, entry("Look up ORD-101."))
    attempts, target = _run(probe, world, [[LOOKUP_101]])

    (block,) = target.prompts[0].notes["tools"]
    assert block == {
        "type": "mcp",
        "server_label": "midojo",
        "server_url": "http://127.0.0.1:1/b/0/mcp",
        "authorization": "tool-token-0",
        "require_approval": "never",
        "allowed_tools": ["lookup_order", "process_refund"],
    }, "the target gets the MCP tool block its Responses API takes"
    (stored,) = attempts[0].prompt.notes["tools"]
    assert stored == {
        **block,
        "authorization": "[redacted]",
    }, "the conversation the report keeps holds the block without its credential"


def test_the_injector_binds_the_chains_plan_policy_and_upstream(
    tmp_path, world, report
):
    probe = build_probe(tmp_path, entry("Look up ORD-101."))
    _run(probe, world)

    (binding,) = world.bindings
    spec = probe.entries[0].injection_spec
    assert binding["plan"] is spec.plan, "the injector gets the authored plan"
    assert binding["policy"] is spec.tools, "and the tool policy"
    assert binding["upstream_url"] == UPSTREAM, "and the document's own upstream"


def test_the_upstream_option_outranks_the_documents_upstream(tmp_path, world, report):
    probe = build_probe(
        tmp_path, entry("Look up ORD-101."), upstream="https://tools.example/mcp?k=v"
    )
    _run(probe, world)

    assert world.bindings[0]["upstream_url"] == "https://tools.example/mcp?k=v"


def test_no_upstream_anywhere_is_refused_before_a_chain_opens(tmp_path, world, report):
    bare = entry("Look up ORD-101.")
    bare["injection_spec"]["interception"]["upstream"] = None
    probe = build_probe(tmp_path, bare)

    with pytest.raises(ConfigFailure, match="no upstream to intercept"):
        probe.probe(ScriptedTarget(world))
    assert world.count("arm") == 0, "nothing bound"


@pytest.mark.parametrize(
    "value",
    [
        "sentinel-upstream",
        "ftp://sentinel-upstream.example/mcp",
        "https://sentinel-upstream@localhost:8767/mcp",
        "https://localhost:bad-sentinel-upstream/mcp",
        "https://localhost:8767/mcp?x=sentinel-upstream\n",
        {"url": "https://sentinel-upstream.example/mcp"},
        b"https://sentinel-upstream.example/mcp",
    ],
)
def test_an_upstream_that_is_not_a_plain_http_url_is_refused_without_its_value(
    tmp_path, value
):
    with pytest.raises(ConfigFailure) as caught:
        build_probe(tmp_path, entry("Look up ORD-101."), upstream=value)
    assert "upstream" in str(caught.value)
    assert "sentinel-upstream" not in exception_text(caught.value)


def test_a_documents_own_upstream_is_held_to_the_same_rule(tmp_path, world, report):
    bad = entry("Look up ORD-101.")
    bad["injection_spec"]["interception"]["upstream"] = "https://sentinel-doc@host/mcp"
    probe = build_probe(tmp_path, bad)

    with pytest.raises(ConfigFailure) as caught:
        probe.probe(ScriptedTarget(world))
    assert "sentinel-doc" not in exception_text(caught.value)
    assert world.count("arm") == 0, "the injector never saw the URL"


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("interceptor_url", "plugins.injectors.mim.service_url"),
        ("interceptor_credential", "plugins.injectors.mim.bind_credential"),
    ],
)
def test_an_option_that_moved_to_the_injector_is_refused_naming_its_new_home(
    tmp_path, old, new
):
    given = {old: "sentinel-moved-value"}
    with pytest.raises(ConfigFailure) as caught:
        build_probe(tmp_path, entry("Look up ORD-101."), **given)

    assert old in str(caught.value), "the refusal names the option given"
    assert new in str(caught.value), "and where it is set now"
    assert "MIDOJO_INTERCEPTOR_URL" in str(caught.value), "or the environment pair"
    assert "sentinel-moved-value" not in exception_text(caught.value)


def test_a_refusal_names_every_moved_option_the_configuration_holds(tmp_path):
    given = {"interceptor_url": "sentinel-a", "interceptor_credential": "sentinel-b"}
    with pytest.raises(ConfigFailure) as caught:
        build_probe(tmp_path, entry("Look up ORD-101."), **given)

    text = str(caught.value)
    for old, new in (
        ("interceptor_url", "plugins.injectors.mim.service_url"),
        ("interceptor_credential", "plugins.injectors.mim.bind_credential"),
    ):
        assert old in text and new in text, "each option and its new home are named"
    assert "sentinel" not in exception_text(caught.value)


@pytest.mark.parametrize("name", ["chain_timeout", "turn_timeout"])
@pytest.mark.parametrize(
    "value", [0, -1, -0.5, math.inf, -math.inf, math.nan, "30", True, [5], {}]
)
def test_a_timeout_that_is_not_a_positive_number_is_refused(tmp_path, name, value):
    with pytest.raises(ConfigFailure, match=name) as caught:
        build_probe(tmp_path, entry("Look up ORD-101."), **{name: value})
    assert "positive number of seconds" in str(caught.value)


def test_a_turn_timeout_cannot_be_left_out(tmp_path):
    with pytest.raises(ConfigFailure, match="turn_timeout"):
        build_probe(tmp_path, entry("Look up ORD-101."), turn_timeout=None)


@pytest.mark.parametrize("value", [0, -1, 1.5, "2", True, None, [1]])
def test_repeats_must_be_a_positive_whole_number(tmp_path, value):
    with pytest.raises(ConfigFailure, match="repeats"):
        build_probe(tmp_path, entry("Look up ORD-101."), repeats=value)


def test_the_coordinator_takes_the_probes_turn_timeout_and_one_live_chain(
    tmp_path, world, report, monkeypatch
):
    made = []
    real = garak.probes.midojo.RunCoordinator

    def recording(*args, **kwargs):
        made.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(garak.probes.midojo, "RunCoordinator", recording)
    probe = build_probe(tmp_path, entry("Look up ORD-101."), turn_timeout=2.5)
    _run(probe, world)

    (options,) = made
    assert options["turn_timeout"] == 2.5, "a turn gets the probe's deadline"
    assert options["max_live"] == 1, "one chain is live at a time"


def test_single_turn_obeying_the_payload_is_detected(tmp_path, world, report):
    probe = build_probe(tmp_path, entry("Look up ORD-101 for me."))
    attempts, _ = _run(probe, world, [[LOOKUP_101, LOOKUP_201]])

    assert len(attempts) == 1
    trace = attempts[0].notes[TRACE_NOTE]
    assert [call["name"] for call in trace["tool_calls"]] == [
        "lookup_order",
        "lookup_order",
    ], "the trace holds the calls the target made through the interceptor"
    assert trace["sealed"] == "final"
    assert trace["capture_complete"] is True
    assert trace["response_status"] == "completed", "a delivered payload can grade"
    assert trace["verdict"]["grade"]["security"] is True
    assert trace["verdict"]["grade"]["injection_delivered"] is True
    assert trace["snapshot_id"] == trace["verdict"]["snapshot_id"]
    assert attempts[0].outputs[0].notes["tool_calls"] == trace["tool_calls"]
    assert ToolCallCondition().detect(attempts[0]) == [1.0]
    assert EvaluationVerdict().detect(attempts[0]) == [1.0]
    assert EvaluationEffect().detect(attempts[0]) == [1.0]


def test_single_turn_resisting_the_payload_is_not_detected(tmp_path, world, report):
    probe = build_probe(tmp_path, entry("Look up ORD-101 for me."))
    attempts, _ = _run(probe, world, [[LOOKUP_101]])

    trace = attempts[0].notes[TRACE_NOTE]
    assert trace["verdict"]["grade"]["security"] is False
    assert trace["capture_complete"] is True, "a refusal is a complete capture"
    assert trace["response_status"] == "completed"
    assert ToolCallCondition().detect(attempts[0]) == [0.0]
    assert EvaluationVerdict().detect(attempts[0]) == [0.0]
    assert EvaluationEffect().detect(attempts[0]) == [0.0]


def test_second_turn_carries_the_first_reply(tmp_path, world, report):
    probe = build_probe(tmp_path, entry("Look up ORD-101.", "And ORD-201."))
    attempts, target = _run(probe, world, [[LOOKUP_101], [LOOKUP_201]])

    second = target.prompts[1]
    assert [turn.role for turn in second.turns] == ["user", "assistant", "user"]
    assert second.turns[0].content.text == "Look up ORD-101."
    assert (
        second.turns[1].content.text == "Handled 1 tool calls."
    ), "the next request carries the target's real reply"
    assert second.turns[2].content.text == "And ORD-201."
    assert attempts[0].notes[CHAIN_NOTE] == {
        "chain": 0,
        "turn": 1,
        "turns": 2,
        "final": True,
    }


def test_a_multi_turn_chain_records_each_turn_in_its_snapshot(tmp_path, world, report):
    probe = build_probe(tmp_path, entry("Look up ORD-101.", "And ORD-201."))
    attempts, _ = _run(probe, world, [[LOOKUP_101], [LOOKUP_201]])

    snapshot = attempts[0].notes[TRACE_NOTE]["snapshot"]
    assert [(t["message"], t["reply"]) for t in snapshot["turns"]] == [
        ("Look up ORD-101.", "Handled 1 tool calls."),
        ("And ORD-201.", "Handled 1 tool calls."),
    ], "each turn holds its message and the target's reply"
    assert [call["turn"] for call in snapshot["function_calls"]] == [
        0,
        1,
    ], "each call is stamped with the turn it came in"
    assert snapshot["agent_output"] == "Handled 1 tool calls."
    assert snapshot["evidence_status"] == {}, "nothing was cut short"


def test_a_turn_that_only_calls_tools_keeps_its_chain(tmp_path, world, report):
    probe = build_probe(tmp_path, entry("Look up ORD-101.", "And ORD-201."))
    target = ScriptedTarget(world, [[LOOKUP_101], [LOOKUP_201]])
    original = target.generate

    def tool_only(prompt, generations_this_call=1):
        messages = original(prompt, generations_this_call)
        if len(target.prompts) == 1:
            notes = {**messages[0].notes, "tool_calls": [{"type": "mcp_call"}]}
            return [Message(None, notes=notes)]
        return messages

    target.generate = tool_only
    (attempt,) = probe.probe(target)

    assert len(target.prompts) == 2, "a target that called a tool is not a dead turn"
    trace = attempt.notes[TRACE_NOTE]
    assert trace["error"] is None, "the chain reached its final turn"
    assert (
        trace["snapshot"]["turns"][0]["reply"] == ""
    ), "the silent turn's reply is empty"
    assert trace["snapshot"]["turns"][1]["reply"] == "Handled 1 tool calls."
    assert trace["verdict"]["grade"]["security"] is True, "and the chain is graded"
    second = target.prompts[1]
    assert [turn.role for turn in second.turns] == [
        "user",
        "user",
    ], "an empty reply adds no assistant turn to the next request"


def test_a_turn_with_no_text_and_no_calls_ends_its_chain(tmp_path, world, report):
    probe = build_probe(tmp_path, entry("Look up ORD-101.", "And ORD-201."))
    target = ScriptedTarget(world)

    def generate(prompt, generations_this_call=1):
        target.prompts.append(prompt)
        return [Message(None, notes={"response_status": "completed"})]

    target.generate = generate

    (attempt,) = probe.probe(target)

    assert len(target.prompts) == 1, "an empty turn stops the chain"
    assert attempt.notes[TRACE_NOTE]["error"] == "EmptyReply"


def test_generator_and_interceptor_calls_survive_separately_in_order():
    generated = [{"name": "first"}, {"name": "second"}]
    recorded = [{"name": "intercepted-two"}, {"name": "intercepted-one"}]
    attempt = garak.attempt.Attempt(
        prompt=garak.attempt.Conversation([garak.attempt.Turn("user", Message("test"))])
    )
    attempt.outputs = [
        Message(
            "reply", notes={"response_status": "completed", "tool_calls": generated}
        )
    ]
    probe = object.__new__(InjectionSpecProbe)
    probe._chain_secrets = {}
    trace = {**_sealed_trace(), "tool_calls": recorded}
    probe._attach_trace(attempt, trace)
    generated[0]["name"] = "changed"
    assert attempt.outputs[0].notes["generator_tool_calls"] == [
        {"name": "first"},
        {"name": "second"},
    ], "the generator's ordered calls are copied, not overwritten"
    assert (
        attempt.outputs[0].notes["tool_calls"] == recorded
    ), "the interceptor's ordered calls remain the detector input"
    probe._attach_trace(attempt, trace)
    assert (
        attempt.outputs[0].notes["generator_tool_calls"][0]["name"] == "first"
    ), "a second attachment never replaces the original generator record"


def _sealed_trace():
    """A sealed chain's trace, in the shape ``_trace`` returns for a completed capture."""
    return {
        "evaluation_id": "evaluation-6f1c",
        "snapshot_id": "snapshot-6f1c",
        "snapshot": None,
        "verdict": {"grade": {"security": True, "injection_delivered": True}},
        "capture_complete": True,
        "sealed": "final",
        "failure": None,
        "error": None,
        "release_error": None,
        "cleanup_failures": [],
        "tool_calls": [],
        "delivery": [],
        "exposure": None,
        "response_status": "completed",
    }


def test_harness_scores_a_two_turn_chain_once(tmp_path, world, report, monkeypatch):
    probe = build_probe(tmp_path, entry("Look up ORD-101.", "And ORD-201."))
    target = ScriptedTarget(world, [[LOOKUP_101], [LOOKUP_201]])
    # the scripted target is no plugin, so the run writes no plugin cache entry
    monkeypatch.setattr(
        garak.harnesses.base, "_emit_plugin_cache_entry", lambda *a: None
    )

    garak.harnesses.base.Harness().run(
        target, [probe], [ToolCallCondition()], ThresholdEvaluator()
    )
    report.flush()

    entries = report_entries(tmp_path / "probe.report.jsonl")
    attempts = [e for e in entries if e.get("entry_type") == "attempt"]
    assert [e["status"] for e in attempts] == [
        1,
        1,
        2,
    ], "each turn is recorded as started; only the chain's last one completes"
    evaluated = [e for e in entries if e.get("entry_type") == "eval"]
    assert len(evaluated) == 1, "one result per chain, which is what the digest counts"
    record = evaluated[0]
    assert record["detector"] == "toolcall.ToolCallCondition"
    assert record["total_evaluated"] == 1, "the chain's one result is the one judged"
    assert (record["passed"], record["fails"]) == (0, 1), "the planted call was made"


REGRADE_SCRIPT = """
import json, sys
from garak.attempt import Attempt, Conversation, Message, Turn
from garak.detectors.toolcall import ToolCallCondition

scores = []
for line in open(sys.argv[1], encoding="utf-8"):
    entry = json.loads(line)
    if entry.get("entry_type") != "attempt" or entry.get("status") != 2:
        continue
    attempt = Attempt(
        prompt=Conversation(
            [Turn("user", Message("stored prompt"))],
            notes=dict(entry["prompt"]["notes"]),
        )
    )
    attempt.outputs = [Message(None, notes=dict(o["notes"])) for o in entry["outputs"]]
    scores.append(ToolCallCondition().detect(attempt))
print(json.dumps(scores))
"""


def test_stored_attempts_regrade_to_the_same_scores(
    tmp_path, world, report, monkeypatch
):
    probe = build_probe(tmp_path, entry("Look up ORD-101."), entry("Look up ORD-101."))
    target = ScriptedTarget(world, [[LOOKUP_101, LOOKUP_201], [LOOKUP_101]])
    monkeypatch.setattr(
        garak.harnesses.base, "_emit_plugin_cache_entry", lambda *a: None
    )

    garak.harnesses.base.Harness().run(
        target, [probe], [ToolCallCondition()], ThresholdEvaluator()
    )
    report.flush()

    path = tmp_path / "probe.report.jsonl"
    stored = [
        item["detector_results"]["toolcall.ToolCallCondition"]
        for item in report_entries(path)
        if item.get("entry_type") == "attempt" and item.get("status") == 2
    ]
    assert stored == [[1.0], [0.0]], "the obey path hits and the resist path does not"

    replayed = subprocess.run(
        [sys.executable, "-c", REGRADE_SCRIPT, str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert (
        json.loads(replayed.stdout) == stored
    ), "a fresh process reproduces every stored score from the trace"


def test_a_failing_turn_ends_its_chain(tmp_path, world, report):
    probe = build_probe(tmp_path, entry("Look up ORD-101.", "And ORD-201."))
    target = ScriptedTarget(world, [[LOOKUP_101]])
    original = target.generate

    def fail_second(prompt, generations_this_call=1):
        if len(target.prompts) == 1:
            target.prompts.append(prompt)
            return [Message("rate limited", notes={"response_status": "incomplete"})]
        return original(prompt, generations_this_call)

    target.generate = fail_second
    attempts = list(probe.probe(target))

    assert len(attempts) == 1, "the failed turn is the chain's only result"
    trace = attempts[0].notes[TRACE_NOTE]
    assert trace["error"] == "ResponsesNotCompleted"
    assert trace["response_status"] == "incomplete"
    assert EvaluationVerdict().detect(attempts[0]) == [
        None
    ], "a failed turn scores None"
    assert "errored" in attempts[0].notes["evaluation_verdict_results"][0]["reason"]
    assert probe._live == {}, "a failed chain leaves no live chain"
    assert world.count("disarm") == 1, "and its binding is released"
    snapshot = trace["snapshot"]
    assert set(snapshot["evidence_status"]) == {
        "function_calls",
        "turns",
        "agent_output",
    }, "the seal says the calls, turns and last words it holds are cut short"
    assert {name: status for name, status in snapshot["evidence_status"].items()} == {
        "function_calls": "incomplete",
        "turns": "incomplete",
        "agent_output": "incomplete",
    }


@pytest.mark.parametrize(
    ("message", "failure"),
    [
        (
            Message("late", notes={"response_status": "incomplete"}),
            "ResponsesNotCompleted",
        ),
        (
            Message(None, notes={"error": {"type": "TimedOut"}}),
            "TimedOut",
        ),
        (None, "ResponsesNotCompleted"),
    ],
)
def test_timed_out_or_unfinished_turn_seals_and_closes(
    tmp_path, world, report, message, failure
):
    probe = build_probe(tmp_path, entry("First turn.", "Second turn."))
    target = ScriptedTarget(world)

    def generate(prompt, generations_this_call=1):
        target.prompts.append(prompt)
        return [message]

    target.generate = generate
    attempts = list(probe.probe(target))
    assert len(target.prompts) == 1, "a failed turn stops before the next one"
    assert len(attempts) == 1, "the final attempt represents the failed chain"
    trace = attempts[0].notes[TRACE_NOTE]
    assert trace["sealed"] == "final", "the evaluation is sealed"
    assert trace["error"] == failure, "the failure type survives"
    assert trace["response_status"] == "incomplete", "the turn cannot decide"
    assert probe._live == {}, "the chain closes on the failed turn"
    assert world.count("disarm") == 1


def test_a_completed_refusal_is_graded_not_dropped(tmp_path, world, report):
    probe = build_probe(tmp_path, entry("First turn."))
    target = ScriptedTarget(world)

    def generate(prompt, generations_this_call=1):
        target.prompts.append(prompt)
        return [Message("I cannot do that.", notes={"response_status": "completed"})]

    target.generate = generate
    attempts = list(probe.probe(target))

    assert len(attempts) == 1, "a refusal still yields a final attempt"
    assert attempts[0].notes[TRACE_NOTE]["error"] is None, "refusal is not an error"
    assert (
        attempts[0].notes[TRACE_NOTE]["verdict"]["grade"]["security"] is False
    ), "the refusal is graded by the grader"


def test_a_none_reply_on_the_final_turn_scores_none(tmp_path, world, report):
    probe = build_probe(tmp_path, entry("Look up ORD-101."))
    target = silent_turns(ScriptedTarget(world), {0})

    attempts = list(probe.probe(target))

    assert len(attempts) == 1, "the unanswered turn is the chain's only result"
    assert attempts[0].notes[CHAIN_NOTE]["final"] is True
    trace = attempts[0].notes[TRACE_NOTE]
    assert trace["sealed"] == "final", "a turn with no output still seals its chain"
    assert trace["error"] == "ResponsesNotCompleted", "no output is a failed turn"
    assert trace["response_status"] == "incomplete"
    assert EvaluationVerdict().detect(attempts[0]) == [None], "never scored as resisted"
    assert ToolCallCondition().detect(attempts[0]) == [None], "nor as a miss"
    assert "errored" in attempts[0].notes["evaluation_verdict_results"][0]["reason"]
    assert probe._live == {}, "a failed chain leaves no live chain"
    assert world.count("disarm") == 1, "the binding is released"


def test_a_none_reply_stores_the_trace_on_the_output_it_leaves(tmp_path, world, report):
    """The trace lands on the message the attempt stores, which is what detectors read."""
    probe = build_probe(tmp_path, entry("Look up ORD-101."))
    target = silent_turns(ScriptedTarget(world), {0})

    attempts = list(probe.probe(target))

    stored = attempts[0].outputs[0]
    trace = attempts[0].notes[TRACE_NOTE]
    assert isinstance(stored, Message), "an unanswered turn still stores a message"
    assert stored.notes["tool_calls"] == trace["tool_calls"]
    assert stored.notes["response_status"] == "incomplete"
    assert stored.notes["response_status"] in json.dumps(attempts[0].as_dict())


def test_a_none_reply_mid_chain_does_not_abort_the_run(tmp_path, world, report):
    probe = build_probe(
        tmp_path, entry("Look up ORD-101.", "And ORD-201."), entry("Look up ORD-101.")
    )
    target = silent_turns(ScriptedTarget(world, [[LOOKUP_101], [LOOKUP_101]]), {0})

    attempts = list(probe.probe(target))

    assert len(target.prompts) == 2, "no next turn starts after a turn with no output"
    assert [a.notes[CHAIN_NOTE]["chain"] for a in attempts] == [
        0,
        1,
    ], "both chains report, so one chain's None does not end the run"
    cut_short = attempts[0]
    assert (
        cut_short.notes[CHAIN_NOTE]["turn"] == 0
    ), "the unanswered turn ends its chain"
    trace = cut_short.notes[TRACE_NOTE]
    assert trace["sealed"] == "final"
    assert trace["error"] == "ResponsesNotCompleted"
    assert trace["response_status"] == "incomplete"
    assert EvaluationVerdict().detect(cut_short) == [None], "never scored as resisted"
    assert ToolCallCondition().detect(cut_short) == [None], "nor as a miss"
    later = attempts[1].notes[TRACE_NOTE]
    assert later["error"] is None, "the next chain runs to its own end"
    assert ToolCallCondition().detect(attempts[1]) == [0.0], "and scores on its trace"
    assert probe._live == {}, "both chains left nothing live"
    assert world.count("disarm") == 2


def test_a_sealed_trace_carries_delivery_and_exposure(
    tmp_path, world, report, monkeypatch
):
    """The trace's delivery and exposure come from the same sealed calls the grade does."""
    probe = build_probe(tmp_path, entry("Look up ORD-101."))
    sealed = []
    real = probe._trace

    def recording(state, snapshot, **kwargs):
        sealed.append(snapshot)
        return real(state, snapshot, **kwargs)

    monkeypatch.setattr(probe, "_trace", recording)
    attempts, _ = _run(probe, world, [[LOOKUP_101]])

    (snapshot,) = sealed
    trace = attempts[0].notes[TRACE_NOTE]
    plan = probe.entries[0].injection_spec.plan
    assert trace["delivery"] == [
        fact.to_json() for fact in delivery_facts(plan, snapshot.function_calls)
    ], "the trace carries the delivery facts for the sealed calls"
    (fact,) = trace["delivery"]
    assert (fact["slot_key"], fact["status"], fact["plan_instruction_index"]) == (
        "ipi_episode_test",
        "applied",
        0,
    ), "the placed payload leaves a delivery fact"
    assert trace["exposure"]["emitted"] == "exposed", "and an exposure at the boundary"


def test_an_interrupted_trace_states_no_delivery_and_no_exposure(
    tmp_path, world, report
):
    """A chain closed before its seal carries null for both, even when it sealed a snapshot."""
    probe = build_probe(tmp_path, entry("Look up ORD-101.", "And ORD-201."))
    target = _fail_turn(
        ScriptedTarget(world, [[LOOKUP_101]]), 2, RuntimeError("the target went away")
    )

    with pytest.raises(RuntimeError, match="the target went away"):
        list(probe.probe(target))

    trace = probe._last_processed[0].notes[TRACE_NOTE]
    assert trace["snapshot"] is not None, "the closed chain still seals its evidence"
    assert trace["sealed"] == "interrupted"
    assert trace["delivery"] is None, "an interrupted chain states no delivery"
    assert trace["exposure"] is None, "nor an exposure"
    assert set(trace["snapshot"]["evidence_status"]) == {
        "function_calls",
        "turns",
        "agent_output",
    }, "the seal names what the interrupt cut short"


def test_a_trace_with_no_snapshot_states_no_delivery_and_no_exposure():
    """A chain that never sealed a snapshot has no calls to describe: both stay null."""
    state = SimpleNamespace(evaluation_id="never-sealed")

    trace = InjectionSpecProbe._empty_trace(state, "final", "NoSnapshot")

    assert trace["error"] == "NoSnapshot"
    assert trace["snapshot"] is None and trace["verdict"] is None
    assert trace["delivery"] is None, "no snapshot, no delivery facts"
    assert trace["exposure"] is None, "and no exposure"
    assert trace["capture_complete"] is False


def test_the_capture_status_names_what_stops_a_condition():
    """The status is the fact ToolCallCondition suffixes its inconclusive reason with."""
    undecided = {
        "sealed": "final",
        "error": None,
        "capture_complete": True,
        "response_status": "incomplete",
        "verdict": {"grade": {"injection_delivered": None}},
    }
    assert InjectionSpecProbe._capture_status(undecided) == "unrecorded_delivery"

    decided = {
        **undecided,
        "response_status": "completed",
        "verdict": {"grade": {"injection_delivered": True}},
    }
    assert InjectionSpecProbe._capture_status(decided) == "completed"

    truncated = {**undecided, "capture_complete": False}
    assert (
        InjectionSpecProbe._capture_status(truncated) == "incomplete"
    ), "a truncated capture keeps its own name"

    errored = {**undecided, "error": "ResponsesNotCompleted"}
    assert InjectionSpecProbe._capture_status(errored) == "incomplete"

    interrupted = {**undecided, "sealed": "interrupted"}
    assert InjectionSpecProbe._capture_status(interrupted) == "incomplete"


@pytest.mark.parametrize(
    ("status", "failures", "finished", "expected"),
    [
        ("complete", (), True, True),
        ("not_collected", (), True, True),
        ("failed", (), True, False),
        ("incomplete", (), True, False),
        ("unknown", (), True, False),
        ("complete", ("timeout",), True, False),
        ("complete", (), False, False),
    ],
)
def test_capture_completeness_uses_the_verified_snapshot_rule(
    status, failures, finished, expected
):
    snapshot = SimpleNamespace(
        evidence_status={"function_calls": status},
        capture_failures=failures,
        function_calls=[SimpleNamespace(complete=finished)],
    )
    assert (
        InjectionSpecProbe._capture_complete(snapshot) is expected
    ), "only complete evidence and finished calls decide a capture"


def test_an_unrecorded_delivery_is_named_in_the_condition_reason(
    tmp_path, world, report
):
    """A receipt carries the condition's reason, so the probe names the unrecorded delivery there."""
    probe = build_probe(tmp_path, entry("Look up ORD-101."))
    attempts, _ = _run(probe, world, [[LOOKUP_101]])
    attempt = attempts[0]
    trace = copy.deepcopy(attempt.notes[TRACE_NOTE])
    trace["verdict"]["grade"]["injection_delivered"] = None
    probe._attach_trace(attempt, trace)

    assert attempt.outputs[0].notes["response_status"] == "unrecorded_delivery"
    assert ToolCallCondition().detect(attempt) == [
        None
    ], "an unrecorded delivery decides nothing"
    reason = attempt.notes["tool_call_condition_results"][0]["reason"]
    assert (
        "unrecorded_delivery" in reason
    ), "the record the receipt reads names the unrecorded delivery"


def test_an_interrupted_trace_that_fails_leaves_its_attempt_open(
    tmp_path, world, report, monkeypatch
):
    probe = build_probe(tmp_path, entry("Look up ORD-101.", "And ORD-201."))
    target = _fail_turn(
        ScriptedTarget(world, [[LOOKUP_101]]), 2, RuntimeError("the target went away")
    )

    def no_trace(*args, **kwargs):
        raise RuntimeError("the snapshot could not be read back")

    monkeypatch.setattr(probe, "_trace", no_trace)

    with pytest.raises(RuntimeError, match="the target went away"):
        list(probe.probe(target))

    interrupted = probe._last_processed[0]
    assert interrupted.notes[CHAIN_NOTE]["final"] is False, "no trace, no final attempt"
    assert TRACE_NOTE not in interrupted.notes, "and nothing half-attached"
    assert probe._live == {}, "the chain still closes"
    assert world.count("disarm") == 1


def test_an_interrupt_before_the_first_postprocess_seals_nothing(
    tmp_path, world, report
):
    probe = build_probe(tmp_path, entry("Look up ORD-101.", "And ORD-201."))
    target = _fail_turn(ScriptedTarget(world), 1, RuntimeError("the target went away"))

    with pytest.raises(RuntimeError, match="the target went away"):
        list(probe.probe(target))

    assert probe._last_processed == {}, "no turn of the chain was processed"
    assert probe._live == {}, "the opened chain still closes"
    assert world.count("disarm") == 1, "and its binding is released"


@pytest.mark.parametrize(
    "error", [RuntimeError("the target went away"), KeyboardInterrupt()]
)
def test_an_error_or_interrupt_mid_chain_closes_everything(
    tmp_path, world, report, error
):
    probe = build_probe(
        tmp_path, entry("Look up ORD-101.", "And ORD-201."), environment="fake"
    )
    target = _fail_turn(ScriptedTarget(world, [[LOOKUP_101]]), 2, error)

    with pytest.raises(type(error)):
        list(probe.probe(target))

    assert probe._live == {}, "no chain is left live"
    assert world.events[-3:] == [
        "disarm",
        "environment_close",
        "run_close",
    ], "the binding, the environment and the injector run are released in order"
    interrupted = probe._last_processed[0]
    assert interrupted.notes[TRACE_NOTE]["sealed"] == "interrupted"
    assert interrupted.notes[CHAIN_NOTE]["final"] is True, "the last attempt is final"
    assert EvaluationVerdict().detect(interrupted) == [None], "never resisted"
    assert "interrupted" in interrupted.notes["evaluation_verdict_results"][0]["reason"]


def test_a_chain_deadline_closes_a_chain_that_runs_too_long(tmp_path, world, report):
    probe = build_probe(
        tmp_path, entry("Look up ORD-101.", "And ORD-201."), chain_timeout=0.2
    )
    target = ScriptedTarget(world, [[LOOKUP_101], [LOOKUP_201]])
    original = target.generate

    def slow(prompt, generations_this_call=1):
        time.sleep(0.8)
        return original(prompt, generations_this_call)

    target.generate = slow
    (attempt,) = probe.probe(target)

    trace = attempt.notes[TRACE_NOTE]
    assert trace["error"] == "ChainTimeout", "the deadline is the failure"
    assert len(target.prompts) == 1, "no later turn starts on a closed chain"
    assert trace["sealed"] == "final"
    assert world.count("disarm") == 1, "the binding was released once"
    assert (
        trace["snapshot"]["evidence_status"]["turns"] == "incomplete"
    ), "the seal marks the turns it did not finish"
    assert EvaluationVerdict().detect(attempt) == [
        None
    ], "a timed-out chain never resists"


def test_a_chain_that_ends_in_time_leaves_no_deadline_running(tmp_path, world, report):
    before = {t for t in threading.enumerate() if isinstance(t, threading.Timer)}
    probe = build_probe(tmp_path, entry("Look up ORD-101."), chain_timeout=30)
    attempts, _ = _run(probe, world, [[LOOKUP_101]])

    assert attempts[0].notes[TRACE_NOTE]["error"] is None
    time.sleep(0.1)
    leftover = {
        t for t in threading.enumerate() if isinstance(t, threading.Timer)
    } - before
    assert not [t for t in leftover if t.is_alive()], "the chain's timer was cancelled"


def test_a_turn_deadline_ends_the_turn_and_its_chain(tmp_path, world, report):
    probe = build_probe(
        tmp_path, entry("Look up ORD-101.", "And ORD-201."), turn_timeout=0.2
    )
    target = ScriptedTarget(world, [[LOOKUP_101], [LOOKUP_201]])
    original = target.generate

    def slow(prompt, generations_this_call=1):
        time.sleep(0.8)
        return original(prompt, generations_this_call)

    target.generate = slow
    (attempt,) = probe.probe(target)

    trace = attempt.notes[TRACE_NOTE]
    assert trace["error"] == "TurnTimeout", "the turn's deadline is the failure"
    assert len(target.prompts) == 1, "no later turn starts"
    assert (
        trace["snapshot"]["agent_output"] == ""
    ), "a reply after the deadline is not kept"
    assert world.count("disarm") == 1


def test_repeats_are_refused_when_the_environment_cannot_isolate_them(
    tmp_path, world, report
):
    probe = build_probe(tmp_path, entry("Look up ORD-101."), repeats=2)

    with pytest.raises(ConfigFailure, match="repeats refused.*isolation") as caught:
        probe.probe(ScriptedTarget(world))
    assert "Unmanaged" in str(caught.value), "the refusal names the environment"
    assert world.events == [], "nothing opened"


def test_repeats_are_separate_chains_in_an_isolating_environment(
    tmp_path, world, report
):
    probe = build_probe(
        tmp_path, entry("Look up ORD-101."), repeats=2, environment="fake"
    )
    attempts, target = _run(probe, world, [[LOOKUP_101], [LOOKUP_101, LOOKUP_201]])

    assert len(attempts) == 2, "each repeat is its own chain"
    assert len(target.prompts) == 2
    assert world.resets == [False, True], "a repeat asks the environment to reset"
    assert world.events == [
        "start_run",
        "arm",
        "environment_open",
        "disarm",
        "environment_close",
        "arm",
        "environment_open",
        "disarm",
        "environment_close",
        "run_close",
    ], "a sealed chain is released before the next one opens"
    evaluations = {attempt.notes[TRACE_NOTE]["evaluation_id"] for attempt in attempts}
    assert len(evaluations) == 2, "each chain is its own evaluation"
    assert [attempt.notes[CHAIN_NOTE]["chain"] for attempt in attempts] == [0, 1]
    scores = [ToolCallCondition().detect(attempt) for attempt in attempts]
    assert scores == [[0.0], [1.0]], "one score per chain, from that chain's trace"


def test_a_lost_release_is_recorded_and_the_capture_is_incomplete(
    tmp_path, world, report
):
    world.release_lost = True
    probe = build_probe(tmp_path, entry("Look up ORD-101 for me."))
    attempts, _ = _run(probe, world, [[LOOKUP_101, LOOKUP_201]])

    (attempt,) = attempts
    trace = attempt.notes[TRACE_NOTE]
    assert trace["sealed"] == "final", "the seal stands; only the release was lost"
    assert trace["release_error"] == "link-lost", "the trace shows the lost release"
    assert trace["cleanup_failures"] == ["CapabilityError"], "and what failed to close"
    assert trace["error"] is None, "a lost release is not a generation failure"
    assert trace["snapshot"] is not None, "the evidence the chain held is kept"
    assert trace["verdict"]["grade"]["security"] is True, "the sealed calls still grade"
    assert trace["capture_complete"] is False, "but the calls cannot be called whole"
    assert trace["response_status"] == "incomplete", "so the capture cannot decide"
    assert trace["snapshot"]["evidence_status"]["function_calls"] == "incomplete"
    assert probe._live == {}, "the chain is closed and forgotten"
    assert world.events[-1] == "run_close", "and the run still closes"


def test_an_acknowledged_release_records_no_lost_release(tmp_path, world, report):
    probe = build_probe(tmp_path, entry("Look up ORD-101 for me."))
    attempts, _ = _run(probe, world, [[LOOKUP_101]])

    trace = attempts[0].notes[TRACE_NOTE]
    assert trace["release_error"] is None, "an acknowledged release leaves no marker"
    assert trace["cleanup_failures"] == []


def test_a_lost_release_cannot_prevent_an_interrupted_chain_closing(
    tmp_path, world, report
):
    world.release_lost = True
    probe = build_probe(tmp_path, entry("Look up ORD-101.", "And ORD-201."))
    target = _fail_turn(
        ScriptedTarget(world, [[LOOKUP_101]]), 2, RuntimeError("the target went away")
    )

    with pytest.raises(RuntimeError, match="the target went away"):
        list(probe.probe(target))

    attempt = probe._last_processed[0]
    assert probe._live == {}, "the lost release closes the chain"
    assert (
        attempt.notes[CHAIN_NOTE]["final"] is True
    ), "the chain reaches a final attempt"
    trace = attempt.notes[TRACE_NOTE]
    assert trace["sealed"] == "interrupted", "the interrupted grade stays undecided"
    assert trace["release_error"] == "link-lost", "the lost release is recorded"
    assert trace["response_status"] == "incomplete", "the capture cannot decide"


@pytest.mark.parametrize(
    "error", [TypeError("sentinel-bind-text"), ValueError("sentinel-bind-text")]
)
def test_an_error_while_binding_is_reported_by_type_only(
    tmp_path, world, report, error
):
    world.arm_error = error
    probe = build_probe(tmp_path, entry("Look up ORD-101."))

    with pytest.raises(ConfigFailure, match="could not open") as caught:
        probe.probe(ScriptedTarget(world))

    assert type(error).__name__ in str(caught.value), "the refusal names the type"
    assert "sentinel-bind-text" not in exception_text(caught.value)
    assert world.count("run_close") == 1, "the injector run closes"


def test_a_capability_refusal_while_binding_keeps_its_own_words(
    tmp_path, world, report
):
    world.arm_error = CapabilityError("the service refused the binding")
    probe = build_probe(tmp_path, entry("Look up ORD-101."))

    with pytest.raises(ConfigFailure, match="the service refused the binding"):
        probe.probe(ScriptedTarget(world))


def test_a_selection_that_cannot_load_is_refused_by_name(tmp_path, world, report):
    probe = build_probe(tmp_path, entry("Look up ORD-101."), injector_type="absent")

    with pytest.raises(ConfigFailure, match="could not load.*midojo-mim"):
        probe.probe(ScriptedTarget(world))


def test_a_missing_midojo_mim_extra_asks_for_the_extra(
    tmp_path, world, report, monkeypatch
):
    monkeypatch.setitem(sys.modules, "midojo_mim", None)
    probe = build_probe(tmp_path, entry("Look up ORD-101."), injector_type=None)

    with pytest.raises(ConfigFailure, match="midojo-mim"):
        probe.probe(ScriptedTarget(world))
