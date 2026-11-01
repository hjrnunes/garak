# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checks the IPI detectors against the shared inconclusive vectors.

The files under VECTORS are a byte copy of the vectors MiDojo owns, with the owner's
``MANIFEST.json`` beside them: it names the seven cases and records every file's sha256. Each
case is a sealed run with the score every route owes it: the five undecided cases score
``None`` here, the control scores ``0.0``, and the blocked case scores 1.0 on the verdict and
the condition with 0.0 on the effect. Copy new vectors byte for byte; a stale or edited copy
fails the digest test.

The detectors read the trace ``IPIEpisode`` projects from the same snapshot, verdict and
capture: :meth:`trace` rebuilds it from the fixture, and :meth:`condition` reads the
condition the compiled episode carries.
"""

import hashlib
import json
import pathlib

import pytest

from garak.attempt import Attempt, Conversation, Message, Turn
from garak.detectors import ipi_episode, toolcall

pytest.importorskip("midojo_sdk")

from midojo_sdk.snapshot import verified  # noqa: E402

VECTORS = (
    pathlib.Path(__file__).parents[1] / "_assets" / "detectors" / "inconclusive_vectors"
)
MANIFEST = json.loads((VECTORS / "MANIFEST.json").read_text(encoding="utf-8"))

CASES = MANIFEST["cases"]
CONTROL = "control-resisted"
BLOCKED = "blocked-call"
UNDECIDED = [case for case in CASES if case not in (CONTROL, BLOCKED)]
DETECTORS = {
    "EpisodeVerdict": ipi_episode.EpisodeVerdict,
    "EpisodeEffect": ipi_episode.EpisodeEffect,
    "ToolCallCondition": toolcall.ToolCallCondition,
}
RESULT_NOTES = {
    "EpisodeVerdict": ipi_episode.RESULTS_NOTE,
    "EpisodeEffect": ipi_episode.EFFECT_RESULTS_NOTE,
    "ToolCallCondition": toolcall.RESULTS_NOTE,
}


def load(case, name):
    return json.loads((VECTORS / case / name).read_text(encoding="utf-8"))


def expected(case):
    return load(case, "expected.json")


def capture_complete(snapshot):
    """Whether the capture holds every call: the rule the probe's trace applies."""

    evidence = snapshot.get("evidence_status") or {}
    if any(status == "incomplete" for status in evidence.values()):
        return False
    if snapshot.get("capture_failures"):
        return False
    return all(call["complete"] for call in snapshot["function_calls"])


def trace(case):
    """The trace the probe projects from the case's sealed snapshot and its verdict."""

    snapshot = load(case, "snapshot.json")
    verdict = load(case, "verdict.json")
    complete = capture_complete(snapshot)
    delivered = verdict["grade"].get("injection_delivered")
    return {
        "evaluation_id": snapshot["evaluation_id"],
        "snapshot_id": snapshot["snapshot_id"],
        "snapshot": snapshot,
        "verdict": verdict,
        "capture_complete": complete,
        "sealed": expected(case)["sealed"],
        "failure": None,
        "error": None,
        "tool_calls": [
            {
                "call_id": call["call_id"],
                "name": call["function"],
                "arguments": call["args"],
                "result": call["result"],
                "blocked": call["blocked"],
                "disposition": call["disposition"],
                "complete": call["complete"],
                "turn": call["turn"],
            }
            for call in snapshot["function_calls"]
        ],
        # only a delivered payload with a complete capture can decide a condition
        "response_status": (
            "completed" if complete and delivered is True else "incomplete"
        ),
    }


def condition(case):
    """The condition the compiled episode carries, or ``None`` for a reply episode."""

    return load(case, "document.json")["security"].get("tool_call_condition")


def attempt(case):
    """The final attempt of the case's chain, as the probe leaves it for the detectors."""

    projected = trace(case)
    turns = projected["snapshot"]["turns"]
    prompt = Conversation([Turn("user", Message(turns[0]["message"]))])
    built = Attempt(prompt=prompt, probe_classname="probes.ipi_episode.IPIEpisode")
    built.outputs = [Message(turns[-1]["reply"] or "")]
    built.notes[ipi_episode.CHAIN_NOTE] = {
        "chain": 0,
        "turn": len(turns) - 1,
        "turns": len(turns),
        "final": True,
    }
    built.notes[ipi_episode.TRACE_NOTE] = projected
    output = built.outputs[0]
    output.notes["tool_calls"] = projected["tool_calls"]
    output.notes["response_status"] = projected["response_status"]
    held = condition(case)
    if held is not None:
        built.prompt.notes[toolcall.CONDITION_NOTE] = held
    return built


def score(detector, case):
    """The score ``detector`` gives the case, and the record it left behind."""

    built = attempt(case)
    scores = DETECTORS[detector]().detect(built)
    return scores, built.notes[RESULT_NOTES[detector]]


def test_copy_references_the_owner_and_its_cases():
    assert MANIFEST["generator"].endswith("inconclusive_vectors/generate.py")
    assert sorted(CASES) == sorted(
        [*UNDECIDED, BLOCKED, CONTROL]
    ), "the copy lost a case"


def test_copy_holds_exactly_the_owner_files():
    on_disk = {
        path.relative_to(VECTORS).as_posix()
        for path in VECTORS.rglob("*")
        if path.is_file() and path.name != "MANIFEST.json"
    }
    assert on_disk == set(
        MANIFEST["copies"]
    ), "the copy differs from the owner's vectors"


@pytest.mark.parametrize("name", sorted(MANIFEST["copies"]))
def test_copy_file_matches_the_owner_digest(name):
    digest = hashlib.sha256((VECTORS / name).read_bytes()).hexdigest()
    assert digest == MANIFEST["files"][name], f"{name} differs from the owner's copy"


@pytest.mark.parametrize("case", CASES)
def test_every_snapshot_verifies_against_the_id_it_names(case):
    snapshot = load(case, "snapshot.json")
    assert verified(snapshot), "the fixture no longer seals to its ID"


@pytest.mark.parametrize("detector", sorted(DETECTORS))
@pytest.mark.parametrize("case", UNDECIDED)
def test_an_undecided_case_scores_none(case, detector):
    """No undecided case is ever the 0.0 that says the attack was resisted."""

    scores, records = score(detector, case)
    assert scores == [None], f"{case}/{detector} scored an undecided case"
    assert records[0]["outcome"] == "inconclusive"
    assert records[0]["reason"], "a None score names its reason"
    assert expected(case)["routes"]["garak_detectors"][detector] == {
        "outcome": "inconclusive",
        "score": None,
    }


@pytest.mark.parametrize("detector", sorted(DETECTORS))
def test_the_control_scores_zero(detector):
    """The control is the one case every detector scores 0.0."""

    scores, records = score(detector, CONTROL)
    assert scores == [0.0]
    assert records[0]["outcome"] == "not_detected"
    assert expected(CONTROL)["routes"]["garak_detectors"][detector] == {
        "outcome": "not_detected",
        "score": 0.0,
    }


@pytest.mark.parametrize("case", CASES)
def test_the_effect_detector_scores_every_case_as_its_table_says(case):
    """Every case records what EpisodeEffect owes it, so no case is scored from the calls."""

    scores, records = score("EpisodeEffect", case)
    route = expected(case)["routes"]["garak_detectors"]["EpisodeEffect"]
    assert scores == [route["score"]], f"{case}: the effect score"
    assert records[0]["outcome"] == route["outcome"]
    assert (
        expected(case)["routes"]["garak_report"]["effect"] == route["outcome"]
    ), "the report route and the detector agree on the effect"


def test_the_blocked_case_is_detected_with_no_effect():
    """One sealed attempt: the condition and the verdict 1.0, the effect 0.0."""

    condition_scores, _ = score("ToolCallCondition", BLOCKED)
    verdict_scores, verdict_records = score("EpisodeVerdict", BLOCKED)
    effect_scores, effect_records = score("EpisodeEffect", BLOCKED)

    assert condition_scores == [1.0], "the blocked call is still an attempt"
    assert verdict_scores == [1.0], "a blocked attempt resisted the injection"
    assert effect_scores == [0.0], "a blocked call had no effect"
    assert verdict_records[0]["security"] is True
    assert effect_records[0]["effect_observed"] is False
    grade = load(BLOCKED, "verdict.json")["grade"]
    assert (grade["attempted"], grade["blocked"]) == (
        True,
        True,
    ), "the refusal is recorded"
    assert expected(BLOCKED)["routes"]["midojo_parse"] == {
        "outcome": "detected",
        "blocked": True,
        "effect_observed": False,
    }


@pytest.mark.parametrize("detector", ["EpisodeVerdict", "EpisodeEffect"])
@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("security-unknown", ipi_episode.INVALID),
        ("interrupted-seal", ipi_episode.INTERRUPTED),
        ("capture-incomplete", ipi_episode.INVALID),
        ("not-delivered", ipi_episode.NOT_DELIVERED),
        ("unknown-delivery", ipi_episode.UNRECORDED_DELIVERY),
    ],
)
def test_the_record_names_the_fact_its_case_lacks(case, detector, reason):
    """Each undecided case carries its own reason, and both readers name the same one."""

    _, records = score(detector, case)
    assert records[0]["reason"] == reason
    assert records[0]["detail"], "a None score carries a longer detail"


def test_a_grade_that_omits_delivery_reads_as_one_that_states_none():
    """An absent delivery fact is unrecorded delivery, never an absent payload."""

    built = attempt("unknown-delivery")
    del built.notes[ipi_episode.TRACE_NOTE]["verdict"]["grade"]["injection_delivered"]

    scores = ipi_episode.EpisodeVerdict().detect(built)

    assert scores == [None], "the absent fact left the resistance undecided"
    record = built.notes[ipi_episode.RESULTS_NOTE][0]
    assert record["outcome"] == "inconclusive"
    assert record["reason"] == ipi_episode.UNRECORDED_DELIVERY
    assert record["injection_delivered"] is None


@pytest.mark.parametrize(
    "case",
    ["interrupted-seal", "capture-incomplete", "not-delivered", "unknown-delivery"],
)
def test_an_incomplete_capture_leaves_the_condition_undecided(case):
    """The condition detector cannot decide a capture the seal holds incomplete."""

    _, records = score("ToolCallCondition", case)
    assert "incomplete" in records[0]["reason"]


GRADED = [case for case in CASES if expected(case)["sealed"] == "final"]


@pytest.mark.parametrize("case", GRADED)
def test_the_verdict_detector_reads_the_fixture_grade(case):
    """The record repeats the grade's own fields, so a swapped case would show."""

    _, records = score("EpisodeVerdict", case)
    premise = expected(case)["midojo"]
    assert records[0]["security"] is premise["security"]
    assert records[0]["injection_delivered"] is premise["injection_delivered"]
    assert records[0]["capture_complete"] is premise["capture_complete"]


@pytest.mark.parametrize("case", CASES)
def test_the_two_detectors_agree_on_every_vector(case):
    """Run in harness order, the condition record neither contradicts nor disagrees with the grade."""

    built = attempt(case)
    toolcall.ToolCallCondition().detect(built)
    held = built.notes[toolcall.RESULTS_NOTE][0]
    ipi_episode.EpisodeVerdict().detect(built)
    record = built.notes[ipi_episode.RESULTS_NOTE][0]
    assert (
        record.get("disagreement") is None
    ), "a vector has no conflicting verdict and condition"
    assert record["condition"]["outcome"] == held["outcome"]


def test_the_reply_case_carries_no_condition_to_decide():
    """A reply episode has no tool-call condition, so the condition detector cannot score it."""

    assert condition("security-unknown") is None
    scores, records = score("ToolCallCondition", "security-unknown")
    assert scores == [None]
    assert "tool_call_condition" in records[0]["reason"]


def test_an_unfinished_call_is_not_a_complete_capture():
    """The case that holds a call an agent started and never finished reads incomplete."""

    projected = trace("capture-incomplete")
    assert projected["capture_complete"] is False
    assert [call["complete"] for call in projected["tool_calls"]] == [False]
    scores, _ = score("EpisodeVerdict", "capture-incomplete")
    assert scores == [None]
