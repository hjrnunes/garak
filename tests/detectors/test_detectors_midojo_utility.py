# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test the utility detector against the sealed MiDojo-owned vectors.

The vectors have no utility check, so each test sets the utility grade that it
needs on the unsealed verdict; the verified snapshot is not touched.
"""

import copy
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("midojo_spec")

from garak.attempt import Attempt, Conversation, Message, Turn  # noqa: E402
from garak.detectors import midojo  # noqa: E402
from garak.resources import metrics  # noqa: E402

VECTORS = Path(__file__).parents[1] / "_assets" / "detectors" / "inconclusive_vectors"
NAME = "midojo.EvaluationUtility"


def load(case, name):
    return json.loads((VECTORS / case / name).read_text(encoding="utf-8"))


def trace(case="control-resisted", utility=True, status="valid"):
    snapshot = load(case, "snapshot.json")
    expected = load(case, "expected.json")
    document = {
        "evaluation_id": snapshot["evaluation_id"],
        "snapshot_id": snapshot["snapshot_id"],
        "snapshot": snapshot,
        "verdict": load(case, "verdict.json"),
        "capture_complete": expected["midojo"]["capture_complete"],
        "sealed": expected["sealed"],
        "error": None,
        "tool_calls": [],
    }
    grade = document["verdict"]["grade"]
    grade["utility"] = utility
    grade["grade_validity"]["utility"] = {"status": status, "reasons": []}
    return document


def attempt(document, outputs=1, chain=None, pair=None):
    built = Attempt(
        prompt=Conversation([Turn("user", Message("Check this order."))]),
        probe_classname="probes.test.Blank",
    )
    built.outputs = [Message("Done.") for _ in range(outputs)]
    if document is not None:
        built.notes[midojo.TRACE_NOTE] = document
    if chain is not None:
        built.notes[midojo.CHAIN_NOTE] = chain
    if pair is not None:
        metrics.mark_pair(built, *pair)
    return built


def run(document, **kwargs):
    built = attempt(document, **kwargs)
    scores = midojo.EvaluationUtility().detect(built)
    return scores, built.notes[midojo.UTILITY_RESULTS_NOTE], built


def test_the_detector_is_a_utility_detector():
    assert (
        midojo.EvaluationUtility.metric_kind == metrics.UTILITY
    ), "the evaluator reports its results apart from attack success"


@pytest.mark.parametrize("utility,score", [(True, 1.0), (False, 0.0)])
def test_a_valid_utility_grade_is_scored(utility, score):
    scores, records, built = run(trace(utility=utility))

    assert scores == [score], "completion scores 1.0, failure 0.0"
    assert (records[0]["outcome"], records[0]["reason"], records[0]["detail"]) == (
        midojo.DETECTED if utility else midojo.NOT_DETECTED,
        None,
        None,
    ), "a decision carries no refusal"
    assert (
        records[0]["utility"] is utility and records[0]["utility_validity"] == "valid"
    )
    assert (
        records[0]["snapshot_id"]
        == load("control-resisted", "snapshot.json")["snapshot_id"]
    ), "the record names the snapshot that was graded"
    assert metrics.UNSCORED_NOTE not in built.notes, "a score needs no explanation"


def test_no_utility_check_is_not_applicable_and_not_a_failure():
    scores, records, built = run(trace(utility=False, status="not_applicable"))

    assert scores == [None], "a spec without a check cannot fail it"
    assert records[0]["reason"] == midojo.NO_UTILITY_CHECK
    assert metrics.unscored_mark(built, NAME) == (
        midojo.NO_UTILITY_CHECK,
        True,
    ), "the evaluator counts it as not applicable"


@pytest.mark.parametrize("utility", [None, "yes", 1])
def test_an_unknown_utility_grade_is_invalid_not_failed(utility):
    scores, records, built = run(trace(utility=utility, status="invalid"))

    assert scores == [None], "an unknown grade is not a failed task"
    assert records[0]["reason"] == midojo.INVALID
    assert metrics.unscored_mark(built, NAME) == (
        midojo.INVALID,
        False,
    ), "an invalid grade is an unknown"


@pytest.mark.parametrize("status", ["valid", "other", None, 4])
def test_a_utility_value_that_is_not_a_boolean_decides_nothing(status):
    scores, records, _ = run(trace(utility="yes", status=status))

    assert scores == [None], "only a boolean is a decided grade"
    assert records[0]["reason"] == midojo.INVALID


@pytest.mark.parametrize(
    "reasons", ["bad", {}, [None], [{"check": "c"}], [{"check": 1, "reason": "r"}]]
)
def test_malformed_validity_reasons_are_invalid(reasons):
    document = trace()
    document["verdict"]["grade"]["grade_validity"]["utility"]["reasons"] = reasons

    scores, records, _ = run(document)

    assert scores == [None], "a validity that cannot be read decides nothing"
    assert (records[0]["reason"], records[0]["utility_validity"]) == (
        midojo.INVALID,
        None,
    ), "no validity is recorded"


@pytest.mark.parametrize(
    "case", ["security-unknown", "blocked-call", "control-resisted"]
)
def test_the_security_grade_does_not_decide_utility(case):
    scores, _, _ = run(trace(case))

    assert scores == [1.0], "utility reads its own grade, whatever the security grade"


@pytest.mark.parametrize(
    "case,reason",
    [
        ("interrupted-seal", midojo.INTERRUPTED),
        ("capture-incomplete", midojo.INCOMPLETE),
    ],
)
def test_an_unsound_seal_decides_nothing(case, reason):
    scores, records, _ = run(trace(case))

    assert scores == [None], "a utility from an unsound seal is not a result"
    assert records[0]["reason"] == reason, "the gate shared with security names it"
    assert records[0]["detail"], "a refusal says why"


@pytest.mark.parametrize(
    "change,reason",
    [
        (lambda d: d.update(error="EvaluationFailure"), midojo.ERRORED),
        (lambda d: d.update(verdict=None), midojo.MISSING),
        (lambda d: d.update(snapshot=None), midojo.DIGEST_MISMATCH),
        (
            lambda d: d["snapshot"].update(agent_output="tampered"),
            midojo.DIGEST_MISMATCH,
        ),
        (lambda d: d["verdict"].update(grade=None), midojo.INVALID),
    ],
)
def test_the_trace_gates_apply_to_utility(change, reason):
    document = trace()
    change(document)

    scores, records, _ = run(document)

    assert scores == [None] and records[0]["reason"] == reason, "same gates as security"


def test_no_trace_is_not_graded():
    scores, records, _ = run(None)

    assert (scores, records[0]["reason"]) == ([None], midojo.MISSING)


def test_an_intermediate_turn_is_not_applicable():
    scores, records, built = run(trace(), chain={"final": False})

    assert scores == [None], "only the final turn carries a grade"
    assert metrics.unscored_mark(built, NAME) == (
        midojo.INTERMEDIATE_TURN,
        True,
    ), "an intermediate turn is not an evaluation, so is not graded twice"


@pytest.mark.parametrize("delivered", [True, False, None])
@pytest.mark.parametrize("role", [None, metrics.BASELINE])
def test_utility_without_an_attacked_role_does_not_ask_for_delivery(role, delivered):
    document = trace()
    document["verdict"]["grade"]["injection_delivered"] = delivered

    scores, _, _ = run(document, pair=("p1", role) if role else None)

    assert scores == [1.0], "a baseline and an unpaired run owe no delivery"


@pytest.mark.parametrize(
    "delivered,reason",
    [(False, midojo.NOT_DELIVERED), (None, midojo.UNRECORDED_DELIVERY)],
)
def test_the_attacked_side_of_a_pair_needs_a_delivered_payload(delivered, reason):
    document = trace()
    document["verdict"]["grade"]["injection_delivered"] = delivered

    scores, records, built = run(document, pair=("p1", metrics.ATTACKED))

    assert scores == [None], "a run that never met the payload is not under attack"
    assert records[0]["reason"] == reason
    assert metrics.unscored_mark(built, NAME) == (reason, False)
    assert records[0]["utility"] is True, "the refusal keeps the grade it withheld"


def test_the_attacked_side_with_a_delivered_payload_is_scored():
    scores, records, _ = run(trace(utility=False), pair=("p1", metrics.ATTACKED))

    assert scores == [0.0], "an attack that costs the task is a failed task"
    assert records[0]["pair"] == {"id": "p1", "role": metrics.ATTACKED}


def test_an_unpaired_record_names_no_pair():
    _, records, _ = run(trace())

    assert records[0]["pair"] is None, "the pair is recorded only when there is one"


def test_every_output_gets_an_independent_record():
    scores, records, _ = run(trace(utility=False), outputs=3)

    assert scores == [0.0] * 3, "each output scores the same sealed evaluation"
    assert records[0] == records[1] and records[0] is not records[1]


def test_scoring_leaves_the_sealed_evidence_unchanged():
    document = trace(utility=False)
    before = copy.deepcopy(document)
    built = attempt(document, pair=("p1", metrics.BASELINE))

    midojo.EvaluationUtility().detect(built)

    assert built.notes[midojo.TRACE_NOTE] == before, "trace, snapshot and verdict stay"


@pytest.mark.parametrize(
    "path", [("utility",), ("grade_validity", "utility", "status")]
)
@pytest.mark.parametrize(
    "value",
    [10**4500, float("nan"), "x" * 10_000, [1], {"a": 1}, 4],
    ids=["huge-int", "nan", "long-text", "list", "object", "number"],
)
def test_hostile_grade_facts_fail_closed_and_stay_reportable(path, value):
    document = trace()
    node = document["verdict"]["grade"]
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value

    scores, records, _ = run(document)

    assert scores == [None], "a malformed grade fact decides nothing"
    assert json.dumps(records), "the record serialises whatever the grade held"
    assert len(json.dumps(records)) < 2_000, "a hostile value is named, not copied"


@pytest.mark.parametrize("value", [None, "bad", []])
def test_malformed_notes_still_receive_a_result(value):
    built = attempt(None)
    built.notes = value

    assert midojo.EvaluationUtility().detect(built) == [None], "unreadable notes"
    assert built.notes[midojo.UTILITY_RESULTS_NOTE][0]["reason"] == midojo.MISSING


def test_missing_optional_dependency_raises_at_construction(monkeypatch):
    monkeypatch.setitem(sys.modules, "midojo_spec", None)

    with pytest.raises(ModuleNotFoundError, match="midojo_spec"):
        midojo.EvaluationUtility()
