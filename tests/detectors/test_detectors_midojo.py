# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test both sealed-evidence detectors against the MiDojo-owned vectors.

The vectors also specify a tool-call condition score; that column is owned by
the condition detector and is not asserted here.
"""

import copy
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("midojo_spec")
from midojo_spec.snapshot import snapshot_digest, verified  # noqa: E402

from garak import _config  # noqa: E402
from garak.attempt import Attempt, Conversation, Message, Turn  # noqa: E402
from garak.detectors import midojo  # noqa: E402
from garak.evaluators.base import ThresholdEvaluator  # noqa: E402
from garak.resources.metrics import UNSCORED_NOTE  # noqa: E402

VECTORS = Path(__file__).parents[1] / "_assets" / "detectors" / "inconclusive_vectors"
CASES = json.loads((VECTORS / "MANIFEST.json").read_text())["cases"]
DETECTORS = (
    (midojo.EvaluationVerdict, midojo.RESULTS_NOTE),
    (midojo.EvaluationEffect, midojo.EFFECT_RESULTS_NOTE),
)


_ARMED = []


class _ExplodingMeta(type):
    """Raise on a type-name read, but only while a test has armed it.

    The test runner also reads type names to report a failure.
    """

    def __getattribute__(cls, key):
        if key == "__name__" and _ARMED:
            raise RuntimeError("name bomb")
        return super().__getattribute__(key)


class _Hostile(metaclass=_ExplodingMeta):
    pass


def load(case, name):
    return json.loads((VECTORS / case / name).read_text(encoding="utf-8"))


def trace(case="control-resisted"):
    snapshot = load(case, "snapshot.json")
    expected = load(case, "expected.json")
    return {
        "evaluation_id": snapshot["evaluation_id"],
        "snapshot_id": snapshot["snapshot_id"],
        "snapshot": snapshot,
        "verdict": load(case, "verdict.json"),
        "capture_complete": expected["midojo"]["capture_complete"],
        "sealed": expected["sealed"],
        "error": None,
        "tool_calls": [],
    }


def attempt(document=None, outputs=1, chain=None):
    built = Attempt(
        prompt=Conversation([Turn("user", Message("Check this order."))]),
        probe_classname="probes.test.Blank",
    )
    built.outputs = [Message("Done.") for _ in range(outputs)]
    if document is not None:
        built.notes[midojo.TRACE_NOTE] = document
    if chain is not None:
        built.notes[midojo.CHAIN_NOTE] = chain
    return built


def run(detector, document, *, outputs=1, chain=None):
    built = attempt(document, outputs, chain)
    cls, name = detector
    scores = cls().detect(built)
    return scores, built.notes[name]


def reseal(document):
    snapshot = document["snapshot"]
    snapshot["snapshot_id"] = snapshot_digest(snapshot)
    document["snapshot_id"] = snapshot["snapshot_id"]
    document["verdict"]["snapshot_id"] = snapshot["snapshot_id"]
    return document


@pytest.mark.parametrize("case", CASES)
def test_owned_vector_scores_and_facts(case):
    expected = load(case, "expected.json")["routes"]["garak_detectors"]
    for detector in DETECTORS:
        cls, _ = detector
        scores, records = run(detector, trace(case))
        want = expected[cls.__name__]
        assert scores == [want["score"]], f"{case}/{cls.__name__}: score"
        assert records[0]["outcome"] == want["outcome"], f"{case}: outcome"
        if scores == [None]:
            assert records[0]["reason"] and records[0]["detail"], f"{case}: why unknown"
        else:
            assert (records[0]["reason"], records[0]["detail"]) == (
                None,
                None,
            ), f"{case}: a decision carries no refusal"


@pytest.mark.parametrize("case", CASES)
def test_vector_snapshots_verify(case):
    assert verified(load(case, "snapshot.json")), f"{case}: sealed snapshot digest"


def test_blocked_call_keeps_validity_and_effect_separate():
    document = trace("blocked-call")
    verdict, vrecords = run(DETECTORS[0], document)
    effect, erecords = run(DETECTORS[1], document)
    assert (verdict, effect) == (
        [1.0],
        [0.0],
    ), "a blocked call complies but has no effect"
    assert vrecords[0]["security"] is True, "the security check held"
    assert erecords[0]["effect_observed"] is False, "the action did not run"


def test_observed_effect_does_not_change_a_false_security_grade():
    document = trace()
    document["verdict"]["grade"]["effect_observed"] = True
    verdict, _ = run(DETECTORS[0], document)
    effect, records = run(DETECTORS[1], document)
    assert (verdict, effect) == ([0.0], [1.0]), "each detector reads its own fact"
    assert records[0]["reason"] is None, "an observed effect is decided"


@pytest.mark.parametrize(
    "case,reason",
    [
        ("security-unknown", midojo.INVALID),
        ("interrupted-seal", midojo.INTERRUPTED),
        ("capture-incomplete", midojo.INCOMPLETE),
        ("not-delivered", midojo.NOT_DELIVERED),
        ("unknown-delivery", midojo.UNRECORDED_DELIVERY),
    ],
)
@pytest.mark.parametrize("detector", DETECTORS)
def test_inconclusive_vector_reason(detector, case, reason):
    scores, records = run(detector, trace(case))
    assert scores == [None], f"{case} cannot establish resistance"
    assert records[0]["reason"] == reason, f"{case} reports its missing evidence"


@pytest.mark.parametrize("detector", DETECTORS)
def test_every_output_gets_an_independent_record(detector):
    scores, records = run(detector, trace(), outputs=3)
    assert scores == [0.0] * 3, "each output scores the same sealed evaluation"
    assert len(records) == 3 and records[0] == records[1], "record shape is stable"
    assert records[0] is not records[1], "records are independent objects"


@pytest.mark.parametrize("detector", DETECTORS)
@pytest.mark.parametrize(
    "change,reason",
    [
        (lambda d: d.update(sealed="interrupted"), midojo.INTERRUPTED),
        (lambda d: d.update(error="EvaluationFailure"), midojo.ERRORED),
        (lambda d: d.update(verdict=None), midojo.MISSING),
        (lambda d: d.update(snapshot=None), midojo.DIGEST_MISMATCH),
        (lambda d: d.update(capture_complete=False), midojo.INCOMPLETE),
        (lambda d: d["verdict"].update(grade=None), midojo.INVALID),
        (
            lambda d: d["verdict"]["grade"].update(
                grade_validity={"security": {"status": "invalid", "reasons": []}}
            ),
            midojo.INVALID,
        ),
        (
            lambda d: d["verdict"]["grade"].update(security=None),
            midojo.INVALID,
        ),
        (
            lambda d: d["verdict"]["grade"].update(injection_delivered=False),
            midojo.NOT_DELIVERED,
        ),
        (
            lambda d: d["verdict"]["grade"].update(injection_delivered=None),
            midojo.UNRECORDED_DELIVERY,
        ),
    ],
)
def test_case_table_refusals(detector, change, reason):
    document = trace()
    change(document)
    scores, records = run(detector, document)
    assert scores == [None], "an undecidable trace never scores as resistance"
    assert records[0]["reason"] == reason, "the gate names its refusal"
    assert (
        isinstance(records[0]["detail"], str) and records[0]["detail"]
    ), "detail exists"


@pytest.mark.parametrize("detector", DETECTORS)
def test_not_graded_is_distinct_from_unknown(detector):
    assert (
        run(detector, None)[1][0]["reason"] == midojo.MISSING
    ), "no trace is not graded"
    scores, records = run(detector, trace(), chain={"final": False})
    assert scores == [None], "an intermediate turn has no final grade"
    assert records[0]["reason"] == midojo.INTERMEDIATE_TURN, "turn is not graded"
    assert run(detector, trace("security-unknown"))[1][0]["reason"] == (
        midojo.INVALID
    ), "an unknown grade differs from a missing grade"


@pytest.mark.parametrize("detector", DETECTORS)
@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d["snapshot"].update(agent_output="tampered"),
        lambda d: d.update(snapshot_id="sha256:other"),
        lambda d: d.update(evaluation_id="other"),
        lambda d: d["verdict"].update(snapshot_id="sha256:other"),
        lambda d: d["verdict"].update(evaluation_id="other"),
        lambda d: d["snapshot"].update(snapshot_id="sha256:other"),
        lambda d: d["snapshot"].update(evaluation_id="other"),
        lambda d: d["snapshot"].pop("agent_output"),
        lambda d: d["snapshot"].update(agent_output=object()),
    ],
)
def test_digest_and_identity_mismatches(detector, mutation):
    document = trace()
    mutation(document)
    scores, records = run(detector, document)
    assert scores == [None], "unverified content never scores"
    assert records[0]["reason"] == midojo.DIGEST_MISMATCH, "bad identity is explicit"


@pytest.mark.parametrize("detector", DETECTORS)
@pytest.mark.parametrize(
    "field,value,fragment",
    [
        ("evidence_status", {"calls": "failed"}, "calls"),
        ("evidence_status", {"calls": "incomplete"}, "calls"),
        ("evidence_status", {"calls": "unknown"}, "calls"),
        ("evidence_status", None, "evidence_status"),
        ("capture_failures", ["capture failed"], "capture_failures"),
        ("function_calls", [{"complete": False}], "function_calls"),
        ("function_calls", [None], "function_calls"),
        ("function_calls", {}, "function_calls"),
        ("function_calls", "none", "function_calls"),
    ],
)
def test_verified_capture_loss_overrides_trace_flag(detector, field, value, fragment):
    document = reseal(trace())
    document["snapshot"][field] = value
    reseal(document)
    assert document["capture_complete"] is True, "the trace flag remains true"
    scores, records = run(detector, document)
    assert scores == [None], "the verified snapshot exposes capture loss"
    assert records[0]["reason"] == midojo.INCOMPLETE, "loss is not resistance"
    assert fragment in records[0]["detail"], "detail names the source of loss"
    assert records[0]["capture_complete"] is False, "record uses derived completeness"


@pytest.mark.parametrize("detector", DETECTORS)
def test_not_collected_source_does_not_indicate_capture_loss(detector):
    document = trace()
    document["snapshot"]["evidence_status"] = {"unused": "not_collected"}
    reseal(document)
    assert run(detector, document)[0] == [0.0], "uncollected sources are not losses"


@pytest.mark.parametrize("detector", DETECTORS)
@pytest.mark.parametrize(
    "value",
    [True, False, "bad", 4, [], [1], None],
)
def test_hostile_traces_fail_closed(detector, value):
    document = trace()
    document["verdict"] = value
    scores, records = run(detector, document)
    assert scores == [None] and records[0]["reason"] == midojo.MISSING, "bad verdict"


@pytest.mark.parametrize("detector", DETECTORS)
@pytest.mark.parametrize("value", [True, False, "bad", 4, [], [1]])
def test_non_object_trace_fails_closed(detector, value):
    built = attempt()
    built.notes[midojo.TRACE_NOTE] = value
    cls, key = detector
    assert cls().detect(built) == [None], "bad trace is not a score"
    assert (
        built.notes[key][0]["reason"] == midojo.MISSING
    ), "trace is absent or malformed"


@pytest.mark.parametrize("detector", DETECTORS)
@pytest.mark.parametrize("value", [None, "bad", []])
def test_malformed_notes_still_receive_a_result(detector, value):
    built = attempt()
    built.notes = value
    cls, key = detector
    assert cls().detect(built) == [None], "unreadable notes cannot score"
    assert built.notes[key][0]["reason"] == midojo.MISSING, "the refusal is recorded"


@pytest.mark.parametrize("detector", DETECTORS)
@pytest.mark.parametrize("value", [True, "bad", [], None])
def test_non_object_grade_fails_closed(detector, value):
    document = trace()
    document["verdict"]["grade"] = value
    assert run(detector, document)[1][0]["reason"] == midojo.INVALID, "bad grade"


@pytest.mark.parametrize("detector", DETECTORS)
@pytest.mark.parametrize("value", [True, "bad", [], None])
def test_non_object_snapshot_fails_closed(detector, value):
    document = trace()
    document["snapshot"] = value
    scores, records = run(detector, document)
    assert scores == [None], "malformed snapshot has no score"
    assert records[0]["reason"] == midojo.DIGEST_MISMATCH, "snapshot cannot verify"


@pytest.mark.parametrize("detector", DETECTORS)
def test_recursive_snapshot_fails_closed(detector):
    document = trace()
    document["snapshot"]["agent_output"] = document["snapshot"]
    assert run(detector, document)[1][0]["reason"] == (
        midojo.DIGEST_MISMATCH
    ), "recursive digest input cannot escape detect"


@pytest.mark.parametrize("detector", DETECTORS)
@pytest.mark.parametrize(
    "reasons",
    [
        "bad",
        {},
        [None],
        [{"check": "c"}],
        [{"reason": "r"}],
        [{"check": 1, "reason": "r"}],
    ],
)
def test_malformed_validity_reasons_are_invalid(detector, reasons):
    document = trace()
    validity = document["verdict"]["grade"]["grade_validity"]["security"]
    validity["status"] = "valid"
    validity["reasons"] = reasons
    scores, records = run(detector, document)
    assert scores == [None], "a validity that cannot be read decides nothing"
    assert records[0]["reason"] == midojo.INVALID, "the grade is refused as invalid"
    assert records[0]["security_validity"] is None, "no validity is recorded"


@pytest.mark.parametrize("detector", DETECTORS)
@pytest.mark.parametrize(
    "path",
    [
        ("security",),
        ("injection_delivered",),
        ("effect_observed",),
        ("grade_validity", "security", "status"),
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        10**4500,
        float("nan"),
        "x" * 10_000,
        [1],
        {"a": 1},
        4,
        _Hostile(),
        type("X" * 10_000, (), {})(),
    ],
    ids=[
        "huge-int",
        "nan",
        "long-text",
        "list",
        "object",
        "number",
        "hostile-type",
        "long-type-name",
    ],
)
def test_hostile_grade_facts_fail_closed_and_stay_reportable(detector, path, value):
    document = trace()
    node = document["verdict"]["grade"]
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    _ARMED.append(True)
    try:
        scores, records = run(detector, document)
    finally:
        _ARMED.clear()
    reads_effect = detector[0] is midojo.EvaluationEffect
    if path == ("effect_observed",) and not reads_effect:
        assert scores == [0.0], "the verdict does not read the effect fact"
    else:
        assert scores == [None], "a malformed grade fact decides nothing"
    assert json.dumps(records), "the record serialises whatever the grade held"
    assert len(json.dumps(records)) < 2_000, "a hostile value is named, not copied"


@pytest.mark.parametrize("detector", DETECTORS)
@pytest.mark.parametrize(
    "statuses",
    [
        {"z" * 10_000: "failed"},
        {"calls": "z" * 10_000},
        {"calls": [1]},
    ],
    ids=["long-source", "long-status", "list-status"],
)
def test_evidence_status_is_named_not_copied(detector, statuses):
    document = trace()
    document["snapshot"]["evidence_status"] = statuses
    reseal(document)
    scores, records = run(detector, document)
    assert scores == [None], "an unreadable status is capture loss"
    assert records[0]["reason"] == midojo.INCOMPLETE, "loss is not resistance"
    assert len(json.dumps(records)) < 2_000, "sealed text does not swell the record"


@pytest.mark.parametrize("detector", DETECTORS)
def test_non_text_evaluation_id_is_not_an_identity(detector):
    document = trace()
    document["snapshot"]["evaluation_id"] = 4
    document["evaluation_id"] = 4
    document["verdict"]["evaluation_id"] = 4
    reseal(document)
    scores, records = run(detector, document)
    assert scores == [None], "an identity that is not text decides nothing"
    assert records[0]["reason"] == midojo.DIGEST_MISMATCH, "the identity is refused"


@pytest.mark.parametrize("detector", DETECTORS)
def test_final_seal_is_checked_before_the_error(detector):
    document = trace()
    document.update(sealed="interrupted", error="EvaluationFailure")
    scores, records = run(detector, document)
    assert scores == [None], "an interrupted evaluation decides nothing"
    assert records[0]["reason"] == midojo.INTERRUPTED, "the first refusal is reported"


@pytest.mark.parametrize("detector", DETECTORS)
def test_unrecorded_effect_and_validity_do_not_conflate(detector):
    document = trace()
    document["verdict"]["grade"]["effect_observed"] = None
    scores, records = run(detector, document)
    want = midojo.UNRECORDED_EFFECT if detector[0] is midojo.EvaluationEffect else None
    assert (scores, records[0]["reason"]) == (
        [None] if want else [0.0],
        want,
    ), "effect absence does not change the security verdict"


@pytest.mark.parametrize("detector", DETECTORS)
def test_missing_optional_dependency_raises_at_construction(detector, monkeypatch):
    monkeypatch.setitem(sys.modules, "midojo_spec", None)
    with pytest.raises(ModuleNotFoundError, match="midojo_spec"):
        detector[0]()


def _marks(detector, document, chain=None):
    cls, _ = detector
    built = attempt(document, chain=chain)
    scores = cls().detect(built)
    return scores, built.notes.get(UNSCORED_NOTE, {}).get(f"midojo.{cls.__name__}")


@pytest.mark.parametrize("detector", DETECTORS)
@pytest.mark.parametrize(
    "case,reason",
    [
        ("security-unknown", midojo.INVALID),
        ("interrupted-seal", midojo.INTERRUPTED),
        ("capture-incomplete", midojo.INCOMPLETE),
        ("not-delivered", midojo.NOT_DELIVERED),
        ("unknown-delivery", midojo.UNRECORDED_DELIVERY),
    ],
)
def test_refusals_are_marked_unknown(detector, case, reason):
    scores, mark = _marks(detector, trace(case))
    assert scores == [None], "the vector decides nothing"
    assert mark == {
        "reason": reason,
        "not_applicable": False,
    }, "an undecided evaluation is an unknown with its reason"


@pytest.mark.parametrize("detector", DETECTORS)
def test_intermediate_turn_is_marked_not_applicable(detector):
    scores, mark = _marks(detector, trace(), chain={"final": False})
    assert scores == [None], "an intermediate turn has no final grade"
    assert mark == {
        "reason": midojo.INTERMEDIATE_TURN,
        "not_applicable": True,
    }, "an intermediate turn is not an evaluation"


@pytest.mark.parametrize("detector", DETECTORS)
@pytest.mark.parametrize("case", ["control-resisted", "blocked-call"])
def test_decided_results_are_not_marked(detector, case):
    scores, mark = _marks(detector, trace(case))
    assert scores[0] is not None, "the vector is decided"
    assert mark is None, "a score needs no explanation"


def test_evaluator_declares_what_leaves_the_attack_success_rate(tmp_path):
    cases = [
        ("control-resisted", None),
        ("blocked-call", None),
        ("not-delivered", None),
        ("not-delivered", None),
        ("unknown-delivery", None),
        ("security-unknown", None),
        ("control-resisted", {"final": False}),
    ]
    detector = midojo.EvaluationVerdict()
    attempts = []
    for case, chain in cases:
        built = attempt(trace(case), chain=chain)
        built.probe_classname = "test.Blank"
        built.detector_results["midojo.EvaluationVerdict"] = detector.detect(built)
        attempts.append(built)

    _config.load_base_config()
    _config.reporting.confidence_interval_method = "none"
    path = tmp_path / "declared.report.jsonl"
    _config.transient.report_filename = str(path)
    _config.transient.reportfile = path.open("w", buffering=1, encoding="utf-8")
    _config.transient.run_id = "declared"
    _config.transient.hitlogfile = None
    _config.system.show_z = False
    _config.system.verbose = 0
    _config.system.narrow_output = False

    summary = ThresholdEvaluator(0.5).evaluate(attempts)

    record = json.loads(
        next(
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if '"entry_type": "eval"' in line
        )
    )
    assert (record["passed"], record["fails"]) == (1, 1), "decided rows are scored"
    assert record["total_evaluated"] == 2, "undelivered rows stay out of the rate"
    assert record["total_processed"] == 7, "every evaluation is in the population"
    assert record["unscored_reasons"] == {
        midojo.INTERMEDIATE_TURN: 1,
        midojo.INVALID: 1,
        midojo.NOT_DELIVERED: 2,
        midojo.UNRECORDED_DELIVERY: 1,
    }, "each excluded evaluation is declared with its reason"
    assert record["not_applicable"] == 1, "an intermediate turn is not an unknown"
    assert (
        summary["detection_counts"]["unscored_reasons"] == record["unscored_reasons"]
    ), "the probe summary declares the same exclusions"
