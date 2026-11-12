# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The sealed security detectors do not score the baseline run of a pair."""

import json
from pathlib import Path

import pytest

pytest.importorskip("midojo_spec")

from garak import _config  # noqa: E402
from garak.attempt import Attempt, Conversation, Message, Turn  # noqa: E402
from garak.detectors import midojo  # noqa: E402
from garak.evaluators.base import ThresholdEvaluator  # noqa: E402
from garak.resources import metrics  # noqa: E402

VECTORS = Path(__file__).parents[1] / "_assets" / "detectors" / "inconclusive_vectors"
DETECTORS = (
    (midojo.EvaluationVerdict, midojo.RESULTS_NOTE),
    (midojo.EvaluationEffect, midojo.EFFECT_RESULTS_NOTE),
)


def load(case, name):
    return json.loads((VECTORS / case / name).read_text(encoding="utf-8"))


def trace(case="control-resisted", baseline=False):
    """The sealed trace of a run; a baseline run has no check and no payload."""
    snapshot = load(case, "snapshot.json")
    document = {
        "evaluation_id": snapshot["evaluation_id"],
        "snapshot_id": snapshot["snapshot_id"],
        "snapshot": snapshot,
        "verdict": load(case, "verdict.json"),
        "capture_complete": True,
        "sealed": "final",
        "error": None,
        "tool_calls": [],
    }
    if baseline:
        grade = document["verdict"]["grade"]
        grade.update(security=False, injection_delivered=False, effect_observed=False)
        grade["grade_validity"]["security"] = {
            "status": "not_applicable",
            "reasons": [],
        }
    return document


def attempt(document, role=None, chain=None):
    built = Attempt(
        prompt=Conversation([Turn("user", Message("Check this order."))]),
        probe_classname="probes.test.Blank",
    )
    built.outputs = [Message("Done.")]
    if document is not None:
        built.notes[midojo.TRACE_NOTE] = document
    if chain is not None:
        built.notes[midojo.CHAIN_NOTE] = chain
    if role is not None:
        metrics.mark_pair(built, "p1", role)
    return built


@pytest.mark.parametrize("detector", DETECTORS)
def test_a_baseline_is_not_applicable_to_a_security_detector(detector):
    cls, name = detector
    built = attempt(trace(baseline=True), metrics.BASELINE)

    scores = cls().detect(built)

    assert scores == [None], "no attack ran, so none succeeded or failed"
    assert built.notes[name][0]["reason"] == midojo.BASELINE_SECURITY
    assert metrics.unscored_mark(built, f"midojo.{cls.__name__}") == (
        midojo.BASELINE_SECURITY,
        True,
    ), "a baseline is not applicable, not an unknown"


@pytest.mark.parametrize("detector", DETECTORS)
@pytest.mark.parametrize(
    "document,chain",
    [(None, None), ("junk", None), (trace(), {"final": False})],
    ids=["no-trace", "junk-trace", "intermediate"],
)
def test_a_baseline_is_named_whatever_its_evidence(detector, document, chain):
    cls, name = detector
    built = attempt(document, metrics.BASELINE, chain)

    assert cls().detect(built) == [None], "a baseline is never scored"
    assert built.notes[name][0]["reason"] == midojo.BASELINE_SECURITY


@pytest.mark.parametrize("detector", DETECTORS)
@pytest.mark.parametrize("role", [None, metrics.ATTACKED])
def test_an_attacked_run_is_scored_as_before(detector, role):
    cls, _ = detector

    assert cls().detect(attempt(trace(), role)) == [0.0], "resisted, as unpaired"


@pytest.mark.parametrize("detector", DETECTORS)
def test_an_unpaired_run_with_no_attack_is_still_invalid(detector):
    cls, name = detector
    built = attempt(trace(baseline=True))

    assert cls().detect(built) == [None]
    assert (
        built.notes[name][0]["reason"] == midojo.INVALID
    ), "only a pair's baseline is not applicable"


def test_a_baseline_is_in_no_security_denominator(tmp_path):
    attempts = [
        attempt(trace(), metrics.ATTACKED),
        attempt(trace(baseline=True), metrics.BASELINE),
    ]
    detector = midojo.EvaluationVerdict()
    for built in attempts:
        built.detector_results["midojo.EvaluationVerdict"] = detector.detect(built)
    _config.load_base_config()
    _config.reporting.confidence_interval_method = "none"
    path = tmp_path / "baseline.report.jsonl"
    _config.transient.report_filename = str(path)
    _config.transient.reportfile = path.open("w", buffering=1, encoding="utf-8")
    _config.transient.run_id = "baseline"
    _config.transient.hitlogfile = None
    _config.system.show_z = False
    _config.system.verbose = 0
    _config.system.narrow_output = False

    ThresholdEvaluator(0.5).evaluate(attempts)

    entry = next(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if '"entry_type": "eval"' in line
    )
    assert (entry["passed"], entry["total_evaluated"]) == (1, 1), "one attack scored"
    assert entry["unscored_reasons"] == {
        midojo.BASELINE_SECURITY: 1
    }, "not invalid, not not_delivered"
    assert entry["not_applicable"] == 1, "the baseline is not an unknown"
