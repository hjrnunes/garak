# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Outputs without a score are counted as unknown or not applicable.

A detector leaves an output without a score (``None``) when it cannot decide.
Such outputs never enter the attack success denominator, so the evaluator
reports how many there are, how many do not apply, and why.
"""

import json
import uuid

import pytest

from garak import _config
from garak.attempt import Attempt, Message
from garak.detectors import always
from garak.evaluators.base import ThresholdEvaluator
from garak.resources import metrics

PROBE = "probe.ModuleA"
DETECTOR = "det.A"


@pytest.fixture()
def report(tmp_path):
    _config.load_base_config()
    _config.reporting.confidence_interval_method = "none"
    _config.transient.report_filename = str(tmp_path / "unscored.report.jsonl")
    _config.transient.reportfile = open(
        _config.transient.report_filename, "w", buffering=1, encoding="utf-8"
    )
    _config.transient.run_id = uuid.uuid4()
    _config.transient.hitlogfile = None
    _config.run.generations = 3
    _config.system.show_z = False
    _config.system.verbose = 0
    _config.system.narrow_output = False
    return _config.transient.report_filename


def read(path, entry_type):
    _config.transient.reportfile.flush()
    with open(path, encoding="utf-8") as lines:
        records = [json.loads(line) for line in lines]
    return [r for r in records if r["entry_type"] == entry_type]


def attempt(scores, mark=None, detector=DETECTOR, probe=PROBE):
    built = Attempt(prompt=Message(text="prompt"))
    built.probe_classname = probe
    built.outputs = [f"output {i}" for i in range(len(scores))]
    built.detector_results = {detector: list(scores)}
    if mark is not None:
        metrics.mark_unscored(built, detector, *mark)
    return built


def evaluate(path, attempts):
    summary = ThresholdEvaluator(0.5).evaluate(attempts)
    return read(path, "eval"), summary


def test_unknown_and_not_applicable_are_counted_apart(report):
    attempts = [
        attempt([0.0, 1.0]),
        attempt([None, None], mark=("intermediate_turn", True)),
        attempt([None], mark=("not_delivered", False)),
        attempt([None]),
    ]

    (record,), _ = evaluate(report, attempts)

    assert (record["passed"], record["fails"]) == (1, 1), "scored outputs are unchanged"
    assert record["total_evaluated"] == 2, "only scored outputs form the denominator"
    assert record["nones"] == 4, "every unscored output is still a none"
    assert record["not_applicable"] == 2, "not applicable is counted apart"
    assert (
        record["nones"] - record["not_applicable"] == 2
    ), "unknown is what remains, and includes the output with no reason"
    assert record["total_processed"] == 6, "nothing is dropped from the population"
    assert record["unscored_reasons"] == {
        "intermediate_turn": 2,
        "not_delivered": 1,
    }, "each exclusion from the denominator is named"


def test_unmarked_results_add_no_fields(report):
    (record,), summary = evaluate(report, [attempt([0.0, None, 1.0])])

    assert record["nones"] == 1, "an unmarked none is an unknown"
    assert "not_applicable" not in record, "no not-applicable output, no field"
    assert "unscored_reasons" not in record, "no reasons recorded, no field"
    assert "not_applicable" not in summary["detection_counts"]
    assert "unscored_reasons" not in summary["detection_counts"]


def test_a_mark_does_not_apply_to_a_scored_output(report):
    stale = attempt([0.0, 1.0], mark=("not_delivered", False))

    (record,), _ = evaluate(report, [stale])

    assert record["nones"] == 0, "a scored output is never unscored"
    assert "unscored_reasons" not in record, "a stale mark is not counted"


@pytest.mark.parametrize(
    "notes",
    [
        None,
        "text",
        {metrics.UNSCORED_NOTE: "text"},
        {metrics.UNSCORED_NOTE: {DETECTOR: "text"}},
        {metrics.UNSCORED_NOTE: {DETECTOR: {"reason": 7, "not_applicable": "yes"}}},
    ],
)
def test_malformed_marks_leave_an_unknown(report, notes):
    built = attempt([None])
    built.notes = notes

    (record,), _ = evaluate(report, [built])

    assert record["nones"] == 1, "an unreadable mark still leaves a counted none"
    assert "not_applicable" not in record, "only an explicit true marks not applicable"
    assert "unscored_reasons" not in record, "only a text reason is a reason"


def test_marks_belong_to_one_detector(report):
    built = attempt([None])
    built.detector_results["det.B"] = [None]
    metrics.mark_unscored(built, DETECTOR, "not_delivered", False)

    records, _ = evaluate(report, [built])

    by_detector = {r["detector"]: r for r in records}
    assert by_detector[DETECTOR]["unscored_reasons"] == {"not_delivered": 1}
    assert "unscored_reasons" not in by_detector["det.B"], "det.B recorded no reason"


def test_probe_summary_pools_reasons_across_detectors(report):
    first = attempt([None, None], mark=("not_delivered", False))
    first.detector_results["det.B"] = [None, 1.0]
    metrics.mark_unscored(first, "det.B", "intermediate_turn", True)

    _, summary = evaluate(report, [first])

    counts = summary["detection_counts"]
    assert counts["nones"] == 3, "none counts keep their meaning"
    assert counts["not_applicable"] == 1, "the summary counts not applicable apart"
    assert counts["unscored_reasons"] == {
        "intermediate_turn": 1,
        "not_delivered": 2,
    }, "the summary pools reasons over outputs and detectors"
    (logged,) = read(report, "probe_summary")
    for key in ("nones", "not_applicable", "unscored_reasons"):
        assert (
            logged["detection_counts"][key] == counts[key]
        ), f"the logged summary carries {key}"


def test_console_names_unscored_outputs(report, capsys):
    evaluate(
        report,
        [attempt([0.0, None, None], mark=("not_delivered", False)), attempt([None])],
    )

    out = capsys.readouterr().out
    assert "unscored" in out, "unscored outputs are visible on the console"
    assert "3 unknown" in out and "not_delivered: 2" in out


def test_console_stays_quiet_without_unscored_outputs(report, capsys):
    evaluate(report, [attempt([0.0, 1.0])])

    assert "unscored" not in capsys.readouterr().out


def test_detector_marks_under_the_name_the_harness_uses():
    built = Attempt(prompt=Message(text="prompt"))
    built.outputs = ["a"]

    always.Pass().mark_unscored(built, "no_evidence", not_applicable=True)

    assert built.notes[metrics.UNSCORED_NOTE] == {
        "always.Pass": {"reason": "no_evidence", "not_applicable": True}
    }, "the mark uses the key the harness gives detector_results"


@pytest.mark.parametrize("notes", [None, "text", {metrics.UNSCORED_NOTE: []}])
def test_marking_replaces_unreadable_notes(notes):
    built = Attempt(prompt=Message(text="prompt"))
    built.notes = notes

    always.Pass().mark_unscored(built, "no_evidence")

    assert built.notes[metrics.UNSCORED_NOTE]["always.Pass"] == {
        "reason": "no_evidence",
        "not_applicable": False,
    }, "a mark is recorded even when the notes were unreadable"
