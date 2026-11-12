# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utility results are reported apart from attack success.

A detector declares a metric kind. A security detector scores an attack that
succeeded; a utility detector scores a legitimate task that was completed.
Utility results must never change an attack success statistic.
"""

import json
import uuid

import pytest

import garak.harnesses.base
from garak import _config
from garak.attempt import Attempt, Message
from garak.detectors import always
from garak.detectors.base import Detector
from garak.evaluators.base import ThresholdEvaluator
from garak.resources import metrics

PROBE = "probe.ModuleA"
SECURITY_DETECTOR = "det.Security"
UTILITY_DETECTOR = "det.Utility"


@pytest.fixture()
def report(tmp_path):
    _config.load_base_config()
    _config.reporting.confidence_interval_method = "none"
    _config.reporting.hit_at_k = [1]
    _config.transient.report_filename = str(tmp_path / "utility.report.jsonl")
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


def attempt(results, kinds=None, unscored=None):
    """An attempt scored by several detectors: ``results`` maps name to scores"""
    built = Attempt(prompt=Message(text="prompt"))
    built.probe_classname = PROBE
    built.outputs = [f"output {i}" for i in range(len(next(iter(results.values()))))]
    built.detector_results = {name: list(scores) for name, scores in results.items()}
    for name, kind in (kinds or {}).items():
        metrics.mark_kind(built, name, kind)
    for name, mark in (unscored or {}).items():
        metrics.mark_unscored(built, name, *mark)
    return built


def utility(scores, mark=None):
    return attempt(
        {UTILITY_DETECTOR: scores},
        kinds={UTILITY_DETECTOR: metrics.UTILITY},
        unscored={UTILITY_DETECTOR: mark} if mark else None,
    )


def evaluate(attempts):
    return ThresholdEvaluator(0.5).evaluate(attempts)


class _UtilityDetector(always.Pass):
    """A detector that scores task completion."""

    metric_kind = metrics.UTILITY


def test_detectors_are_security_unless_declared():
    assert Detector.metric_kind == metrics.SECURITY, "security is the default"
    assert (
        _UtilityDetector.metric_kind == metrics.UTILITY
    ), "a detector declares its kind with a class attribute"


@pytest.mark.parametrize(
    "detector_class,kind", [(always.Pass, None), (_UtilityDetector, "utility")]
)
def test_harness_records_only_a_declared_kind(detector_class, kind):
    built = Attempt(prompt=Message(text="prompt"))
    built.outputs = ["a"]

    garak.harnesses.base.Harness()._run_detector([built], detector_class())

    (name,) = built.detector_results
    assert (
        built.notes.get(metrics.KIND_NOTE, {}).get(name) == kind
    ), "the harness carries a utility kind to the evaluator; security needs no mark"


def test_detector_with_unknown_kind_is_refused_at_construction():
    class Misspelt(always.Pass):
        """A detector whose kind is a typo."""

        metric_kind = "utilty"

    with pytest.raises(ValueError, match="metric_kind"):
        Misspelt()


def test_unknown_kind_is_refused():
    built = attempt({SECURITY_DETECTOR: [0.0]})
    with pytest.raises(ValueError, match="metric kind"):
        metrics.mark_kind(built, SECURITY_DETECTOR, "utilty")


@pytest.mark.parametrize(
    "notes",
    [
        None,
        "text",
        {metrics.KIND_NOTE: []},
        {metrics.KIND_NOTE: {SECURITY_DETECTOR: 1}},
    ],
)
def test_unreadable_kind_reads_as_security(notes):
    built = attempt({SECURITY_DETECTOR: [0.0]})
    built.notes = notes
    assert (
        metrics.metric_kind(built, SECURITY_DETECTOR) == metrics.SECURITY
    ), "only an explicit utility mark moves a detector out of the security report"


def test_utility_never_changes_attack_success_statistics(report):
    security = [[0.0, 1.0, 1.0], [None, 0.0, 0.0]]
    alone = [attempt({SECURITY_DETECTOR: scores}) for scores in security]
    with_utility = [
        attempt(
            {SECURITY_DETECTOR: scores, UTILITY_DETECTOR: [1.0, 1.0, 0.0]},
            kinds={UTILITY_DETECTOR: metrics.UTILITY},
        )
        for scores in security
    ]

    alone_summary = evaluate(alone)
    alone_eval = read(report, "eval")
    with_summary = evaluate(with_utility)
    all_eval = read(report, "eval")

    assert (
        all_eval[len(alone_eval) :] == alone_eval
    ), "the eval record is the same with a utility detector beside it"
    assert (
        with_summary["detection_counts"] == alone_summary["detection_counts"]
    ), "the detection counts are the same with a utility detector beside it"
    assert all_eval[-1]["detector"] == SECURITY_DETECTOR, "only security is an eval"
    assert all_eval[-1]["total_evaluated"] == 5, "the denominator is security outputs"


def test_utility_is_reported_in_its_own_entry(report):
    attempts = [
        utility([1.0, 1.0, 0.0]),
        utility([0.0, None, 1.0], mark=("no_utility_check", True)),
    ]

    summary = evaluate(attempts)

    assert read(report, "eval") == [], "utility does not make an eval record"
    (record,) = read(report, "utility_eval")
    assert record["detector"] == UTILITY_DETECTOR
    assert (record["completed"], record["not_completed"]) == (
        3,
        2,
    ), "a score of 1.0 is a completed task"
    assert record["total_evaluated"] == 5, "the rate is over scored outputs"
    assert record["nones"] == 1, "an unscored output is counted apart"
    assert record["not_applicable"] == 1, "not applicable is counted apart"
    assert record["unscored_reasons"] == {"no_utility_check": 1}
    assert record["total_processed"] == 6
    for absent in ("passed", "fails", "hit_at_k", "confidence", "intents"):
        assert absent not in record, f"utility has no {absent}: it is not an attack"
    assert summary["utility_counts"] == {
        "detectors": [UTILITY_DETECTOR],
        "completed": 3,
        "not_completed": 2,
        "nones": 1,
        "not_applicable": 1,
        "unscored_reasons": {"no_utility_check": 1},
    }, "the probe summary carries utility beside, not inside, detection_counts"
    assert summary["detection_counts"] == {
        "detectors": [],
        "passed": 0,
        "fails": 0,
        "nones": 0,
    }, "no attack success count includes utility"
    (logged,) = read(report, "probe_summary")
    assert logged["utility_counts"] == summary["utility_counts"]


def test_utility_leaves_no_hit_log_and_no_interval(report):
    _config.reporting.confidence_interval_method = "bootstrap"
    _config.reporting.bootstrap_min_sample_size = 1
    _config.reporting.bootstrap_confidence_level = 0.95
    evaluate([utility([1.0, 0.0, 1.0] * 40)])

    assert _config.transient.hitlogfile is None, "a completed task is not a hit"
    (record,) = read(report, "utility_eval")
    assert not any(
        key.startswith("confidence") for key in record
    ), "a utility rate has no attack success interval"


def test_utility_follows_the_evaluator_threshold(report):
    evaluate([utility([0.49, 0.5, 0.51, 1.0])])
    (record,) = read(report, "utility_eval")
    assert (record["completed"], record["not_completed"]) == (
        3,
        1,
    ), "completion is a score at or above the evaluator threshold"


def test_a_probe_with_only_utility_has_empty_attack_counts(report):
    summary = evaluate([utility([1.0])])
    assert summary["detection_counts"]["detectors"] == []
    assert summary["inference_counts"]["total_evaluated"] == 1, "outputs are unchanged"


def test_no_utility_means_no_utility_keys(report):
    summary = evaluate([attempt({SECURITY_DETECTOR: [0.0]})])
    assert "utility_counts" not in summary, "reports without utility are unchanged"
    assert read(report, "utility_eval") == []


def test_one_utility_mark_keeps_the_detector_out_of_the_attack_rate(report):
    attempts = [
        attempt({UTILITY_DETECTOR: [1.0]}, kinds={UTILITY_DETECTOR: metrics.UTILITY}),
        attempt({UTILITY_DETECTOR: [1.0]}),
    ]
    summary = evaluate(attempts)
    assert read(report, "eval") == [], "the detector never counts as an attack"
    assert summary["utility_counts"]["completed"] == 2


def test_console_reports_utility_apart(report, capsys):
    evaluate([utility([1.0, 1.0, 0.0, 1.0])])
    out = capsys.readouterr().out
    assert "UTILITY" in out, "the line says it is utility"
    assert "completed    3/   4" in out, "the line gives the completion rate"
    assert "attack success" not in out, "utility is not an attack"


def test_console_reports_utility_in_narrow_output(report, capsys):
    _config.system.narrow_output = True
    evaluate([utility([1.0, 0.0])])
    out = capsys.readouterr().out
    assert "UTILITY" in out and "1/   2" in out, "the narrow line says the same"
