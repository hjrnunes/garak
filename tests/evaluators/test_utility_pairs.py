# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The evaluator reports the utility drop over the valid pairs of a probe."""

import json
import uuid

import pytest

from garak import _config
from garak.attempt import Attempt, Message
from garak.evaluators.base import ThresholdEvaluator
from garak.resources import metrics

PROBE = "probe.ModuleA"
UTILITY_DETECTOR = "det.Utility"
OTHER_DETECTOR = "det.OtherUtility"


@pytest.fixture()
def report(tmp_path):
    _config.load_base_config()
    _config.reporting.confidence_interval_method = "none"
    _config.reporting.hit_at_k = [1]
    _config.transient.report_filename = str(tmp_path / "pairs.report.jsonl")
    _config.transient.reportfile = open(
        _config.transient.report_filename, "w", buffering=1, encoding="utf-8"
    )
    _config.transient.run_id = uuid.uuid4()
    _config.transient.hitlogfile = None
    _config.run.generations = 1
    _config.system.show_z = False
    _config.system.verbose = 0
    _config.system.narrow_output = False
    return _config.transient.report_filename


def read(path, entry_type):
    _config.transient.reportfile.flush()
    with open(path, encoding="utf-8") as lines:
        return [r for r in map(json.loads, lines) if r["entry_type"] == entry_type]


def attempt(score, role=None, pair="p1", reason=None, detector=UTILITY_DETECTOR):
    built = Attempt(prompt=Message(text="prompt"), probe_classname=PROBE)
    built.outputs = ["output"]
    built.detector_results = {detector: [score]}
    metrics.mark_kind(built, detector, metrics.UTILITY)
    if role is not None:
        metrics.mark_pair(built, pair, role)
    if reason is not None:
        metrics.mark_unscored(built, detector, reason)
    return built


def pair(benign, attacked, name):
    return [
        attempt(
            attacked, metrics.ATTACKED, name, "errored" if attacked is None else None
        ),
        attempt(benign, metrics.BASELINE, name),
    ]


def evaluate(attempts):
    return ThresholdEvaluator(0.5).evaluate(attempts)


def test_pairs_are_reported_beside_the_utility_counts(report):
    attempts = [*pair(1.0, 1.0, "a"), *pair(1.0, 0.0, "b")]

    summary = evaluate(attempts)

    (record,) = read(report, "utility_eval")
    want = {
        "pairs": 2,
        "valid": 2,
        "invalid": 0,
        "benign_completed": 2,
        "attacked_completed": 1,
        "benign_utility": 1.0,
        "utility_under_attack": 0.5,
        "drop": 0.5,
    }
    assert record["pairs"] == want, "the utility_eval entry carries the pair summary"
    assert summary["utility_counts"]["pairs"] == want, "so does the probe summary"
    (logged,) = read(report, "probe_summary")
    assert logged["utility_counts"]["pairs"] == want, "and so does the logged one"


def test_the_counts_cover_the_attempts_under_test_not_the_baselines(report):
    evaluate([*pair(1.0, 1.0, "a"), *pair(1.0, 0.0, "b")])

    (record,) = read(report, "utility_eval")

    assert (record["completed"], record["not_completed"]) == (
        1,
        1,
    ), "the rate is utility under attack, not a mix of benign and attacked runs"
    assert record["total_evaluated"] == 2


def test_a_scored_attacked_chain_of_an_invalid_pair_stays_in_the_plain_counts(report):
    errored = attempt(None, metrics.BASELINE, "b", "errored")

    evaluate([*pair(1.0, 1.0, "a"), attempt(0.0, metrics.ATTACKED, "b"), errored])

    (record,) = read(report, "utility_eval")
    assert (record["completed"], record["total_evaluated"]) == (
        1,
        2,
    ), "every scored chain under test is counted, so the counts reconcile"
    assert record["pairs"]["invalid_reasons"] == {"baseline:errored": 1}
    assert record["pairs"]["utility_under_attack"] == 1.0, "the pair rate is not"


def test_a_run_without_a_pair_has_no_pairs_member(report):
    summary = evaluate([attempt(1.0), attempt(0.0)])

    (record,) = read(report, "utility_eval")
    assert "pairs" not in record, "reports without pairs are unchanged"
    assert "pairs" not in summary["utility_counts"]


def test_an_unscored_side_makes_the_pair_invalid_and_is_declared(report):
    summary = evaluate([*pair(1.0, 0.0, "a"), *pair(1.0, None, "b")])

    (record,) = read(report, "utility_eval")
    pairs = record["pairs"]
    assert (pairs["valid"], pairs["invalid"]) == (1, 1)
    assert pairs["invalid_reasons"] == {"attacked:errored": 1}
    assert pairs["drop"] == 1.0, "the invalid pair is in no rate"
    assert record["unscored_reasons"] == {
        "errored": 1
    }, "the attempt under test still declares why it has no score"
    assert summary["utility_counts"]["pairs"] == pairs


@pytest.mark.parametrize(
    "benign,attacked,drop", [(0.5, 0.49, 1.0), (0.5, 0.5, 0.0), (0.49, 0.5, -1.0)]
)
def test_pairs_follow_the_evaluator_threshold(report, benign, attacked, drop):
    evaluate(pair(benign, attacked, "a"))

    (record,) = read(report, "utility_eval")
    assert record["pairs"]["drop"] == drop, "completion is a score at or above it"


def test_pairs_are_pooled_over_the_utility_detectors_of_a_probe(report):
    other = [
        attempt(1.0, metrics.ATTACKED, "x", detector=OTHER_DETECTOR),
        attempt(1.0, metrics.BASELINE, "x", detector=OTHER_DETECTOR),
    ]

    summary = evaluate([*pair(1.0, 0.0, "a"), *other])

    assert len(read(report, "utility_eval")) == 2, "each detector has its own entry"
    pooled = summary["utility_counts"]["pairs"]
    assert (pooled["pairs"], pooled["valid"]) == (2, 2), "the probe summary pools them"
    assert pooled["drop"] == 0.5, "over pooled counts"


def test_pairs_leave_attack_success_statistics_alone(report):
    security = Attempt(prompt=Message(text="prompt"), probe_classname=PROBE)
    security.outputs = ["output"]
    security.detector_results = {"det.Security": [1.0]}
    alone = evaluate([security])
    (eval_alone,) = read(report, "eval")

    paired = evaluate([security, *pair(1.0, 0.0, "a")])

    assert paired["detection_counts"] == alone["detection_counts"]
    assert read(report, "eval")[-1] == eval_alone, "the eval record is unchanged"


def test_console_reports_the_drop(report, capsys):
    evaluate([*pair(1.0, 1.0, "a"), *pair(1.0, 0.0, "b"), *pair(1.0, None, "c")])

    out = capsys.readouterr().out

    assert "pairs:" in out and "2/3 valid" in out, "the line counts valid pairs"
    assert "benign 100.00%, under attack 50.00%, drop 50.00%" in out
    assert "invalid attacked:errored: 1" in out, "the invalid pair is declared"


def test_console_says_so_when_no_pair_is_valid(report, capsys):
    evaluate(pair(1.0, None, "a"))

    out = capsys.readouterr().out

    assert "0/1 valid" in out and "drop" not in out, "no rate is shown for no pair"
