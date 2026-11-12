# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A baseline run is benign, so no security detector scores it."""

import json
import uuid

import pytest

import garak.harnesses.base
from garak import _config
from garak.attempt import Attempt, Message
from garak.detectors import always
from garak.evaluators.base import ThresholdEvaluator
from garak.resources import metrics

PROBE = "probe.ModuleA"


class _Utility(always.Pass):
    """A detector that scores task completion."""

    metric_kind = metrics.UTILITY


class _Spy(always.Fail):
    """A security detector that notes that it ran."""

    def detect(self, attempt):
        _Spy.ran += 1
        return super().detect(attempt)


_Spy.ran = 0


def built(role=None, outputs=2):
    attempt = Attempt(prompt=Message(text="prompt"), probe_classname=PROBE)
    attempt.outputs = [f"output {i}" for i in range(outputs)]
    if role is not None:
        metrics.mark_pair(attempt, "p1", role)
    return attempt


def run(attempts, detector):
    garak.harnesses.base.Harness()._run_detector(attempts, detector)
    return [list(a.detector_results.values())[0] for a in attempts]


@pytest.fixture(autouse=True)
def _reset_spy():
    _Spy.ran = 0


def test_a_security_detector_leaves_a_baseline_unscored_and_does_not_run():
    baseline = built(metrics.BASELINE)

    (scores,) = run([baseline], _Spy())

    assert scores == [None, None], "one unscored entry per output"
    assert _Spy.ran == 0, "the detector never sees a baseline"
    (name,) = baseline.detector_results
    assert metrics.unscored_mark(baseline, name) == (
        metrics.BASELINE_SECURITY,
        True,
    ), "a baseline is not applicable to a security metric, not an unknown"
    assert metrics.metric_kind(baseline, name) == metrics.SECURITY


@pytest.mark.parametrize("role", [None, metrics.ATTACKED])
def test_a_security_detector_scores_every_other_attempt(role):
    attempt = built(role)

    (scores,) = run([attempt], _Spy())

    assert scores == [1.0, 1.0] and _Spy.ran == 1, "an attacked run is an attack"
    assert metrics.UNSCORED_NOTE not in attempt.notes, "nothing to explain"


def test_a_utility_detector_scores_a_baseline():
    baseline = built(metrics.BASELINE)

    (scores,) = run([baseline], _Utility())

    assert scores == [0.0, 0.0], "a baseline measures utility without the attack"
    (name,) = baseline.detector_results
    assert metrics.metric_kind(baseline, name) == metrics.UTILITY
    assert metrics.UNSCORED_NOTE not in baseline.notes


def test_a_pair_note_with_no_readable_role_is_no_baseline():
    attempt = built()
    attempt.notes[metrics.PAIR_NOTE] = {"id": "p1", "role": "baseline "}

    (scores,) = run([attempt], _Spy())

    assert scores == [1.0, 1.0], "only a readable baseline role is skipped"


@pytest.fixture()
def report(tmp_path):
    _config.load_base_config()
    _config.reporting.confidence_interval_method = "none"
    _config.reporting.hit_at_k = [1]
    _config.transient.report_filename = str(tmp_path / "baseline.report.jsonl")
    _config.transient.reportfile = open(
        _config.transient.report_filename, "w", buffering=1, encoding="utf-8"
    )
    _config.transient.run_id = uuid.uuid4()
    _config.transient.hitlogfile = None
    _config.run.generations = 2
    _config.system.show_z = False
    _config.system.verbose = 0
    _config.system.narrow_output = False
    return _config.transient.report_filename


def test_a_baseline_stays_out_of_the_attack_success_rate(report):
    attempts = [built(metrics.ATTACKED), built(metrics.BASELINE), built(None)]
    run(attempts, _Spy())

    ThresholdEvaluator(0.5).evaluate(attempts)

    _config.transient.reportfile.flush()
    with open(report, encoding="utf-8") as lines:
        (entry,) = [r for r in map(json.loads, lines) if r["entry_type"] == "eval"]
    assert (entry["total_evaluated"], entry["fails"]) == (4, 4), "two attacks scored"
    assert entry["total_processed"] == 6, "the baseline's outputs are processed"
    assert entry["not_applicable"] == 2, "the baseline is not an unknown"
    assert entry["unscored_reasons"] == {
        metrics.BASELINE_SECURITY: 2
    }, "the exclusion is declared with its reason"
