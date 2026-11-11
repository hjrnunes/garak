# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attack success statistics must not depend on how a report is processed.

``metric_kinds_baseline.report.jsonl`` and ``metric_kinds_baseline.digests.json``
in ``tests/_assets/analyze`` were written by the evaluator and the report digest
before detectors could declare a metric kind. The digests file also holds the
statistics of the other saved reports in that directory. These tests run the
current evaluator and digest over the same saved data and expect identical
``eval`` and ``detection_counts`` values.
"""

import json
import uuid
from pathlib import Path

import pytest

from garak import _config
from garak.attempt import Attempt, Message
from garak.evaluators.base import ThresholdEvaluator
import garak.analyze.report_digest as report_digest

ASSETS = Path("tests/_assets/analyze")
BASELINE = ASSETS / "metric_kinds_baseline.report.jsonl"
GOLDENS = json.loads((ASSETS / "metric_kinds_baseline.digests.json").read_text())


def _records(path):
    with open(path, encoding="utf-8") as report:
        return [json.loads(line) for line in report if line.strip()]


def _statistics(digest):
    """Pick the statistics out of a digest, as the baseline run recorded them."""
    evals, probe_counts = {}, {}
    for group, probes in digest["eval"].items():
        for probe, entries in probes.items():
            if probe == "_summary":
                continue
            probe_counts[probe] = entries["_summary"]["probe_counts"]
            for detector, entry in entries.items():
                if detector != "_summary":
                    evals.setdefault(group, {}).setdefault(probe, {})[detector] = {
                        key: entry.get(key)
                        for key in ("passed", "total_evaluated", "absolute_score")
                    }
    return {
        "eval": evals,
        "probe_counts": probe_counts,
        "technique_intent_matrix": digest["technique_intent_matrix"],
    }


def _sorted_detectors(value):
    """Order the detector lists, which the evaluator writes from a set."""
    if isinstance(value, dict):
        return {
            key: sorted(item) if key == "detectors" else _sorted_detectors(item)
            for key, item in value.items()
        }
    return value


@pytest.fixture()
def run_report(tmp_path):
    _config.load_base_config()
    _config.reporting.confidence_interval_method = "none"
    _config.transient.report_filename = str(tmp_path / "replay.report.jsonl")
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


def _saved_attempts():
    attempts = {}
    for record in _records(BASELINE):
        if record["entry_type"] != "attempt":
            continue
        attempt = Attempt(prompt=Message(text="saved prompt"))
        attempt.probe_classname = record["probe_classname"]
        attempt.seq = record["seq"]
        attempt.goal = record["goal"]
        attempt.outputs = [output["text"] for output in record["outputs"]]
        attempt.detector_results = record["detector_results"]
        attempt.intent = record["intent"]
        attempts.setdefault(record["probe_classname"], []).append(attempt)
    return attempts


def test_evaluator_reproduces_saved_eval_and_summary_records(run_report):
    evaluator = ThresholdEvaluator(0.5)
    for attempts in _saved_attempts().values():
        evaluator.evaluate(attempts)
    _config.transient.reportfile.flush()

    wanted = ("eval", "probe_summary")
    replayed = [r for r in _records(run_report) if r["entry_type"] in wanted]
    saved = [r for r in _records(BASELINE) if r["entry_type"] in wanted]

    assert len(saved) == 7, "the saved run has five eval and two probe summary rows"
    assert [_sorted_detectors(r) for r in replayed] == [
        _sorted_detectors(r) for r in saved
    ], "a replay of saved attempts must write the saved eval and summary values"


@pytest.mark.parametrize("name", sorted(GOLDENS))
def test_digest_reproduces_saved_statistics(name):
    path = (
        BASELINE if name == "metric_kinds_baseline" else ASSETS / f"{name}.report.jsonl"
    )
    _config.load_base_config()
    _config.reporting.taxonomy = None

    digest = report_digest.build_digest(str(path))

    assert _sorted_detectors(_statistics(digest)) == _sorted_detectors(
        GOLDENS[name]
    ), f"{name}: digest statistics must match the baseline run"
