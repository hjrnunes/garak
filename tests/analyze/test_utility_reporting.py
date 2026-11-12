# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report tools keep utility results apart from attack success.

A utility detector writes ``utility_eval`` entries, and a note on each attempt
that names it as utility. The report tools must carry those entries, must not
read a completed task as an attack, and must not fail on outputs without a score.
"""

import io
import json
from pathlib import Path

import pytest

from garak import _config
from garak.analyze import report_digest
from garak.analyze.aggregate_reports import _process_file_body
from garak.analyze.analyze_log import analyze_log
from garak.analyze.qual_review import build_review

BASELINE = Path("tests/_assets/analyze/metric_kinds_baseline.report.jsonl")
UTILITY = "midojo.TaskCompleted"
UTILITY_EVAL = {
    "entry_type": "utility_eval",
    "probe": "test.Blank",
    "detector": UTILITY,
    "completed": 3,
    "not_completed": 1,
    "nones": 2,
    "total_evaluated": 4,
    "total_processed": 6,
    "not_applicable": 1,
    "unscored_reasons": {"no_utility_check": 1, "errored": 1},
}


def _lines(records):
    return "".join(json.dumps(record) + "\n" for record in records)


def _attempt(probe, scores, utility_scores):
    return {
        "entry_type": "attempt",
        "uuid": "6a5d1c1e-0000-4000-8000-000000000001",
        "status": 2,
        "probe_classname": probe,
        "prompt": "a prompt",
        "outputs": [f"output {i}" for i in range(len(scores))],
        "detector_results": {"always.Fail": scores, UTILITY: utility_scores},
        "notes": {"metric_kind": {UTILITY: "utility"}},
    }


@pytest.fixture()
def digest_pair(tmp_path):
    _config.load_base_config()
    _config.reporting.taxonomy = None
    saved = BASELINE.read_text(encoding="utf-8")
    with_utility = tmp_path / "utility.report.jsonl"
    with_utility.write_text(saved + _lines([UTILITY_EVAL]), encoding="utf-8")
    return (
        report_digest.build_digest(str(BASELINE)),
        report_digest.build_digest(str(with_utility)),
    )


def test_digest_without_utility_has_no_utility_section(digest_pair):
    without, _ = digest_pair
    assert "utility" not in without, "a report without utility is unchanged"


def test_digest_reports_utility_in_its_own_section(digest_pair):
    _, with_utility = digest_pair
    assert with_utility["utility"] == {
        "test.Blank": {
            UTILITY: {
                "completed": 3,
                "not_completed": 1,
                "nones": 2,
                "total_evaluated": 4,
                "total_processed": 6,
                "not_applicable": 1,
                "unscored_reasons": {"no_utility_check": 1, "errored": 1},
                "score": 0.75,
            }
        }
    }, "utility is listed by probe and detector with its completion rate"


def test_digest_attack_success_ignores_utility(digest_pair):
    without, with_utility = digest_pair
    for key in ("eval", "technique_intent_matrix"):
        assert with_utility[key] == without[key], f"{key} is the same with utility"


def test_utility_without_scored_outputs_has_no_rate(tmp_path):
    _config.load_base_config()
    _config.reporting.taxonomy = None
    empty = dict(UTILITY_EVAL, completed=0, not_completed=0, total_evaluated=0)
    path = tmp_path / "empty.report.jsonl"
    path.write_text(BASELINE.read_text(encoding="utf-8") + _lines([empty]))
    digest = report_digest.build_digest(str(path))
    assert digest["utility"]["test.Blank"][UTILITY]["score"] is None, "no rate over 0"


def test_aggregate_keeps_utility_entries():
    source = io.StringIO(_lines([UTILITY_EVAL, {"entry_type": "digest"}]))
    sink = io.StringIO()

    _process_file_body(source, sink, "aggregate-run")

    assert [json.loads(line) for line in sink.getvalue().splitlines()] == [
        UTILITY_EVAL
    ], "an aggregated report keeps its utility entries"


def test_analyze_log_neither_fails_on_unscored_nor_hits_on_utility(tmp_path, capsys):
    path = tmp_path / "utility.report.jsonl"
    path.write_text(
        _lines(
            [
                _attempt("test.Blank", [1.0, None], [1.0, 1.0]),
                {
                    "entry_type": "eval",
                    "probe": "test.Blank",
                    "detector": "always.Fail",
                    "passed": 0,
                    "fails": 1,
                    "nones": 1,
                    "total_evaluated": 1,
                    "total_processed": 2,
                },
                UTILITY_EVAL,
            ]
        ),
        encoding="utf-8",
    )

    analyze_log(str(path))

    lines = capsys.readouterr().out.splitlines()
    hits = [line for line in lines if line.startswith("test.Blank\t")]
    assert [line.split("\t")[1:3] for line in hits] == [
        ["always.Fail", "100.00%"],
        ["always.Fail", "0.0000"],
    ], "a hit rate counts scored outputs, the eval row is attack success"
    assert any(
        line.startswith("##") and UTILITY in line and "3/" in line for line in lines
    ), "the completed tasks are summarised, marked as not an attack"


def test_qual_review_neither_fails_on_unscored_nor_reviews_utility(tmp_path):
    path = tmp_path / "utility.report.jsonl"
    path.write_text(
        _lines(
            [
                _attempt("ansiescape.AnsiRaw", [1.0, None], [1.0, 1.0]),
                {
                    "entry_type": "eval",
                    "probe": "ansiescape.AnsiRaw",
                    "detector": "always.Fail",
                    "passed": 0,
                    "fails": 1,
                    "nones": 1,
                    "total_evaluated": 1,
                    "total_processed": 2,
                },
            ]
        ),
        encoding="utf-8",
    )

    review = build_review(str(path))

    (result,) = review["tier_1_probe_results"]
    assert result["probe_detector"] == "ansiescape.AnsiRaw+always.Fail"
    assert len(result["failing_examples"]) == 1, "only the scored hit is an example"
    assert result["passing_examples"] == [], "an unscored output is no example"
