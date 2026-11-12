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
from garak.resources import metrics

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


def test_digest_pools_utility_results_repeated_for_a_probe_and_detector(tmp_path):
    _config.load_base_config()
    _config.reporting.taxonomy = None
    second = {
        "entry_type": "utility_eval",
        "probe": "test.Blank",
        "detector": UTILITY,
        "completed": 1,
        "not_completed": 3,
        "nones": 1,
        "total_evaluated": 4,
        "total_processed": 5,
        "unscored_reasons": {"errored": 1},
    }
    other_probe = dict(UTILITY_EVAL, probe="test.Other")
    aggregate = io.StringIO()
    _process_file_body(
        io.StringIO(_lines([UTILITY_EVAL, second, other_probe])), aggregate, "agg"
    )
    path = tmp_path / "aggregate.report.jsonl"
    path.write_text(BASELINE.read_text(encoding="utf-8") + aggregate.getvalue())

    utility = report_digest.build_digest(str(path))["utility"]

    assert utility["test.Blank"][UTILITY] == {
        "completed": 4,
        "not_completed": 4,
        "nones": 3,
        "total_evaluated": 8,
        "total_processed": 11,
        "not_applicable": 1,
        "unscored_reasons": {"no_utility_check": 1, "errored": 2},
        "score": 0.5,
    }, "the counts of every result add, and the rate is over the sum"
    assert utility["test.Other"][UTILITY]["completed"] == 3, "probes stay apart"


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


def _half_marked_report(tmp_path, probe, extra=()):
    """A report whose evaluator held one utility mark for a detector on two attempts"""
    marked = _attempt(probe, [0.0], [1.0])
    unmarked = dict(marked, notes={}, uuid="6a5d1c1e-0000-4000-8000-000000000002")
    path = tmp_path / "half.report.jsonl"
    path.write_text(_lines([marked, unmarked, *extra]), encoding="utf-8")
    return path


def _summary(probe):
    return {"entry_type": "probe_summary", "probe": probe}


def _utility_scope(tmp_path, rows):
    path = tmp_path / "scope.report.jsonl"
    path.write_text(_lines(rows), encoding="utf-8")
    return metrics.UtilityScope(str(path))


def test_utility_scope_covers_every_attempt_of_the_probe(tmp_path):
    marked = _attempt("test.Blank", [0.0], [1.0])
    scope = _utility_scope(tmp_path, [marked, dict(marked, notes={})])
    assert scope.covers("test.Blank", UTILITY), "one mark makes the detector utility"
    assert not scope.covers("test.Other", UTILITY), "the mark belongs to its probe"
    assert not scope.covers("test.Blank", "always.Fail"), "and to its detector"


@pytest.mark.parametrize("ending", ["eval", "utility_eval", "probe_summary"])
def test_utility_scope_moves_to_the_next_evaluation_after_its_entries(tmp_path, ending):
    marked = _attempt("test.Blank", [0.0], [1.0])
    rows = [
        dict(marked, notes={}),
        {"entry_type": ending},
        marked,
        {"entry_type": ending},
    ]
    scope = _utility_scope(tmp_path, rows)

    covered = []
    for row in rows:
        scope.observe(row)
        covered.append(scope.covers("test.Blank", UTILITY))

    assert covered == [
        False,
        False,
        True,
        True,
    ], "only the second evaluation has the mark"


def test_utility_scope_ignores_the_mark_of_an_unfinished_attempt(tmp_path):
    started = dict(_attempt("test.Blank", [0.0], [1.0]), status=1)
    scope = _utility_scope(tmp_path, [started])
    assert not scope.covers("test.Blank", UTILITY), "only completed attempts count"


def _two_evaluations(probe, summaries):
    """A security evaluation and a utility evaluation of one probe and detector"""
    security = dict(_attempt(probe, [0.0], [1.0]), notes={})
    utility = dict(
        _attempt(probe, [0.0], [1.0]), uuid="6a5d1c1e-0000-4000-8000-000000000003"
    )
    security_eval = {
        "entry_type": "eval",
        "probe": probe,
        "detector": UTILITY,
        "passed": 0,
        "fails": 1,
        "nones": 0,
        "total_evaluated": 1,
        "total_processed": 1,
    }
    utility_eval = dict(UTILITY_EVAL, probe=probe)
    summary = [_summary(probe)] if summaries else []
    return [security, security_eval, *summary, utility, utility_eval, *summary]


@pytest.mark.parametrize("summaries", [True, False])
def test_analyze_log_keeps_a_security_hit_when_another_evaluation_is_utility(
    tmp_path, capsys, summaries
):
    path = tmp_path / "aggregate.report.jsonl"
    path.write_text(_lines(_two_evaluations("test.Blank", summaries)), encoding="utf-8")

    analyze_log(str(path))

    hits = [
        line.split("\t")[1:3]
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("test.Blank\t")
    ]
    assert hits == [
        [UTILITY, "100.00%"],
        [UTILITY, "0.0000"],
    ], "the security attempt is a hit, the utility attempt is not"


@pytest.mark.parametrize("summaries", [True, False])
def test_qual_review_keeps_a_security_example_when_another_evaluation_is_utility(
    tmp_path, summaries
):
    path = tmp_path / "aggregate.report.jsonl"
    path.write_text(
        _lines(_two_evaluations("ansiescape.AnsiRaw", summaries)), encoding="utf-8"
    )

    (result,) = build_review(str(path))["tier_1_probe_results"]

    assert len(result["failing_examples"]) == 1, "only the security attempt is one"


def test_analyze_log_never_hits_on_a_utility_detector_with_a_missing_mark(
    tmp_path, capsys
):
    path = _half_marked_report(tmp_path, "test.Blank")

    analyze_log(str(path))

    hits = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("test.Blank\t")
    ]
    assert hits == [], "a completed task is not a hit on any attempt of the probe"


def test_qual_review_never_reviews_a_utility_detector_with_a_missing_mark(tmp_path):
    utility_eval = {
        "entry_type": "eval",
        "probe": "ansiescape.AnsiRaw",
        "detector": UTILITY,
        "passed": 0,
        "fails": 2,
        "nones": 0,
        "total_evaluated": 2,
        "total_processed": 2,
    }
    path = _half_marked_report(tmp_path, "ansiescape.AnsiRaw", [utility_eval])

    review = build_review(str(path))

    (result,) = review["tier_1_probe_results"]
    assert result["probe_detector"] == f"ansiescape.AnsiRaw+{UTILITY}"
    assert result["failing_examples"] == [], "a completed task is no failing example"


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
