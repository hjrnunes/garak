# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report tools carry the utility drop of paired runs and pool it over counts."""

import json
from pathlib import Path

import pytest

from garak import _config
from garak.analyze import report_digest
from garak.analyze.analyze_log import analyze_log
from garak.exception import ReportIncompatibleError

BASELINE = Path("tests/_assets/analyze/metric_kinds_baseline.report.jsonl")
DETECTOR = "midojo.EvaluationUtility"


def pairs(valid, benign, attacked, invalid=0, reasons=None):
    counts = {
        "pairs": valid + invalid,
        "valid": valid,
        "invalid": invalid,
        "benign_completed": benign,
        "attacked_completed": attacked,
        "benign_utility": benign / valid if valid else None,
        "utility_under_attack": attacked / valid if valid else None,
        "drop": (benign - attacked) / valid if valid else None,
    }
    if reasons:
        counts["invalid_reasons"] = reasons
    return counts


def entry(probe="test.Blank", **members):
    return {
        "entry_type": "utility_eval",
        "probe": probe,
        "detector": DETECTOR,
        "completed": 1,
        "not_completed": 1,
        "nones": 0,
        "total_evaluated": 2,
        "total_processed": 2,
        **members,
    }


def digest(tmp_path, records):
    _config.load_base_config()
    _config.reporting.taxonomy = None
    path = tmp_path / "pairs.report.jsonl"
    path.write_text(
        BASELINE.read_text(encoding="utf-8")
        + "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return report_digest.build_digest(str(path))


def test_the_digest_carries_the_pairs_of_a_utility_detector(tmp_path):
    summary = pairs(2, 2, 1)

    section = digest(tmp_path, [entry(pairs=summary)])["utility"]

    assert section["test.Blank"][DETECTOR]["pairs"] == summary, "pairs are reported"


def test_the_digest_names_what_its_two_utilities_cover(tmp_path):
    section = digest(tmp_path, [entry(pairs=pairs(1, 1, 0, 1))])["utility"]

    detector = section["test.Blank"][DETECTOR]
    assert detector["covers"] == {
        "counts": "all scored attacked chains",
        "pairs": "valid pairs only",
    }, "the plain score and the pair rates are labelled by what they cover"
    assert detector["score"] == 0.5, "the plain utility under attack"
    assert detector["pairs"]["utility_under_attack"] == 0.0, "and the valid pairs'"
    assert detector["pairs"]["invalid"] == 1, "with the invalid count beside them"


def test_the_digest_has_no_pairs_without_a_pair(tmp_path):
    section = digest(tmp_path, [entry()])["utility"]

    assert "pairs" not in section["test.Blank"][DETECTOR], "reports are unchanged"
    assert "covers" not in section["test.Blank"][DETECTOR], "and have no label"


def test_the_digest_pools_pairs_over_counts_not_rates(tmp_path):
    records = [
        entry(pairs=pairs(1, 1, 1)),
        entry(pairs=pairs(3, 3, 0, 1, {"baseline:errored": 1})),
        entry("test.Other", pairs=pairs(1, 1, 0)),
    ]

    section = digest(tmp_path, records)["utility"]

    assert section["test.Blank"][DETECTOR]["pairs"] == {
        "pairs": 5,
        "valid": 4,
        "invalid": 1,
        "benign_completed": 4,
        "attacked_completed": 1,
        "benign_utility": 1.0,
        "utility_under_attack": 0.25,
        "drop": 0.75,
        "invalid_reasons": {"baseline:errored": 1},
    }, "the rates are over the pooled valid pairs"
    assert section["test.Other"][DETECTOR]["pairs"]["drop"] == 1.0, "by probe"


def test_pairs_leave_the_digest_attack_success_alone(tmp_path):
    plain = digest(tmp_path, [entry()])
    paired = digest(tmp_path, [entry(pairs=pairs(1, 1, 0))])

    for key in ("eval", "technique_intent_matrix"):
        assert paired[key] == plain[key], f"{key} is the same with pairs"


BAD_PAIRS = [
    None,
    "x",
    [],
    {"pairs": 1},
    {**pairs(1, 1, 0), "valid": "1"},
    {**pairs(1, 2, 1)},
    {**pairs(1, 1, 2)},
    {**pairs(1, 1, 0), "pairs": 3},
]
BAD_PAIR_IDS = [
    "none",
    "text",
    "list",
    "missing-counts",
    "text-count",
    "benign-above-valid",
    "attacked-above-valid",
    "counts-do-not-add-up",
]


@pytest.mark.parametrize("bad", BAD_PAIRS, ids=BAD_PAIR_IDS)
def test_a_malformed_pairs_member_is_an_incompatible_report(tmp_path, bad):
    with pytest.raises(ReportIncompatibleError):
        digest(tmp_path, [entry(pairs=bad)])


@pytest.mark.parametrize("bad", BAD_PAIRS, ids=BAD_PAIR_IDS)
def test_analyze_log_refuses_a_malformed_pairs_member_alike(tmp_path, bad):
    path = tmp_path / "pairs.report.jsonl"
    path.write_text(json.dumps(entry(pairs=bad)) + "\n", encoding="utf-8")

    with pytest.raises(ReportIncompatibleError):
        analyze_log(str(path))


def test_analyze_log_summarises_the_drop(tmp_path, capsys):
    path = tmp_path / "pairs.report.jsonl"
    path.write_text(json.dumps(entry(pairs=pairs(2, 2, 1, 1))) + "\n", encoding="utf-8")

    analyze_log(str(path))

    lines = capsys.readouterr().out.splitlines()
    assert (
        "## paired drop: valid pairs only: 0.5000 over 2/3 valid pairs, 1 invalid"
        in lines
    ), "the drop is named, with its scope and the invalid pairs beside it"
    assert (
        "## utility, not an attack: test.Blank midojo.EvaluationUtility completed 1/2"
        " (utility under attack: all scored attacked chains)"
    ) in lines, "and the plain count is named as the other figure"


def test_analyze_log_labels_nothing_without_pairs(tmp_path, capsys):
    path = tmp_path / "plain.report.jsonl"
    path.write_text(json.dumps(entry()) + "\n", encoding="utf-8")

    analyze_log(str(path))

    out = capsys.readouterr().out
    assert "utility under attack" not in out and "paired drop" not in out


def test_analyze_log_computes_the_drop_from_the_counts(tmp_path, capsys):
    rates = ("benign_utility", "utility_under_attack", "drop")
    counts = {k: v for k, v in pairs(2, 2, 1, 1).items() if k not in rates}
    path = tmp_path / "pairs.report.jsonl"
    path.write_text(json.dumps(entry(pairs=counts)) + "\n", encoding="utf-8")

    analyze_log(str(path))

    assert (
        "## paired drop: valid pairs only: 0.5000 over 2/3 valid pairs, 1 invalid"
        in capsys.readouterr().out
    ), "a report without its rates still reads, from its counts"


def test_analyze_log_says_nothing_of_a_drop_without_a_valid_pair(tmp_path, capsys):
    path = tmp_path / "pairs.report.jsonl"
    path.write_text(json.dumps(entry(pairs=pairs(0, 0, 0, 1))) + "\n", encoding="utf-8")

    analyze_log(str(path))

    assert "drop" not in capsys.readouterr().out, "no rate for no valid pair"
