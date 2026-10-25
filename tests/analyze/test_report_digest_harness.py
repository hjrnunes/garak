# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report digests of results written by a harness rather than a probe

EarlyStopHarness writes one summary attempt per intent stub, named after the
harness (``garak.harnesses.earlystop.EarlyStopHarness``) and scored by the
harness itself (``{"EarlyStop": [score]}``), followed by a single eval entry
for that harness/score pair. The fixture report mirrors that output."""

import json
import shutil

import pytest

from garak import _config
import garak.analyze.report_digest as report_digest

EARLYSTOP_REPORT = "tests/_assets/analyze/earlystop.report.jsonl"
HARNESS_GROUP = "harnesses.earlystop"
HARNESS_RESULT = "harnesses.earlystop.EarlyStopHarness"


@pytest.fixture
def earlystop_report(tmp_path):
    report_path = tmp_path / "earlystop.report.jsonl"
    shutil.copy(EARLYSTOP_REPORT, report_path)
    return report_path


def _append_eval(report_path, probe, detector, passed, total):
    with open(report_path, "a", encoding="utf-8") as reportfile:
        reportfile.write(
            json.dumps(
                {
                    "entry_type": "eval",
                    "probe": probe,
                    "detector": detector,
                    "passed": passed,
                    "fails": total - passed,
                    "nones": 0,
                    "total_evaluated": total,
                    "total_processed": total,
                }
            )
            + "\n"
        )


@pytest.mark.parametrize(
    "name, expected",
    [
        (
            "garak.harnesses.earlystop.EarlyStopHarness",
            ["harnesses.earlystop", "EarlyStopHarness"],
        ),
        (
            "harnesses.earlystop.EarlyStopHarness",
            ["harnesses.earlystop", "EarlyStopHarness"],
        ),
        ("probes.dan.Dan_11_0", ["dan", "Dan_11_0"]),
        ("dan.Dan_11_0", ["dan", "Dan_11_0"]),
    ],
)
def test_split_result_source(name, expected):
    assert list(report_digest._split_result_source(name)) == expected


def test_digest_earlystop_report(earlystop_report):
    _config.load_base_config()
    _config.reporting.taxonomy = None

    digest = report_digest.build_digest(str(earlystop_report))

    assert list(digest["eval"].keys()) == [HARNESS_GROUP]
    group = digest["eval"][HARNESS_GROUP]

    summary = group["_summary"]
    assert summary["group"] == HARNESS_GROUP
    assert "Early stopping harness" in summary["doc"]
    assert summary["group_link"].endswith(f"garak.{HARNESS_GROUP}.html")
    assert summary["score"] == pytest.approx(2 / 3)

    assert set(group.keys()) == {"_summary", HARNESS_RESULT}
    harness_result = group[HARNESS_RESULT]
    assert harness_result["_summary"]["probe_name"] == HARNESS_RESULT
    assert harness_result["_summary"]["probe_descr"] == "Early stopping harness"
    assert harness_result["_summary"]["probe_tags"] == []

    assert set(harness_result.keys()) == {"_summary", "EarlyStop"}
    verdict = harness_result["EarlyStop"]
    assert verdict["detector_name"] == "EarlyStop"
    assert verdict["absolute_score"] == pytest.approx(2 / 3)
    assert verdict["relative_score"] == "n/a"
    assert verdict["calibration_used"] is False
    assert verdict["detector_defcon"] == verdict["absolute_defcon"]
    assert verdict["total_evaluated"] == 3
    assert verdict["passed"] == 2

    assert digest["meta"]["calibration_used"] is False
    assert "EarlyStopHarness" in report_digest.build_html(digest)


def test_digest_earlystop_report_with_taxonomy(earlystop_report):
    _config.load_base_config()
    _config.reporting.taxonomy = "owasp"

    digest = report_digest.build_digest(str(earlystop_report))

    # harness results carry no taxonomy tags; they keep their harness group
    # rather than being filed as uncategorised
    assert list(digest["eval"].keys()) == [HARNESS_GROUP]
    assert digest["eval"][HARNESS_GROUP]["_summary"]["group"] == HARNESS_GROUP
    assert HARNESS_RESULT in digest["eval"][HARNESS_GROUP]


def test_digest_mixed_harness_and_probe_results(earlystop_report):
    _config.load_base_config()
    _config.reporting.taxonomy = None
    _append_eval(earlystop_report, "test.Blank", "always.Pass", passed=1, total=1)

    digest = report_digest.build_digest(str(earlystop_report))

    assert sorted(digest["eval"].keys()) == [HARNESS_GROUP, "test"]
    assert "test.Blank" in digest["eval"]["test"]
    assert "always.Pass" in digest["eval"]["test"]["test.Blank"]
    assert HARNESS_RESULT in digest["eval"][HARNESS_GROUP]


def test_digest_unknown_harness_raises(earlystop_report):
    _config.load_base_config()
    _config.reporting.taxonomy = None
    _append_eval(
        earlystop_report,
        "garak.harnesses.does_not_exist.NoSuchHarness",
        "EarlyStop",
        passed=1,
        total=1,
    )

    with pytest.raises(report_digest.ReportIncompatibleError) as exc_info:
        report_digest.build_digest(str(earlystop_report))
    assert "harnesses.does_not_exist" in str(exc_info.value)
