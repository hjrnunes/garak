# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report digests of results written by a harness rather than a probe

EarlyStopHarness logs every baseline and attack attempt it scores, then writes
one summary attempt per intent stub, named after the harness
(``garak.harnesses.earlystop.EarlyStopHarness``) and scored by the harness
itself (``{"EarlyStop": [score]}``): rejected stubs first, then accepted stubs
carrying the conversation of the attack that succeeded. A single eval entry
for that harness/score pair follows. The fixture report was written by
``EarlyStopHarness.run()`` with stubbed intents, probes, detector and target:
three stubs, all refused at baseline, one jailbroken by the attack probe."""

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
    assert (
        list(report_digest._split_result_source(name)) == expected
    ), "harness names keep their harnesses. module; probe names split as before"


def test_digest_earlystop_report(earlystop_report):
    _config.load_base_config()
    _config.reporting.taxonomy = None

    digest = report_digest.build_digest(str(earlystop_report))

    assert list(digest["eval"].keys()) == [
        HARNESS_GROUP
    ], "harness results should form one group named after the harness"
    group = digest["eval"][HARNESS_GROUP]

    summary = group["_summary"]
    assert summary["group"] == HARNESS_GROUP, "group name should be the harness"
    assert (
        "Early stopping harness" in summary["doc"]
    ), "group doc should come from the harness module docstring"
    assert summary["group_link"].endswith(
        f"garak.{HARNESS_GROUP}.html"
    ), "group link should point at the harness reference page"
    assert summary["score"] == pytest.approx(
        2 / 3
    ), "group score should be the share of stubs not jailbroken"

    assert set(group.keys()) == {
        "_summary",
        HARNESS_RESULT,
    }, "only the harness summary should be reported, not in-between attempts"
    harness_result = group[HARNESS_RESULT]
    probe_summary = harness_result["_summary"]
    assert (
        probe_summary["probe_name"] == HARNESS_RESULT
    ), "result should be named after the harness class"
    assert (
        probe_summary["probe_descr"] == "Early stopping harness"
    ), "description should come from the harness module docstring"
    assert probe_summary["probe_tags"] == [], "harness results carry no tags"

    assert set(harness_result.keys()) == {
        "_summary",
        "EarlyStop",
    }, "the harness verdict should be reported as its only score"
    verdict = harness_result["EarlyStop"]
    assert verdict["detector_name"] == "EarlyStop", "verdict keeps its name"
    assert verdict["absolute_score"] == pytest.approx(
        2 / 3
    ), "two of three stubs resisted every attack"
    assert verdict["relative_score"] == "n/a", "harness results have no calibration"
    assert verdict["calibration_used"] is False, "no calibration for harness results"
    assert (
        verdict["detector_defcon"] == verdict["absolute_defcon"]
    ), "without calibration the rating should rest on the absolute score"
    assert verdict["total_evaluated"] == 3, "one evaluation per stub"
    assert verdict["passed"] == 2, "two stubs were never jailbroken"

    assert digest["meta"]["calibration_used"] is False, "no calibration was used"
    assert "EarlyStopHarness" in report_digest.build_html(
        digest
    ), "HTML report should include the harness result"


def test_digest_earlystop_report_with_taxonomy(earlystop_report):
    _config.load_base_config()
    _config.reporting.taxonomy = "owasp"

    digest = report_digest.build_digest(str(earlystop_report))

    assert list(digest["eval"].keys()) == [
        HARNESS_GROUP
    ], "untagged harness results should keep their harness group, not 'other'"
    assert (
        digest["eval"][HARNESS_GROUP]["_summary"]["group"] == HARNESS_GROUP
    ), "group should be named after the harness, not 'Uncategorized'"
    assert (
        HARNESS_RESULT in digest["eval"][HARNESS_GROUP]
    ), "harness result should be listed in its group"


def test_digest_mixed_harness_and_probe_results(earlystop_report):
    _config.load_base_config()
    _config.reporting.taxonomy = None
    _append_eval(earlystop_report, "test.Blank", "always.Pass", passed=1, total=1)

    digest = report_digest.build_digest(str(earlystop_report))

    assert sorted(digest["eval"].keys()) == [
        HARNESS_GROUP,
        "test",
    ], "harness and probe results should get separate groups"
    assert "test.Blank" in digest["eval"]["test"], "probe result grouped as before"
    assert (
        "always.Pass" in digest["eval"]["test"]["test.Blank"]
    ), "probe detector reported as before"
    assert (
        HARNESS_RESULT in digest["eval"][HARNESS_GROUP]
    ), "harness result reported alongside probe results"


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
    assert "harnesses.does_not_exist" in str(
        exc_info.value
    ), "error should name the unknown harness"
