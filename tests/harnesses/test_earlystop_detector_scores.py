# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for how EarlyStopHarness reads detector scores and reports failures.

A detector that gives None for an attempt makes no decision: it must neither
overrule another detector's hit nor count as a hit itself.
"""

import logging
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from garak import _config, evaluators
from garak.attempt import Attempt, Conversation, Turn, Message, ATTEMPT_STARTED
from garak.harnesses.earlystop import EarlyStopHarness, _is_rejected
from garak.intents import TextStub


def _make_attempt(output_text="Sure, here's how to do that"):
    stub = TextStub("T999test", "Refund the order")
    attempt = Attempt(
        probe_classname="test.TestProbe",
        intent=stub.intent,
        notes={"stub": stub},
        goal=stub.content,
        status=ATTEMPT_STARTED,
        prompt=Conversation(turns=[Turn("user", Message(stub.content))]),
    )
    attempt.outputs = [Message(output_text)]
    return attempt


def _detector(name, scores):
    detector = MagicMock()
    detector.detectorname = name
    detector.detect = MagicMock(return_value=scores)
    return detector


@pytest.fixture
def evaluator_env():
    """Set up the global environment needed for _is_rejected."""
    _config.load_base_config()

    temp_report_file = tempfile.NamedTemporaryFile(
        mode="w+", delete=False, encoding="utf-8"
    )
    _config.transient.reportfile = temp_report_file
    _config.transient.report_filename = temp_report_file.name

    yield evaluators.ThresholdEvaluator()

    temp_report_file.close()


def test_none_beside_a_hit_does_not_overrule_it(evaluator_env, caplog):
    caplog.set_level(logging.WARNING)
    detectors = [
        _detector("toolcall.ToolCallCondition", [1.0]),
        _detector("rubric_judge.RubricJudge", [None]),
    ]

    assert not _is_rejected(
        _make_attempt(), detectors, evaluator_env
    ), "a hit with no other real score is accepted"
    assert "rubric_judge.RubricJudge" in caplog.text


def test_detector_returning_none_outright_is_no_decision(evaluator_env, caplog):
    caplog.set_level(logging.WARNING)
    detectors = [
        _detector("toolcall.ToolCallCondition", [1.0]),
        _detector("rubric_judge.RubricJudge", None),
    ]

    assert not _is_rejected(_make_attempt(), detectors, evaluator_env)
    assert "rubric_judge.RubricJudge" in caplog.text


def test_none_beside_a_safe_score_is_rejected(evaluator_env, caplog):
    caplog.set_level(logging.WARNING)
    detectors = [
        _detector("toolcall.ToolCallCondition", [0.0]),
        _detector("rubric_judge.RubricJudge", [None]),
    ]

    assert _is_rejected(
        _make_attempt(), detectors, evaluator_env
    ), "the real safe score still rejects"
    assert "rubric_judge.RubricJudge" in caplog.text


def test_none_alone_is_rejected_with_a_warning(evaluator_env, caplog):
    caplog.set_level(logging.WARNING)
    detectors = [_detector("rubric_judge.RubricJudge", [None])]

    assert _is_rejected(
        _make_attempt(), detectors, evaluator_env
    ), "no real score is never a jailbreak hit"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        "rubric_judge.RubricJudge" in r.getMessage() for r in warnings
    ), "the warning names the detector that gave no score"


def test_real_scores_are_still_evaluated_per_output(evaluator_env, caplog):
    caplog.set_level(logging.WARNING)
    detectors = [_detector("toolcall.ToolCallCondition", [1.0, 1.0])]

    assert not _is_rejected(_make_attempt(), detectors, evaluator_env)
    assert caplog.text == "", "real scores log no warning"


def test_baseline_failure_is_logged_with_its_traceback(evaluator_env, caplog):
    caplog.set_level(logging.ERROR)
    stub = TextStub("T999test", "Refund the order")
    baseline_probe = MagicMock()
    baseline_probe.probe.side_effect = AttributeError("no notes")

    with patch("garak.harnesses.earlystop.IntentProbe", return_value=baseline_probe):
        accepted, rejected = EarlyStopHarness()._collect_baseline(
            MagicMock(), [], evaluator_env, [stub]
        )

    assert accepted == [], "a failed baseline accepts nothing"
    assert [a.notes["stub"] for a in rejected] == [stub]
    failures = [r for r in caplog.records if "baseline" in r.getMessage()]
    assert failures and failures[0].exc_info is not None, "traceback is logged"
    assert "AttributeError: no notes" in caplog.text
