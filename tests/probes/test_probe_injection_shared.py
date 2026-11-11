# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests of scripted injection source and turn handling."""

import pytest

from garak.attempt import Message
from garak.probes._injection_shared import resolve_source, turn_failure


def test_source_prefers_an_existing_file(tmp_path):
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    assert resolve_source(str(source)) == source, "an absolute source is preserved"


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        (None, "ResponsesNotCompleted"),
        (Message("", notes={"error": {"type": "TimedOut"}}), "TimedOut"),
        (Message("", notes={"response_status": "incomplete"}), "ResponsesNotCompleted"),
        (Message("", notes={"response_status": "completed"}), "EmptyReply"),
        (Message("refused", notes={"response_status": "completed"}), None),
    ],
)
def test_turn_failure_names_an_unfinished_response(output, expected):
    assert turn_failure(output) == expected, "the turn's disposition is explicit"
