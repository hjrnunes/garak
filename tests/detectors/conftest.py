# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path

import pytest

FAKE_JUDGE = (
    Path(__file__).parents[1]
    / "_assets"
    / "detectors"
    / "reply_judge"
    / "fake_judge.py"
)


def load_fake_judge():
    """The fake judge service module: a file of the asset directory, which no package holds."""
    name = "reply_judge_fake_judge"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, FAKE_JUDGE)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # a dataclass reads its module while it is made
        spec.loader.exec_module(module)
    return sys.modules[name]


@pytest.fixture
def fake_judge():
    """A fake judge service on a free loopback port, closed after the test."""
    service = load_fake_judge().FakeJudge()
    yield service
    service.close()
