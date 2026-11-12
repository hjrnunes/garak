# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guard against a similarly named but unrelated runtime distribution."""

import sys
from types import ModuleType

import pytest

from garak.exception import ConfigFailure
from garak.probes.midojo import _midojo
from tests_runtime_midojo import require_runtime


def test_an_unrelated_midojo_package_fails_instead_of_skipping(monkeypatch):
    fake = ModuleType("midojo")
    fake.__path__ = []
    monkeypatch.setitem(sys.modules, "midojo", fake)
    for name in ("midojo.injection_spec", "midojo.evaluation_handle"):
        monkeypatch.delitem(sys.modules, name, raising=False)

    with pytest.raises(ConfigFailure, match="not the MiDojo harness.*package-index"):
        _midojo()
    with pytest.raises(pytest.fail.Exception, match="not the MiDojo harness"):
        require_runtime()
