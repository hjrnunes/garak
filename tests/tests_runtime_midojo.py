# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Require the real MiDojo harness in tests that exercise a running handle."""

import importlib

import pytest


def require_runtime():
    """Skip only an absent harness; fail explicitly for an unrelated distribution."""
    try:
        module = importlib.import_module("midojo")
    except ModuleNotFoundError as exc:
        if exc.name == "midojo":
            pytest.skip("the optional MiDojo harness is absent")
        raise
    try:
        importlib.import_module("midojo.injection_spec")
        importlib.import_module("midojo.evaluation_handle")
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.fail(
            "installed midojo is not the MiDojo harness: the plain package-index "
            "name is unrelated; install a trusted source pin",
            pytrace=False,
        )
    return module
