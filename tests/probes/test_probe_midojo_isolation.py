# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A repeated chain needs an environment that isolates attempts, not one that resets."""

import sys
from types import ModuleType

import pytest

pytest.importorskip("midojo_spec")

from garak.environments.base import Environment, EnvironmentScope  # noqa: E402
from garak.exception import ConfigFailure  # noqa: E402
from garak.probes.midojo import CHAIN_NOTE  # noqa: E402

from injection_chain import (  # noqa: E402
    ScriptedTarget,
    build_probe,
    entry,
    report,  # noqa: F401
    world,  # noqa: F401
)

LOOKUP_101 = ("lookup_order", {"order_id": "ORD-101"})


class _Scope(EnvironmentScope):
    def observe(self):
        return {}

    def close(self):
        pass


@pytest.fixture()
def fresh(monkeypatch):
    """An environment that declares isolation and cannot reset; its reset requests."""
    resets = []

    class Fresh(Environment):
        """Gives each chain a fresh target, which an operator declares."""

        __module__ = "garak.environments.fresh"
        isolation = "disposable"
        capabilities = frozenset()

        def open(self, chain, *, reset=False):
            resets.append(reset)
            return _Scope()

    module = ModuleType(Fresh.__module__)
    module.DEFAULT_CLASS = "Fresh"
    module.Fresh = Fresh
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return resets


def test_a_repeat_runs_in_an_isolating_environment_that_cannot_reset(
    tmp_path, world, report, fresh
):
    probe = build_probe(
        tmp_path, entry("Look up ORD-101."), repeats=2, environment="fresh"
    )

    attempts = list(probe.probe(ScriptedTarget(world, [[LOOKUP_101], [LOOKUP_101]])))

    assert [a.notes[CHAIN_NOTE]["chain"] for a in attempts] == [
        0,
        1,
    ], "each repeat is its own chain"
    assert fresh == [False, False], "a fresh target needs no reset"


def test_repeats_are_refused_without_declared_isolation_and_say_what_is_needed(
    tmp_path, world, report
):
    probe = build_probe(tmp_path, entry("Look up ORD-101."), repeats=2)

    with pytest.raises(ConfigFailure, match="repeats refused") as caught:
        probe.probe(ScriptedTarget(world))

    assert "declares isolation" in str(
        caught.value
    ), "the refusal says what the environment must do"
    assert "environment_type" in str(caught.value), "and where to choose one"
    assert world.events == [], "nothing opened"


def test_the_shipped_disposable_environment_allows_repeats(tmp_path, world, report):
    probe = build_probe(
        tmp_path, entry("Look up ORD-101."), repeats=2, environment="disposable"
    )

    attempts = list(probe.probe(ScriptedTarget(world, [[LOOKUP_101], [LOOKUP_101]])))

    assert len(attempts) == 2, "a declared fresh target runs each repeat as a chain"
