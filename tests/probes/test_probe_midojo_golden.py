# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test that the injection-spec probe seals what the MiDojo runtime sealed.

Each consumer fixture (``tests/_assets/evidence/consumer_ipi``) holds the compiled
source entry, the snapshot the runtime sealed for a target's run of it and the
verdict it graded. The probe runs the source file unchanged against a fake
injector and a target that makes the recorded calls at the recorded times, on a
pinned clock and evaluation id, and the trace must hold the runtime's snapshot,
its digest and its verdict.
"""

import functools
import json
from pathlib import Path

import pytest

pytest.importorskip("midojo_spec")

import garak.probes.midojo  # noqa: E402
from garak.attempt import Message  # noqa: E402
from garak.evidence.collector import Collector  # noqa: E402
from garak.probes.midojo import InjectionSpecProbe, TRACE_NOTE  # noqa: E402

from injection_chain import (  # noqa: E402
    UPSTREAM,
    FakeArming,
    ScriptedTarget,
    config,
    report,  # noqa: F401
    world,  # noqa: F401
)
from sealed_replay import ScriptedClock, call_fields  # noqa: E402

CONSUMER = Path(__file__).parents[1] / "_assets" / "evidence" / "consumer_ipi"

#: the runtime graded ``reply-judged`` with a scripted judge service, which a probe
#: test has no use for; its snapshot is the one this test compares
SNAPSHOT_ONLY = {"reply-judged"}
#: ``incomplete`` holds a call that never finished, which no finished call records
REPLAYED = ["blocked", "delivered", "reply-judged", "sequential", "undelivered"]


class RecordedTarget(ScriptedTarget):
    """A target that makes the calls a sealed snapshot holds, at the times it holds them."""

    def __init__(self, world, snapshot, clock):
        super().__init__(world)
        self.snapshot = snapshot
        self.clock = clock

    def generate(self, prompt, generations_this_call=1):
        self.prompts.append(prompt)
        index = len(self.prompts) - 1
        arming = self.arming(prompt)
        for call in self.snapshot["function_calls"]:
            if call["turn"] == index:
                self.clock.at(call["timestamp"])
                arming.record({**call_fields(call), "call_id": call["call_id"]})
        turn = self.snapshot["turns"][index]
        self.clock.at(turn["ended_at"])
        return [Message(turn["reply"], notes={"response_status": "completed"})]


def _run_recorded(case, monkeypatch, world):
    """Run a consumer source file against the recorded target; return its attempts."""
    folder = CONSUMER / case
    snapshot = json.loads((folder / "snapshot.json").read_text(encoding="utf-8"))
    clock = ScriptedClock(snapshot["turns"][0]["started_at"])

    monkeypatch.setattr(
        garak.probes.midojo, "Collector", functools.partial(Collector, clock=clock)
    )
    monkeypatch.setattr(
        InjectionSpecProbe, "_new_evaluation_id", lambda self: snapshot["evaluation_id"]
    )
    release = FakeArming.disarm

    def disarm(self):
        clock.at(snapshot["sealed_at"])
        return release(self)

    monkeypatch.setattr(FakeArming, "disarm", disarm)

    probe = InjectionSpecProbe(
        config_root=config(folder / "ipi-episodes.json", upstream=UPSTREAM)
    )
    starting = probe._generator_precall_hook

    def start_turn(generator, attempt=None):
        note = attempt.notes[garak.probes.midojo.CHAIN_NOTE]
        clock.at(snapshot["turns"][note["turn"]]["started_at"])
        return starting(generator, attempt)

    monkeypatch.setattr(probe, "_generator_precall_hook", start_turn)
    return list(probe.probe(RecordedTarget(world, snapshot, clock))), folder, snapshot


@pytest.mark.parametrize("case", REPLAYED)
def test_the_trace_holds_the_snapshot_the_runtime_sealed(
    case, monkeypatch, world, report
):
    attempts, _, snapshot = _run_recorded(case, monkeypatch, world)

    trace = attempts[-1].notes[TRACE_NOTE]
    assert trace["sealed"] == "final" and trace["error"] is None
    assert (
        trace["snapshot_id"] == snapshot["snapshot_id"]
    ), "the seal digests as the runtime's"
    assert trace["snapshot"] == snapshot, "the trace's snapshot is the runtime's value"


@pytest.mark.parametrize("case", sorted(set(REPLAYED) - SNAPSHOT_ONLY))
def test_the_trace_holds_the_verdict_the_runtime_graded(
    case, monkeypatch, world, report
):
    attempts, folder, _ = _run_recorded(case, monkeypatch, world)

    verdict = json.loads((folder / "verdict.json").read_text(encoding="utf-8"))
    trace = attempts[-1].notes[TRACE_NOTE]
    assert trace["verdict"] == verdict, "the grader's verdict is the runtime's"


def test_a_two_turn_fixture_seals_once_on_its_last_turn(monkeypatch, world, report):
    attempts, _, snapshot = _run_recorded("sequential", monkeypatch, world)

    assert len(snapshot["turns"]) == 2, "the fixture is a two-turn chain"
    [attempt] = attempts
    assert attempt.notes[garak.probes.midojo.CHAIN_NOTE] == {
        "chain": 0,
        "turn": 1,
        "turns": 2,
        "final": True,
    }, "the attempt the chain leaves is its last turn's"
