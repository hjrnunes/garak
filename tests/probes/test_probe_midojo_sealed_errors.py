# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression for errors retained by the sealed per-call trace.

Each saved document is the sealed snapshot of a real two-turn chain whose agent
made calls the service answered with an error. The test replays the chain's
turns and calls through the probe and the run coordinator, so a call error has
to survive the collector, the seal and the trace to reach the final attempt.
"""

import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("midojo_spec")

from garak.evidence.records import PLACEMENT_FIELDS  # noqa: E402
from garak.probes.midojo import TRACE_NOTE  # noqa: E402

from injection_chain import (  # noqa: E402
    ScriptedTarget,
    build_probe,
    entry,
    report,  # noqa: F401
    world,  # noqa: F401
)
from sealed_replay import call_fields  # noqa: E402

ASSETS = Path(__file__).resolve().parents[1] / "_assets" / "probes" / "sealed_errors"
MANIFEST = json.loads((ASSETS / "MANIFEST.json").read_text(encoding="utf-8"))


def _as_recorded(call):
    """A saved call's members as the service records them.

    The saved seals come from an older harness, which wrote an outcome that
    placed nothing with empty placement members; the collector takes those
    members only from an outcome that placed a payload.
    """
    fields = call_fields(call)
    fields["injection_outcomes"] = [
        {
            name: value
            for name, value in outcome.items()
            if value is not None or name not in PLACEMENT_FIELDS
        }
        for outcome in call["injection_outcomes"]
    ]
    return fields


@pytest.mark.parametrize("name", sorted(MANIFEST))
def test_saved_chain_keeps_every_call_error_on_the_final_attempt(
    tmp_path, world, report, name
):
    """The sealed snapshot stays unedited and each error survives the trace in order."""
    raw = (ASSETS / name).read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest() == MANIFEST[name]["sha256"]
    ), "seal is unedited"
    saved = json.loads(raw)
    expected = [call["error"] for call in saved["function_calls"] if call["error"]]
    assert len(expected) == MANIFEST[name]["error_count"], "all errors are expected"
    turns = sorted(saved["turns"], key=lambda turn: turn["index"])
    script = [
        [
            _as_recorded(call)
            for call in saved["function_calls"]
            if call["turn"] == turn["index"]
        ]
        for turn in turns
    ]
    probe = build_probe(
        tmp_path,
        entry(*(turn["message"] for turn in turns), instructions=[], condition=None),
    )
    target = ScriptedTarget(world, script, replies=[turn["reply"] for turn in turns])

    (attempt,) = probe.probe(target)

    trace = attempt.notes[TRACE_NOTE]
    assert trace["error"] is None, "the chain ran both turns"
    assert trace["snapshot"]["agent_output"] == saved["agent_output"]
    assert [
        call["error"] for call in trace["tool_calls"] if call["error"]
    ] == expected, "the trace preserves each sealed error in order"
    assert [
        call["error"]
        for call in attempt.outputs[0].notes["tool_calls"]
        if call["error"]
    ] == expected, "the final attempt carries the same errors for its detectors"
