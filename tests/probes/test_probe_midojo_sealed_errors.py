# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression for errors retained by the sealed per-call trace."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from garak.attempt import Attempt, Conversation, Message, Turn
from garak.probes.midojo import InjectionSpecProbe, TRACE_NOTE, _midojo
from tests_runtime_midojo import require_runtime

require_runtime()

from midojo.control_plane.client import snapshot_from_response  # noqa: E402
from midojo.control_plane.models import SnapshotResponse  # noqa: E402
from midojo.types import Environment  # noqa: E402
from midojo_spec.snapshot import verified  # noqa: E402

ASSETS = Path(__file__).resolve().parents[1] / "_assets" / "probes" / "sealed_errors"
MANIFEST = json.loads((ASSETS / "MANIFEST.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", sorted(MANIFEST))
def test_saved_seal_keeps_every_call_error_on_the_final_attempt(name):
    """The sealed snapshot remains byte-identical and each error survives the trace."""
    raw = (ASSETS / name).read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest() == MANIFEST[name]["sha256"]
    ), "seal is unedited"
    document = json.loads(raw)
    verified(document)
    snapshot = snapshot_from_response(
        SnapshotResponse.model_validate(document), Environment
    )
    assert (
        snapshot.snapshot_id == document["snapshot_id"]
    ), "harness reads original seal"
    expected = [call["error"] for call in document["function_calls"] if call["error"]]
    assert len(expected) == MANIFEST[name]["error_count"], "all errors are represented"

    probe = object.__new__(InjectionSpecProbe)
    probe._midojo = _midojo()
    verdict = SimpleNamespace(
        grade=SimpleNamespace(injection_delivered=True),
        to_document=lambda: {"grade": {"injection_delivered": True}},
    )
    handle = SimpleNamespace(
        evaluation_id=snapshot.evaluation_id, grade=lambda: verdict
    )
    trace = probe._trace(handle, snapshot, sealed="final")
    attempt = Attempt(prompt=Conversation([Turn("user", Message("test"))]))
    attempt.outputs = [Message("reply", notes={"response_status": "completed"})]
    probe._attach_trace(attempt, trace)

    assert [
        call["error"] for call in trace["tool_calls"] if call["error"]
    ] == expected, "the trace preserves each sealed error in order"
    assert [
        call["error"]
        for call in attempt.notes[TRACE_NOTE]["tool_calls"]
        if call["error"]
    ] == expected, "the final attempt carries the same errors"
