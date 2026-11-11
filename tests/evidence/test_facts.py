"""Late facts, capture failures, turns and observations."""

import pytest

pytest.importorskip("midojo_spec")

from garak.evidence.facts import (
    AGENT_OUTPUT,
    CaptureFailure,
    LateFact,
    LateFactError,
    Observation,
    Turn,
    cut_reason,
    json_payload,
)
from garak.evidence.records import RejectedError, thaw


def fact(**changes):
    """A minimal late fact with caller-supplied changes."""
    values = dict(
        event="finish_call",
        stage="sealed",
        by="interceptor",
        at="2026-01-01T00:00:00+00:00",
        evidence=("function_calls",),
        payload={"call_id": "one", "result": "late"},
        call_id="one",
    )
    values.update(changes)
    return LateFact(**values)


def test_late_fact_identity_ignores_stage_and_time():
    """The same write kept twice is one fact, whatever its stage or arrival."""
    first = fact()
    again = fact(stage="closing", at="2027-01-01T00:00:00+00:00")
    other_call = fact(payload={"call_id": "two", "result": "late"})
    other_writer = fact(by="host")
    assert first.identity() == again.identity(), "stage or time split one write"
    assert first.identity() != other_call.identity(), "another call is another write"
    assert (
        first.identity() != other_writer.identity()
    ), "another writer is another write"


def test_late_fact_payload_is_frozen_json():
    """A payload is frozen, and data JSON cannot hold is refused."""
    payload = json_payload({"nested": {"list": [1, True, None]}})
    with pytest.raises(RejectedError, match="unsupported_value"):
        json_payload({"bad": {1, 2}})
    assert thaw(payload)["nested"]["list"] == [1, True, None], "payload lost its values"


def test_late_fact_error_is_a_late_marked_rejection():
    """A late fact refusal names its kind and carries the late marker."""
    error = LateFactError("sealed", "kept as a late fact")
    assert isinstance(error, RejectedError), "a late fact refusal is a rejection"
    assert error.kind == "sealed", "wrong refusal kind"
    assert error.late_fact is True, "the late marker is missing"
    with pytest.raises(ValueError):
        LateFactError("call_conflict", "not a late kind")


def test_capture_failure_identity_and_reason_cut():
    """Identity is call, function and phase; the reason is cut at the shared length."""
    failure = CaptureFailure(
        call_id="one",
        function="tool",
        phase="start",
        reason="r" * 5000,
        stage="open",
        by="interceptor",
        at="2026-01-01T00:00:00+00:00",
    )
    assert len(failure.reason) == 2048, "reason was not cut at the shared length"
    assert len(cut_reason("r" * 5000)) == 2048, "cut helper ignored the length"
    same = CaptureFailure(
        call_id="one",
        function="tool",
        phase="start",
        reason="another attempt's error",
        stage="sealed",
        by="host",
        at="2027-01-01T00:00:00+00:00",
    )
    assert failure.identity() == same.identity(), "identity included mutable fields"
    assert failure.to_json() == {
        "call_id": "one",
        "function": "tool",
        "phase": "start",
        "reason": "r" * 2048,
        "stage": "open",
        "by": "interceptor",
        "at": "2026-01-01T00:00:00+00:00",
    }, "failure JSON differs from the sealed reader's schema"
    with pytest.raises(ValueError):
        CaptureFailure(
            call_id="one",
            function="tool",
            phase="deliver",
            reason="r",
            stage="open",
            by="host",
            at="2026-01-01T00:00:00+00:00",
        )


def test_turn_json_shape():
    """A turn exports the sealed snapshot's field order."""
    turn = Turn(
        index=0,
        message="hello",
        reply="hi",
        started_at="2026-01-01T00:00:00+00:00",
        ended_at="2026-01-01T00:00:01+00:00",
    )
    assert list(turn.to_json()) == [
        "index",
        "message",
        "reply",
        "started_at",
        "ended_at",
    ], "turn keys differ from the sealed reader's schema"


def test_observation_flattens_evidence_keys():
    """Each part's status is keyed by source and part."""
    observation = Observation(
        source="runtime",
        observed={"findings": []},
        evidence={"findings": "complete", "workdir": "incomplete"},
    )
    assert observation.flattened() == {
        "runtime.findings": "complete",
        "runtime.workdir": "incomplete",
    }, "evidence keys were not flattened by source"


def test_observation_evidence_cannot_change_after_it_is_recorded():
    """The statuses an observation holds are fixed, whoever kept a reference."""
    statuses = {"findings": "complete"}
    observation = Observation(source="runtime", observed={}, evidence=statuses)
    statuses["findings"] = "failed"
    assert observation.evidence["findings"] == "complete", "the caller's dict leaked in"
    with pytest.raises(TypeError):
        observation.evidence["findings"] = "failed"


def test_late_fact_evidence_keys():
    """A fact's keys are exactly the evidence it marks incomplete."""
    assert fact(evidence=("turns", AGENT_OUTPUT)).evidence_keys() == frozenset(
        {"turns", "agent_output"}
    ), "evidence keys lost a key"
    assert fact(evidence=[]).evidence_keys() == frozenset(), "empty keys grew"
