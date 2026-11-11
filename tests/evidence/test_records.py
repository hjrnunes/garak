"""Frozen evidence values round-trip into verifiable snapshot calls."""

import dataclasses
import hashlib
import json
import math
import sys

import pytest

pytest.importorskip("midojo_spec")
from midojo_spec import injection, recorder, snapshot

from garak.evidence.records import (
    BODY_FIELDS,
    DISPOSITIONS,
    PROVENANCES,
    REFUSED_DISPOSITIONS,
    CallBody,
    CallRecord,
    InvalidRequestError,
    RejectedError,
    freeze,
    thaw,
)


@pytest.mark.parametrize(
    "value",
    [
        None,
        "word",
        True,
        1,
        1.0,
        -0.0,
        [],
        {},
        {"b": [True, 1, 1.0, {"x": -0.0}], "a": None},
    ],
)
def test_round_trip(value):
    """Freeze and thaw retain exact value types, nested order, and float signs."""
    rebuilt = thaw(freeze(value))
    assert snapshot.dumps(snapshot.canonical(rebuilt)) == snapshot.dumps(
        snapshot.canonical(value)
    ), "frozen data changed types or order"
    if type(value) is dict:
        assert list(rebuilt) == list(value), "mapping order changed"


@pytest.mark.parametrize(
    ("value", "location"),
    [
        ({"a": [0, 1, (2,)]}, "a[2]"),
        ({"a": {1}}, "a"),
        ({"a": b"binary"}, "a"),
        ({"a": math.nan}, "a"),
        ({"a": math.inf}, "a"),
        ({1: "bad"}, "[1]"),
        ({"a": type("SubStr", (str,), {})("a")}, "a"),
        ({"a": type("SubDict", (dict,), {})()}, "a"),
        ({"a": type("SubList", (list,), {})()}, "a"),
        ({"a": type("SubInt", (int,), {})(1)}, "a"),
    ],
)
def test_unsupported(value, location):
    """Refusals name the path rather than silently coercing a value."""
    with pytest.raises(RejectedError, match="unsupported_value") as caught:
        freeze(value)
    assert location in caught.value.detail, "refusal lost the value location"


def test_int_the_digest_cannot_encode():
    """An integer past the interpreter's digit limit is refused, not leaked as ValueError."""
    previous = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(4300)
    try:
        with pytest.raises(RejectedError, match="unsupported_value") as caught:
            CallBody.build(function="f", args={"n": [10**5000]}, result="r")
        assert "args.n[0]" in caught.value.detail, "refusal lost the value location"
        assert freeze(10**1000) == 10**1000, "an encodable wide integer was refused"
    finally:
        sys.set_int_max_str_digits(previous)


def test_bounded_depth_and_cycle():
    """Depth 100 succeeds; depth 101 and a cycle refuse without recursion overflow."""
    value = 1
    for _ in range(100):
        value = [value]
    assert thaw(freeze(value)) == value, "depth limit excluded depth 100"
    with pytest.raises(RejectedError, match="unsupported_value"):
        freeze([value])
    cycle = []
    cycle.append(cycle)
    with pytest.raises(RejectedError, match="unsupported_value"):
        freeze(cycle)


def record(**changes):
    """Make a record with the same frozen fields the ledger installs."""
    values = dict(function="tool", args={"nested": [1]}, result="ok")
    values.update(changes)
    body = CallBody.build(**values)
    return CallRecord(
        **{name: getattr(body, name) for name in (*BODY_FIELDS, "disposition")},
        call_id="call-1",
        complete=True,
        provenance="host",
        timestamp="2026-01-01T00:00:00+00:00",
        turn=None,
    )


def test_record_is_immutable_and_json_is_detached():
    """Neither attributes nor nested data can mutate a recorded call."""
    call = record(structured_content={"items": [{"a": 1}]})
    with pytest.raises(dataclasses.FrozenInstanceError):
        call.result = "changed"
    with pytest.raises(TypeError):
        call.args["nested"] = []
    assert type(call.args["nested"]) is tuple, "nested list remains mutable"
    data = call.to_json()
    assert list(data) == [
        "call_id",
        "function",
        "args",
        "result",
        "error",
        "blocked",
        "disposition",
        "structured_content",
        "injection_outcomes",
        "complete",
        "provenance",
        "timestamp",
        "turn",
        "pre_environment",
        "post_environment",
    ], "call keys differ from the sealed reader's schema"
    data["args"]["nested"].append(2)
    data["structured_content"]["items"][0]["a"] = 2
    assert call.to_json()["args"] == {"nested": [1]}, "JSON mutated the call"
    assert call.to_json()["structured_content"] == {
        "items": [{"a": 1}]
    }, "JSON mutated details"


def test_snapshot_digest_and_verification():
    """The standard snapshot codec verifies calls exported from frozen records."""
    payload = {
        field: (
            [record().to_json()]
            if field == "function_calls"
            else (
                []
                if field in ("turns", "capture_failures")
                else (
                    {}
                    if field
                    in (
                        "pre_environment",
                        "environment",
                        "observations",
                        "evidence_status",
                    )
                    else (
                        None
                        if field in ("run_id", "user_task_id", "injection_task_id")
                        else "test"
                    )
                )
            )
        )
        for field in snapshot.CONTENT_FIELDS
    }
    payload["snapshot_id"] = snapshot.snapshot_digest(payload)
    assert snapshot.verified(payload) is payload, "snapshot digest did not verify"


@pytest.mark.parametrize(
    "outcome",
    [
        {
            "instruction_index": 0,
            "slot_key": None,
            "occurrence": 1,
            "status": "applied",
            "detail": "done",
            "selected": True,
            "transformation": "identity",
            "emitted": ["content_block"],
        },
        {
            "instruction_index": 0,
            "slot_key": "a",
            "occurrence": 0,
            "status": "unmatched",
            "detail": "no match",
        },
    ],
)
def test_valid_outcomes(outcome):
    """Keep valid outcomes without inserting absent facts."""
    assert record(injection_outcomes=[outcome]).to_json()["injection_outcomes"] == [
        outcome
    ], "outcome changed during freezing"


@pytest.mark.parametrize(
    "changes",
    [
        {"occurrence": 0},
        {
            "occurrence": 0,
            "status": "unmatched",
            "selected": True,
            "transformation": None,
            "emitted": [],
        },
        {"selected": False},
        {"instruction_index": True},
        {"occurrence": -1},
        {"status": "else"},
        {"detail": 1},
        {"unknown": 2},
        {"emitted": ["unknown"], "selected": False, "transformation": None},
        {"payload_digest": "wrong"},
        {"slot": {}},
    ],
)
def test_invalid_outcomes(changes):
    """Reject malformed outcomes before they become sealed evidence."""
    outcome = {
        "instruction_index": 0,
        "slot_key": None,
        "occurrence": 1,
        "status": "applied",
        "detail": "done",
    }
    outcome.update(changes)
    with pytest.raises(InvalidRequestError):
        record(injection_outcomes=[outcome])


PLACEMENT = {
    "instruction_index": 0,
    "slot_key": "slot",
    "occurrence": 1,
    "status": "applied",
    "detail": "placed",
    "payload_digest": "sha256:" + "0" * 64,
    "placed_digest": "sha256:" + "1" * 64,
}


def slot_data(**target):
    """Return the shared codec's own data for a slot on the given target."""
    return injection.InjectionSlot(injection.InjectionTarget(**target)).to_data()


def target_id(target):
    """Hash a target as given, as a reader that does not decode it would."""
    raw = json.dumps(target, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(raw.encode()).hexdigest()


def test_placed_slot_is_the_shared_slot():
    """Keep a slot exactly as the shared codec encodes it."""
    slot = slot_data(tool="get_weather")
    kept = record(injection_outcomes=[{**PLACEMENT, "slot": slot}])
    assert kept.to_json()["injection_outcomes"][0]["slot"] == slot, "slot changed"


def test_cyclic_slot_is_unsupported_json():
    """Refuse cyclic slot data as unsupported JSON, naming where the cycle is."""
    slot = slot_data(tool="example")
    slot["target"]["loop"] = slot["target"]
    with pytest.raises(RejectedError, match="unsupported_value") as caught:
        record(injection_outcomes=[{**PLACEMENT, "slot": slot}])
    assert (
        "injection_outcomes[0].slot.target.loop" in caught.value.detail
    ), "cycle location was lost"


@pytest.mark.parametrize(
    "slot",
    [
        {
            "id": target_id({"not_a_target_field": 1}),
            "target": {"not_a_target_field": 1},
            "mode": "append",
        },
        {
            "id": target_id({"tool": "get_weather"}),
            "target": {"tool": "get_weather"},
            "mode": "append",
        },
        {**slot_data(tool="get_weather"), "id": "sha256:" + "0" * 64},
        {**slot_data(tool="get_weather"), "mode": "prepend"},
        {**slot_data(tool="get_weather"), "extra": 1},
    ],
)
def test_invalid_slot(slot):
    """Refuse a slot the shared codec cannot decode or would encode differently."""
    with pytest.raises(InvalidRequestError, match="slot is invalid"):
        record(injection_outcomes=[{**PLACEMENT, "slot": slot}])


def test_shared_vocabulary_and_rejection_kinds():
    """Collector vocabulary agrees with the recorder and verifying consumer."""
    assert DISPOSITIONS == set(recorder.Disposition.__args__), "dispositions diverged"
    assert REFUSED_DISPOSITIONS == recorder.REFUSED, "refusals diverged"
    assert PROVENANCES == {"host", "interceptor", "agent"}, "provenances diverged"
    assert {status.value for status in injection.OutcomeStatus} == {
        "applied",
        "partial",
        "fallback",
        "unmatched",
        "no_result",
        "too_large",
        "skipped_occurrence",
    }, "statuses diverged"
    for kind in ("call_conflict", "invalid_disposition", "unsupported_value"):
        assert kind in {
            entry.value for entry in recorder.RejectionKind
        }, "unknown rejection kind"
