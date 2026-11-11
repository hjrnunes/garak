"""Idempotent intake and channel-bound provenance."""

from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("midojo_spec")

from garak.evidence.ledger import CallLedger
from garak.evidence.records import InvalidRequestError, RejectedError


def intake(channel, **changes):
    """Submit a minimal complete call with caller-supplied changes."""
    body = {"function": "tool", "args": {"a": 1}, "result": "ok"}
    body.update(changes)
    return channel.record_call(**body)


def test_identical_replay_keeps_first_record():
    """Repeated IDs retain the first writer's metadata and object identity."""
    ledger = CallLedger(clock=lambda: "first")
    original = intake(ledger.channel("host"), call_id="one")
    replay = intake(ledger.channel("agent"), call_id="one")
    assert original.inserted and not replay.inserted, "replay inserted a call"
    assert replay.record is original.record, "replay changed record identity"
    assert (replay.record.provenance, replay.record.timestamp) == (
        "host",
        "first",
    ), "replay changed metadata"
    assert len(ledger) == 1, "replay changed ledger size"


@pytest.mark.parametrize(
    "changes",
    [
        {"function": "other"},
        {"args": {"a": 2}},
        {"result": "other"},
        {"error": "failure"},
        {"blocked": True},
        {"structured_content": {}},
        {"injection_outcomes": []},
    ],
)
def test_conflicting_body_is_atomic(changes):
    """Every body field is compared before a duplicate ID is admitted."""
    ledger = CallLedger()
    channel = ledger.channel("host")
    original = intake(channel, call_id="one")
    before = ledger.calls()
    with pytest.raises(RejectedError) as caught:
        intake(channel, call_id="one", **changes)
    assert caught.value.kind == "call_conflict", "wrong conflict kind"
    assert ledger.calls() == before == (original.record,), "conflict changed calls"


@pytest.mark.parametrize(
    ("original", "other"),
    [(True, 1), (1, 1.0), (0.0, -0.0)],
)
def test_type_for_type_comparison(original, other):
    """Numerical coercion and floating-point equality cannot hide a conflict."""
    ledger = CallLedger()
    channel = ledger.channel("host")
    intake(channel, call_id="one", args={"value": original})
    with pytest.raises(RejectedError, match="call_conflict"):
        intake(channel, call_id="one", args={"value": other})
    assert len(ledger) == 1, "numeric conflict inserted a call"


def test_mapping_order_is_ignored():
    """Object key order cannot create a conflict between otherwise equal calls."""
    channel = CallLedger().channel("host")
    first = intake(channel, call_id="one", args={"x": 1, "y": 2})
    replay = intake(channel, call_id="one", args={"y": 2, "x": 1})
    assert (
        replay.record is first.record and not replay.inserted
    ), "order caused conflict"


def test_disposition_agreement():
    """A replay may omit the disposition but never name one the record lacks."""
    ledger = CallLedger()
    channel = ledger.channel("host")
    unnamed = intake(channel, call_id="one")
    for named in ("forwarded", "simulated"):
        with pytest.raises(RejectedError) as caught:
            intake(channel, call_id="one", disposition=named)
        assert caught.value.kind == "call_conflict", "replay added a disposition"
    assert unnamed.record.disposition is None, "first record gained a disposition"
    named = intake(channel, call_id="two", disposition="forwarded")
    assert (
        intake(channel, call_id="two", disposition="forwarded").record is named.record
    ), "same disposition did not replay"
    assert (
        intake(channel, call_id="two").record is named.record
    ), "unnamed replay changed record"
    with pytest.raises(RejectedError) as caught:
        intake(channel, call_id="two", disposition="simulated")
    assert caught.value.kind == "call_conflict", "different named disposition accepted"
    with pytest.raises(RejectedError) as caught:
        intake(channel, call_id="three", disposition="denied")
    assert caught.value.kind == "invalid_disposition", "blocked mismatch accepted"
    assert len(ledger) == 2, "invalid disposition inserted a call"


@pytest.mark.parametrize("provenance", ["host", "interceptor", "agent"])
def test_provenance_is_channel_bound(provenance):
    """Only the channel chooses the source of a recorded call."""
    ledger = CallLedger()
    record = intake(ledger.channel(provenance)).record
    assert record.to_json()["provenance"] == provenance, "channel provenance lost"
    with pytest.raises(TypeError):
        intake(ledger.channel(provenance), provenance="host")
    with pytest.raises(ValueError):
        ledger.channel("unknown")


def test_generated_id_and_injected_metadata():
    """ID generation happens once per new call; supplied metadata is retained."""
    ids = iter(("first", "second"))
    ledger = CallLedger(
        clock=lambda: "2026-01-01T00:00:00+00:00",
        new_call_id=lambda: next(ids),
        current_turn=lambda: 4,
    )
    channel = ledger.channel("host")
    first, second = intake(channel), intake(channel)
    assert (first.record.call_id, second.record.call_id) == (
        "first",
        "second",
    ), "generated IDs did not advance"
    assert ledger.get(first.record.call_id) is first.record, "generated ID is unstable"
    assert first.record.timestamp == "2026-01-01T00:00:00+00:00", "clock was ignored"
    assert first.record.turn == 4, "current turn was ignored"
    assert len(ledger) == 2, "missing IDs deduplicated calls"


@pytest.mark.parametrize("bad", ["", "a" * 257, 1, True, type("Sub", (str,), {})("id")])
def test_invalid_call_id(bad):
    """Invalid caller IDs refuse the request without changing the ledger."""
    ledger = CallLedger()
    with pytest.raises(InvalidRequestError):
        intake(ledger.channel("host"), call_id=bad)
    assert not ledger.calls(), "bad ID inserted a call"


def test_insertion_order_and_rejection_atomicity():
    """Snapshots preserve insertion order through conflicts and bad data."""
    ledger = CallLedger()
    channel = ledger.channel("host")
    for index in range(3):
        intake(channel, call_id=str(index))
    before = ledger.calls()
    with pytest.raises(RejectedError, match="unsupported_value"):
        intake(channel, call_id="four", args={"bad": (1,)})
    with pytest.raises(RejectedError, match="call_conflict"):
        intake(channel, call_id="1", result="different")
    with pytest.raises(RejectedError, match="invalid_disposition"):
        intake(channel, call_id="four", disposition="blocked")
    assert ledger.calls() == before, "rejection changed the snapshot"
    assert tuple(record.call_id for record in before) == (
        "0",
        "1",
        "2",
    ), "insertion order changed"


def test_concurrent_replay_and_distinct_ids():
    """Concurrent writers linearise at one insert per ID."""
    ledger = CallLedger()
    channel = ledger.channel("interceptor")
    with ThreadPoolExecutor(max_workers=16) as pool:
        replays = list(pool.map(lambda _: intake(channel, call_id="one"), range(16)))
        distinct = list(
            pool.map(lambda i: intake(channel, call_id=f"other-{i}"), range(16))
        )
    assert sum(receipt.inserted for receipt in replays) == 1, "replay inserted twice"
    assert all(
        receipt.record is replays[0].record for receipt in replays
    ), "replays returned different records"
    assert all(receipt.inserted for receipt in distinct), "distinct ID was dropped"
    assert len(ledger) == 17, "concurrent calls were lost"
