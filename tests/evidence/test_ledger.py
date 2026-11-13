"""Idempotent intake and channel-bound provenance."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("midojo_spec")

from garak.evidence.ledger import CallLedger
from garak.evidence.records import (
    CallBody,
    InvalidRequestError,
    RejectedError,
)


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


def test_clock_that_reads_the_ledger_does_not_stop_intake():
    """The ledger stamps under its lock, so a stamp that reads it must not wait on itself."""
    seen = {}

    def clock():
        if "ledger" in seen:
            len(seen["ledger"])
        return "t"

    ledger = seen["ledger"] = CallLedger(clock=clock)
    writer = threading.Thread(
        target=lambda: intake(ledger.channel("host"), call_id="one"), daemon=True
    )
    writer.start()
    writer.join(2.0)
    assert not writer.is_alive(), "intake waited on its own lock"
    assert len(ledger) == 1, "the call did not reach the ledger"


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


def started(ledger, call_id="one", **changes):
    """Start a call on the ledger with caller-supplied changes."""
    values = dict(function="tool", args={"a": 1}, disposition=None)
    values.update(changes)
    body = CallBody.build_start(**values)
    return ledger.start("host", call_id, body, body.encoded(("function", "args")))


def test_start_appends_an_unfinished_record():
    """A start holds no result and keeps the channel's provenance and turn."""
    ledger = CallLedger(
        clock=lambda: "2026-01-01T00:00:00+00:00", current_turn=lambda: 2
    )
    receipt = started(ledger)
    record = receipt.record
    assert (receipt.inserted, record.complete, record.result) == (
        True,
        False,
        None,
    ), "a start completed the call"
    assert (record.provenance, record.timestamp, record.turn) == (
        "host",
        "2026-01-01T00:00:00+00:00",
        2,
    ), "a start lost its metadata"
    assert ledger.unfinished() == (record,), "the start is not unfinished"


def test_unfinished_calls_come_back_in_start_order():
    """The unfinished list follows start order, whatever order finishes arrive in."""
    ledger = CallLedger()
    ids = [f"call-{index}" for index in range(12)]
    for call_id in ids:
        started(ledger, call_id)
    body = CallBody.build(function="tool", args={"a": 1}, result="ok")
    done = ("call-3", "call-0", "call-7")
    for call_id in done:
        ledger.complete(call_id, body, body.encoded())
    assert [record.call_id for record in ledger.unfinished()] == [
        call_id for call_id in ids if call_id not in done
    ], "unfinished calls left start order"


def test_start_needs_a_call_id():
    """A start without an ID cannot be paired with its finish, so it is refused."""
    ledger = CallLedger(new_call_id=lambda: "assigned")
    body = CallBody.build_start(function="tool", args={"a": 1})
    with pytest.raises(InvalidRequestError):
        ledger.start("host", None, body, body.encoded(("function", "args")))
    assert len(ledger) == 0, "a start without an ID was recorded"


def test_completion_happens_in_place():
    """A finish completes the record at its position, keeping its metadata."""
    ledger = CallLedger(
        clock=lambda: "2026-01-01T00:00:00+00:00", current_turn=lambda: 3
    )
    first = started(ledger, "one")
    other = started(ledger, "two")
    body = CallBody.build(function="tool", args={"a": 1}, result="ok")
    receipt = ledger.complete("one", body, body.encoded())
    finished = receipt.record
    assert finished.complete and finished.result == "ok", "the call did not complete"
    assert finished.call_id == "one" and finished.turn == 3, "identity changed"
    assert finished.provenance == "host", "the finish changed the provenance"
    assert finished.timestamp == first.record.timestamp, "the finish changed the time"
    assert ledger.calls() == (finished, other.record), "the record left its position"
    assert ledger.unfinished() == (other.record,), "a finished call stayed unfinished"


def test_completion_keeps_a_start_disposition_when_finish_omits_one():
    """A finish that names no disposition keeps the start's."""
    ledger = CallLedger()
    started(ledger, "one", disposition="forwarded")
    body = CallBody.build(function="tool", args={"a": 1}, result="ok")
    finished = ledger.complete("one", body, body.encoded()).record
    assert finished.disposition == "forwarded", "the start's disposition was lost"
    body = CallBody.build(
        function="tool", args={"a": 1}, result="ok", disposition="forwarded"
    )
    finished = ledger.complete("one", body, body.encoded()).record
    assert finished.disposition == "forwarded", "a named one replaced the start's"
