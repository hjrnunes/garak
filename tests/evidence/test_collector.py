"""The staged collector: admission, completion, late facts, drain and the seal."""

import copy
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("midojo_spec")
from midojo_spec import snapshot as shared

from garak.evidence.collector import Collector, CollectorChannel, Stage
from garak.evidence.facts import LateFactError
from garak.evidence.links import CaptureLink
from garak.evidence.records import InvalidRequestError, RejectedError


class Clock:
    """A fake monotonic clock the tests advance by hand."""

    def __init__(self, start=0.0):
        self.now = start

    def monotonic(self):
        return self.now

    def clock(self):
        return f"2026-01-01T{int(self.now // 60):02d}:{int(self.now % 60):02d}+00:00"

    def advance(self, seconds):
        self.now += seconds


def collector(**changes):
    """A collector on a fake clock with caller-supplied changes."""
    time_source = Clock()
    values = dict(
        evaluation_id="ev-1",
        monotonic=time_source.monotonic,
        clock=time_source.clock,
    )
    values.update(changes)
    return Collector(**values), time_source


def sealed_collector(**changes):
    """A collector already drained and sealed with a plain output."""
    unit, clock = collector(**changes)
    unit.begin_closure(0.0)
    clock.advance(1.0)
    unit.drain()
    unit.seal("done")
    return unit, clock


def test_start_then_finish_keeps_one_record():
    """A finish completes the start in place, keeping its metadata and position."""
    unit, _ = collector()
    channel = unit.channel("interceptor")
    started = channel.start_call("one", function="tool", args={"n": 1})
    finished = channel.finish_call(
        "one", function="tool", args={"n": 1}, result="ok", disposition="forwarded"
    )
    assert unit.stage is Stage.OPEN, "intake changed the stage"
    records = _records(unit)
    assert len(records) == 1, "start and finish made two records"
    assert records[0] is finished.record, "the completed record left its position"
    assert finished.record.complete and finished.record.result == "ok", "no completion"
    assert finished.record.provenance == "interceptor", "finish changed the provenance"
    assert finished.record.timestamp == started.record.timestamp, "time changed"
    assert finished.record.turn == started.record.turn, "turn changed"
    assert finished.record.disposition == "forwarded", "finish lost its disposition"


def _records(unit):
    return [call for call in _ledger(unit).calls()]


def _ledger(unit):
    return unit._ledger


def test_start_replay_at_every_stage():
    """An identical start replays while open, closing and after the seal."""
    unit, clock = collector()
    channel = unit.channel("host")
    first = channel.start_call("one", function="tool", args={"n": 1})
    again = channel.start_call("one", function="tool", args={"n": 1})
    assert (again.inserted, again.record is first.record) == (
        False,
        True,
    ), "an open replay inserted"
    unit.begin_closure(5.0)
    again = channel.start_call("one", function="tool", args={"n": 1})
    assert not again.inserted, "a closing replay inserted"
    clock.advance(6.0)
    unit.drain()
    unit.seal("done")
    again = channel.start_call("one", function="tool", args={"n": 1})
    assert not again.inserted, "a sealed replay inserted"
    assert not unit.late_facts, "a replay was kept as a late fact"


def test_start_conflict_is_rejected_at_every_stage():
    """Another start body under the same ID conflicts, and after the seal is late."""
    unit, clock = collector()
    channel = unit.channel("host")
    channel.start_call("one", function="tool", args={"n": 1})
    for change in ({"args": {"n": 2}}, {"function": "other"}):
        with pytest.raises(RejectedError) as caught:
            channel.start_call(
                "one",
                function=change.get("function", "tool"),
                args=change.get("args", {"n": 1}),
            )
        assert caught.value.kind == "call_conflict", "conflict not rejected while open"
    unit.begin_closure(5.0)
    with pytest.raises(RejectedError) as caught:
        channel.start_call("one", function="tool", args={"n": 2})
    assert caught.value.kind == "call_conflict", "conflict not rejected while closing"
    clock.advance(6.0)
    unit.drain()
    unit.seal("done")
    with pytest.raises(LateFactError) as caught:
        channel.start_call("one", function="tool", args={"n": 3})
    assert caught.value.late_fact, "a sealed conflict was not marked late"


def test_start_disposition_rules():
    """A start may name a disposition a stored record lacks only when it matches."""
    unit, _ = collector()
    channel = unit.channel("host")
    channel.start_call("one", function="tool", args={"n": 1})
    with pytest.raises(RejectedError) as caught:
        channel.start_call(
            "one", function="tool", args={"n": 1}, disposition="forwarded"
        )
    assert caught.value.kind == "call_conflict", "a replay added a disposition"
    named = channel.start_call(
        "two", function="tool", args={"n": 1}, disposition="forwarded"
    )
    replay = channel.start_call(
        "two", function="tool", args={"n": 1}, disposition="forwarded"
    )
    assert replay.record is named.record and not replay.inserted, "named replay failed"
    with pytest.raises(RejectedError) as caught:
        channel.start_call(
            "two", function="tool", args={"n": 1}, disposition="simulated"
        )
    assert caught.value.kind == "call_conflict", "another disposition accepted"


@pytest.mark.parametrize("disposition", ["denied", "blocked"])
def test_refused_disposition_start_is_rejected(disposition):
    """A denied or blocked call never runs, so it is never started."""
    unit, _ = collector()
    channel = unit.channel("host")
    with pytest.raises(RejectedError) as caught:
        channel.start_call(
            "one", function="tool", args={"n": 1}, disposition=disposition
        )
    assert caught.value.kind == "invalid_disposition", "refused start accepted"
    assert not _records(unit), "refused start inserted a record"


def test_finish_completion_rules():
    """A finish completes an unfinished record under the MiDojo disposition rule."""
    unit, _ = collector()
    channel = unit.channel("agent")
    channel.start_call("one", function="tool", args={"n": 1})
    finished = channel.finish_call(
        "one", function="tool", args={"n": 1}, result="ok", disposition="forwarded"
    )
    assert finished.record.complete, "named finish did not complete"
    channel.start_call("two", function="tool", args={"n": 2})
    omitted = channel.finish_call("two", function="tool", args={"n": 2}, result="ok")
    assert omitted.record.disposition is None, "an omitted disposition invented one"
    channel.start_call("three", function="tool", args={"n": 3}, disposition="forwarded")
    keeps = channel.finish_call("three", function="tool", args={"n": 3}, result="ok")
    assert (
        keeps.record.disposition == "forwarded"
    ), "finish lost the start's disposition"
    with pytest.raises(RejectedError) as caught:
        channel.finish_call("three", function="tool", args={"n": 3}, result="again")
    assert caught.value.kind == "call_conflict", "a second finish was accepted"


@pytest.mark.parametrize(
    ("start_disposition", "finish_disposition", "blocked", "kind"),
    [
        ("forwarded", "simulated", False, "call_conflict"),
        ("forwarded", "forwarded", False, None),
        (None, "forwarded", False, None),
        (None, None, False, None),
    ],
)
def test_start_finish_disposition_matrix(
    start_disposition, finish_disposition, blocked, kind
):
    """Named dispositions must agree; an omitted one inherits the start's."""
    unit, _ = collector()
    channel = unit.channel("host")
    channel.start_call(
        "one", function="tool", args={"n": 1}, disposition=start_disposition
    )
    if kind is None:
        record = channel.finish_call(
            "one",
            function="tool",
            args={"n": 1},
            result="ok",
            blocked=blocked,
            disposition=finish_disposition,
        ).record
        expected = finish_disposition or start_disposition
        assert record.disposition == expected, "completed record lost the disposition"
    else:
        with pytest.raises(RejectedError) as caught:
            channel.finish_call(
                "one",
                function="tool",
                args={"n": 1},
                result="ok",
                blocked=blocked,
                disposition=finish_disposition,
            )
        assert caught.value.kind == kind, "disagreeing dispositions accepted"


def test_finish_mismatch_and_replay():
    """A finish naming a complete record replays only an identical body."""
    unit, _ = collector()
    channel = unit.channel("host")
    whole = channel.record_call(function="tool", args={"n": 1}, result="ok")
    replay = channel.finish_call("missing", function="tool", args={"n": 1}, result="ok")
    assert replay.inserted, "a finish of an unknown ID did not record whole"
    again = channel.finish_call(
        whole.record.call_id, function="tool", args={"n": 1}, result="ok"
    )
    assert not again.inserted and again.record is whole.record, "finish replay failed"
    with pytest.raises(RejectedError) as caught:
        channel.finish_call(
            whole.record.call_id, function="tool", args={"n": 1}, result="other"
        )
    assert caught.value.kind == "call_conflict", "finish conflict accepted"


@pytest.mark.parametrize(
    ("function", "args"),
    [("other", {"n": 1}), ("tool", {"n": 2}), ("other", {"n": 2})],
)
def test_finish_must_name_the_function_and_args_it_started(function, args):
    """A finish that differs from its start conflicts and leaves the start open."""
    unit, _ = collector()
    channel = unit.channel("host")
    started = channel.start_call("one", function="tool", args={"n": 1})
    with pytest.raises(RejectedError) as caught:
        channel.finish_call("one", function=function, args=args, result="ok")
    assert caught.value.kind == "call_conflict", "a mismatched finish was accepted"
    records = _records(unit)
    assert records == [started.record], "the refused finish changed the ledger"
    assert not records[0].complete, "the refused finish completed the start"
    finished = channel.finish_call("one", function="tool", args={"n": 1}, result="ok")
    assert finished.record.complete, "a refused finish blocked the matching finish"


@pytest.mark.parametrize(
    "outcome",
    [{"error": "boom"}, {"blocked": True}, {"error": "refused", "blocked": True}],
)
def test_finish_fills_error_and_blocked(outcome):
    """A finish completes the started record with the error and blocked it names."""
    unit, _ = collector()
    channel = unit.channel("agent")
    started = channel.start_call("one", function="tool", args={"n": 1})
    assert (started.record.error, started.record.blocked) == (
        None,
        False,
    ), "a start carried an outcome"
    finished = channel.finish_call(
        "one", function="tool", args={"n": 1}, result="", **outcome
    )
    record = finished.record
    assert _records(unit) == [record], "the finish did not complete the start in place"
    assert record.complete and record.result == "", "the finish was not complete"
    assert (record.error, record.blocked) == (
        outcome.get("error"),
        outcome.get("blocked", False),
    ), "the finish lost its error or blocked"
    unit.close(0.0)
    sealed = unit.seal("done").to_json()["function_calls"]
    assert [(call["error"], call["blocked"]) for call in sealed] == [
        (outcome.get("error"), outcome.get("blocked", False))
    ], "the sealed call lost its error or blocked"


def test_whole_record_path_is_unchanged():
    """A finish without an ID records whole, with a fresh ID, as before."""
    unit, _ = collector()
    channel = unit.channel("host")
    first = channel.finish_call(function="tool", args={"n": 1}, result="ok")
    second = channel.record_call(function="tool", args={"n": 1}, result="ok")
    assert first.inserted and second.inserted, "whole records were deduplicated"
    assert first.record.call_id != second.record.call_id, "IDs collided"


def test_stage_machine_and_admission():
    """A new call only in open; an admitted one finishes in open or closing."""
    unit, clock = collector()
    channel = unit.channel("host")
    admitted = channel.start_call("one", function="tool", args={"n": 1})
    unit.begin_closure(5.0)
    assert unit.stage is Stage.CLOSING, "closure did not begin"
    unit.begin_closure(5.0)
    assert unit.stage is Stage.CLOSING, "a second closure changed the deadline"
    with pytest.raises(LateFactError) as caught:
        channel.start_call("two", function="tool", args={"n": 2})
    assert caught.value.kind == "closing", "a new call was admitted while closing"
    finished = channel.finish_call("one", function="tool", args={"n": 1}, result="ok")
    assert finished.record.complete, "an admitted call could not finish while closing"
    clock.advance(6.0)
    with pytest.raises(LateFactError) as caught:
        channel.finish_call("two", function="tool", args={"n": 2}, result="ok")
    assert caught.value.kind == "closing", "a new call was admitted past the deadline"
    with pytest.raises(RejectedError) as caught:
        channel.finish_call("one", function="tool", args={"n": 1}, result="again")
    assert (
        caught.value.kind == "call_conflict"
    ), "a conflicting finish past the deadline"
    assert unit.stage is Stage.COLLECTING, "deadline did not demote the stage"


def test_on_time_demotes_without_drain():
    """An event arriving past the deadline finds the collector collecting."""
    unit, clock = collector()
    channel = unit.channel("host")
    channel.start_call("one", function="tool", args={"n": 1})
    unit.begin_closure(1.0)
    clock.advance(2.0)
    with pytest.raises(LateFactError):
        channel.start_call("two", function="tool", args={"n": 2})
    assert unit.stage is Stage.COLLECTING, "the stage stayed closing past the deadline"


def test_seal_requires_collecting():
    """Sealing from open or closing is refused with a typed closure_pending."""
    unit, _ = collector()
    with pytest.raises(RejectedError) as caught:
        unit.seal("done")
    assert caught.value.kind == "closure_pending", "sealing from open was allowed"
    unit.begin_closure(1.0)
    with pytest.raises(RejectedError) as caught:
        unit.seal("done")
    assert caught.value.kind == "closure_pending", "sealing from closing was allowed"


def test_seal_is_idempotent_and_late_for_another_output():
    """Sealing again returns the same snapshot, or keeps another output late."""
    unit, _ = collector()
    unit.close(0.0)
    first = unit.seal("done")
    again = unit.seal("done")
    assert again is first, "an identical retry resealed"
    with pytest.raises(LateFactError) as caught:
        unit.seal("another")
    assert caught.value.kind == "sealed", "another output was not kept late"
    assert first.verified(), "a late seal changed the snapshot"
    assert (
        unit.evidence_status.get("agent_output") == "incomplete"
    ), "the overlay missed the late seal"


def test_late_facts_kept_once_per_distinct_write():
    """The same refused write is kept once; a different one is kept again."""
    unit, clock = collector()
    channel = unit.channel("interceptor")
    unit.begin_closure(1.0)
    clock.advance(2.0)
    for _ in range(3):
        with pytest.raises(LateFactError):
            channel.start_call("one", function="tool", args={"n": 1})
    with pytest.raises(LateFactError):
        channel.start_call("one", function="tool", args={"n": 2})
    facts = unit.late_facts
    assert len(facts) == 2, "late facts were kept per refusal, not per write"
    assert {fact.call_id for fact in facts} == {"one"}, "late facts lost the call"
    assert all(
        fact.evidence_keys() == {"function_calls"} for fact in facts
    ), "late facts marked the wrong evidence"


def test_late_fact_payload_refused_before_recording():
    """A payload JSON cannot hold is refused and leaves no late fact."""
    unit, clock = collector()
    channel = unit.channel("host")
    unit.begin_closure(1.0)
    clock.advance(2.0)
    with pytest.raises(RejectedError) as caught:
        channel.start_call("one", function="tool", args={"bad": {1, 2}})
    assert caught.value.kind == "unsupported_value", "unsupported payload mislabelled"
    assert not unit.late_facts, "an unsupported payload was kept as a late fact"


def test_late_facts_never_change_the_sealed_digest():
    """Every later write changes only the overlay, never the snapshot."""
    unit, clock = collector()
    channel = unit.channel("agent")
    channel.record_call(function="tool", args={"n": 1}, result="ok")
    unit.close(0.0)
    snapshot = unit.seal("done")
    digest = snapshot.snapshot_id
    for _ in range(3):
        with pytest.raises(LateFactError):
            channel.finish_call("late", function="tool", args={"n": 1}, result="x")
    with pytest.raises(LateFactError):
        unit.start_turn("a late turn")
    assert snapshot.snapshot_id == digest and snapshot.verified(), "digest changed"
    assert len(unit.late_facts) == 2, "distinct late writes were collapsed"
    assert unit.evidence_status == {
        "function_calls": "incomplete",
        "turns": "incomplete",
    }, "the overlay did not fold the late facts"


def test_late_facts_before_the_seal_fold_into_the_snapshot():
    """A late fact kept during closing or collecting marks the sealed status."""
    unit, clock = collector()
    channel = unit.channel("host")
    unit.begin_closure(1.0)
    clock.advance(2.0)
    with pytest.raises(LateFactError):
        channel.start_call("one", function="tool", args={"n": 1})
    snapshot = unit.seal("done")
    assert (
        snapshot.evidence_status["function_calls"] == "incomplete"
    ), "a pre-seal late fact did not mark the snapshot"


def test_capture_failures_identity_replay_and_lateness():
    """A capture failure replays at any stage; a new one after the seal is late."""
    unit, clock = collector()
    channel = unit.channel("interceptor")
    first = channel.report_capture_failure("one", "tool", "start", "unreachable")
    again = channel.report_capture_failure("one", "tool", "start", "another error")
    assert again is first, "the same identity kept two failures"
    other = channel.report_capture_failure("one", "tool", "finish", "unreachable")
    assert other is not first, "another phase is another failure"
    unit.close(0.0)
    unit.seal("done")
    replay = channel.report_capture_failure("one", "tool", "start", "resent")
    assert replay is first, "a kept failure did not replay after the seal"
    with pytest.raises(LateFactError):
        channel.report_capture_failure("two", "tool", "start", "new loss")
    assert len(unit.late_facts) == 1, "a replayed report was kept late"
    assert unit.evidence_status["function_calls"] == "incomplete", "overlay missed it"
    snapshot_calls = unit.seal("done").to_json()["capture_failures"]
    assert len(snapshot_calls) == 2, "the sealed failures changed"


def test_capture_failure_never_touches_records():
    """A capture failure adds no record and changes none."""
    unit, _ = collector()
    channel = unit.channel("host")
    channel.record_call(function="tool", args={"n": 1}, result="ok")
    before = _records(unit)
    channel.report_capture_failure("one", "tool", "finish", "lost")
    assert _records(unit) == before, "a capture failure changed the records"
    snapshot = unit.close(0.0) and None or unit.seal("done")
    assert (
        snapshot.evidence_status["function_calls"] == "incomplete"
    ), "a capture failure did not mark the capture"


def test_turns_stamp_calls_and_refuse_overlaps():
    """A turn starts only once the previous has its reply, and calls carry its index."""
    unit, _ = collector()
    channel = unit.channel("host")
    first = unit.start_turn("hello")
    with pytest.raises(RejectedError) as caught:
        unit.start_turn("again")
    assert caught.value.kind == "turn_pending", "overlapping turns allowed"
    stamped = channel.start_call("one", function="tool", args={"n": 1})
    assert stamped.record.turn == 0, "the call missed the started turn"
    unit.record_reply(0, "hi")
    with pytest.raises(RejectedError) as caught:
        unit.record_reply(0, "again")
    assert caught.value.kind == "turn_answered", "a second reply was accepted"
    with pytest.raises(RejectedError) as caught:
        unit.record_reply(5, "never started")
    assert caught.value.kind == "turn_not_started", "an unknown turn was answered"
    second = unit.start_turn("more")
    assert second.index == 1, "turn indices did not advance"
    later = channel.start_call("two", function="tool", args={"n": 2})
    assert later.record.turn == 1, "the call kept the old turn"
    turns = unit.close(0.0) and None or unit.seal("done").to_json()["turns"]
    assert [turn["reply"] for turn in turns] == ["hi", None], "turn replies changed"


def test_turns_outside_open_are_late():
    """A turn or reply outside open is a late fact keyed turns."""
    unit, clock = collector()
    unit.start_turn("hello")
    unit.begin_closure(1.0)
    clock.advance(2.0)
    with pytest.raises(LateFactError):
        unit.record_reply(0, "late")
    assert unit.late_facts[0].evidence_keys() == {"turns"}, "wrong evidence key"
    unit.seal("done")
    assert unit.evidence_status["turns"] == "incomplete", "overlay missed the turn"


def test_observations_replace_and_flatten():
    """A source's observations replace whole, and its keys flatten by source."""
    unit, _ = collector()
    unit.record_observations("runtime", {"a": 1}, {"findings": "complete"})
    unit.record_observations("runtime", {"b": 2}, {"findings": "incomplete"})
    snapshot = unit.close(0.0) and None or unit.seal("done")
    data = snapshot.to_json()
    assert data["observations"] == {"runtime": {"b": 2}}, "observations did not replace"
    assert snapshot.evidence_status == {
        "runtime.findings": "incomplete"
    }, "evidence keys were not flattened"
    with pytest.raises(RejectedError, match="unsupported_value"):
        unit.record_observations("runtime", {"b": 2}, {"findings": "unknown"})


def test_observations_after_the_seal_are_late_by_part():
    """A late observation is keyed by each part it carries."""
    unit, _ = collector()
    unit.record_observations("runtime", {"a": 1}, {"findings": "complete"})
    unit.close(0.0)
    unit.seal("done")
    with pytest.raises(LateFactError):
        unit.record_observations("runtime", {"b": 2}, {"workdir": "complete"})
    assert unit.late_facts[0].evidence_keys() == {"runtime.workdir"}, "wrong late key"


def test_environment_replacement_rules():
    """The environment changes while open or closing, and is late after that."""
    unit, clock = collector(pre_environment={"balance": 10})
    unit.replace_environment({"balance": 5})
    unit.begin_closure(5.0)
    unit.replace_environment({"balance": 4})
    clock.advance(6.0)
    with pytest.raises(LateFactError):
        unit.replace_environment({"balance": 1})
    assert unit.late_facts[0].evidence_keys() == {"environment"}, "wrong late key"
    snapshot = unit.seal("done")
    assert snapshot.evidence_status["environment"] == "incomplete", "overlay missed it"
    assert snapshot.to_json()["environment"] == {"balance": 4}, "environment changed"
    assert snapshot.to_json()["pre_environment"] == {"balance": 10}, "pre-env changed"


def test_capture_status_rules():
    """Unfinished session calls and agent-mode emptiness mark the capture."""
    agent_unit, _ = collector(interception="agent")
    snapshot = agent_unit.close(0.0) and None or agent_unit.seal("done")
    assert snapshot.evidence_status == {
        "function_calls": "incomplete"
    }, "an empty agent-mode capture read complete"
    host_unit, _ = collector(interception="host")
    snapshot = host_unit.close(0.0) and None or host_unit.seal("done")
    assert snapshot.evidence_status == {}, "an empty host capture was marked"
    unfinished_agent, _ = collector(interception="host")
    channel = unfinished_agent.channel("agent")
    channel.start_call("one", function="tool", args={"n": 1})
    snapshot = unfinished_agent.close(0.0) and None or unfinished_agent.seal("done")
    assert (
        snapshot.evidence_status["function_calls"] == "incomplete"
    ), "an unfinished agent call read complete"
    unfinished_host, _ = collector(interception="host")
    channel = unfinished_host.channel("host")
    channel.start_call("one", function="tool", args={"n": 1})
    snapshot = unfinished_host.close(0.0) and None or unfinished_host.seal("done")
    assert snapshot.evidence_status == {}, "an unfinished host call marked the capture"


def test_seal_incomplete_keys_are_marked():
    """The caller's incomplete keys mark the snapshot's status."""
    unit, _ = collector()
    unit.close(0.0)
    snapshot = unit.seal("done", incomplete=("function_calls", "agent_output"))
    assert snapshot.evidence_status == {
        "function_calls": "incomplete",
        "agent_output": "incomplete",
    }, "named keys were not marked"
    with pytest.raises(InvalidRequestError):
        unit.seal("done", incomplete=("",))


def test_drain_wakes_on_finish():
    """A drain in progress wakes when the last admitted call finishes."""
    unit, clock = collector()
    channel = unit.channel("host")
    channel.start_call("one", function="tool", args={"n": 1})
    unit.begin_closure(30.0)
    results = {}

    def draining():
        results["unfinished"] = unit.drain()

    thread = threading.Thread(target=draining)
    thread.start()
    time.sleep(0.05)
    channel.finish_call("one", function="tool", args={"n": 1}, result="ok")
    thread.join(timeout=5.0)
    assert not thread.is_alive(), "drain did not wake on the finish"
    assert results["unfinished"] == (), "the finished call stayed unfinished"
    assert unit.stage is Stage.COLLECTING, "drain did not collect"


def test_drain_never_waits_past_the_deadline():
    """A drain returns at the deadline with the unfinished calls listed."""
    unit, clock = collector()
    channel = unit.channel("agent")
    started = channel.start_call("one", function="tool", args={"n": 1})
    unit.begin_closure(2.0)

    def expire():
        time.sleep(0.05)
        clock.advance(3.0)

    threading.Thread(target=expire).start()
    began = time.monotonic()
    unfinished = unit.drain()
    waited = time.monotonic() - began
    assert waited < 1.0, f"drain waited {waited:.2f}s past its deadline"
    assert unfinished == (started.record,), "drain lost the unfinished call"
    assert unit.stage is Stage.COLLECTING, "drain did not collect"


def test_drain_is_a_noop_when_collecting_or_sealed():
    """Draining after collection or the seal returns the unfinished list at once."""
    unit, _ = collector()
    channel = unit.channel("host")
    started = channel.start_call("one", function="tool", args={"n": 1})
    unit.begin_closure(0.0)
    assert unit.drain() == (started.record,), "collecting drain lost the call"
    unit.seal("done")
    assert unit.drain() == (started.record,), "sealed drain lost the call"
    assert unit.stage is Stage.SEALED, "a sealed drain changed the stage"


def test_close_is_closure_plus_drain():
    """Close begins closure and drains in one step."""
    unit, _ = collector()
    channel = unit.channel("host")
    channel.start_call("one", function="tool", args={"n": 1})
    channel.finish_call("one", function="tool", args={"n": 1}, result="ok")
    assert unit.close(1.0) == (), "close returned unfinished calls"
    assert unit.stage is Stage.COLLECTING, "close did not collect"


@pytest.mark.parametrize(
    "wait", [float("nan"), float("inf"), float("-inf"), -1, -0.5, 2**2000, True, "1"]
)
def test_closure_wait_must_be_finite_and_not_negative(wait):
    """A wait that no deadline can follow is refused and leaves the collector open."""
    unit, _ = collector()
    with pytest.raises(InvalidRequestError):
        unit.begin_closure(wait)
    assert unit.stage is Stage.OPEN, "a refused wait began the closure"


# --- the two required scenarios ---


def test_mitm_vanishes_mid_call():
    """An interceptor's link is lost mid-call: the call stays unfinished and the seal says so."""
    unit, clock = collector(interception="host")
    link = unit.open_link("wire")
    channel = unit.channel("interceptor", link)
    started = channel.start_call("one", function="tool", args={"n": 1})
    link.lose("the binding stream ended")
    assert unit.links["wire"] == "lost", "the link read as live after the loss"
    unit.begin_closure(2.0)
    clock.advance(3.0)
    unfinished = unit.drain()
    assert [record.call_id for record in unfinished] == ["one"], "drain lost the call"
    snapshot = unit.seal("the run ended")
    assert (
        snapshot.evidence_status["function_calls"] == "incomplete"
    ), "a lost link read as a complete capture"
    sealed_call = snapshot.to_json()["function_calls"][0]
    assert sealed_call["complete"] is False, "the sealed call was completed"
    assert sealed_call["result"] is None, "a result was invented for the lost call"
    digest = snapshot.snapshot_id
    with pytest.raises(LateFactError) as caught:
        channel.finish_call("one", function="tool", args={"n": 1}, result="recovered")
    assert caught.value.kind == "sealed", "the late finish was not refused"
    assert caught.value.late_fact, "the late finish was not marked late"
    assert snapshot.snapshot_id == digest and snapshot.verified(), "digest changed"
    assert [fact.event for fact in unit.late_facts] == ["finish_call"], "wrong facts"
    assert unit.evidence_status["function_calls"] == "incomplete", "overlay missed it"


def test_stalled_dispatcher_with_expired_binding_at_seal():
    """A stalled dispatcher that never reports the loss still marks the capture."""
    unit, clock = collector(interception="host")
    link = unit.open_link("wire", stall_after=5.0, bound=lambda: False)
    channel = unit.channel("interceptor", link)
    channel.start_call("one", function="tool", args={"n": 1})
    unit.begin_closure(1.0)
    clock.advance(10.0)
    began = time.monotonic()
    unfinished = unit.drain()
    assert time.monotonic() - began < 1.0, "the seal waited for the dispatcher"
    assert [record.call_id for record in unfinished] == ["one"], "drain lost the call"
    snapshot = unit.seal("done")
    assert (
        snapshot.evidence_status["function_calls"] == "incomplete"
    ), "an expired binding read as a complete capture"
    assert unit.links["wire"] in ("lost", "stalled"), "the link read as live"


def test_interceptor_channel_provenance_is_interceptor():
    """A link-dispatched interceptor call carries interceptor provenance."""
    unit, _ = collector(interception="host")
    link = unit.open_link("wire")
    channel = unit.channel("interceptor", link)
    record = channel.start_call("one", function="tool", args={"n": 1}).record
    assert record.provenance == "interceptor", "the channel lost its provenance"
    finished = channel.finish_call(
        "one", function="tool", args={"n": 1}, result="ok"
    ).record
    assert finished.provenance == "interceptor", "the finish changed the provenance"


@pytest.mark.parametrize("provenance", ["bogus", "", None, b"host", "Host"])
def test_channel_refuses_unknown_provenance(provenance):
    """A channel's provenance is one the ledger knows, never what a caller invents."""
    unit, _ = collector()
    with pytest.raises(ValueError):
        unit.channel(provenance)


def test_channel_touches_its_link_per_frame():
    """Every frame a channel handles beats its link."""
    unit, clock = collector()
    link = unit.open_link("wire", stall_after=5.0)
    channel = unit.channel("interceptor", link)
    channel.start_call("one", function="tool", args={"n": 1})
    clock.advance(3.0)
    assert link.state(clock.now) == "live", "a handled frame did not beat the link"
    clock.advance(10.0)
    assert link.state(clock.now) == "stalled", "the link kept beating alone"
    channel.finish_call("one", function="tool", args={"n": 1}, result="ok")
    assert link.state(clock.now) == "live", "the finish did not beat the link"


def test_released_links_never_mark_the_capture():
    """A released link is clean, whatever its binding says afterwards."""
    unit, clock = collector(interception="host")
    link = unit.open_link("wire", bound=lambda: True)
    channel = unit.channel("interceptor", link)
    channel.start_call("one", function="tool", args={"n": 1})
    channel.finish_call("one", function="tool", args={"n": 1}, result="ok")
    link.release()
    unit.close(0.0)
    snapshot = unit.seal("done")
    assert snapshot.evidence_status == {}, "a released link marked the capture"
    assert unit.links["wire"] == "released", "the released link read otherwise"


def test_link_summary_is_not_part_of_the_digest():
    """The links summary is diagnostics only; it never enters the snapshot."""
    unit, clock = collector()
    link = unit.open_link("wire")
    link.lose("gone")
    unit.close(0.0)
    snapshot = unit.seal("done")
    assert "links" not in snapshot.to_json(), "the links summary entered the digest"


def test_release_after_loss_still_marks_the_capture():
    """A release that follows a loss does not make the capture complete."""
    unit, _ = collector(interception="host")
    link = unit.open_link("wire")
    channel = unit.channel("interceptor", link)
    channel.record_call(function="tool", args={"n": 1}, result="ok")
    link.lose("binding stream dropped")
    link.release()
    unit.close(0.0)
    assert unit.seal("done").evidence_status == {
        "function_calls": "incomplete"
    }, "a release after a loss hid the loss"


def test_a_link_name_opens_once():
    """A second link under a name would hide the first one's loss, so it is refused."""
    unit, _ = collector()
    first = unit.open_link("wire")
    with pytest.raises(InvalidRequestError):
        unit.open_link("wire")
    first.lose("gone")
    assert unit.links == {"wire": "lost"}, "the refused open displaced the link"


def test_concurrent_opens_of_one_link_name_admit_one():
    """Racing opens of one name leave one registered link and refuse the rest."""
    unit, _ = collector()
    barrier = threading.Barrier(8)

    def opener(_):
        barrier.wait()
        try:
            unit.open_link("wire")
        except InvalidRequestError:
            return False
        return True

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(opener, range(8)))
    assert results.count(True) == 1, "more than one open of the name succeeded"
    assert list(unit.links) == ["wire"], "the registered links changed"


def test_channel_refuses_a_link_the_collector_did_not_open():
    """A link the seal never reads must not carry a channel's calls."""
    unit, _ = collector()
    foreign = collector()[0].open_link("wire")
    with pytest.raises(InvalidRequestError):
        unit.channel("interceptor", foreign)
    with pytest.raises(InvalidRequestError):
        unit.channel("interceptor", CaptureLink(name="wire"))
    own = unit.open_link("wire")
    assert unit.channel("interceptor", own).link is own, "an own link was refused"


def test_link_opened_after_the_deadline_is_a_late_fact():
    """A link opened once the closure deadline has passed arrives too late."""
    unit, clock = collector()
    unit.begin_closure(1.0)
    clock.advance(2.0)
    with pytest.raises(LateFactError):
        unit.open_link("new")
    assert unit.stage is Stage.COLLECTING, "the deadline did not end the closure"
    assert [fact.event for fact in unit.late_facts] == ["open_link"], "no late fact"
    assert unit.links == {}, "a late link was registered"


def test_closure_deadline_must_be_finite():
    """A wait whose deadline overflows is refused, as an infinite wait is."""
    unit, _ = collector(monotonic=lambda: 1e308)
    with pytest.raises(InvalidRequestError):
        unit.begin_closure(1e308)
    assert unit.stage is Stage.OPEN, "a refused wait began the closure"


def test_start_without_a_call_id_is_refused_at_every_stage():
    """A start needs an ID to pair with its finish, and never becomes a late fact."""
    unit, _ = collector(new_call_id=lambda: "assigned")
    channel = unit.channel("host")
    with pytest.raises(InvalidRequestError):
        channel.start_call(None, function="tool", args={})
    unit.begin_closure(5.0)
    with pytest.raises(InvalidRequestError):
        channel.start_call(None, function="tool", args={})
    assert unit.late_facts == (), "a start without an ID became a late fact"
    assert unit.close(0.0) == (), "a start without an ID was recorded"


@pytest.mark.parametrize("disposition", [[], {}, 3, "invented"])
def test_start_refuses_a_malformed_disposition(disposition):
    """A disposition no call can carry is a typed refusal, not a TypeError."""
    unit, _ = collector()
    with pytest.raises(InvalidRequestError):
        unit.channel("host").start_call(
            "one", function="tool", args={}, disposition=disposition
        )
    assert unit.close(0.0) == (), "a refused start was recorded"


def test_record_call_takes_an_explicit_id_and_replays():
    """A whole call may name its ID, and a replay returns the first record."""
    unit, _ = collector()
    channel = unit.channel("host")
    first = channel.record_call(
        call_id="one", function="tool", args={"n": 1}, result="ok"
    )
    replay = channel.record_call(
        call_id="one", function="tool", args={"n": 1}, result="ok"
    )
    assert first.inserted and first.record.call_id == "one", "the ID was not kept"
    assert replay.record is first.record and not replay.inserted, "replay inserted"
    with pytest.raises(RejectedError) as caught:
        channel.record_call(call_id="one", function="tool", args={"n": 2}, result="ok")
    assert caught.value.kind == "call_conflict", "another body was not a conflict"


# --- threads and shape ---


def test_concurrent_start_finish_and_seal():
    """Concurrent writers linearise: one insert per ID, one snapshot."""
    unit, clock = collector()
    channel = unit.channel("interceptor")

    def work(index):
        call_id = f"call-{index}"
        channel.start_call(call_id, function="tool", args={"n": index})
        return channel.finish_call(
            call_id, function="tool", args={"n": index}, result="ok"
        )

    with ThreadPoolExecutor(max_workers=16) as pool:
        receipts = list(pool.map(work, range(32)))
    assert all(receipt.record.complete for receipt in receipts), "a finish was lost"
    assert len(_records(unit)) == 32, "concurrent starts lost or duplicated calls"
    unit.close(0.0)
    with ThreadPoolExecutor(max_workers=8) as pool:
        snapshots = list(pool.map(lambda _: unit.seal("done"), range(8)))
    assert all(snapshot is snapshots[0] for snapshot in snapshots), "resealing raced"
    assert snapshots[0].verified(), "the raced snapshot does not verify"


def test_concurrent_identical_starts_insert_once():
    """Concurrent identical replays insert exactly one record."""
    unit, _ = collector()
    channel = unit.channel("agent")
    with ThreadPoolExecutor(max_workers=16) as pool:
        receipts = list(
            pool.map(
                lambda _: channel.start_call("one", function="tool", args={"n": 1}),
                range(16),
            )
        )
    assert sum(receipt.inserted for receipt in receipts) == 1, "replay inserted twice"
    assert all(
        receipt.record is receipts[0].record for receipt in receipts
    ), "replays returned different records"


def test_live_collector_refuses_pickling_and_copying():
    """A collector is live state, so it never pickles or copies."""
    unit, _ = collector()
    with pytest.raises(TypeError):
        pickle.dumps(unit)
    with pytest.raises(TypeError):
        copy.copy(unit)
    with pytest.raises(TypeError):
        copy.deepcopy(unit)


def test_snapshot_json_matches_the_sealed_reader():
    """The sealed JSON passes the strict consumer reader's shape checks."""
    unit, _ = collector()
    channel = unit.channel("interceptor")
    channel.start_call("one", function="tool", args={"n": 1})
    channel.finish_call(
        "one",
        function="tool",
        args={"n": 1},
        result="ok",
        disposition="forwarded",
        structured_content={"items": []},
        injection_outcomes=[],
    )
    channel.start_call("two", function="tool", args={"n": 2})
    unit.start_turn("hello")
    unit.record_reply(0, "hi")
    unit.record_observations("runtime", {"a": 1}, {"findings": "complete"})
    unit.close(0.0)
    snapshot = unit.seal("done")
    data = {**snapshot.to_json(), "snapshot_id": snapshot.snapshot_id}
    assert shared.verified(data) is data, "the shared codec refused the snapshot"
    assert _call_shapes(data["function_calls"]), "a call failed the reader's shape"
    assert _turn_shapes(data["turns"]), "a turn failed the reader's shape"
    assert _failure_shapes(data["capture_failures"]), "a failure failed the shape"
    assert data["evidence_status"] == {
        "runtime.findings": "complete",
        "function_calls": "incomplete",
    }, "evidence status was lost"


CALL_FIELDS = frozenset(
    {
        "call_id",
        "function",
        "args",
        "result",
        "error",
        "blocked",
        "disposition",
        "complete",
        "provenance",
        "timestamp",
        "turn",
        "pre_environment",
        "post_environment",
        "structured_content",
        "injection_outcomes",
    }
)
TURN_FIELDS = frozenset({"index", "message", "reply", "started_at", "ended_at"})
FAILURE_FIELDS = frozenset(
    {"call_id", "function", "phase", "reason", "stage", "by", "at"}
)


def _call_shapes(calls):
    return all(
        set(call) == CALL_FIELDS
        and all(
            isinstance(call[key], str) for key in ("call_id", "function", "timestamp")
        )
        and isinstance(call["args"], dict)
        and (call["result"] is None or isinstance(call["result"], str))
        and (call["error"] is None or isinstance(call["error"], str))
        and all(type(call[key]) is bool for key in ("blocked", "complete"))
        and (call["disposition"] is None or isinstance(call["disposition"], str))
        and call["provenance"] in ("host", "interceptor", "agent")
        and (call["turn"] is None or type(call["turn"]) is int)
        and call["pre_environment"] is None
        and call["post_environment"] is None
        and (
            call["structured_content"] is None
            or isinstance(call["structured_content"], dict)
        )
        and (
            call["injection_outcomes"] is None
            or isinstance(call["injection_outcomes"], list)
        )
        for call in calls
    )


def _turn_shapes(turns):
    return all(
        set(turn) == TURN_FIELDS
        and type(turn["index"]) is int
        and isinstance(turn["message"], str)
        and (turn["reply"] is None or isinstance(turn["reply"], str))
        and isinstance(turn["started_at"], str)
        and (turn["ended_at"] is None or isinstance(turn["ended_at"], str))
        for turn in turns
    )


def _failure_shapes(failures):
    return all(
        set(failure) == FAILURE_FIELDS
        and all(isinstance(failure[key], str) for key in FAILURE_FIELDS)
        and failure["phase"] in ("start", "finish")
        for failure in failures
    )
