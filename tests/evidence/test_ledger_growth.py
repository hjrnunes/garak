"""Per-call growth tests: rising late cost or retention indicates whole-ledger copies."""

import copy
import gc
import statistics
import time
import tracemalloc
from unittest.mock import patch

import pytest

pytest.importorskip("midojo_spec")

from garak.evidence.ledger import CallLedger
from garak.evidence.records import RejectedError


def run_calls(observer=None, ledger=None):
    """Run 500 calls with replays, a few conflicts and substantial call data."""
    if ledger is None:
        ledger = CallLedger()
    channel = ledger.channel("host")
    args = {"payload": "a" * 320, "parts": [True, 1, 1.0]}
    elapsed = []
    inserted = 0
    for index in range(500):
        identifier = f"call-{index - 3 if index % 4 == 3 else index}"
        result = "r" * 2048 if index not in (111, 223, 335, 447) else "conflict"
        start = time.perf_counter_ns()
        try:
            receipt = channel.record_call(
                call_id=identifier, function="tool", args=args, result=result
            )
            inserted += receipt.inserted
        except RejectedError as error:
            assert error.kind == "call_conflict", "unexpected growth-run refusal"
        elapsed.append(time.perf_counter_ns() - start)
        if observer is not None:
            observer(inserted)
    return ledger, elapsed


def test_bounded_growth_and_no_copy():
    """Late calls should not get slower or retain more as the ledger grows."""
    times = [run_calls()[1] for _ in range(5)]
    first = min(sum(run[:100]) / 100 for run in times)
    last = min(sum(run[-100:]) / 100 for run in times)
    assert last <= 3 * first, f"late calls grew: {last / first:.2f}x"

    retained = {}
    tracemalloc.start()
    try:
        baseline = tracemalloc.get_traced_memory()[0]

        def observe(count):
            if count in (100, 275, 375) and count not in retained:
                retained[count] = tracemalloc.get_traced_memory()[0]

        ledger, _ = run_calls(observe)
        assert len(ledger) == 375, "run inserted the wrong number of records"
    finally:
        tracemalloc.stop()
    early = retained[100] - baseline
    late = retained[375] - retained[275]
    assert late <= 1.5 * early, f"late retention grew: {late / early:.2f}x"

    copies = []

    def count_copy(*_args, **_kwargs):
        copies.append(True)
        raise AssertionError("whole-ledger copy attempted")

    with (
        patch.object(copy, "copy", count_copy),
        patch.object(copy, "deepcopy", count_copy),
    ):
        run_calls()
    assert not copies, "intake invoked copy"


LIST_WHOLE = (
    "__iter__",
    "__reversed__",
    "__contains__",
    "__eq__",
    "__add__",
    "__radd__",
    "__mul__",
    "__rmul__",
    "copy",
    "index",
    "count",
)
DICT_WHOLE = ("__iter__", "keys", "values", "items", "copy", "__eq__", "__or__")


def watched(base, whole, seen):
    """Subclass a container so that operations on all of its entries are logged."""

    def logging(name):
        def method(self, *args, **kwargs):
            seen.append(f"{base.__name__}.{name}")
            return getattr(base, name)(self, *args, **kwargs)

        return method

    members = {name: logging(name) for name in whole}
    if base is list:

        def __getitem__(self, key):
            if isinstance(key, slice):
                seen.append("list.slice")
            return list.__getitem__(self, key)

        members["__getitem__"] = __getitem__
    return type(f"Watched{base.__name__.title()}", (base,), members)


def watch_ledger(ledger):
    """Replace the ledger's containers with watched ones and return the log."""
    seen = []
    ledger._calls = watched(list, LIST_WHOLE, seen)(ledger._calls)
    ledger._positions = watched(dict, DICT_WHOLE, seen)(ledger._positions)
    ledger._encoded_by_id = watched(dict, DICT_WHOLE, seen)(ledger._encoded_by_id)
    return seen


def test_intake_touches_single_entries_only():
    """Intake, replay, lookup and refusal use no whole-container method."""
    ledger = CallLedger()
    seen = watch_ledger(ledger)
    run_calls(ledger=ledger)
    channel = ledger.channel("agent")
    for index in range(20):
        channel.record_call(function="tool", args={"n": index}, result="ok")
        ledger.get(f"call-{index}")
    assert len(ledger) == 395, "run inserted the wrong number of records"
    assert not seen, f"whole-ledger operations during intake: {sorted(set(seen))}"


WHOLE_LEDGER_OPERATIONS = {
    "list": lambda ledger: list(ledger._calls),
    "tuple": lambda ledger: tuple(ledger._calls),
    "slice": lambda ledger: ledger._calls[:],
    "iteration": lambda ledger: sum(1 for _ in ledger._calls),
    "shallow copy": lambda ledger: copy.copy(ledger._calls),
    "index scan": lambda ledger: ledger._calls.index(ledger._calls[0]),
    "dict copy": lambda ledger: dict(ledger._positions),
    "dict items": lambda ledger: list(ledger._encoded_by_id.items()),
    "dict keys": lambda ledger: sorted(ledger._positions),
}


@pytest.mark.parametrize(
    "operation",
    list(WHOLE_LEDGER_OPERATIONS.values()),
    ids=list(WHOLE_LEDGER_OPERATIONS),
)
def test_whole_ledger_operations_are_detected(operation):
    """The guard fails when intake uses a container's own whole-container method."""

    class Regressed(CallLedger):
        """A ledger whose intake does work proportional to the calls it holds."""

        def _insert(self, *args, **kwargs):
            if len(self._calls) > 1:
                operation(self)
            return super()._insert(*args, **kwargs)

    ledger = Regressed()
    seen = watch_ledger(ledger)
    run_calls(ledger=ledger)
    assert seen, "guard missed a whole-ledger operation"


SCALE_CALLS = 12000
FLAT_RATIO = 1.5


def late_over_early(ledger_class):
    """Time a long run and return late over early median per-call cost."""
    channel = ledger_class().channel("host")
    elapsed = []
    collecting = gc.isenabled()
    gc.disable()
    try:
        for index in range(SCALE_CALLS):
            start = time.perf_counter_ns()
            channel.record_call(
                call_id=f"id-{index}",
                function="tool",
                args={"n": index % 7},
                result="ok",
            )
            elapsed.append(time.perf_counter_ns() - start)
    finally:
        if collecting:
            gc.enable()
    return statistics.median(elapsed[-1000:]) / statistics.median(elapsed[500:1500])


def test_per_call_time_is_flat_at_scale():
    """A long run is as fast at the end as near the start, in at least one of three."""
    ratios = []
    for _ in range(3):
        ratios.append(late_over_early(CallLedger))
        if ratios[-1] <= FLAT_RATIO:
            return
    pytest.fail(f"late calls grew in every run: {ratios}")


REACHING_AROUND_THE_GUARD = {
    "list method": lambda ledger: list.copy(ledger._calls),
    "dict method": lambda ledger: dict.copy(ledger._positions),
    "base iteration": lambda ledger: sum(1 for _ in list.__iter__(ledger._calls)),
}


@pytest.mark.parametrize(
    "operation",
    list(REACHING_AROUND_THE_GUARD.values()),
    ids=list(REACHING_AROUND_THE_GUARD),
)
def test_scale_run_detects_work_that_avoids_the_watched_methods(operation):
    """Calling a built-in method directly skips the guard but not a long timed run."""

    class Regressed(CallLedger):
        """A ledger whose intake visits every earlier call without a watched method."""

        def _insert(self, *args, **kwargs):
            operation(self)
            return super()._insert(*args, **kwargs)

    assert late_over_early(Regressed) > FLAT_RATIO, "long run missed whole-ledger work"
