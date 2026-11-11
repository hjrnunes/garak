"""The staged evidence collector: one evaluation's calls, turns and observations.

A collector moves through four stages. ``open`` admits every write.
``closing`` (``begin_closure``) admits no new call but lets calls already
started finish, until none is unfinished or the closure deadline passes.
``collecting`` (``drain``) takes the final observations; the environment no
longer changes. ``sealed`` (``seal``) fixes an immutable snapshot whose ID is
the shared digest of its content.

A write a stage no longer admits is refused as :class:`LateFactError` and kept
once per distinct write as a late fact. Late facts never change a sealed
snapshot or its digest; they appear only in the collector's overlay status.
Late facts kept before the seal are folded into the snapshot's evidence status.

Time is injectable: ``clock()`` stamps record and turn timestamps,
``monotonic()`` judges deadlines and stall ages. Each event takes its arrival
time from ``monotonic()``; an event arriving in ``closing`` at or after the
deadline finds the collector ``collecting`` before it is judged, so a finish
after the deadline is a late fact even when nobody called ``drain()``.

The collector does not grade: it exposes records and statuses.
"""

from __future__ import annotations

import math
import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from garak.evidence.facts import (
    FUNCTION_CALLS,
    Observation,
    Turn,
    CaptureFailure,
    LateFact,
    LateFactError,
    cut_reason,
    json_payload,
    require_text,
)
from garak.evidence.intake import Stage, finish_call, start_call
from garak.evidence.ledger import CallLedger, Receipt
from garak.evidence.links import CaptureLink
from garak.evidence.records import (
    PROVENANCES,
    CallRecord,
    InvalidRequestError,
    freeze,
)
from garak.evidence.seal import Snapshot
from garak.evidence.sealing import SealingMixin
from garak.evidence.world import WorldMixin

__all__ = ["Collector", "CollectorChannel", "Stage"]

INTERCEPTIONS = frozenset({"host", "agent"})
#: The longest wait drain takes before re-reading the clock, so a deadline
#: another thread's clock advanced past still ends the wait.
_WAIT_QUANTUM = 0.05
_LATE_DETAIL = "the write is recorded as a late fact"


def _default_clock() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_monotonic() -> float:
    import time

    return time.monotonic()


def _default_call_id() -> str:
    return str(uuid.uuid4())


@dataclass(frozen=True, slots=True)
class CollectorChannel:
    """A recording capability whose provenance is fixed, and whose link stays live.

    Every frame the channel handles touches its link, so a dispatcher that
    stops sending frames stops beating and the seal reads the stall.
    """

    collector: Collector
    provenance: str
    link: CaptureLink | None = None

    def __post_init__(self) -> None:
        if type(self.provenance) is not str or self.provenance not in PROVENANCES:
            raise ValueError("unknown call provenance")
        if self.link is not None and not self.collector._owns(self.link):
            raise InvalidRequestError("the link was not opened on this collector")

    def start_call(
        self,
        call_id: str,
        *,
        function: str,
        args: dict,
        disposition: str | None = None,
    ) -> Receipt:
        """Record that a call started: an unfinished record the finish completes."""
        self.collector._touch(self.link)
        return start_call(
            self.collector,
            self.provenance,
            call_id,
            function=function,
            args=args,
            disposition=disposition,
        )

    def finish_call(
        self,
        call_id: str | None = None,
        **fields: Any,
    ) -> Receipt:
        """Finish the call ``call_id`` started, or record it whole when it was not."""
        return self._finish(call_id, **fields)

    def record_call(self, *, call_id: str | None = None, **fields: Any) -> Receipt:
        """Record a complete call, or return the first record for an identical replay."""
        return self._finish(call_id, **fields)

    def _finish(self, call_id: str | None, **fields: Any) -> Receipt:
        """Forward a finish to the collector, touching the link first."""
        self.collector._touch(self.link)
        return finish_call(self.collector, self.provenance, call_id, **fields)

    def report_capture_failure(
        self, call_id: str, function: str, phase: str, reason: str
    ) -> CaptureFailure:
        """Report a call record that could not be delivered; kept once per identity."""
        self.collector._touch(self.link)
        return self.collector._report_capture_failure(
            self.provenance, call_id, function, phase, reason
        )


class Collector(WorldMixin, SealingMixin):
    """One evaluation's staged evidence collector.

    A collector is a live object: it refuses pickling and copying. Its ledger
    is shared, not copied; records are immutable and shared with every reader.
    """

    def __init__(
        self,
        evaluation_id: str,
        *,
        run_id: str | None = None,
        user_task_id: str | None = None,
        injection_task_id: str | None = None,
        interception: str = "host",
        pre_environment: Mapping[str, Any] | None = None,
        clock: Callable[[], str] | None = None,
        monotonic: Callable[[], float] | None = None,
        new_call_id: Callable[[], str] | None = None,
    ):
        require_text(evaluation_id, "evaluation_id")
        if interception not in INTERCEPTIONS:
            raise InvalidRequestError("interception must be 'host' or 'agent'")
        self.evaluation_id = evaluation_id
        self.run_id = run_id
        self.user_task_id = user_task_id
        self.injection_task_id = injection_task_id
        self.interception = interception
        self._clock = clock or _default_clock
        self._monotonic = monotonic or _default_monotonic
        self._stage = Stage.OPEN
        self._deadline: float | None = None
        self._turns: list[Turn] = []
        self._turn_lock = threading.RLock()
        self._state_lock = threading.RLock()
        self._condition = threading.Condition(self._state_lock)
        self._late_facts: list[LateFact] = []
        self._late_identities: set[tuple[str, str, str | None, str]] = set()
        self._capture_failures: dict[tuple[str, str, str], CaptureFailure] = {}
        self._observations: dict[str, Observation] = {}
        self._pre_environment = (
            None if pre_environment is None else freeze(dict(pre_environment))
        )
        self._environment = self._pre_environment
        self._links: dict[str, CaptureLink] = {}
        self._snapshot: Snapshot | None = None
        self._ledger = CallLedger(
            clock=self._clock,
            new_call_id=new_call_id or _default_call_id,
            current_turn=self._current_turn,
        )

    # --- stage and time ---

    @property
    def stage(self) -> Stage:
        """The collector's current stage."""
        with self._state_lock:
            return self._stage

    @property
    def links(self) -> dict[str, str]:
        """A diagnostics-only summary of each link's state; never part of the digest."""
        now = self._monotonic()
        with self._state_lock:
            return {name: link.state(now) for name, link in self._links.items()}

    @property
    def late_facts(self) -> tuple[LateFact, ...]:
        """The late facts kept so far, in arrival order."""
        with self._state_lock:
            return tuple(self._late_facts)

    @property
    def evidence_status(self) -> dict[str, str]:
        """The sealed status overlaid with later late facts and capture failures."""
        with self._state_lock:
            statuses = dict(self._sealed_status())
            for fact in self._late_facts:
                for key in fact.evidence_keys():
                    statuses[key] = "incomplete"
            if self._capture_failures:
                statuses[FUNCTION_CALLS] = "incomplete"
            return statuses

    def _on_time(self, at: float) -> None:
        """Demote ``closing`` to ``collecting`` once the deadline has passed.

        The caller holds the state lock.
        """
        if (
            self._stage is Stage.CLOSING
            and self._deadline is not None
            and at >= self._deadline
        ):
            self._stage = Stage.COLLECTING
            self._condition.notify_all()

    def begin_closure(self, wait_seconds: float) -> Stage:
        """Stop admitting new calls; admitted calls may finish until the deadline.

        ``wait_seconds`` must be a finite number that is not negative; a wait
        no deadline can follow is refused, so ``drain`` always ends.
        """
        if type(wait_seconds) not in (int, float):
            raise InvalidRequestError("wait_seconds must be a number")
        try:
            finite = math.isfinite(wait_seconds)
        except OverflowError:
            finite = False
        if not finite or wait_seconds < 0:
            raise InvalidRequestError("wait_seconds must be finite and not negative")
        with self._condition:
            if self._stage is Stage.OPEN:
                deadline = self._monotonic() + wait_seconds
                if not math.isfinite(deadline):
                    raise InvalidRequestError("the closure deadline must be finite")
                self._stage = Stage.CLOSING
                self._deadline = deadline
            return self._stage

    def drain(self) -> tuple[CallRecord, ...]:
        """Wait until no unfinished call remains or the deadline passes, then collect.

        Already ``collecting`` or ``sealed``, this returns the unfinished list
        at once. It never waits past the deadline: each wait is bounded by the
        time left, and a clock another thread advances is re-read at least
        every quantum, so a deadline that passes without a finish still ends
        the wait.
        """
        with self._condition:
            while self._stage is Stage.CLOSING:
                if not self._ledger.unfinished():
                    break
                remaining = (
                    self._deadline - self._monotonic()
                    if self._deadline is not None
                    else None
                )
                if remaining is None or remaining <= 0:
                    break
                self._condition.wait(min(remaining, _WAIT_QUANTUM))
            if self._stage is Stage.CLOSING:
                self._stage = Stage.COLLECTING
                self._condition.notify_all()
            return self._ledger.unfinished()

    def close(self, wait_seconds: float) -> tuple[CallRecord, ...]:
        """Begin closure and drain it."""
        self.begin_closure(wait_seconds)
        return self.drain()

    # --- channels and links ---

    def channel(
        self, provenance: str, link: CaptureLink | None = None
    ) -> CollectorChannel:
        """Return a channel that fixes provenance and touches ``link`` per frame."""
        return CollectorChannel(self, provenance, link)

    def open_link(
        self,
        name: str,
        *,
        stall_after: float | None = None,
        bound: Callable[[], bool] | None = None,
    ) -> CaptureLink:
        """Open a capture link whose state the seal reads from the binding itself.

        A name opens once: a second link under it would hide the first one's
        loss from the seal, so a reconnection takes a new name.
        """
        require_text(name, "link name")
        link = CaptureLink(name=name, stall_after=stall_after, bound=bound)
        arrival = self._monotonic()
        link.beat(arrival)
        at = self._clock()
        with self._condition:
            self._on_time(arrival)
            if self._stage not in (Stage.OPEN, Stage.CLOSING):
                self._record_late(
                    "open_link",
                    "host",
                    at,
                    evidence=(FUNCTION_CALLS,),
                    payload={"name": name, "stall_after": stall_after},
                )
                raise self._late_error(self._stage)
            if name in self._links:
                raise InvalidRequestError(f"link '{name}' is already open")
            self._links[name] = link
            return link

    def _owns(self, link: CaptureLink) -> bool:
        """Whether ``link`` is the one this collector registered under its name."""
        with self._state_lock:
            return self._links.get(getattr(link, "name", None)) is link

    def _touch(self, link: CaptureLink | None) -> None:
        if link is not None:
            link.beat(self._monotonic())

    # --- calls ---

    def _report_capture_failure(
        self, provenance: str, call_id: str, function: str, phase: str, reason: str
    ) -> CaptureFailure:
        """Keep one capture failure per identity, marking the capture incomplete."""
        require_text(call_id, "call_id")
        require_text(function, "function")
        require_text(reason, "reason")
        if phase not in ("start", "finish"):
            raise InvalidRequestError("phase must be 'start' or 'finish'")
        arrival = self._monotonic()
        at = self._clock()
        cut = cut_reason(reason)
        identity = (call_id, function, phase)
        with self._condition:
            self._on_time(arrival)
            stage = self._stage
            known = self._capture_failures.get(identity)
            if known is not None:
                return known
            if stage is Stage.SEALED:
                self._record_late(
                    "report_capture_failure",
                    provenance,
                    at,
                    evidence=(FUNCTION_CALLS,),
                    payload={
                        "call_id": call_id,
                        "function": function,
                        "phase": phase,
                        "reason": cut,
                    },
                    call_id=call_id,
                )
                raise self._late_error(stage)
            failure = CaptureFailure(
                call_id=call_id,
                function=function,
                phase=phase,
                reason=cut,
                stage=str(stage),
                by=provenance,
                at=at,
            )
            self._capture_failures[identity] = failure
            return failure

    # --- late facts ---

    def _late_error(self, stage: Stage) -> LateFactError:
        """The refusal a kept late fact raises."""
        if stage is Stage.SEALED:
            return LateFactError("sealed", f"the evaluation is sealed: {_LATE_DETAIL}")
        return LateFactError("closing", f"the evaluation is closing: {_LATE_DETAIL}")

    def _record_late(
        self,
        event: str,
        by: str,
        at: str,
        *,
        evidence: tuple[str, ...],
        payload: Mapping[str, Any],
        call_id: str | None = None,
    ) -> None:
        """Keep one late fact per distinct write; the caller holds the state lock.

        A payload JSON cannot hold is refused as ``unsupported_value`` before
        anything is kept.
        """
        fact = LateFact(
            event=event,
            stage=str(self._stage),
            by=by,
            at=at,
            evidence=evidence,
            payload=json_payload(payload),
            call_id=call_id,
        )
        identity = fact.identity()
        if identity in self._late_identities:
            return
        self._late_identities.add(identity)
        self._late_facts.append(fact)

    # --- a live collector never copies ---

    def __reduce__(self) -> Any:
        raise TypeError("a live collector cannot be pickled")

    def __copy__(self) -> Any:
        raise TypeError("a live collector cannot be copied")

    def __deepcopy__(self, memo: dict[int, Any]) -> Any:
        raise TypeError("a live collector cannot be copied")
