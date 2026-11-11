"""Single-writer call intake with stable IDs and idempotent replay."""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import NamedTuple

from garak.evidence.records import (
    BODY_FIELDS,
    PROVENANCES,
    CallBody,
    CallRecord,
    Disposition,
    Provenance,
    RejectedError,
    check_call_id,
)


class Receipt(NamedTuple):
    """The recorded call and whether this intake inserted it."""

    record: CallRecord
    inserted: bool


class CallLedger:
    """Append-only calls of one evaluation with constant-time ID lookup."""

    def __init__(
        self,
        *,
        clock: Callable[[], str] | None = None,
        new_call_id: Callable[[], str] | None = None,
        current_turn: Callable[[], int | None] | None = None,
    ):
        self._clock = clock or (lambda: datetime.now(timezone.utc).isoformat())
        self._new_call_id = new_call_id or (lambda: str(uuid.uuid4()))
        self._current_turn = current_turn or (lambda: None)
        self._calls: list[CallRecord] = []
        self._positions: dict[str, int] = {}
        self._encoded_by_id: dict[str, str] = {}
        self._lock = threading.Lock()

    def __len__(self) -> int:
        """Return the number of distinct recorded calls."""
        with self._lock:
            return len(self._calls)

    def get(self, call_id: str) -> CallRecord | None:
        """Look up a recorded call by ID without scanning the ledger."""
        with self._lock:
            position = self._positions.get(call_id)
            return None if position is None else self._calls[position]

    def calls(self) -> tuple[CallRecord, ...]:
        """Return an ordered snapshot sharing immutable records with the ledger."""
        with self._lock:
            return tuple(self._calls)

    def channel(self, provenance: Provenance) -> CallChannel:
        """Bind the authority of subsequent writes to a channel."""
        return CallChannel(self, provenance)

    def _insert(
        self,
        provenance: Provenance,
        call_id: str | None,
        body: CallBody,
        encoded: str,
    ) -> Receipt:
        """Append a validated body, or return the first record for an exact replay.

        The caller has already validated and encoded the body, so the lock is
        held only for index lookups and one append.
        """
        assigned = call_id if call_id is not None else self._new_call_id()
        check_call_id(assigned)
        with self._lock:
            position = self._positions.get(assigned)
            if position is not None:
                existing = self._calls[position]
                if (
                    call_id is not None
                    and self._encoded_by_id[assigned] == encoded
                    and (
                        body.disposition is None
                        or body.disposition == existing.disposition
                    )
                ):
                    return Receipt(existing, False)
                raise RejectedError(
                    "call_conflict",
                    f"call_id '{assigned}' is already recorded with another body",
                )
            record = CallRecord(
                **{name: getattr(body, name) for name in (*BODY_FIELDS, "disposition")},
                call_id=assigned,
                complete=True,
                provenance=provenance,
                timestamp=self._clock(),
                turn=self._current_turn(),
            )
            self._positions[assigned] = len(self._calls)
            self._calls.append(record)
            self._encoded_by_id[assigned] = encoded
            return Receipt(record, True)


@dataclass(frozen=True, slots=True)
class CallChannel:
    """A recording capability that fixes provenance independently of call data."""

    ledger: CallLedger
    provenance: Provenance

    def __post_init__(self) -> None:
        if type(self.provenance) is not str or self.provenance not in PROVENANCES:
            raise ValueError("unknown call provenance")

    def record_call(
        self,
        *,
        call_id: str | None = None,
        function: str,
        args: dict,
        result: str,
        error: str | None = None,
        blocked: bool = False,
        disposition: Disposition | None = None,
        structured_content: dict | None = None,
        injection_outcomes: list | None = None,
    ) -> Receipt:
        """Record a complete call, or return the first record for an identical replay.

        An existing ID with another body is refused. A replay may leave the
        disposition out, but one it names must be the first record's. Validation
        and lossless encoding run before taking the writer lock.
        """
        check_call_id(call_id)
        body = CallBody.build(
            function=function,
            args=args,
            result=result,
            error=error,
            blocked=blocked,
            disposition=disposition,
            structured_content=structured_content,
            injection_outcomes=injection_outcomes,
        )
        return self.ledger._insert(self.provenance, call_id, body, body.encoded())
