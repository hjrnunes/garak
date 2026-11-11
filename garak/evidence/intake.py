"""Call intake rules: what a start or a finish does with the record it names.

These are the collector's call decisions, split out so the collector holds
the stage machine and the seal, and this module holds the ledger rules a
write against a call obeys:

* a start appends an unfinished record the finish completes in place;
* a replay at any stage returns the first record and inserts nothing;
* another body under the same ID is ``call_conflict`` until the seal, and a
  late fact after it;
* the disposition rule on completion: the start's and the finish's
  dispositions disagree only when both are named and differ; a start that
  named none lets the finish name one, which the completed record then holds;
  a finish that omits it keeps the start's.
"""

from __future__ import annotations

import enum
from typing import Any

from garak.evidence.facts import FUNCTION_CALLS
from garak.evidence.ledger import Receipt
from garak.evidence.records import (
    BODY_FIELDS,
    REFUSED_DISPOSITIONS,
    CallBody,
    CallRecord,
    InvalidRequestError,
    RejectedError,
    check_call_id,
)


class Stage(enum.StrEnum):
    """Where a collector is in its lifecycle."""

    OPEN = "open"
    CLOSING = "closing"
    COLLECTING = "collecting"
    SEALED = "sealed"


#: The fields a start compares: what the call is, not what it returned.
START_FIELDS: tuple[str, ...] = ("function", "args")
# What a finish decides to do with the record it names.
REPLAY, COMPLETE, LATE = "replay", "complete", "late"


def start_call(
    collector: Any,
    provenance: str,
    call_id: str,
    *,
    function: str,
    args: dict,
    disposition: str | None,
) -> Receipt:
    """Admit, replay or refuse a call's start against the collector's stage."""
    check_call_id(call_id)
    if call_id is None:
        raise InvalidRequestError("a start needs a call_id to pair with its finish")
    arrival = collector._monotonic()
    at = collector._clock()
    body = CallBody.build_start(function=function, args=args, disposition=disposition)
    if body.disposition in REFUSED_DISPOSITIONS:
        raise RejectedError(
            "invalid_disposition",
            f"A {disposition} call never runs, so it isn't started: "
            "its finish records it whole",
        )
    start_encoded = body.encoded(START_FIELDS)
    with collector._condition:
        collector._on_time(arrival)
        stage = collector._stage
        existing = collector._ledger.get(call_id) if call_id is not None else None
        if existing is not None:
            if _start_replay(
                collector,
                call_id,
                stage,
                body=body,
                start_encoded=start_encoded,
                existing=existing,
            ):
                return Receipt(existing, False)
        elif stage is Stage.OPEN:
            return collector._ledger.start(provenance, call_id, body, start_encoded)
        collector._record_late(
            "start_call",
            provenance,
            at,
            evidence=(FUNCTION_CALLS,),
            payload={
                "call_id": call_id,
                "function": function,
                "args": args,
                "disposition": disposition,
            },
            call_id=call_id,
        )
        raise collector._late_error(stage)


def _start_replay(
    collector: Any,
    call_id: str,
    stage: Stage,
    *,
    body: CallBody,
    start_encoded: str,
    existing: CallRecord,
) -> bool:
    """Whether a start replays a record its ID already names.

    A replay may say less than the stored record but never more: it agrees
    when it names no disposition or the stored one.
    """
    conflict = (
        f"call_id '{call_id}' is already recorded with another "
        "function, args or disposition"
    )
    if collector._ledger.start_encoded(call_id) != start_encoded:
        if stage is not Stage.SEALED:
            raise RejectedError("call_conflict", conflict)
        return False
    if body.disposition is not None and body.disposition != existing.disposition:
        if stage is not Stage.SEALED:
            raise RejectedError("call_conflict", conflict)
        return False
    return True


def finish_call(
    collector: Any,
    provenance: str,
    call_id: str | None,
    **fields: Any,
) -> Receipt:
    """Admit, complete, replay or refuse a call's finish against the stage."""
    check_call_id(call_id)
    arrival = collector._monotonic()
    at = collector._clock()
    body = CallBody.build(**fields)
    whole_encoded = body.encoded(BODY_FIELDS)
    with collector._condition:
        collector._on_time(arrival)
        stage = collector._stage
        existing = collector._ledger.get(call_id) if call_id is not None else None
        if existing is not None:
            action = _finish_action(
                collector,
                stage,
                call_id,
                body=body,
                whole_encoded=whole_encoded,
                existing=existing,
            )
            if action is REPLAY:
                return Receipt(existing, False)
            if action is COMPLETE:
                receipt = collector._ledger.complete(
                    existing.call_id, body, whole_encoded
                )
                collector._condition.notify_all()
                return receipt
        elif stage is Stage.OPEN:
            return collector._ledger._insert(provenance, call_id, body, whole_encoded)
        collector._record_late(
            "finish_call",
            provenance,
            at,
            evidence=(FUNCTION_CALLS,),
            payload={**fields, "call_id": call_id},
            call_id=call_id,
        )
        raise collector._late_error(stage)


def _finish_action(
    collector: Any,
    stage: Stage,
    call_id: str | None,
    *,
    body: CallBody,
    whole_encoded: str,
    existing: CallRecord,
) -> str:
    """Decide what a finish naming an existing record does with it."""
    dispositions_agree = (
        body.disposition is None
        or existing.disposition is None
        or body.disposition == existing.disposition
    )

    def conflict(detail: str) -> None:
        if stage is not Stage.SEALED:
            raise RejectedError("call_conflict", detail)

    if existing.complete:
        if (
            call_id is not None
            and collector._ledger.encoded(existing.call_id) == whole_encoded
            and (body.disposition is None or body.disposition == existing.disposition)
        ):
            return REPLAY
        conflict(f"call_id '{call_id}' is already recorded with another body")
        return LATE
    if collector._ledger.start_encoded(existing.call_id) != body.encoded(START_FIELDS):
        conflict(f"call_id '{call_id}' was started with another function or args")
        return LATE
    if not dispositions_agree:
        conflict(
            f"call_id '{call_id}' was started {existing.disposition}, so its "
            f"finish can't name {body.disposition}"
        )
        return LATE
    if (
        existing.disposition is not None
        and (existing.disposition in REFUSED_DISPOSITIONS) != body.blocked
    ):
        conflict(
            f"call_id '{call_id}' was started {existing.disposition}, so its "
            "finish can't disagree about being blocked"
        )
        return LATE
    if stage not in (Stage.OPEN, Stage.CLOSING):
        return LATE
    return COMPLETE
