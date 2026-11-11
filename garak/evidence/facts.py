"""Late facts, capture failures, turns and observations: the collector's records.

These values are the collector's own evidence records, distinct from call
records: a late fact is a write a stage refused, a capture failure is a record
a layer could not deliver, a turn is one message and its reply, and an
observation is one source's collected data with the status of each part.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from garak.evidence.codec import encode
from garak.evidence.records import (
    InvalidRequestError,
    RejectedError,
    freeze,
    thaw,
)

#: The statuses an evidence part may report.
EVIDENCE_STATUSES = frozenset({"complete", "incomplete", "failed", "not_collected"})
#: The evidence key naming the call capture.
FUNCTION_CALLS = "function_calls"
#: The evidence key naming the conversation.
TURNS = "turns"
#: The evidence key naming the environment.
ENVIRONMENT = "environment"
#: The evidence key naming the agent's final output.
AGENT_OUTPUT = "agent_output"
#: The provenances whose unfinished calls mark the call capture incomplete:
#: a layer on the agent's side may fail open, so its report is the only one.
SESSION_PROVENANCE = frozenset({"agent", "interceptor"})
#: The reason length used when the shared recorder vocabulary is not installed.
FALLBACK_REASON_MAX_LENGTH = 2048
_LATE_KINDS = frozenset({"closing", "sealed"})
_PHASES = frozenset({"start", "finish"})


def cut_reason(reason: str) -> str:
    """Cut a capture failure's reason at the length the shared vocabulary allows.

    The shared recorder package is optional at import time; without it the
    documented length stands.
    """
    try:
        from garak.evidence.codec import MissingSnapshotCodecError, recorder_codec

        limit = recorder_codec().CAPTURE_REASON_MAX_LENGTH
    except MissingSnapshotCodecError:
        limit = FALLBACK_REASON_MAX_LENGTH
    return reason[:limit]


class LateFactError(RejectedError):
    """A refusal whose write the collector kept as a late fact.

    ``late_fact`` is True and ``kind`` is ``"closing"`` or ``"sealed"``; the
    evidence keys the kept fact marks incomplete are on the fact itself, which
    the collector's ``late_facts`` tuple exposes.
    """

    late_fact = True

    def __init__(self, kind: str, detail: str):
        if kind not in _LATE_KINDS:
            raise ValueError(f"unknown late-fact kind '{kind}'")
        super().__init__(kind, detail)


@dataclass(frozen=True, slots=True)
class LateFact:
    """A write a stage refused, kept once per distinct write.

    ``at`` is the refusal's timestamp; ``evidence`` names the evidence-status
    keys the fact marks incomplete; ``payload`` is the write as sent, frozen
    JSON data.
    """

    event: str
    stage: str
    by: str
    at: str
    evidence: tuple[str, ...]
    payload: Mapping[str, Any]
    call_id: str | None = None

    def __post_init__(self) -> None:
        if type(self.evidence) is not tuple:
            object.__setattr__(self, "evidence", tuple(self.evidence))

    def identity(self) -> tuple[str, str, str | None, str]:
        """What makes this write the same as another: event, writer, call, payload."""
        return (self.event, self.by, self.call_id, encode(thaw(self.payload)))

    def evidence_keys(self) -> frozenset[str]:
        """The evidence-status keys this fact marks incomplete."""
        return frozenset(self.evidence)


@dataclass(frozen=True, slots=True)
class CaptureFailure:
    """A call record the capture layer could not deliver.

    Identity is ``(call_id, function, phase)``: a later report of the same
    identity replays, whatever its reason. A capture failure marks
    ``function_calls`` incomplete; it never adds, changes or removes a call
    record.
    """

    call_id: str
    function: str
    phase: str
    reason: str
    stage: str
    by: str
    at: str

    def __post_init__(self) -> None:
        if self.phase not in _PHASES:
            raise ValueError(f"capture phase '{self.phase}' is unknown")
        object.__setattr__(self, "reason", cut_reason(self.reason))

    def identity(self) -> tuple[str, str, str]:
        """What makes this failure the same as another."""
        return (self.call_id, self.function, self.phase)

    def to_json(self) -> dict[str, Any]:
        """Export fresh JSON values in the sealed snapshot's field order."""
        return {
            "call_id": self.call_id,
            "function": self.function,
            "phase": self.phase,
            "reason": self.reason,
            "stage": self.stage,
            "by": self.by,
            "at": self.at,
        }


@dataclass(frozen=True, slots=True)
class Turn:
    """One user message in the conversation and the agent's reply to it."""

    index: int
    message: str
    reply: str | None
    started_at: str
    ended_at: str | None

    def to_json(self) -> dict[str, Any]:
        """Export fresh JSON values in the sealed snapshot's field order."""
        return {
            "index": self.index,
            "message": self.message,
            "reply": self.reply,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }


@dataclass(frozen=True, slots=True)
class Observation:
    """One source's observed data, replaced whole by each later record.

    ``evidence`` maps each part the source reports to its status; the seal
    flattens the keys to ``<source>.<part>``. The statuses are copied into a
    read-only mapping, so neither the caller's dict nor a returned
    observation can change what the seal reads.
    """

    source: str
    observed: Mapping[str, Any]
    evidence: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence)))

    def flattened(self) -> dict[str, str]:
        """This source's evidence statuses, flattened to ``<source>.<part>``."""
        return {
            f"{self.source}.{part}": status for part, status in self.evidence.items()
        }


def json_payload(data: Mapping[str, Any]) -> Mapping[str, Any]:
    """Freeze a late fact's payload, refusing data JSON cannot hold."""
    return freeze(dict(data))


def require_text(value: Any, what: str) -> str:
    """Refuse anything but exact non-empty text, naming what it is."""
    if type(value) is not str or not value:
        raise InvalidRequestError(f"{what} must be non-empty text")
    return value
