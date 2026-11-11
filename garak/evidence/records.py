"""Frozen call bodies and sealed-snapshot-compatible call records."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping

from garak.exception import GarakException
from garak.evidence.codec import encode, injection_codec

CALL_ID_MAX_LENGTH = 256
MAX_NESTING_DEPTH = 100
Provenance = Literal["host", "interceptor", "agent"]
PROVENANCES = frozenset({"host", "interceptor", "agent"})
Disposition = Literal["forwarded", "denied", "simulated", "blocked"]
DISPOSITIONS = frozenset({"forwarded", "denied", "simulated", "blocked"})
REFUSED_DISPOSITIONS = frozenset({"denied", "blocked"})
BODY_FIELDS = (
    "function",
    "args",
    "result",
    "error",
    "blocked",
    "structured_content",
    "injection_outcomes",
)
OUTCOME_FIELDS = frozenset(
    {"instruction_index", "slot_key", "occurrence", "status", "detail"}
)
FACT_FIELDS = frozenset({"selected", "transformation", "emitted"})
PLACEMENT_FIELDS = frozenset({"payload_digest", "placed_digest", "slot"})
OUTCOME_STATUSES = frozenset(
    {
        "applied",
        "partial",
        "fallback",
        "unmatched",
        "no_result",
        "too_large",
        "skipped_occurrence",
    }
)
PLACED_STATUSES = frozenset({"applied", "partial", "fallback"})
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
# Integers wider than this are test-converted to text: the interpreter refuses
# some, and a value the digest cannot encode must never reach a record.
_WIDE_INT_BITS = 1024


class RejectedError(GarakException):
    """A well-formed call violates a ledger rule or carries unsupported data."""

    def __init__(self, kind: str, detail: str):
        self.kind = kind
        self.detail = detail
        super().__init__(f"{kind}: {detail}")


class InvalidRequestError(ValueError):
    """A request has a missing field or a field of the wrong shape or type."""


def _freeze(value: Any, depth: int) -> Any:
    kind = type(value)
    if depth > MAX_NESTING_DEPTH:
        raise RejectedError("unsupported_value", f": {kind.__name__} (nesting limit)")
    if kind is int and value.bit_length() > _WIDE_INT_BITS:
        try:
            str(value)
        except ValueError as exc:
            raise RejectedError(
                "unsupported_value", ": int (too many digits to encode)"
            ) from exc
    if value is None or kind in (str, bool, int):
        return value
    if kind is float and math.isfinite(value):
        return value
    if kind is list:
        items = []
        for index, item in enumerate(value):
            try:
                items.append(_freeze(item, depth + 1))
            except RejectedError as exc:
                raise RejectedError(exc.kind, f"[{index}]{exc.detail}") from exc
        return tuple(items)
    if kind is dict:
        items = {}
        for key, item in value.items():
            if type(key) is not str:
                raise RejectedError(
                    "unsupported_value", f"[{key!r}]: {type(key).__name__}"
                )
            try:
                items[key] = _freeze(item, depth + 1)
            except RejectedError as exc:
                raise RejectedError(exc.kind, f".{key}{exc.detail}") from exc
        return MappingProxyType(items)
    raise RejectedError("unsupported_value", f": {kind.__name__}")


def freeze(value: Any) -> Any:
    """Freeze exact JSON types recursively, preserving object order and value types.

    A refusal includes the path from the root; recursion depth also bounds cycles.
    """
    try:
        return _freeze(value, 0)
    except RejectedError as exc:
        raise RejectedError(exc.kind, f"value{exc.detail}") from exc


def thaw(value: Any) -> Any:
    """Return a fresh JSON value from a frozen value, preserving insertion order."""
    if isinstance(value, MappingProxyType):
        return {key: thaw(item) for key, item in value.items()}
    if type(value) is tuple:
        return [thaw(item) for item in value]
    return value


def _freeze_field(name: str, value: Any) -> Any:
    try:
        return _freeze(value, 0)
    except RejectedError as exc:
        raise RejectedError(exc.kind, f"{name}{exc.detail}") from exc


def _slot_shape(slot: Any) -> bool:
    if type(slot) is not dict or set(slot) != {"id", "target", "mode"}:
        return False
    try:
        decoded = injection_codec().InjectionSlot.from_data(slot)
    except ValueError:
        return False
    return encode(decoded.to_data()) == encode(slot)


def _check_outcome(item: Any, index: int) -> None:
    """Reject an outcome a verifying sealed-snapshot reader cannot interpret."""
    where = f"injection_outcomes[{index}]"
    if type(item) is not dict or not (
        OUTCOME_FIELDS <= item.keys() <= OUTCOME_FIELDS | FACT_FIELDS | PLACEMENT_FIELDS
    ):
        raise InvalidRequestError(f"{where} has invalid fields")
    if FACT_FIELDS & item.keys() and not FACT_FIELDS <= item.keys():
        raise InvalidRequestError(f"{where} has partial delivery facts")
    for field in ("instruction_index", "occurrence"):
        if type(item[field]) is not int or item[field] < 0:
            raise InvalidRequestError(f"{where}.{field} must be a non-negative int")
    if item["slot_key"] is not None and type(item["slot_key"]) is not str:
        raise InvalidRequestError(f"{where}.slot_key must be text or None")
    if type(item["status"]) is not str or item["status"] not in OUTCOME_STATUSES:
        raise InvalidRequestError(f"{where}.status is unknown")
    if type(item["detail"]) is not str:
        raise InvalidRequestError(f"{where}.detail must be text")
    if FACT_FIELDS <= item.keys():
        if type(item["selected"]) is not bool:
            raise InvalidRequestError(f"{where}.selected must be bool")
        if item["transformation"] not in (None, "identity", "json_string") or (
            item["transformation"] is not None
            and type(item["transformation"]) is not str
        ):
            raise InvalidRequestError(f"{where}.transformation is invalid")
        emitted = item["emitted"]
        if type(emitted) is not list or any(
            type(part) is not str or part not in {"content_block", "structured_content"}
            for part in emitted
        ):
            raise InvalidRequestError(f"{where}.emitted is invalid")
    if item["occurrence"] == 0 and (
        item["status"] != "unmatched"
        or (
            FACT_FIELDS <= item.keys()
            and (
                item["selected"]
                or item["transformation"] is not None
                or item["emitted"]
            )
        )
    ):
        raise InvalidRequestError(
            f"{where} occurrence zero requires unmatched with empty facts"
        )
    if PLACEMENT_FIELDS & item.keys():
        if item["status"] not in PLACED_STATUSES:
            raise InvalidRequestError(f"{where} placement requires a placed status")
        for name in ("payload_digest", "placed_digest"):
            if name in item and (
                type(item[name]) is not str or not _DIGEST.fullmatch(item[name])
            ):
                raise InvalidRequestError(f"{where}.{name} must be a SHA-256 digest")
        if "slot" in item and not _slot_shape(item["slot"]):
            raise InvalidRequestError(f"{where}.slot is invalid")


def check_call_id(call_id: Any) -> None:
    """Reject IDs other than None or exact non-empty text of bounded length."""
    if call_id is not None and (
        type(call_id) is not str or not 1 <= len(call_id) <= CALL_ID_MAX_LENGTH
    ):
        raise InvalidRequestError("call_id must be text of length 1 to 256")


@dataclass(frozen=True, slots=True, eq=False)
class CallBody:
    """One immutable, validated call payload, independent of its recording source."""

    function: str
    args: Mapping[str, Any]
    result: str | None
    error: str | None
    blocked: bool
    disposition: Disposition | None
    structured_content: Mapping[str, Any] | None
    injection_outcomes: tuple | None

    @classmethod
    def build(
        cls,
        *,
        function: str,
        args: dict,
        result: str,
        error: str | None = None,
        blocked: bool = False,
        disposition: Disposition | None = None,
        structured_content: dict | None = None,
        injection_outcomes: list | None = None,
    ) -> CallBody:
        """Validate a complete call and detach its JSON values from the caller."""
        if type(function) is not str or not function:
            raise InvalidRequestError("function must be non-empty text")
        if type(args) is not dict:
            raise InvalidRequestError("args must be an object")
        if type(result) is not str:
            raise InvalidRequestError("result must be text")
        if error is not None and type(error) is not str:
            raise InvalidRequestError("error must be text or None")
        if type(blocked) is not bool:
            raise InvalidRequestError("blocked must be bool")
        if disposition is not None and (
            type(disposition) is not str or disposition not in DISPOSITIONS
        ):
            raise InvalidRequestError("disposition is unknown")
        if structured_content is not None and type(structured_content) is not dict:
            raise InvalidRequestError("structured_content must be an object or None")
        if injection_outcomes is not None and type(injection_outcomes) is not list:
            raise InvalidRequestError("injection_outcomes must be a list or None")
        if disposition is not None and (disposition in REFUSED_DISPOSITIONS) != blocked:
            raise RejectedError(
                "invalid_disposition",
                f"A {disposition} call must have blocked={not blocked}",
            )
        frozen_outcomes = _freeze_field("injection_outcomes", injection_outcomes)
        if injection_outcomes is not None:
            for index, item in enumerate(injection_outcomes):
                _check_outcome(item, index)
        return cls(
            function,
            _freeze_field("args", args),
            result,
            error,
            blocked,
            disposition,
            _freeze_field("structured_content", structured_content),
            frozen_outcomes,
        )

    def encoded(self, fields: tuple[str, ...] = BODY_FIELDS) -> str:
        """Compare selected body fields type for type, excluding disposition by default."""
        return encode({name: thaw(getattr(self, name)) for name in fields})


@dataclass(frozen=True, slots=True, eq=False)
class CallRecord(CallBody):
    """A completed call and its immutable ID, provenance, time, and turn."""

    call_id: str
    complete: bool
    provenance: Provenance
    timestamp: str
    turn: int | None

    def to_json(self) -> dict[str, Any]:
        """Export fresh JSON values in the sealed call record's field order."""
        return {
            "call_id": self.call_id,
            "function": self.function,
            "args": thaw(self.args),
            "result": self.result,
            "error": self.error,
            "blocked": self.blocked,
            "disposition": self.disposition,
            "structured_content": thaw(self.structured_content),
            "injection_outcomes": thaw(self.injection_outcomes),
            "complete": self.complete,
            "provenance": self.provenance,
            "timestamp": self.timestamp,
            "turn": self.turn,
            "pre_environment": None,
            "post_environment": None,
        }
