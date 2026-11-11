"""The immutable sealed snapshot, named by the shared digest of its content."""

from __future__ import annotations

import copy
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from garak.evidence.facts import CaptureFailure, Turn
from garak.evidence.records import CallRecord, freeze, thaw


class Snapshot:
    """What the seal fixes: the shared vocabulary's content, frozen, and its digest.

    ``snapshot_id`` is ``midojo_spec.snapshot.snapshot_digest`` of the content,
    computed once at construction; no caller chooses it. ``to_json()`` returns
    a fresh, fully detached JSON object each call, so mutating what it returns
    changes neither the digest nor a later read. ``verified()`` recomputes the
    digest from what the snapshot holds now and says whether it still matches.
    The snapshot may be pickled and copied: it is a pure frozen value, and
    copying it returns it unchanged.
    """

    __slots__ = ("_content", "snapshot_id")

    def __init__(
        self,
        *,
        evaluation_id: str,
        agent_output: str,
        pre_environment: Mapping[str, Any] | None,
        environment: Mapping[str, Any] | None,
        function_calls: Iterable[CallRecord],
        turns: Iterable[Turn],
        observations: Mapping[str, Mapping[str, Any]],
        evidence_status: Mapping[str, str],
        sealed_at: str,
        capture_failures: Iterable[CaptureFailure],
        run_id: str | None = None,
        user_task_id: str | None = None,
        injection_task_id: str | None = None,
    ):
        content: dict[str, Any] = {
            "evaluation_id": evaluation_id,
            "run_id": run_id,
            "user_task_id": user_task_id,
            "injection_task_id": injection_task_id,
            "agent_output": agent_output,
            "pre_environment": (
                freeze(dict(pre_environment)) if pre_environment is not None else None
            ),
            "environment": (
                freeze(dict(environment)) if environment is not None else None
            ),
            "function_calls": tuple(function_calls),
            "turns": tuple(turns),
            "observations": MappingProxyType(
                {
                    source: freeze(dict(observed))
                    for source, observed in observations.items()
                }
            ),
            "evidence_status": MappingProxyType(dict(evidence_status)),
            "sealed_at": sealed_at,
            "capture_failures": tuple(capture_failures),
        }
        object.__setattr__(self, "_content", content)
        object.__setattr__(self, "snapshot_id", self._digest(content))

    def _digest(self, content: Mapping[str, Any]) -> str:
        from garak.evidence.codec import snapshot_codec

        return snapshot_codec().snapshot_digest(self._json_object(content))

    def _json_object(self, content: Mapping[str, Any] | None = None) -> dict[str, Any]:
        held = self._content if content is None else content
        return {
            "evaluation_id": held["evaluation_id"],
            "run_id": held["run_id"],
            "user_task_id": held["user_task_id"],
            "injection_task_id": held["injection_task_id"],
            "agent_output": held["agent_output"],
            "pre_environment": thaw(held["pre_environment"]),
            "environment": thaw(held["environment"]),
            "function_calls": [call.to_json() for call in held["function_calls"]],
            "turns": [turn.to_json() for turn in held["turns"]],
            "observations": {
                source: thaw(observed)
                for source, observed in held["observations"].items()
            },
            "evidence_status": dict(held["evidence_status"]),
            "sealed_at": held["sealed_at"],
            "capture_failures": [
                failure.to_json() for failure in held["capture_failures"]
            ],
        }

    def to_json(self) -> dict[str, Any]:
        """Return a fresh, detached JSON object of everything the digest names."""
        return copy.deepcopy(self._json_object())

    def verified(self) -> bool:
        """Recompute the digest and say whether it still names this content."""
        return self._digest(self._content) == self.snapshot_id

    def __getattr__(self, name: str) -> Any:
        content = object.__getattribute__(self, "_content")
        if name in content:
            return content[name]
        raise AttributeError(f"a sealed snapshot has no field '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("a sealed snapshot is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("a sealed snapshot is immutable")

    def __reduce__(self) -> Any:
        """A snapshot is a pure frozen value: pickle it through its JSON form."""
        return (rebuild_snapshot, (self.to_json(),))

    def __copy__(self) -> Snapshot:
        """A snapshot is immutable, so a copy is the snapshot itself."""
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> Snapshot:
        """A snapshot is immutable, so a copy is the snapshot itself."""
        return self

    def __repr__(self) -> str:
        return f"Snapshot(snapshot_id={self.snapshot_id!r})"


def rebuild_snapshot(data: Mapping[str, Any]) -> Snapshot:
    """Rebuild a snapshot from the JSON ``to_json()`` returned, digest intact.

    Records are rebuilt through ``CallRecord.from_json``, so an unfinished
    call comes back with ``complete`` false and ``result`` None, as it sealed.
    """
    return Snapshot(
        evaluation_id=data["evaluation_id"],
        agent_output=data["agent_output"],
        pre_environment=data["pre_environment"],
        environment=data["environment"],
        function_calls=[CallRecord.from_json(call) for call in data["function_calls"]],
        turns=[Turn(**turn) for turn in data["turns"]],
        observations={
            source: dict(observed) for source, observed in data["observations"].items()
        },
        evidence_status=data["evidence_status"],
        sealed_at=data["sealed_at"],
        capture_failures=[
            CaptureFailure(
                call_id=failure["call_id"],
                function=failure["function"],
                phase=failure["phase"],
                reason=failure["reason"],
                stage=failure["stage"],
                by=failure["by"],
                at=failure["at"],
            )
            for failure in data["capture_failures"]
        ],
        run_id=data["run_id"],
        user_task_id=data["user_task_id"],
        injection_task_id=data["injection_task_id"],
    )
