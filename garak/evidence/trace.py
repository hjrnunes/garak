"""The snapshot and verdict values a trace states for one sealed evaluation.

``snapshot_value(snapshot)`` is the sealed snapshot as the MiDojo runtime
serves it: the content ``Snapshot.to_json()`` returns, with the snapshot's ID
among it, in the order of the runtime's snapshot response. Serialised with
sorted keys, it is the runtime's value byte for byte. ``trace_values(document,
snapshot)`` returns that value and the ``midojo-verdict-v1`` document that
:func:`garak.detectors.grading.grade_sealed` writes for the same snapshot, the
two members a consumer reads from a trace.

A served snapshot holds an object for ``pre_environment`` and ``environment``,
and a snapshot sealed with ``None`` for either digests differently from one
sealed with ``{}``. :class:`TraceRefused` names such a snapshot rather than
state a value the runtime never serves: seal an ``unmanaged`` runtime's
evaluation with ``pre_environment={}``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from garak.detectors.grading import grade_sealed
from garak.evidence.injection_spec import InjectionSpec
from garak.evidence.seal import Snapshot
from garak.exception import GarakException

# The runtime's snapshot response lists its members in this order.
SNAPSHOT_KEYS = (
    "evaluation_id",
    "run_id",
    "snapshot_id",
    "user_task_id",
    "injection_task_id",
    "agent_output",
    "pre_environment",
    "environment",
    "function_calls",
    "turns",
    "observations",
    "evidence_status",
    "sealed_at",
    "capture_failures",
)
_ENVIRONMENTS = ("pre_environment", "environment")


class TraceRefused(GarakException):
    """The snapshot cannot be stated as the runtime serves it, so no trace value is written."""


def snapshot_value(snapshot: Snapshot) -> dict[str, Any]:
    """``snapshot`` as the runtime serves it: a fresh JSON object, its ID included."""
    if not isinstance(snapshot, Snapshot):
        raise TraceRefused(
            f"a sealed snapshot is required, not {type(snapshot).__name__}"
        )
    if not snapshot.verified():
        raise TraceRefused("the snapshot no longer matches its digest")
    content = snapshot.to_json()
    for name in _ENVIRONMENTS:
        if content[name] is None:
            raise TraceRefused(
                f"the snapshot's {name} is null: a served snapshot holds an object there, and a null "
                "digests differently from {}"
            )
    content["snapshot_id"] = snapshot.snapshot_id
    return {key: content[key] for key in SNAPSHOT_KEYS}


def trace_values(
    document: Mapping[str, Any] | InjectionSpec, snapshot: Snapshot
) -> dict[str, dict[str, Any]]:
    """The trace's ``snapshot`` and ``verdict`` for ``snapshot`` graded with ``document``.

    ``document`` is a mapping the reader checks in full, or the
    :class:`InjectionSpec` that already passed it. Raises
    :class:`TraceRefused`, or what :func:`grade_sealed` raises, and returns
    neither value then.
    """
    value = snapshot_value(snapshot)
    return {
        "snapshot": value,
        "verdict": grade_sealed(document, snapshot).to_document(),
    }
