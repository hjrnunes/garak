# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a detector's results mean, and why an output has no result.

A detector scores an output with a number, or with ``None`` when it cannot.
The evaluator counts every ``None`` as unscored and keeps it out of the rate.
A detector can say why, so that the report declares each exclusion instead of
hiding it. An output is *not applicable* when the check does not apply to it at
all, and *unknown* otherwise.

A detector also has a metric kind. A score of 1.0 always means that the
behaviour the detector looks for occurred. For a ``security`` detector that
behaviour is a successful attack; for a ``utility`` detector it is a completed
legitimate task. The evaluator reports the two kinds apart so that a utility
result cannot change an attack success rate.
"""

from typing import Optional, Tuple

SECURITY = "security"
UTILITY = "utility"
METRIC_KINDS = (SECURITY, UTILITY)

# keys in ``attempt.notes``, each mapping a detector name, as it appears in
# ``attempt.detector_results``, to what is known about that detector's results
UNSCORED_NOTE = "unscored"  # {"reason": str, "not_applicable": bool}
KIND_NOTE = "metric_kind"  # a metric kind; written only for a non-security kind


def _marks(attempt, note: str) -> dict:
    """The mapping under ``note``, replacing notes that cannot hold one"""
    if not isinstance(attempt.notes, dict):
        attempt.notes = {}
    marks = attempt.notes.get(note)
    if not isinstance(marks, dict):
        marks = attempt.notes[note] = {}
    return marks


def _read(attempt, note: str, detector_name: str):
    """The entry for a detector under ``note``, or None when unreadable"""
    notes = attempt.notes if isinstance(attempt.notes, dict) else {}
    marks = notes.get(note)
    return marks.get(detector_name) if isinstance(marks, dict) else None


def mark_unscored(
    attempt, detector_name: str, reason: str, not_applicable: bool = False
) -> None:
    """Record why a detector left the outputs of an attempt without a score

    :param attempt: the attempt the detector ran on
    :type attempt: garak.attempt.Attempt
    :param detector_name: the detector's key in ``attempt.detector_results``
    :type detector_name: str
    :param reason: a short identifier, counted by the evaluator
    :type reason: str
    :param not_applicable: True if the check does not apply to this attempt
    :type not_applicable: bool
    """
    _marks(attempt, UNSCORED_NOTE)[detector_name] = {
        "reason": reason,
        "not_applicable": not_applicable,
    }


def unscored_mark(attempt, detector_name: str) -> Tuple[Optional[str], bool]:
    """Read what :func:`mark_unscored` recorded; unreadable notes read as unknown

    :returns: the reason, or None if none was recorded, and whether the check
        was not applicable
    """
    mark = _read(attempt, UNSCORED_NOTE, detector_name)
    if not isinstance(mark, dict):
        return None, False
    reason = mark.get("reason")
    return (
        reason if isinstance(reason, str) else None,
        mark.get("not_applicable") is True,
    )


def mark_kind(attempt, detector_name: str, kind: str) -> None:
    """Record the metric kind of a detector that ran on an attempt

    A security detector needs no record: security is what the evaluator
    assumes.

    :param attempt: the attempt the detector ran on
    :type attempt: garak.attempt.Attempt
    :param detector_name: the detector's key in ``attempt.detector_results``
    :type detector_name: str
    :param kind: one of ``METRIC_KINDS``
    :type kind: str
    :raises ValueError: if the kind is not a metric kind
    """
    if kind not in METRIC_KINDS:
        raise ValueError(
            f"unknown metric kind {kind!r}, expected one of {METRIC_KINDS}"
        )
    if kind != SECURITY:
        _marks(attempt, KIND_NOTE)[detector_name] = kind


def metric_kind(attempt, detector_name: str) -> str:
    """Read what :func:`mark_kind` recorded; anything else reads as security"""
    return UTILITY if _read(attempt, KIND_NOTE, detector_name) == UTILITY else SECURITY
