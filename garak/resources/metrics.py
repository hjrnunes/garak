# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a detector's results mean, and why an output has no result.

A detector scores an output with a number, or with ``None`` when it cannot.
The evaluator counts every ``None`` as unscored and keeps it out of the rate.
A detector can say why, so that the report declares each exclusion instead of
hiding it. An output is *not applicable* when the check does not apply to it at
all, and *unknown* otherwise.
"""

from typing import Optional, Tuple

# key in ``attempt.notes`` mapping a detector name, as it appears in
# ``attempt.detector_results``, to ``{"reason": str, "not_applicable": bool}``
UNSCORED_NOTE = "unscored"


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
    if not isinstance(attempt.notes, dict):
        attempt.notes = {}
    marks = attempt.notes.get(UNSCORED_NOTE)
    if not isinstance(marks, dict):
        marks = attempt.notes[UNSCORED_NOTE] = {}
    marks[detector_name] = {"reason": reason, "not_applicable": not_applicable}


def unscored_mark(attempt, detector_name: str) -> Tuple[Optional[str], bool]:
    """Read what :func:`mark_unscored` recorded; unreadable notes read as unknown

    :returns: the reason, or None if none was recorded, and whether the check
        was not applicable
    """
    notes = attempt.notes if isinstance(attempt.notes, dict) else {}
    marks = notes.get(UNSCORED_NOTE)
    mark = marks.get(detector_name) if isinstance(marks, dict) else None
    if not isinstance(mark, dict):
        return None, False
    reason = mark.get("reason")
    return (
        reason if isinstance(reason, str) else None,
        mark.get("not_applicable") is True,
    )
