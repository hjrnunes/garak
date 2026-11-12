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

A probe can run each attacked chain beside a benign baseline of the same task.
The two attempts of such a pair share an id, and each says which role it has.
A baseline is never an attack, so a security detector leaves it unscored. The
drop in utility is counted over the pairs whose two sides both have a result.
"""

import json
from typing import Callable, Iterable, Optional, Tuple

SECURITY = "security"
UTILITY = "utility"
METRIC_KINDS = (SECURITY, UTILITY)

ATTACKED = "attacked"
BASELINE = "baseline"
PAIR_ROLES = (ATTACKED, BASELINE)
_PAIR_COUNTS = ("pairs", "valid", "invalid", "benign_completed", "attacked_completed")
BASELINE_SECURITY = "baseline_security"  # why a security detector skips a baseline

# keys in ``attempt.notes``, each mapping a detector name, as it appears in
# ``attempt.detector_results``, to what is known about that detector's results
UNSCORED_NOTE = "unscored"  # {"reason": str, "not_applicable": bool}
KIND_NOTE = "metric_kind"  # a metric kind; written only for a non-security kind
PAIR_NOTE = "utility_pair"  # {"id": str, "role": str}, one per attempt
_EVALUATION_ENTRIES = (
    "eval",
    "utility_eval",
    "probe_summary",
)  # written by the evaluator


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

    A security detector needs no record, because security is what the evaluator
    assumes, but its run replaces the record of an earlier run of the same
    detector as utility.

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
        return
    marks = attempt.notes.get(KIND_NOTE) if isinstance(attempt.notes, dict) else None
    if isinstance(marks, dict):
        marks.pop(detector_name, None)
        if not marks:
            del attempt.notes[KIND_NOTE]


def kind_in_notes(notes, detector_name: str) -> str:
    """Read the kind that :func:`mark_kind` recorded in a notes mapping, such as
    the ``notes`` of an attempt entry in a report; anything else reads as security
    """
    marks = notes.get(KIND_NOTE) if isinstance(notes, dict) else None
    recorded = marks.get(detector_name) if isinstance(marks, dict) else None
    return UTILITY if recorded == UTILITY else SECURITY


def metric_kind(attempt, detector_name: str) -> str:
    """Read what :func:`mark_kind` recorded; anything else reads as security"""
    return kind_in_notes(attempt.notes, detector_name)


def mark_pair(attempt, pair_id: str, role: str) -> None:
    """Record that an attempt is one side of a pair

    :param attempt: the attempt, which keeps the record in its notes
    :type attempt: garak.attempt.Attempt
    :param pair_id: names the pair; both sides of a pair carry the same id
    :type pair_id: str
    :param role: one of ``PAIR_ROLES``
    :type role: str
    :raises ValueError: if the id is not non-empty text or the role is unknown
    """
    if type(pair_id) is not str or not pair_id:
        raise ValueError("a pair id must be non-empty text")
    if role not in PAIR_ROLES:
        raise ValueError(f"unknown pair role {role!r}, expected one of {PAIR_ROLES}")
    if not isinstance(attempt.notes, dict):
        attempt.notes = {}
    attempt.notes[PAIR_NOTE] = {"id": pair_id, "role": role}


def pair_of(attempt) -> Optional[Tuple[str, str]]:
    """Read what :func:`mark_pair` recorded; unreadable notes read as no pair

    :returns: the pair id and the role, or None if the attempt is in no pair
    """
    notes = attempt.notes if isinstance(attempt.notes, dict) else {}
    mark = notes.get(PAIR_NOTE)
    if not isinstance(mark, dict):
        return None
    pair_id, role = mark.get("id"), mark.get("role")
    if type(pair_id) is not str or not pair_id or role not in PAIR_ROLES:
        return None
    return pair_id, role


def is_baseline(attempt) -> bool:
    """True if :func:`mark_pair` made the attempt the benign side of a pair"""
    pair = pair_of(attempt)
    return pair is not None and pair[1] == BASELINE


def _side(
    held: list, role: str, detector_name: str, completed: Callable[[float], bool]
) -> Tuple[Optional[bool], Optional[str]]:
    """Whether the one attempt on a side completed its task, or why it does not count"""
    if len(held) != 1:
        return None, "unpaired"
    attempt = held[0]
    scores = attempt.detector_results.get(detector_name)
    if not isinstance(scores, list) or len(scores) != 1:
        return None, f"{role}:outputs"
    if scores[0] is None:
        reason, _ = unscored_mark(attempt, detector_name)
        return None, f"{role}:{reason or 'unscored'}"
    return completed(float(scores[0])), None


def pair_summary(
    attempts: Iterable, detector_name: str, completed: Callable[[float], bool]
) -> Optional[dict]:
    """Count a utility detector's results over the pairs the attempts hold

    A pair is valid when each side is one attempt with one scored output. Any
    other pair is invalid, counted under the reason of its first side that has
    no result, and takes no part in the rates, so a result that could not be
    decided never reads as a task that failed.

    :param attempts: the attempts a utility detector scored
    :param detector_name: the detector's key in ``attempt.detector_results``
    :type detector_name: str
    :param completed: tells whether a score is a completed task
    :returns: the counts and rates over valid pairs (a rate is None when no
        pair is valid; ``drop`` is the benign rate less the rate under attack),
        or None if no attempt is in a pair
    """
    pairs: dict = {}
    for attempt in attempts:
        pair = pair_of(attempt)
        if pair is not None:
            sides = pairs.setdefault(pair[0], {role: [] for role in PAIR_ROLES})
            sides[pair[1]].append(attempt)
    if not pairs:
        return None
    valid = benign = attacked = 0
    invalid_reasons: dict = {}
    for sides in pairs.values():
        results = {}
        reason = None
        for role in PAIR_ROLES:
            results[role], why = _side(sides[role], role, detector_name, completed)
            reason = reason or why
        if reason is not None:
            invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
            continue
        valid += 1
        benign += results[BASELINE]
        attacked += results[ATTACKED]
    summary = {
        "pairs": len(pairs),
        "valid": valid,
        "invalid": len(pairs) - valid,
        "benign_completed": benign,
        "attacked_completed": attacked,
    }
    if invalid_reasons:
        summary["invalid_reasons"] = dict(sorted(invalid_reasons.items()))
    return _with_rates(summary)


def _with_rates(counts: dict) -> dict:
    """Set the rates of pair counts; with no valid pair they are unknown"""
    valid = counts["valid"]
    benign, attacked = counts["benign_completed"], counts["attacked_completed"]
    counts["benign_utility"] = benign / valid if valid else None
    counts["utility_under_attack"] = attacked / valid if valid else None
    counts["drop"] = (benign - attacked) / valid if valid else None
    return counts


def _checked_counts(summary) -> Tuple[dict, dict]:
    """The counts and invalid reasons of a pair summary that can be one

    :raises ValueError: if a count is missing or is not a non-negative integer,
        if the valid and invalid pairs do not make up the pairs, if a side
        completed more tasks than there are valid pairs, or if the reasons are
        not a mapping of text to such counts
    """
    if not isinstance(summary, dict):
        raise ValueError("a pair summary must be a mapping")
    counts = {}
    for key in _PAIR_COUNTS:
        count = summary.get(key)
        if type(count) is not int or count < 0:
            raise ValueError(f"pair summary {key!r} must be a non-negative integer")
        counts[key] = count
    if counts["valid"] + counts["invalid"] != counts["pairs"]:
        raise ValueError("pair summary valid and invalid pairs do not make up pairs")
    for key in ("benign_completed", "attacked_completed"):
        if counts[key] > counts["valid"]:
            raise ValueError(f"pair summary {key!r} is more than its valid pairs")
    reasons = summary.get("invalid_reasons", {})
    if not isinstance(reasons, dict) or any(
        type(reason) is not str or type(count) is not int or count < 0
        for reason, count in reasons.items()
    ):
        raise ValueError("pair summary invalid_reasons must map text to counts")
    return counts, reasons


def pool_pairs(pooled: Optional[dict], summary: dict) -> dict:
    """Add the counts of a :func:`pair_summary` to pooled counts

    The rates are computed again over the pooled valid pairs.

    :param pooled: what this function returned before, or None to begin
    :param summary: a summary, such as one read from a report
    :returns: the pooled counts, a new mapping
    :raises ValueError: if the summary cannot be one, so that a report that
        holds an impossible count is refused and not pooled into a rate
    """
    total = {key: 0 for key in _PAIR_COUNTS}
    reasons: dict = {}
    for source in (summary,) if pooled is None else (pooled, summary):
        counts, source_reasons = _checked_counts(source)
        for key in _PAIR_COUNTS:
            total[key] += counts[key]
        for reason, count in source_reasons.items():
            reasons[reason] = reasons.get(reason, 0) + count
    if reasons:
        total["invalid_reasons"] = dict(sorted(reasons.items()))
    return _with_rates(total)


class _EvaluationCounter:
    """Numbers the evaluations of a report in the order that the report lists them

    A report lists the completed attempts of an evaluation, then the entries
    that the evaluator wrote for it. The first completed attempt after such an
    entry belongs to the next evaluation.
    """

    def __init__(self):
        self.index = 0
        self._evaluated = False

    def see(self, record) -> None:
        if not isinstance(record, dict):
            return
        entry_type = record.get("entry_type")
        if entry_type in _EVALUATION_ENTRIES:
            self._evaluated = True
        elif entry_type == "attempt" and record.get("status") == 2 and self._evaluated:
            self.index += 1
            self._evaluated = False


class UtilityScope:
    """Which probe and detector pairs a report treats as utility, evaluation by evaluation

    The evaluator reports a detector as utility for all the attempts it
    evaluates together if any of them carries the utility mark. A tool that
    reads a report one attempt at a time would read an attempt without the mark
    as an attack, so it makes a scope before it reads, passes each record it
    reads to :meth:`observe`, and asks :meth:`covers` about each completed
    attempt. Evaluations of the same probe and detector, as an aggregate of runs
    holds, stay apart.

    :param report_path: path of a report JSONL file
    :type report_path: str
    """

    def __init__(self, report_path: str):
        self._evaluations = []  # per evaluation, the (probe, detector) pairs marked
        self._reading = _EvaluationCounter()
        counter = _EvaluationCounter()
        with open(report_path, "r", encoding="utf-8") as report_file:
            for line in report_file:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                counter.see(record)
                if not isinstance(record, dict):
                    continue
                if record.get("entry_type") != "attempt" or record.get("status") != 2:
                    continue
                results = record.get("detector_results")
                for detector_name in results if isinstance(results, dict) else ():
                    if kind_in_notes(record.get("notes"), detector_name) != UTILITY:
                        continue
                    while len(self._evaluations) <= counter.index:
                        self._evaluations.append(set())
                    self._evaluations[counter.index].add(
                        (record.get("probe_classname"), detector_name)
                    )

    def observe(self, record) -> None:
        """Note a record of the report that the tool has read, in report order"""
        self._reading.see(record)

    def covers(self, probe_classname: str, detector_name: str) -> bool:
        """True if the evaluation that the observed records have reached treats
        the detector as utility for the probe"""
        index = self._reading.index
        return (
            index < len(self._evaluations)
            and (probe_classname, detector_name) in self._evaluations[index]
        )
