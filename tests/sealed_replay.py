"""Helpers for tests that record a sealed snapshot's evidence again.

A snapshot document names its turns, calls and times. A test that wants the
Garak collector to seal the same document records each fact at its recorded
time on a scripted clock and compares the result with the document.
"""

import json
from functools import lru_cache
from pathlib import Path

GRADING_CORPUS = (
    Path(__file__).parent / "_assets" / "detectors" / "grading" / "corpus.jsonl"
)

#: Members of a recorded call that the collector sets itself.
DERIVED_CALL_MEMBERS = frozenset(
    {
        "call_id",
        "complete",
        "post_environment",
        "pre_environment",
        "provenance",
        "timestamp",
        "turn",
    }
)


class ScriptedClock:
    """A clock that reads whatever time the test last set."""

    def __init__(self, stamp=None):
        self.stamp = stamp

    def at(self, stamp):
        self.stamp = stamp

    def __call__(self):
        return self.stamp


def call_fields(call):
    """The members of a snapshot's call that a channel takes, for ``record_call``."""
    return {
        name: value for name, value in call.items() if name not in DERIVED_CALL_MEMBERS
    }


@lru_cache(maxsize=1)
def _grading_corpus():
    base, documents, snapshots, cases = None, {}, {}, {}
    for line in GRADING_CORPUS.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        if entry["kind"] == "header":
            base = entry["base"]
        elif entry["kind"] == "document":
            documents[entry["name"]] = {**base["document"], **entry["delta"]}
        elif entry["kind"] == "snapshot":
            snapshots[entry["name"]] = {**base["snapshot"], **entry["delta"]}
        else:
            cases[entry["id"]] = entry
    return documents, snapshots, cases


def grading_case(case_id):
    """The document, the sealed snapshot and the runtime's outcome for one corpus case."""
    documents, snapshots, cases = _grading_corpus()
    case = cases[case_id]
    return (
        json.loads(json.dumps(documents[case["document"]])),
        json.loads(json.dumps(snapshots[case["snapshot"]])),
        case["oracle"],
    )
