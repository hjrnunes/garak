# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test that the Garak collector seals what the MiDojo runtime sealed.

The seven shared inconclusive vectors and the consumer's six injected-agent
fixtures each hold a sealed snapshot that names its turns, calls and times. The
collector records them again on a scripted clock, and the seal must have the
snapshot id and the bytes the runtime wrote. No runtime takes part, so the codec
and the collector stand on their own.
"""

import copy
import json
from pathlib import Path

import pytest

pytest.importorskip("midojo_spec")

from garak.evidence import trace  # noqa: E402
from garak.evidence.records import InvalidRequestError  # noqa: E402
from garak.evidence.seal import rebuild_snapshot  # noqa: E402

from sealed_replay import replay_snapshot  # noqa: E402

ASSETS = Path(__file__).parents[1] / "_assets"
VECTORS = ASSETS / "detectors" / "inconclusive_vectors"
CONSUMER = ASSETS / "evidence" / "consumer_ipi"

VECTOR_CASES = json.loads((VECTORS / "MANIFEST.json").read_text())["cases"]
CONSUMER_CASES = [
    "blocked",
    "delivered",
    "incomplete",
    "reply-judged",
    "sequential",
    "undelivered",
]
SEALED = [("vector", case) for case in VECTOR_CASES] + [
    ("consumer", case) for case in CONSUMER_CASES
]
#: graded with a scripted judge service, which this test has no use for
SCRIPTED_JUDGE = ("consumer", "reply-judged")

#: members of the interception service's full outcome that MiDojo never seals
SERVICE_ONLY = {
    "scope": "evaluation",
    "call_id": "10000000-0000-4000-8000-000000000000",
    "tool": "list_notes",
    "admission_order": 0,
}


@pytest.fixture(autouse=True)
def no_judge_service(monkeypatch):
    for name in ("JUDGE_URL", "JUDGE_TOKEN"):
        monkeypatch.delenv(name, raising=False)


def sealed_of(kind, case):
    """The folder, the document that was graded and the snapshot text the runtime wrote."""
    if kind == "vector":
        folder = VECTORS / case
        document = json.loads((folder / "document.json").read_text(encoding="utf-8"))
    else:
        folder = CONSUMER / case
        episodes = json.loads((folder / "ipi-episodes.json").read_text("utf-8"))
        document = episodes[0]["injection_spec"]
    return folder, document, (folder / "snapshot.json").read_text(encoding="utf-8")


def text_of(value):
    """The bytes the runtime's fixtures hold for ``value``."""
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def case_ids(pairs):
    return [pytest.param(kind, case, id=f"{kind}/{case}") for kind, case in pairs]


@pytest.mark.parametrize("kind, case", case_ids(SEALED))
def test_the_collector_seals_the_snapshot_the_runtime_sealed(kind, case):
    _, _, snapshot_text = sealed_of(kind, case)
    served = json.loads(snapshot_text)

    sealed = replay_snapshot(served)

    assert (
        sealed.snapshot_id == served["snapshot_id"]
    ), "the seal digests as the runtime's"
    assert (
        text_of(trace.snapshot_value(sealed)) == snapshot_text
    ), "the seal's value is the runtime's, byte for byte"


@pytest.mark.parametrize("kind, case", case_ids(sorted(set(SEALED) - {SCRIPTED_JUDGE})))
def test_the_grader_grades_the_collectors_seal_as_the_runtime_graded(kind, case):
    folder, document, snapshot_text = sealed_of(kind, case)

    sealed = replay_snapshot(json.loads(snapshot_text))
    values = trace.trace_values(document, sealed)

    assert text_of(values["verdict"]) == (folder / "verdict.json").read_text(
        encoding="utf-8"
    ), "the verdict, byte for byte"


def test_a_changed_fact_changes_the_digest():
    _, _, snapshot_text = sealed_of("consumer", "delivered")
    served = json.loads(snapshot_text)
    served["function_calls"][1]["args"]["amount"] = 121

    assert (
        replay_snapshot(served).snapshot_id != served["snapshot_id"]
    ), "the digest covers the recorded arguments"


@pytest.mark.parametrize("name", sorted(SERVICE_ONLY))
def test_an_outcome_field_the_runtime_never_seals_is_refused(name):
    """The consumer's reader refuses these fields too, so the codec leaves them out."""
    _, _, snapshot_text = sealed_of("consumer", "delivered")
    served = json.loads(snapshot_text)
    served["function_calls"][0]["injection_outcomes"][0][name] = SERVICE_ONLY[name]

    with pytest.raises(InvalidRequestError, match="invalid fields"):
        replay_snapshot(served)


@pytest.mark.parametrize("name", sorted(SERVICE_ONLY))
def test_an_outcome_field_the_runtime_never_seals_breaks_the_digest(name):
    """A snapshot that carried the field could never equal the one the runtime seals."""
    _, _, snapshot_text = sealed_of("consumer", "delivered")
    served = json.loads(snapshot_text)
    wider = copy.deepcopy(served)
    wider["function_calls"][0]["injection_outcomes"][0][name] = SERVICE_ONLY[name]

    assert (
        rebuild_snapshot(wider).snapshot_id != served["snapshot_id"]
    ), "the digest encodes every member of an outcome"
