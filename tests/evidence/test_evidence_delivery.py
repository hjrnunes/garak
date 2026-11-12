# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check delivery facts and exposure against what the MiDojo runtime derives.

``tests/_assets/detectors/grading/corpus.jsonl`` pairs injection spec documents
and sealed snapshots with the delivery facts and the exposure the runtime
writes for them, so the comparison needs no runtime. ``generate.py`` in the
same directory rewrites the corpus.
"""

import dataclasses
import json
from functools import lru_cache
from pathlib import Path

import pytest

pytest.importorskip("midojo_spec")
from midojo_spec import injection  # noqa: E402

from garak.evidence import delivery  # noqa: E402
from garak.evidence.seal import rebuild_snapshot  # noqa: E402

CORPUS = (
    Path(__file__).parents[1] / "_assets" / "detectors" / "grading" / "corpus.jsonl"
)
FACT_KEYS = [
    "call_id",
    "function",
    "disposition",
    "turn",
    "interceptor_instruction_index",
    "plan_instruction_index",
    "slot_key",
    "occurrence",
    "status",
    "selected",
    "transformation",
    "emitted",
    "payload_digest",
    "placed_digest",
    "slot",
]
EXPOSURE_KEYS = ["emitted", "agent_tool_result", "model_input", "instructions"]
INSTRUCTION_KEYS = [
    "plan_instruction_index",
    "channel",
    "slot_key",
    "placed_by",
    "exposure",
    "reason",
]


@lru_cache(maxsize=1)
def corpus():
    base, documents, snapshots, cases = None, {}, {}, []
    for line in CORPUS.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        if entry["kind"] == "header":
            base = entry["base"]
        elif entry["kind"] == "document":
            documents[entry["name"]] = {**base["document"], **entry["delta"]}
        elif entry["kind"] == "snapshot":
            snapshots[entry["name"]] = {**base["snapshot"], **entry["delta"]}
        else:
            cases.append(entry)
    return documents, snapshots, cases


def delivery_cases():
    return [case for case in corpus()[2] if "exposure" in case["oracle"]]


def derive(case):
    documents, snapshots, _ = corpus()
    plan = injection.InjectionPlan.from_data(documents[case["document"]]["plan"])
    sealed = rebuild_snapshot(snapshots[case["snapshot"]])
    calls = sealed.function_calls
    return (
        delivery.delivery_facts(plan, calls),
        delivery.exposure(
            plan,
            calls,
            turns=len(sealed.turns),
            evidence_status=sealed.evidence_status,
        ),
    )


def canonical(value):
    return json.dumps(value, sort_keys=True)


def test_corpus_holds_the_delivery_cases():
    assert len(delivery_cases()) >= 40, "the delivery group carries raw oracle values"


@pytest.mark.parametrize("case", delivery_cases(), ids=lambda case: case["id"])
def test_delivery_facts_match_the_runtime(case):
    facts, _ = derive(case)
    assert canonical([fact.to_json() for fact in facts]) == canonical(
        case["oracle"]["delivery_facts"]
    ), "each fact equals the runtime's, in call then instruction order"


@pytest.mark.parametrize("case", delivery_cases(), ids=lambda case: case["id"])
def test_exposure_matches_the_runtime(case):
    _, exposed = derive(case)
    assert canonical(exposed.to_json()) == canonical(
        case["oracle"]["exposure"]
    ), "the exposure equals the runtime's at every boundary, reason included"


def test_documents_are_written_in_the_runtimes_key_order():
    cases = {case["id"]: case for case in delivery_cases()}
    facts, exposed = derive(cases["delivery.placed-with-slot"])
    written = facts[0].to_json()
    assert list(written) == FACT_KEYS, "a fact keeps the runtime's field order"
    assert list(written["slot"]) == ["id", "target", "mode"], "a slot is re-encoded"
    assert list(exposed.to_json()) == EXPOSURE_KEYS, "an exposure keeps its order"
    assert (
        list(exposed.to_json()["instructions"][0]) == INSTRUCTION_KEYS
    ), "an instruction exposure keeps its order"


def test_prompt_instruction_has_no_agent_tool_result_boundary():
    cases = {case["id"]: case for case in delivery_cases()}
    _, exposed = derive(cases["delivery.prompt-sent"])
    assert list(exposed.to_json()["instructions"][0]["exposure"]) == [
        "emitted",
        "model_input",
    ], "a turn message is not a tool result"


def test_legacy_slot_key_reads_as_slot_key():
    cases = {case["id"]: case for case in delivery_cases()}
    facts, _ = derive(cases["delivery.emitted-legacy-key"])
    assert facts[0].slot_key == "planted_item:seed_0", "probe_key names the slot key"
    assert "probe_key" not in facts[0].to_json(), "a fact is written under slot_key"


def test_exposure_at_names_each_boundary():
    cases = {case["id"]: case for case in delivery_cases()}
    _, exposed = derive(cases["delivery.emitted"])
    assert [exposed.at(boundary) for boundary in delivery.EXPOSURE_BOUNDARIES] == [
        "exposed",
        "unknown",
        "unknown",
    ], "only emission is observed"
    with pytest.raises(KeyError):
        exposed.at("elsewhere")


def test_results_are_immutable_and_detached():
    cases = {case["id"]: case for case in delivery_cases()}
    facts, exposed = derive(cases["delivery.placed-with-slot"])
    with pytest.raises(dataclasses.FrozenInstanceError):
        facts[0].status = "unmatched"
    with pytest.raises(dataclasses.FrozenInstanceError):
        exposed.emitted = "unknown"
    facts[0].to_json()["slot"]["target"]["tool"] = "other"
    facts[0].to_json()["emitted"].append("elsewhere")
    exposed.to_json()["instructions"][0]["exposure"]["emitted"] = "unknown"
    assert facts[0].to_json()["slot"]["target"]["tool"] == "lookup_order"
    assert facts[0].to_json()["emitted"] == ["structured_content", "content_block"]
    assert exposed.to_json()["instructions"][0]["exposure"]["emitted"] == "exposed"


def refused_cases():
    return [case for case in corpus()[2] if case["group"] == "outcome-refused"]


def test_corpus_holds_the_refused_outcomes():
    assert len(refused_cases()) >= 45, "every damage the runtime's record refuses"
    assert all("error" in case["oracle"] for case in refused_cases())


@pytest.mark.parametrize("case", refused_cases(), ids=lambda case: case["id"])
def test_an_outcome_the_runtime_refuses_is_refused(case):
    documents, snapshots, _ = corpus()
    plan = injection.InjectionPlan.from_data(documents[case["document"]]["plan"])
    sealed = rebuild_snapshot(snapshots[case["snapshot"]])
    with pytest.raises(ValueError, match="malformed injection outcome"):
        delivery.delivery_facts(plan, sealed.function_calls)
    with pytest.raises(ValueError, match="malformed injection outcome"):
        delivery.exposure(
            plan,
            sealed.function_calls,
            turns=len(sealed.turns),
            evidence_status=sealed.evidence_status,
        )
