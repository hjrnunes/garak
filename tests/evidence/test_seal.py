"""The immutable sealed snapshot and its digest."""

import copy
import pickle

import pytest

pytest.importorskip("midojo_spec")
from midojo_spec import snapshot as shared

from garak.evidence.facts import CaptureFailure, Turn
from garak.evidence.records import CallBody, CallRecord
from garak.evidence.seal import Snapshot, rebuild_snapshot


def call(call_id="one", complete=True, result="ok"):
    """A record built the way the ledger builds one, unfinished ones included."""
    if complete:
        body = CallBody.build(function="tool", args={"n": 1}, result=result)
    else:
        body = CallBody.build_start(function="tool", args={"n": 1})
    return CallRecord(
        function=body.function,
        args=body.args,
        result=body.result,
        error=body.error,
        blocked=body.blocked,
        disposition=body.disposition,
        structured_content=body.structured_content,
        injection_outcomes=body.injection_outcomes,
        call_id=call_id,
        complete=complete,
        provenance="interceptor",
        timestamp="2026-01-01T00:00:00+00:00",
        turn=0,
    )


def sealed(**changes):
    """A snapshot with caller-supplied changes."""
    values = dict(
        evaluation_id="ev-1",
        agent_output="done",
        pre_environment={"balance": 10},
        environment={"balance": 9},
        function_calls=[call()],
        turns=[
            Turn(
                index=0,
                message="hello",
                reply="hi",
                started_at="2026-01-01T00:00:00+00:00",
                ended_at="2026-01-01T00:00:01+00:00",
            )
        ],
        observations={"runtime": {"findings": ["f"]}},
        evidence_status={"runtime.findings": "complete"},
        sealed_at="2026-01-01T00:00:02+00:00",
        capture_failures=[],
    )
    values.update(changes)
    return Snapshot(**values)


def test_snapshot_id_is_the_shared_digest():
    """The ID is what the shared codec computes over the same content."""
    snapshot = sealed()
    data = snapshot.to_json()
    assert snapshot.snapshot_id == shared.snapshot_digest(data), "ID is not the digest"
    assert shared.verified({**data, "snapshot_id": snapshot.snapshot_id}) is not None


def test_to_json_is_detached_and_fresh():
    """Mutating the exported JSON changes neither the digest nor a later read."""
    snapshot = sealed()
    first = snapshot.to_json()
    first["agent_output"] = "tampered"
    first["function_calls"][0]["result"] = "invented"
    first["evidence_status"]["function_calls"] = "incomplete"
    second = snapshot.to_json()
    assert second["agent_output"] == "done", "JSON mutated the snapshot"
    assert second["function_calls"][0]["result"] == "ok", "JSON mutated a call"
    assert second is not first, "to_json returned the same object"
    assert snapshot.verified(), "a mutated export changed the digest"


def test_mutating_the_inputs_does_not_change_the_digest():
    """A caller editing the objects it passed in changes nothing."""
    environment = {"balance": 10}
    turns = [Turn(index=0, message="m", reply=None, started_at="t", ended_at=None)]
    snapshot = sealed(environment=environment, turns=turns)
    digest = snapshot.snapshot_id
    environment["balance"] = 999
    turns.append(Turn(index=1, message="m2", reply=None, started_at="t", ended_at=None))
    assert snapshot.snapshot_id == digest, "the inputs changed the digest"
    assert snapshot.verified(), "the snapshot no longer verifies"


def test_snapshot_is_immutable():
    """A field write is refused."""
    snapshot = sealed()
    with pytest.raises(AttributeError):
        snapshot.agent_output = "changed"


def test_snapshot_copies_and_pickles_as_a_frozen_value():
    """Copying returns the snapshot; pickling round-trips the digest."""
    snapshot = sealed()
    assert copy.copy(snapshot) is snapshot, "a copy was made of a frozen value"
    assert copy.deepcopy(snapshot) is snapshot, "a deep copy was made"
    rebuilt = pickle.loads(pickle.dumps(snapshot))
    assert rebuilt.snapshot_id == snapshot.snapshot_id, "pickling changed the ID"
    assert rebuilt.to_json() == snapshot.to_json(), "pickling changed the content"
    assert rebuilt.verified(), "the rebuilt snapshot does not verify"


def test_tampered_content_fails_verification():
    """A snapshot whose content changed under its ID fails the shared check."""
    snapshot = sealed()
    data = snapshot.to_json()
    data["agent_output"] = "tampered"
    with pytest.raises(shared.DigestFault):
        shared.verified({**data, "snapshot_id": snapshot.snapshot_id})
    object.__setattr__(snapshot, "snapshot_id", "sha256:" + "0" * 64)
    assert not snapshot.verified(), "verification accepted a wrong ID"


def test_unfinished_calls_seal_and_rebuild():
    """An unfinished call keeps complete false and result null, through a rebuild."""
    unfinished = sealed(function_calls=[call(complete=False, result=None)])
    data = unfinished.to_json()
    sealed_call = data["function_calls"][0]
    assert sealed_call["complete"] is False, "unfinished call sealed complete"
    assert sealed_call["result"] is None, "unfinished call invented a result"
    rebuilt = rebuild_snapshot(data)
    assert rebuilt.snapshot_id == unfinished.snapshot_id, "rebuild changed the ID"
    assert (
        rebuilt.to_json()["function_calls"][0]["complete"] is False
    ), "rebuild completed the call"


def test_capture_failures_seal_in_field_order():
    """A capture failure exports the sealed reader's exact shape."""
    failure = CaptureFailure(
        call_id="one",
        function="tool",
        phase="finish",
        reason="unreachable",
        stage="collecting",
        by="interceptor",
        at="2026-01-01T00:00:01+00:00",
    )
    snapshot = sealed(capture_failures=[failure])
    data = snapshot.to_json()
    assert data["capture_failures"] == [failure.to_json()], "failure JSON changed"
    assert shared.verified({**data, "snapshot_id": snapshot.snapshot_id}) is not None


def test_empty_observations_and_environments_digest():
    """None environments and empty observations are content too."""
    snapshot = sealed(pre_environment=None, environment=None, observations={}, turns=[])
    data = snapshot.to_json()
    assert data["pre_environment"] is None and data["environment"] is None
    assert snapshot.verified(), "empty content did not verify"
