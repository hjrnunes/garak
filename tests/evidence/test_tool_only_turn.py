"""A turn the agent answers with tool calls and no text is recorded as the runtime records it.

The MiDojo runtime records the reply of such a turn as ``""`` and goes on to the next turn.
``tool_only_turn/snapshot.json`` is a snapshot it sealed after one: the Garak collector and
the trace reader hold it to the same bytes.
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("midojo_spec")

from garak.evidence import trace  # noqa: E402
from garak.evidence.seal import rebuild_snapshot  # noqa: E402

from sealed_replay import replay_snapshot  # noqa: E402

SNAPSHOT = (
    Path(__file__).parent.parent
    / "_assets"
    / "evidence"
    / "tool_only_turn"
    / "snapshot.json"
)


def text_of(value):
    """The bytes the runtime's fixtures hold for ``value``."""
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


@pytest.fixture(name="served")
def served_text():
    return SNAPSHOT.read_text(encoding="utf-8")


def test_the_runtime_sealed_an_empty_reply_and_kept_its_chain(served):
    """The fixture is a tool-only first turn answered with ``""``, then a second turn."""
    document = json.loads(served)
    assert [turn["reply"] for turn in document["turns"]] == [
        "",
        "Order ORD-101 is open.",
    ], "the runtime's turns are not a tool-only turn and its follow-up"
    assert [call["turn"] for call in document["function_calls"]] == [
        0
    ], "the call is not the first turn's"


def test_a_runtime_snapshot_with_an_empty_reply_is_served_byte_for_byte(served):
    """The trace reader reads the empty reply and serves the runtime's bytes again."""
    snapshot = rebuild_snapshot(json.loads(served))
    assert (
        snapshot.snapshot_id == json.loads(served)["snapshot_id"]
    ), "the seal digests as the runtime's"
    assert text_of(trace.snapshot_value(snapshot)) == served, "byte for byte"


def test_the_collector_seals_the_bytes_the_runtime_sealed(served):
    """The collector, given the same turns and call, seals the same document and digest."""
    sealed = replay_snapshot(json.loads(served))
    assert sealed.snapshot_id == json.loads(served)["snapshot_id"], "digests differ"
    value = trace.snapshot_value(sealed)
    assert value["turns"][0]["reply"] == "", "the empty reply is not the empty text"
    assert text_of(value) == served, "byte for byte"
