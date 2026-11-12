"""Regenerate ``snapshot.json``: a snapshot the MiDojo runtime sealed after a turn the agent answered with tool calls and no text.

The runtime records such a reply as ``""``, and its first turn keeps its chain: the
second turn starts once the first has its (empty) reply. The Garak collector is held to
the same bytes by ``tests/evidence/test_tool_only_turn.py``. Run this script with an
interpreter that has the ``midojo`` runtime installed, at the commit the other copies of
the runtime's fixtures name (``../consumer_ipi/MANIFEST.json``)::

    <midojo venv>/bin/python tests/_assets/evidence/tool_only_turn/generate.py --write
    <midojo venv>/bin/python tests/_assets/evidence/tool_only_turn/generate.py --check

``--check`` regenerates in memory and exits non-zero when ``snapshot.json`` differs, so the
file is never edited by hand. The script imports nothing from Garak. It seals the case the
way the runtime's own inconclusive-vector generator seals its in-process cases: under a
pinned clock and pinned ids, through the in-process control plane's own writes, from the
episode document of the ``capture-incomplete`` vector (``../../detectors/inconclusive_vectors``).
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import sys
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
SNAPSHOT = HERE / "snapshot.json"
SEED = (
    HERE.parent.parent
    / "detectors"
    / "inconclusive_vectors"
    / "capture-incomplete"
    / "document.json"
)
# The clock every seal starts from, the step each reading of it takes, and a version-4
# UUID with a counter in its last bytes: unique, and the same on every run.
PINNED_START = datetime(2026, 1, 1, tzinfo=UTC)
PINNED_STEP = timedelta(milliseconds=1)
PINNED_UUID = 0x10000000000040008000000000000000
# The first turn's tool call, and the replies: the first turn is answered with no text.
CALL = ("lookup_order", {"order_id": "ORD-101"})
RESULT = {"order_id": "ORD-101", "status": "open"}
REPLIES = ("", "Order ORD-101 is open.")


@contextmanager
def pinned():
    """Seal under a pinned clock and pinned ids, leaving every runtime file as it is."""
    from midojo.control_plane import service, state, store

    ticks, names, evaluations = (
        itertools.count(),
        itertools.count(1),
        itertools.count(1),
    )

    def now() -> str:
        return (PINNED_START + PINNED_STEP * next(ticks)).isoformat()

    def uuid4() -> uuid.UUID:
        return uuid.UUID(int=PINNED_UUID + next(names))

    def token_hex(size: int = 5) -> str:
        return f"{next(evaluations):0{2 * size}x}"

    resume = (state._now, service._now, uuid.uuid4, store.secrets.token_hex)
    state._now = service._now = now
    uuid.uuid4 = uuid4
    store.secrets.token_hex = token_hex
    try:
        yield
    finally:
        state._now, service._now, uuid.uuid4, store.secrets.token_hex = resume


async def run(built):
    """Run the two turns by the control plane's own writes and return the sealed snapshot."""
    from midojo.control_plane.config import AppConfig
    from midojo.control_plane.inprocess_control import InProcessControlClient
    from midojo.control_plane.service import ControlPlane

    client = InProcessControlClient(
        ControlPlane({}, config=AppConfig(closure_wait_seconds=0))
    )
    opened = await client.open_evaluation(built, interception="host")
    recorder = client.recorder(opened.session_token)
    turns = opened.turns
    if turns is None or len(turns) < len(REPLIES):
        raise ValueError("the episode names too few turns to run")
    name, arguments = CALL
    for index, reply in enumerate(REPLIES):
        await client.start_turn(None, opened.id, turns[index])
        if index == 0:
            await recorder.start_call(
                call_id="call-0", function=name, args=arguments, disposition="forwarded"
            )
            await recorder.finish_call(
                call_id="call-0",
                function=name,
                args=arguments,
                result=json.dumps(RESULT),
                disposition="forwarded",
            )
        await client.record_turn_reply(None, opened.id, index, reply)
    await client.complete_evaluation(None, opened.id, REPLIES[-1])
    return await client.snapshot(
        None, opened.id, environment_type=built.environment_type
    )


def served(snapshot, environment_type) -> dict:
    """The snapshot as ``GET .../snapshot`` serves it, checked: it reads back to the sealed one."""
    from midojo.control_plane.client import snapshot_from_response
    from midojo.control_plane.models import SnapshotResponse
    from midojo.control_plane.routers.runs import snapshot_response
    from midojo.control_plane.state import snapshot_digest
    from midojo_spec.snapshot import verified

    document = snapshot_response(snapshot).model_dump(mode="json")
    again = snapshot_from_response(
        SnapshotResponse.model_validate(document), environment_type
    )
    verified(document)
    assert (
        again.snapshot_id
        == snapshot.snapshot_id
        == document["snapshot_id"]
        == snapshot_digest(again)
    )
    return document


def generate() -> str:
    """The text of ``snapshot.json``."""
    from midojo.injection_spec import InjectionSpec

    built = InjectionSpec.from_document(json.loads(SEED.read_text(encoding="utf-8")))
    with pinned():
        snapshot = asyncio.run(run(built))
        document = served(snapshot, built.environment_type)
    turns = document["turns"]
    assert (
        turns[0]["reply"] == "" and turns[1]["reply"] == REPLIES[1]
    ), "the turns lost their replies"
    return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="write snapshot.json")
    mode.add_argument(
        "--check", action="store_true", help="fail when snapshot.json differs"
    )
    arguments = parser.parse_args(argv)
    text = generate()
    if arguments.write:
        SNAPSHOT.write_text(text, encoding="utf-8")
        return 0
    if SNAPSHOT.read_text(encoding="utf-8") != text:
        print("snapshot.json differs from the runtime's seal", file=sys.stderr)
        return 1
    print("snapshot.json matches the runtime's seal")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
