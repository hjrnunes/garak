# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run MiDojo episode documents as multi-turn tool-use probes.

Each source entry holds an ``episode`` (a MiDojo episode document) and an opaque
``metadata`` object. The probe reads the document through MiDojo's own reader, so
the plan, the tools and the checks are exactly what the document authored: the
payload is already in the document's plan, filled there by whoever compiled the
episode. The probe adds no knowledge of its own about the episode's contents.

Every chain is one episode driven through a MiDojo episode handle: the handle
runs the interception layer and, when the runtime needs it, a sandbox. The
probe opens one handle per chain in ``_create_init_attempts``, starts the
episode's first turn, and puts the interceptor's MCP tool block in
``Conversation.notes["tools"]`` as the request's tool list. The target's calls
reach the interceptor from the generator, and the handle records them whatever
the runtime is. The probe never holds a handle in an attempt's notes, so
``IterativeProbe``'s deep copy of the attempt queue carries only plain data.

At each turn's end ``_postprocess_attempt`` records the reply and either starts
the next turn or seals and grades the episode. A turn that a request rejected,
that never finished, or that neither spoke nor called a tool ends its chain
early; a turn that answers with tool calls alone does not, because MiDojo grades
those calls. The sealed trace goes on the
final attempt: the recorded tool calls (with their dispositions and whether each
finished), the evaluation and snapshot IDs, MiDojo's verdict, and whether the
capture is complete. ``detectors.toolcall.ToolCallCondition`` re-grades the
trace with no MiDojo code, and ``detectors.ipi_episode.EpisodeVerdict``
cross-checks it against the verdict.

The probe needs a Responses-style generator: one that takes the request's
``notes["tools"]`` as its tool list and reports ``response_status`` (and any
``tool_calls``) on each output message. Only such a generator can carry the
interceptor's tool block to the target and say whether a response finished.
Every other generator fails closed: a turn with no output, or with a response
status the generator never set, ends its chain as errored, and the chain scores
``None`` rather than as resisted.

``probe()`` returns only each chain's final attempt: an intermediate turn stays
in the report at status 1 and is never evaluated, so a two-turn chain scores
once. The harness, the evaluator and ``garak.analyze.report_digest`` all read
that shape as one result. Chains run one at a time, so at most one handle (and
so at most one interceptor, and one sandbox) is live at any moment.
"""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, List

import garak.attempt
from garak import _config
from garak.exception import ConfigFailure
import garak.probes
from garak.probes.injection import resolve_source, turn_failure

CHAIN_NOTE = "ipi_episode"
TRACE_NOTE = "episode_trace"
METADATA_NOTE = "source_metadata"
CONDITION_NOTE = "tool_call_condition"
EMPTY_REPLY = "EmptyReply"

_EXTRA_HINT = (
    "the ipi_episode plugins need the optional 'midojo' extra (Python 3.12 or "
    "newer): install it with `python -m pip install -e '.[midojo]'`"
)


def _turn_failure(output: garak.attempt.Message | None) -> str | None:
    """Name why a turn ends its chain early, or return None.

    ``turn_failure`` reads a reply-less response as a dead conversation. An
    episode's turn can answer with tool calls alone, and MiDojo grades those
    calls, so only a turn that neither speaks nor calls a tool fails.
    """
    failure = turn_failure(output)
    if failure == EMPTY_REPLY and (output.notes or {}).get("tool_calls"):
        return None
    return failure


def _midojo() -> SimpleNamespace:
    """The MiDojo pieces this probe drives, imported on first use.

    Nothing in this module imports MiDojo at import time, so the plugin lists
    without the extra installed and only running the probe needs it.
    """
    try:
        from midojo import episode_handle
        from midojo.control_plane.routers.runs import snapshot_response
        from midojo.episode import Episode
        from midojo.episode_document import (
            EpisodeDocumentError,
            UnsupportedDocumentVersion,
        )
        from midojo.runtimes import runtime_capabilities
    except ImportError as exc:
        raise ConfigFailure(_EXTRA_HINT) from exc
    return SimpleNamespace(
        episode_handle=episode_handle,
        snapshot_response=snapshot_response,
        Episode=Episode,
        EpisodeDocumentError=EpisodeDocumentError,
        UnsupportedDocumentVersion=UnsupportedDocumentVersion,
        runtime_capabilities=runtime_capabilities,
    )


def plain(value: Any) -> Any:
    """``value`` as plain JSON data: MiDojo reads documents into read-only mappings."""
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def load_episode_entries(source: Path, owner: str) -> List[dict]:
    """Load the episode entries of a JSON or JSONL source.

    A JSON file holds a single object or a list of objects; a JSONL file one
    object per line. Entries that are not objects holding an ``episode`` are
    skipped with a warning.
    """
    if source.suffix == ".jsonl":
        entries = [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        data = json.loads(source.read_text(encoding="utf-8"))
        entries = data if isinstance(data, list) else [data]

    kept = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or "episode" not in entry:
            logging.warning(
                "%s: skipping malformed episode entry %d in %s", owner, index, source
            )
            continue
        kept.append(entry)
    return kept


@dataclass(frozen=True)
class EpisodeEntry:
    """One source entry: the episode to run and what the entry declares beside it."""

    episode: Any
    metadata: Any
    condition: Any


class IPIEpisode(garak.probes.IterativeProbe):
    """Drive MiDojo episode documents, one episode per chain

    Reads entries of ``{"episode": <episode document>, "metadata": <any JSON>}``
    from ``source_path``. ``episode_source`` stays as a deprecated alias that
    applies only when ``source_path`` is unset; naming both with different files
    is refused. The document is read with MiDojo's reader, so a
    missing or unsupported ``schema_version`` is refused before any handle
    opens. ``metadata`` travels into every attempt's
    ``notes["source_metadata"]`` unread. The ``tool_call_condition`` the
    document's security check carries (or the entry's own) goes on every
    turn's prompt notes for ``detectors.toolcall.ToolCallCondition``.

    The episode's turns are the chain's turns: each is one request, and a later
    request carries the earlier turns and the target's real replies. The
    episode's plan supplies the payload; the probe never rewrites it. A target
    turn that fails (a rejected request, an unfinished response, or neither a
    reply nor a tool call) ends its chain early, and that turn is the chain's
    final attempt.

    ``repeats`` runs the same episode as that many separate chains, each its own
    evaluation, and is refused when the runtime resets neither by a declared
    reset nor by a fresh target. One reply per turn means ``generations`` is 1,
    and chains run one at a time whatever ``parallel_attempts`` says, so a
    runtime that starts a sandbox holds one at a time.

    Every chain is sealed and graded at its end, and its handle closes there, so
    at most one handle is live. ``probe()`` seals any chain an error interrupted
    and closes every remaining handle in a ``finally``: each open episode holds a
    listening interceptor.
    """

    extra_dependency_names = ["midojo"]
    # the cached imports and the parsed entries hold MiDojo data that does not
    # pickle; the state carries the source documents instead
    _unsafe_attributes = ["_midojo", "entries"]

    lang = "en"
    goal = "elicit harmful behaviour through an injected MiDojo episode"
    doc_uri = (
        "https://owasp.org/www-project-top-10-for-large-language-model-applications/"
    )
    # instructions injected via other contexts (tool results, documents, ...)
    intent = "T009indirect"
    active = False  # opt-in: requires an external source_path file
    tags = [
        "avid-effect:security:S0403",
        "owasp:llm01",
        "quality:Security:PromptStability",
    ]
    tier = garak.probes.Tier.INFORMATIONAL
    primary_detector = "toolcall.ToolCallCondition"
    extended_detectors = ["ipi_episode.EpisodeVerdict", "ipi_episode.EpisodeEffect"]

    DEFAULT_PARAMS = garak.probes.IterativeProbe.DEFAULT_PARAMS | {
        # a .json or .jsonl file holding episode entries
        "source_path": "injection/example_episode.json",
        # deprecated alias for source_path; honoured only when source_path is unset
        "episode_source": None,
        # an MCP server URL that wins over each document's interception upstream
        "upstream": None,
        # how many separate chains run the same episode
        "repeats": 1,
        "end_condition": "verify",
        "follow_prompt_cap": False,
    }

    def __init__(self, config_root=_config):
        try:
            super().__init__(config_root=config_root)
        except ModuleNotFoundError as exc:
            raise ConfigFailure(_EXTRA_HINT) from exc
        self._load_unsafe()
        self.entries: List[EpisodeEntry] = []
        source = self._source_path()
        self._entry_documents = load_episode_entries(source, self.__class__.__name__)
        for entry in self._entry_documents:
            self.entries.append(self._read_entry(entry))
        if not self.entries:
            raise ConfigFailure(
                f"{self.__class__.__name__}: {source} holds no episode entries"
            )
        # one agent reply per turn: the handle records exactly one reply
        self.generations = 1
        # chains run in turn order so only one episode is live at a time
        self.parallel_attempts = 1
        self._chains = len(self.entries) * self.repeats
        self.max_calls_per_conv = (
            sum(len(entry.episode.turns) for entry in self.entries) * self.repeats
        )
        self._refuse_unreset_repeats()
        self._handles = {}
        self._last_processed = {}
        self._next_attempts = {}
        self._pending_starts = list(range(self._chains))
        self._sealed = set()

    # --- reading the source ---

    def _load_unsafe(self):
        """Cache the MiDojo pieces and read the entries, which hold MiDojo data."""
        self._midojo = _midojo()
        self.entries = [
            self._read_entry(document)
            for document in getattr(self, "_entry_documents", [])
        ]

    def _source_path(self) -> Path:
        """The episode file the parameters name, refusing two different answers."""
        source = self.source_path
        alias = self.episode_source
        if alias is not None:
            logging.warning(
                "%s: episode_source is deprecated and will be removed; "
                "use source_path",
                self.__class__.__name__,
            )
            if source not in (None, self.DEFAULT_PARAMS["source_path"]) and str(
                source
            ) != str(alias):
                raise ConfigFailure(
                    f"{self.__class__.__name__}: source_path {source!r} and the "
                    f"deprecated episode_source {alias!r} name different files"
                )
            source = alias
        if source is None:
            raise ConfigFailure(
                f"{self.__class__.__name__}: no source_path names an episode file"
            )
        return resolve_source(source)

    def _read_entry(self, entry: dict) -> EpisodeEntry:
        document = entry["episode"]
        try:
            episode = self._midojo.Episode.from_document(document)
        except self._midojo.UnsupportedDocumentVersion as exc:
            raise ConfigFailure(
                f"{self.__class__.__name__}: episode document version "
                f"{exc.version!r} is not supported; supported: "
                f"{', '.join(exc.supported)}"
            ) from exc
        except self._midojo.EpisodeDocumentError as exc:
            raise ConfigFailure(
                f"{self.__class__.__name__}: the episode document is refused: {exc}"
            ) from exc
        except ValueError as exc:
            raise ConfigFailure(
                f"{self.__class__.__name__}: the episode document is refused: {exc}"
            ) from exc
        if not episode.turns:
            raise ConfigFailure(
                f"{self.__class__.__name__}: the episode document holds no turns "
                "to script; a host-driven episode has nothing for this probe to send"
            )
        condition = entry.get(CONDITION_NOTE)
        if condition is None and isinstance(episode.security, Mapping):
            condition = episode.security.get(CONDITION_NOTE)
        return EpisodeEntry(
            episode=episode,
            metadata=copy.deepcopy(entry.get("metadata")),
            condition=None if condition is None else plain(condition),
        )

    def _refuse_unreset_repeats(self) -> None:
        """Refuse repeats a runtime cannot keep apart, before a handle opens."""
        if self.repeats <= 1:
            return
        for entry in self.entries:
            capabilities = self._midojo.runtime_capabilities(entry.episode.runtime)
            if not capabilities.isolates_attempts:
                raise ConfigFailure(
                    f"{self.__class__.__name__}: repeats={self.repeats} would run "
                    f"the same episode against the {capabilities.name!r} runtime, "
                    "which neither resets its target nor gives each evaluation a "
                    "fresh one; set repeats to 1 or declare a reset"
                )

    # --- the chain hooks ---

    def _start_chain(self, chain: int) -> garak.attempt.Attempt:
        """Open the chain's handle and mint its first-turn attempt."""
        entry = self.entries[chain // self.repeats]
        repeat = chain % self.repeats
        try:
            handle = self._midojo.episode_handle.open(
                entry.episode,
                upstream=self.upstream,
                reset=repeat > 0,
            )
        except self._midojo.EpisodeDocumentError as exc:
            raise ConfigFailure(
                f"{self.__class__.__name__}: no upstream to intercept: {exc}"
            ) from exc
        self._handles[chain] = handle
        handle.start_turn(entry.episode.turns[0])
        return self._turn_attempt(chain, 0, [], handle)

    def _create_init_attempts(self) -> Iterable[garak.attempt.Attempt]:
        return [self._start_chain(self._pending_starts.pop(0))]

    def _generate_next_attempts(
        self, last_attempt: garak.attempt.Attempt
    ) -> Iterable[garak.attempt.Attempt]:
        minted = self._next_attempts.pop(last_attempt.notes[CHAIN_NOTE]["chain"], [])
        if minted:
            return minted
        if self._pending_starts:
            return [self._start_chain(self._pending_starts.pop(0))]
        return []

    def _turn_attempt(
        self, chain: int, index: int, earlier: list, handle
    ) -> garak.attempt.Attempt:
        entry = self.entries[chain // self.repeats]
        notes = {"tools": [handle.tool_block()]}
        if entry.condition is not None:
            notes[CONDITION_NOTE] = copy.deepcopy(entry.condition)
        user = garak.attempt.Turn(
            "user",
            garak.attempt.Message(entry.episode.turns[index], lang=self.lang),
        )
        prompt = garak.attempt.Conversation([*earlier, user], notes=notes)
        attempt = self._create_attempt(prompt)
        attempt.notes[CHAIN_NOTE] = {
            "chain": chain,
            "turn": index,
            "turns": len(entry.episode.turns),
            "final": index == len(entry.episode.turns) - 1,
        }
        attempt.notes[METADATA_NOTE] = copy.deepcopy(entry.metadata)
        return attempt

    def _postprocess_attempt(self, this_attempt) -> garak.attempt.Attempt:
        trace = self._advance_chain(this_attempt)
        processed = super()._postprocess_attempt(this_attempt)
        self._last_processed[this_attempt.notes[CHAIN_NOTE]["chain"]] = processed
        if trace is not None:
            self._attach_trace(processed, trace)
        return processed

    def _advance_chain(self, this_attempt) -> dict | None:
        """Record the reply, then start the next turn or seal the chain.

        Returns the chain's trace on its final turn, and None while it goes on.
        """
        state = this_attempt.notes[CHAIN_NOTE]
        chain, index = state["chain"], state["turn"]
        handle = self._handles[chain]
        output = this_attempt.outputs[0] if this_attempt.outputs else None
        if output is not None and not handle.closed and not handle.sealed:
            handle.record_reply(index, output.text or "")
        # a turn with no output has no reply to record, so the seal leaves that
        # turn unanswered; the failure below still ends the chain
        failure = _turn_failure(output)
        if failure is None and not state["final"]:
            earlier = copy.deepcopy(this_attempt.prompt.turns)
            if output is not None and output.text:
                earlier.append(
                    garak.attempt.Turn("assistant", garak.attempt.Message(output.text))
                )
            entry = self.entries[chain // self.repeats]
            handle.start_turn(entry.episode.turns[index + 1])
            self._next_attempts[chain] = [
                self._turn_attempt(chain, index + 1, earlier, handle)
            ]
            return None
        state["final"] = True
        self._sealed.add(chain)
        try:
            trace = self._seal_trace(handle)
        finally:
            self._close_handle(chain)
        if failure is not None:
            # a turn the target never answered leaves the chain's condition undecided
            trace["error"] = failure
            trace["response_status"] = "incomplete"
        return trace

    # --- the sealed trace ---

    def _seal_trace(self, handle) -> dict:
        snapshot = handle.seal()
        return self._trace(handle, snapshot, sealed="final")

    def _trace(self, handle, snapshot, *, sealed: str) -> dict:
        verdict = handle.grade()
        served = self._midojo.snapshot_response(snapshot).model_dump(mode="json")
        complete = self._capture_complete(snapshot)
        calls = [
            {
                "call_id": call.call_id,
                "name": call.function,
                "arguments": copy.deepcopy(call.args),
                "result": call.result,
                "blocked": call.blocked,
                "disposition": call.disposition,
                "complete": call.complete,
                "turn": call.turn,
            }
            for call in snapshot.function_calls
        ]
        return {
            "evaluation_id": handle.evaluation_id,
            "snapshot_id": snapshot.snapshot_id,
            "snapshot": served,
            "verdict": verdict.to_document(),
            "capture_complete": complete,
            "sealed": sealed,
            "failure": None,
            "error": None,
            "tool_calls": calls,
            # only a delivered payload with a complete capture can decide a condition
            "response_status": (
                "completed"
                if complete and verdict.grade.injection_delivered is True
                else "incomplete"
            ),
        }

    @staticmethod
    def _capture_complete(snapshot) -> bool:
        evidence = snapshot.evidence_status or {}
        if any(status == "incomplete" for status in evidence.values()):
            return False
        if snapshot.capture_failures:
            return False
        return all(call.complete for call in snapshot.function_calls)

    def _attach_trace(self, attempt, trace: dict) -> None:
        attempt.notes[TRACE_NOTE] = trace
        output = self._stored_output(attempt)
        if output is None:
            # a turn the target never answered stores no message; the condition
            # detector still needs a stored output to read this trace from
            output = garak.attempt.Message("")
            self._store_output(attempt, output)
        output.notes = {
            **(output.notes or {}),
            "tool_calls": copy.deepcopy(trace["tool_calls"]),
            "response_status": self._capture_status(trace),
        }

    @staticmethod
    def _capture_status(trace: dict) -> str:
        """Name why this capture cannot decide a condition, or that it can.

        ``toolcall.ToolCallCondition`` suffixes an inconclusive reason with the
        output's ``response_status``, so a recording whose grade states no
        delivery fact is named here: a reader of the receipt can then tell an
        unrecorded delivery from a capture that really was incomplete.
        """
        if trace["sealed"] != "final" or trace["error"] is not None:
            return "incomplete"
        if not trace["capture_complete"]:
            return "incomplete"
        grade = (trace.get("verdict") or {}).get("grade") or {}
        delivered = grade.get("injection_delivered")
        if delivered is True:
            return "completed"
        if delivered is None:
            return "unrecorded_delivery"
        return "incomplete"

    @staticmethod
    def _stored_output(attempt) -> garak.attempt.Message | None:
        """The message ``attempt.outputs`` reads, when its last turn holds one."""
        if not attempt.conversations:
            return None
        assistant = [
            turn for turn in attempt.conversations[0].turns if turn.role == "assistant"
        ]
        if not assistant:
            return None
        content = assistant[-1].content
        return content if isinstance(content, garak.attempt.Message) else None

    @staticmethod
    def _store_output(attempt, output: garak.attempt.Message) -> None:
        """Write ``output`` where ``attempt.outputs`` reads it: the last assistant turn."""
        assistant = [
            turn for turn in attempt.conversations[0].turns if turn.role == "assistant"
        ]
        if assistant:
            assistant[-1].content = output
            return
        attempt.conversations[0].turns.append(garak.attempt.Turn("assistant", output))

    # --- the run ---

    def probe(self, generator) -> Iterable[garak.attempt.Attempt]:
        """Run every chain, then seal what an error interrupted and close the handles."""
        self._handles = {}
        self._last_processed = {}
        self._next_attempts = {}
        self._pending_starts = list(range(self._chains))
        self._sealed = set()
        try:
            attempts = super().probe(generator)
        finally:
            try:
                self._seal_interrupted()
            finally:
                self._close_handles()
        return [
            attempt
            for attempt in attempts
            if (attempt.notes.get(CHAIN_NOTE) or {}).get("final")
        ]

    def _seal_interrupted(self) -> None:
        """Seal every chain an error or an interrupt cut short, and mark its last attempt final."""
        for chain, handle in list(self._handles.items()):
            if chain in self._sealed:
                continue
            try:
                handle.close()
                self._sealed.add(chain)
                attempt = self._last_processed.get(chain)
                if attempt is None:
                    continue
                trace = self._interrupted_trace(handle)
                attempt.notes[CHAIN_NOTE]["final"] = True
                self._attach_trace(attempt, trace)
            except (RuntimeError, ValueError) as exc:
                logging.warning(
                    "IPIEpisode: couldn't seal interrupted chain %d: %s", chain, exc
                )

    def _interrupted_trace(self, handle) -> dict:
        """The trace of a chain closed before its seal: no verdict is trustworthy."""
        result = handle.result
        snapshot = None if result is None else result.snapshot
        if snapshot is None:
            error = None if result is None else result.snapshot_error
            return {
                "evaluation_id": handle.evaluation_id,
                "snapshot_id": None,
                "snapshot": None,
                "verdict": None,
                "capture_complete": False,
                "sealed": "interrupted",
                "failure": None,
                "error": type(error).__name__ if error is not None else "NoSnapshot",
                "tool_calls": [],
                "response_status": "incomplete",
            }
        return self._trace(handle, snapshot, sealed="interrupted")

    def _close_handle(self, chain: int) -> None:
        """Close and forget a chain's handle once the chain is sealed.

        A sealed chain needs neither its interceptor nor its sandbox, and
        closing here keeps at most one handle live however many chains run.
        """
        handle = self._handles.pop(chain, None)
        if handle is None:
            return
        try:
            handle.close()
        except (RuntimeError, ValueError) as exc:
            logging.warning("IPIEpisode: couldn't close chain %d: %s", chain, exc)

    def _close_handles(self) -> None:
        for chain in list(self._handles):
            self._close_handle(chain)
