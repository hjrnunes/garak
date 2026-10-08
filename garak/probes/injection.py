# SPDX-FileCopyrightText: Portions Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Indirect prompt injection probes.

Replays pre-built conversations (authored offline in OpenAI chat-completion format)
against a target **unchanged**. The conversations may contain tool calls and
tool-call results, so a harmful payload can be injected anywhere -- including as the
result of a tool call, modelling indirect prompt injection.

Each conversation carries an optional ``tools`` schema (forwarded to the target so it
can emit new tool calls) and a natural-language ``judge_description`` describing what a
successful attack looks like. The description is consumed by
``detectors.injection_judge.InjectionJudge``. An optional ``tool_call_condition``
states declaratively which tool calls mean the attack succeeded; it is evaluated by
``detectors.toolcall.ToolCallCondition``. An optional ``judge_rubric`` holds a yes/no
question, criteria, and facts for ``detectors.rubric_judge.RubricJudge``.

``SequentialInjection`` reads the same entries but holds only user messages: it
sends them one per request, each later request carrying the earlier turns and the
target's real replies.
"""

import copy
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import garak.attempt
from garak import _config
from garak.data import path as data_path
from garak.exception import GarakException
import garak.probes

# entry keys copied into the conversation notes
_ENTRY_NOTES = (
    "tools",
    "tool_choice",
    "judge_description",
    "tool_call_condition",
    "judge_rubric",
)
# notes a judge reads; a sequential probe sends them on the judged turns only
_JUDGE_NOTES = ("judge_description", "judge_rubric")
_GRADING_NOTES = ("tool_call_condition", *_JUDGE_NOTES)
JUDGE_TURNS = ("final", "all")


def resolve_source(conversation_source: str) -> Path:
    """Resolve a conversation source to an existing file path."""
    candidate = Path(conversation_source)
    if candidate.is_file():
        return candidate
    try:
        return Path(data_path / conversation_source)
    except GarakException as e:
        raise GarakException(
            f"Could not find conversation_source '{conversation_source}'"
        ) from e


def load_entries(source: Path, owner: str) -> List[dict]:
    """Load the conversation entries of a JSON or JSONL source.

    A JSON file holds a single object or a list of objects; a JSONL file one object
    per line. Entries that are not objects with a ``messages`` key are skipped.
    """
    entries: List[dict] = []
    with open(source, "r", encoding="utf-8") as f:
        if source.suffix == ".jsonl":
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        else:
            data = json.load(f)
            entries = data if isinstance(data, list) else [data]

    cleaned = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict) or "messages" not in entry:
            logging.warning(
                "%s: skipping malformed conversation entry %d in %s",
                owner,
                i,
                source,
            )
            continue
        cleaned.append(entry)
    return cleaned


def entry_notes(entry: dict) -> dict:
    """Return the conversation notes an entry declares: tools and grading data."""
    notes = {}
    for key in _ENTRY_NOTES:
        value = entry.get(key)
        if key in ("tools", "judge_description"):
            if value:
                notes[key] = value
        elif value is not None:
            notes[key] = value
    return notes


class IndirectInjection(garak.probes.Probe):
    """Replay pre-built (possibly tool-calling) conversations to model indirect prompt injection

    Loads conversations from a JSON/JSONL source and sends each one to the target
    without modification. An LLM-as-a-judge detector scores the target response using
    the per-conversation description; when a conversation carries a
    tool_call_condition, a deterministic detector also checks the target's tool calls,
    and when it carries a judge_rubric, a rubric judge answers its yes/no question.
    """

    lang = "en"
    goal = "elicit harmful behaviour via an injected conversation"
    doc_uri = (
        "https://owasp.org/www-project-top-10-for-large-language-model-applications/"
    )
    # instructions injected via other contexts (tool results, documents, ...)
    intent = "T009indirect"
    active = False  # opt-in: requires an external conversation_source file
    tags = [
        "avid-effect:security:S0403",
        "owasp:llm01",
        "quality:Security:PromptStability",
    ]
    tier = garak.probes.Tier.INFORMATIONAL
    primary_detector = "injection_judge.InjectionJudge"
    extended_detectors = ["toolcall.ToolCallCondition", "rubric_judge.RubricJudge"]

    DEFAULT_PARAMS = garak.probes.Probe.DEFAULT_PARAMS | {
        # path to a .json (single object or list of objects) or .jsonl file; may be an
        # absolute path or a path relative to a garak data directory
        "conversation_source": "injection/example_conversations.json",
    }

    def __init__(self, config_root=_config):
        super().__init__(config_root=config_root)
        self.prompts = []
        self._conversation_notes = []
        for entry in self._load_source():
            conversation = self._build_conversation(entry)
            self.prompts.append(conversation)

    def _resolve_source(self) -> Path:
        """Resolve conversation_source to an existing file path."""
        return resolve_source(self.conversation_source)

    def _load_source(self) -> List[dict]:
        """Load raw conversation entries from the source file.

        Accepts a JSON file containing either a single object or a list of objects, or
        a JSONL file with one object per line. Each object is expected to have a
        ``messages`` key (OpenAI messages array) and optional ``tools``,
        ``tool_choice``, ``judge_description``, ``tool_call_condition``, and
        ``judge_rubric`` keys.
        """
        return load_entries(self._resolve_source(), self.__class__.__name__)

    def _build_conversation(self, entry: dict) -> garak.attempt.Conversation:
        """Build a Conversation from a raw entry, stashing tools + judge criteria in notes."""
        return garak.attempt.Conversation.from_openai(
            entry["messages"], notes=entry_notes(entry)
        )


@dataclass
class Script:
    """The user turns of one sequential conversation and the notes of its entry."""

    texts: List[str]
    notes: dict


def script_of(entry: dict) -> Script | None:
    """Return an entry's script, or None unless it holds only user text messages."""
    messages = entry.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    if not all(
        isinstance(m, dict)
        and m.get("role") == "user"
        and isinstance(m.get("content"), str)
        for m in messages
    ):
        return None
    return Script([m["content"] for m in messages], entry_notes(entry))


def turn_failure(output: garak.attempt.Message | None) -> str | None:
    """Name why a turn's output ends the conversation, or return None.

    A request the API rejected names its error type; a response that did not
    finish is ``ResponsesNotCompleted``; a finished response without reply text is
    ``EmptyReply``.
    """
    notes = (output.notes or {}) if output is not None else {}
    error = notes.get("error")
    if isinstance(error, dict):
        return error.get("type") or "ResponsesRequestFailed"
    if notes.get("response_status") != "completed":
        return "ResponsesNotCompleted"
    if not output.text:
        return "EmptyReply"
    return None


class SequentialInjection(garak.probes.IterativeProbe):
    """Send scripted user turns one by one, each after the target's real replies

    Reads the entries ``IndirectInjection`` reads, keeping those whose messages are
    all user turns. Each turn is one request and one attempt; a later request
    carries the earlier user turns, the ``mcp_call`` items of each earlier response
    (the generator's ``output_items`` note), and the target's real replies, so the
    target needs no server-side state. The first turn that fails (a rejected
    request, an unfinished response, or no reply text) ends its conversation;
    ``notes["sequence"]["failure"]`` names why, and no attempt of that
    conversation is graded.

    Attempt ``notes["sequence"]`` holds ``conversation`` (the entry's position),
    ``turn_index`` (from 0), ``turn_count``, and ``call_offset`` (the tool calls
    made by earlier turns). Each output's ``tool_calls`` note holds every call
    made so far in the conversation, so the detectors see the whole conversation.
    The ``tool_call_condition`` travels on every turn; ``judge_rubric`` and
    ``judge_description`` on the final turn, or on every turn when
    ``judge_turns`` is ``all``. The scripts, not ``max_calls_per_conv``, bound
    each conversation; with several generations, the first one continues it.
    """

    lang = "en"
    goal = "elicit harmful behaviour over a scripted multi-turn conversation"
    doc_uri = (
        "https://owasp.org/www-project-top-10-for-large-language-model-applications/"
    )
    # instructions injected via other contexts (tool results, documents, ...)
    intent = "T009indirect"
    active = False  # opt-in: requires an external conversation_source file
    tags = [
        "avid-effect:security:S0403",
        "owasp:llm01",
        "quality:Security:PromptStability",
    ]
    tier = garak.probes.Tier.INFORMATIONAL
    primary_detector = "injection_judge.InjectionJudge"
    extended_detectors = ["toolcall.ToolCallCondition", "rubric_judge.RubricJudge"]

    DEFAULT_PARAMS = garak.probes.IterativeProbe.DEFAULT_PARAMS | {
        # a .json or .jsonl file, as for IndirectInjection
        "conversation_source": "injection/example_sequential.json",
        # "final" or "all": the turns that carry the judge notes
        "judge_turns": "final",
        "end_condition": "verify",
        "follow_prompt_cap": False,
    }

    def __init__(self, config_root=_config):
        super().__init__(config_root=config_root)
        if self.judge_turns not in JUDGE_TURNS:
            raise ValueError(
                f"judge_turns must be one of {JUDGE_TURNS}, not {self.judge_turns!r}"
            )
        self.scripts: List[Script] = []
        source = resolve_source(self.conversation_source)
        for i, entry in enumerate(load_entries(source, self.__class__.__name__)):
            script = script_of(entry)
            if script is None:
                logging.warning(
                    "%s: skipping entry %d in %s: messages must be user text turns",
                    self.__class__.__name__,
                    i,
                    source,
                )
                continue
            self.scripts.append(script)
        self.max_calls_per_conv = max((len(s.texts) for s in self.scripts), default=0)
        self._calls: dict = {}
        self._failed: set = set()

    def probe(self, generator) -> Iterable[garak.attempt.Attempt]:
        """Run every script; strip the grading notes of conversations that failed."""
        self._calls = {}
        self._failed = set()
        attempts = super().probe(generator)
        for attempt in attempts:
            if attempt.notes["sequence"]["conversation"] in self._failed:
                for key in _GRADING_NOTES:
                    attempt.prompt.notes.pop(key, None)
        return attempts

    def _create_init_attempts(self) -> Iterable[garak.attempt.Attempt]:
        return [self._turn_attempt(c, 0, []) for c in range(len(self.scripts))]

    def _generate_next_attempts(
        self, last_attempt: garak.attempt.Attempt
    ) -> Iterable[garak.attempt.Attempt]:
        sequence = last_attempt.notes["sequence"]
        index = sequence["turn_index"] + 1
        if "failure" in sequence or index >= sequence["turn_count"]:
            return []
        output = last_attempt.outputs[0]
        items = (output.notes or {}).get("output_items")
        reply = garak.attempt.Message(
            output.text, notes={"output_items": items} if items else None
        )
        earlier = [
            *copy.deepcopy(last_attempt.prompt.turns),
            garak.attempt.Turn("assistant", reply),
        ]
        return [self._turn_attempt(sequence["conversation"], index, earlier)]

    def _turn_attempt(
        self, conversation: int, index: int, earlier: list
    ) -> garak.attempt.Attempt:
        script = self.scripts[conversation]
        judged = self.judge_turns == "all" or index == len(script.texts) - 1
        notes = {
            k: v for k, v in script.notes.items() if judged or k not in _JUDGE_NOTES
        }
        user = garak.attempt.Turn(
            "user", garak.attempt.Message(script.texts[index], lang=self.lang)
        )
        prompt = garak.attempt.Conversation([*earlier, user], notes=notes)
        attempt = self._create_attempt(prompt)
        attempt.notes["sequence"] = {
            "conversation": conversation,
            "turn_index": index,
            "turn_count": len(script.texts),
            "call_offset": len(self._calls.get(conversation, [])),
        }
        return attempt

    def _postprocess_attempt(self, this_attempt) -> garak.attempt.Attempt:
        self._close_turn(this_attempt)
        return super()._postprocess_attempt(this_attempt)

    def _close_turn(self, attempt: garak.attempt.Attempt) -> None:
        """Record why a turn failed, and put every call so far on its output."""
        sequence = attempt.notes["sequence"]
        output = attempt.outputs[0] if attempt.outputs else None
        failure = turn_failure(output)
        if failure is not None:
            sequence["failure"] = failure
            self._failed.add(sequence["conversation"])
        if output is None:
            return
        calls = self._calls.setdefault(sequence["conversation"], [])
        calls.extend((output.notes or {}).get("tool_calls", []))
        output.notes = {**(output.notes or {}), "tool_calls": list(calls)}
