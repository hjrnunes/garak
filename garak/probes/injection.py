# SPDX-FileCopyrightText: Portions Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conversation replay probes.

Replays pre-built conversations (authored offline in OpenAI chat-completion format,
with system, user and assistant messages) against a target **unchanged**.

Each conversation carries an optional ``tools`` schema, forwarded to the target so it
can call tools. An optional ``tool_call_condition`` states declaratively which tool
calls mean the attack succeeded; it is evaluated by
``detectors.toolcall.ToolCallCondition``. An optional ``judge_rubric`` holds a yes/no
question, criteria, and facts for ``detectors.rubric_judge.RubricJudge``.
"""

import json
import logging
from typing import List

import garak.attempt
from garak import _config
import garak.probes
from garak.probes._injection_shared import resolve_source

# entry keys copied into the conversation notes
_ENTRY_NOTES = (
    "tools",
    "tool_choice",
    "tool_call_condition",
    "judge_rubric",
)


def load_entries(source, owner: str) -> List[dict]:
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
        if key == "tools":
            if value:
                notes[key] = value
        elif value is not None:
            notes[key] = value
    return notes


class IndirectInjection(garak.probes.Probe):
    """Replay pre-built conversations against a target, unchanged

    Loads conversations from a JSON/JSONL source and sends each one to the target
    without modification. When a conversation carries a tool_call_condition, a
    deterministic detector checks the target's tool calls; when it carries a
    judge_rubric, a rubric judge answers its yes/no question.
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
    primary_detector = "toolcall.ToolCallCondition"
    extended_detectors = ["rubric_judge.RubricJudge"]

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

    def _load_source(self) -> List[dict]:
        """Load raw conversation entries from the source file.

        Accepts a JSON file containing either a single object or a list of objects, or
        a JSONL file with one object per line. Each object is expected to have a
        ``messages`` key (OpenAI messages array) and optional ``tools``,
        ``tool_choice``, ``tool_call_condition``, and ``judge_rubric`` keys.
        """
        return load_entries(
            resolve_source(self.conversation_source), self.__class__.__name__
        )

    def _build_conversation(self, entry: dict) -> garak.attempt.Conversation:
        """Build a Conversation from a raw entry, stashing tools + grading data in notes."""
        return garak.attempt.Conversation.from_openai(
            entry["messages"], notes=entry_notes(entry)
        )
