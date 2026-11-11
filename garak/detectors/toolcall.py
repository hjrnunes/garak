# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for reading the tool calls a target emitted.

Generators capture the calls a target made into ``Message.notes["tool_calls"]``.
The helpers here decode and normalise that capture, in both the Responses API
and the chat-completions shape, so a detector can judge it without a model.
"""

import json
from collections.abc import Mapping
from typing import Any, List, Optional, Tuple

from garak.attempt import Message


def decode_payload(raw: Any) -> Any:
    """Decode a raw tool-call argument or result payload.

    Strings are parsed as JSON, and a parse failure gives ``None``; any other value
    passes through unchanged. Text that Python cannot parse, such as an integer
    with more digits than its conversion limit or a document nested too deeply,
    is a parse failure.
    """
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except (ValueError, RecursionError):
        return None


def _json_shaped(value: Any, ancestors: Optional[set] = None) -> bool:
    """Say whether ``value`` is built only from what a JSON document holds.

    Types are matched exactly, so no method of a caller's object runs, and a
    value that holds itself is not JSON.
    """
    kind = type(value)
    if value is None or any(kind is scalar for scalar in (bool, int, float, str)):
        return True
    if kind is not list and kind is not dict:
        return False
    ancestors = set() if ancestors is None else ancestors
    if id(value) in ancestors:
        return False
    ancestors.add(id(value))
    try:
        if kind is list:
            return all(_json_shaped(item, ancestors) for item in value)
        return all(
            type(key) is str and _json_shaped(item, ancestors)
            for key, item in value.items()
        )
    finally:
        ancestors.discard(id(value))


def _text(value: Any) -> Optional[str]:
    return value if type(value) is str else None


def _first_present(call: Mapping, keys: Tuple[str, ...]) -> Any:
    for key in keys:
        if call.get(key) is not None:
            return call[key]
    return None


def normalise_tool_call(call: Any) -> dict:
    """Normalise one captured tool call from either capture shape.

    Handles Responses API items (``name``, ``arguments``, ``output``, and for an
    ``mcp_call`` also ``input``; the ``input`` of another item is a custom tool's
    freeform text and is not read) and chat-completions items
    (``function.name``, ``function.arguments``). Returns a dict with ``name``
    (text or ``None``), ``arguments`` (decoded), and ``result`` (decoded), plus
    ``blocked`` when the capture marks the call with a boolean ``blocked`` (the
    target emitted the call but it did not run).
    """
    if not isinstance(call, Mapping):
        return {"name": None, "arguments": None, "result": None}
    function = call.get("function")
    if isinstance(function, Mapping):
        normalised = {
            "name": _text(function.get("name")),
            "arguments": decode_payload(function.get("arguments")),
            "result": None,
        }
    else:
        sources = ("arguments", "raw_arguments")
        if _text(call.get("type")) == "mcp_call":
            sources += ("input",)
        normalised = {
            "name": _text(_first_present(call, ("name", "tool_name"))),
            "arguments": decode_payload(_first_present(call, sources)),
            "result": decode_payload(
                _first_present(call, ("result", "output", "raw_result"))
            ),
        }
    if isinstance(call.get("blocked"), bool):
        normalised["blocked"] = call["blocked"]
    return normalised


def read_tool_calls(output: Message) -> Tuple[Optional[List[dict]], bool]:
    """Read and normalise the tool calls captured on an output message.

    Returns ``(calls, complete)``. ``calls`` is ``None`` when the notes are not
    an object or the ``tool_calls`` note is present but not a list. A missing
    note means the target made no tool call. ``complete`` is true only when the
    ``response_status`` note says the response finished (``completed``, or the
    note is absent) and every captured entry is an object: an entry that is not
    may have been a call to any tool.
    """
    notes = {} if output.notes is None else output.notes
    if not isinstance(notes, Mapping):
        return None, False
    raw = notes.get("tool_calls", [])
    if not isinstance(raw, list):
        return None, False
    complete = _text(notes.get("response_status", "completed")) == "completed"
    complete = complete and all(isinstance(call, Mapping) for call in raw)
    return [normalise_tool_call(call) for call in raw], complete


def _blocked(call: dict) -> bool:
    return call.get("blocked") is True


def _usable_arguments(call: dict) -> Optional[Mapping]:
    arguments = call.get("arguments")
    if isinstance(arguments, Mapping) and _json_shaped(arguments):
        return arguments
    return None
