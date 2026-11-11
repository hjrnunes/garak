# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared source and turn handling for scripted injection probes."""

from pathlib import Path

from garak.attempt import Message
from garak.data import path as data_path
from garak.exception import GarakException


def resolve_source(source: str) -> Path:
    """Resolve a source against the filesystem and the packaged data directory."""
    candidate = Path(source)
    if candidate.is_file():
        return candidate
    try:
        return Path(data_path / source)
    except GarakException as exc:
        raise GarakException(f"Could not find injection source '{source}'") from exc


def turn_failure(output: Message | None) -> str | None:
    """Name a rejected, unfinished or empty reply, if any."""
    notes = (output.notes or {}) if output is not None else {}
    error = notes.get("error")
    if isinstance(error, dict):
        return error.get("type") or "ResponsesRequestFailed"
    if notes.get("response_status") != "completed":
        return "ResponsesNotCompleted"
    if not output.text:
        return "EmptyReply"
    return None
