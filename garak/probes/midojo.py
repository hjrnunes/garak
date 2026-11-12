# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drive injection-spec documents as serial, multi-turn MiDojo evaluations.

Each source entry holds an ``injection_spec`` document and optional opaque
``metadata``. The authored plan supplies the payload unchanged. One handle
runs each chain; only the final attempt is graded. The final attempt carries
the sealed evaluation in ``notes["evaluation_trace"]``. No handle is stored
on an attempt, which the iterative base deep-copies between turns.
"""

from __future__ import annotations

import copy
import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable
from urllib.parse import urlsplit

import garak.attempt
import garak.probes
from garak import _config
from garak.cli import redact_credentials
from garak.exception import ConfigFailure, GarakException
from garak.probes._injection_shared import resolve_source, turn_failure

CHAIN_NOTE = "injection_spec_chain"
TRACE_NOTE = "evaluation_trace"
METADATA_NOTE = "source_metadata"
CONDITION_NOTE = "tool_call_condition"
INJECTION_SPEC_KEY = "injection_spec"
EMPTY_REPLY = "EmptyReply"

_EXTRA_HINT = (
    "the injection-spec probe needs the MiDojo harness (Python 3.12 or newer); "
    "install the 'midojo-runtime' extra with a trusted source pin, never the "
    "unrelated 'midojo' package from the package index"
)
_WRONG_PACKAGE = (
    "the installed 'midojo' is not the MiDojo harness "
    "(missing midojo.injection_spec or midojo.evaluation_handle); "
    "the plain package-index name belongs to an unrelated package"
)


def _midojo() -> SimpleNamespace:
    """Import the runtime only when the probe runs; reject a wrong distribution."""
    try:
        from midojo import evaluation_handle
        from midojo.control_plane.routers.runs import snapshot_response
        from midojo.delivery import delivery_facts, exposure
        from midojo.injection_spec import InjectionSpec
        from midojo.injection_spec_document import (
            InjectionSpecDocumentError,
            SUPPORTED_DOCUMENT_VERSIONS,
            UnsupportedDocumentVersion,
        )
        from midojo.interception_host import RemoteInterception
        from midojo.runtimes import runtime_capabilities
    except ModuleNotFoundError as exc:
        if exc.name == "midojo":
            raise ConfigFailure(_EXTRA_HINT) from exc
        raise ConfigFailure(_WRONG_PACKAGE) from exc
    except ImportError as exc:
        raise ConfigFailure(_WRONG_PACKAGE) from exc
    return SimpleNamespace(
        evaluation_handle=evaluation_handle,
        snapshot_response=snapshot_response,
        delivery_facts=delivery_facts,
        exposure=exposure,
        InjectionSpec=InjectionSpec,
        InjectionSpecDocumentError=InjectionSpecDocumentError,
        SUPPORTED_DOCUMENT_VERSIONS=SUPPORTED_DOCUMENT_VERSIONS,
        UnsupportedDocumentVersion=UnsupportedDocumentVersion,
        RemoteInterception=RemoteInterception,
        runtime_capabilities=runtime_capabilities,
    )


def plain(value: Any) -> Any:
    """Convert immutable document mappings and tuples to plain JSON values."""
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return value


def load_injection_spec_entries(source: Path, owner: str) -> list[dict]:
    """Load JSON or JSONL entries, skipping malformed records with a warning."""
    contents = None
    try:
        contents = source.read_text(encoding="utf-8")
    except OSError:
        pass
    if contents is None:
        raise ConfigFailure(f"{owner}: source_path cannot be read") from None
    if source.suffix == ".jsonl":
        entries = [json.loads(line) for line in contents.splitlines() if line.strip()]
    else:
        data = json.loads(contents)
        entries = data if isinstance(data, list) else [data]
    kept = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or INJECTION_SPEC_KEY not in entry:
            logging.warning(
                "%s: skipping malformed injection-spec entry %d",
                owner,
                index,
            )
            continue
        kept.append(entry)
    return kept


@dataclass(frozen=True)
class InjectionSpecEntry:
    """One validated document, its metadata and its optional condition."""

    injection_spec: Any
    metadata: Any
    condition: Any


class InjectionSpecProbe(garak.probes.IterativeProbe):
    """Drive one MiDojo injection spec per serial chain.

    The source is a JSON or JSONL file of entries with an ``injection_spec``
    document and optional ``metadata``. Its turns become one request apiece;
    later turns carry real earlier replies. The authored plan is never rewritten.
    Failed or unfinished turns end their chain; tool-only turns can continue.
    Every handle closes after sealing, including when an exception interrupts
    the run. The target needs a Responses-style generator that accepts MCP tools
    from conversation notes and records response status and tool calls.
    """

    extra_dependency_names = ["midojo"]
    _unsafe_attributes = ["_midojo", "entries"]

    source_path: str | None
    upstream: str | None
    interceptor_url: str | None
    interceptor_credential: str | None
    repeats: int

    lang = "en"
    goal = "elicit harmful behaviour through an injected MiDojo injection spec"
    doc_uri = "https://genai.owasp.org/llmrisk/llm01-prompt-injection/"
    intent = "T009indirect"
    active = False
    tags = [
        "avid-effect:security:S0403",  # planted instructions change tool use
        "owasp:llm01",  # injection in a lower-trust surface
        "quality:Security:PromptStability",  # tests resistance to the payload
    ]
    tier = garak.probes.Tier.INFORMATIONAL
    primary_detector = "toolcall.ToolCallCondition"
    extended_detectors = ["midojo.EvaluationVerdict", "midojo.EvaluationEffect"]

    DEFAULT_PARAMS = garak.probes.IterativeProbe.DEFAULT_PARAMS | {
        "source_path": "injection/example_injection_spec.json",
        "upstream": None,
        "interceptor_url": None,
        "interceptor_credential": None,
        "repeats": 1,
        "end_condition": "verify",
        "follow_prompt_cap": False,
    }

    def __init__(self, config_root=_config):
        invalid_end_condition = False
        try:
            super().__init__(config_root=config_root)
        except ModuleNotFoundError as exc:
            if exc.name == "midojo":
                raise ConfigFailure(_EXTRA_HINT) from exc
            raise ConfigFailure(_WRONG_PACKAGE) from exc
        except ValueError:
            invalid_end_condition = True
        if invalid_end_condition:
            raise ConfigFailure(
                f"{self.__class__.__name__}: end_condition must be detector or verify"
            ) from None
        self._load_unsafe()
        self._refuse_partial_interceptor()
        self._validate_upstream()
        invalid_source = False
        if type(self.source_path) is not str or not self.source_path.strip():
            invalid_source = True
        else:
            try:
                source = resolve_source(self.source_path)
            except (GarakException, TypeError, ValueError, OSError):
                invalid_source = True
        if invalid_source:
            raise ConfigFailure(
                f"{self.__class__.__name__}: source_path must name an injection-spec file"
            ) from None
        self._entry_documents = load_injection_spec_entries(
            source, self.__class__.__name__
        )
        self.entries = [self._read_entry(entry) for entry in self._entry_documents]
        if not self.entries:
            raise ConfigFailure(f"{self.__class__.__name__}: source has no entries")
        self.generations = 1
        self.parallel_attempts = 1
        self._chains = len(self.entries) * self.repeats
        self.max_calls_per_conv = (
            sum(len(entry.injection_spec.turns) for entry in self.entries)
            * self.repeats
        )
        self._refuse_unreset_repeats()
        self._handles = {}
        self._last_processed = {}
        self._next_attempts = {}
        self._pending_starts = list(range(self._chains))
        self._sealed = set()

    def _load_unsafe(self):
        """Restore the runtime imports and documents after unpickling."""
        self._midojo = _midojo()
        self.entries = [
            self._read_entry(entry) for entry in getattr(self, "_entry_documents", [])
        ]

    def _read_entry(self, entry: dict) -> InjectionSpecEntry:
        document = entry[INJECTION_SPEC_KEY]
        supported = ", ".join(self._midojo.SUPPORTED_DOCUMENT_VERSIONS)
        if isinstance(document, Mapping):
            if "schema_version" not in document:
                raise ConfigFailure(
                    f"{self.__class__.__name__}: injection-spec document "
                    f"requires schema_version; supported: {supported}"
                )
            if type(document["schema_version"]) is not str:
                raise ConfigFailure(
                    f"{self.__class__.__name__}: injection-spec document "
                    f"schema_version must be a string; supported: {supported}"
                )
        refusal = None
        try:
            spec = self._midojo.InjectionSpec.from_document(document)
        except self._midojo.UnsupportedDocumentVersion as exc:
            version = repr(exc.version[:80])[:96]
            if len(exc.version) > 80:
                version += "..."
            refusal = (
                f"{self.__class__.__name__}: injection-spec document version "
                f"{version} is not supported; supported: {supported}"
            )
        except (self._midojo.InjectionSpecDocumentError, ValueError) as exc:
            refusal = (
                f"{self.__class__.__name__}: injection-spec document refused: {exc}"
            )
        if refusal is not None:
            raise ConfigFailure(refusal) from None
        if not spec.turns:
            raise ConfigFailure(
                f"{self.__class__.__name__}: a host-driven injection spec "
                "has no turns to send"
            )
        condition = entry.get(CONDITION_NOTE)
        if condition is None and isinstance(spec.security, Mapping):
            condition = spec.security.get(CONDITION_NOTE)
        return InjectionSpecEntry(
            spec,
            copy.deepcopy(entry.get("metadata")),
            None if condition is None else plain(condition),
        )

    def _refuse_unreset_repeats(self):
        if self.repeats <= 1:
            return
        for entry in self.entries:
            capabilities = self._midojo.runtime_capabilities(
                entry.injection_spec.runtime
            )
            if not capabilities.isolates_attempts:
                raise ConfigFailure(
                    f"{self.__class__.__name__}: repeats needs a "
                    f"resettable or disposable runtime; {capabilities.name!r} is neither"
                )

    def _refuse_partial_interceptor(self):
        if (self.interceptor_url is None) != (self.interceptor_credential is None):
            missing = (
                "interceptor_url"
                if self.interceptor_url is None
                else "interceptor_credential"
            )
            raise ConfigFailure(
                f"{self.__class__.__name__}: remote interception needs both "
                f"interceptor_url and interceptor_credential; {missing} is missing"
            )
        self._validate_interceptor_options()

    def _validate_interceptor_options(self):
        self._validate_remote_pair(self.interceptor_url, self.interceptor_credential)

    def _validate_remote_pair(self, url, credential):
        if url is None and credential is None:
            return
        if (
            type(credential) is not str
            or not credential.strip()
            or any(ord(char) < 32 or ord(char) == 127 for char in credential)
        ):
            raise ConfigFailure(
                f"{self.__class__.__name__}: interceptor_credential "
                "must be a non-empty string"
            )
        valid = False
        if (
            type(url) is str
            and url
            and not any(
                char.isspace() or ord(char) < 32 or ord(char) == 127 for char in url
            )
        ):
            try:
                parsed = urlsplit(url)
                valid = (
                    parsed.scheme in ("http", "https", "ws", "wss")
                    and bool(parsed.hostname)
                    and parsed.port != 0
                    and parsed.username is None
                    and parsed.password is None
                    and not parsed.query
                    and not parsed.fragment
                )
            except ValueError:
                valid = False
        if not valid:
            raise ConfigFailure(
                f"{self.__class__.__name__}: interceptor_url must be a "
                "non-empty HTTP(S) or WS(S) URL without user info, query or fragment"
            )

    def _validate_upstream(self):
        url = self.upstream
        if not isinstance(url, str):
            return
        valid = False
        if url and not any(
            char.isspace() or ord(char) < 32 or ord(char) == 127 for char in url
        ):
            try:
                parsed = urlsplit(url)
                valid = (
                    parsed.scheme in ("http", "https")
                    and bool(parsed.hostname)
                    and parsed.port != 0
                    and parsed.username is None
                    and parsed.password is None
                )
            except ValueError:
                pass
        if not valid:
            raise ConfigFailure(
                f"{self.__class__.__name__}: upstream must be an HTTP(S) URL "
                "with a host and no user info"
            ) from None

    def _interception_option(self):
        self._refuse_partial_interceptor()
        if self.interceptor_url is None:
            from midojo_spec.link import CREDENTIAL_ENV, URL_ENV

            url, credential = os.environ.get(URL_ENV), os.environ.get(CREDENTIAL_ENV)
            if (url is None) != (credential is None):
                raise ConfigFailure(
                    f"{self.__class__.__name__}: remote interception environment "
                    "needs both interceptor_url and interceptor_credential"
                ) from None
            if url is not None:
                # Apply the same admission without keeping environment secrets on the probe.
                self._validate_remote_pair(url, credential)
            return {}
        self._validate_remote_pair(self.interceptor_url, self.interceptor_credential)
        interception = None
        try:
            interception = self._midojo.RemoteInterception(
                self.interceptor_url, self.interceptor_credential
            )
        except (TypeError, ValueError):
            pass
        if interception is None:
            raise ConfigFailure(
                f"{self.__class__.__name__}: remote interceptor options are invalid"
            ) from None
        return {"interception": interception}

    def _start_chain(self, chain: int) -> garak.attempt.Attempt:
        if self.end_condition not in ("detector", "verify"):
            raise ConfigFailure(
                f"{self.__class__.__name__}: end_condition must be detector or verify"
            )
        entry = self.entries[chain // self.repeats]
        repeat = chain % self.repeats
        self._validate_upstream()
        handle = None
        try:
            handle = self._midojo.evaluation_handle.open(
                entry.injection_spec,
                upstream=self.upstream,
                reset=repeat > 0,
                **self._interception_option(),
            )
        except self._midojo.InjectionSpecDocumentError:
            raise ConfigFailure(
                f"{self.__class__.__name__}: no upstream to intercept"
            ) from None
        except (TypeError, ValueError):
            pass
        if handle is None:
            raise ConfigFailure(
                f"{self.__class__.__name__}: evaluation could not open"
            ) from None
        self._handles[chain] = handle
        handle.start_turn(entry.injection_spec.turns[0])
        return self._turn_attempt(chain, 0, [], handle)

    def _create_init_attempts(self) -> Iterable[garak.attempt.Attempt]:
        return [self._start_chain(self._pending_starts.pop(0))]

    def _generate_next_attempts(self, last_attempt) -> Iterable[garak.attempt.Attempt]:
        minted = self._next_attempts.pop(last_attempt.notes[CHAIN_NOTE]["chain"], [])
        if minted:
            return minted
        if self._pending_starts:
            return [self._start_chain(self._pending_starts.pop(0))]
        return []

    def _turn_attempt(self, chain: int, index: int, earlier: list, handle):
        entry = self.entries[chain // self.repeats]
        notes = {"tools": [redact_credentials(handle.tool_block())]}
        if entry.condition is not None:
            notes[CONDITION_NOTE] = copy.deepcopy(entry.condition)
        user = garak.attempt.Turn(
            "user",
            garak.attempt.Message(entry.injection_spec.turns[index], lang=self.lang),
        )
        attempt = self._create_attempt(
            garak.attempt.Conversation([*earlier, user], notes=notes)
        )
        attempt.notes[CHAIN_NOTE] = {
            "chain": chain,
            "turn": index,
            "turns": len(entry.injection_spec.turns),
            "final": index == len(entry.injection_spec.turns) - 1,
        }
        attempt.notes[METADATA_NOTE] = copy.deepcopy(entry.metadata)
        return attempt

    def _execute_attempt(self, this_attempt):
        """Give the generator the live block without putting it on a report-bound attempt."""
        self._generator_precall_hook(self.generator, this_attempt)
        request = copy.deepcopy(this_attempt.prompt)
        chain = this_attempt.notes[CHAIN_NOTE]["chain"]
        request.notes["tools"] = [self._handles[chain].tool_block()]
        this_attempt.outputs = self.generator.generate(
            request, generations_this_call=self.generations
        )
        if self.post_buff_hook:
            this_attempt = self._postprocess_buff(this_attempt)
        this_attempt = self._postprocess_hook(this_attempt)
        self._generator_cleanup()
        return copy.deepcopy(this_attempt)

    @staticmethod
    def _turn_failure(output):
        failure = turn_failure(output)
        if failure == EMPTY_REPLY and (output.notes or {}).get("tool_calls"):
            return None
        return failure

    def _postprocess_attempt(self, this_attempt):
        trace = self._advance_chain(this_attempt)
        processed = super()._postprocess_attempt(this_attempt)
        self._last_processed[this_attempt.notes[CHAIN_NOTE]["chain"]] = processed
        if trace is not None:
            self._attach_trace(processed, trace)
        return processed

    def _advance_chain(self, attempt):
        state = attempt.notes[CHAIN_NOTE]
        chain, index = state["chain"], state["turn"]
        handle = self._handles[chain]
        entry = self.entries[chain // self.repeats]
        output = attempt.outputs[0] if attempt.outputs else None
        if output is not None and not handle.closed and not handle.sealed:
            handle.record_reply(index, output.text or "")
        failure = self._turn_failure(output)
        if failure is None and not state["final"]:
            earlier = copy.deepcopy(attempt.prompt.turns)
            if output is not None and output.text:
                earlier.append(
                    garak.attempt.Turn("assistant", garak.attempt.Message(output.text))
                )
            handle.start_turn(entry.injection_spec.turns[index + 1])
            self._next_attempts[chain] = [
                self._turn_attempt(chain, index + 1, earlier, handle)
            ]
            return None
        state["final"] = True
        self._sealed.add(chain)
        try:
            trace = self._seal_trace(handle, entry.injection_spec.plan)
        finally:
            release_error = self._close_handle(chain)
        if release_error is not None:
            trace["release_error"] = release_error
        if failure is not None:
            trace["error"] = failure
            trace["response_status"] = "incomplete"
        return trace

    def _seal_trace(self, handle, plan):
        return self._trace(handle, handle.seal(), sealed="final", plan=plan)

    def _trace(self, handle, snapshot, *, sealed: str, plan=None):
        """Describe the sealed snapshot, with its delivery and exposure facts."""
        verdict = handle.grade()
        served = self._midojo.snapshot_response(snapshot).model_dump(mode="json")
        complete = self._capture_complete(snapshot)
        calls = [
            {
                "call_id": call.call_id,
                "name": call.function,
                "arguments": copy.deepcopy(call.args),
                "result": call.result,
                "error": call.error,
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
            "release_error": None,
            "tool_calls": calls,
            "delivery": self._delivery(plan, snapshot),
            "exposure": self._exposure(plan, snapshot),
            "response_status": (
                "completed"
                if complete and verdict.grade.injection_delivered is True
                else "incomplete"
            ),
        }

    def _delivery(self, plan, snapshot):
        if plan is None:
            return None
        return [
            fact.model_dump(mode="json")
            for fact in self._midojo.delivery_facts(plan, snapshot.function_calls)
        ]

    def _exposure(self, plan, snapshot):
        if plan is None:
            return None
        return self._midojo.exposure(
            plan,
            snapshot.function_calls,
            turns=len(snapshot.turns),
            evidence_status=snapshot.evidence_status,
        ).model_dump(mode="json")

    @staticmethod
    def _capture_complete(snapshot):
        from garak.detectors.midojo import _capture_problem

        evidence = {
            "evidence_status": dict(snapshot.evidence_status),
            "capture_failures": list(snapshot.capture_failures),
            "function_calls": [
                {"complete": call.complete} for call in snapshot.function_calls
            ],
        }
        return _capture_problem({"capture_complete": True}, evidence) is None

    def _attach_trace(self, attempt, trace):
        attempt.notes[TRACE_NOTE] = trace
        output = self._stored_output(attempt)
        if output is None:
            output = garak.attempt.Message("")
            self._store_output(attempt, output)
        original = (output.notes or {}).get("tool_calls")
        preserved = (output.notes or {}).get("generator_tool_calls")
        output.notes = {
            **(output.notes or {}),
            **(
                {"generator_tool_calls": copy.deepcopy(original)}
                if original is not None and preserved is None
                else {}
            ),
            "tool_calls": copy.deepcopy(trace["tool_calls"]),
            "response_status": self._capture_status(trace),
        }

    @staticmethod
    def _capture_status(trace):
        if trace["sealed"] != "final" or trace["error"] is not None:
            return "incomplete"
        if not trace["capture_complete"]:
            return "incomplete"
        delivered = ((trace.get("verdict") or {}).get("grade") or {}).get(
            "injection_delivered"
        )
        if delivered is True:
            return "completed"
        if delivered is None:
            return "unrecorded_delivery"
        return "incomplete"

    @staticmethod
    def _stored_output(attempt):
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
    def _store_output(attempt, output):
        assistant = [
            turn for turn in attempt.conversations[0].turns if turn.role == "assistant"
        ]
        if assistant:
            assistant[-1].content = output
        else:
            attempt.conversations[0].turns.append(
                garak.attempt.Turn("assistant", output)
            )

    def probe(self, generator):
        """Run every chain and close all handles even when a turn raises."""
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

    def _seal_interrupted(self):
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
            except (RuntimeError, ValueError):
                logging.warning(
                    "InjectionSpecProbe: couldn't seal interrupted chain %d", chain
                )

    def _interrupted_trace(self, handle):
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
                "release_error": None,
                "tool_calls": [],
                "delivery": None,
                "exposure": None,
                "response_status": "incomplete",
            }
        return self._trace(handle, snapshot, sealed="interrupted")

    def _close_handle(self, chain):
        handle = self._handles.pop(chain, None)
        if handle is None:
            return None
        try:
            handle.close()
        except (RuntimeError, ValueError):
            logging.warning("InjectionSpecProbe: couldn't close chain %d", chain)
        error = getattr(handle, "close_error", None)
        return getattr(error, "reason", None)

    def _close_handles(self):
        for chain in list(self._handles):
            self._close_handle(chain)
