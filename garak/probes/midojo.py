# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drive injection-spec documents as serial, multi-turn evaluations.

Each source entry holds an ``injection_spec`` document and optional opaque
``metadata``. The authored plan supplies the payload unchanged. One chain scope
of the run coordinator runs each entry; only the final attempt is graded. The
final attempt carries the sealed evaluation in ``notes["evaluation_trace"]``.
No chain scope is stored on an attempt, which the iterative base deep-copies
between turns.
"""

from __future__ import annotations

import contextlib
import copy
import json
import logging
import math
import os
import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, NamedTuple
from urllib.parse import urlsplit

import garak.attempt
import garak.probes
from garak import _config
from garak.cli import credential_values, redact_credentials, scrub_secrets
from garak.coordinator import (
    ContextRequest,
    RunCoordinator,
    TurnTimeout,
    select_components,
)
from garak.detectors.grading import GradingRefused, grade_sealed
from garak.evidence.collector import Collector
from garak.evidence.delivery import delivery_facts, exposure
from garak.evidence.facts import AGENT_OUTPUT, FUNCTION_CALLS, TURNS
from garak.evidence.injection_spec import (
    SUPPORTED_DOCUMENT_VERSIONS,
    InjectionSpec,
    InjectionSpecDocumentError,
    UnsupportedDocumentVersion,
)
from garak.evidence.records import RejectedError
from garak.evidence.trace import TraceRefused, snapshot_value
from garak.exception import CapabilityError, ConfigFailure, GarakException
from garak.probes._injection_shared import resolve_source, turn_failure

CHAIN_NOTE = "injection_spec_chain"
TRACE_NOTE = "evaluation_trace"
METADATA_NOTE = "source_metadata"
CONDITION_NOTE = "tool_call_condition"
INJECTION_SPEC_KEY = "injection_spec"
EMPTY_REPLY = "EmptyReply"
MAX_PAYLOAD_BYTES = 8192

# What a chain cut short keeps unknown: the calls it did not see, the turns it
# did not finish and the agent's last words.
CUT_SHORT = (FUNCTION_CALLS, TURNS, AGENT_OUTPUT)
INJECTOR_NEEDS = frozenset({"mcp_tools", "tool_output", "binding_stream"})
# Options the probe read before the injector took the binding over, and where each one is set now.
MOVED_OPTIONS = {
    "interceptor_url": "plugins.injectors.mim.service_url",
    "interceptor_credential": "plugins.injectors.mim.bind_credential",
}
_CLOSURE_ERRORS = (
    CapabilityError,
    RuntimeError,
    OSError,
    TimeoutError,
    ValueError,
    ExceptionGroup,
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


@dataclass
class _Chain:
    """One live chain: its scope, what it last heard and the deadline that bounds it."""

    entry: InjectionSpecEntry
    evaluation_id: str
    stack: contextlib.ExitStack
    scope: Any
    reply: str = ""
    failure: str | None = None
    timer: threading.Timer | None = None
    expired: threading.Event = field(default_factory=threading.Event)

    def stop_deadline(self):
        if self.timer is not None:
            self.timer.cancel()


class _Sealed(NamedTuple):
    """What closing one chain left: its snapshot, how its release went, and why it has none."""

    snapshot: Any
    release_error: str | None
    cleanup_failures: list
    error: str | None


class InjectionSpecProbe(garak.probes.IterativeProbe):
    """Drive one injection spec per serial chain.

    The source is a JSON or JSONL file of entries with an ``injection_spec``
    document and optional ``metadata``. Its turns become one request apiece;
    later turns carry real earlier replies. The authored plan is never rewritten.
    Failed or unfinished turns end their chain; tool-only turns can continue.
    Every chain is released after sealing, including when an exception
    interrupts the run, and a chain cut short seals with its calls, turns and
    agent output marked incomplete. The run coordinator takes the injector
    (``plugins.injector_type``, default ``mim``) and the environment
    (``plugins.environment_type``, default ``unmanaged``) from the plugin
    configuration. The target needs a Responses-style generator that accepts MCP
    tools from conversation notes and records response status and tool calls.
    """

    extra_dependency_names = ["midojo_spec"]
    _unsafe_attributes = ["entries"]

    source_path: str | None
    upstream: str | None
    repeats: int
    chain_timeout: float | None
    turn_timeout: float

    lang = "en"
    goal = "elicit harmful behaviour through an injected injection spec"
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
        "repeats": 1,
        "chain_timeout": None,
        "turn_timeout": 60,
        "end_condition": "verify",
        "follow_prompt_cap": False,
    }

    def __init__(self, config_root=_config):
        invalid_end_condition = False
        try:
            super().__init__(config_root=config_root)
        except ValueError:
            invalid_end_condition = True
        if invalid_end_condition:
            raise ConfigFailure(
                f"{self.__class__.__name__}: end_condition must be detector or verify"
            ) from None
        self._config_root = None if config_root is _config else config_root
        self._check_options()
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
        self.max_calls_per_conv = (
            sum(len(entry.injection_spec.turns) for entry in self.entries)
            * self.repeats
        )
        self._reset_state()

    def _reset_state(self):
        self._coordinator = None
        self._live = {}
        self._opening = {}
        self._presented = []
        self._chain_secrets = {}
        self._last_processed = {}
        self._next_attempts = {}
        self._pending_starts = list(range(len(self.entries) * self.repeats))
        self._sealed = set()
        self._refusal = None

    def _load_unsafe(self):
        """Restore the documents after unpickling."""
        self.entries = [
            self._read_entry(entry) for entry in getattr(self, "_entry_documents", [])
        ]

    def _check_options(self):
        """Refuse an option that moved or holds the wrong kind of value, without echoing it."""
        self._refuse_moved_options()
        self._validate_options()

    def _refuse_moved_options(self):
        given = [old for old in MOVED_OPTIONS if hasattr(self, old)]
        if given:
            raise ConfigFailure(
                f"{self.__class__.__name__}: {', '.join(given)} no longer "
                "configure this probe; set "
                f"{', '.join(MOVED_OPTIONS[old] for old in given)} for the "
                "injector (or the MIDOJO_INTERCEPTOR_URL and "
                "MIDOJO_INTERCEPTOR_CREDENTIAL environment variables)"
            )

    def _validate_options(self):
        owner = self.__class__.__name__
        if type(self.repeats) is not int or self.repeats < 1:
            raise ConfigFailure(f"{owner}: repeats must be a positive integer")
        for name, optional in (("chain_timeout", True), ("turn_timeout", False)):
            value = getattr(self, name)
            if value is None and optional:
                continue
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ConfigFailure(
                    f"{owner}: {name} must be a finite positive number of seconds"
                    + (" or null" if optional else "")
                )
        if self.upstream is not None:
            self._check_upstream(self.upstream, "upstream")

    def _check_upstream(self, url, where):
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
                f"{self.__class__.__name__}: {where} must be an HTTP(S) URL "
                "with a host and no user info"
            ) from None

    def _upstream(self, spec):
        """The endpoint the injector forwards to: the option, else the document's own."""
        if self.upstream is not None:
            return self.upstream
        url = spec.interception.upstream
        if url is None:
            raise ConfigFailure(
                f"{self.__class__.__name__}: no upstream to intercept; set upstream "
                "or the document's interception.upstream"
            )
        self._check_upstream(url, "the document's interception.upstream")
        return url

    def _read_entry(self, entry: dict) -> InjectionSpecEntry:
        document = entry[INJECTION_SPEC_KEY]
        supported = ", ".join(SUPPORTED_DOCUMENT_VERSIONS)
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
        self._admit_payloads(document)
        refusal = None
        try:
            spec = InjectionSpec.from_document(document)
        except UnsupportedDocumentVersion as exc:
            version = repr(exc.version[:80])[:96]
            if len(exc.version) > 80:
                version += "..."
            refusal = (
                f"{self.__class__.__name__}: injection-spec document version "
                f"{version} is not supported; supported: {supported}"
            )
        except (InjectionSpecDocumentError, ValueError) as exc:
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

    @staticmethod
    def _admit_payloads(document):
        """Refuse unsafe plan payloads before a chain opens or payloads reach logs."""
        plan = document.get("plan") if isinstance(document, dict) else None
        instructions = plan.get("instructions") if isinstance(plan, dict) else None
        if not isinstance(instructions, list):
            return
        for index, instruction in enumerate(instructions):
            if not isinstance(instruction, dict):
                continue
            payload = instruction.get("payload")
            if type(payload) is not str or not payload:
                raise ConfigFailure(
                    f"InjectionSpecProbe: plan instruction {index} needs a non-empty string payload"
                )
            try:
                size = len(payload.encode("utf-8"))
            except UnicodeEncodeError as exc:
                raise ConfigFailure(
                    f"InjectionSpecProbe: plan instruction {index} needs UTF-8 payload text"
                ) from exc
            if size > MAX_PAYLOAD_BYTES:
                raise ConfigFailure(
                    f"InjectionSpecProbe: plan instruction {index} exceeds "
                    f"the {MAX_PAYLOAD_BYTES}-byte payload cap"
                )

    def _new_evaluation_id(self) -> str:
        """An opaque name for one chain's evaluation."""
        return uuid.uuid4().hex

    def _new_collector(self, evaluation_id: str, spec) -> Collector:
        """The collector one chain records into: the document's provenance, no modelled environment."""
        return Collector(
            evaluation_id,
            user_task_id=spec.user_task_id,
            injection_task_id=spec.injection_task_id,
            interception=spec.interception.mode,
            pre_environment={},
        )

    def _collector_for(self, evaluation_id: str) -> Collector:
        return self._new_collector(evaluation_id, self._opening[evaluation_id])

    @staticmethod
    def _tool_block(block) -> dict:
        """The tool block a Responses-style target takes for the chain's endpoint."""
        return {
            "type": "mcp",
            "server_label": "midojo",
            "server_url": block.tool_url,
            "authorization": block.tool_token,
            "require_approval": "never",
            "allowed_tools": list(block.tools),
        }

    def _presented_credential(self):
        """The credential the injector presents to its service, if one is set."""
        injector = getattr(self._coordinator, "injector", None)
        configured = getattr(injector, "bind_credential", None)
        if configured:
            return configured
        return os.environ.get("MIDOJO_INTERCEPTOR_CREDENTIAL")

    def _sample_credential(self):
        """Remember a presented credential: the environment can change before a trace comes back."""
        value = self._presented_credential()
        if isinstance(value, str) and value.strip() and value not in self._presented:
            self._presented.append(value)

    def _refuse(self, reason: str, secrets=()):
        """Stop the run for ``reason``; the iterative base only logs what is raised in its loop.

        An injector's refusal is its own text, which can repeat what it was handed,
        so the reason leaves with every credential the probe knows masked.
        """
        reason = scrub_secrets(reason, [*self._presented, *secrets])
        self._refusal = f"{self.__class__.__name__}: {reason}"
        raise ConfigFailure(self._refusal)

    def _start_chain(self, chain: int) -> garak.attempt.Attempt:
        if self.end_condition not in ("detector", "verify"):
            raise ConfigFailure(
                f"{self.__class__.__name__}: end_condition must be detector or verify"
            )
        entry = self.entries[chain // self.repeats]
        repeat = chain % self.repeats
        spec = entry.injection_spec
        upstream = self._upstream(spec)
        evaluation_id = self._new_evaluation_id()
        self._sample_credential()
        scope = self._coordinator.chain(
            evaluation_id,
            upstream_url=upstream,
            plan=spec.plan,
            policy=spec.tools,
            repeated=repeat > 0,
            reset=repeat > 0,
        )
        stack = contextlib.ExitStack()
        self._opening[evaluation_id] = spec
        refusal = None
        try:
            stack.enter_context(scope)
        except (*_CLOSURE_ERRORS, TypeError) as exc:
            # only a capability refusal's text is the injector's own; any other
            # error can carry what the injector was handed, so it is named by type
            refusal = str(exc) if type(exc) is CapabilityError else type(exc).__name__
        finally:
            self._opening.pop(evaluation_id, None)
        if refusal is not None:
            self._sample_credential()
            block = scope.tool_block
            self._refuse(
                f"evaluation could not open: {refusal}",
                () if block is None else credential_values(self._tool_block(block)),
            )
        if scope.tool_block is None:
            stack.close()
            self._refuse("the injector gave no tool endpoint")
        # what the chain presented is fixed here: the environment can change
        # before the trace comes back, and the service echoes what it received.
        # The credential is sampled on both sides of the open, which reads it.
        self._sample_credential()
        self._chain_secrets[chain] = [
            *credential_values(self._tool_block(scope.tool_block)),
            *self._presented,
        ]
        state = _Chain(entry, evaluation_id, stack, scope)
        self._live[chain] = state
        self._arm_deadline(state)
        return self._turn_attempt(chain, 0, [], state)

    def _arm_deadline(self, state: _Chain):
        if self.chain_timeout is None:
            return
        state.timer = threading.Timer(self.chain_timeout, self._expire, args=(state,))
        state.timer.daemon = True
        state.timer.start()

    def _expire(self, state: _Chain):
        """Close a chain whose deadline passed; its turn in flight ends when the target returns."""
        state.expired.set()
        try:
            state.scope.cancel("deadline")
        except _CLOSURE_ERRORS as exc:
            logging.warning(
                "%s: chain deadline closure failed: %s",
                self.__class__.__name__,
                type(exc).__name__,
            )

    def _create_init_attempts(self) -> Iterable[garak.attempt.Attempt]:
        return [self._start_chain(self._pending_starts.pop(0))]

    def _generate_next_attempts(self, last_attempt) -> Iterable[garak.attempt.Attempt]:
        minted = self._next_attempts.pop(last_attempt.notes[CHAIN_NOTE]["chain"], [])
        if minted:
            return minted
        if self._pending_starts:
            return [self._start_chain(self._pending_starts.pop(0))]
        return []

    def _turn_attempt(self, chain: int, index: int, earlier: list, state: _Chain):
        entry = state.entry
        notes = {
            "tools": [redact_credentials(self._tool_block(state.scope.tool_block))]
        }
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
        """Run one turn in its scope, giving the generator the live block off the report-bound attempt."""
        note = this_attempt.notes[CHAIN_NOTE]
        state = self._live[note["chain"]]
        self._generator_precall_hook(self.generator, this_attempt)
        request = copy.deepcopy(this_attempt.prompt)
        request.notes["tools"] = [self._tool_block(state.scope.tool_block)]
        state.reply = ""
        try:
            with state.scope.turn(
                state.entry.injection_spec.turns[note["turn"]]
            ) as turn:
                this_attempt.outputs = self.generator.generate(
                    request, generations_this_call=self.generations
                )
                if self.post_buff_hook:
                    this_attempt = self._postprocess_buff(this_attempt)
                this_attempt = self._postprocess_hook(this_attempt)
                self._generator_cleanup()
                output = this_attempt.outputs[0] if this_attempt.outputs else None
                if output is not None:
                    reply = output.text or ""
                    turn.reply(reply)
                    state.reply = reply
        except (TurnTimeout, RejectedError, RuntimeError) as exc:
            # a call the target makes after the turn's deadline is refused by the
            # closed chain, so the deadline's record names the cause
            failed = state.scope.result["failed_turn"] or {}
            if isinstance(exc, TurnTimeout) or failed.get("kind") == "timeout":
                state.failure = "TurnTimeout"
            elif state.expired.is_set() or state.scope.closed:
                state.failure = "ChainTimeout"
            else:
                raise
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
        note = attempt.notes[CHAIN_NOTE]
        chain, index = note["chain"], note["turn"]
        state = self._live[chain]
        output = attempt.outputs[0] if attempt.outputs else None
        failure = state.failure or self._turn_failure(output)
        if failure is None and state.expired.is_set():
            failure = "ChainTimeout"
        if failure is None and not note["final"]:
            earlier = copy.deepcopy(attempt.prompt.turns)
            if output is not None and output.text:
                earlier.append(
                    garak.attempt.Turn("assistant", garak.attempt.Message(output.text))
                )
            self._next_attempts[chain] = [
                self._turn_attempt(chain, index + 1, earlier, state)
            ]
            return None
        note["final"] = True
        self._sealed.add(chain)
        sealed = self._finish_chain(chain, state, CUT_SHORT if failure else ())
        if sealed.snapshot is None:
            trace = self._empty_trace(state, "final", sealed.error)
        else:
            trace = self._trace(
                state,
                sealed.snapshot,
                sealed="final",
                plan=state.entry.injection_spec.plan,
            )
        trace["release_error"] = sealed.release_error
        trace["cleanup_failures"] = sealed.cleanup_failures
        if failure is not None:
            trace["error"] = failure
            trace["response_status"] = "incomplete"
        return trace

    def _finish_chain(self, chain: int, state: _Chain, incomplete: tuple) -> _Sealed:
        """Release and seal one chain: what it left, however its release went."""
        snapshot, error = None, None
        try:
            snapshot = state.scope.seal(state.reply, incomplete=incomplete)
        except _CLOSURE_ERRORS as exc:
            snapshot = state.scope.result["snapshot"]
            error = type(exc).__name__
        finally:
            state.stop_deadline()
            self._live.pop(chain, None)
        try:
            state.stack.close()
        except _CLOSURE_ERRORS:
            logging.warning(
                "%s: couldn't close chain %d", self.__class__.__name__, chain
            )
        cleanup = list(state.scope.result["cleanup_failures"])
        release_error = None
        if "CapabilityError" in cleanup:
            from midojo_spec.link import LINK_LOST_REASON

            release_error = LINK_LOST_REASON
        return _Sealed(snapshot, release_error, cleanup, error or "NoSnapshot")

    def _values(self, spec, snapshot):
        """The trace's snapshot and verdict, or the refusal that left one out."""
        try:
            value = snapshot_value(snapshot)
        except TraceRefused as exc:
            return None, None, type(exc).__name__
        try:
            return value, grade_sealed(spec, snapshot).to_document(), None
        except GradingRefused as exc:
            return value, None, type(exc).__name__

    def _trace(self, state, snapshot, *, sealed: str, plan=None):
        """Describe the sealed snapshot, with its delivery and exposure facts."""
        value, verdict, error = self._values(state.entry.injection_spec, snapshot)
        complete = self._capture_complete(snapshot)
        calls = [
            {
                "call_id": call.call_id,
                "name": call.function,
                "arguments": plain(call.args),
                "result": call.result,
                "error": call.error,
                "blocked": call.blocked,
                "disposition": call.disposition,
                "complete": call.complete,
                "turn": call.turn,
            }
            for call in snapshot.function_calls
        ]
        delivered = ((verdict or {}).get("grade") or {}).get("injection_delivered")
        return {
            "evaluation_id": state.evaluation_id,
            "snapshot_id": snapshot.snapshot_id,
            "snapshot": value,
            "verdict": verdict,
            "capture_complete": complete,
            "sealed": sealed,
            "failure": None,
            "error": error,
            "release_error": None,
            "cleanup_failures": [],
            "tool_calls": calls,
            "delivery": self._delivery(plan, snapshot),
            "exposure": self._exposure(plan, snapshot),
            "response_status": (
                "completed" if complete and delivered is True else "incomplete"
            ),
        }

    @staticmethod
    def _empty_trace(state, sealed: str, error: str):
        """The trace of a chain that left no snapshot to describe."""
        return {
            "evaluation_id": state.evaluation_id,
            "snapshot_id": None,
            "snapshot": None,
            "verdict": None,
            "capture_complete": False,
            "sealed": sealed,
            "failure": None,
            "error": error,
            "release_error": None,
            "cleanup_failures": [],
            "tool_calls": [],
            "delivery": None,
            "exposure": None,
            "response_status": "incomplete",
        }

    @staticmethod
    def _delivery(plan, snapshot):
        if plan is None:
            return None
        return [
            fact.to_json() for fact in delivery_facts(plan, snapshot.function_calls)
        ]

    @staticmethod
    def _exposure(plan, snapshot):
        if plan is None:
            return None
        return exposure(
            plan,
            snapshot.function_calls,
            turns=len(snapshot.turns),
            evidence_status=snapshot.evidence_status,
        ).to_json()

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

    def _trace_secrets(self, attempt):
        """Return the credentials a tool or the service could echo into a trace.

        A trace holds what the tools returned and what the service reported, and
        either can repeat a credential it received: the chain's tool bearer, or
        the credential the injector presents to the service.
        """
        chain = (attempt.notes.get(CHAIN_NOTE) or {}).get("chain")
        return [s for s in self._chain_secrets.get(chain, ()) if isinstance(s, str)]

    def _attach_trace(self, attempt, trace):
        """Keep ``trace`` on the attempt, scrubbed once here of every credential it could hold."""
        trace = scrub_secrets(trace, self._trace_secrets(attempt))
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

    def _refuse_repeats(self, environment):
        """Refuse repeated chains against an environment that cannot isolate attempts."""
        refusal = None
        try:
            environment.check_attempt(repeated=True)
        except CapabilityError as exc:
            refusal = str(exc)
        if refusal is not None:
            raise ConfigFailure(
                f"{self.__class__.__name__}: repeats refused: {refusal}"
            )

    def probe(self, generator):
        """Run every chain and release all of them even when a turn raises."""
        self._check_options()
        self._reset_state()
        # a refusal inside the iterative base's loop is logged, not raised
        for entry in self.entries:
            self._upstream(entry.injection_spec)
        request = ContextRequest(
            injector="mim",
            environment="unmanaged",
            required_injector=INJECTOR_NEEDS,
        )
        unavailable = None
        try:
            injector, environment = select_components(
                request, config_root=self._config_root
            )
        except (GarakException, ValueError) as exc:
            unavailable = type(exc.__cause__ or exc).__name__
        if unavailable is not None:
            raise ConfigFailure(
                f"{self.__class__.__name__}: the selected injector or environment "
                f"could not load ({unavailable}); the default 'mim' injector needs "
                "the 'midojo-mim' extra"
            )
        coordinator = RunCoordinator(
            injector,
            environment,
            collector_factory=self._collector_for,
            max_live=1,
            turn_timeout=self.turn_timeout,
            request=request,
            generator=generator,
        )
        attempts = None
        try:
            with coordinator:
                self._coordinator = coordinator
                self._sample_credential()
                try:
                    if self.repeats > 1:
                        self._refuse_repeats(environment)
                    attempts = super().probe(generator)
                finally:
                    try:
                        self._seal_interrupted()
                    finally:
                        for state in list(self._live.values()):
                            state.stop_deadline()
                        self._coordinator = None
        except _CLOSURE_ERRORS as exc:
            if attempts is None:
                raise
            logging.warning(
                "%s: the injector run did not close cleanly: %s",
                self.__class__.__name__,
                type(exc).__name__,
            )
        if not attempts and self._refusal is not None:
            raise ConfigFailure(self._refusal)
        return [
            attempt
            for attempt in attempts
            if (attempt.notes.get(CHAIN_NOTE) or {}).get("final")
        ]

    def _seal_interrupted(self):
        for chain, state in list(self._live.items()):
            if chain in self._sealed:
                continue
            self._sealed.add(chain)
            try:
                sealed = self._finish_chain(chain, state, CUT_SHORT)
                attempt = self._last_processed.get(chain)
                if attempt is None:
                    continue
                if sealed.snapshot is None:
                    trace = self._empty_trace(state, "interrupted", sealed.error)
                else:
                    trace = self._trace(state, sealed.snapshot, sealed="interrupted")
                trace["release_error"] = sealed.release_error
                trace["cleanup_failures"] = sealed.cleanup_failures
                attempt.notes[CHAIN_NOTE]["final"] = True
                self._attach_trace(attempt, trace)
            except (RuntimeError, ValueError):
                logging.warning(
                    "InjectionSpecProbe: couldn't seal interrupted chain %d", chain
                )
