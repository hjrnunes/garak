# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Judge a captured reply against an attempt's rubric using shared judge rules."""

import importlib
import inspect
import json
import os
import threading
from collections.abc import Mapping
from contextlib import contextmanager
from functools import wraps
from urllib.parse import urlsplit
from typing import Any, List

import openai
import requests

from garak import _config
from garak.attempt import Attempt, Conversation, Message, Turn
from garak.detectors.base import Detector
from garak.detectors.judge import ModelAsJudge
from garak.detectors.toolcall import normalise_tool_call
from garak.exception import GarakException, GeneratorBackoffTrigger
from garak.generators.openai import OpenAICompatible

RUBRIC_NOTE = "judge_rubric"
RESULTS_NOTE = "rubric_judge_results"
_SCORES = {"supported": 1.0, "contradicted": 0.0}
_SUPPRESSED_PARAMS = {
    "n",
    "stop",
    "max_tokens",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
}
_MAX_ANSWER_BYTES = 1024 * 1024
# Different detectors can share one generator and its mutable chat resource.
_NATIVE_CREATE_LOCK = threading.RLock()
_NATIVE_CALL_STATE = threading.local()
_SERVICE_REASONS = frozenset(
    {
        "judge_completed",
        "judge_parse_failed",
        "judge_response_invalid",
        "judge_evidence_invalid",
        "judge_support_missing",
        "judge_support_unresolved",
    }
)
_CALL_ERRORS = frozenset(
    {
        "APITimeoutError",
        "APIConnectionError",
        "JSONDecodeError",
        "AuthenticationError",
        "PermissionDeniedError",
        "NotFoundError",
        "ConflictError",
        "UnprocessableEntityError",
        "RateLimitError",
        "InternalServerError",
        "APIStatusError",
        "GarakException",
        "OpenAIError",
    }
)


class _JudgeRetryBlocked(GarakException):
    """Stop a generator retry before it reaches the chat resource."""


@contextmanager
def _native_request_guard(rules):
    if getattr(_NATIVE_CALL_STATE, "active", False):
        raise rules.JudgeCallError("ReentrantJudgeCall")
    with _NATIVE_CREATE_LOCK:
        _NATIVE_CALL_STATE.active = True
        try:
            yield
        finally:
            _NATIVE_CALL_STATE.active = False


def build_evidence(attempt: Attempt, output: Message) -> dict:
    """Build the Garak-specific evidence view from a conversation and captured reply."""
    turns = list(attempt.prompt.turns) if attempt.prompt is not None else []
    user_indices = [i for i, turn in enumerate(turns) if turn.role == "user"]
    last_user = user_indices[-1] if user_indices else None
    user_text = turns[last_user].content.text if last_user is not None else None
    history = [
        {"role": turn.role, "content": turn.content.text if turn.content else None}
        for turn in turns[:last_user]
    ]
    notes = output.notes or {}
    messages = (
        [{"role": "assistant", "content": output.text}]
        if isinstance(output.text, str)
        else []
    )
    raw_calls = notes.get("tool_calls", [])
    availability = "captured"
    if isinstance(raw_calls, list):
        calls = [normalise_tool_call(call) for call in raw_calls]
    else:
        calls = []
        availability = "not_captured"
    complete = (
        "complete"
        if notes.get("response_status", "completed") == "completed"
        else "partial"
    )
    return {
        "user_text": user_text,
        "history": history,
        "messages": messages,
        "tool_calls": calls,
        "availability": {"messages": "captured", "tool_calls": availability},
        "completeness": {"messages": complete, "tool_calls": complete},
    }


def _shared(name: str):
    """Name a missing optional judge module without leaking its import failure."""
    try:
        module = importlib.import_module(f"midojo_spec.{name}")
    except ImportError as exc:
        raise GarakException(f"RubricJudge requires midojo_spec.{name}") from exc
    if name == "rubric_judge" and not callable(
        getattr(module, "strip_code_fence", None)
    ):
        raise GarakException(
            "RubricJudge requires midojo_spec.rubric_judge.strip_code_fence"
        )
    return module


def _record(
    rules,
    reason,
    request=None,
    *,
    verdict="unresolved",
    refs=None,
    raw=None,
    model=None,
    prompt=None,
):
    return {
        **rules.record(verdict, [] if refs is None else refs, reason, request, raw),
        "model": model,
        "prompt": prompt,
    }


def _judge_reply(rules, rubric, evidence, call, model):
    """Delegate parsing and support checks, then add Garak's sent model and prompt."""
    raw = None

    def capture(messages):
        nonlocal raw
        raw = call(messages)
        if isinstance(raw, bytes):
            try:
                raw = raw.decode("utf-8")
            except UnicodeDecodeError:
                return raw  # Let the shared rule refuse undecodable bytes.
        return raw

    try:
        result = rules.judge_output(rubric, evidence, capture)
    except (ValueError, RecursionError):
        if not isinstance(raw, str):
            raise
        request = rules.build_request(rubric, evidence)
        return _record(
            rules,
            "judge_parse_failed",
            request,
            raw=raw,
            model=model,
            prompt=rules.judge_messages(request),
        )
    return {**result, "model": model, "prompt": rules.judge_messages(result["request"])}


def _service_request(judge, rules, rubric, evidence):
    """Make a final-turn request; refuse evidence the service cannot reproduce."""
    history = evidence["history"]
    if (
        not isinstance(evidence["user_text"], str)
        or not isinstance(evidence["messages"], list)
        or len(evidence["messages"]) != 1
        or not isinstance(evidence["messages"][0]["content"], str)
        or not evidence["messages"][0]["content"]
        or len(history) % 2
        or any(
            pair[0] != {"role": "user", "content": pair[0].get("content")}
            or pair[1] != {"role": "assistant", "content": pair[1].get("content")}
            or not all(isinstance(t.get("content"), str) and t["content"] for t in pair)
            for pair in (history[i : i + 2] for i in range(0, len(history), 2))
        )
    ):
        return None
    prior = [
        {"message": history[i]["content"], "reply": history[i + 1]["content"]}
        for i in range(0, len(history), 2)
    ]
    calls = []
    for call in evidence["tool_calls"]:
        if not isinstance(call.get("name"), str):
            return None
        try:
            result = json.dumps(call["result"], ensure_ascii=False, allow_nan=False)
            json.dumps(call["arguments"], ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError, OverflowError, RecursionError):
            return None
        calls.append(
            {
                "function": call["name"],
                "args": call["arguments"],
                "result": result,
                "error": None,
                "complete": True,
                "blocked": call.get("blocked", False),
                "turn": len(prior),
            }
        )
    complete = evidence["completeness"]["messages"] == "complete"
    body = {
        "claim": rubric["question"],
        "rubric": {
            "criteria": rubric.get("criteria", []),
            "facts": rubric.get("facts", []),
            "runtime_facts": rubric.get("runtime_facts", {}),
            "turns": [t["message"] for t in prior] + [evidence["user_text"]],
            "judge_turns": "final",
        },
        "message": evidence["user_text"],
        "reply": evidence["messages"][0]["content"],
        "prior_turns": prior,
        "tool_calls": calls,
        "tool_calls_complete": complete,
    }
    try:
        request = {"schema": judge.SCHEMA, "id": judge.request_id(body), **body}
    except (TypeError, ValueError, OverflowError, RecursionError):
        return None
    if judge.request_problem(request) is not None:
        return None
    rebuilt = rules.build_evidence(
        request["message"],
        [
            turn
            for pair in prior
            for turn in (
                {"role": "user", "content": pair["message"]},
                {"role": "assistant", "content": pair["reply"]},
            )
        ],
        request["reply"],
        calls,
        complete=complete and rules.is_complete(calls),
    )
    return request if rebuilt == evidence else None


class RubricJudge(ModelAsJudge):
    """Judge a rubric question against each captured reply, preserving evidence refs.

    Each output records its verdict and request; only supported and contradicted score.
    The native backend asks an OpenAI-compatible model, service asks a judge endpoint,
    and replay uses an exchange log without contacting either.
    """

    DEFAULT_PARAMS = ModelAsJudge.DEFAULT_PARAMS | {
        "backend": "native",
        "service_url": None,
        "service_token_env": None,
        "judge_model": None,
        "replay_path": None,
    }
    extra_dependency_names = ["midojo_spec"]
    _unsafe_attributes = ["rules", "judge", "exchange", "replay"]
    doc_uri = "https://genai.owasp.org/llmrisk/llm01-prompt-injection/"
    active = True
    tags = [
        "owasp:llm01",  # an injected instruction can change a target's reply
        "quality:Security:PromptStability",  # checks reply stability against a rubric
    ]
    description = "Judge a captured reply against a per-attempt rubric"
    hit_desc = "Judge established that the answer to the rubric question is yes"
    pass_desc = "Judge established that the answer to the rubric question is no"

    def __init__(self, config_root=_config):
        # ModelAsJudge.__init__ loads a generator unconditionally; bypass that
        # step only when the backend cannot make a model request.
        Detector.__init__(self, config_root=config_root)
        self.rules = _shared("rubric_judge")
        if self.backend == "native":
            self._load_generator()
            self._apply_request_controls()
        elif self.backend == "service":
            self.judge = _shared("judge")
            self.exchange = _shared("judge_exchange")
            self.replay = _shared("judge_replay")
            url = urlsplit(self.service_url or "")
            if (
                url.scheme not in ("http", "https")
                or not url.hostname
                or url.username is not None
                or url.password is not None
                or url.fragment
                or not self.service_token_env
                or not isinstance(self.judge_model, str)
                or not self.judge_model
            ):
                raise GarakException("RubricJudge service configuration is invalid")
        elif self.backend == "replay":
            self.judge = _shared("judge")
            self.exchange = _shared("judge_exchange")
            self.replay = _shared("judge_replay")
            if (
                not self.replay_path
                or not isinstance(self.judge_model, str)
                or not self.judge_model
            ):
                raise GarakException(
                    "RubricJudge replay needs replay_path and judge_model"
                )
            try:
                self.rows = self.exchange.read_exchanges(self.replay_path)
            except (OSError, ValueError) as exc:
                raise GarakException(
                    f"RubricJudge replay cannot read exchanges ({type(exc).__name__})"
                ) from None
        else:
            raise GarakException(
                "RubricJudge backend must be native, service or replay"
            )

    def _load_unsafe(self):
        """Restore module references after Configurable unpickles a detector."""
        self.rules = _shared("rubric_judge")
        if self.backend in ("service", "replay"):
            self.judge = _shared("judge")
            self.exchange = _shared("judge_exchange")
            self.replay = _shared("judge_replay")

    def _apply_request_controls(self):
        generator = self.evaluation_generator
        self._validate_chat_generator()
        generator.temperature = 0.0
        generator.supports_multiple_generations = False
        for attr in ("vary_temp_each_call", "vary_seed_each_call"):
            if hasattr(generator, attr):
                setattr(generator, attr, False)
        generator.retry_json = False
        create = generator.generator.create
        allowed = {
            "model",
            "messages",
            "temperature",
            "max_completion_tokens",
            "response_format",
            "extra_body",
        }
        generator.suppressed_params = (
            set(getattr(generator, "suppressed_params", ()))
            | _SUPPRESSED_PARAMS
            | (set(inspect.signature(create).parameters) - allowed)
        )
        body = self.rules.request_body(generator.name, [])
        extra = {
            "max_completion_tokens": body["max_completion_tokens"],
            "response_format": body["response_format"],
        }
        if "extra_body" in inspect.signature(create).parameters:
            extra["extra_body"] = {"chat_template_kwargs": body["chat_template_kwargs"]}
        generator.extra_params = extra
        self._disable_sdk_retries()

    def _disable_sdk_retries(self):
        generator = self.evaluation_generator
        client = getattr(generator, "client", None)
        if client is not None and hasattr(client, "with_options"):
            generator.client = client.with_options(max_retries=0)
            generator.generator = generator.client.chat.completions
        self._validate_chat_generator()

    def _validate_chat_generator(self):
        generator = self.evaluation_generator
        client = getattr(generator, "client", None)
        chat = getattr(getattr(client, "chat", None), "completions", None)
        if chat is None or getattr(generator, "generator", None) != chat:
            raise GarakException(
                "RubricJudge native requires a chat.completions generator "
                "(for example, OpenAICompatible)"
            )

    def detect(self, attempt: Attempt) -> List[float | None]:
        notes = attempt.prompt.notes if attempt.prompt is not None else None
        rubric = notes.get(RUBRIC_NOTE) if isinstance(notes, Mapping) else None
        records = [
            self._judge_output(attempt, rubric, output) for output in attempt.outputs
        ]
        attempt.notes[RESULTS_NOTE] = records
        return [_SCORES.get(record["verdict"]) for record in records]

    def _judge_output(self, attempt, rubric, output):
        rules = self.rules
        if rubric is None:
            return _record(rules, "judge_not_declared")
        if (
            not isinstance(rubric, dict)
            or not isinstance(rubric.get("question"), str)
            or not rubric["question"].strip()
        ):
            return _record(rules, "judge_spec_invalid")
        if output is None:
            return _record(rules, "no_output")
        evidence = build_evidence(attempt, output)
        request = rules.build_request(rubric, evidence)
        if self.backend == "native":
            return _judge_reply(
                rules,
                rubric,
                evidence,
                self._native_call,
                self.evaluation_generator.name,
            )
        sent = _service_request(self.judge, rules, rubric, evidence)
        if sent is None:
            return _record(rules, "judge_request_not_expressible", request)
        prompt = rules.judge_messages(request)
        if self.backend == "service":
            return self._service_result(sent, request, prompt)
        return self._replay_result(sent, rubric, evidence, request)

    def _native_call(self, messages):
        conversation = Conversation(
            turns=[
                Turn(role=item["role"], content=Message(item["content"]))
                for item in messages
            ]
        )
        generator = self.evaluation_generator
        first_error = None
        blocked = False
        try:
            with _native_request_guard(self.rules):
                if isinstance(generator, OpenAICompatible):
                    if generator.client is None:
                        generator._load_unsafe()
                        self._disable_sdk_retries()
                    resource = generator.generator
                    create = resource.create
                    body = self.rules.request_body(generator.name, messages)
                    expected = {
                        **{
                            k: v for k, v in body.items() if k != "chat_template_kwargs"
                        },
                        "extra_body": {
                            "chat_template_kwargs": body["chat_template_kwargs"]
                        },
                    }
                    created = False

                    @wraps(create)
                    def create_once(*args, **kwargs):
                        nonlocal created, blocked, first_error
                        if created:
                            blocked = True
                            raise _JudgeRetryBlocked(
                                "RubricJudge blocked a repeated create"
                            )
                        created = True
                        if args or kwargs != expected:
                            raise GarakException(
                                "RubricJudge native request differs from shared body"
                            )
                        try:
                            return create(*args, **kwargs)
                        except openai.OpenAIError as exc:
                            first_error = type(exc).__name__
                            raise

                    resource.create = create_once
                    try:
                        responses = generator.generate(conversation)
                    finally:
                        resource.create = create
                else:
                    responses = generator.generate(conversation)
        except (
            openai.OpenAIError,
            GarakException,
            GeneratorBackoffTrigger,
            json.JSONDecodeError,
            ValueError,
            OSError,
        ) as exc:
            label = first_error if blocked and first_error else type(exc).__name__
            raise self.rules.JudgeCallError(label) from None
        if blocked:
            raise self.rules.JudgeCallError(first_error or "GarakException")
        if not responses or not responses[0]:
            return None
        # a generator with bounded retries answers a failed request with a
        # failed generation, which names the error as a blocked retry does
        error = (responses[0].notes or {}).get("error")
        if responses[0].text is None and isinstance(error, dict):
            if error.get("type") in _CALL_ERRORS:
                raise self.rules.JudgeCallError(error["type"])
        return responses[0].text

    def _post(self, sent):
        token = os.environ.get(self.service_token_env)
        if not token:
            raise ValueError("service token is absent")
        session = requests.Session()
        session.trust_env = False
        try:
            with session.post(
                self.service_url,
                json=sent,
                headers={"Authorization": f"Bearer {token}"},
                timeout=30,
                allow_redirects=False,
                stream=True,
            ) as response:
                response.raise_for_status()
                if response.status_code != 200:
                    raise requests.HTTPError("judge service did not return HTTP 200")
                if (
                    response.headers.get("Content-Length")
                    and int(response.headers["Content-Length"]) > _MAX_ANSWER_BYTES
                ):
                    raise ValueError("service answer too large")
                data = bytearray()
                for chunk in response.iter_content(chunk_size=8192):
                    data.extend(chunk)
                    if len(data) > _MAX_ANSWER_BYTES:
                        raise ValueError("service answer too large")
                return json.loads(data), token
        finally:
            session.close()

    def _service_result(self, sent, request, prompt):
        rules = self.rules
        try:
            answer, sent_token = self._post(sent)
        except (
            requests.RequestException,
            ValueError,
            UnicodeDecodeError,
            OSError,
            RecursionError,  # an answer nested past what the parser reads
        ) as exc:
            return _record(
                rules,
                f"judge_failed:{type(exc).__name__}",
                request,
                model=None,
                prompt=prompt,
            )

        def refuse(reason):
            return _record(rules, reason, request, model=None, prompt=prompt)

        if self.judge.response_problem(answer, sent["id"]) is not None:
            return refuse("judge_response_invalid")
        if answer.get("id") != sent["id"]:
            return refuse("judge_response_id_mismatch")
        details = answer.get("details")
        if (
            not isinstance(details, dict)
            or details.get("prompt_version") != rules.PROMPT_VERSION
        ):
            return refuse("judge_prompt_version_mismatch")
        identity = self.replay.identity_problem(
            answer.get("model"),
            details.get("prompt_version"),
            expected_model=self.judge_model,
            expected_prompt_version=rules.PROMPT_VERSION,
        )
        if identity:
            return refuse(f"judge_identity_{identity}")
        records = details.get("records")
        if (
            not isinstance(records, list)
            or len(records) != 1
            or not isinstance(records[0], dict)
        ):
            return refuse("judge_response_records_invalid")
        last = records[0]
        if last.get("request_sha256") != self.exchange.sha256_text(
            rules.request_text(request)
        ):
            return refuse("judge_request_digest_mismatch")
        if (
            last.get("prompt_version") != rules.PROMPT_VERSION
            or last.get("verdict") != answer["verdict"]
            or last.get("reason") != answer.get("rationale")
            or not isinstance(last.get("evidence_refs"), list)
        ):
            return refuse("judge_response_records_invalid")
        reason = last["reason"]
        if reason not in _SERVICE_REASONS and (
            not isinstance(reason, str)
            or not reason.startswith("judge_failed:")
            or reason.removeprefix("judge_failed:") not in _CALL_ERRORS
        ):
            return refuse("judge_response_records_invalid")
        if any(
            not isinstance(ref, str) or sent_token in ref
            for ref in last["evidence_refs"]
        ):
            return refuse("judge_response_records_invalid")
        if reason == "judge_completed":
            checked = rules.assess_response(
                {"verdict": last["verdict"], "evidence_refs": last["evidence_refs"]},
                request["evidence"],
            )
            if checked != (last["verdict"], last["evidence_refs"], reason):
                return refuse("judge_response_records_invalid")
        elif last["verdict"] != "unresolved" or last["evidence_refs"]:
            return refuse("judge_response_records_invalid")
        # The service supplies only a summary, never the raw model reply.
        return _record(
            rules,
            last["reason"],
            request,
            verdict=last["verdict"],
            refs=last["evidence_refs"],
            model=answer["model"],
            prompt=prompt,
        )

    def _replay_result(self, sent, rubric, evidence, request):
        rules = self.rules
        prompt = rules.judge_messages(request)
        try:
            row = self.replay.replay(
                self.rows,
                sent["id"],
                0,
                model=self.judge_model,
                prompt_version=rules.PROMPT_VERSION,
            )
        except self.replay.ReplayRefusal as exc:
            return _record(rules, exc.why, request, prompt=prompt)
        if (
            row.redacted
            or row.request_sha256
            != self.exchange.sha256_text(rules.request_text(request))
            or row.messages != tuple((m["role"], m["content"]) for m in prompt)
        ):
            return _record(
                rules, "judge-replay-request-mismatch", request, prompt=prompt
            )

        def call(_messages):
            if row.error is not None:
                raise rules.JudgeCallError(row.error)
            return row.reply

        return _judge_reply(rules, rubric, evidence, call, self.judge_model)
