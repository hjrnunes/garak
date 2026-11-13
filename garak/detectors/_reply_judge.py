# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The reply judge: a check that asks a judge service whether a claim holds of the agent's reply.

``reply_judge`` sends the judged reply (the agent's output), the user message it
answers, the claim and the rubric to a judge service as one ``midojo.judge.v1``
request, and maps the service's verdict to the check's: ``supported`` is True,
``contradicted`` False and ``unresolved`` unknown (None). A request that gets no
valid answer is unknown, never False.

This module reads the check, builds the request, keeps the
:class:`JudgeRecord` a grading holds of each request, asks the service
(:meth:`ReplyJudge.assess`, over ``_judge_http``) and answers a re-grade from
earlier records (:class:`RecordedJudge`) without a connection. The request id,
the response rules and the replay identity check are those of ``midojo_spec``
(``judge`` and ``judge_replay``), loaded when first needed so that they have
one source.

Assessing never raises for what the service or the connection does: every
failure is a record with its reason, and the check is unknown. The bearer token
is read from the environment for the one request, sent in its header, and
removed from every text the service sent back.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
from collections.abc import Iterable, MutableMapping, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from garak.detectors import _judge_http
from garak.evidence.codec import INSTALL_HINT, MissingSnapshotCodecError
from garak.evidence.records import thaw

if TYPE_CHECKING:
    from garak.detectors._checks import Context

NAME = "reply_judge"
VERDICTS = ("supported", "contradicted", "unresolved")
PASSED: dict[str, bool | None] = {
    "supported": True,
    "contradicted": False,
    "unresolved": None,
}
DEFAULT_TIMEOUT_SECONDS = 60.0

_SPEC_KEYS = frozenset(
    {
        "claim",
        "rubric",
        "url",
        "url_env",
        "token_env",
        "timeout_seconds",
        "prior_turns",
        "tool_calls",
    }
)
_RECORD_FIELDS = (
    "request_id",
    "endpoint",
    "request",
    "sent",
    "http_status",
    "verdict",
    "rationale",
    "model",
    "details",
    "failure",
)
_RECORD_REQUIRED = frozenset({"request_id", "endpoint", "request", "sent"})
# The record's fields that hold what the service sent (or a failure quoting it): the ones redaction reads.
_SERVICE_TEXT = ("rationale", "model", "details", "failure")


def _judge_spec() -> Any:
    try:
        from midojo_spec import judge
    except ImportError as exc:
        raise MissingSnapshotCodecError(INSTALL_HINT) from exc
    return judge


def _replay_spec() -> Any:
    try:
        from midojo_spec import judge_replay
    except ImportError as exc:
        raise MissingSnapshotCodecError(INSTALL_HINT) from exc
    return judge_replay


class SettingsError(ValueError):
    """A ``reply_judge`` check that cannot be read; the message names the rule, never a value."""


class UnbuildableError(ValueError):
    """A check whose own settings cannot make a request; its message is safe to show."""


@dataclass(frozen=True)
class JudgeRecord:
    """What one reply judge check asked its service and what it was answered, for the grade's audit.

    ``request`` is the body sent (or, when ``sent`` is False, the body that
    would have been), never the bearer token; ``endpoint`` is the URL without
    its user info, query or fragment, None when no URL was configured. An
    answered request has its ``http_status``; a valid answer has its
    ``verdict``, and the ``rationale``, ``model`` and ``details`` the service
    gave. ``failure`` says why the check is unknown when no valid answer came.
    """

    request_id: str
    endpoint: str | None
    request: dict[str, Any]
    sent: bool
    http_status: int | None = None
    verdict: str | None = None
    rationale: str | None = None
    model: str | None = None
    details: dict[str, Any] | None = None
    failure: str | None = None

    def to_json(self) -> dict[str, Any]:
        """A fresh JSON object, its fields in the order the grade's ``judgements`` hold them."""
        return {name: copy.deepcopy(getattr(self, name)) for name in _RECORD_FIELDS}

    @classmethod
    def from_json(cls, data: Any) -> JudgeRecord:
        """The record a JSON object holds, or ValueError naming the first rule it breaks."""
        if (problem := _record_problem(data)) is not None:
            raise ValueError(problem)
        kept = {name: copy.deepcopy(data[name]) for name in data}
        for name in ("request", "details"):
            if kept.get(name) is not None:
                kept[name] = dict(kept[name])
        return cls(**kept)


def _plain_json(value: Any) -> bool:
    """Whether ``value`` is JSON data of exact types: objects with text keys, lists, finite numbers, no subclass."""
    pending = [value]
    while pending:
        item = pending.pop()
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                return False
            pending.extend(item.values())
        elif type(item) is list:
            pending.extend(item)
        elif type(item) is float and not math.isfinite(item):
            return False
        elif item is not None and type(item) not in (str, int, float, bool):
            return False
    return True


def _record_problem(data: Any) -> str | None:
    if not isinstance(data, Mapping):
        return "a judgement record must be an object"
    if outside := len(set(data) - set(_RECORD_FIELDS)):
        return f"a judgement record has {outside} field(s) outside the schema"
    if missing := sorted(_RECORD_REQUIRED - set(data)):
        return f"a judgement record has no {', '.join(missing)}"
    text = (str,)
    expected: tuple[tuple[str, tuple[type, ...], bool], ...] = (
        ("request_id", text, False),
        ("endpoint", text, True),
        ("request", (Mapping,), False),
        ("sent", (bool,), False),
        ("http_status", (int,), True),
        ("rationale", text, True),
        ("model", text, True),
        ("details", (Mapping,), True),
        ("failure", text, True),
    )
    for name, kinds, optional in expected:
        value = data.get(name)
        if value is None and optional:
            continue
        # bool is an int to Python, not to a status code; a subclass of str may compare
        # equal to text it does not spell, so replay would trust what it did not check
        if (
            not isinstance(value, kinds)
            or (kinds == (int,) and type(value) is bool)
            or (kinds == text and type(value) is not str)
        ):
            return f"judgement record {name} is not valid"
    verdict = data.get("verdict")
    if verdict is not None and (type(verdict) is not str or verdict not in VERDICTS):
        return "judgement record verdict is not valid"
    if not _plain_json(data["request"]) or not _plain_json(data.get("details")):
        return "judgement record request or details is not JSON data"
    if not _request_is_the_one_named(data["request_id"], data["request"]):
        return "judgement record request is not the request its request_id names"
    # A request that holds no reply was never asked (it is ``{}``, the schema and
    # id alone, or the claim and rubric alone), so what is kept of it is a failure.
    # An answer is trusted for the request id it names, so it must hold the request.
    if "reply" not in data["request"] and (
        data.get("failure") is None
        or any(
            data.get(name) is not None
            for name in ("verdict", "rationale", "model", "details")
        )
    ):
        return "judgement record has an answer, or no failure, for a request it does not hold"
    return None


def _request_is_the_one_named(request_id: str, request: Mapping[str, Any]) -> bool:
    """Whether ``request`` is what a grading kept for the request ``request_id`` names.

    A request's id is the digest of its other fields, so a record that holds
    some other request than the one it answers has no claim to answer it, and
    neither has one that holds a request the schema does not allow, whatever
    its digest. Two requests carry no fields to digest: ``{}``, of a grading
    that failed before it could say what it asked (it has no id), and the
    schema with the id alone, of a request that no record answered. One that
    could not be built holds the claim and rubric alone (with the schema and
    its id), and only for that one are the message and reply the schema
    requires not held against it. Any other request is read as it is.
    """
    if not request:
        return request_id == ""
    spec = _judge_spec()
    if request.get("schema") != spec.SCHEMA or request.get("id") != request_id:
        return False
    keys = set(request)
    if keys == {"schema", "id"}:
        return True
    shown = dict(request)
    if keys == {"schema", "id", "claim", "rubric"}:
        shown.update(message=None, reply=None)
    try:
        return spec.request_problem(shown) is None and bool(
            spec.id_holds(dict(request))
        )
    except RecursionError:  # data too deep to read or to digest
        return False


# --- reading the check --------------------------------------------------------------


def _text(value: str) -> str:
    """``value`` with each lone surrogate escaped (``\\ud800``): text UTF-8 can hold, the same every time."""
    return value.encode("utf-8", "backslashreplace").decode("utf-8")


def _not_utf8(value: Any) -> bool:
    """Whether text in ``value`` (a string, or the keys and values of a mapping or list) has a lone surrogate."""
    if type(value) is str:
        return value != _text(value)
    if type(value) is dict:
        return any(_not_utf8(key) or _not_utf8(item) for key, item in value.items())
    if type(value) in (list, tuple):
        return any(_not_utf8(item) for item in value)
    return False


def _sort_key(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def json_data(value: Any) -> Any:
    """``value`` as JSON data, the same every time; TypeError for a type JSON does not hold.

    Text is itself with each lone surrogate escaped (``\\ud800``), as UTF-8
    cannot hold one; integers, booleans and None are themselves; a finite
    float is itself and a non-finite one null; a list or tuple is a list; a
    dict whose keys are all text, still distinct once escaped, is an object,
    and any other dict a list of ``[key, value]`` pairs sorted by the whole
    pair's JSON, so no key is lost or merged and equal dicts write alike.
    """
    kind = type(value)
    if kind is str:
        return _text(value)
    if value is None or kind in (bool, int):
        return value
    if kind is float:
        return value if math.isfinite(value) else None
    if kind in (list, tuple):
        return [json_data(item) for item in value]
    if kind is dict:
        pairs = [[json_data(key), json_data(item)] for key, item in value.items()]
        keys = [key for key, _ in pairs]
        if all(type(key) is str for key in value) and len(set(keys)) == len(keys):
            return dict(pairs)
        return sorted(pairs, key=_sort_key)
    raise TypeError(f"a {kind.__module__}.{kind.__qualname__} is not JSON data")


@dataclass(frozen=True)
class ReplyJudge:
    """Passes when the judge service finds ``claim`` supported by the reply, fails when it finds it contradicted.

    ``url`` is the service's URL, or ``url_env`` names the environment
    variable that holds it; ``token_env`` names the one holding its bearer
    token, if it takes one (a literal token is refused, so none lands in a
    document). Both are read when the check is assessed. ``prior_turns`` and
    ``tool_calls`` add the earlier turns and the recorded tool calls to the
    request.
    """

    claim: str
    rubric: str | dict[str, Any] | None = None
    url: str | None = None
    url_env: str | None = None
    token_env: str | None = None
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    prior_turns: bool = False
    tool_calls: bool = False

    @property
    def label(self) -> str:
        """The check as its reasons name it."""
        return f'reply judge "{_text(str(self.claim))}"'

    def request(self, ctx: Context) -> dict[str, Any]:
        """The ``midojo.judge.v1`` request body for ``ctx``: its ``id`` is the digest of the rest.

        The claim and rubric must be UTF-8 text, as reading the check requires
        (one built some other way may not be): else :class:`UnbuildableError`.
        Everything else in it, what the agent and its tools produced, is
        written as JSON data by :func:`json_data`, a lone surrogate escaped.
        """
        for name in ("claim", "rubric"):
            if _not_utf8(getattr(self, name)):
                raise UnbuildableError(
                    f"the check's {name} is not UTF-8 text: it has a lone surrogate"
                )
        turns = ctx.turns
        body: dict[str, Any] = {
            "claim": self.claim,
            "rubric": self.rubric,
            "message": turns[-1].message if turns else None,
            "reply": ctx.agent_output,
        }
        if self.prior_turns:
            body["prior_turns"] = [
                {"message": turn.message, "reply": turn.reply} for turn in turns[:-1]
            ]
        if self.tool_calls:
            body["tool_calls"] = [
                {
                    "function": call.function,
                    "args": thaw(call.args),
                    "result": call.result,
                    "error": call.error,
                    "complete": call.complete,
                    "blocked": call.blocked,
                    "turn": call.turn,
                }
                for call in ctx.function_calls
            ]
            body["tool_calls_complete"] = ctx.function_calls_complete and all(
                call.complete for call in ctx.function_calls
            )
        body = json_data(body)
        return {
            "schema": _judge_spec().SCHEMA,
            "id": _judge_spec().request_id(body),
            **body,
        }

    def unbuilt_request(self) -> dict[str, Any]:
        """What a request that could not be built is known to hold: the check's own claim and rubric, escaped.

        It does not depend on what failed: the claim and rubric go through the
        same escaping as every request's text, and when even that fails the
        request holds neither.
        """
        try:
            body = json_data({"claim": self.claim, "rubric": self.rubric})
        except Exception:  # noqa: BLE001 - nothing of the claim can be told
            body = {}
        spec = _judge_spec()
        return {"schema": spec.SCHEMA, "id": spec.request_id(body), **body}

    def outcome(self, record: JudgeRecord) -> tuple[bool | None, str]:
        """The check's value and the words for its reason, from what the service answered.

        Nothing the service sent goes into the words: not its model or
        rationale (the record keeps those), and a verdict only as the schema's
        own word for it. A failure's text is this module's own. For an unknown
        value the words are why it is unknown.
        """
        request = f"request {record.request_id}"
        if record.failure is not None:
            return None, f"{record.failure}; {request}"
        verdict = next(
            (name for name in PASSED if name == record.verdict), "unresolved"
        )
        if PASSED[verdict] is None:
            return None, f"the judge service left it unresolved ({request})"
        return PASSED[verdict], f"{verdict} ({request})"

    def assess(
        self,
        ctx: Context,
        judgements: MutableMapping[str, JudgeRecord],
        replay: RecordedJudge | None = None,
    ) -> tuple[bool | None, str]:
        """Ask the service once per request, and give the check's value with the words for its reason: never an exception.

        ``judgements`` holds the record of every request this grading has made
        (the check asked again with the same request reads its record), and
        ``replay``, when given, answers from earlier records instead of the
        service. The known failures have their own reasons. Any other
        exception on the way (resolving the settings, building, sending,
        decoding, validating or recording) is caught here as a backstop:
        unknown, with the exception's type only (its message may hold the
        token or a header), and a record of how far the request got. Only a
        missing ``midojo_spec`` is raised, as the installation's fault.
        """
        progress = _Progress()
        try:
            record = self._judgement(ctx, judgements, replay, progress)
        except ReplayRefusal as unanswered:
            return self._unanswered(judgements, unanswered, progress)
        except MissingSnapshotCodecError:
            raise
        except Exception as exc:  # noqa: BLE001 - the backstop, as the docstring says
            record = progress.failed(self, exc)
            key = progress.key or json.dumps(["failed", record.request_id])
            judgements.setdefault(key, record)
        return self.outcome(record)

    def _unanswered(
        self,
        judgements: MutableMapping[str, JudgeRecord],
        unanswered: ReplayRefusal,
        progress: _Progress,
    ) -> tuple[bool | None, str]:
        """A replay with no record it can answer from: unknown, naming the id, never a verdict.

        The refusal is kept as a failed record, as every unanswered request
        is, so the grade shows which request had no answer. The reason is the
        refusal's own: its code and the id's first eight hex digits.
        """
        why = str(unanswered)
        request = progress.request or {
            "schema": _judge_spec().SCHEMA,
            "id": unanswered.request_id,
        }
        record = JudgeRecord(
            request_id=unanswered.request_id,
            endpoint=progress.endpoint,
            request=request,
            sent=False,
            failure=why,
        )
        key = progress.key or json.dumps(["replay", unanswered.request_id])
        judgements.setdefault(key, record)
        return None, why

    def _replay_endpoint(self) -> tuple[bool, str | None]:
        """The endpoint this check would ask, when the check alone names it: ``(known, endpoint)``.

        A literal ``url`` names the endpoint the live path would clean (None
        when it is not a URL one could read); a check with no URL at all names
        None. A ``url_env`` variable is read only when the check is assessed,
        so a replay, which reads no environment, cannot know which endpoint it
        held then: the endpoint is unknown.
        """
        if self.url is None:
            return (False, None) if self.url_env is not None else (True, None)
        endpoint = _judge_http.parse_url(self.url)
        return True, None if endpoint is None else _text(endpoint.text)

    def _judgement(
        self,
        ctx: Context,
        judgements: MutableMapping[str, JudgeRecord],
        replay: RecordedJudge | None,
        progress: _Progress,
    ) -> JudgeRecord:
        try:
            request = self.request(ctx)
        except MissingSnapshotCodecError:
            raise
        except (
            Exception
        ) as exc:  # noqa: BLE001 - unknown, with the request as far as known
            request = self.unbuilt_request()
            why = str(exc) if isinstance(exc, UnbuildableError) else type(exc).__name__
            unbuilt = JudgeRecord(
                request_id=request["id"],
                endpoint=None,
                request=request,
                sent=False,
                failure=f"the judge request could not be built ({why})",
            )
            key = json.dumps(
                ["unbuilt", request["id"], self.url, self.url_env, self.token_env]
            )
            return judgements.setdefault(key, unbuilt)
        progress.request = request
        progress.key = json.dumps(
            [
                request["id"],
                self.url,
                self.url_env,
                self.token_env,
                self.timeout_seconds,
            ]
        )
        record = judgements.get(progress.key)
        if record is None and replay is not None:
            # A re-grade answers from its records, by request id and, when the check names one, endpoint.
            endpoint_known, endpoint = self._replay_endpoint()
            record = judgements[progress.key] = replay.ask(
                request, endpoint=endpoint, endpoint_known=endpoint_known
            )
        if record is None:
            record = judgements[progress.key] = self.ask(request, progress)
        return record

    def ask(
        self, request: dict[str, Any], progress: _Progress | None = None
    ) -> JudgeRecord:
        """Send ``request`` to the service and keep what it answered, or why there is no answer.

        A URL or token that cannot be used is a failure before anything is
        sent: its reason names where it came from, never the token, and shows
        no URL text. ``progress`` notes how far it got, for the backstop in
        :meth:`assess`.
        """
        progress = progress or _Progress()
        progress.request = request
        fields: dict[str, Any] = {
            "request_id": request["id"],
            "request": request,
            "endpoint": None,
            "sent": False,
        }
        source = "the check's url" if self.url is not None else _text(str(self.url_env))
        url = self.url if self.url is not None else os.environ.get(self.url_env or "")
        if not url:
            return JudgeRecord(
                **fields, failure=f"no judge service URL: {source} is not set"
            )
        # An environment value that is not UTF-8 decodes to lone surrogates.
        if _not_utf8(url):
            return JudgeRecord(
                **fields, failure=f"the judge service URL in {source} is not UTF-8 text"
            )
        endpoint = _judge_http.parse_url(url)
        if endpoint is None:
            return JudgeRecord(
                **fields, failure="the judge service URL is not a valid URL"
            )
        progress.endpoint = fields["endpoint"] = _text(endpoint.text)
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        token = None
        if self.token_env is not None:
            name = _text(str(self.token_env))
            token = os.environ.get(self.token_env)
            if not token:
                return JudgeRecord(
                    **fields, failure=f"no judge service token: {name} is not set"
                )
            if not _header_safe(token):
                failure = f"the judge service token in {name} can't be sent: it isn't printable ASCII"
                return JudgeRecord(**fields, failure=failure)
            headers["Authorization"] = f"Bearer {token}"
        record = _redacted(self._exchange(endpoint, headers, fields, progress), token)
        _dumped(record)  # the grade dumps it; one that cannot be is the backstop's
        return record

    def _exchange(
        self,
        endpoint: _judge_http.Endpoint,
        headers: dict[str, str],
        fields: dict[str, Any],
        progress: _Progress,
    ) -> JudgeRecord:
        content = json.dumps(fields["request"], ensure_ascii=False).encode("utf-8")
        fields = {**fields, "sent": True}
        try:
            progress.sent = True  # from here on, the request may have gone out
            answer = _judge_http.post(
                endpoint, content, headers, timeout=self.timeout_seconds
            )
        except TimeoutError as exc:
            failure = f"the judge service timed out after {self.timeout_seconds:g}s ({type(exc).__name__})"
            return JudgeRecord(**fields, failure=failure)
        except _judge_http.AnswerTooLarge as exc:
            failure = f"the judge service's answer is longer than {_judge_http.MAX_ANSWER_BYTES} bytes"
            return JudgeRecord(**fields, http_status=exc.status, failure=failure)
        except _judge_http.UnreadableCoding as exc:
            # The coding is the service's text: the reason says only that there is one.
            failure = "the judge service's answer is in a content coding this check does not decode"
            return JudgeRecord(**fields, http_status=exc.status, failure=failure)
        except _judge_http.TRANSPORT_ERRORS as exc:
            # The exception's message may hold the URL; its type says enough.
            failure = f"the judge service request failed ({type(exc).__name__})"
            return JudgeRecord(**fields, failure=failure)
        progress.http_status = answer.status
        sent = {**fields, "http_status": answer.status}
        if not 200 <= answer.status < 300:
            return JudgeRecord(
                **sent, failure=f"the judge service answered HTTP {answer.status}"
            )
        try:
            body = json.loads(
                answer.body, parse_constant=_refuse_constant, parse_float=_finite_float
            )
        except _NonFiniteError as exc:
            return JudgeRecord(**sent, failure=f"the judge service's answer {exc}")
        except (ValueError, RecursionError):  # a decoding error is a ValueError
            return JudgeRecord(**sent, failure="the judge service's answer is not JSON")
        spec = _judge_spec()
        problem = spec.response_problem(body, fields["request_id"])
        if problem is not None:
            failure = (
                f"the judge service's answer is not a {spec.SCHEMA} response: {problem}"
            )
            return JudgeRecord(**sent, failure=failure)
        return JudgeRecord(
            **sent,
            verdict=body["verdict"],
            # JSON's \ud800 escape decodes to a lone surrogate: escaped, as in a request, so the record dumps.
            rationale=json_data(body.get("rationale")),
            model=json_data(body.get("model")),
            details=json_data(body.get("details")),
        )


def parse_reply_judge(value: Any) -> ReplyJudge:
    """A :class:`ReplyJudge` from its document form, or :class:`SettingsError` naming what is wrong with it."""
    if not isinstance(value, dict):
        raise SettingsError(f"{NAME}: expected a mapping, got {type(value).__name__}")
    if "token" in value:
        raise SettingsError(
            f"{NAME}: a literal token is not accepted; name the variable that holds it with token_env"
        )
    if unknown := sorted(set(value) - _SPEC_KEYS, key=str):
        raise SettingsError(f"{NAME}: unknown keys {unknown}")
    claim = value.get("claim")
    if type(claim) is not str or not claim.strip():
        raise SettingsError(f"{NAME}: claim must be a non-empty string")
    rubric = value.get("rubric")
    if rubric is not None and type(rubric) not in (str, dict):
        raise SettingsError(f"{NAME}: rubric must be a string or a mapping")
    for name in ("claim", "rubric", "url", "url_env", "token_env"):
        if _not_utf8(value.get(name)):
            raise SettingsError(
                f"{NAME}: {name} is not UTF-8 text: it has a lone surrogate"
            )
    if isinstance(rubric, dict):
        try:
            json.dumps(rubric, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise SettingsError(f"{NAME}: rubric must be JSON data") from exc
    if ("url" in value) == ("url_env" in value):
        raise SettingsError(f"{NAME}: give exactly one of url and url_env")
    for name in ("url", "url_env", "token_env"):
        if name in value and (type(value[name]) is not str or not value[name]):
            raise SettingsError(f"{NAME}: {name} must be a non-empty string")
    timeout = value.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise SettingsError(f"{NAME}: timeout_seconds must be a positive number")
    for name in ("prior_turns", "tool_calls"):
        if type(value.get(name, False)) is not bool:
            raise SettingsError(f"{NAME}: {name} must be true or false")
    return ReplyJudge(
        claim=claim,
        rubric=copy.deepcopy(rubric),
        url=value.get("url"),
        url_env=value.get("url_env"),
        token_env=value.get("token_env"),
        timeout_seconds=float(timeout),
        prior_turns=value.get("prior_turns", False),
        tool_calls=value.get("tool_calls", False),
    )


# --- asking ---------------------------------------------------------------------------


def _dumped(record: JudgeRecord) -> None:
    """Raise when ``record`` cannot be written as the grade writes it: strict UTF-8 and no NaN."""
    json.dumps(record.to_json(), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _header_safe(token: str) -> bool:
    """Whether ``token`` can go in an Authorization header as it is: printable ASCII, no control character."""
    return all(" " <= character <= "~" for character in token)


@dataclass
class _Progress:
    """How far one judge request got: what the backstop's record says when a step fails unexpectedly."""

    request: dict[str, Any] | None = None
    key: str | None = None
    endpoint: str | None = None
    sent: bool = False
    http_status: int | None = None

    def failed(self, check: ReplyJudge, exc: Exception) -> JudgeRecord:
        """The record of a request stopped by ``exc``: its type only, nothing the service answered, no token."""
        failure = f"the judge check failed ({type(exc).__name__})"
        try:
            request = (
                self.request if self.request is not None else check.unbuilt_request()
            )
            record = JudgeRecord(
                request_id=request["id"],
                endpoint=self.endpoint,
                request=request,
                sent=self.sent,
                http_status=self.http_status,
                failure=failure,
            )
            _dumped(record)
            return record
        except Exception:  # noqa: BLE001 - nothing of the request can be told
            return JudgeRecord(
                request_id="",
                endpoint=None,
                request={},
                sent=self.sent,
                failure=failure,
            )


class _NonFiniteError(ValueError):
    """The answer has ``NaN``, ``Infinity`` or ``-Infinity``, which JSON does not, or a number no float holds."""


def _refuse_constant(name: str) -> Any:
    # The parser only hands over these three; the reason names them in this module's own words all the same.
    known = {"NaN": "NaN", "Infinity": "Infinity", "-Infinity": "-Infinity"}
    raise _NonFiniteError(
        f"is not JSON: it has {known.get(name, 'a constant JSON has no word for')}"
    )


def _finite_float(text: str) -> float:
    number = float(text)
    if not math.isfinite(number):
        # Not the number itself: the service wrote it.
        raise _NonFiniteError("has a number no float holds")
    return number


def _redacted(record: JudgeRecord, token: str | None) -> JudgeRecord:
    """``record`` with ``token`` replaced wherever the service's text holds it (one it echoed back, say).

    Only the values of the fields the service filled in are read (keys and
    values of ``details`` included), never a field name or the verdict, so the
    record keeps its schema whatever the token is. When it cannot be redacted,
    the record keeps none of the service's text and says so.
    """
    if not token:
        return record
    try:
        from garak.cli import scrub_secrets

        return dataclasses.replace(
            record,
            **{
                name: scrub_secrets(getattr(record, name), [token])
                for name in _SERVICE_TEXT
            },
        )
    except Exception as exc:  # noqa: BLE001 - keep none of the service's text instead
        return JudgeRecord(
            request_id=record.request_id,
            endpoint=record.endpoint,
            request=record.request,
            sent=record.sent,
            http_status=record.http_status,
            failure=f"the judge service's answer could not be redacted ({type(exc).__name__})",
        )


# --- replay --------------------------------------------------------------------------


class ReplayRefusal(LookupError):  # noqa: N818 - the replay contract names the leaves
    """A replay that cannot answer a request: its check is unknown, with the reason's ``id8``.

    ``request_id`` is the id that was asked for and ``id8`` its first eight
    hex digits, which the check's reason names after ``why``.
    """

    why = "judge-replay"

    def __init__(self, request: Mapping[str, Any]) -> None:
        self.request_id = str(request["id"])
        self.id8 = self.request_id.removeprefix("sha256:")[:8]
        super().__init__(f"{self.why}:{self.id8}")


class UnrecordedJudgeRequest(ReplayRefusal):
    """A replay was asked for a request its records do not hold: the check is unknown, never False."""

    why = "judge-replay-unrecorded"


class IdentityRefusedJudgeRequest(ReplayRefusal):
    """Records have the id, but none has the requested model and prompt version."""

    why = "judge-replay-identity"

    def __init__(self, request: Mapping[str, Any], reason: str) -> None:
        self.reason = reason
        super().__init__(request)


class AmbiguousJudgeRequest(ReplayRefusal):
    """Several records answer one request id, cannot be told apart, and disagree.

    Checks that differ only by endpoint share a request id, which hashes the
    body alone. When the check does not name its endpoint and the records
    decide differently, no record is guessed: the answer is unknown, so a
    recorded failure never replays as a verdict.
    """

    why = "judge-replay-ambiguous"


def _decides_alike(left: JudgeRecord, right: JudgeRecord) -> bool:
    return (left.verdict, left.failure) == (right.verdict, right.failure)


class RecordedJudge:
    """Answers ``reply_judge`` requests from records an earlier grading kept, and never opens a connection.

    ``records`` are :class:`JudgeRecord` objects, or their JSON objects (as a
    grade's ``judgements`` holds them), kept by request id; one that is not a
    grade's record, its request not the one its request id names or an answer
    for a request it does not hold, is a ValueError. What is kept is a copy
    made after that check, and an answer is a copy of it: changing the records
    given, or an answer, afterwards changes no later answer. ``model`` and
    ``prompt_version`` are required: a record without either identity never
    answers, even when it is the only candidate. A request no record names is
    an :class:`UnrecordedJudgeRequest`, which the check turns into unknown with
    ``judge-replay-unrecorded:<id8>``: a replay never guesses a verdict.

    A request id hashes the request body alone, so checks that differ only by
    endpoint share one id. :meth:`ask` then matches the check's own endpoint
    when the caller passes it with ``endpoint_known=True``. When the records
    for one id disagree and that cannot tell them apart, the answer is an
    :class:`AmbiguousJudgeRequest`. Records that decide alike answer alike, so
    any one of them answers. A record whose ``failure`` is set replays as
    unknown, as it graded the first time.
    """

    def __init__(
        self,
        records: Iterable[JudgeRecord | Mapping[str, Any]],
        *,
        model: str,
        prompt_version: str,
    ) -> None:
        _replay_spec().identity_problem(
            model,
            prompt_version,
            expected_model=model,
            expected_prompt_version=prompt_version,
        )
        self.model = model
        self.prompt_version = prompt_version
        self._records: dict[str, list[JudgeRecord]] = {}
        for number, record in enumerate(records):
            try:
                # A record built by hand is held to what one read from JSON is,
                # and only the copy that was read answers: nothing here holds the
                # caller's record, so changing it afterwards changes no answer.
                read = JudgeRecord.from_json(
                    record.to_json() if isinstance(record, JudgeRecord) else record
                )
            except ValueError as exc:
                raise ValueError(f"judgements[{number}]: {exc}") from None
            self._records.setdefault(read.request_id, []).append(read)

    def __len__(self) -> int:
        """How many records it holds, repeated request ids included."""
        return sum(len(group) for group in self._records.values())

    def __contains__(self, request_id: object) -> bool:
        return request_id in self._records

    def ask(
        self,
        request: Mapping[str, Any],
        *,
        endpoint: str | None = None,
        endpoint_known: bool = False,
    ) -> JudgeRecord:
        """The record for ``request``, or a :class:`ReplayRefusal` naming why it cannot be known.

        Every record must match this replay's model and prompt version before
        the endpoint can narrow the candidates. ``endpoint`` narrows them when
        ``endpoint_known`` says the check alone names it. One candidate
        answers. Several that decide alike answer alike, so the first answers;
        several that disagree are an :class:`AmbiguousJudgeRequest`.
        """
        candidates = self._records.get(str(request["id"]))
        if not candidates:
            raise UnrecordedJudgeRequest(request)
        identity_problem = _replay_spec().identity_problem
        problems = [
            identity_problem(
                record.model,
                record.details.get("prompt_version") if record.details else None,
                expected_model=self.model,
                expected_prompt_version=self.prompt_version,
            )
            for record in candidates
        ]
        matching_identity = [
            record
            for record, problem in zip(candidates, problems, strict=True)
            if problem is None
        ]
        if not matching_identity:
            raise IdentityRefusedJudgeRequest(request, problems[0] or "missing")
        candidates = matching_identity
        if endpoint_known:
            matching = [record for record in candidates if record.endpoint == endpoint]
            candidates = matching or candidates
        first = candidates[0]
        if all(_decides_alike(first, other) for other in candidates[1:]):
            # A copy: what the caller does to the answer must not reach the next one.
            return copy.deepcopy(first)
        raise AmbiguousJudgeRequest(request)
