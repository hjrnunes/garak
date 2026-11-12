# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test asking a judge service: the verdict, every failure, the token and the record.

The service is a fake on 127.0.0.1 (``tests/_assets/detectors/reply_judge/fake_judge.py``):
no model is called and nothing leaves the machine. Every failure is unknown
with its reason, never False, and never an exception; the bearer token goes
only into the request's Authorization header.
"""

import dataclasses
import json
import logging
import os
import socket
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("midojo_spec")

from midojo_spec.judge import SCHEMA  # noqa: E402

from garak.detectors import _checks, _judge_http, _reply_judge  # noqa: E402
from garak.detectors._reply_judge import (  # noqa: E402
    JudgeRecord,
    RecordedJudge,
    ReplyJudge,
)
from garak.evidence.codec import MissingSnapshotCodecError  # noqa: E402
from garak.evidence.facts import Turn  # noqa: E402
from garak.evidence.records import CallRecord  # noqa: E402

TOKEN = "FAKE-TEST-VALUE-0000"  # a made-up bearer token: no service takes it
SNEAKY = "sn3aky\\tok\"en'-4be1"  # a backslash and both quotes: repr and JSON spell it differently
CLAIM = "The reply tells the user to wire money to account 4421"
REPLY = "Please wire the balance to account 4421 today."
LONE = "\ud800"
STAMP = "2026-01-01T00:00:00.000{:03d}+00:00"
URL_VAR, TOKEN_VAR = "TEST_JUDGE_URL", "TEST_JUDGE_TOKEN"


def call(function="transfer", args=None, *, n=0):
    return CallRecord.from_json(
        {
            "call_id": f"call-{n}",
            "function": function,
            "args": args or {},
            "result": "ok",
            "error": None,
            "blocked": False,
            "disposition": "forwarded",
            "structured_content": None,
            "injection_outcomes": None,
            "complete": True,
            "provenance": "interceptor",
            "timestamp": STAMP.format(n),
            "turn": 0,
            "pre_environment": None,
            "post_environment": None,
        }
    )


def context(reply=REPLY, turns=None, calls=()):
    held = [Turn(0, "Pay my bills", reply, STAMP.format(1), STAMP.format(2))]
    return _checks.Context(
        agent_output=reply,
        pre_environment=None,
        post_environment=None,
        function_calls=tuple(calls),
        turns=tuple(held if turns is None else turns),
        evidence_status={},
    )


def check(url, **settings):
    return ReplyJudge(claim=CLAIM, url=url, **settings)


def assess(judge_check, ctx=None, kept=None, replay=None):
    kept = {} if kept is None else kept
    passed, words = judge_check.assess(ctx or context(), kept, replay)
    return passed, words, kept


def only(kept):
    [record] = kept.values()
    return record


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def no_secret_in(secret, *texts):
    return [text for text in texts if secret in text] == []


def dumps(record):
    return json.dumps(record.to_json(), ensure_ascii=False)


# --- the three verdicts ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("verdict", "passed"),
    [("supported", True), ("contradicted", False), ("unresolved", None)],
)
def test_a_verdict_maps_to_the_checks_value_and_the_record_keeps_the_answer(
    fake_judge, verdict, passed
):
    value, words, kept = assess(check(fake_judge.url(f"/{verdict}")))
    record = only(kept)
    assert value is passed
    assert (record.verdict, record.http_status, record.model, record.sent) == (
        verdict,
        200,
        fake_judge.MODEL,
        True,
    )
    assert record.rationale == f"the reply makes the claim {verdict}"
    assert record.details == {
        "evidence_refs": ["reply"],
        "prompt_version": fake_judge.PROMPT_VERSION,
    }
    assert record.failure is None
    assert record.endpoint == fake_judge.url(f"/{verdict}")
    assert record.request_id in words and verdict in words
    assert "makes the claim" not in words and fake_judge.MODEL not in words


def test_a_minimal_answer_has_only_a_schema_and_a_verdict(fake_judge):
    value, _, kept = assess(check(fake_judge.url("/minimal")))
    record = only(kept)
    assert value is False
    assert (record.verdict, record.rationale, record.model, record.details) == (
        "contradicted",
        None,
        None,
        None,
    )


# --- the request as sent --------------------------------------------------------------------


def test_the_request_on_the_wire_is_the_one_the_record_keeps(fake_judge):
    calls = [call("transfer", {"to": "4421", "memo": ["a", {"b": 1.5}]})]
    turns = [
        Turn(0, "Hi", "Hello", STAMP.format(0), STAMP.format(1)),
        Turn(1, "Pay", REPLY, STAMP.format(2), STAMP.format(3)),
    ]
    rubric = {"criteria": ["an explicit account number counts"]}
    _, _, kept = assess(
        check(
            fake_judge.url("/supported"),
            rubric=rubric,
            prior_turns=True,
            tool_calls=True,
        ),
        context(turns=turns, calls=calls),
    )
    [received] = fake_judge.requests
    assert received.body == only(kept).request
    assert received.body["rubric"] == rubric and received.body["message"] == "Pay"
    assert received.body["prior_turns"] == [{"message": "Hi", "reply": "Hello"}]
    assert received.body["tool_calls"][0]["args"] == {
        "to": "4421",
        "memo": ["a", {"b": 1.5}],
    }
    assert received.headers["content-type"] == "application/json"
    assert received.headers["accept"] == "application/json"
    assert (
        "authorization" not in received.headers
    ), "no token is configured, none is sent"


def test_text_the_agent_produced_with_lone_surrogates_is_sent_escaped(fake_judge):
    turns = [
        Turn(0, f"hi {LONE}", f"hello {LONE}", STAMP.format(0), STAMP.format(1)),
        Turn(1, "Pay", f"Wire {LONE}", STAMP.format(2), STAMP.format(3)),
    ]
    calls = [call("lookup", {f"key {LONE}": [f"v {LONE}"]})]
    value, _, kept = assess(
        check(fake_judge.url("/supported"), prior_turns=True, tool_calls=True),
        context(reply=f"Wire {LONE}", turns=turns, calls=calls),
    )
    [received] = fake_judge.requests
    assert value is True
    assert received.body["reply"] == "Wire \\ud800"
    assert received.body["prior_turns"] == [
        {"message": "hi \\ud800", "reply": "hello \\ud800"}
    ]
    assert received.body["tool_calls"][0]["args"] == {"key \\ud800": ["v \\ud800"]}
    dumps(only(kept))


def test_any_success_status_is_an_answer(fake_judge):
    value, _, kept = assess(check(fake_judge.url("/created")))
    assert value is True and only(kept).http_status == 201


def test_the_body_is_the_request_in_json_with_its_text_as_it_is(fake_judge):
    claim = "wire the € to Zoë"
    _, _, kept = assess(ReplyJudge(claim=claim, url=fake_judge.url("/supported")))
    [received] = fake_judge.requests
    assert received.raw == json.dumps(only(kept).request, ensure_ascii=False).encode()
    assert claim.encode() in received.raw, "not escaped to ASCII"


def test_the_endpoint_keeps_no_user_info_or_query_and_the_query_is_sent(fake_judge):
    url = f"http://user:secret@{fake_judge.host_port}/supported?key=secret#frag"
    _, _, kept = assess(check(url))
    assert only(kept).endpoint == fake_judge.url("/supported")
    assert fake_judge.requests[0].query == "key=secret"
    assert "secret" not in dumps(only(kept))


# --- a check asked again ---------------------------------------------------------------------


def test_a_check_asked_again_with_the_same_request_in_one_grading_asks_once(fake_judge):
    checked = check(fake_judge.url("/supported"))
    kept: dict = {}
    first = assess(checked, kept=kept)[:2]
    second = assess(checked, kept=kept)[:2]
    assert first == second == (True, first[1])
    assert len(fake_judge.requests) == 1 and len(kept) == 1


def test_another_grading_asks_again(fake_judge):
    checked = check(fake_judge.url("/supported"))
    assess(checked)
    assess(checked)
    assert len(fake_judge.requests) == 2


def test_a_failed_answer_is_kept_too_and_not_asked_again(fake_judge):
    checked = check(fake_judge.url("/500"))
    kept: dict = {}
    assess(checked, kept=kept)
    assess(checked, kept=kept)
    assert len(fake_judge.requests) == 1 and only(kept).http_status == 500


@pytest.mark.parametrize(
    "other",
    [
        {"timeout_seconds": 30.0},
        {"token_env": TOKEN_VAR},
        {"claim": CLAIM + " today"},
        {"prior_turns": True},
    ],
)
def test_a_check_that_asks_another_question_or_another_way_asks_again(
    fake_judge, monkeypatch, other
):
    monkeypatch.setenv(TOKEN_VAR, TOKEN)
    settings = {"claim": CLAIM, "url": fake_judge.url("/supported")}
    kept: dict = {}
    assess(ReplyJudge(**settings), kept=kept)
    assess(ReplyJudge(**{**settings, **other}), kept=kept)
    assert len(fake_judge.requests) == 2 and len(kept) == 2


def test_a_check_that_names_another_url_or_another_url_variable_asks_again(
    fake_judge, monkeypatch
):
    monkeypatch.setenv(URL_VAR, fake_judge.url("/supported"))
    monkeypatch.setenv(f"{URL_VAR}_2", fake_judge.url("/supported"))
    kept: dict = {}
    for settings in (
        {"url": fake_judge.url("/supported")},
        {"url": fake_judge.url("/contradicted")},
        {"url_env": URL_VAR},
        {"url_env": f"{URL_VAR}_2"},
    ):
        assess(ReplyJudge(claim=CLAIM, **settings), kept=kept)
    assert len(fake_judge.requests) == 4 and len(kept) == 4


# --- every failure is unknown, with its reason ------------------------------------------------

FAILURES = {
    "/500": "the judge service answered HTTP 500",
    "/404": "the judge service answered HTTP 404",
    "/redirect": "the judge service answered HTTP 302",
    "/multiple-choices": "the judge service answered HTTP 300",
    "/not-json": "the judge service's answer is not JSON",
    "/empty": "the judge service's answer is not JSON",
    "/recursion": "the judge service's answer is not JSON",
    "/array": f"not a {SCHEMA} response: the answer is an array, not an object",
    "/no-verdict": "it has no verdict",
    "/bad-verdict": 'verdict is not one of "supported", "contradicted", "unresolved"',
    "/verdict-type": "verdict is a boolean, not a string",
    "/verdict-list": "verdict is an array, not a string",
    "/verdict-object": "verdict is an object, not a string",
    "/no-schema": f"not a {SCHEMA} response: it has no schema",
    "/wrong-schema": f'schema is not "{SCHEMA}"',
    "/rationale-type": "rationale is an array, not a string",
    "/model-type": "model is a number, not a string",
    "/details-type": "details is a string, not an object",
    "/unknown-field": "it has 1 field outside the schema",
    "/other-id": "id is not the request's id",
    "/deep-details": "details nests deeper than 64",
    "/details-NaN": "the judge service's answer is not JSON: it has NaN",
    "/details-Infinity": "the judge service's answer is not JSON: it has Infinity",
    "/details--Infinity": "the judge service's answer is not JSON: it has -Infinity",
    "/huge-number": "the judge service's answer has a number no float holds",
}
STATUS = {"/500": 500, "/404": 404, "/redirect": 302, "/multiple-choices": 300}


@pytest.mark.parametrize("path", FAILURES)
def test_a_failed_answer_is_unknown_with_its_reason(fake_judge, path):
    value, words, kept = assess(check(fake_judge.url(path)))
    record = only(kept)
    assert value is None
    assert FAILURES[path] in words and record.request_id in words
    assert record.sent and record.verdict is None
    assert FAILURES[path] in (record.failure or "")
    if path in STATUS:
        assert record.failure == FAILURES[path], "a status says no more than its number"
    assert record.http_status == STATUS.get(path, 200)
    assert (record.rationale, record.model, record.details) == (None, None, None)
    assert [r.path for r in fake_judge.requests] == [path], "a redirect is not followed"
    dumps(record)


def test_an_answer_longer_than_the_limit_is_unknown_and_names_it(fake_judge):
    value, words, kept = assess(check(fake_judge.url("/big"), timeout_seconds=30))
    assert value is None
    assert f"answer is longer than {_judge_http.MAX_ANSWER_BYTES} bytes" in words
    assert only(kept).http_status == 200 and only(kept).verdict is None


@pytest.mark.parametrize(
    "path",
    ["/gzip", "/gzip-capitalised", "/deflate", "/deflate-raw", "/deflate-then-gzip"]
    + ["/deflate-split", "/gzip-split", "/identity-coding"],
)
def test_an_answer_in_a_content_coding_is_read_as_the_plain_one_is(fake_judge, path):
    value, words, kept = assess(check(fake_judge.url(path)))
    assert value is True and words.startswith("supported")
    record = only(kept)
    assert (record.verdict, record.failure) == ("supported", None)
    assert record.http_status == 200


@pytest.mark.parametrize("path", ["/br", "/zstd", "/unknown-coding", "/gzip-then-br"])
def test_an_answer_in_a_coding_the_check_does_not_decode_is_unknown_never_a_verdict(
    fake_judge, path
):
    # Each of these answers is a valid "supported" as it is sent.
    reason = (
        "the judge service's answer is in a content coding this check does not decode"
    )
    value, words, kept = assess(check(fake_judge.url(path)))
    record = only(kept)
    assert value is None
    assert reason in words and record.request_id in words
    assert record.failure == reason, "the reason holds no text the service sent"
    assert record.sent and record.http_status == 200, "the request was answered"
    assert (record.verdict, record.rationale, record.model, record.details) == (
        None,
        None,
        None,
        None,
    )
    assert [r.path for r in fake_judge.requests] == [path]
    dumps(record)


@pytest.mark.parametrize(
    ("path", "failure", "status"),
    [
        ("/gzip-not-gzip", "the judge service request failed (DecodingError)", None),
        (
            "/deflate-not-deflate",
            "the judge service request failed (DecodingError)",
            None,
        ),
        (
            "/deflate-raw-split",
            "the judge service request failed (DecodingError)",
            None,
        ),
        ("/gzip-truncated", "the judge service's answer is not JSON", 200),
        ("/gzip-bomb", "the judge service's answer is longer than", 200),
    ],
)
def test_an_answer_its_coding_does_not_hold_is_unknown_with_its_reason(
    fake_judge, path, failure, status
):
    value, words, kept = assess(check(fake_judge.url(path), timeout_seconds=30))
    record = only(kept)
    assert value is None and failure in words
    assert record.sent and record.http_status == status and record.verdict is None


def test_a_port_of_zero_is_connected_to_and_fails_like_any_closed_port():
    value, words, kept = assess(check("http://127.0.0.1:0/judge"))
    record = only(kept)
    assert value is None and record.sent
    assert record.endpoint == "http://127.0.0.1:0/judge"
    assert (record.failure or "").startswith("the judge service request failed (")


def test_a_timeout_is_unknown(fake_judge):
    started = time.monotonic()
    value, words, kept = assess(check(fake_judge.url("/slow"), timeout_seconds=0.2))
    assert time.monotonic() - started < 1.5
    assert value is None
    assert "the judge service timed out after 0.2s (TimeoutError)" in words
    assert only(kept).http_status is None and only(kept).sent


def test_a_refused_connection_is_unknown():
    value, words, kept = assess(check(f"http://127.0.0.1:{free_port()}/judge"))
    assert value is None
    assert "the judge service request failed (ConnectionRefusedError)" in words
    assert only(kept).sent


def test_a_url_with_another_scheme_is_unknown_and_names_it():
    value, words, kept = assess(check("ftp://127.0.0.1/judge"))
    assert value is None
    assert "the judge service request failed (UnsupportedProtocol)" in words
    assert only(kept).endpoint == "ftp://127.0.0.1/judge"


@pytest.mark.parametrize("url", ["not a url", "http://h:abc/", "http:///p", "//host/p"])
def test_a_url_that_is_not_one_is_unknown_and_sends_nothing(url):
    value, words, kept = assess(check(url))
    record = only(kept)
    assert value is None and "the judge service URL is not a valid URL" in words
    assert (record.sent, record.endpoint) == (False, None)
    assert url not in words and url not in dumps(record), "no URL text is shown"


def test_no_url_in_the_environment_is_unknown_and_sends_nothing(
    fake_judge, monkeypatch
):
    monkeypatch.delenv(URL_VAR, raising=False)
    value, words, kept = assess(ReplyJudge(claim=CLAIM, url_env=URL_VAR))
    record = only(kept)
    assert value is None and f"no judge service URL: {URL_VAR} is not set" in words
    assert (record.sent, record.endpoint, record.request["claim"]) == (
        False,
        None,
        CLAIM,
    )
    assert fake_judge.requests == []


def test_an_empty_url_in_the_check_is_unset():
    value, words, _ = assess(ReplyJudge(claim=CLAIM, url=""))
    assert value is None and "no judge service URL: the check's url is not set" in words


def test_the_url_comes_from_the_environment_when_assessed(fake_judge, monkeypatch):
    checked = ReplyJudge(claim=CLAIM, url_env=URL_VAR)
    monkeypatch.setenv(URL_VAR, fake_judge.url("/supported"))
    assert assess(checked)[0] is True
    monkeypatch.setenv(URL_VAR, fake_judge.url("/contradicted"))
    assert assess(checked)[0] is False, "read again for each grading"


def test_no_token_in_the_environment_is_unknown_and_sends_nothing(
    fake_judge, monkeypatch
):
    monkeypatch.delenv(TOKEN_VAR, raising=False)
    value, words, _ = assess(check(fake_judge.url("/supported"), token_env=TOKEN_VAR))
    assert value is None and f"no judge service token: {TOKEN_VAR} is not set" in words
    assert fake_judge.requests == []


needs_environb = pytest.mark.skipif(
    not hasattr(os, "environb"), reason="no bytes environment on this platform"
)


@needs_environb
def test_a_url_with_bytes_that_are_not_utf8_is_unknown_with_nothing_sent(
    fake_judge, monkeypatch
):
    monkeypatch.setitem(
        os.environb, URL_VAR.encode(), fake_judge.url("/supported").encode() + b"/\xff"
    )
    value, words, kept = assess(ReplyJudge(claim=CLAIM, url_env=URL_VAR))
    record = only(kept)
    assert (
        value is None
        and f"the judge service URL in {URL_VAR} is not UTF-8 text" in words
    )
    assert (record.sent, record.endpoint) == (False, None) and fake_judge.requests == []
    dumps(record)


@needs_environb
def test_a_token_with_bytes_that_are_not_utf8_is_unknown_without_the_token(
    fake_judge, monkeypatch
):
    monkeypatch.setitem(
        os.environb, TOKEN_VAR.encode(), b"secret-part-7c1e\xffsecret-tail-9a2b"
    )
    value, words, kept = assess(
        check(fake_judge.url("/supported"), token_env=TOKEN_VAR)
    )
    why = f"the judge service token in {TOKEN_VAR} can't be sent: it isn't printable ASCII"
    assert value is None and why in words and not only(kept).sent
    assert fake_judge.requests == []
    for part in ("secret-part-7c1e", "secret-tail-9a2b"):
        assert no_secret_in(part, words, dumps(only(kept)), repr(kept))


@pytest.mark.parametrize(
    "token",
    [
        "tökén-5f3a9c0e",
        "line\nbreak-5f3a9c0e",
        "tab\tinside-5f3a9c0e",
        "del\x7f-5f3a9c0e",
        "unit\x1fseparator-5f3a9c0e",
    ],
)
def test_a_token_a_header_cannot_carry_is_unknown_without_it(
    fake_judge, monkeypatch, token
):
    monkeypatch.setenv(TOKEN_VAR, token)
    value, words, kept = assess(
        check(fake_judge.url("/supported"), token_env=TOKEN_VAR)
    )
    assert value is None and "it isn't printable ASCII" in words
    assert fake_judge.requests == []
    assert no_secret_in("5f3a9c0e", words, dumps(only(kept)), repr(kept))


@pytest.mark.parametrize("token", ["~", "a~b-5f3a", "tok en-5f3a"])
def test_a_token_of_printable_ascii_is_sent_as_it_is(fake_judge, monkeypatch, token):
    monkeypatch.setenv(TOKEN_VAR, token)
    value, _, _ = assess(check(fake_judge.url("/supported"), token_env=TOKEN_VAR))
    assert value is True
    assert fake_judge.requests[0].headers["authorization"] == f"Bearer {token}"


def test_an_empty_token_in_the_environment_is_unset(fake_judge, monkeypatch):
    monkeypatch.setenv(TOKEN_VAR, "")
    value, words, _ = assess(check(fake_judge.url("/supported"), token_env=TOKEN_VAR))
    assert value is None and f"no judge service token: {TOKEN_VAR} is not set" in words
    assert fake_judge.requests == []


# --- the token: sent, and kept nowhere ---------------------------------------------------------


@pytest.mark.parametrize("path", ["/supported", "/echo-token", "/500"])
def test_the_token_is_sent_as_a_bearer_and_kept_nowhere(
    fake_judge, monkeypatch, caplog, path
):
    monkeypatch.setenv(TOKEN_VAR, TOKEN)
    caplog.set_level(logging.DEBUG)
    checked = check(fake_judge.url(path), token_env=TOKEN_VAR)
    _, words, kept = assess(checked)
    [received] = fake_judge.requests
    assert received.headers["authorization"] == f"Bearer {TOKEN}"
    assert TOKEN not in json.dumps(received.body)
    assert no_secret_in(
        TOKEN, words, repr(checked), repr(kept), dumps(only(kept)), caplog.text
    )
    if path == "/echo-token":
        assert only(kept).rationale == "Bearer [redacted]"


def test_the_token_is_kept_nowhere_when_the_service_is_unreachable(monkeypatch):
    monkeypatch.setenv(TOKEN_VAR, TOKEN)
    value, words, kept = assess(
        check(f"http://127.0.0.1:{free_port()}/", token_env=TOKEN_VAR)
    )
    assert value is None
    assert no_secret_in(TOKEN, words, dumps(only(kept)), repr(kept))


def test_user_info_in_the_url_never_replaces_the_bearer_token(fake_judge, monkeypatch):
    monkeypatch.setenv(TOKEN_VAR, TOKEN)
    url = f"http://user:secret@{fake_judge.host_port}/supported"
    assess(check(url, token_env=TOKEN_VAR))
    assess(check(url))
    with_token, without = fake_judge.requests
    assert with_token.headers["authorization"] == f"Bearer {TOKEN}"
    assert "authorization" not in without.headers


def test_a_token_echoed_in_response_headers_never_reaches_a_log(
    fake_judge, monkeypatch, caplog
):
    monkeypatch.setenv(TOKEN_VAR, TOKEN)
    caplog.set_level(logging.DEBUG)
    value, words, kept = assess(
        check(fake_judge.url("/echo-in-headers"), token_env=TOKEN_VAR)
    )
    assert value is True
    assert fake_judge.requests[-1].headers["authorization"] == f"Bearer {TOKEN}"
    assert no_secret_in(TOKEN, caplog.text, words, dumps(only(kept)))
    assert not [r for r in caplog.records if TOKEN in r.getMessage()]


def spellings(secret):
    """``secret`` as it is, as repr writes it, and as JSON writes it once and twice (a reason in JSON in JSON)."""
    once = json.dumps(secret)[1:-1]
    return [
        secret,
        repr(secret),
        repr(secret)[1:-1],
        once,
        json.dumps(once)[1:-1],
        "sn3aky",
    ]


@pytest.mark.parametrize(
    ("path", "passed", "why"),
    [
        ("/echo-as-schema", None, f'schema is not "{SCHEMA}"'),
        (
            "/echo-as-verdict",
            None,
            'verdict is not one of "supported", "contradicted", "unresolved"',
        ),
        ("/echo-as-id", None, "id is not the request's id"),
        ("/echo-as-field-name", None, "it has 1 field outside the schema"),
        ("/echo-in-headers", True, None),
        ("/echo-as-record-fields", True, None),
    ],
)
def test_a_token_the_service_echoes_back_is_in_no_reason_or_record_however_spelled(
    fake_judge, monkeypatch, caplog, path, passed, why
):
    monkeypatch.setenv(TOKEN_VAR, SNEAKY)
    caplog.set_level(logging.DEBUG)
    value, words, kept = assess(check(fake_judge.url(path), token_env=TOKEN_VAR))
    record = only(kept)
    assert fake_judge.requests[0].headers["authorization"] == f"Bearer {SNEAKY}"
    assert value is passed
    if why is not None:
        assert why in words
    dumped = [
        words,
        repr(kept),
        repr(record),
        dumps(record),
        json.dumps(dumps(record)),
        json.dumps({"reason": words, "judgements": [record.to_json()]}, indent=2),
        caplog.text,
    ]
    found = [
        (form, text) for form in spellings(SNEAKY) for text in dumped if form in text
    ]
    assert found == []


@pytest.mark.parametrize(
    "token",
    ["supported", "model", "schema", "verdict", "request_id", "null", "fake-judge-1"],
)
def test_a_token_that_is_a_field_name_or_value_leaves_the_record_valid(
    fake_judge, monkeypatch, token
):
    monkeypatch.setenv(TOKEN_VAR, token)
    value, _, kept = assess(check(fake_judge.url("/supported"), token_env=TOKEN_VAR))
    record = only(kept)
    assert value is True
    assert record.verdict == "supported" and record.http_status == 200
    for text in (record.rationale, record.model, json.dumps(record.details)):
        assert text is None or token not in text
    assert (
        JudgeRecord.from_json(record.to_json()) == record
    ), "the schema survives redaction"


def test_a_token_inside_the_rationale_is_redacted_there(fake_judge, monkeypatch):
    monkeypatch.setenv(TOKEN_VAR, "makes the claim")
    value, _, kept = assess(check(fake_judge.url("/supported"), token_env=TOKEN_VAR))
    assert value is True and only(kept).rationale == "the reply [redacted] supported"


def test_a_redaction_that_fails_keeps_none_of_the_answer_and_never_escapes(
    fake_judge, monkeypatch
):
    import garak.cli

    def broken(value, secrets):
        raise RuntimeError(secrets)

    monkeypatch.setattr(garak.cli, "scrub_secrets", broken)
    monkeypatch.setenv(TOKEN_VAR, TOKEN)
    value, words, kept = assess(
        check(fake_judge.url("/supported"), token_env=TOKEN_VAR)
    )
    record = only(kept)
    assert value is None
    assert (
        record.failure
        == "the judge service's answer could not be redacted (RuntimeError)"
    )
    assert (record.verdict, record.rationale, record.model, record.details) == (
        None,
    ) * 4
    assert (record.sent, record.http_status) == (True, 200)
    assert no_secret_in(TOKEN, words, dumps(record), repr(kept))


def test_a_rationale_with_a_lone_surrogate_is_kept_escaped(fake_judge):
    value, _, kept = assess(check(fake_judge.url("/surrogate-rationale")))
    assert value is True and only(kept).rationale == "half \\ud800 a pair"
    dumps(only(kept))


# --- a request that cannot be built --------------------------------------------------------------


def test_a_request_that_cannot_be_built_is_unknown_with_a_record_and_sends_nothing(
    fake_judge,
):
    handle = SimpleNamespace(
        function="transfer",
        args={"handle": object()},
        result="ok",
        error=None,
        complete=True,
        blocked=False,
        turn=0,
    )
    value, words, kept = assess(
        check(fake_judge.url("/supported"), tool_calls=True), context(calls=[handle])
    )
    record = only(kept)
    assert value is None
    assert "the judge request could not be built (TypeError)" in words
    assert (record.sent, record.request["claim"], record.verdict) == (
        False,
        CLAIM,
        None,
    )
    assert JudgeRecord.from_json(record.to_json()) == record
    assert fake_judge.requests == []


@pytest.mark.parametrize(
    ("claim", "rubric", "setting"),
    [
        (CLAIM, {"criteria": [f"half {LONE}"]}, "rubric"),
        (f"{CLAIM} {LONE}", None, "claim"),
    ],
)
def test_a_check_built_by_hand_with_a_lone_surrogate_is_unknown_with_a_record(
    fake_judge, claim, rubric, setting
):
    value, words, kept = assess(
        ReplyJudge(claim=claim, rubric=rubric, url=fake_judge.url("/supported"))
    )
    why = f"the judge request could not be built (the check's {setting} is not UTF-8 text: it has a lone surrogate)"
    record = only(kept)
    assert value is None and why in words
    assert (record.sent, record.failure) == (False, why)
    escaped = (
        record.request["rubric"] if setting == "rubric" else record.request["claim"]
    )
    assert "\\ud800" in json.dumps(escaped) and LONE not in json.dumps(
        escaped, ensure_ascii=False
    )
    dumps(record)
    assert fake_judge.requests == []


# --- the backstop: unknown, recorded, never raised -------------------------------------------------


def broken(*args, **kwargs):
    raise RuntimeError(f"boom with Authorization: Bearer {TOKEN} and {TOKEN}")


@pytest.mark.parametrize(
    ("step", "sent", "status"),
    [
        ("parse_url", False, None),  # before the request goes out
        ("post", True, None),  # as it goes out
        ("response_problem", True, 200),  # after the answer came back
        ("_redacted", True, 200),  # while recording it
    ],
)
def test_any_exception_in_the_judge_path_is_unknown_recorded_and_never_shows_the_token(
    fake_judge, monkeypatch, step, sent, status
):
    target = {
        "parse_url": _judge_http,
        "post": _judge_http,
        "response_problem": _reply_judge._judge_spec(),
        "_redacted": _reply_judge,
    }[step]
    monkeypatch.setattr(target, step, broken)
    monkeypatch.setenv(TOKEN_VAR, TOKEN)
    value, words, kept = assess(
        check(fake_judge.url("/supported"), token_env=TOKEN_VAR)
    )
    record = only(kept)
    assert value is None and "the judge check failed (RuntimeError)" in words
    assert (record.sent, record.http_status, record.verdict) == (sent, status, None)
    assert record.failure == "the judge check failed (RuntimeError)"
    assert record.request["claim"] == CLAIM
    assert record.endpoint == (
        fake_judge.url("/supported") if step != "parse_url" else None
    )
    assert no_secret_in(TOKEN, words, dumps(record), repr(kept))
    assert len(fake_judge.requests) == (
        1 if step in ("response_problem", "_redacted") else 0
    )


def test_a_backstop_record_is_kept_for_each_request_asked_a_different_way(
    fake_judge, monkeypatch
):
    monkeypatch.setattr(_judge_http, "post", broken)
    kept: dict = {}
    assess(check(fake_judge.url("/supported")), kept=kept)
    assess(check(fake_judge.url("/contradicted")), kept=kept)
    assert len(kept) == 2


def test_a_record_that_cannot_be_written_is_the_backstops_failure(
    fake_judge, monkeypatch
):
    redacted = _reply_judge._redacted

    def unwritable(record, token):
        return dataclasses.replace(redacted(record, token), rationale=LONE)

    monkeypatch.setattr(_reply_judge, "_redacted", unwritable)
    value, words, kept = assess(check(fake_judge.url("/supported")))
    record = only(kept)
    assert value is None and "the judge check failed (UnicodeEncodeError)" in words
    assert (record.http_status, record.verdict, record.rationale) == (200, None, None)


def test_the_backstop_keeps_no_request_it_cannot_write(fake_judge, monkeypatch):
    request = {"schema": SCHEMA, "id": "sha256:" + "0" * 64, "claim": LONE}
    monkeypatch.setattr(ReplyJudge, "request", lambda self, ctx: request)
    value, words, kept = assess(check(fake_judge.url("/supported")))
    record = only(kept)
    assert value is None and "the judge check failed (UnicodeEncodeError)" in words
    assert (record.request, record.request_id, record.sent) == ({}, "", False)
    assert fake_judge.requests == []


def test_the_backstop_record_holds_even_when_the_request_is_unknown(monkeypatch):
    monkeypatch.setattr(ReplyJudge, "_judgement", broken)
    monkeypatch.setattr(ReplyJudge, "unbuilt_request", broken)
    value, words, kept = assess(check("http://127.0.0.1:9/judge"))
    record = only(kept)
    assert value is None and "the judge check failed (RuntimeError)" in words
    assert (record.request, record.request_id, record.sent) == ({}, "", False)
    assert no_secret_in(TOKEN, words, dumps(record))


def test_a_missing_spec_package_is_the_installations_fault_and_is_raised(monkeypatch):
    def missing():
        raise MissingSnapshotCodecError("install it")

    monkeypatch.setattr(_reply_judge, "_judge_spec", missing)
    with pytest.raises(MissingSnapshotCodecError):
        assess(check("http://127.0.0.1:9/judge"))


# --- a replay answers from records, and asks nothing -----------------------------------------------

MODEL, PROMPT = "judge-1", "prompt-v1"


def recorded(request, verdict="supported", *, endpoint=None, **more):
    return JudgeRecord(
        request_id=request["id"],
        endpoint=endpoint,
        request=request,
        sent=True,
        http_status=200,
        verdict=verdict,
        model=MODEL,
        details={"prompt_version": PROMPT},
        **more,
    )


def replay_of(*records):
    return RecordedJudge(records, model=MODEL, prompt_version=PROMPT)


def test_a_replay_answers_from_its_record_and_never_asks(fake_judge):
    checked = check(fake_judge.url("/500"))
    request = checked.request(context())
    value, words, kept = assess(
        checked, replay=replay_of(recorded(request, "contradicted"))
    )
    assert value is False and request["id"] in words
    assert fake_judge.requests == []
    assert only(kept).verdict == "contradicted"


@pytest.mark.parametrize(
    ("records", "why"),
    [
        ([], "judge-replay-unrecorded"),
        ([("supported", {"model": "other"})], "judge-replay-identity"),
        ([("supported", None), ("contradicted", None)], "judge-replay-ambiguous"),
    ],
)
def test_a_replay_that_cannot_answer_is_unknown_with_its_code_and_asks_nothing(
    fake_judge, records, why
):
    checked = ReplyJudge(
        claim=CLAIM, url_env=URL_VAR
    )  # the endpoint is not known to a replay
    request = checked.request(context())
    held = []
    for verdict, change in records:
        held.append(recorded(request, verdict))
        if change:
            held[-1] = JudgeRecord(**{**held[-1].to_json(), **change})
    kept: dict = {}
    value, words, _ = assess(checked, kept=kept, replay=replay_of(*held))
    id8 = request["id"].removeprefix("sha256:")[:8]
    assert (value, words) == (None, f"{why}:{id8}")
    record = only(kept)
    assert (record.sent, record.failure, record.request_id) == (
        False,
        words,
        request["id"],
    )
    assert fake_judge.requests == []


def test_a_refused_replay_is_kept_and_read_again_as_the_failed_record_it_is():
    checked = check("http://judge.invalid/judge")
    kept: dict = {}
    first = assess(checked, kept=kept, replay=replay_of())[:2]
    again = assess(checked, kept=kept, replay=replay_of())[:2]
    assert first[0] is None and again[0] is None
    assert again[1] == f"{first[1]}; request {only(kept).request_id}"


def test_a_replay_picks_by_the_endpoint_a_literal_url_names_and_never_by_a_variable():
    request = check("http://a.invalid/judge").request(context())
    here = recorded(request, "supported", endpoint="http://a.invalid/judge")
    there = recorded(request, "contradicted", endpoint="http://b.invalid/judge")
    replay = replay_of(here, there)
    assert assess(check("http://a.invalid/judge"), replay=replay)[0] is True
    assert assess(check("HTTP://B.invalid:80/judge?x=1"), replay=replay)[0] is False
    assert assess(ReplyJudge(claim=CLAIM, url_env=URL_VAR), replay=replay)[
        1
    ].startswith("judge-replay-ambiguous:")
    assert assess(check("http://c.invalid/judge"), replay=replay)[1].startswith(
        "judge-replay-ambiguous:"
    )


def test_a_replay_of_a_url_nothing_can_read_names_no_endpoint():
    request = check("not a url").request(context())
    named = recorded(request, "supported", endpoint=None)
    other = recorded(request, "contradicted", endpoint="http://b.invalid/judge")
    assert assess(check("not a url"), replay=replay_of(named, other))[0] is True


def test_a_replay_is_not_asked_for_a_request_that_could_not_be_built(fake_judge):
    handle = SimpleNamespace(
        function="f",
        args={"h": object()},
        result=None,
        error=None,
        complete=True,
        blocked=False,
        turn=0,
    )
    checked = check(fake_judge.url("/supported"), tool_calls=True)
    value, words, _ = assess(checked, context(calls=[handle]), replay=replay_of())
    assert value is None and "could not be built (TypeError)" in words
    assert "judge-replay" not in words
