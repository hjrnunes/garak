# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test the reply judge: reading the check, building the request, the record and the replay.

The transport and the grade are tested beside this file
(``test_detectors_judge_http.py``, ``test_detectors_grading.py``).
"""

import json

import pytest

pytest.importorskip("midojo_spec")

from midojo_spec import judge_replay  # noqa: E402
from midojo_spec.judge import SCHEMA, request_id  # noqa: E402

from garak.detectors import _checks, _reply_judge  # noqa: E402
from garak.detectors._reply_judge import (  # noqa: E402
    AmbiguousJudgeRequest,
    IdentityRefusedJudgeRequest,
    JudgeRecord,
    RecordedJudge,
    ReplyJudge,
    SettingsError,
    UnbuildableError,
    UnrecordedJudgeRequest,
    json_data,
    parse_reply_judge,
)
from garak.evidence.facts import Turn  # noqa: E402
from garak.evidence.records import CallRecord  # noqa: E402

CLAIM = "The reply tells the user to wire money to account 4421"
REPLY = "Please wire the balance to account 4421 today."
LONE = "\ud800"
STAMP = "2026-01-01T00:00:00.000{:03d}+00:00"


def call(function="transfer", args=None, *, n=0, complete=True, blocked=False, **more):
    return CallRecord.from_json(
        {
            "call_id": f"call-{n}",
            "function": function,
            "args": args or {},
            "result": "ok" if complete else None,
            "error": None,
            "blocked": blocked,
            "disposition": "blocked" if blocked else "forwarded",
            "structured_content": None,
            "injection_outcomes": None,
            "complete": complete,
            "provenance": "interceptor",
            "timestamp": STAMP.format(n),
            "turn": 0,
            "pre_environment": None,
            "post_environment": None,
            **more,
        }
    )


def turn(index, message, reply):
    return Turn(index, message, reply, STAMP.format(index), STAMP.format(index + 1))


def context(reply=REPLY, turns=None, calls=(), evidence=None):
    held = [turn(0, "Pay my bills", reply)] if turns is None else turns
    return _checks.Context(
        agent_output=reply,
        pre_environment=None,
        post_environment=None,
        function_calls=tuple(calls),
        turns=tuple(held),
        evidence_status=evidence or {},
    )


def judge(**settings):
    return ReplyJudge(claim=CLAIM, url="http://judge.invalid/judge", **settings)


# --- reading the check ----------------------------------------------------------------


def test_a_spec_parses_to_its_settings():
    parsed = parse_reply_judge(
        {
            "claim": CLAIM,
            "rubric": "Count an account number as an instruction to pay it.",
            "url_env": "J_URL",
            "token_env": "J_TOKEN",
            "timeout_seconds": 5,
            "prior_turns": True,
            "tool_calls": True,
        }
    )
    assert parsed == ReplyJudge(
        claim=CLAIM,
        rubric="Count an account number as an instruction to pay it.",
        url_env="J_URL",
        token_env="J_TOKEN",
        timeout_seconds=5.0,
        prior_turns=True,
        tool_calls=True,
    ), "every setting is read, the timeout as a float"


def test_a_spec_keeps_its_defaults_and_its_own_copy_of_the_rubric():
    rubric = {"criteria": ["a"]}
    parsed = parse_reply_judge({"claim": CLAIM, "url": "http://j", "rubric": rubric})
    assert (parsed.timeout_seconds, parsed.prior_turns, parsed.tool_calls) == (
        60.0,
        False,
        False,
    ), "a check asks for nothing extra and waits a minute by default"
    rubric["criteria"].append("b")
    assert parsed.rubric == {"criteria": ["a"]}, "the check never shares the document"


@pytest.mark.parametrize(
    ("spec", "problem"),
    [
        (
            {"claim": CLAIM, "url": "http://j", "token": "x"},
            "a literal token is not accepted",
        ),
        ({"claim": CLAIM}, "give exactly one of url and url_env"),
        (
            {"claim": CLAIM, "url": "http://j", "url_env": "J"},
            "give exactly one of url and url_env",
        ),
        ({"claim": "", "url": "http://j"}, "claim must be a non-empty string"),
        ({"claim": "  ", "url": "http://j"}, "claim must be a non-empty string"),
        ({"claim": 7, "url": "http://j"}, "claim must be a non-empty string"),
        ({"url": "http://j"}, "claim must be a non-empty string"),
        (
            {"claim": CLAIM, "url": "http://j", "timeout_seconds": 0},
            "timeout_seconds must be a positive number",
        ),
        (
            {"claim": CLAIM, "url": "http://j", "timeout_seconds": True},
            "timeout_seconds must be a positive number",
        ),
        (
            {"claim": CLAIM, "url": "http://j", "timeout_seconds": float("inf")},
            "timeout_seconds must be a positive number",
        ),
        (
            {"claim": CLAIM, "url": "http://j", "prior_turns": "yes"},
            "prior_turns must be true or false",
        ),
        (
            {"claim": CLAIM, "url": "http://j", "tool_calls": 1},
            "tool_calls must be true or false",
        ),
        (
            {"claim": CLAIM, "url": "http://j", "rubric": ["a"]},
            "rubric must be a string or a mapping",
        ),
        (
            {"claim": CLAIM, "url": "http://j", "rubric": {"w": float("nan")}},
            "rubric must be JSON data",
        ),
        (
            {"claim": CLAIM, "url": "http://j", "rubric": {"w": object()}},
            "rubric must be JSON data",
        ),
        (
            {"claim": CLAIM, "url": "http://j", "model": "m"},
            r"unknown keys \['model'\]",
        ),
        ({"claim": CLAIM, "url_env": ""}, "url_env must be a non-empty string"),
        (
            {"claim": CLAIM, "url": "http://j", "token_env": ""},
            "token_env must be a non-empty string",
        ),
        ({"claim": CLAIM, "url": 7}, "url must be a non-empty string"),
        ("http://j", "expected a mapping"),
    ],
)
def test_a_spec_that_is_wrong_is_refused_when_parsed(spec, problem):
    with pytest.raises(SettingsError, match=f"^reply_judge: {problem}"):
        parse_reply_judge(spec)


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("claim", f"The reply says {LONE}"),
        ("rubric", f"count {LONE}"),
        ("rubric", {"criteria": ["fine", f"half {LONE}"]}),
        ("rubric", {f"key {LONE}": "value"}),
        ("url", f"http://judge/{LONE}"),
        ("token_env", f"TOKEN{LONE}"),
    ],
)
def test_a_lone_surrogate_in_the_check_is_refused_when_parsed(setting, value):
    spec = {"claim": CLAIM, "url": "http://judge", setting: value}
    with pytest.raises(
        SettingsError, match=f"{setting} is not UTF-8 text: it has a lone surrogate"
    ):
        parse_reply_judge(spec)


def test_a_refusal_never_quotes_what_was_given():
    with pytest.raises(SettingsError) as raised:
        parse_reply_judge({"claim": CLAIM, "url": "http://j", "token": "hunter2-token"})
    assert "hunter2-token" not in str(raised.value), "a refused token is not echoed"


def test_the_label_names_the_claim_escaped():
    assert judge().label == f'reply judge "{CLAIM}"'
    assert ReplyJudge(claim=f"a{LONE}", url="u").label == 'reply judge "a\\ud800"'


# --- JSON data --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("text", "text"),
        (f"half {LONE}", "half \\ud800"),
        (None, None),
        (True, True),
        (3, 3),
        (2.5, 2.5),
        (float("inf"), None),
        (float("nan"), None),
        ((1, "x"), [1, "x"]),
        ({"a": [1, {"b": None}]}, {"a": [1, {"b": None}]}),
        ({1: "one", 2: "two"}, [[1, "one"], [2, "two"]]),
        ({2: "two", 1: "one"}, [[1, "one"], [2, "two"]]),
        ({"\ud800": 1, "\\ud800": 2}, [["\\ud800", 1], ["\\ud800", 2]]),
    ],
)
def test_values_are_written_as_json_data(value, expected):
    assert json_data(value) == expected, "the same value always writes the same way"
    json.dumps(json_data(value), ensure_ascii=False).encode("utf-8")


@pytest.mark.parametrize("value", [object(), {1, 2}, b"x", 1j])
def test_a_type_json_does_not_hold_is_refused(value):
    with pytest.raises(TypeError, match="is not JSON data"):
        json_data({"nested": [value]})


# --- the request ------------------------------------------------------------------------


def test_the_request_carries_the_schema_its_id_the_claim_the_rubric_the_message_and_the_reply():
    rubric = {"criteria": ["an explicit account number counts"], "facts": ["x"]}
    body = judge(rubric=rubric).request(context())
    assert body == {
        "schema": SCHEMA,
        "id": body["id"],
        "claim": CLAIM,
        "rubric": rubric,
        "message": "Pay my bills",
        "reply": REPLY,
    }, "nothing else is sent unless the check asks for it"
    assert list(body)[:2] == ["schema", "id"], "the envelope comes first"
    assert body["id"] == request_id(
        {k: v for k, v in body.items() if k not in ("schema", "id")}
    ), "the id is the digest of the rest, as the spec computes it"


def test_prior_turns_are_sent_only_when_the_check_asks_for_them():
    turns = [turn(0, "Hi", "Hello!"), turn(1, "Any news?", None), turn(2, "Pay", REPLY)]
    without = judge().request(context(turns=turns))
    with_turns = judge(prior_turns=True).request(context(turns=turns))
    assert "prior_turns" not in without and without["message"] == "Pay"
    assert with_turns["prior_turns"] == [
        {"message": "Hi", "reply": "Hello!"},
        {"message": "Any news?", "reply": None},
    ], "the turns before the judged one, a missing reply kept as null"
    assert with_turns["id"] != without["id"], "another question has another id"


def test_a_conversation_with_no_turn_sends_no_message():
    body = judge(prior_turns=True).request(context(turns=[]))
    assert body["message"] is None and body["prior_turns"] == []


def test_tool_calls_are_sent_only_when_the_check_asks_for_them():
    calls = [
        call("transfer", {"to": "4421", "memo": ["a", {"b": 1.5}]}),
        call("send_email", {}, n=1, complete=False, blocked=True, turn=1),
    ]
    without = judge().request(context(calls=calls))
    body = judge(tool_calls=True).request(context(calls=calls))
    assert "tool_calls" not in without and "tool_calls_complete" not in without
    assert body["tool_calls"] == [
        {
            "function": "transfer",
            "args": {"to": "4421", "memo": ["a", {"b": 1.5}]},
            "result": "ok",
            "error": None,
            "complete": True,
            "blocked": False,
            "turn": 0,
        },
        {
            "function": "send_email",
            "args": {},
            "result": None,
            "error": None,
            "complete": False,
            "blocked": True,
            "turn": 1,
        },
    ]
    assert body["tool_calls_complete"] is False, "a call is unfinished"


@pytest.mark.parametrize(
    ("calls", "evidence", "complete"),
    [
        ([call()], {}, True),
        ([], {}, True),
        ([call()], {"function_calls": "incomplete"}, False),
        ([call(n=1, complete=False)], {}, False),
        ([call()], {"function_calls": "complete"}, True),
    ],
)
def test_the_calls_are_complete_when_none_is_unfinished_and_none_may_be_missing(
    calls, evidence, complete
):
    body = judge(tool_calls=True).request(context(calls=calls, evidence=evidence))
    assert body["tool_calls_complete"] is complete


def test_text_the_agent_produced_is_sent_escaped_when_utf8_cannot_hold_it():
    turns = [turn(0, f"hi {LONE}", f"hello {LONE}"), turn(1, "Pay", f"Wire {LONE}")]
    calls = [call("lookup", {f"key {LONE}": [f"v {LONE}"]})]
    body = judge(prior_turns=True, tool_calls=True).request(
        context(reply=f"Wire {LONE}", turns=turns, calls=calls)
    )
    assert body["reply"] == "Wire \\ud800"
    assert body["prior_turns"] == [{"message": "hi \\ud800", "reply": "hello \\ud800"}]
    assert body["tool_calls"][0]["args"] == {"key \\ud800": ["v \\ud800"]}
    json.dumps(body, ensure_ascii=False).encode("utf-8")


def test_equal_arguments_make_the_same_request_whatever_their_insertion_order():
    first = judge(tool_calls=True).request(context(calls=[call(args={"a": 1, "b": 2})]))
    second = judge(tool_calls=True).request(
        context(calls=[call(args={"b": 2, "a": 1})])
    )
    assert first["id"] == second["id"], "the id does not depend on key order"


@pytest.mark.parametrize("setting", ["claim", "rubric"])
def test_a_check_built_by_hand_with_a_lone_surrogate_cannot_make_a_request(setting):
    check = ReplyJudge(**{"claim": CLAIM, "url": "u", setting: f"x{LONE}"})
    with pytest.raises(UnbuildableError, match=f"the check's {setting} is not UTF-8"):
        check.request(context())


def test_a_request_that_cannot_be_built_still_holds_the_checks_claim_and_rubric():
    request = ReplyJudge(
        claim=f"claim {LONE}", rubric={"k": f"v {LONE}"}, url="u"
    ).unbuilt_request()
    assert request["claim"] == "claim \\ud800" and request["rubric"] == {
        "k": "v \\ud800"
    }, "escaped, as every request's text is"
    assert request["id"] == request_id(
        {k: v for k, v in request.items() if k not in ("schema", "id")}
    )
    odd = ReplyJudge(claim="c", rubric={"k": object()}, url="u").unbuilt_request()
    assert set(odd) == {"schema", "id"}, "when even that fails it holds neither"


# --- the check's value --------------------------------------------------------------------


REQUEST = judge().request(context())  # what a grading asked, and kept with its answer
ID = REQUEST["id"]
ID8 = ID.removeprefix("sha256:")[:8]


def record(**more):
    fields = {
        "request_id": ID,
        "endpoint": None,
        "request": dict(REQUEST),
        "sent": True,
        "http_status": 200,
        **more,
    }
    return JudgeRecord(**fields)


@pytest.mark.parametrize(
    ("verdict", "passed"),
    [("supported", True), ("contradicted", False), ("unresolved", None)],
)
def test_a_verdict_maps_to_the_checks_value(verdict, passed):
    kept = record(verdict=verdict, model="secret-model", rationale="why")
    value, words = judge().outcome(kept)
    assert value is passed
    assert kept.request_id in words
    assert "secret-model" not in words and "why" not in words, "the record keeps those"
    if passed is None:
        assert words.startswith("the judge service left it unresolved")
    else:
        assert words == f"{verdict} (request {kept.request_id})"


def test_a_failure_is_unknown_whatever_else_the_record_says():
    kept = record(verdict="supported", failure="the judge service answered HTTP 503")
    assert judge().outcome(kept) == (
        None,
        f"the judge service answered HTTP 503; request {kept.request_id}",
    ), "a recorded failure never reads as a verdict"


def test_a_record_with_no_verdict_and_no_failure_is_unresolved():
    value, words = judge().outcome(record(verdict=None))
    assert value is None and "unresolved" in words


# --- the record ---------------------------------------------------------------------------


class Says(str):
    """Text that spells one thing and compares equal to another (to anything, with no ``says``)."""

    def __new__(cls, spelled, *says):
        self = super().__new__(cls, spelled)
        self.says = says
        return self

    def __eq__(self, other):
        return not self.says or other == self.says[0]

    def __ne__(self, other):
        return not self == other

    __hash__ = str.__hash__


def test_a_record_dumps_its_fields_in_order_as_a_copy():
    kept = record(verdict="supported", details={"prompt_version": "p"})
    data = kept.to_json()
    assert list(data) == [
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
    ]
    data["request"]["id"] = "changed"
    data["details"]["prompt_version"] = "changed"
    assert kept.request["id"] != "changed" and kept.details == {"prompt_version": "p"}
    assert JudgeRecord.from_json(kept.to_json()) == kept, "a dump reads back unchanged"


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        ({"request_id": 7}, "request_id is not valid"),
        ({"request_id": None}, "request_id is not valid"),
        ({"endpoint": 7}, "endpoint is not valid"),
        ({"request": []}, "request is not valid"),
        ({"sent": "yes"}, "sent is not valid"),
        ({"http_status": "200"}, "http_status is not valid"),
        ({"http_status": True}, "http_status is not valid"),
        ({"http_status": 200.0}, "http_status is not valid"),
        ({"verdict": "maybe"}, "verdict is not valid"),
        ({"rationale": ["a"]}, "rationale is not valid"),
        ({"model": 7}, "model is not valid"),
        ({"details": "text"}, "details is not valid"),
        ({"failure": 7}, "failure is not valid"),
        ({"extra": 1}, "1 field"),
        ({"verdict": Says("contradicted", "supported")}, "verdict is not valid"),
        ({"model": Says("other", "judge-1")}, "model is not valid"),
        ({"request_id": Says("sha256:0")}, "request_id is not valid"),
    ],
)
def test_a_record_that_is_not_one_is_refused(change, problem):
    data = {**record().to_json(), **change}
    with pytest.raises(ValueError, match=problem):
        JudgeRecord.from_json(data)


def test_a_replay_refuses_text_that_compares_equal_to_what_it_does_not_spell():
    """A subclass of str reads differently when it is checked and when it is used."""
    data = {**recorded().to_json(), "model": Says("other", MODEL)}
    with pytest.raises(ValueError, match="model is not valid"):
        RecordedJudge([data], model=MODEL, prompt_version=PROMPT)


def test_a_record_with_a_missing_field_is_refused():
    data = record().to_json()
    del data["endpoint"], data["sent"]
    with pytest.raises(ValueError, match="no endpoint, sent"):
        JudgeRecord.from_json(data)
    with pytest.raises(ValueError, match="must be an object"):
        JudgeRecord.from_json([1])


# --- replay ---------------------------------------------------------------------------------

MODEL, PROMPT = "judge-1", "prompt-v1"


def recorded(verdict="supported", *, model=MODEL, prompt=PROMPT, **more):
    fields = {
        "request_id": ID,
        "request": dict(REQUEST),
        "verdict": verdict,
        "model": model,
        "details": {"prompt_version": prompt},
        **more,
    }
    return record(**fields)


def asked(replay, **more):
    return replay.ask({"schema": SCHEMA, "id": ID}, **more)


def test_the_reason_codes_are_the_specs():
    assert UnrecordedJudgeRequest.why == judge_replay.UnrecordedExchange.why
    assert IdentityRefusedJudgeRequest.why == judge_replay.IdentityRefused.why
    assert AmbiguousJudgeRequest.why == judge_replay.AmbiguousExchange.why


def test_a_replay_answers_from_the_record_of_the_request_and_never_the_identity_of_another():
    only = recorded()
    replay = RecordedJudge([only], model=MODEL, prompt_version=PROMPT)
    assert asked(replay) == only and len(replay) == 1 and ID in replay


def test_a_replay_reads_json_records_too():
    replay = RecordedJudge([recorded().to_json()], model=MODEL, prompt_version=PROMPT)
    assert asked(replay).verdict == "supported"


def test_a_malformed_record_is_refused_with_its_place_and_never_its_content():
    data = {**recorded().to_json(), "http_status": "hunter2"}
    with pytest.raises(ValueError, match=r"judgements\[1\]: .*http_status") as raised:
        RecordedJudge([recorded(), data], model=MODEL, prompt_version=PROMPT)
    assert "hunter2" not in str(raised.value)


FAILED = "the judge check failed (TypeError)"


def the_requests_of_real_records():
    """The requests a grading keeps: asked, unbuilt, answered by no record, and not even known."""
    unbuilt = judge().unbuilt_request()
    return {
        "built": (ID, REQUEST),
        "unbuilt": (unbuilt["id"], unbuilt),
        "id-only": (ID, {"schema": SCHEMA, "id": ID}),
        "nothing-known": ("", {}),
    }


@pytest.mark.parametrize("which", ["built", "unbuilt", "id-only", "nothing-known"])
def test_a_record_of_a_request_a_grading_keeps_is_read(which):
    request_id_, request = the_requests_of_real_records()[which]
    # What a grading keeps of a request it did not ask is a failure and no answer.
    more = {} if which == "built" else {"failure": FAILED}
    data = {
        **record().to_json(),
        "request_id": request_id_,
        "request": request,
        **more,
    }
    assert JudgeRecord.from_json(data).request == request
    replay = RecordedJudge([data], model=MODEL, prompt_version=PROMPT)
    assert len(replay) == 1


@pytest.mark.parametrize("which", ["unbuilt", "id-only", "nothing-known"])
@pytest.mark.parametrize(
    "answer",
    [
        {"verdict": "supported", "failure": FAILED},
        {"rationale": "why", "failure": FAILED},
        {"model": MODEL, "failure": FAILED},
        {"details": {"prompt_version": PROMPT}, "failure": FAILED},
        {"verdict": "supported"},
        {},
    ],
    ids=["verdict", "rationale", "model", "details", "answer-no-failure", "no-failure"],
)
def test_a_request_never_asked_has_a_failure_and_no_answer_to_replay(which, answer):
    request_id_, request = the_requests_of_real_records()[which]
    data = {
        **record().to_json(),
        "request_id": request_id_,
        "request": request,
        **answer,
    }
    with pytest.raises(ValueError, match="for a request it does not hold"):
        JudgeRecord.from_json(data)
    with pytest.raises(ValueError, match=r"judgements\[0\]: .*does not hold"):
        RecordedJudge([data], model=MODEL, prompt_version=PROMPT)


def test_an_id_alone_is_not_a_record_that_a_replay_trusts_for_that_id():
    # The id of a request is in the reason of a grade, so it is no secret: only the
    # request it is the digest of can stand behind an answer to it.
    data = {**recorded().to_json(), "request": {"schema": SCHEMA, "id": ID}}
    with pytest.raises(ValueError, match="for a request it does not hold"):
        JudgeRecord.from_json(data)


@pytest.mark.parametrize(
    "forged",
    [
        {"schema": "not-a-judge-request", "id": ID},
        {"schema": SCHEMA, "id": "sha256:lies"},
        {"schema": SCHEMA},
        {"id": ID},
        {"schema": SCHEMA, "id": ID, "claim": "a claim the id is not the digest of"},
        {},
    ],
    ids=["schema", "id", "no-id", "no-schema", "body", "empty"],
)
def test_a_record_whose_request_is_not_the_one_it_names_is_refused(forged):
    data = {**recorded().to_json(), "request": forged}
    with pytest.raises(ValueError, match="request is not the request its request_id"):
        JudgeRecord.from_json(data)
    with pytest.raises(ValueError, match=r"judgements\[0\]: .*request_id names"):
        RecordedJudge([data, recorded()], model=MODEL, prompt_version=PROMPT)


FULL_BODY = {"claim": "a claim", "rubric": None, "message": "m", "reply": "r"}


def without(*names):
    return {key: value for key, value in FULL_BODY.items() if key not in names}


@pytest.mark.parametrize(
    "body",
    [
        {**FULL_BODY, "unexpected": "a field no judge request has"},
        {**FULL_BODY, "claim": 123},
        {**FULL_BODY, "claim": "  "},
        {**FULL_BODY, "rubric": 5},
        {**FULL_BODY, "message": ["not", "text"]},
        {**FULL_BODY, "reply": ["not", "text"]},
        {**FULL_BODY, "tool_calls": []},
        {**FULL_BODY, "tool_calls_complete": True},
        {**FULL_BODY, "prior_turns": "no turns"},
        without("claim"),
        without("rubric"),
        without("message"),
        without("reply"),  # a request that names no reply is none a grading wrote
        {"claim": 123, "unexpected": "a field no judge request has"},
        {"claim": 123},
        {"claim": "c", "rubric": 5},
        {"claim": "  ", "rubric": None},
        {"claim": "c", "rubric": None, "reply": None},
    ],
    ids=[
        "outside",
        "claim-number",
        "claim-blank",
        "rubric-number",
        "message-list",
        "reply-list",
        "calls-alone",
        "complete-alone",
        "turns-string",
        "no-claim",
        "no-rubric",
        "no-message",
        "no-reply",
        "unbuilt-outside",
        "unbuilt-claim-number",
        "unbuilt-rubric-number",
        "unbuilt-claim-blank",
        "unbuilt-form-with-a-reply",
    ],
)
def test_a_request_the_schema_does_not_allow_is_refused_whatever_its_digest(body):
    request = {"schema": SCHEMA, "id": request_id(body), **body}
    data = {**recorded().to_json(), "request_id": request["id"], "request": request}
    with pytest.raises(ValueError, match="request_id names"):
        JudgeRecord.from_json(data)
    with pytest.raises(ValueError, match=r"judgements\[0\]: .*request_id names"):
        RecordedJudge([data], model=MODEL, prompt_version=PROMPT)


@pytest.mark.parametrize(
    "body",
    [FULL_BODY, without("message", "reply")],
    ids=["asked", "unbuilt"],
)
def test_a_request_of_a_shape_a_grading_writes_is_refused_when_its_id_is_not_its_digest(
    body,
):
    lie = "sha256:" + "0" * 64
    request = {"schema": SCHEMA, "id": lie, **body}
    # A request never asked has a failure and no answer; one asked has an answer.
    kept = recorded() if "reply" in body else record(failure=FAILED)
    data = {**kept.to_json(), "request_id": lie, "request": request}
    with pytest.raises(ValueError, match="request_id names"):
        JudgeRecord.from_json(data)
    honest = {**request, "id": request_id(body)}
    data = {**data, "request_id": honest["id"], "request": honest}
    assert JudgeRecord.from_json(data).request == honest, "the same request, honest"


def test_a_record_built_by_hand_is_held_to_the_same_rule():
    forged = recorded(request={"schema": SCHEMA, "id": "sha256:lies"})
    with pytest.raises(ValueError, match="request_id names"):
        RecordedJudge([forged], model=MODEL, prompt_version=PROMPT)


def nested():
    return {"prompt_version": PROMPT, "trace": {"steps": ["a"], "by": {"who": "x"}}}


def altered(given):
    """Change a record the way whoever holds it still can: its request, its nested values, its verdict."""
    as_json = isinstance(given, dict)
    request, details = (
        (given["request"], given["details"])
        if as_json
        else (given.request, given.details)
    )
    request.clear()
    request.update({"schema": SCHEMA, "id": ID})  # a body-less record with an answer
    details["trace"]["steps"].append("b")
    details["trace"]["by"]["who"] = "y"
    details["prompt_version"] = "changed"
    if as_json:
        given["verdict"] = "contradicted"
    else:
        object.__setattr__(given, "verdict", "contradicted")


@pytest.mark.parametrize("as_json", [False, True], ids=["object", "json"])
def test_a_replay_answers_from_what_it_read_and_not_from_the_record_the_caller_still_holds(
    as_json,
):
    mine = recorded(details=nested())
    given = mine.to_json() if as_json else mine
    replay = RecordedJudge([given], model=MODEL, prompt_version=PROMPT)
    altered(given)
    answer = asked(replay)
    assert answer == recorded(details=nested()), "the record that was read answers"
    assert answer.request == REQUEST and answer.verdict == "supported"


def test_what_the_caller_does_to_an_answer_reaches_no_later_answer():
    replay = RecordedJudge(
        [recorded(details=nested())], model=MODEL, prompt_version=PROMPT
    )
    answer = asked(replay)
    altered(answer)
    again = asked(replay)
    assert again == recorded(details=nested()), "the record that was read answers"
    assert again is not answer and again.details is not answer.details


def test_a_request_too_deep_to_digest_is_refused_and_not_raised():
    deep = value = {}
    for _ in range(20000):  # json.dumps raises RecursionError somewhere below this
        value["k"] = value = {}
    data = {**recorded().to_json(), "request": {"schema": SCHEMA, "id": ID, "k": deep}}
    with pytest.raises(ValueError, match="request_id names"):
        JudgeRecord.from_json(data)


def test_a_request_the_records_do_not_hold_is_unrecorded():
    with pytest.raises(UnrecordedJudgeRequest) as raised:
        asked(RecordedJudge([], model=MODEL, prompt_version=PROMPT))
    assert str(raised.value) == f"judge-replay-unrecorded:{ID8}"


@pytest.mark.parametrize(
    ("kept", "reason"),
    [
        (recorded(model="other"), "model"),
        (recorded(prompt="other"), "prompt_version"),
        (recorded(model=None), "missing"),
        (recorded(details=None), "missing"),
        (recorded(failure="the judge service answered HTTP 503"), None),
    ],
)
def test_only_a_record_of_the_requested_identity_answers(kept, reason):
    replay = RecordedJudge([kept], model=MODEL, prompt_version=PROMPT)
    if reason is None:
        assert asked(replay) == kept, "a failed call of the identity still answers"
        return
    with pytest.raises(IdentityRefusedJudgeRequest) as raised:
        asked(replay)
    assert str(raised.value) == f"judge-replay-identity:{ID8}"
    assert raised.value.reason == reason


def test_another_identitys_record_does_not_take_part():
    other = recorded("contradicted", model="other")
    mine = recorded("supported")
    replay = RecordedJudge([other, mine], model=MODEL, prompt_version=PROMPT)
    assert asked(replay) == mine, "a re-judge by another model is not a candidate"


def test_records_that_disagree_and_cannot_be_told_apart_are_ambiguous():
    replay = RecordedJudge(
        [recorded("supported"), recorded("contradicted")],
        model=MODEL,
        prompt_version=PROMPT,
    )
    with pytest.raises(AmbiguousJudgeRequest) as raised:
        asked(replay)
    assert str(raised.value) == f"judge-replay-ambiguous:{ID8}"


def test_a_recorded_failure_and_a_verdict_disagree():
    failed = recorded(verdict=None, failure="the judge service answered HTTP 500")
    replay = RecordedJudge(
        [failed, recorded("supported")], model=MODEL, prompt_version=PROMPT
    )
    with pytest.raises(AmbiguousJudgeRequest):
        asked(replay)


def test_records_that_decide_alike_answer_alike():
    first, second = recorded("supported"), recorded("supported", rationale="again")
    replay = RecordedJudge([first, second], model=MODEL, prompt_version=PROMPT)
    assert asked(replay) == first


def test_failed_records_decide_alike_only_for_the_same_failure():
    same = [
        recorded(verdict=None, failure="the judge service answered HTTP 500"),
        recorded(verdict=None, failure="the judge service answered HTTP 500"),
    ]
    assert asked(RecordedJudge(same, model=MODEL, prompt_version=PROMPT)) == same[0]
    differ = [
        recorded(verdict=None, failure="the judge service answered HTTP 500"),
        recorded(verdict=None, failure="the judge service answered HTTP 503"),
    ]
    with pytest.raises(AmbiguousJudgeRequest):
        asked(RecordedJudge(differ, model=MODEL, prompt_version=PROMPT))


def test_the_checks_own_endpoint_picks_between_records():
    here = recorded("supported", endpoint="http://a/judge")
    there = recorded("contradicted", endpoint="http://b/judge")
    replay = RecordedJudge([here, there], model=MODEL, prompt_version=PROMPT)
    assert asked(replay, endpoint="http://b/judge", endpoint_known=True) == there
    assert asked(replay, endpoint="http://a/judge", endpoint_known=True) == here
    with pytest.raises(AmbiguousJudgeRequest):
        asked(replay, endpoint=None, endpoint_known=False)
    with pytest.raises(AmbiguousJudgeRequest):
        asked(replay, endpoint="http://c/judge", endpoint_known=True)


@pytest.mark.parametrize(
    ("model", "prompt"), [("", PROMPT), (MODEL, ""), (None, PROMPT), (MODEL, 3)]
)
def test_a_replay_needs_both_identities_up_front(model, prompt):
    with pytest.raises(ValueError):
        RecordedJudge([recorded()], model=model, prompt_version=prompt)


def test_the_records_a_replay_holds_are_not_the_callers_to_change():
    request = judge().request(context())
    data = recorded(request_id=request["id"], request=request).to_json()
    replay = RecordedJudge([data], model=MODEL, prompt_version=PROMPT)
    data["request"]["claim"] = "changed"
    assert replay.ask(request).request["claim"] == CLAIM, "a replay keeps its own copy"
