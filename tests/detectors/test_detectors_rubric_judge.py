# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise Garak evidence, backend identity and shared-rule delegation offline."""

import dataclasses
import hashlib
import json
import random
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest

pytest.importorskip("midojo_spec.rubric_judge")
from midojo_spec import judge as shared_judge
from midojo_spec import judge_exchange as exchange
from midojo_spec import rubric_judge as rules

from garak.attempt import Attempt, Conversation, Message, Turn
from garak.detectors import rubric_judge as detector
from garak.exception import GarakException
from garak.generators.openai import OpenAICompatible, OpenAIResponsesGenerator

MODEL = "test-judge"
RUBRIC = {"question": "Does the reply say yes?", "criteria": ["Read the reply."]}
YES = json.dumps({"verdict": "supported", "evidence_refs": ["messages[0].content"]})
NO = json.dumps({"verdict": "contradicted", "evidence_refs": ["messages[0].content"]})
UNKNOWN = json.dumps({"verdict": "unresolved", "evidence_refs": []})
VECTORS = Path(__file__).parents[1] / "_assets" / "detectors" / "rubric_judge_vectors"
MANIFEST = json.loads((VECTORS / "MANIFEST.json").read_text(encoding="utf-8"))
FENCES = json.loads((VECTORS / "fences.json").read_text(encoding="utf-8"))


def attempt(rubric=RUBRIC, outputs=None, turns=None):
    prompt = Conversation(
        turns=turns or [Turn("user", Message("Is the answer yes?"))],
        notes={} if rubric is None else {detector.RUBRIC_NOTE: rubric},
    )
    built = Attempt(prompt=prompt, probe_classname="probes.test.Blank")
    built.outputs = [Message("Yes.")] if outputs is None else outputs
    return built


def configured(backend="service", **params):
    values = {"backend": backend, "judge_model": MODEL, **params}
    return detector.RubricJudge(
        config_root={"detectors": {"rubric_judge": {"RubricJudge": values}}}
    )


def fake_chat(fake):
    class Resource:
        def create(self, *, extra_body=None):
            pass

    fake.client = SimpleNamespace(chat=SimpleNamespace(completions=Resource()))
    fake.generator = fake.client.chat.completions
    return fake


def test_evidence_preserves_history_and_tool_capture():
    built = attempt(
        turns=[
            Turn("system", Message("Keep receipts.")),
            Turn("user", Message("First question")),
            Turn("assistant", Message("First reply")),
            Turn("user", Message("Second question")),
        ],
        outputs=[
            Message(
                "Done",
                notes={
                    "tool_calls": [
                        {
                            "name": "read",
                            "arguments": '{"id": 1}',
                            "output": '{"ok": true}',
                        }
                    ],
                    "response_status": "in_progress",
                },
            )
        ],
    )
    evidence = detector.build_evidence(built, built.outputs[0])
    assert evidence["user_text"] == "Second question", "last user turn is the request"
    assert evidence["history"][0]["role"] == "system", "history retains non-user turns"
    assert evidence["tool_calls"][0]["result"] == {
        "ok": True
    }, "tool results are decoded"
    assert (
        evidence["completeness"]["messages"] == "partial"
    ), "unfinished output stays partial"


@pytest.mark.parametrize(
    "rubric,outputs,reason",
    [
        (None, [Message("Yes.")], "judge_not_declared"),
        ({"question": 3}, [Message("Yes.")], "judge_spec_invalid"),
        (RUBRIC, [None], "no_output"),
    ],
)
def test_preflight_records_do_not_ask_backend(rubric, outputs, reason):
    built = attempt(rubric, outputs)
    judge = configured(
        "service",
        service_url="http://127.0.0.1:1/",
        service_token_env="TEST_JUDGE_TOKEN",
    )
    assert judge.detect(built) == [None], "preflight cannot score"
    assert (
        built.notes[detector.RESULTS_NOTE][0]["reason"] == reason
    ), "reason identifies preflight"
    assert set(built.notes[detector.RESULTS_NOTE][0]) == {
        "verdict",
        "evidence_refs",
        "reason",
        "prompt_version",
        "request",
        "raw_response",
        "model",
        "prompt",
    }, "every record has the same keys"


def test_shared_judge_output_is_the_detector_decision():
    built = attempt()
    evidence = detector.build_evidence(built, built.outputs[0])
    expected = rules.judge_output(RUBRIC, evidence, lambda _: YES)
    got = detector._judge_reply(rules, RUBRIC, evidence, lambda _: YES, MODEL)
    assert all(
        got[key] == value for key, value in expected.items()
    ), "shared rules own judgement"


def test_native_sends_shared_request_body(monkeypatch):
    built = attempt()

    class FakeGenerator:
        name = MODEL
        temperature = None
        suppressed_params = set()
        extra_params = {}
        retry_json = True

        def generate(self, conversation):
            self.messages = [
                {"role": turn.role, "content": turn.content.text}
                for turn in conversation.turns
            ]
            return [Message(YES)]

    fake = fake_chat(FakeGenerator())
    monkeypatch.setattr(
        detector.RubricJudge,
        "_load_generator",
        lambda self: setattr(self, "evaluation_generator", fake),
    )
    judge = configured("native")
    assert judge.detect(built) == [1.0], "supported reply scores a hit"
    expected = rules.request_body(
        MODEL, rules.judge_messages(built.notes[detector.RESULTS_NOTE][0]["request"])
    )
    sent = {
        "model": fake.name,
        "messages": fake.messages,
        "temperature": fake.temperature,
        "max_completion_tokens": fake.extra_params["max_completion_tokens"],
        "response_format": fake.extra_params["response_format"],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    assert json.dumps(sent, sort_keys=True) == json.dumps(
        expected, sort_keys=True
    ), "native request matches shared body"
    assert fake.retry_json is False, "JSON retries are disabled"


@pytest.fixture
def fake_service(tmp_path, monkeypatch):
    path = tmp_path / "judge-record.jsonl"
    log = exchange.ExchangeLog(path)
    state = {"reply": YES, "model": MODEL, "version": rules.PROMPT_VERSION, "calls": 0}
    token = "sentinel-private-token"
    monkeypatch.setenv("TEST_JUDGE_TOKEN", token)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            state["calls"] += 1
            assert (
                self.headers.get("Authorization") == f"Bearer {token}"
            ), "service gets token only in header"
            sent = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if "replies" in state:
                state["reply"] = state["replies"][sent["reply"]]
            assert (
                shared_judge.request_problem(sent) is None
            ), "service request has valid shape"
            assert shared_judge.id_holds(sent), "request id hashes all fields"

            def calls(_budget, observer):
                def call(messages):
                    if observer:
                        observer(
                            messages,
                            exchange.Observed(
                                started_at=exchange.now_iso(),
                                duration_ms=1,
                                content=(
                                    state["reply"]
                                    if not isinstance(
                                        state["reply"], rules.JudgeCallError
                                    )
                                    else None
                                ),
                                error=(
                                    state["reply"].label
                                    if isinstance(state["reply"], rules.JudgeCallError)
                                    else None
                                ),
                            ),
                        )
                    if isinstance(state["reply"], rules.JudgeCallError):
                        raise state["reply"]
                    return state["reply"]

                return call

            result = service_judge(sent, calls, state["model"], log)
            result["details"]["prompt_version"] = state["version"]
            if state.get("alter"):
                state["alter"](result)
            body = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    yield f"http://127.0.0.1:{server.server_port}/", state, path, token
    server.shutdown()
    server.server_close()
    worker.join()


def service_judge(sent, calls, model, log):
    """Implement the service's final-turn projection using the shared contract."""
    from midojo_spec.judge_exchange import Exchange

    def observed_calls(budget, _observer):
        def run(messages):
            def write(prompt, observed):
                log.append(
                    Exchange.observed(
                        observed,
                        request_id=sent["id"],
                        ask=0,
                        model=model,
                        prompt_version=rules.PROMPT_VERSION,
                        messages=prompt,
                    )
                )

            return calls(budget, write)(messages)

        return run

    prior = sent["prior_turns"]
    history = [
        turn
        for pair in prior
        for turn in (
            {"role": "user", "content": pair["message"]},
            {"role": "assistant", "content": pair["reply"]},
        )
    ]
    evidence = rules.build_evidence(
        sent["message"],
        history,
        sent["reply"],
        sent["tool_calls"],
        complete=sent["tool_calls_complete"] and rules.is_complete(sent["tool_calls"]),
    )
    rubric = {
        "question": sent["claim"],
        **{
            k: sent["rubric"].get(k, v)
            for k, v in (
                ("criteria", []),
                ("facts", []),
                ("runtime_facts", {}),
            )
        },
    }
    record = rules.judge_output(rubric, evidence, observed_calls(180, None))
    summary = {
        **{
            k: record[k]
            for k in ("verdict", "evidence_refs", "reason", "prompt_version")
        },
        "request_sha256": exchange.sha256_text(rules.request_text(record["request"])),
        "response_sha256": (
            None
            if record["raw_response"] is None
            else exchange.sha256_text(record["raw_response"])
        ),
    }
    return {
        "schema": shared_judge.SCHEMA,
        "id": sent["id"],
        "verdict": record["verdict"],
        "rationale": record["reason"],
        "model": model,
        "details": {"prompt_version": rules.PROMPT_VERSION, "records": [summary]},
    }


@pytest.mark.parametrize(
    "reply,verdict,reason,refs",
    [
        (YES, "supported", "judge_completed", ["messages[0].content"]),
        (NO, "contradicted", "judge_completed", ["messages[0].content"]),
        (UNKNOWN, "unresolved", "judge_completed", []),
        (
            json.dumps({"verdict": "supported", "evidence_refs": ["facts[0]"]}),
            "unresolved",
            "judge_support_unresolved",
            [],
        ),
        (
            json.dumps(
                {"verdict": "supported", "evidence_refs": ["tool_calls[0].result"]}
            ),
            "supported",
            "judge_completed",
            ["tool_calls[0].result"],
        ),
        ("broken", "unresolved", "judge_parse_failed", []),
        (
            rules.JudgeCallError("APITimeoutError"),
            "unresolved",
            "judge_failed:APITimeoutError",
            [],
        ),
    ],
)
def test_three_backends_same_decision(
    fake_service, monkeypatch, reply, verdict, reason, refs
):
    url, state, path, _ = fake_service
    state["reply"] = reply
    output = Message(
        "Yes.",
        notes={
            "tool_calls": [
                {"name": "read", "arguments": "{}", "result": '{"ok":true}'},
            ]
        },
    )
    original = attempt(outputs=[output])
    native = configured_native(monkeypatch, reply)
    service = configured(
        "service", service_url=url, service_token_env="TEST_JUDGE_TOKEN"
    )
    first = native.detect(original)
    native_record = original.notes[detector.RESULTS_NOTE][0]
    remote = attempt(outputs=[output])
    second = service.detect(remote)
    replay = configured("replay", replay_path=str(path))
    saved = attempt(outputs=[output])
    third = replay.detect(saved)
    records = [
        native_record,
        remote.notes[detector.RESULTS_NOTE][0],
        saved.notes[detector.RESULTS_NOTE][0],
    ]
    for key in (
        "verdict",
        "evidence_refs",
        "reason",
        "model",
        "prompt_version",
        "request",
        "prompt",
    ):
        assert (
            records[0][key] == records[1][key] == records[2][key]
        ), f"{key} agrees across backends"
    assert [r["verdict"] for r in records] == [
        verdict
    ] * 3, "all report the expected verdict"
    assert [r["reason"] for r in records] == [
        reason
    ] * 3, "all report the expected reason"
    assert [r["evidence_refs"] for r in records] == [
        refs
    ] * 3, "all report the expected references"
    assert first == second == third, "all score the same evidence"


def configured_native(monkeypatch, reply):
    class FakeGenerator:
        name = MODEL
        suppressed_params = set()
        extra_params = {}

        def generate(self, _conversation):
            if isinstance(reply, rules.JudgeCallError):
                raise openai.APITimeoutError(request=None)
            return [Message(reply)]

    monkeypatch.setattr(
        detector.RubricJudge,
        "_load_generator",
        lambda self: setattr(self, "evaluation_generator", fake_chat(FakeGenerator())),
    )
    return configured("native")


def test_vendored_fence_vectors_match_manifest():
    assert {path.name for path in VECTORS.iterdir()} == {
        "MANIFEST.json",
        "fences.json",
    }, "only the declared oracle is vendored"
    assert hashlib.sha256((VECTORS / "fences.json").read_bytes()).hexdigest() == (
        MANIFEST["files"]["fences.json"]
    ), "the vendored vector bytes match their owner's digest"
    assert (
        len(FENCES["strips"]) == len(FENCES["vectors"]) == 34
    ), "all pinned fence cases are present"


@pytest.mark.parametrize("case", FENCES["strips"], ids=range(34))
def test_shared_fence_strip_matches_frozen_oracle(case):
    assert (
        rules.strip_code_fence(case["content"]) == case["stripped"]
    ), "the shared fence reader matches the frozen output"


def test_long_opening_fence_finishes_without_backtracking():
    program = (
        "from midojo_spec.rubric_judge import strip_code_fence; "
        "text = '```json' + ' ' * 5000 + '\\n{}'; "
        "assert strip_code_fence(text) == text"
    )
    done = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, timeout=8
    )
    assert done.returncode == 0, "a long fence finishes within a generous bound"


@pytest.mark.parametrize("case", FENCES["vectors"], ids=lambda item: item["id"])
def test_fence_vectors_agree_across_native_service_and_replay(
    fake_service, monkeypatch, case
):
    url, state, path, _ = fake_service
    model = case["record"]["model"]
    canned = case["canned"]
    state["model"] = model
    state["reply"] = (
        rules.JudgeCallError(canned["raise"])
        if "raise" in canned
        else canned["content"]
    )
    content = case["midojo"]
    calls = [
        {
            "name": call["function"],
            "arguments": call["args"],
            "result": call["result"],
        }
        for call in content["function_calls"]
    ]

    def make_attempt():
        return attempt(
            rubric=case["rubric"],
            turns=[Turn("user", Message(content["user_text"]))],
            outputs=[Message(content["agent_output"], notes={"tool_calls": calls})],
        )

    class VectorGenerator:
        name = model
        suppressed_params = set()
        extra_params = {}

        def generate(self, conversation):
            self.messages = [
                {"role": turn.role, "content": turn.content.text}
                for turn in conversation.turns
            ]
            if "raise" in canned:
                raise openai.APIConnectionError(
                    request=httpx.Request("POST", "http://127.0.0.1/")
                )
            return [Message(canned["content"])]

    monkeypatch.setattr(
        detector.RubricJudge,
        "_load_generator",
        lambda self: setattr(
            self, "evaluation_generator", fake_chat(VectorGenerator())
        ),
    )
    native = configured("native")
    original = make_attempt()
    native_scores = native.detect(original)
    native_record = original.notes[detector.RESULTS_NOTE][0]
    service = configured(
        "service",
        judge_model=model,
        service_url=url,
        service_token_env="TEST_JUDGE_TOKEN",
    )
    remote = make_attempt()
    service_scores = service.detect(remote)
    service_record = remote.notes[detector.RESULTS_NOTE][0]
    replay = configured("replay", judge_model=model, replay_path=str(path))
    saved = make_attempt()
    replay_scores = replay.detect(saved)
    replay_record = saved.notes[detector.RESULTS_NOTE][0]

    assert (
        native_scores == service_scores == replay_scores
    ), "all backends score the same evidence"
    assert (
        native_record == replay_record == case["record"]
    ), "native and replay match the frozen full record"
    # The judge service supplies only a digest: the exchange row, not its answer,
    # holds the original reply. Check the complete record against that row too.
    assert (
        service_record["raw_response"] is None
    ), "the service never returns model text"
    row = exchange.read_exchanges(path)[0]
    assert (
        row.reply == case["record"]["raw_response"]
    ), "the service records the original reply"
    assert {**service_record, "raw_response": row.reply} == case[
        "record"
    ], "the service summary and exchange together match the frozen record"
    assert rules.request_body(model, native_record["prompt"]) == {
        **{k: v for k, v in case["http_body"].items() if k != "extra_body"},
        **case["http_body"]["extra_body"],
    }, "the judge request matches the frozen request body"


def test_old_shared_rubric_module_fails_loudly(monkeypatch):
    monkeypatch.delattr(rules, "strip_code_fence")
    with pytest.raises(
        GarakException, match="midojo_spec.rubric_judge.strip_code_fence"
    ):
        configured(
            "service",
            service_url="http://127.0.0.1:1/",
            service_token_env="TEST_JUDGE_TOKEN",
        )


def test_multi_output_keeps_order_and_scores(fake_service):
    url, state, _, _ = fake_service
    judge = configured("service", service_url=url, service_token_env="TEST_JUDGE_TOKEN")
    built = attempt(outputs=[Message("Yes."), None, Message("No.")])
    assert judge.detect(built) == [1.0, None, 1.0], "each output has its own score"
    assert [r["reason"] for r in built.notes[detector.RESULTS_NOTE]] == [
        "judge_completed",
        "no_output",
        "judge_completed",
    ], "records keep output order"
    assert state["calls"] == 2, "only actual outputs ask the service"


def test_service_refuses_unexpressible_evidence_without_post(fake_service):
    url, state, _, _ = fake_service
    built = attempt(
        turns=[Turn("system", Message("Keep receipts.")), Turn("user", Message("Ask"))]
    )
    judge = configured("service", service_url=url, service_token_env="TEST_JUDGE_TOKEN")
    assert judge.detect(built) == [None], "non-paired history cannot be represented"
    assert (
        built.notes[detector.RESULTS_NOTE][0]["reason"]
        == "judge_request_not_expressible"
    ), "refusal is stable"
    assert state["calls"] == 0, "no evidence-changing request was sent"


@pytest.mark.parametrize(
    "changed,reason",
    [
        ({"model": None}, "judge_identity_missing"),
        ({"model": "other-judge"}, "judge_identity_model"),
        ({"version": "other-version"}, "judge_prompt_version_mismatch"),
    ],
)
def test_service_refuses_wrong_identity(fake_service, changed, reason):
    url, state, _, _ = fake_service
    state.update(changed)
    built = attempt()
    configured("service", service_url=url, service_token_env="TEST_JUDGE_TOKEN").detect(
        built
    )
    record = built.notes[detector.RESULTS_NOTE][0]
    assert record["reason"] == reason, "service identity is mandatory"
    assert record["verdict"] == "unresolved", "unverified service cannot decide"


@pytest.mark.parametrize(
    "model,version,reason",
    [
        (None, rules.PROMPT_VERSION, "judge-replay-identity"),
        ("other", rules.PROMPT_VERSION, "judge-replay-identity"),
        (MODEL, "other", "judge-replay-identity"),
    ],
)
def test_replay_refuses_sole_wrong_identity(fake_service, model, version, reason):
    url, _, path, _ = fake_service
    built = attempt()
    configured("service", service_url=url, service_token_env="TEST_JUDGE_TOKEN").detect(
        built
    )
    rows = exchange.read_exchanges(path)
    modified = dataclasses.replace(rows[0], model=model, prompt_version=version)
    alternate = path.parent / "other.jsonl"
    exchange.ExchangeLog(alternate).append(modified)
    saved = attempt()
    assert configured("replay", replay_path=str(alternate)).detect(saved) == [
        None
    ], "wrong identity never scores"
    assert (
        saved.notes[detector.RESULTS_NOTE][0]["reason"] == reason
    ), "shared replay refuses sole candidate"


def test_replay_refuses_ambiguity_and_absent_request(fake_service):
    url, _, path, _ = fake_service
    configured("service", service_url=url, service_token_env="TEST_JUDGE_TOKEN").detect(
        attempt()
    )
    row = exchange.read_exchanges(path)[0]
    other = exchange.ExchangeLog(path)
    other.append(
        dataclasses.replace(
            row,
            reply=NO,
            response_sha256=exchange.sha256_text(NO),
        )
    )
    built = attempt()
    configured("replay", replay_path=str(path)).detect(built)
    assert (
        built.notes[detector.RESULTS_NOTE][0]["reason"] == "judge-replay-ambiguous"
    ), "conflicts refuse replay"
    different = attempt(outputs=[Message("Different.")])
    configured("replay", replay_path=str(path)).detect(different)
    assert (
        different.notes[detector.RESULTS_NOTE][0]["reason"] == "judge-replay-unrecorded"
    ), "absent key refuses replay"


def test_service_token_never_enters_records_or_errors(fake_service):
    url, _, _, secret = fake_service
    judge = configured("service", service_url=url, service_token_env="TEST_JUDGE_TOKEN")
    built = attempt()
    judge.detect(built)
    assert secret not in repr(judge), "repr contains no token"
    assert secret not in repr(built.notes), "notes contain no token"
    with pytest.raises(GarakException) as failure:
        configured(
            "service",
            service_url=f"http://{secret}@127.0.0.1/",
            service_token_env="TEST_JUDGE_TOKEN",
        )
    assert secret not in str(failure.value), "URL credential is never repeated"


@pytest.mark.parametrize(
    "alter,reason",
    [
        (lambda answer: answer.update(id="wrong"), "judge_response_invalid"),
        (lambda answer: answer.pop("id"), "judge_response_id_mismatch"),
        (
            lambda answer: answer["details"]["records"][0].update(
                request_sha256="0" * 64
            ),
            "judge_request_digest_mismatch",
        ),
        (
            lambda answer: answer["details"]["records"][0].update(
                evidence_refs=["tool_calls[9].result"]
            ),
            "judge_response_records_invalid",
        ),
    ],
)
def test_service_rejects_invalid_answer_or_request_digest(fake_service, alter, reason):
    url, state, _, _ = fake_service
    state["alter"] = alter
    built = attempt()
    assert configured(
        "service", service_url=url, service_token_env="TEST_JUDGE_TOKEN"
    ).detect(built) == [None], "unverified answer cannot score"
    assert (
        built.notes[detector.RESULTS_NOTE][0]["reason"] == reason
    ), "refusal names the violated rule"


def test_service_answer_nested_past_the_parser_is_unresolved(monkeypatch):
    body = b"[" * (1 << 20)  # the longest answer the check reads

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    monkeypatch.setenv("TEST_JUDGE_TOKEN", "sentinel-private-token")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        judge = configured(
            "service",
            service_url=f"http://127.0.0.1:{server.server_port}/",
            service_token_env="TEST_JUDGE_TOKEN",
        )
        built = attempt()
        assert judge.detect(built) == [None], "a nested answer cannot score"
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
    assert built.notes[detector.RESULTS_NOTE][0]["reason"].startswith(
        "judge_failed:"
    ), "an answer the parser cannot read is a failed call, not an exception"


def test_service_drops_hostile_reason_and_secret_reference(fake_service):
    url, state, _, token = fake_service

    def alter(answer):
        last = answer["details"]["records"][0]
        last["reason"] = answer["rationale"] = token
        last["evidence_refs"] = [token]

    state["alter"] = alter
    built = attempt()
    configured("service", service_url=url, service_token_env="TEST_JUDGE_TOKEN").detect(
        built
    )
    assert token not in repr(
        built.notes
    ), "a service cannot echo credentials into notes"
    assert (
        built.notes[detector.RESULTS_NOTE][0]["reason"]
        == "judge_response_records_invalid"
    ), "hostile reason is refused"


@pytest.mark.parametrize(
    "output,turns",
    [
        (Message("Yes.", notes={"tool_calls": "not a list"}), None),
        (Message("Yes.", notes={"tool_calls": [{"arguments": "{}", "name": 1}]}), None),
        (
            Message(
                "Yes.", notes={"tool_calls": [{"name": "read", "result": float("nan")}]}
            ),
            None,
        ),
        (
            Message("Yes.", notes={"tool_calls": [{"name": "read", "blocked": True}]}),
            None,
        ),
        (
            Message("Yes."),
            [Turn("assistant", Message("First")), Turn("user", Message("Last"))],
        ),
    ],
)
def test_service_never_sends_unexpressible_evidence(fake_service, output, turns):
    url, state, _, _ = fake_service
    built = attempt(outputs=[output], turns=turns)
    assert configured(
        "service", service_url=url, service_token_env="TEST_JUDGE_TOKEN"
    ).detect(built) == [None], "unexpressible evidence refuses"
    assert state["calls"] == 0, "service never sees a different evidence view"


def test_replay_is_offline_and_needs_no_provider_credentials(fake_service, monkeypatch):
    url, _, path, _ = fake_service
    configured("service", service_url=url, service_token_env="TEST_JUDGE_TOKEN").detect(
        attempt()
    )
    monkeypatch.delenv("TEST_JUDGE_TOKEN")
    monkeypatch.delenv("OPENAICOMPATIBLE_API_KEY", raising=False)
    monkeypatch.setattr(
        detector.RubricJudge,
        "_load_generator",
        lambda self: pytest.fail("replay must not load a generator"),
    )
    monkeypatch.setattr(
        detector.RubricJudge,
        "_post",
        lambda self, sent: pytest.fail("replay must not post"),
    )
    saved = attempt()
    replay = configured("replay", replay_path=str(path))
    assert replay.detect(saved) == [
        1.0
    ], "replay uses the recorded reply without credentials"


@pytest.mark.parametrize("redacted", [True, False])
def test_replay_refuses_unverifiable_prompt(fake_service, redacted):
    url, _, path, _ = fake_service
    configured("service", service_url=url, service_token_env="TEST_JUDGE_TOKEN").detect(
        attempt()
    )
    row = exchange.read_exchanges(path)[0]
    alternative = path.parent / "unverifiable.jsonl"
    change = (
        {"redacted": True}
        if redacted
        else {"messages": (("system", "wrong"), row.messages[-1])}
    )
    exchange.ExchangeLog(alternative).append(dataclasses.replace(row, **change))
    built = attempt()
    replay = configured("replay", replay_path=str(alternative))
    assert replay.detect(built) == [None], "unverifiable prompt cannot score"
    assert built.notes[detector.RESULTS_NOTE][0]["reason"] == (
        "judge-replay-request-mismatch"
    ), "prompt mismatch has a stable refusal"


def test_unscorable_attempts_have_the_same_record_on_every_backend(
    fake_service, monkeypatch
):
    _, state, path, _ = fake_service
    path.touch()
    native = configured_native(monkeypatch, YES)
    service = configured(
        "service",
        service_url="http://127.0.0.1:1/",
        service_token_env="TEST_JUDGE_TOKEN",
    )
    replay = configured("replay", replay_path=str(path))
    for rubric, outputs in (
        (None, [Message("Yes.")]),
        ({"question": False}, [Message("Yes.")]),
        (RUBRIC, [None]),
    ):
        records = []
        for backend in (native, service, replay):
            built = attempt(rubric=rubric, outputs=outputs)
            assert backend.detect(built) == [None], "preflight cannot score"
            records.append(built.notes[detector.RESULTS_NOTE][0])
        assert records[0] == records[1] == records[2], "preflight records agree"
    assert state["calls"] == 0, "no preflight case asks the service"


@pytest.mark.parametrize(
    "module,backend",
    [
        ("rubric_judge", "native"),
        ("judge", "service"),
        ("judge_replay", "replay"),
    ],
)
def test_missing_shared_module_names_the_module(
    module, backend, fake_service, monkeypatch
):
    _, _, path, _ = fake_service
    imported = detector.importlib.import_module

    def unavailable(name):
        if name == f"midojo_spec.{module}":
            raise ModuleNotFoundError(name)
        return imported(name)

    monkeypatch.setattr(detector.importlib, "import_module", unavailable)
    if backend == "native":
        monkeypatch.setattr(detector.RubricJudge, "_load_generator", lambda self: None)
    with pytest.raises(GarakException, match=f"midojo_spec.{module}"):
        configured(
            backend,
            service_url="http://127.0.0.1:1/",
            service_token_env="TEST_JUDGE_TOKEN",
            replay_path=str(path),
        )


def sdk_generator(monkeypatch, create, *, generator_type=OpenAICompatible):
    """Load an offline chat resource with the real Garak create path."""

    class Resource:
        def create(
            self,
            *,
            model,
            messages,
            n=None,
            temperature=None,
            seed=None,
            top_p=None,
            logprobs=None,
            max_completion_tokens=None,
            response_format=None,
            extra_body=None,
        ):
            return create(
                {k: v for k, v in locals().items() if k not in ("self", "create")}
            )

    fake = generator_type.__new__(generator_type)
    fake.name = MODEL
    fake.client = SimpleNamespace(
        chat=SimpleNamespace(completions=Resource()),
        completions=object(),
        close=lambda: None,
    )
    fake.generator = fake.client.chat.completions
    fake.suppressed_params = set()
    fake.extra_params = {"logprobs": True, "top_p": 0.9}
    fake.seed = 123
    fake._rng = random.Random()
    fake.supports_multiple_generations = True
    fake.temperature = 0.8
    fake.top_p = 0.9
    fake.logprobs = True
    fake.retry_json = True
    fake.transient_retry_codes = [429, 503]
    monkeypatch.setattr(
        detector.RubricJudge,
        "_load_generator",
        lambda self: setattr(self, "evaluation_generator", fake),
    )
    return configured("native")


def test_native_timeout_is_one_create_without_generator_backoff(monkeypatch):
    calls = []

    def create(kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise openai.APITimeoutError(request=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=YES))]
        )

    judge = sdk_generator(monkeypatch, create)
    built = attempt()
    assert judge.detect(built) == [None], "timeout never retries a judge request"
    assert len(calls) == 1, "one create is the maximum even on timeout"
    assert built.notes[detector.RESULTS_NOTE][0]["reason"] == (
        "judge_failed:APITimeoutError"
    ), "the timeout remains unresolved"


def test_native_success_uses_one_create_and_exact_shared_body(monkeypatch):
    calls = []

    def create(kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=YES))]
        )

    judge = sdk_generator(monkeypatch, create)
    built = attempt()
    assert judge.detect(built) == [1.0], "one successful create scores"
    assert len(calls) == 1, "a successful judge call is sent once"
    sent = {k: v for k, v in calls[0].items() if v is not None}
    body = rules.request_body(MODEL, built.notes[detector.RESULTS_NOTE][0]["prompt"])
    assert sent == {
        **{k: v for k, v in body.items() if k != "chat_template_kwargs"},
        "extra_body": {"chat_template_kwargs": body["chat_template_kwargs"]},
    }, "configured seed, sampling, and extra parameters cannot alter the request"


@pytest.mark.parametrize("question", ["", "  \t "])
def test_blank_question_refuses_all_backends_before_call(
    fake_service, monkeypatch, question
):
    url, state, path, _ = fake_service
    path.touch()
    judges = (
        configured_native(monkeypatch, YES),
        configured("service", service_url=url, service_token_env="TEST_JUDGE_TOKEN"),
        configured("replay", replay_path=str(path)),
    )
    records = []
    for judge in judges:
        built = attempt(rubric={"question": question})
        assert judge.detect(built) == [None], "blank question never scores"
        records.append(built.notes[detector.RESULTS_NOTE][0])
    assert records[0] == records[1] == records[2], "preflight agrees exactly"
    assert records[0]["reason"] == "judge_spec_invalid", "blank claim is invalid"
    assert state["calls"] == 0, "no invalid request reaches the service"


@pytest.mark.parametrize(
    "raw", ['{"verdict":' + "9" * 5000 + "}", "[" * 10000 + "0" + "]" * 10000]
)
def test_hostile_json_parse_is_unresolved_with_original_reply(
    fake_service, monkeypatch, raw
):
    url, _, path, _ = fake_service
    configured("service", service_url=url, service_token_env="TEST_JUDGE_TOKEN").detect(
        attempt()
    )
    original = exchange.read_exchanges(path)[0]
    alternate = path.parent / "hostile.jsonl"
    exchange.ExchangeLog(alternate).append(
        dataclasses.replace(
            original, reply=raw, response_sha256=exchange.sha256_text(raw)
        )
    )
    native = configured_native(monkeypatch, raw)
    replay = configured("replay", replay_path=str(alternate))
    for judge in (native, replay):
        built = attempt()
        assert judge.detect(built) == [None], "invalid JSON cannot score"
        record = built.notes[detector.RESULTS_NOTE][0]
        assert record["reason"] == "judge_parse_failed", "parse error is classified"
        assert record["raw_response"] == raw, "original response is preserved"
        assert record["model"] == MODEL and record["prompt"], "identity is retained"


@pytest.mark.parametrize("replacement", [None, "replacement-token"])
def test_service_checks_token_sent_even_after_environment_changes(
    fake_service, monkeypatch, replacement
):
    url, state, _, token = fake_service
    judge = configured("service", service_url=url, service_token_env="TEST_JUDGE_TOKEN")
    post = judge._post

    def rotate(sent):
        result = post(sent)
        if replacement is None:
            monkeypatch.delenv("TEST_JUDGE_TOKEN")
        else:
            monkeypatch.setenv("TEST_JUDGE_TOKEN", replacement)
        return result

    monkeypatch.setattr(judge, "_post", rotate)
    built = attempt()
    assert judge.detect(built) == [1.0], "token rotation cannot invalidate a valid ref"
    assert token not in repr(built.notes), "the sent token stays private"
    assert state["calls"] == 1, "the service saw the request"


def test_service_refuses_reference_containing_sent_token_after_rotation(
    fake_service, monkeypatch
):
    url, state, _, token = fake_service
    state["alter"] = lambda answer: answer["details"]["records"][0].update(
        evidence_refs=[f"messages[0].content:{token}"]
    )
    judge = configured("service", service_url=url, service_token_env="TEST_JUDGE_TOKEN")
    post = judge._post

    def rotate(sent):
        result = post(sent)
        monkeypatch.setenv("TEST_JUDGE_TOKEN", "different")
        return result

    monkeypatch.setattr(judge, "_post", rotate)
    built = attempt()
    assert judge.detect(built) == [None], "token in evidence ref is refused"
    assert token not in repr(built.notes), "refused token never enters notes"


def test_native_refuses_responses_resource_and_allows_chat_subclass(monkeypatch):
    class ChatSubclass(OpenAICompatible):
        pass

    chat = sdk_generator(
        monkeypatch,
        lambda _: SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=YES))]
        ),
        generator_type=ChatSubclass,
    )
    assert chat.detect(attempt()) == [1.0], "chat subclass is supported"
    responses = OpenAIResponsesGenerator.__new__(OpenAIResponsesGenerator)
    responses.name = MODEL
    responses.suppressed_params = set()
    responses.extra_params = {}
    responses.client = SimpleNamespace(
        chat=SimpleNamespace(completions=object()),
        responses=SimpleNamespace(create=lambda **kwargs: None),
        close=lambda: None,
    )
    responses.generator = responses.client.responses
    monkeypatch.setattr(
        detector.RubricJudge,
        "_load_generator",
        lambda self: setattr(self, "evaluation_generator", responses),
    )
    with pytest.raises(GarakException, match="chat.completions"):
        configured("native")


def test_multi_output_records_match_across_all_backends(fake_service, monkeypatch):
    url, state, path, _ = fake_service
    state["replies"] = {"Yes.": YES, "No.": NO, "Hostile.": "not json"}
    outputs = [Message("Yes."), None, Message("No."), Message("Hostile.")]

    native = configured_native(monkeypatch, YES)
    service = configured(
        "service", service_url=url, service_token_env="TEST_JUDGE_TOKEN"
    )
    current = attempt(outputs=outputs)
    remaining = iter(output.text for output in outputs if output is not None)

    def routed(_messages):
        return state["replies"][next(remaining)]

    monkeypatch.setattr(native, "_native_call", routed)
    native_scores = native.detect(current)
    remote = attempt(outputs=outputs)
    service_scores = service.detect(remote)
    replay = configured("replay", replay_path=str(path))
    saved = attempt(outputs=outputs)
    replay_scores = replay.detect(saved)
    assert (
        native_scores == service_scores == replay_scores == [1.0, None, 0.0, None]
    ), "scores remain in output order"
    native_records = current.notes[detector.RESULTS_NOTE]
    service_records = remote.notes[detector.RESULTS_NOTE]
    replay_records = saved.notes[detector.RESULTS_NOTE]
    rows = exchange.read_exchanges(path)
    assert len(rows) == 3, "one exchange is recorded per judged output"
    for i, (n, s, r) in enumerate(zip(native_records, service_records, replay_records)):
        assert n == r, f"native and replay full record {i} agree"
        if outputs[i] is None:
            assert n == s, "empty output has identical preflight records"
        else:
            assert {
                **s,
                "raw_response": rows.pop(0).reply,
            } == n, f"service summary and exchange reproduce full record {i}"
    assert [r["reason"] for r in native_records] == [
        "judge_completed",
        "no_output",
        "judge_completed",
        "judge_parse_failed",
    ], "notes preserve distinct verdicts and hostile output"


@pytest.mark.parametrize(
    "text",
    ['{"verdict":' + "9" * 5000 + "}", "[" * 10000 + "0" + "]" * 10000],
)
def test_hostile_bytes_and_text_match_on_native_and_replay(
    fake_service, monkeypatch, text
):
    url, _, path, _ = fake_service
    configured("service", service_url=url, service_token_env="TEST_JUDGE_TOKEN").detect(
        attempt()
    )
    row = exchange.read_exchanges(path)[0]
    records = []
    for reply in (text, text.encode("utf-8")):
        native = configured_native(monkeypatch, reply)
        saved = attempt()
        assert native.detect(saved) == [None], "hostile reply cannot score"
        records.append(saved.notes[detector.RESULTS_NOTE][0])
        monkeypatch.setattr(
            detector._shared("judge_replay"),
            "replay",
            lambda *_args, **_kwargs: dataclasses.replace(row, reply=reply),
        )
        replay = configured("replay", replay_path=str(path))
        saved = attempt()
        assert replay.detect(saved) == [None], "replayed hostile reply cannot score"
        records.append(saved.notes[detector.RESULTS_NOTE][0])
    assert all(
        record == records[0] for record in records
    ), "both types and backends agree"
    assert (
        records[0]["reason"] == "judge_parse_failed"
    ), "hostile JSON is a parse failure"
    assert records[0]["raw_response"] == text, "shared path retains decoded text"


@pytest.mark.parametrize(
    "reply,expected",
    [(YES.encode("utf-8"), YES), (b"\xff", None)],
)
def test_native_and_replay_share_utf8_reply_handling(
    fake_service, monkeypatch, reply, expected
):
    url, _, path, _ = fake_service
    configured("service", service_url=url, service_token_env="TEST_JUDGE_TOKEN").detect(
        attempt()
    )
    row = exchange.read_exchanges(path)[0]
    native = configured_native(monkeypatch, reply)
    monkeypatch.setattr(
        detector._shared("judge_replay"),
        "replay",
        lambda *_args, **_kwargs: dataclasses.replace(row, reply=reply),
    )
    replay = configured("replay", replay_path=str(path))
    records = []
    for backend in (native, replay):
        built = attempt()
        backend.detect(built)
        records.append(built.notes[detector.RESULTS_NOTE][0])
    assert records[0] == records[1], "both backends decode or refuse identically"
    assert records[0]["raw_response"] == expected, "shared raw reply is preserved"
    assert records[0]["reason"] == (
        "judge_completed" if expected is not None else "judge_parse_failed"
    ), "invalid UTF-8 is refused without raw content"


def test_native_respects_chat_subclass_refusal_without_create(monkeypatch):
    class RefusingChat(OpenAICompatible):
        calls = 0

        def _call_model(self, prompt, generations_this_call=1):
            self.calls += 1
            return [None]

    created = []
    judge = sdk_generator(
        monkeypatch, lambda kwargs: created.append(kwargs), generator_type=RefusingChat
    )
    built = attempt()
    assert judge.detect(built) == [None], "subclass refusal cannot score"
    assert judge.evaluation_generator.calls == 1, "the subclass sees the request"
    assert created == [], "refused requests never reach create"
    assert built.notes[detector.RESULTS_NOTE][0]["reason"] == (
        "judge_parse_failed"
    ), "the refusal has a stable unresolved reason"


def test_native_dispatches_chat_subclass_with_shared_request_body(monkeypatch):
    class ForwardingChat(OpenAICompatible):
        calls = 0

        def _call_model(self, prompt, generations_this_call=1):
            self.calls += 1
            return super()._call_model(prompt, generations_this_call)

    created = []

    def create(kwargs):
        created.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=YES))]
        )

    judge = sdk_generator(monkeypatch, create, generator_type=ForwardingChat)
    built = attempt()
    assert judge.detect(built) == [1.0], "forwarding subclass scores"
    assert (
        judge.evaluation_generator.calls == len(created) == 1
    ), "subclass and resource each run once"
    body = rules.request_body(MODEL, built.notes[detector.RESULTS_NOTE][0]["prompt"])
    assert {k: v for k, v in created[0].items() if v is not None} == {
        **{k: v for k, v in body.items() if k != "chat_template_kwargs"},
        "extra_body": {"chat_template_kwargs": body["chat_template_kwargs"]},
    }, "the subclass sends only the shared request body"


def test_native_subclass_retry_cannot_create_twice(monkeypatch):
    class ForwardingChat(OpenAICompatible):
        calls = 0

        def _call_model(self, prompt, generations_this_call=1):
            self.calls += 1
            return super()._call_model(prompt, generations_this_call)

    created = []

    def create(kwargs):
        created.append(kwargs)
        if len(created) == 1:
            raise openai.APITimeoutError(request=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=YES))]
        )

    delays = []
    monkeypatch.setattr("backoff._sync.time.sleep", delays.append)
    judge = sdk_generator(monkeypatch, create, generator_type=ForwardingChat)
    built = attempt()
    assert judge.detect(built) == [None], "a transient failure cannot score a retry"
    assert len(created) == 1, "a retry cannot send a second request"
    assert judge.evaluation_generator.calls == 1, "the subclass handles the first call"
    assert built.notes[detector.RESULTS_NOTE][0]["reason"] == (
        "judge_failed:APITimeoutError"
    ), "the first failure keeps its error label"
    assert all(
        0 <= delay <= 1 for delay in delays
    ), "the first Fibonacci backoff has at most one second of jitter"


def test_shared_native_generator_serialises_guards_across_detectors(monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    second_started = threading.Event()
    second_done = threading.Event()
    created = []
    outcomes = {}

    def create(kwargs):
        created.append(kwargs)
        if len(created) == 1:
            entered.set()
            if not release.wait(timeout=8):
                raise RuntimeError("first request was not released")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=YES))]
        )

    first = sdk_generator(monkeypatch, create)
    second = configured("native")
    assert (
        first.evaluation_generator is second.evaluation_generator
    ), "both detectors use the same generator and resource"
    resource = first.evaluation_generator.generator
    original_create = resource.create

    def run(judge, number):
        if number == 1:
            second_started.set()
        try:
            built = attempt()
            outcomes[number] = (
                judge.detect(built),
                built.notes[detector.RESULTS_NOTE][0]["reason"],
            )
        except (GarakException, RuntimeError, AttributeError) as exc:
            outcomes[number] = type(exc).__name__
        finally:
            if number == 1:
                second_done.set()

    workers = (
        threading.Thread(target=run, args=(first, 0), daemon=True),
        threading.Thread(target=run, args=(second, 1), daemon=True),
    )
    workers[0].start()
    try:
        assert entered.wait(timeout=5), "first call reaches create"
        workers[1].start()
        assert second_started.wait(timeout=5), "second call starts while first waits"
        second_done.wait(timeout=2)
    finally:
        release.set()
        for worker in workers:
            if worker.ident is not None:
                worker.join(timeout=5)
    assert not any(worker.is_alive() for worker in workers), "both calls finish"
    assert outcomes == {
        0: ([1.0], "judge_completed"),
        1: ([1.0], "judge_completed"),
    }, "each detector judges its own successful reply"
    assert len(created) == 2, "each judgement makes exactly one create"
    assert resource.create == original_create, "the original resource is restored"


def test_native_reloads_closed_generator_with_sdk_retries_disabled(monkeypatch):
    class ReloadableChat(OpenAICompatible):
        reload_count = 0

        def _load_unsafe(self):
            self.reload_count += 1
            resource = self.saved_resource
            chat = SimpleNamespace(completions=resource)

            def with_options(*, max_retries):
                return SimpleNamespace(
                    chat=chat,
                    completions=object(),
                    max_retries=max_retries,
                    close=lambda: None,
                )

            self.client = SimpleNamespace(
                chat=chat,
                completions=object(),
                with_options=with_options,
                close=lambda: None,
            )
            self.generator = resource

    created = []

    def create(kwargs):
        created.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=YES))]
        )

    judge = sdk_generator(monkeypatch, create, generator_type=ReloadableChat)
    generator = judge.evaluation_generator
    generator.saved_resource = generator.generator
    generator.close()
    built = attempt()
    assert judge.detect(built) == [1.0], "closed generator reloads and scores"
    assert generator.reload_count == 1, "the generator owns its client reload"
    assert generator.client.max_retries == 0, "reloaded SDK client disables retries"
    assert len(created) == 1, "the reloaded resource is called once"


@pytest.mark.parametrize("error_type", [ValueError, openai.OpenAIError])
def test_native_reload_failure_records_unresolved_without_create(
    monkeypatch, error_type
):
    class FailingReload(OpenAICompatible):
        def _load_unsafe(self):
            raise error_type("reload unavailable")

    created = []
    judge = sdk_generator(
        monkeypatch, lambda kwargs: created.append(kwargs), generator_type=FailingReload
    )
    judge.evaluation_generator.close()
    built = attempt()
    assert judge.detect(built) == [None], "reload failure is unresolved"
    assert built.notes[detector.RESULTS_NOTE][0]["reason"] == (
        f"judge_failed:{error_type.__name__}"
    ), "failure names only its exception type"
    assert created == [], "failed reload never asks the resource"


@pytest.mark.parametrize("same_detector", [False, True], ids=["shared", "self"])
def test_reentrant_native_judgement_refuses_only_inner(monkeypatch, same_detector):
    class ReentrantChat(OpenAICompatible):
        entering = False

        def generate(self, conversation, generations_this_call=1, typecheck=True):
            if not self.entering:
                self.entering = True
                try:
                    built = attempt()
                    self.inner_scores = self.nested.detect(built)
                    self.inner_record = built.notes[detector.RESULTS_NOTE][0]
                finally:
                    self.entering = False
            return super().generate(conversation, generations_this_call, typecheck)

    created = []

    def create(kwargs):
        created.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=YES))]
        )

    outer = sdk_generator(monkeypatch, create, generator_type=ReentrantChat)
    generator = outer.evaluation_generator
    generator.nested = outer if same_detector else configured("native")
    resource = generator.generator
    original_create = resource.create
    for count in (1, 2):
        built = attempt()
        assert outer.detect(built) == [1.0], "outer judgement scores its one reply"
        assert built.notes[detector.RESULTS_NOTE][0]["reason"] == (
            "judge_completed"
        ), "nested refusal cannot change the outer verdict"
        assert generator.inner_scores == [None], "inner judgement stays unresolved"
        assert generator.inner_record["reason"] == (
            "judge_failed:ReentrantJudgeCall"
        ), "inner refusal has a named and bounded reason"
        assert len(created) == count, "only the outer judgement calls create"
        assert resource.create == original_create, "the original create is restored"


@pytest.mark.parametrize("interrupt", [False, True], ids=["error", "interrupt"])
def test_native_reentry_flag_clears_after_guard_failure(monkeypatch, interrupt):
    class PrivateInterrupt(KeyboardInterrupt):
        pass

    created = []

    def create(kwargs):
        created.append(kwargs)
        if len(created) == 1:
            if interrupt:
                raise PrivateInterrupt("offline interruption")
            raise GarakException("offline failure")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=YES))]
        )

    judge = sdk_generator(monkeypatch, create)
    resource = judge.evaluation_generator.generator
    original_create = resource.create
    if interrupt:
        with pytest.raises(PrivateInterrupt):
            judge.detect(attempt())
    else:
        failed = attempt()
        assert judge.detect(failed) == [None], "generator failure is unresolved"
        assert failed.notes[detector.RESULTS_NOTE][0]["reason"] == (
            "judge_failed:GarakException"
        ), "ordinary failure keeps its existing reason"
    assert resource.create == original_create, "failed call restores create"
    built = attempt()
    assert judge.detect(built) == [1.0], "next call on the same thread can score"
    assert len(created) == 2, "each judgement reaches create at most once"
    assert resource.create == original_create, "successful call restores create"
