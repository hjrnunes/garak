# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise Garak evidence, backend identity and shared-rule delegation offline."""

import dataclasses
import hashlib
import json
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
