# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test the replay probes end to end against a scripted OpenAI-compatible endpoint.

Each case runs the garak command line in this process, with the target a real
``OpenAIResponsesGenerator`` and the judge a real native ``RubricJudge``, and the
endpoint scripted by ``replay_endpoint.py`` (cases in ``cases.json``, both beside
the README in ``tests/_assets/probes/injection``). The report file is what is
checked. One case replays a reply recorded in a run of the earlier route, and its
attempt must equal the one that run wrote (see the README beside it).
"""

import json
import sys
from pathlib import Path

import httpx
import pytest
import respx
import yaml

from garak import cli

ASSETS = Path(__file__).parents[1] / "_assets" / "probes" / "injection"
sys.path.insert(0, str(ASSETS))
from replay_endpoint import Endpoint  # noqa: E402

CASES = json.loads((ASSETS / "cases.json").read_text(encoding="utf-8"))
URI = "http://192.0.2.10:9000/v1/"
API_KEY = "sentinel-endpoint-key-must-not-be-kept"


def run_case(tmp_path, case):
    """Run a case through the command line; return its finished attempts and endpoint."""
    source = tmp_path / "source.json"
    source.write_text(json.dumps(case["entries"]), encoding="utf-8")
    client = {
        "uri": URI,
        "api_key": API_KEY,
        "max_retries": 0,
        "client_options": {"timeout": 180.0},
    }
    module, name = case["probe"].split(".")
    config = {
        "system": {
            "lite": True,
            "parallel_requests": False,
            "parallel_attempts": False,
        },
        "run": {"generations": 1},
        "reporting": {"report_dir": str(tmp_path), "report_prefix": "run"},
        "plugins": {
            "generators": {
                "openai": {
                    "OpenAIResponsesGenerator": client
                    | {"max_tokens": 4096, "extra_params": {"tool_choice": "auto"}}
                }
            },
            "probes": {
                module: {
                    name: {"conversation_source": str(source)}
                    | case.get("probe_params", {})
                }
            },
            "detectors": {
                "rubric_judge": {
                    "RubricJudge": {
                        "detector_model_type": "openai.OpenAICompatible",
                        "detector_model_name": "test-model",
                        "detector_model_config": client,
                    }
                }
            },
        },
    }
    config_file = tmp_path / "run.yaml"
    config_file.write_text(yaml.safe_dump(config), encoding="utf-8")
    config_file.chmod(0o600)

    endpoint = Endpoint(case)

    def answer(request):
        body = json.loads(request.content) if request.content else None
        status, payload = endpoint.answer(request.method, request.url.path, body)
        return httpx.Response(status, json=payload)

    with respx.mock(assert_all_called=False) as router:
        router.route(url__startswith=URI).mock(side_effect=answer)
        cli.main(
            [
                "--config",
                str(config_file),
                "--target_type",
                "openai.OpenAIResponsesGenerator",
                "--target_name",
                "test-model",
                "--probes",
                case["probe"],
                "--detectors",
                ",".join(case["detectors"]),
            ]
        )
    rows = [
        json.loads(line)
        for line in (tmp_path / "run.report.jsonl").read_text("utf-8").splitlines()
    ]
    return [
        r for r in rows if r["entry_type"] == "attempt" and r["status"] == 2
    ], endpoint


def _recorded_case(record):
    """Return the case that makes a target answer as the recorded attempt shows."""
    prompt, output = record["prompt"], record["outputs"][0]
    entry = {
        "messages": [
            {"role": t["role"], "content": t["content"]["text"]}
            for t in prompt["turns"]
        ],
        "tools": prompt["notes"]["tools"],
        "judge_rubric": prompt["notes"]["judge_rubric"],
    }
    notes = output["notes"]
    usage = json.loads(json.dumps(notes["response"]["usage"]))
    # the endpoint sent no cache_write_tokens; the SDK reads the missing field as None
    usage["input_tokens_details"].pop("cache_write_tokens")
    judge = record["notes"]["rubric_judge_results"][0]
    return {
        "probe": record["probe_classname"],
        "detectors": list(record["detector_results"]),
        "entries": [entry],
        "replies": [
            {
                "text": output["text"],
                "items": notes["output_items"],
                "status": notes["response_status"],
                "response": notes["response"] | {"usage": usage},
            }
        ],
        "judge": judge["raw_response"],
    }


def _scores(attempts, detector="toolcall.ToolCallCondition"):
    return [a["detector_results"][detector] for a in attempts]


def test_a_recorded_reply_gives_the_attempt_the_earlier_route_wrote(tmp_path):
    pytest.importorskip("midojo_spec")
    recorded = json.loads((ASSETS / "scn-002.attempt.jsonl").read_text("utf-8"))

    attempts, endpoint = run_case(tmp_path, _recorded_case(recorded))

    [attempt] = attempts
    attempt.pop("uuid"), recorded.pop("uuid")
    assert attempt == recorded, "the attempt holds what the earlier route recorded"
    assert len(endpoint.requested("responses")) == 1, "one request to the target"
    assert len(endpoint.requested("completions")) == 1, "one request to the judge"


def test_a_single_turn_is_hit_or_missed_by_the_calls_the_target_made(tmp_path):
    attempts, _ = run_case(tmp_path, CASES["single-command"])

    assert _scores(attempts) == [
        [1.0],
        [0.0],
    ], "the refund call is a hit, the lookup not"
    assert [a["outputs"][0]["notes"]["tool_calls"][0]["name"] for a in attempts] == [
        "issue_refund",
        "get_orders",
    ], "each attempt holds the calls its target made"


def test_a_later_turn_resends_the_earlier_calls_and_replies(tmp_path):
    attempts, endpoint = run_case(tmp_path, CASES["sequential-command"])

    assert _scores(attempts) == [[0.0], [1.0]], "the condition holds on the last turn"
    assert [a["notes"]["sequence"]["call_offset"] for a in attempts] == [0, 1]
    second = endpoint.requested("responses")[1]["input"]
    assert [i.get("role") or i["type"] for i in second] == [
        "user",
        "mcp_call",
        "assistant",
        "user",
    ], "the second request holds the first turn, its call and reply, and the second turn"


def test_judge_turns_final_grades_the_last_turn_only(tmp_path):
    pytest.importorskip("midojo_spec")

    attempts, endpoint = run_case(tmp_path, CASES["sequential-reply"])

    verdicts = [a["notes"]["rubric_judge_results"][0] for a in attempts]
    assert [v["reason"] for v in verdicts] == ["judge_not_declared", "judge_completed"]
    assert _scores(attempts, "rubric_judge.RubricJudge") == [[None], [1.0]]
    assert len(endpoint.requested("completions")) == 1, "the judge was asked once"


def test_a_failed_single_turn_keeps_its_error_and_gets_no_verdict(tmp_path):
    attempts, endpoint = run_case(tmp_path, CASES["failed-single"])

    assert attempts[0]["prompt"]["notes"] == {}, "the grading notes are gone"
    assert _scores(attempts) == [[None]], "an outage is not a pass"
    [output] = attempts[0]["outputs"]
    assert output["text"] is None, "a rejected request leaves no reply"
    assert output["notes"]["error"] == {
        "type": "InternalServerError",
        "status_code": 503,
        "body": '{"error":{"message":"unavailable"}}',
    }
    assert len(endpoint.requested("responses")) == 1, "max_retries 0 sends one request"
    assert API_KEY not in json.dumps(attempts), "the report holds no credential"


def test_a_failed_sequential_turn_ends_the_conversation_ungraded(tmp_path):
    attempts, endpoint = run_case(tmp_path, CASES["failed-sequential"])

    [attempt] = attempts
    assert attempt["notes"]["sequence"]["failure"] == "InternalServerError"
    assert len(endpoint.requested("responses")) == 1, "no request after the failure"
    assert attempt["prompt"]["notes"] == {}, "the grading notes are gone"
    assert _scores(attempts) == [[None]], "and the detector gives no verdict"
