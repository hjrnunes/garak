# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hold the reply judge in the grader to what the MiDojo runtime grades.

``tests/_assets/detectors/reply_judge/corpus.json`` holds, for each scripted
case, what the runtime wrote when it graded a sealed snapshot with a
``reply_judge`` check against the fake judge service, and what that service
received. Each case here runs through ``grade_sealed`` against the same fake
service, a real server on 127.0.0.1, and must give the same verdict document
and send the same requests. The comparison needs no runtime and calls no model.

The grader sends the request with the standard library's ``http.client``, where
the runtime uses ``httpx``, so it differs where the two libraries do: the name
of the exception a failure reports, a URL the standard library cannot form a
request to, and the user and password of a URL. ``DEVIATIONS`` names each case
that differs and states what the grader does instead. A case that stops
differing fails until it leaves the table.

The grader also stops where the runtime goes on. It decodes ``gzip`` and
``deflate`` answers only, and an answer in any other content coding is unknown,
whatever its bytes are, with a reason of its own and the status it came with:
never a verdict. ``httpx`` reads an answer in a coding it does not know as it
is (``answer.unknown-coding``, plain JSON that names ``x-unknown``, is a
verdict there), and decodes ``br`` and ``zstd`` when the package for each is
installed. Those two have no case, as the corpus is the same wherever it is
checked. A judge service that does not compress its answers, as the
pipeline's own (``judge-serve``) does not, never meets this difference. The
grader likewise reads no more than 8 MiB of an answer, where ``httpx`` reads
any length; that has no case either.
"""

import json
import socket
from pathlib import Path

import pytest

pytest.importorskip("midojo_spec")

from garak.detectors import grading  # noqa: E402
from garak.evidence.seal import rebuild_snapshot  # noqa: E402

ASSETS = Path(__file__).parents[1] / "_assets" / "detectors" / "reply_judge"
CORPUS = json.loads((ASSETS / "corpus.json").read_text(encoding="utf-8"))
CASES = CORPUS["cases"]
ENVIRONMENT_NAMES = ("JUDGE_URL", "JUDGE_TOKEN", "TEST_JUDGE_URL", "TEST_JUDGE_TOKEN")
HOST, DEAD = "{host}", "{dead}"
NOT_A_URL = "the judge service URL is not a valid URL"

# The runtime's error in the grader's terms: its refusal of the document and of
# the records are classes of the grader, and a missing identity is a TypeError.
REFUSALS = {
    "document:InjectionSpecDocumentError": grading.DocumentRefused,
    "grade:TypeError": TypeError,
    "grade:ValueError": grading.GradingRefused,
    "grade:ValidationError": grading.GradingRefused,
}


def swapped(value, pairs):
    text = json.dumps(value, ensure_ascii=False)
    for old, new in pairs:
        text = text.replace(old, new)
    return json.loads(text)


def stdlib_names(case):
    """A failure names the exception ``http.client`` and ``socket`` raise, not httpx's."""
    names = [
        ("(ReadTimeout)", "(TimeoutError)"),
        ("(ConnectError)", "(ConnectionRefusedError)"),
    ]
    return swapped(case["oracle"], names), case["received"]


def not_a_url(case):
    """A URL the standard library cannot form a request to is no request: nothing is sent."""
    failed = [
        (f"the judge service request failed ({name})", NOT_A_URL)
        for name in ("UnsupportedProtocol", "ConnectError")
    ]
    expected = swapped(case["oracle"], failed)
    for record in expected["verdict"]["grade"]["judgements"]:
        record["endpoint"], record["sent"] = None, False
    return expected, []


def no_userinfo(case):
    """The user and password of a URL are dropped, not sent as Basic credentials."""
    return case["oracle"], [
        {**seen, "authorization": None} for seen in case["received"]
    ]


def unread_coding(case):
    """An answer in a coding the grader does not decode is unknown, as an answer that is not JSON is."""
    twin = next(item for item in CASES if item["id"] == "failure.not-json")
    reason = (
        "the judge service's answer is in a content coding this check does not decode"
    )
    changed = [
        ("/not-json", "/unknown-coding"),
        ("the judge service's answer is not JSON", reason),
    ]
    return swapped(twin["oracle"], changed), case["received"]


DEVIATIONS = {
    "answer.unknown-coding": unread_coding,
    "failure.timeout": stdlib_names,
    "failure.refused-connection": stdlib_names,
    "token.unreachable": stdlib_names,
    "failure.no-scheme": not_a_url,
    "failure.no-host": not_a_url,
    "failure.bad-port": not_a_url,
    "environment.not-a-url": not_a_url,
    "endpoint.user-info": no_userinfo,
}


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def seen(received):
    """What the service saw of one request: its target, the headers that matter, and its body."""
    return {
        "path": received.path,
        "query": received.query,
        "authorization": received.headers.get("authorization"),
        "content_type": received.headers.get("content-type"),
        "accept": received.headers.get("accept"),
        "body": received.body,
    }


def run(case, service, monkeypatch):
    """What the grader makes of ``case`` against ``service``: its verdict document or its refusal, and what the service saw."""
    places = [(HOST, service.host_port), (DEAD, f"127.0.0.1:{free_port()}")]
    document = swapped(
        {
            **CORPUS["documents"]["base"],
            "security": case["security"],
            "utility": case["utility"],
        },
        places,
    )
    snapshot = rebuild_snapshot(CORPUS["snapshots"][case["snapshot"]])
    for name in ENVIRONMENT_NAMES:
        monkeypatch.delenv(name, raising=False)
    for name, value in swapped(case["env"], places).items():
        monkeypatch.setenv(name, value)
    kwargs = {}
    if case["replay"] is not None:
        replay = swapped(case["replay"], places)
        kwargs["judgements"] = replay["records"]
        for member, keyword in (
            ("model", "judge_model"),
            ("prompt_version", "judge_prompt_version"),
        ):
            if member in replay:
                kwargs[keyword] = replay[member]
    back = [(new, old) for old, new in places]
    try:
        verdict = grading.grade_sealed(document, snapshot, **kwargs).to_document()
    except (grading.GradingRefused, TypeError) as exc:
        outcome = {"refused": type(exc)}
    else:
        outcome = {"verdict": swapped(verdict, back)}
    return outcome, swapped([seen(item) for item in service.requests], back)


def test_the_corpus_is_the_format_this_test_reads():
    assert CORPUS["format"] == "reply-judge-oracle-v1"
    ids = [case["id"] for case in CASES]
    assert len(ids) == len(set(ids)), "case ids are unique"
    assert {case["group"] for case in CASES} == {
        "verdict",
        "request",
        "failure",
        "answer",
        "environment",
        "token",
        "endpoint",
        "compose",
        "outcome",
        "replay",
        "refused",
    }
    assert set(DEVIATIONS) <= set(ids), "a deviation names a case of the corpus"


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_a_case_is_graded_as_the_runtime_graded_it(case, fake_judge, monkeypatch):
    got, requests = run(case, fake_judge, monkeypatch)
    expected = case["oracle"]
    if "error" in expected:
        assert got == {"refused": REFUSALS[expected["error"]]}, "the same refusal"
        assert requests == [], "nothing was sent"
        return
    sent = case["received"]
    if case["id"] in DEVIATIONS:
        changed = DEVIATIONS[case["id"]](case)
        assert changed != (expected, sent), "the case differs from the runtime's"
        expected, sent = changed
    assert got == expected, "the verdict document, record for record"
    assert requests == sent, "the service received the same requests"
