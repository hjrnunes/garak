# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test the snapshot and verdict values a trace states.

The runtime's own sealed documents are the oracle: the seven shared
inconclusive vectors, the consumer's injected-agent fixtures
(``tests/_assets/evidence/consumer_ipi``) and the snapshots in the grading
corpus. Each file holds the value the MiDojo runtime served or wrote, so the
comparison needs no runtime. Two of the thirteen sealed documents hold a reply
judge check. The vector's verdict is the one graded with no judge service to
ask. The consumer fixture's is the one the runtime graded again from the
recorded judgement of a scripted service, with its endpoint scrubbed. That
record names no prompt version, so no judge identity replays it: a scripted
service gives the same answer, and the verdicts match but for that endpoint.
"""

import contextlib
import copy
import hashlib
import json
import threading
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

pytest.importorskip("midojo_spec")

from garak.detectors import grading  # noqa: E402
from garak.evidence import trace  # noqa: E402
from garak.evidence.injection_spec import InjectionSpec  # noqa: E402
from garak.evidence.seal import rebuild_snapshot  # noqa: E402

ASSETS = Path(__file__).parents[1] / "_assets"
VECTORS = ASSETS / "detectors" / "inconclusive_vectors"
CONSUMER = ASSETS / "evidence" / "consumer_ipi"
GRADING_CORPUS = ASSETS / "detectors" / "grading" / "corpus.jsonl"

VECTOR_CASES = json.loads((VECTORS / "MANIFEST.json").read_text())["cases"]
CONSUMER_CASES = [
    "blocked",
    "delivered",
    "incomplete",
    "reply-judged",
    "sequential",
    "undelivered",
]
SCRIPTED_JUDGE = ("consumer", "reply-judged")


@pytest.fixture(autouse=True)
def no_judge_service(monkeypatch):
    for name in ("JUDGE_URL", "JUDGE_TOKEN"):
        monkeypatch.delenv(name, raising=False)


@contextlib.contextmanager
def scripted_judge(answer):
    """A judge service on a loopback port that gives ``answer`` to each request, and keeps the requests."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - http.server's name
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            service.asked.append(request)
            body = json.dumps({**answer, "id": request["id"]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    service = type("Service", (), {})()
    service.asked = []
    service.url = f"http://127.0.0.1:{server.server_port}/judge"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield service
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def text_of(value):
    """The bytes the runtime's fixtures hold for ``value``."""
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def sealed_of(kind, case):
    """The document, the sealed snapshot and the two files the runtime wrote for them."""
    if kind == "vector":
        folder = VECTORS / case
        document = json.loads((folder / "document.json").read_text(encoding="utf-8"))
    else:
        folder = CONSUMER / case
        episodes = json.loads(
            (folder / "ipi-episodes.json").read_text(encoding="utf-8")
        )
        document = episodes[0]["injection_spec"]
    snapshot_text = (folder / "snapshot.json").read_text(encoding="utf-8")
    verdict_text = (folder / "verdict.json").read_text(encoding="utf-8")
    snapshot = rebuild_snapshot(json.loads(snapshot_text))
    return document, snapshot, snapshot_text, verdict_text


SEALED = [("vector", case) for case in VECTOR_CASES] + [
    ("consumer", case) for case in CONSUMER_CASES
]


def sealed_cases(pairs):
    return [
        pytest.param(kind, case, id=f"{kind}/{case}") for kind, case in sorted(pairs)
    ]


def test_thirteen_documents_are_sealed():
    assert len(SEALED) == 13
    assert len(VECTOR_CASES) == 7


@pytest.mark.parametrize(
    "kind, case", [pytest.param(k, c, id=f"{k}/{c}") for k, c in SEALED]
)
def test_snapshot_value_is_the_one_the_runtime_served(kind, case):
    _, snapshot, snapshot_text, _ = sealed_of(kind, case)
    assert (
        snapshot.snapshot_id == json.loads(snapshot_text)["snapshot_id"]
    ), "the seal digests as the runtime's"
    assert text_of(trace.snapshot_value(snapshot)) == snapshot_text, "byte for byte"


@pytest.mark.parametrize("kind, case", sealed_cases(set(SEALED) - {SCRIPTED_JUDGE}))
def test_trace_values_are_the_ones_the_runtime_wrote(kind, case):
    document, snapshot, snapshot_text, verdict_text = sealed_of(kind, case)
    values = trace.trace_values(document, snapshot)
    assert list(values) == ["snapshot", "verdict"]
    assert text_of(values["snapshot"]) == snapshot_text
    assert text_of(values["verdict"]) == verdict_text, "the verdict, byte for byte"


def test_a_scripted_judges_verdict_is_the_one_the_runtime_wrote(monkeypatch):
    document, snapshot, snapshot_text, verdict_text = sealed_of(*SCRIPTED_JUDGE)
    [kept] = json.loads(verdict_text)["grade"]["judgements"]
    answer = {
        "schema": "midojo.judge.v1",
        **{name: kept[name] for name in ("verdict", "rationale", "model")},
    }
    with scripted_judge(answer) as service:
        monkeypatch.setenv("JUDGE_URL", service.url)
        monkeypatch.setenv("JUDGE_TOKEN", "FAKE-TEST-VALUE-0000")
        values = trace.trace_values(document, snapshot)
    assert service.asked == [
        kept["request"]
    ], "the request is the runtime's, id included"
    [record] = values["verdict"]["grade"]["judgements"]
    assert record["endpoint"] == service.url
    record["endpoint"] = None  # the fixture scrubs the port of its live service
    assert text_of(values["snapshot"]) == snapshot_text
    assert text_of(values["verdict"]) == verdict_text, "the verdict, byte for byte"


@pytest.mark.parametrize("kind, case", sealed_cases(SEALED))
def test_a_spec_that_was_read_gives_the_same_values(kind, case):
    document, snapshot, _, _ = sealed_of(kind, case)
    spec = InjectionSpec.from_document(document)
    assert trace.trace_values(spec, snapshot) == trace.trace_values(document, snapshot)


def test_the_vendored_consumer_files_are_the_ones_named():
    manifest = json.loads((CONSUMER / "MANIFEST.json").read_text())
    assert "cb12e9ac527ba3f611e61878ca6a8ad00fa9b3e1" in manifest["source"]
    held = sorted(
        str(p.relative_to(CONSUMER))
        for p in CONSUMER.rglob("*.json")
        if p.name != "MANIFEST.json"
    )
    assert held == sorted(manifest["files"])
    for name, digest in manifest["files"].items():
        assert (
            hashlib.sha256((CONSUMER / name).read_bytes()).hexdigest() == digest
        ), name
    assert {name.split("/")[0] for name in held} == set(CONSUMER_CASES)


# --- the snapshot value -----------------------------------------------------------


def delivered():
    return sealed_of("consumer", "delivered")


def test_snapshot_value_lists_the_members_in_the_order_of_the_runtimes_response():
    _, snapshot, _, _ = delivered()
    value = trace.snapshot_value(snapshot)
    assert list(value) == [
        "evaluation_id",
        "run_id",
        "snapshot_id",
        "user_task_id",
        "injection_task_id",
        "agent_output",
        "pre_environment",
        "environment",
        "function_calls",
        "turns",
        "observations",
        "evidence_status",
        "sealed_at",
        "capture_failures",
    ]
    assert list(value) == list(trace.SNAPSHOT_KEYS)
    assert set(snapshot.to_json()) | {"snapshot_id"} == set(
        trace.SNAPSHOT_KEYS
    ), "no member is dropped"
    assert value["snapshot_id"] == snapshot.snapshot_id


def test_each_snapshot_value_is_a_fresh_object():
    _, snapshot, _, _ = delivered()
    first = trace.snapshot_value(snapshot)
    expected = copy.deepcopy(first)
    first["function_calls"].clear()
    first["environment"]["added"] = 1
    assert trace.snapshot_value(snapshot) == expected
    assert snapshot.verified()


@pytest.mark.parametrize("name", ["environment", "pre_environment"])
def test_a_snapshot_sealed_with_no_environment_states_no_trace(name):
    _, snapshot, _, _ = delivered()
    data = snapshot.to_json()
    data[name] = None
    unobserved = rebuild_snapshot(data)
    assert (
        unobserved.snapshot_id != snapshot.snapshot_id
    ), "a null digests differently from {}"
    with pytest.raises(trace.TraceRefused, match=name):
        trace.snapshot_value(unobserved)
    with pytest.raises(trace.TraceRefused, match=name):
        trace.trace_values(sealed_of("consumer", "delivered")[0], unobserved)


@pytest.mark.parametrize("value", [None, {}, "snapshot", delivered()[2]])
def test_only_a_sealed_snapshot_has_a_value(value):
    with pytest.raises(trace.TraceRefused, match="sealed snapshot"):
        trace.snapshot_value(value)


def test_a_snapshot_that_no_longer_matches_its_digest_has_no_value():
    _, snapshot, _, _ = delivered()
    changed = rebuild_snapshot({**snapshot.to_json(), "agent_output": "other"})
    object.__setattr__(changed, "snapshot_id", snapshot.snapshot_id)
    with pytest.raises(trace.TraceRefused, match="digest"):
        trace.snapshot_value(changed)


def test_the_verdict_names_the_snapshot_it_graded():
    document, snapshot, _, _ = delivered()
    values = trace.trace_values(document, snapshot)
    assert values["verdict"]["snapshot_id"] == values["snapshot"]["snapshot_id"]
    assert values["verdict"]["evaluation_id"] == values["snapshot"]["evaluation_id"]
    assert values["verdict"]["schema_version"] == grading.VERDICT_DOCUMENT_V1


def test_a_document_the_reader_refuses_gives_no_value():
    document, snapshot, _, _ = delivered()
    document = copy.deepcopy(document)
    document["verifiers"] = ["custom"]
    with pytest.raises(grading.DocumentRefused, match="verifiers"):
        trace.trace_values(document, snapshot)


# --- the snapshots the runtime sealed for the grading corpus ----------------------


@lru_cache(maxsize=1)
def corpus_snapshots():
    base, snapshots = None, {}
    for line in GRADING_CORPUS.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        if entry["kind"] == "header":
            base = entry["base"]
        elif entry["kind"] == "snapshot":
            snapshots[entry["name"]] = {**base["snapshot"], **entry["delta"]}
    return snapshots


@pytest.mark.parametrize("name", sorted(corpus_snapshots()))
def test_snapshot_value_of_every_corpus_snapshot_is_the_runtimes(name):
    data = corpus_snapshots()[name]
    snapshot = rebuild_snapshot(data)
    if data["environment"] is None or data["pre_environment"] is None:
        with pytest.raises(trace.TraceRefused):
            trace.snapshot_value(snapshot)
        return
    assert json.dumps(trace.snapshot_value(snapshot), sort_keys=True) == json.dumps(
        data, sort_keys=True
    ), "the value is the runtime's snapshot, its ID included"
