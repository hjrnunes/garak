# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A fake judge service for tests: a real HTTP server on 127.0.0.1 that answers by the request's path.

``/supported``, ``/contradicted`` and ``/unresolved`` answer a valid
``midojo.judge.v1`` response with that verdict. Each other path answers one way
a service can fail (see ``ANSWERS``). Every request is kept, with its headers
and body, in ``requests``. No model is called.

This file imports nothing from Garak or from the MiDojo runtime, so the one
server answers the scripted cases of both graders. It follows the fake judge
service the MiDojo runtime tests its reply judge against.
"""

from __future__ import annotations

import gzip
import json
import threading
import time
import zlib
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

MODEL = "fake-judge-1"
PROMPT_VERSION = "fake-judge-prompt-v1"
SCHEMA = "midojo.judge.v1"
BIG = 8 * 1024 * 1024 + 1


@dataclass(frozen=True)
class Received:
    path: str
    headers: dict[str, str]
    body: dict[str, Any]
    query: str = ""
    raw: bytes = b""


def _valid(verdict: str):
    def answer(request: dict[str, Any], headers: dict[str, str]) -> tuple[int, Any]:
        return 200, {
            "schema": SCHEMA,
            "id": request["id"],
            "verdict": verdict,
            "rationale": f"the reply makes the claim {verdict}",
            "model": MODEL,
            "details": {"evidence_refs": ["reply"], "prompt_version": PROMPT_VERSION},
        }

    return answer


_DROP = object()


def _with(**changes: Any):
    def answer(request: dict[str, Any], headers: dict[str, str]) -> tuple[int, Any]:
        _, body = _valid("supported")(request, headers)
        body = {**body, **changes}
        return 200, {key: value for key, value in body.items() if value is not _DROP}

    return answer


def _nested(depth: int) -> dict:
    value: dict = {}
    for _ in range(depth - 1):
        value = {"deeper": value}
    return value


def _spliced(old: str, new: str):
    """A valid answer written as JSON text with ``old`` replaced by ``new``."""

    def answer(request: dict[str, Any], headers: dict[str, str]) -> tuple[int, Any]:
        text = json.dumps(_valid("supported")(request, headers)[1])
        assert old in text
        return 200, text.replace(old, new).encode()

    return answer


def _echoing(place: str):
    """A valid answer but for the bearer token it was sent, echoed back as ``place``."""

    def answer(request: dict[str, Any], headers: dict[str, str]) -> tuple[int, Any]:
        token = headers.get("authorization", "").removeprefix("Bearer ")
        body = _valid("supported")(request, headers)[1]
        if place == "field-name":
            body[token] = 1
        elif place == "record-fields":
            body.update(
                rationale=f"echo {token}",
                model=token,
                details={token: [token, {"k": token}]},
            )
        else:
            body[place] = token
        return 200, body

    return answer


def _raw_deflate(data: bytes) -> bytes:
    compressor = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    return compressor.compress(data) + compressor.flush()


# The Content-Encoding a path answers with; its answer is already coded.
ENCODINGS: dict[str, str] = {}
# The length of the first of the two chunks a path's answer is sent in, as chunked transfer.
SPLITS: dict[str, int] = {}


def _coded(path: str, coding: str, compress, split: int | None = None):
    """A valid answer that ``compress`` has made into ``coding``, sent with that header."""
    ENCODINGS[path] = coding
    if split is not None:
        SPLITS[path] = split

    def answer(request: dict[str, Any], headers: dict[str, str]) -> tuple[int, Any]:
        _, body = _valid("supported")(request, headers)
        return 200, compress(json.dumps(body).encode())

    return answer


ANSWERS = {
    "/supported": _valid("supported"),
    "/contradicted": _valid("contradicted"),
    "/unresolved": _valid("unresolved"),
    "/minimal": lambda request, headers: (
        200,
        {"schema": SCHEMA, "verdict": "contradicted"},
    ),
    "/500": lambda request, headers: (500, {"error": "boom"}),
    "/404": lambda request, headers: (404, {"error": "no such judge"}),
    "/redirect": lambda request, headers: (302, None),
    "/multiple-choices": lambda request, headers: (300, None),
    "/created": lambda request, headers: (
        201,
        _valid("supported")(request, headers)[1],
    ),
    "/not-json": lambda request, headers: (200, b"<html>supported</html>"),
    # Nested deeper than the JSON parser reads: it raises RecursionError.
    "/recursion": lambda request, headers: (200, b"[" * 200000 + b"]" * 200000),
    "/empty": lambda request, headers: (200, b""),
    "/array": lambda request, headers: (200, ["supported"]),
    "/no-verdict": _with(verdict=_DROP),
    "/bad-verdict": _with(verdict="maybe"),
    "/verdict-type": _with(verdict=True),
    "/verdict-list": _with(verdict=[]),
    "/verdict-object": _with(verdict={}),
    "/no-schema": _with(schema=_DROP),
    "/wrong-schema": _with(schema="midojo.judge.v0"),
    "/rationale-type": _with(rationale=["a", "list"]),
    "/model-type": _with(model=7),
    "/details-type": _with(details="text"),
    "/unknown-field": _with(confidence=0.9),
    "/other-id": _with(id="sha256:" + "0" * 64),
    "/deep-details": _with(details=_nested(100)),
    "/echo-token": lambda request, headers: (
        200,
        {
            "schema": SCHEMA,
            "verdict": "supported",
            "rationale": headers.get("authorization", ""),
        },
    ),
    # A valid answer whose reason phrase and a header echo the Authorization header (see the handler).
    "/echo-in-headers": _valid("supported"),
    # A valid answer but for a number too large for a float, which Python's json module decodes as inf.
    "/huge-number": _spliced('["reply"]', "1e400"),
    # A valid answer whose rationale is JSON's escape for a lone surrogate, which decodes to one.
    "/surrogate-rationale": _spliced(
        '"the reply makes the claim supported"', '"half \\ud800 a pair"'
    ),
    "/big": lambda request, headers: (200, b" " * BIG),
    "/big-failure": lambda request, headers: (500, b" " * BIG),
}
for _place in ("schema", "verdict", "id", "field-name", "record-fields"):
    ANSWERS[f"/echo-as-{_place}"] = _echoing(_place)
# A valid answer in each content coding a client may be sent, and in codings it is not.
ANSWERS["/gzip"] = _coded("/gzip", "gzip", gzip.compress)
ANSWERS["/gzip-capitalised"] = _coded("/gzip-capitalised", "GZip", gzip.compress)
ANSWERS["/deflate"] = _coded("/deflate", "deflate", zlib.compress)
ANSWERS["/deflate-raw"] = _coded("/deflate-raw", "deflate", _raw_deflate)
ANSWERS["/deflate-then-gzip"] = _coded(
    "/deflate-then-gzip",
    "deflate, gzip",
    lambda data: gzip.compress(zlib.compress(data)),
)
# A first chunk of one byte: a zlib stream that begins so is still zlib, a raw one is not read as raw.
ANSWERS["/deflate-split"] = _coded("/deflate-split", "deflate", zlib.compress, split=1)
ANSWERS["/deflate-raw-split"] = _coded(
    "/deflate-raw-split", "deflate", _raw_deflate, split=1
)
ANSWERS["/gzip-split"] = _coded("/gzip-split", "gzip", gzip.compress, split=1)
ANSWERS["/identity-coding"] = _coded("/identity-coding", "identity", lambda data: data)
ANSWERS["/unknown-coding"] = _coded("/unknown-coding", "x-unknown", lambda data: data)
# Valid answers, as they are, that name codings the grader does not decode. The two
# that httpx decodes with a package are not in the corpus: what it makes of them
# depends on what is installed.
ANSWERS["/br"] = _coded("/br", "br", lambda data: data)
ANSWERS["/zstd"] = _coded("/zstd", "zstd", lambda data: data)
ANSWERS["/gzip-then-br"] = _coded("/gzip-then-br", "gzip, br", gzip.compress)
ANSWERS["/gzip-truncated"] = _coded(
    "/gzip-truncated", "gzip", lambda data: gzip.compress(data)[:-12]
)
ANSWERS["/gzip-not-gzip"] = _coded("/gzip-not-gzip", "gzip", lambda data: data)
ANSWERS["/deflate-not-deflate"] = _coded(
    "/deflate-not-deflate", "deflate", lambda data: data
)
# Small on the wire, longer than the limit once decoded.
ANSWERS["/gzip-bomb"] = _coded(
    "/gzip-bomb", "gzip", lambda data: gzip.compress(b" " * BIG)
)
# The same, in two chunks that each decode to less than the limit and together to more.
ANSWERS["/gzip-bomb-in-chunks"] = _coded(
    "/gzip-bomb-in-chunks", "gzip", lambda data: gzip.compress(b" " * BIG), split=1000
)
# Answers whose details hold a constant JSON does not have, as Python's json module writes it.
for _constant in ("NaN", "Infinity", "-Infinity"):
    ANSWERS[f"/details-{_constant}"] = _spliced('["reply"]', _constant)


class FakeJudge:
    """A judge service listening on a free loopback port until ``close``."""

    MODEL = MODEL
    PROMPT_VERSION = PROMPT_VERSION

    def __init__(self) -> None:
        self.requests: list[Received] = []
        judge = self

        class Handler(BaseHTTPRequestHandler):
            # A connection stays open until its client closes it.
            protocol_version = "HTTP/1.1"

            def handle(self) -> None:
                try:
                    super().handle()
                except (BrokenPipeError, ConnectionResetError):
                    pass  # the client closed with part of the answer unread

            def do_POST(self) -> None:  # noqa: N802 - http.server's name
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                body = json.loads(raw)
                headers = {key.lower(): value for key, value in self.headers.items()}
                self.path, _, query = self.path.partition("?")
                judge.requests.append(Received(self.path, headers, body, query, raw))
                if self.path == "/slow":
                    time.sleep(2)
                    self.path = "/supported"
                status, answer = ANSWERS[self.path](body, headers)
                content = (
                    answer if isinstance(answer, bytes) else json.dumps(answer).encode()
                )
                if self.path == "/echo-in-headers":
                    self.send_response(status, headers.get("authorization", ""))
                    self.send_header("X-Echo", headers.get("authorization", ""))
                else:
                    self.send_response(status)
                if status in (300, 302):
                    self.send_header("Location", "/supported")
                if self.path in ENCODINGS:
                    self.send_header("Content-Encoding", ENCODINGS[self.path])
                self.send_header("Content-Type", "application/json")
                if self.path in SPLITS:
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    cut = SPLITS[self.path]
                    for part in (content[:cut], content[cut:]):
                        self.wfile.write(f"{len(part):x}\r\n".encode() + part + b"\r\n")
                    self.wfile.write(b"0\r\n\r\n")
                    return
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
        )
        self._thread.start()

    @property
    def host_port(self) -> str:
        host, port = self._server.server_address[:2]
        return f"{host}:{port}"

    def url(self, path: str) -> str:
        return f"http://{self.host_port}{path}"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
