# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A server that echoes a tool credential in its response leaves it out of every log record."""

import asyncio
import gc
import http.server
import io
import json
import logging
import threading

import pytest

from garak.attempt import Conversation, Message, Turn
from garak.cli import REDACTED
from garak.generators import openai as openai_generator
from garak.generators.openai import OpenAIResponsesGenerator

BEARER = "sentinel-tool-bearer-echoed-by-the-server"
TOOLS = [
    {
        "type": "mcp",
        "server_label": "bank",
        "server_url": "http://tools.test/sse",
        "authorization": BEARER,
    }
]
RESPONSE = {
    "id": "resp_1",
    "object": "response",
    "created_at": 0,
    "model": "m",
    "status": "completed",
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
    "output": [],
}
# the channels a server has for an echo, besides the body the generator scrubs
CHANNELS = {
    "header": {"status": 200, "headers": {"X-Echoed-Token": BEARER}},
    "reason phrase": {"status": 200, "reason": f"OK {BEARER}"},
    "rejection reason phrase": {"status": 400, "reason": f"Bad Request {BEARER}"},
    "rejection header": {"status": 404, "headers": {"X-Echoed-Token": BEARER}},
}


@pytest.fixture(autouse=True)
def empty_registry():
    """A test that fails halfway leaves no secret for the tests after it.

    The reset takes no lock: a failed test can leave the lock it swapped in held
    by a thread that never finishes.
    """
    yield
    openai_generator._ACTIVE_SECRETS.clear()
    openai_generator._ACTIVE_SNAPSHOT = ()


@pytest.fixture
def server():
    """A local API that answers every POST as its ``reply`` says."""
    reply = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_POST(self):
            request = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if reply.get("redirect") and self.path.endswith("/responses"):
                token = json.loads(request)["tools"][0]["authorization"]
                self.send_response(307)
                self.send_header("Location", f"{reply['uri']}/echo/{token}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            body = json.dumps(reply.get("json", RESPONSE)).encode()
            self.send_response(reply["status"], reply.get("reason"))
            for name, value in reply.get("headers", {}).items():
                self.send_header(name, value)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    reply["uri"] = f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    yield reply
    httpd.shutdown()
    httpd.server_close()
    thread.join()


def _generate(uri):
    generator = OpenAIResponsesGenerator(
        name="m",
        config_root={
            "generators": {
                "openai": {"OpenAIResponsesGenerator": {"uri": uri, "api_key": "k"}}
            }
        },
    )
    return generator._call_model(
        Conversation([Turn("user", Message("hello"))], notes={"tools": TOOLS})
    )


def _written(records):
    """What each handler would write for each record: message and traceback."""
    return [(record.name, logging.Formatter().format(record)) for record in records]


@pytest.mark.parametrize("channel", CHANNELS)
def test_a_response_that_echoes_the_bearer_leaves_it_out_of_every_record(
    channel, server, caplog
):
    server.update(CHANNELS[channel])
    with caplog.at_level(logging.DEBUG):
        outputs = _generate(server["uri"])

    assert outputs, "the call finished"
    written = _written(caplog.records)
    assert not [
        name for name, text in written if BEARER in text
    ], "no logger wrote the bearer"
    assert BEARER not in caplog.text, "nor did the handler"
    marked = {name for name, text in written if REDACTED in text}
    assert "openai._base_client" in marked, "the client logged the response, gap marked"
    assert any(
        name.startswith("httpx") or name.startswith("httpcore") for name in marked
    ), "so did the HTTP libraries under it"


@pytest.mark.parametrize("name", ["openai._base_client", "httpcore.http11"])
def test_a_handler_on_a_logger_of_the_client_leaves_the_bearer_out_too(name, server):
    """A record reaches the handlers of its own logger before the root's."""
    server.update(CHANNELS["header"])
    sink = io.StringIO()
    handler = logging.StreamHandler(sink)
    logger = logging.getLogger(name)
    level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        _generate(server["uri"])
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)

    assert sink.getvalue(), "the handler saw the client's records"
    assert BEARER not in sink.getvalue(), "but not the bearer a header echoed"
    assert REDACTED in sink.getvalue(), "with the gap marked"


def test_a_redirect_that_carries_the_bearer_leaves_it_out_of_the_warning(
    server, caplog
):
    """The client follows the server's redirect, so the URL it reports is the server's."""
    server.update({"status": 418, "redirect": True})
    with caplog.at_level(logging.DEBUG):
        (output,) = _generate(server["uri"])

    assert output.notes["error"]["status_code"] == 418, "the rejection came from there"
    warnings = [text for name, text in _written(caplog.records) if "HTTP 418" in text]
    assert any(
        f"/echo/{REDACTED}" in text for text in warnings
    ), "the warning names the URL, gap marked"
    assert not [
        name for name, text in _written(caplog.records) if BEARER in text
    ], "no logger wrote the bearer"


def test_a_response_that_echoes_nothing_is_logged_as_received(server, caplog):
    server.update({"status": 200, "headers": {"X-Request-Id": "req_abc123"}})
    with caplog.at_level(logging.DEBUG):
        _generate(server["uri"])

    assert "req_abc123" in caplog.text, "records of a call with no echo stay whole"


def test_the_secrets_are_forgotten_when_the_call_ends():
    with pytest.raises(RuntimeError):
        with openai_generator._logs_scrubbed_of([BEARER]):
            assert BEARER in openai_generator._ACTIVE_SECRETS, "the call registers them"
            raise RuntimeError("the call failed")

    assert not openai_generator._ACTIVE_SECRETS, "a failed call forgets them too"


def test_a_secret_stays_registered_until_every_call_using_it_ends():
    with openai_generator._logs_scrubbed_of([BEARER]):
        with openai_generator._logs_scrubbed_of([BEARER]):
            pass
        assert BEARER in openai_generator._ACTIVE_SECRETS, "one call is still in flight"

    assert not openai_generator._ACTIVE_SECRETS, "the last call forgets it"


def _record(msg, args=None, exc_info=None):
    return logging.LogRecord("x", logging.DEBUG, __file__, 1, msg, args, exc_info)


def test_the_filter_takes_no_lock_when_a_finalizer_logs_while_one_is_held(monkeypatch):
    """Garbage collection runs between any two bytecodes, a lock block's too.

    The lock is swapped for a throwaway that cannot be re-entered, so a filter
    that takes it hangs this test and not the module's own lock.
    """
    lock = threading.Lock()
    monkeypatch.setattr(openai_generator, "_ACTIVE_SECRETS_LOCK", lock)
    sink = io.StringIO()
    handler = logging.StreamHandler(sink)
    handler.addFilter(openai_generator._SCRUB_ACTIVE_SECRETS)
    root = logging.getLogger()
    finished = threading.Event()

    def collect_while_locked():
        with lock:
            loop = asyncio.new_event_loop()
            future = loop.create_future()
            future.set_exception(RuntimeError("never retrieved"))
            loop.close()
            cycle = [future]
            cycle.append(cycle)
            del future, cycle
            gc.collect()
        finished.set()

    root.addHandler(handler)
    try:
        threading.Thread(target=collect_while_locked, daemon=True).start()
        assert finished.wait(5), "the filter waits for a lock its own thread holds"
    finally:
        root.removeHandler(handler)

    assert "never retrieved" in sink.getvalue(), "the finalizer logged through it"


def test_the_registry_lock_can_be_taken_again_by_its_holder():
    lock = openai_generator._ACTIVE_SECRETS_LOCK
    with lock:
        taken = lock.acquire(blocking=False)
        if taken:
            lock.release()

    assert taken, "a holder that logs from a finalizer cannot wait on itself"


def test_the_filter_follows_the_calls_in_flight():
    record = _record("%s", (BEARER,))
    other = "second-call-bearer"

    with openai_generator._logs_scrubbed_of([BEARER]):
        with openai_generator._logs_scrubbed_of([other]):
            openai_generator._ScrubActiveSecrets().filter(record)
            assert BEARER not in record.getMessage(), "the first call is covered"
        later = _record("%s %s", (BEARER, other))
        openai_generator._ScrubActiveSecrets().filter(later)
        assert BEARER not in later.getMessage(), "still in flight after the second ends"
        assert other in later.getMessage(), "the second is forgotten when it ends"

    after = _record("%s", (BEARER,))
    openai_generator._ScrubActiveSecrets().filter(after)
    assert after.getMessage() == BEARER, "the last call to end leaves none"


def _string_whose_hash_fails_after(hashes_allowed, text):
    class Hostile(str):
        calls = 0

        def __hash__(self):
            Hostile.calls += 1
            if Hostile.calls > hashes_allowed:
                raise ValueError("hash failed")
            return str.__hash__(self)

    return Hostile(text)


@pytest.mark.parametrize("hashes_allowed", [0, 1])
def test_a_secret_with_a_failing_hash_registers_as_a_plain_string(hashes_allowed):
    """A failure halfway through registration would leave a secret behind."""
    other = "second-call-bearer"
    hostile = _string_whose_hash_fails_after(hashes_allowed, other)

    with openai_generator._logs_scrubbed_of([BEARER, hostile]):
        assert {type(s) for s in openai_generator._ACTIVE_SECRETS} == {str}, "plain"
        record = _record("%s %s", (BEARER, other))
        openai_generator._ScrubActiveSecrets().filter(record)
        assert BEARER not in record.getMessage(), "the first secret is scrubbed"
        assert other not in record.getMessage(), "so is the one with the bad hash"

    assert not openai_generator._ACTIVE_SECRETS, "the call forgets both"
    after = _record("%s %s", (BEARER, other))
    openai_generator._ScrubActiveSecrets().filter(after)
    assert after.getMessage() == f"{BEARER} {other}", "no later record is scrubbed"


def test_only_non_empty_strings_are_registered():
    with openai_generator._logs_scrubbed_of(["", None, 7, BEARER]):
        assert list(openai_generator._ACTIVE_SECRETS) == [BEARER], "as scrub_secrets"


def test_the_filter_leaves_a_record_alone_when_no_call_is_in_flight():
    record = _record("%s", (BEARER,))

    assert openai_generator._ScrubActiveSecrets().filter(record) is True
    assert record.getMessage() == BEARER, "no call, no scrubbing"


def test_the_filter_scrubs_a_record_whose_arguments_do_not_fit_its_message():
    record = _record("%s %s", (BEARER,))

    with openai_generator._logs_scrubbed_of([BEARER]):
        assert openai_generator._ScrubActiveSecrets().filter(record) is True

    assert BEARER not in record.getMessage(), "a bad record does not raise or leak"


def test_the_filter_scrubs_the_traceback_of_a_record():
    try:
        raise ValueError(f"rejected {BEARER}")
    except ValueError:
        import sys

        record = _record("failed", exc_info=sys.exc_info())

    with openai_generator._logs_scrubbed_of([BEARER]):
        openai_generator._ScrubActiveSecrets().filter(record)

    assert BEARER not in logging.Formatter().format(record), "the traceback is scrubbed"
    assert "ValueError" in record.exc_text, "and still a traceback"
