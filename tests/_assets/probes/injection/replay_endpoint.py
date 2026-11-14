"""A scripted OpenAI-compatible endpoint for running the replay probes offline.

``Endpoint`` answers ``POST .../responses`` with the replies of a case in order,
and ``POST .../chat/completions`` with the case's judge reply, and keeps every
request body it receives. A reply holds ``text``, ``calls`` (tool calls made by
the target, each with ``name``, ``arguments`` and ``output``), ``items`` (output
items to send as they are), ``status`` and ``response`` (fields of the response
object), or ``http`` (a status and body to answer with instead).

Serve one case over HTTP with ``python replay_endpoint.py CASES.json CASE PORT``.
"""

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _reply_body(reply: dict, number: int) -> dict:
    calls = [
        {
            "type": "mcp_call",
            "id": f"mcp_{number}_{i}",
            "server_label": "tools",
            "name": call["name"],
            "arguments": json.dumps(call.get("arguments", {})),
            "output": call.get("output", "ok"),
        }
        for i, call in enumerate(reply.get("calls", []))
    ]
    output = list(reply.get("items", calls))
    if reply.get("text") is not None:
        output.append(
            {
                "type": "message",
                "id": f"msg_{number}",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {"type": "output_text", "text": reply["text"], "annotations": []}
                ],
            }
        )
    usage = {
        "input_tokens": 10,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 2,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 12,
    }
    return {
        "id": f"resp_{number}",
        "object": "response",
        "created_at": 0,
        "model": "test-model",
        "status": reply.get("status", "completed"),
        "output": output,
        "usage": usage,
    } | reply.get("response", {})


class Endpoint:
    """Answer the requests of one case, in order, and keep what they asked."""

    def __init__(self, case: dict):
        self.replies = list(case.get("replies", []))
        self.judge = case.get("judge")
        self.requests = []

    def answer(self, method: str, path: str, body=None):
        """Return the status code and JSON body for a request."""
        if method == "GET":
            return 200, {"data": []}
        self.requests.append((path.rsplit("/", 1)[-1], body))
        if path.endswith("/responses") and self.replies:
            reply = self.replies.pop(0)
            if "http" in reply:
                return reply["http"]["status"], reply["http"]["body"]
            return 200, _reply_body(reply, len(self.requests))
        if path.endswith("/chat/completions") and self.judge is not None:
            message = {"role": "assistant", "content": self.judge}
            return 200, {
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        return 404, {"error": {"message": "no scripted reply for this request"}}

    def requested(self, kind: str) -> list:
        """Return the bodies of the requests made to ``responses`` or ``completions``."""
        return [body for path, body in self.requests if path.startswith(kind)]


def serve(case: dict, port: int) -> None:
    """Serve one case over HTTP on the loopback address until interrupted."""
    endpoint = Endpoint(case)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, payload):
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._send(*endpoint.answer("GET", self.path))

        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self._send(*endpoint.answer("POST", self.path, json.loads(raw or b"{}")))

        def log_message(self, *args):
            pass

    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    cases = json.load(open(sys.argv[1], encoding="utf-8"))
    serve(cases[sys.argv[2]], int(sys.argv[3]))
