# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A Responses request the API rejects yields an output that says why."""

import httpx
import respx

from garak.attempt import Conversation, Message, Turn
from garak.generators.openai import OpenAIResponsesGenerator

URI = "http://errors.test/v1"
REJECTION = b'{"error": {"message": "unsupported tool", "code": "bad_tool"}}'
RESPONSES_OK = {
    "id": "resp_1",
    "object": "response",
    "created_at": 0,
    "model": "m",
    "status": "completed",
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
    "output": [
        {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "done", "annotations": []}],
        }
    ],
}


def _generate(response):
    generator = OpenAIResponsesGenerator(
        name="m",
        config_root={
            "generators": {
                "openai": {
                    "OpenAIResponsesGenerator": {
                        "uri": URI,
                        "api_key": "sk-test-fake-key",
                    }
                }
            }
        },
    )
    with respx.mock(base_url=URI) as router:
        router.post("/responses").mock(return_value=response)
        return generator._call_model(Conversation([Turn("user", Message("hello"))]))


def test_rejected_request_notes_the_error_type_and_raw_body():
    outputs = _generate(httpx.Response(400, content=REJECTION))

    assert len(outputs) == 1, "a failed generation still yields one output"
    assert outputs[0].text is None, "a failed generation has no text"
    assert outputs[0].notes["error"] == {
        "type": "BadRequestError",
        "status_code": 400,
        "body": REJECTION.decode(),
    }, "the note names the error and carries the response body unparsed"


def test_successful_output_carries_no_error_note():
    outputs = _generate(httpx.Response(200, json=RESPONSES_OK))

    assert outputs[0].text == "done"
    assert "error" not in outputs[0].notes, "only a failed generation gets the note"
