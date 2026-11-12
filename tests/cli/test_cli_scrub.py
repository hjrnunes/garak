# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text that came back from a server holds none of the credentials the request carried."""

from unittest.mock import MagicMock

import pytest

from garak import cli

SECRET = "sentinel-secret-must-not-be-kept"
QUOTED = 'sentinel-"quoted"-secret'


class _Model:
    """A stand-in for an SDK model: it dumps itself to a dict."""

    def __init__(self, **fields):
        self.fields = fields

    def model_dump(self, mode="python", by_alias=False):
        assert (mode, by_alias) == ("json", True), "the dump is the JSON, aliased one"
        return dict(self.fields)


def test_a_secret_is_replaced_wherever_a_string_sits():
    value = {
        f"key {SECRET}": [f"a {SECRET}", (f"b {SECRET}",)],
        "model": _Model(note=f"c {SECRET}"),
    }

    assert cli.scrub_secrets(value, [SECRET]) == {
        f"key {cli.REDACTED}": [f"a {cli.REDACTED}", [f"b {cli.REDACTED}"]],
        "model": {"note": f"c {cli.REDACTED}"},
    }, "dict keys, list and tuple members and models are all scrubbed"


def test_a_secret_is_matched_as_written_and_as_json_escapes_it():
    escaped = QUOTED.replace('"', '\\"')

    assert (
        cli.scrub_secrets(f"plain {QUOTED} escaped {escaped}", [QUOTED])
        == f"plain {cli.REDACTED} escaped {cli.REDACTED}"
    ), "a server can echo either spelling"


def test_the_longest_secret_goes_first():
    assert (
        cli.scrub_secrets("Bearer abc", ["abc", "Bearer abc"]) == cli.REDACTED
    ), "a value and its last word do not leave a stray prefix"


@pytest.mark.parametrize("secrets", [[], [""], [None, ""], [3]])
def test_a_secret_that_is_not_text_leaves_the_value_alone(secrets):
    assert cli.scrub_secrets("keep this", secrets) == "keep this", "nothing to match"


def test_an_object_that_only_answers_to_model_dump_is_left_alone():
    pretender = MagicMock()

    assert cli.scrub_secrets({"x": pretender}, [SECRET]) == {
        "x": pretender
    }, "a mock answers to any method, and must not be dumped without end"


def test_what_is_not_text_comes_back_as_it_is():
    value = {"count": 3, "ratio": 1.5, "flag": True, "none": None}

    assert (
        cli.scrub_secrets(value, [SECRET]) == value
    ), "numbers, booleans and None stay"
