# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checks the tool-call condition detector against the shared contract vectors.

The vector files come from the tool-call-condition-v2 contract, which the
scenario producer owns. Every implementation of the contract must reproduce the
results in them. Copy new versions of the files byte for byte and update the
digests below; a stale or edited copy fails the digest test.
"""

import hashlib
import json
import pathlib

import pytest

from garak.detectors import toolcall

VECTOR_DIR = (
    pathlib.Path(__file__).parents[1] / "_assets" / "detectors" / "tool_call_condition"
)

VECTOR_SHA256 = {
    "validate.json": "8977048ddf5a7d8843d4406d03cfd7694469093511b002314f04594b16d5b4a3",
    "evaluate.json": "893e357f61eb0c71b4821445d3c3dd4171bfc929fc444e8e20cb1294111c8e6d",
}


CONTRACT_VERSION = "tool-call-condition-v2"


def load_vector_file(name):
    return json.loads((VECTOR_DIR / name).read_text(encoding="utf-8"))


def load_vectors(name):
    return load_vector_file(name)["vectors"]


VALIDATE_VECTORS = load_vectors("validate.json")
EVALUATE_VECTORS = load_vectors("evaluate.json")


@pytest.mark.parametrize("name", sorted(VECTOR_SHA256))
def test_vector_file_matches_recorded_digest(name):
    digest = hashlib.sha256((VECTOR_DIR / name).read_bytes()).hexdigest()
    assert digest == VECTOR_SHA256[name], f"{name} differs from the contract copy"


@pytest.mark.parametrize("name", sorted(VECTOR_SHA256))
def test_vector_file_states_the_implemented_version(name):
    assert (
        load_vector_file(name)["version"] == CONTRACT_VERSION
    ), f"{name} is not a {CONTRACT_VERSION} vector file"


def test_vector_ids_are_unique():
    for vectors in (VALIDATE_VECTORS, EVALUATE_VECTORS):
        ids = [vector["id"] for vector in vectors]
        assert len(ids) == len(set(ids)), "vector ids repeat"


@pytest.mark.parametrize("vector", VALIDATE_VECTORS, ids=lambda v: v["id"])
def test_validate_condition_reproduces_vector(vector):
    error = toolcall.validate_condition(vector["condition"])
    assert (error is None) == vector["valid"], "validity differs from the vector"
    assert error == vector["error"], "problem text differs from the vector"


@pytest.mark.parametrize("vector", EVALUATE_VECTORS, ids=lambda v: v["id"])
def test_evaluate_condition_reproduces_vector(vector):
    assert (
        toolcall.validate_condition(vector["condition"]) is None
    ), "an evaluate vector holds an invalid condition"
    got = toolcall.evaluate_condition(
        vector["condition"], vector["calls"], vector["complete"]
    )
    want = (vector["outcome"], vector["reason"], vector["matched_calls"])
    assert got == want, "outcome, reason, or matched calls differ from the vector"
