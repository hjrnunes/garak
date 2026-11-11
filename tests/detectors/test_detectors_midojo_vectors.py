# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the MiDojo-owned vectors byte for byte without installing MiDojo.

The sealed detector tests assert the EvaluationVerdict and EvaluationEffect
columns. The tool-call condition column belongs to its own detector tests.
"""

import hashlib
import json
from pathlib import Path

import pytest

VECTORS = Path(__file__).parents[1] / "_assets" / "detectors" / "inconclusive_vectors"
MANIFEST = json.loads((VECTORS / "MANIFEST.json").read_text(encoding="utf-8"))


def test_exact_cases_and_files():
    assert len(MANIFEST["cases"]) == 7, "the owner pins seven sealed cases"
    assert set(MANIFEST["files"]) >= set(MANIFEST["copies"]), "every copy has a digest"
    expected = {"MANIFEST.json", *MANIFEST["copies"]}
    actual = {
        path.relative_to(VECTORS).as_posix()
        for path in VECTORS.rglob("*")
        if path.is_file()
    }
    assert actual == expected, "the vendored copy has exactly the owner's files"
    assert set(MANIFEST["copies"]) == {
        f"{case}/{name}"
        for case in MANIFEST["cases"]
        for name in ("document.json", "expected.json", "snapshot.json", "verdict.json")
    }, "each case has the four expected documents"


@pytest.mark.parametrize("name", sorted(MANIFEST["copies"]))
def test_vendored_bytes_match_manifest(name):
    actual = hashlib.sha256((VECTORS / name).read_bytes()).hexdigest()
    assert actual == MANIFEST["files"][name], f"{name} differs from its owner digest"
