"""Checks the vendored tool-call-condition contract against the producer's.

The files under ``tests/_assets/detectors/tool_call_condition`` are a byte copy
of the contract that the scenario producer owns. The digest tests in
``test_detectors_toolcall_vectors.py`` compare the copy with its own reference,
so a copy that has fallen behind the producer still passes them. This module
compares the copy with the producer's files.

Set ``TOOL_CALL_CONDITION_PRODUCER`` to the producer's checkout root, the
directory that holds ``data/contracts/tool-call-condition``, and run with
``-rs`` to see why a test skips::

    TOOL_CALL_CONDITION_PRODUCER=/path/to/producer \\
        python -m pytest -rs tests/detectors/test_detectors_toolcall_copy.py

Without the variable, or with a checkout that holds no contract, the comparison
skips and says why. The other tests need no producer.
"""

import os
import pathlib
import re
import shutil

import pytest

PRODUCER_VARIABLE = "TOOL_CALL_CONDITION_PRODUCER"
PRODUCER_CONTRACT = pathlib.Path("data/contracts/tool-call-condition")
COPY = (
    pathlib.Path(__file__).parents[1] / "_assets" / "detectors" / "tool_call_condition"
)


def files_under(root):
    """Every file under root by relative path, the reference included."""
    return {
        path.relative_to(root).as_posix(): path
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def differences(copy, producer):
    """Name each file that is missing from the copy, only in it, or not equal byte for byte."""
    copied, original = files_under(copy), files_under(producer)
    found = []
    for name in sorted(set(copied) | set(original)):
        if name not in copied:
            found.append(f"{name} is missing from the copy")
        elif name not in original:
            found.append(f"{name} is not in the producer's contract")
        elif copied[name].read_bytes() != original[name].read_bytes():
            found.append(f"{name} differs from the producer's file")
    return found


def producer_contract(environ):
    """The producer's contract directory, or a skip that says what to set or where it looked."""
    checkout = environ.get(PRODUCER_VARIABLE)
    if not checkout:
        pytest.skip(
            f"{PRODUCER_VARIABLE} is not set; set it to the producer checkout "
            f"that holds {PRODUCER_CONTRACT}"
        )
    contract = pathlib.Path(checkout) / PRODUCER_CONTRACT
    if not contract.is_dir():
        pytest.skip(f"the producer contract directory {contract} does not exist")
    return contract


def test_copy_equals_the_producers_contract():
    found = differences(COPY, producer_contract(os.environ))
    assert not found, "the copy differs from the producer's contract:\n" + "\n".join(
        found
    )


@pytest.fixture
def producer(tmp_path):
    """A stand-in producer contract that equals the copy."""
    contract = tmp_path / PRODUCER_CONTRACT
    shutil.copytree(COPY, contract)
    return contract


def test_a_producer_equal_to_the_copy_gives_no_difference(producer):
    assert differences(COPY, producer) == [], "an equal contract shows a difference"


def change_one_byte(path):
    """Make a file differ by one byte; an empty file gains one."""
    data = bytearray(path.read_bytes())
    if data:
        data[-1] ^= 1
    else:
        data.append(1)
    path.write_bytes(bytes(data))


@pytest.mark.parametrize("name", sorted(files_under(COPY)))
def test_one_changed_byte_is_named(producer, name):
    change_one_byte(producer / name)
    assert differences(COPY, producer) == [
        f"{name} differs from the producer's file"
    ], "one changed byte is not named"


def test_a_byte_added_to_an_empty_file_is_named(tmp_path):
    copy, other = tmp_path / "copy", tmp_path / "other"
    for root in (copy, other):
        root.mkdir()
        (root / "empty").write_bytes(b"")
    assert differences(copy, other) == [], "equal empty files show a difference"
    change_one_byte(other / "empty")
    assert differences(copy, other) == [
        "empty differs from the producer's file"
    ], "a byte added to an empty file is not named"


def test_a_file_only_the_producer_has_is_named(producer):
    newer = producer / "tool-call-condition-v2" / "README.md"
    newer.parent.mkdir()
    newer.write_text("a later version\n", encoding="utf-8")
    assert differences(COPY, producer) == [
        "tool-call-condition-v2/README.md is missing from the copy"
    ], "a file the copy lacks is not named"


def test_a_file_only_the_copy_has_is_named(producer):
    (producer / "tool-call-condition-v1" / "schema.json").unlink()
    assert differences(COPY, producer) == [
        "tool-call-condition-v1/schema.json is not in the producer's contract"
    ], "a file the producer lacks is not named"


def test_without_the_variable_the_comparison_skips_and_names_it():
    with pytest.raises(pytest.skip.Exception, match=PRODUCER_VARIABLE):
        producer_contract({})


def test_a_checkout_without_the_contract_skips_and_names_the_path(tmp_path):
    expected = re.escape(str(tmp_path / PRODUCER_CONTRACT))
    with pytest.raises(pytest.skip.Exception, match=expected):
        producer_contract({PRODUCER_VARIABLE: str(tmp_path)})
