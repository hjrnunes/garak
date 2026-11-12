# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A pair of attempts, one attacked and one benign, and the utility drop over valid pairs."""

import pytest

from garak.attempt import Attempt, Message
from garak.resources import metrics

DETECTOR = "det.Utility"


def completed(score):
    return score > 0.5


def attempt(role=None, pair="p1", scores=(1.0,), reason=None, detector=DETECTOR):
    built = Attempt(prompt=Message(text="prompt"))
    built.detector_results = {detector: list(scores)}
    if role is not None:
        metrics.mark_pair(built, pair, role)
    if reason is not None:
        metrics.mark_unscored(built, detector, reason)
    return built


def pair(benign, attacked, name="p1"):
    """One attacked and one baseline attempt; None leaves a side unscored."""
    return [
        attempt(
            metrics.ATTACKED,
            name,
            [attacked],
            "errored" if attacked is None else None,
        ),
        attempt(metrics.BASELINE, name, [benign]),
    ]


def summary(attempts, detector=DETECTOR):
    return metrics.pair_summary(attempts, detector, completed)


def test_a_pair_note_reads_back():
    built = attempt(metrics.BASELINE, "abc")

    assert metrics.pair_of(built) == ("abc", metrics.BASELINE), "the note round-trips"
    assert metrics.is_baseline(built), "a baseline attempt says so"
    assert not metrics.is_baseline(
        attempt(metrics.ATTACKED)
    ), "an attacked attempt is not a baseline"


@pytest.mark.parametrize(
    "notes",
    [
        None,
        {},
        {metrics.PAIR_NOTE: "p1"},
        {metrics.PAIR_NOTE: {"id": "", "role": metrics.BASELINE}},
        {metrics.PAIR_NOTE: {"id": 7, "role": metrics.BASELINE}},
        {metrics.PAIR_NOTE: {"id": "p1", "role": "other"}},
        {metrics.PAIR_NOTE: {"id": "p1"}},
    ],
    ids=[
        "no-notes",
        "no-note",
        "not-a-mapping",
        "empty-id",
        "int-id",
        "bad-role",
        "no-role",
    ],
)
def test_an_unreadable_pair_note_is_no_pair(notes):
    built = attempt()
    built.notes = notes

    assert metrics.pair_of(built) is None, "unreadable notes read as no pair"
    assert not metrics.is_baseline(built), "and never as a baseline"


@pytest.mark.parametrize(
    "pair_id,role", [("", metrics.BASELINE), (7, metrics.BASELINE), ("p1", "x")]
)
def test_a_pair_is_marked_with_a_text_id_and_a_known_role(pair_id, role):
    with pytest.raises(ValueError):
        metrics.mark_pair(attempt(), pair_id, role)


def test_no_pair_in_the_attempts_gives_no_summary():
    assert (
        summary([attempt(), attempt()]) is None
    ), "a run with no baseline has no pairs"


def test_the_drop_is_counted_over_valid_pairs():
    attempts = [
        *pair(1.0, 1.0, "a"),
        *pair(1.0, 0.0, "b"),
        *pair(1.0, 0.0, "c"),
        *pair(0.0, 0.0, "d"),
    ]

    assert summary(attempts) == {
        "pairs": 4,
        "valid": 4,
        "invalid": 0,
        "benign_completed": 3,
        "attacked_completed": 1,
        "benign_utility": 0.75,
        "utility_under_attack": 0.25,
        "drop": 0.5,
    }, "rates and drop are over the valid pairs"


def test_a_drop_can_be_negative():
    result = summary(pair(0.0, 1.0))

    assert result["drop"] == -1.0, "an attack that helps shows as a negative drop"


def test_an_invalid_pair_never_enters_the_rates():
    attempts = [*pair(1.0, 0.0, "good"), *pair(1.0, None, "bad")]

    result = summary(attempts)

    assert (result["pairs"], result["valid"], result["invalid"]) == (2, 1, 1)
    assert result["invalid_reasons"] == {
        "attacked:errored": 1
    }, "the reason names the side"
    assert (result["benign_utility"], result["utility_under_attack"]) == (1.0, 0.0)
    assert result["drop"] == 1.0, "the invalid pair's baseline stays out of the rate"


def test_an_unscored_side_without_a_reason_is_named_as_such():
    built = [
        attempt(metrics.ATTACKED, "p", [1.0]),
        attempt(metrics.BASELINE, "p", [None]),
    ]

    assert summary(built)["invalid_reasons"] == {"baseline:unscored": 1}


@pytest.mark.parametrize(
    "attempts,reason",
    [
        ([attempt(metrics.ATTACKED, "p")], "unpaired"),
        ([attempt(metrics.BASELINE, "p")], "unpaired"),
        (
            [
                attempt(metrics.ATTACKED, "p"),
                attempt(metrics.ATTACKED, "p"),
                attempt(metrics.BASELINE, "p"),
            ],
            "unpaired",
        ),
        (
            [
                attempt(metrics.ATTACKED, "p", [1.0, 1.0]),
                attempt(metrics.BASELINE, "p"),
            ],
            "attacked:outputs",
        ),
        (
            [attempt(metrics.ATTACKED, "p"), attempt(metrics.BASELINE, "p", [])],
            "baseline:outputs",
        ),
    ],
    ids=[
        "no-baseline",
        "no-attacked",
        "two-attacked",
        "attacked-two-outputs",
        "baseline-no-output",
    ],
)
def test_a_pair_without_one_scored_output_per_side_is_invalid(attempts, reason):
    result = summary(attempts)

    assert (result["valid"], result["invalid_reasons"]) == (
        0,
        {reason: 1},
    ), "each side needs exactly one attempt with exactly one score"


def test_no_valid_pair_leaves_the_rates_unknown():
    result = summary(pair(1.0, None))

    assert [
        result[key] for key in ("benign_utility", "utility_under_attack", "drop")
    ] == [
        None,
        None,
        None,
    ], "an empty denominator is unknown, never zero"


def test_the_summary_reads_only_the_named_detector():
    attempts = [
        *pair(1.0, 0.0),
        attempt(metrics.ATTACKED, "q", [1.0], detector="det.Other"),
        attempt(metrics.BASELINE, "q", [1.0], detector="det.Other"),
    ]

    result = summary(attempts)

    assert (result["valid"], result["invalid_reasons"]) == (
        1,
        {"attacked:outputs": 1},
    ), "a pair the named detector did not score is invalid"


def test_attempts_without_a_pair_note_are_left_out():
    result = summary([attempt(), *pair(1.0, 1.0)])

    assert result["pairs"] == 1, "only attempts that carry a pair note are paired"


def test_pooled_summaries_add_counts_and_recompute_rates():
    first = summary([*pair(1.0, 1.0, "a"), *pair(1.0, 0.0, "b")])
    second = summary([*pair(1.0, 0.0, "c"), *pair(1.0, None, "d")])

    pooled = metrics.pool_pairs(metrics.pool_pairs(None, first), second)

    assert pooled == {
        "pairs": 4,
        "valid": 3,
        "invalid": 1,
        "benign_completed": 3,
        "attacked_completed": 1,
        "benign_utility": 1.0,
        "utility_under_attack": 1 / 3,
        "drop": 2 / 3,
        "invalid_reasons": {"attacked:errored": 1},
    }, "rates are over the pooled valid pairs, not an average of rates"


def test_pooling_leaves_its_inputs_alone():
    first = summary(pair(1.0, 0.0))
    before = dict(first)

    metrics.pool_pairs(metrics.pool_pairs(None, first), first)

    assert first == before, "a summary is not changed by being pooled"


GOOD = {
    "pairs": 2,
    "valid": 1,
    "invalid": 1,
    "benign_completed": 1,
    "attacked_completed": 0,
}


@pytest.mark.parametrize(
    "bad",
    [
        None,
        "x",
        [],
        {"pairs": 1},
        {**GOOD, "valid": "1"},
        {**GOOD, "valid": 1.0},
        {**GOOD, "valid": True},
        {**GOOD, "invalid": -1, "pairs": 0},
        {**GOOD, "pairs": 3},
        {**GOOD, "benign_completed": 2},
        {**GOOD, "attacked_completed": 2},
        {**GOOD, "invalid_reasons": ["x"]},
        {**GOOD, "invalid_reasons": {"x": -1}},
        {**GOOD, "invalid_reasons": {"x": "1"}},
        {**GOOD, "invalid_reasons": {1: 1}},
    ],
    ids=[
        "none",
        "text",
        "list",
        "missing-counts",
        "text-count",
        "float-count",
        "bool-count",
        "negative-count",
        "counts-do-not-add-up",
        "benign-above-valid",
        "attacked-above-valid",
        "reasons-list",
        "reason-negative",
        "reason-text-count",
        "reason-not-text",
    ],
)
def test_a_summary_that_cannot_be_one_is_refused_not_pooled(bad):
    with pytest.raises(ValueError):
        metrics.pool_pairs(None, bad)


def test_a_refused_summary_is_refused_beside_a_good_one():
    pooled = metrics.pool_pairs(None, GOOD)

    with pytest.raises(ValueError):
        metrics.pool_pairs(pooled, {**GOOD, "benign_completed": 2})
