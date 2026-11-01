# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deprecated aliases for :mod:`garak.detectors.midojo`.

The detectors moved to :mod:`garak.detectors.midojo`, and their classes were
renamed with them: ``EpisodeVerdict`` is now
:class:`~garak.detectors.midojo.EvaluationVerdict` and ``EpisodeEffect``
:class:`~garak.detectors.midojo.EvaluationEffect`. This module keeps the old
plugin names loadable until the planned alias removal, and re-exports the names
a test or a report reader imports from it. A detector loaded here reads the same
traces and writes the same records as the renamed one.
"""

from garak.detectors.midojo import (  # noqa: F401  (re-exported for old importers)
    CHAIN_NOTE,
    DIGEST_MISMATCH,
    EFFECT_RESULTS_NOTE,
    ERRORED,
    INCOMPLETE,
    INTERMEDIATE_TURN,
    INTERRUPTED,
    INVALID,
    LEGACY_CHAIN_NOTE,
    LEGACY_TRACE_NOTE,
    MISSING,
    NOT_DELIVERED,
    RESULTS_NOTE,
    TRACE_NOTE,
    UNRECORDED_DELIVERY,
    UNRECORDED_EFFECT,
    EvaluationEffect,
    EvaluationVerdict,
)


class EpisodeVerdict(EvaluationVerdict):
    """Deprecated alias for :class:`garak.detectors.midojo.EvaluationVerdict`

    Reads the same sealed trace and writes the same records, so a configuration
    or a report that names ``detectors.ipi_episode.EpisodeVerdict`` still runs.
    New work should name ``detectors.midojo.EvaluationVerdict``.
    """


class EpisodeEffect(EvaluationEffect):
    """Deprecated alias for :class:`garak.detectors.midojo.EvaluationEffect`

    Reads the same sealed trace and writes the same records, so a configuration
    or a report that names ``detectors.ipi_episode.EpisodeEffect`` still runs.
    New work should name ``detectors.midojo.EvaluationEffect``.
    """
