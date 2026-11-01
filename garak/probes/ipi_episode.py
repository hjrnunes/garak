# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deprecated aliases for :mod:`garak.probes.midojo`.

The probe moved to :mod:`garak.probes.midojo`, and its classes were renamed
with it: ``IPIEpisode`` is now :class:`~garak.probes.midojo.InjectionSpecProbe`
and ``EpisodeEntry`` :class:`~garak.probes.midojo.InjectionSpecEntry`. This
module keeps the old plugin name loadable until the planned alias removal, and
re-exports the names a source or a test imports from it. A probe loaded here
reads the same documents and writes the same notes as the renamed one.
"""

from garak.probes.midojo import (  # noqa: F401  (re-exported for old importers)
    CHAIN_NOTE,
    CONDITION_NOTE,
    EMPTY_REPLY,
    INJECTION_SPEC_KEY,
    LEGACY_CHAIN_NOTE,
    LEGACY_INJECTION_SPEC_KEY,
    LEGACY_TRACE_NOTE,
    METADATA_NOTE,
    TRACE_NOTE,
    InjectionSpecEntry,
    InjectionSpecProbe,
    entry_holds_injection_spec,
    entry_injection_spec,
    load_injection_spec_entries,
    plain,
)

EpisodeEntry = InjectionSpecEntry
load_episode_entries = load_injection_spec_entries


class IPIEpisode(InjectionSpecProbe):
    """Deprecated alias for :class:`garak.probes.midojo.InjectionSpecProbe`

    Loads the same documents as the renamed probe and stamps the same notes on
    its attempts, so a configuration, a report or a receipt that names
    ``probes.ipi_episode.IPIEpisode`` still runs and still reads. New work
    should name ``probes.midojo.InjectionSpecProbe``.
    """
