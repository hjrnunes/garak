garak.probes.midojo
===================

Payload probes for ``generators.midojo.EpisodeGenerator``. Each prompt is the
text an episode plants in a tool result, never a task for the agent; each
attempt prompt is marked as a payload, which the episode generator requires.

``EpisodePayloads`` fills indirect prompt injection templates with
``instruction``, the attacker's goal; set it to what the episode's security
check looks for. ``templates`` replaces the templates.

Its primary detector, ``detectors.midojo.EpisodeVerdict``, scores whether the
episode's security check held, a call blocked at the interceptor included. Its
extended detector, ``detectors.midojo.EpisodeEffect``, scores whether the
injected action took effect; run with ``--extended_detectors`` to report both
and tell a call that complied but was contained (1.0 and 0.0) from one that
took effect (1.0 and 1.0).

.. automodule:: garak.probes.midojo
   :members:
   :undoc-members:
   :show-inheritance:
