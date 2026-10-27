garak.probes.midojo
===================

Payload probes for ``generators.midojo.EpisodeGenerator``. Each prompt is the
text an episode plants in a tool result, never a task for the agent; each
attempt prompt is marked as a payload, which the episode generator requires.

``EpisodePayloads`` fills indirect prompt injection templates with
``instruction``, the attacker's goal; set it to what the episode's security
check looks for. ``templates`` replaces the templates.

.. automodule:: garak.probes.midojo
   :members:
   :undoc-members:
   :show-inheritance:
