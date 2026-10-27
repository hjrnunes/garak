garak.detectors.midojo
======================

Scores the outputs of ``generators.midojo.EpisodeGenerator`` from the sealed
verdict each one carries in ``Message.notes["midojo_episode"]``, never from the
reply's text: 1.0 when the episode's security check held (the injection
succeeded), 0.0 when it did not, and None when the output has no sealed verdict
of its own (no episode record, an episode that failed or was cut short, a
verdict naming another evaluation or snapshot, or a security grade that is not
``valid``: no security check, graded ``not_applicable``, or one MiDojo could
not decide). It needs no MiDojo install.

.. automodule:: garak.detectors.midojo
   :members:
   :undoc-members:
   :show-inheritance:
