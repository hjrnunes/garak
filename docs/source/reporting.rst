Reporting
=========

By default, ``garak`` outputs:

* a JSONL file, with the name ``garak.<uuid>.report.jsonl``, that stores progress and outcomes from a scan
* an HTML report summarising scores
* a JSONL hit log, describing all the attempts from the run that were scored successful

For information on how detectors classify responses as hits or passes, and how detector performance is measured, see :doc:`detector_metrics`.

Report JSONL
------------

The report JSON consists of JSON rows. Each row has an ``entry_type`` field.
Different entry types have different other fields.
Attempt-type entries have uuid and status fields.
Status can be 0 (not sent to target), 1 (with target response but not evaluated), or 2 (with response and evaluation).
Eval-type entries are added after each probe/detector pair completes, and list the results used to compute the score.
When the probe is an ``IntentProbe`` (and therefore tags each attempt with an intent),
the corresponding ``eval`` entry also carries an optional ``intents`` field mapping
each intent name to its ``passed``, ``total_evaluated`` and ``nones`` counts
(``nones`` are unscoreable outputs, mirroring the top-level ``nones`` and excluded
from ``total_evaluated``).

The ``digest`` entry (added by ``report_digest``) carries a
``technique_intent_matrix`` field built from those ``eval`` ``intents`` counts: a
``technique -> intent`` cross-tab keyed on each probe's ``demon:*`` tags
(independent of ``reporting.taxonomy``). Each intent cell holds ``score``
(``passed / total_evaluated``, or ``null`` when ``total_evaluated`` is 0),
``passed``, ``total_evaluated``, ``nones`` and ``probe_detector_pairs`` (the
``[probe, detector]`` combinations whose evals produced that cell), plus a
per-technique ``_summary`` of ``n_intents`` and ``n_detectors``. Counts are
micro-averaged
(pooled) across contributing probes and detectors, so ``total_evaluated`` is an
evaluation count (attempt × detector). An intent whose outputs were all unscoreable
surfaces as ``0/0`` with ``nones`` > 0, signalling that the target never produced a
usable response for it.

.. seealso::

   :doc:`cas` for running intent-based scans and interpreting per-intent results.

Unscored outputs
^^^^^^^^^^^^^^^^

A detector scores an output with a number, or with ``None`` when it cannot.
An unscored output is counted in ``nones`` and never enters ``total_evaluated``, the denominator of the attack success rate.
``total_processed`` is the number of outputs the detector saw: ``total_evaluated`` plus ``nones``.

A detector can say why it left outputs unscored by calling ``Detector.mark_unscored`` with a reason, and with ``not_applicable=True`` when the check does not apply to the output at all, such as an intermediate turn of a multi-turn attack.
The ``eval`` entry, and the ``detection_counts`` of the ``probe_summary`` entry, then carry:

* ``unscored_reasons``: each reason, with the number of outputs it explains
* ``not_applicable``: how many of the ``nones`` were not applicable. The remaining ``nones`` are unknown: the detector could not decide.

Both fields appear only when a detector gave a reason, so entries from other detectors are unchanged.
The console prints an ``unscored:`` line under each detector that left outputs unscored.

Evaluations whose injection was not delivered, or whose delivery was not recorded, stay out of the attack success denominator and are declared in ``unscored_reasons`` as ``not_delivered`` and ``unrecorded_delivery`` (see :doc:`detectors/midojo`).

Utility results
^^^^^^^^^^^^^^^

A detector has a ``metric_kind``, either ``security`` (the default) or ``utility``.
A score of 1.0 always means that the behaviour the detector looks for occurred.
For a ``security`` detector that behaviour is a successful attack, and the score feeds the attack success rate.
For a ``utility`` detector it is a legitimate task that was completed, and the score never reaches an attack success rate.

The harness records a utility detector under ``attempt.notes["metric_kind"]``, so the kind stays in the report, and a later run of the same detector as a security detector removes the record.
If any attempt of a probe carries the record for a detector, the evaluator and the report tools treat that detector as utility for every attempt of the probe.
The evaluator then writes a ``utility_eval`` entry instead of an ``eval`` entry:

.. code-block:: json

   {"entry_type": "utility_eval", "probe": "test.Blank", "detector": "midojo.TaskCompleted", "completed": 3, "not_completed": 1, "nones": 2, "total_evaluated": 4, "total_processed": 6}

``completed`` counts the scores that the evaluator would call a hit.
The entry has no ``passed``, ``fails``, ``hit_at_k`` or confidence fields, and a utility detector writes no hit log row.
It carries ``not_applicable`` and ``unscored_reasons`` in the same way as an ``eval`` entry.
The ``probe_summary`` entry gains ``utility_counts``, with the detectors, ``completed``, ``not_completed``, ``nones`` and the unscored fields, only when a utility detector ran. ``detection_counts`` never includes utility.
The ``digest`` entry gains a ``utility`` section, keyed by probe and then by detector, with the counts and a ``score`` of ``completed / total_evaluated``, or ``null`` when no output was scored.
If a report holds several utility results for one probe and detector, as an aggregate of runs does, the digest adds their counts.
A report without a utility detector has none of these entries or fields.

Confidence Intervals (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Confidence intervals are enabled by default using the bootstrap method (see ``reporting.confidence_interval_method`` in :doc:`configurable`). Eval entries include bootstrap confidence intervals for attack success rates when sample size ≥ 30:

* ``confidence``: Confidence level (e.g., "0.95")
* ``confidence_lower``: Lower bound (0-1 scale)
* ``confidence_upper``: Upper bound (0-1 scale)

These intervals account for sampling uncertainty. When detector performance metrics (sensitivity/specificity) are available, they also account for detector imperfection. Otherwise, a perfect detector is assumed.

hit@k Attack Success Rate
^^^^^^^^^^^^^^^^^^^^^^^^^^

Pooled ASR reports the fraction of *all* generations that breach the target. That understates real risk: an attacker doesn't need the target to fail most of the time, only once, and can keep retrying. A jailbreak that succeeds on 1 reply in 5 is a working jailbreak, yet pooled ASR records it as a mild 20%.

``hit@k`` reframes the metric per prompt -- "given ``k`` attempts, does at least one breach the target?" -- and averages across prompts, following the pass@k estimator of Chen et al., `Evaluating Large Language Models Trained on Code <https://arxiv.org/abs/2107.03374>`_ (2021); it is named hit@k here because garak scores an attack success as a hit. ``hit@1`` recovers the familiar per-prompt success rate; larger ``k`` shows how fast a persistent attacker's odds climb.

Every prompt is always scored at ``k`` equal to the number of generations it actually got, so a run with ``run.generations: 5`` always reports ``hit@5``: with ``k`` at the full generation count the estimator collapses to "was this prompt breached at least once". ``reporting.hit_at_k`` (default ``[1]``) adds further k values below that. Set it to ``null`` to drop the metric entirely.

Each ``eval`` entry then carries a ``hit_at_k`` field:

.. code-block:: json

   "hit_at_k": {"1": {"score": 0.2, "prompts": 5}, "5": {"score": 1.0, "prompts": 5}}

Each entry holds ``score`` (mean hit@k over prompts, on a 0-1 scale) and ``prompts`` (how many prompts were eligible). A prompt with fewer than ``k`` scoreable generations cannot be estimated for that ``k`` and is excluded from that entry; a ``k`` with no eligible prompt is omitted. Where prompts got differing numbers of generations, as in probes that end conversations early, the always-on entry is keyed ``n`` rather than an integer and covers every prompt at its own generation count.

The same figures follow the attack success rate in the CLI summary, e.g. ``hit@1: 20.00%  hit@5: 100.00%``, and are copied into each detector's entry in the ``digest`` object written to the report, next to ``total_evaluated`` and ``passed``. Note that the digest's ``absolute_score`` is a pass rate while ``hit_at_k`` scores are hit rates.

Recalculating Confidence Intervals
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For reports created before CI support or to experiment with different parameters, use the standalone ``rebuild_cis`` tool:

.. code-block:: bash

   python -m garak.analyze.rebuild_cis -r path/to/garak.uuid.report.jsonl

By default, this writes the rebuilt report to a new file (e.g. ``garak.uuid.rebuilt.report.jsonl``) without modifying the original. To overwrite the original report in-place, use the ``-w`` flag:

.. code-block:: bash

   python -m garak.analyze.rebuild_cis -r path/to/report.jsonl -w

To write to a specific output path:

.. code-block:: bash

   python -m garak.analyze.rebuild_cis -r path/to/report.jsonl -o path/to/rebuilt.jsonl

To override bootstrap config defaults:

.. code-block:: bash

   python -m garak.analyze.rebuild_cis -r report.jsonl --bootstrap_num_iterations 50000 --bootstrap_confidence_level 0.99

.. note::

   ``rebuild_cis`` updates only the JSONL report file. To regenerate the HTML report
   after recalculating CIs, run ``digest_report`` separately:

   .. code-block:: bash

      python -m garak.analyze.report_digest -r path/to/report.jsonl -o path/to/report.html

Report HTML
-----------

The report HTML presents core items from the run.
Runs are broken down into:

1. modules/taxonomy entries
2. probes within those categories
3. detectors for each probe

Results given are both absolute and relative.

During console output, attack success rates may include confidence intervals displayed as: ``(attack success rate: 45.23% [40.50%, 50.30%])``.
The bracketed values show the lower and upper bounds of the requested (default 95%) confidence interval as percentages, preserving the asymmetry of the bootstrap distribution.
The relative ones are in terms of a Z-score computed against a set of recently tested other models and systems.
For Z-scores, 0 is average, negative is worse, positive is better.
Both absolute and relative scores are placed into one of five grades, ranging from 1 (worst) to 5 (best).
This scale follows the NORAD DEFCON categorisation (with less dire consequences).
Bounds for these categories are developed over many runs.
The absolute scores are only alarming or reassuring for very poor or very good Z-scores.
The relative scores assume the middle 10% is average, the bottom 15% is terrible, and the top 15% is great.

DEFCON scores are aggregated using a minimum, to avoid obscuring important failures.

.. toctree::
   :maxdepth: 2

   reporting.calibration