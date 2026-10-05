garak.detectors.rubric_judge
============================

A strict yes/no semantic judge over a target's captured reply.

``RubricJudge`` reads a rubric carried on each attempt
(``attempt.prompt.notes["judge_rubric"]``, set for example by
``probes.injection.IndirectInjection``) with a ``question``, ``criteria``,
``facts``, and ``runtime_facts``. For every output it builds an evidence view
holding the last user turn, the earlier prompt turns, the reply text, and the
tool calls the generator captured (normalised as in ``detectors.toolcall``),
and sends one JSON request to the judge model.

The judge returns a verdict (``supported``, ``contradicted``, or
``unresolved``) and the evidence paths it relied on. A decisive verdict is kept
only when every path resolves to captured reply text or a non-null value at or
below a tool-call result field (for example ``tool_calls[0].result`` or
``tool_calls[0].result.status``); otherwise it becomes ``unresolved``. The detector scores 1.0 for
``supported``, 0.0 for ``contradicted``, and None otherwise, including when
the attempt has no rubric, the rubric is invalid, or the judge call fails. One
record per output, with the verdict, the paths, a reason, and the request and
raw judge response, is written to ``attempt.notes["rubric_judge_results"]``.

The judge model is instantiated via the generator interface and must inherit
``OpenAICompatible``. Requests use temperature 0, at most 512 completion
tokens, a JSON-object response format, and disable thinking through
``chat_template_kwargs``.

.. automodule:: garak.detectors.rubric_judge
   :members:
   :undoc-members:
   :show-inheritance:
