garak.detectors.toolcall
========================

Deterministic detectors over the tool calls a target emitted.

``ToolCallCondition`` evaluates a declarative condition carried on each attempt
(``attempt.prompt.notes["tool_call_condition"]``, set for example by
``probes.injection.IndirectInjection``) against the tool calls the generator
captured in ``Message.notes["tool_calls"]``. Both the Responses API and the
chat-completions capture shapes are read.

A condition is a list of comparisons that must all hold:

* ``value`` compares a tool-call argument with a literal or another argument
  (``eq``, ``ne``, ``gt``, ``ge``, ``lt``, ``le``, ``in``, ``not_in``);
* ``not_called`` holds when no call to a tool (optionally with given argument
  values) was made;
* ``order`` holds for a call made without an earlier call to a required tool
  (optionally on the same argument value).

Evaluation is three-valued. The detector scores 1.0 when the condition holds,
0.0 when it does not, and None when the capture is incomplete, a relevant call
cannot be decoded, the condition is invalid, or the attempt has no condition.
One record per output, with the outcome, a reason, and the indices of matched
calls, is written to ``attempt.notes["tool_call_condition_results"]``.

.. automodule:: garak.detectors.toolcall
   :members:
   :undoc-members:
   :show-inheritance:
