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
  (optionally on the same argument value, compared with the same equality as
  ``value``; an earlier call that lacks the argument never counts).

Conditions are evaluated under the tool-call-condition-v2 contract. A captured
call may be marked ``blocked`` (the target emitted it but it did not run, for
example because a guard refused it): a blocked call never counts as the earlier
call an ``order`` requires, but it can still be the call a ``value`` or
``order`` comparison matches, and it still makes ``not_called`` fail.

Evaluation is three-valued. Absence is only proven by a complete capture: under
an incomplete one, a ``not_called`` with no matching call, a missing call, and
an ``order`` with no earlier call are all unknown. The detector scores 1.0 when the condition holds,
0.0 when it does not, and None when the capture is incomplete, a relevant call
cannot be decoded, the condition is invalid, or the attempt has no condition.
One record per output, with the outcome, a reason, and the indices of matched
calls, is written to ``attempt.notes["tool_call_condition_results"]``.

.. automodule:: garak.detectors.toolcall
   :members:
   :undoc-members:
   :show-inheritance:
