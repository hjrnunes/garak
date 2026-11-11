garak.detectors.toolcall
========================

Deterministic detectors over the tool calls a target emitted.

``ToolCallCondition`` evaluates a declarative condition carried on each attempt
(``attempt.prompt.notes["tool_call_condition"]``, set for example by
``probes.injection.IndirectInjection``) against the tool calls the generator
captured in ``Message.notes["tool_calls"]``. Both the Responses API and the
chat-completions capture shapes are read. A Responses entry holds its arguments
in ``arguments`` or, failing that, for an ``mcp_call``, in ``input``. The
``input`` of any other call, such as a custom tool's freeform text, is not an
arguments object and is not read.

A condition is a list of comparisons that must all hold:

* ``value`` compares a tool-call argument with a literal or another argument
  (``eq``, ``ne``, ``gt``, ``ge``, ``lt``, ``le``, ``in``, ``not_in``);
* ``not_called`` holds when no call to a tool (optionally with given argument
  values) was made;
* ``order`` holds for a call made without an earlier call to a required tool
  (optionally on the same argument value, compared with the same equality as
  ``value``; an earlier call that lacks the argument never counts).

When two numbers are compared directly, an integer beyond float range counts
as an infinity, in ``same_argument`` too, so two such integers of the same sign
are equal. Inside lists and objects, integers are compared exactly.

Conditions are evaluated under the ``tool-call-condition-v1`` contract. A
captured call may be marked ``blocked`` (the target emitted it but it did not
run, for example because a guard refused it): a blocked call never counts as
the earlier call an ``order`` requires, but it can still be the call a ``value``
or ``order`` comparison matches, and it still makes ``not_called`` fail.

Evaluation is three-valued. Absence is only proven by a complete capture: under
an incomplete one, a ``not_called`` with no matching call, a missing call, and
an ``order`` with no earlier call are all unknown. A capture is incomplete when
the response did not finish (``response_status`` other than ``completed``) or
holds an entry that is not an object. A call that is present still decides a
comparison under an incomplete capture.

The detector scores 1.0 when the condition holds and 0.0 when it does not. It
scores None when the evidence cannot decide: an incomplete capture leaves the
outcome open, a relevant call cannot be decoded, the capture is not readable
(the output is not a message, its notes are not an object or ``tool_calls`` is
not a list), the condition is invalid, or the attempt has no condition. A
condition or a call's arguments that hold a value no JSON document holds, such
as a cyclic list or a byte string, are unreadable, so the condition is invalid
and the call's arguments are unusable. Arguments that are a JSON string are
decoded; text Python cannot parse, such as an integer of more than 4300 digits,
counts as undecodable.

One record per output, with the outcome, a reason, and the indices of matched
calls, is written to ``attempt.notes["tool_call_condition_results"]``.

.. automodule:: garak.detectors.toolcall
   :members:
   :undoc-members:
   :show-inheritance:
