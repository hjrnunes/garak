garak.detectors.toolcall
========================

Helpers and a condition language over the tool calls a target emitted.

Generators capture the calls a target made into ``Message.notes["tool_calls"]``,
in the Responses API shape (``function_call`` and ``mcp_call`` items,
``arguments`` and ``output`` as JSON strings) or the chat-completions shape
(``function.name``, ``function.arguments``). A Responses entry holds its
arguments in ``arguments`` or, failing that, for an ``mcp_call``, in ``input``. The
``input`` of any other call, such as a custom tool's freeform text, is not an
arguments object and is not read. The helpers here
decode and normalise both shapes into one call dict, keep a boolean ``blocked``
mark, and read the capture off an output together with its completeness flag. A
capture is incomplete when the response did not finish or holds an entry that is
not an object.

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

Conditions follow the ``tool-call-condition-v1`` contract. A captured call may
be marked ``blocked`` (the target emitted it but it did not run, for example
because a guard refused it): a blocked call never counts as the earlier call
an ``order`` requires, but it can still be the call a ``value`` or ``order``
comparison matches, and it still makes ``not_called`` fail.

Evaluation is three-valued. Absence is only proven by a complete capture:
under an incomplete one, a ``not_called`` with no matching call, a missing
call, and an ``order`` with no earlier call are all unknown.

.. automodule:: garak.detectors.toolcall
   :members:
   :undoc-members:
   :show-inheritance:
