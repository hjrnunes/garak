garak.detectors.toolcall
========================

Helpers for reading the tool calls a target emitted.

Generators capture the calls a target made into ``Message.notes["tool_calls"]``,
in the Responses API shape (``function_call`` and ``mcp_call`` items,
``arguments`` and ``output`` as JSON strings) or the chat-completions shape
(``function.name``, ``function.arguments``). A Responses entry holds its
arguments in ``arguments`` or, failing that, for an ``mcp_call``, in ``input``. The
``input`` of any other call, such as a custom tool's freeform text, is not an
arguments object and is not read. The helpers here
decode and normalise both shapes into one call dict, keep a boolean ``blocked``
mark, and read the capture off an output together with its completeness flag, so
a detector can judge tool use without a model. A capture is incomplete when the
response did not finish or holds an entry that is not an object.

.. automodule:: garak.detectors.toolcall
   :members:
   :undoc-members:
   :show-inheritance:
