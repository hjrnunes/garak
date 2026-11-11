garak.probes.midojo
===================

Drive a MiDojo injection-spec document through one evaluation handle per
chain. The probe reads JSON or JSONL entries with an ``injection_spec`` and
optional ``metadata``. The document's plan holds the authored payload; the
probe never rewrites it. A missing, malformed or unsupported schema version is
refused before any handle opens; the refusal names the harness's supported
versions and truncates an unsupported version's representation to 99 characters.

The ``midojo-runtime`` extra is temporary while this route moves to the
coordinator and components. **Do not install it from the package index.**
The plain ``midojo`` name there belongs to an unrelated package. Pin a trusted
MiDojo harness source, including its SDK, interceptor and spec packages,
before installing the extra. The ``midojo`` extra supplies only the lightweight
``midojo-spec`` contracts and remains separate.

The target needs a Responses-style generator that takes MCP tools from
``Conversation.notes["tools"]`` and returns a ``response_status`` and any
``tool_calls`` in each output's notes. A generator that cannot mark the
response complete ends the chain as incomplete rather than resisted. The
probe sends each turn after the preceding real response, runs chains serially
and closes each handle on sealing or interruption.

The final attempt's ``notes["injection_spec_chain"]`` identifies its chain,
turn and final state; ``notes["source_metadata"]`` carries opaque metadata.
``notes["evaluation_trace"]`` contains the sealed snapshot, its IDs, grade,
delivery and exposure, per-call errors, capture status and any release error.
The grade comes from the MiDojo runtime's evaluation handle until this route
moves to the coordinator and components.
Its output's ``tool_calls`` contains the interceptor's records; the generator's
original records remain in ``generator_tool_calls``. The trace is graded by
``toolcall.ToolCallCondition``, ``midojo.EvaluationVerdict`` and
``midojo.EvaluationEffect``.

The tool block's bearer credential and ``interceptor_credential`` must never
appear in serialisable conversation notes, the trace, a report or DEBUG logs.
A remote interceptor needs both ``interceptor_url`` and
``interceptor_credential``; alternatively the harness reads its environment
pair. A lost release is recorded as ``release_error`` after sealing and does
not overturn the sealed verdict.

.. automodule:: garak.probes.midojo
   :members:
   :undoc-members:
   :show-inheritance:

   .. show-asr::
