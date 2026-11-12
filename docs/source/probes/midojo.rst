garak.probes.midojo
===================

Drive a MiDojo injection-spec document through one evaluation handle per
chain. The probe reads JSON or JSONL entries with an ``injection_spec`` and
optional ``metadata``. The document's plan holds the authored payload; the
probe never rewrites it. A missing, malformed or unsupported schema version is
refused before any handle opens; the refusal names the harness's supported
versions and truncates an unsupported version's representation to 99 characters.
Each plan instruction needs a non-empty UTF-8 string payload of at most
8192 bytes. The cap bounds a hostile document at this route's entry point;
the refusal identifies the instruction by its index without printing content.

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

The probe gives the generator the live per-binding tool token so it can make
tool calls. The Responses generator scrubs that token from the errors it
records. Other generators are outside the probe's control: anything a
generator emits is its own output, which the probe does not scrub.

The tool block's bearer credential and ``interceptor_credential`` must never
appear in serialisable conversation notes, the trace, a report or DEBUG logs.
A remote interceptor needs both ``interceptor_url`` and
``interceptor_credential``. The credential must be a non-empty string with
non-whitespace content and no control characters. The URL must be an HTTP(S)
or WS(S) URL without user info, a query or a fragment. Paths and proxy
prefixes are accepted. Invalid options are refused without echoing their values.
Alternatively the harness reads its environment pair. A lost release is
recorded as ``release_error`` after sealing and does not overturn the sealed
verdict.

The environment pair follows the same URL and credential admission rules as
explicit interceptor options. A string ``upstream`` needs an HTTP(S) URL with
a host and no user info; ports, paths and queries are accepted. Non-string
upstream transport objects remain available for in-process interception.
Unreadable source files and invalid handle-open options produce value-free
configuration errors.

.. automodule:: garak.probes.midojo
   :members:
   :undoc-members:
   :show-inheritance:

   .. show-asr::
