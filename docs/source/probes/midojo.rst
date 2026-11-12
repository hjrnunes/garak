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
tool calls. A credential garak holds, which is the tool block's bearer,
``interceptor_credential`` or the generator's API key, does not appear in a
report, conversation notes, the console or ``garak.log`` through the 26 paths
in the table below when an API server or a tool server echoes it back. The
echo can come in a body, an error, a header, a reason phrase, a redirect URL, a
call ID or the trace. The Responses generator and the probe scrub every such
value. The guarantee is best effort against a server that misbehaves on
purpose, and it does not extend to code garak does not control. Four limits
follow:

* The model's own text is its own output and is not scrubbed. Other generators
  are outside the probe's control for the same reason.
* A scrub matches a credential as written or as a JSON string escapes it. A
  server that re-encodes a credential before it echoes it is out of reach.
* A log handler you add while a request runs is not covered, because it sees
  that request's records as the client wrote them. A handler that exists when
  the request starts is covered, whichever logger it sits on.
* The probe reads the environment credential before and after a chain opens. A
  credential that another thread changes between the two reads is out of reach.

.. csv-table:: Server-origin values on the Responses route
   :header: "Value", "Lands in", "Treatment"
   :widths: 34, 36, 30

   "Response headers and request ID", "client and HTTP library DEBUG records", "scrubbed by the log filter"
   "Response reason phrase", "client and HTTP library records", "scrubbed by the log filter"
   "Status code and retry-control headers", "retry decisions, error note", "non-secret by construction"
   "Request options and earlier output items", "DEBUG log", "scrubbed by the client filter"
   "Authentication or permission error text", "exception, logs", "scrubbed"
   "Bad-request or non-retryable error text and body", "logs, error note, report", "scrubbed"
   "Exception type and status code", "error note, trace error", "non-secret by construction"
   "Retryable error body and its cause", "the exception's cause, in memory", "in memory only: the retry loop has no try limit, so no handler logs it"
   "JSON decode failure", "exception log", "non-secret by construction"
   "Response ID and model", "output notes, report", "scrubbed"
   "Response usage, including unknown fields and keys", "output notes, report", "scrubbed"
   "Response error and incomplete details", "output notes, report", "scrubbed"
   "Response status string", "output notes, report", "scrubbed"
   "Tool-call fields, including ID and call ID", "output notes, report", "scrubbed"
   "Whole tool-call item, with unknown fields and keys", "output notes, report, later requests", "scrubbed"
   "Model output text", "output text, report, hit log, later turns", "model-authored"
   "Reasoning summary", "output notes, report", "scrubbed"
   "Tool-block authorization", "the live request, and the chain's record of it in memory", "scrubbed copy on the report; the record is never written"
   "Snapshot, tool-call arguments, results and errors", "trace, output notes, report, hit log", "scrubbed in the trace"
   "Verdict, delivery, exposure and capture details", "trace, detector records, report", "scrubbed in the trace"
   "Evaluation and snapshot IDs", "trace, report", "non-secret by construction"
   "Close-error reason", "trace release error", "scrubbed in the trace"
   "Failure and capture status names", "trace, output notes", "non-secret by construction"
   "Detector outcomes, indices, grades and reasons", "detector records, console, summary", "built from the scrubbed trace and fixed codes"
   "Serialised output in the report and hit log", "report, hit log", "scrubbed upstream of the writers, except the model's own text"
   "Request URL after a redirect", "post-request warning log", "scrubbed"

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
