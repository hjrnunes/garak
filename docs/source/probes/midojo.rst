garak.probes.midojo
===================

Run MiDojo injection-spec documents as serial, multi-turn chains on the run
coordinator. The probe reads JSON or JSONL entries with an ``injection_spec``
and optional ``metadata``. The document's plan holds the authored payload; the
probe never rewrites it. A missing, malformed or unsupported schema version is
refused before any chain opens; the refusal names the supported versions and
truncates an unsupported version's representation to 99 characters.
Each plan instruction needs a non-empty UTF-8 string payload of at most
8192 bytes. The cap bounds a hostile document at this route's entry point;
the refusal identifies the instruction by its index without printing content.

Components and install
----------------------

The probe opens one chain scope per entry on the run coordinator (see
:doc:`../coordinator`). ``plugins.injector_type`` selects the injector that
places the payload in tool output (default ``mim``, see
:doc:`../injectors/mim`). ``plugins.environment_type`` selects the environment
that hosts the target (default ``unmanaged``). Garak records the chain in its
own collector, seals it, and grades the sealed snapshot with
``garak.detectors.grading``. No MiDojo runtime is installed or imported.

Install the ``midojo`` extra for the ``midojo-spec`` contracts. A run through
the shipped Mim injector also needs the ``midojo-mim`` extra. The
``midojo-runtime`` extra no longer exists.

The target needs a Responses-style generator that takes MCP tools from
``Conversation.notes["tools"]`` and returns a ``response_status`` and any
``tool_calls`` in each output's notes. A generator that cannot mark the
response complete ends the chain as incomplete rather than resisted. The
probe sends each turn after the preceding real response, runs chains serially
and releases each chain on sealing or interruption.

The final attempt's ``notes["injection_spec_chain"]`` identifies its chain,
turn and final state; ``notes["source_metadata"]`` carries opaque metadata.
``notes["evaluation_trace"]`` contains the sealed snapshot, its IDs, grade,
delivery and exposure, per-call errors, capture status, ``release_error`` and
``cleanup_failures``. ``cleanup_failures`` lists the exception type names of
the errors that closing the chain raised, and a release the injector never
acknowledged also sets ``release_error``. Apart from ``cleanup_failures``, the
trace keeps the keys and value shapes the earlier route wrote. Its output's
``tool_calls`` contains the interceptor's records; the generator's original
records remain in ``generator_tool_calls``. The trace is graded by
``toolcall.ToolCallCondition``, ``midojo.EvaluationVerdict`` and
``midojo.EvaluationEffect``.

Options
-------

.. csv-table:: Probe options
   :header: "Option", "Default", "Meaning"
   :widths: 22, 14, 64

   "``source_path``", "shipped example", "The injection-spec file."
   "``upstream``", "none", "The HTTP(S) URL the injector forwards tool calls to, with a host and no user info; ports, paths and queries are accepted. It takes precedence over the document's ``interception.upstream``. A run with neither is refused."
   "``repeats``", "1", "Chains per entry. A value above 1 needs an environment that isolates attempts and resets between them. The shipped ``unmanaged`` environment does neither, so the run is refused when it starts."
   "``chain_timeout``", "off", "Seconds for a whole chain: a finite positive number, or null. A chain that runs longer closes and records ``ChainTimeout``."
   "``turn_timeout``", "60", "Seconds for one turn: a finite positive number. The coordinator enforces it and records ``TurnTimeout``."
   "``end_condition``, ``follow_prompt_cap``", "``verify``, false", "As in the iterative base probe."

Each option is refused without echoing its value when it holds the wrong kind
of value.

Moving from the earlier options
-------------------------------

The probe no longer reads ``interceptor_url`` or ``interceptor_credential``,
and it has no alias for either. A configuration that sets one of them fails
when the probe loads. The message names every moved option the configuration
holds and the key that replaces it, and it never prints a value. On the command
line, the console says only that the probe failed to load, and ``garak.log``
has the full message.

.. csv-table:: Earlier probe options and the keys that replace them
   :header: "Earlier setting", "Now", "Change"
   :widths: 30, 34, 36

   "``probes.midojo.InjectionSpecProbe.source_path``", "same", "None."
   "``probes.midojo.InjectionSpecProbe.upstream``", "same", "None for an HTTP(S) string. A non-string transport object is refused, because a chain takes a URL."
   "``probes.midojo.InjectionSpecProbe.repeats``", "same", "A value above 1 needs an isolating environment, selected with ``plugins.environment_type``."
   "``probes.midojo.InjectionSpecProbe.end_condition``, ``follow_prompt_cap``", "same", "None."
   "``probes.midojo.InjectionSpecProbe.interceptor_url``", "``plugins.injectors.mim.service_url``", "Moved. It accepts ``http`` or ``https`` only: write ``http`` for ``ws`` and ``https`` for ``wss``. It accepts a base URL or one that ends in ``/v1/bindings``, with an optional proxy path prefix. User info, a query and a fragment are refused."
   "``probes.midojo.InjectionSpecProbe.interceptor_credential``", "``plugins.injectors.mim.bind_credential``", "Moved. The scrubbing that covered the old key covers the new one."
   "``MIDOJO_INTERCEPTOR_URL``, ``MIDOJO_INTERCEPTOR_CREDENTIAL``", "same", "The Mim injector reads them. Both or neither."
   "none: interception inside the MiDojo runtime", "none: a loopback service the Mim injector starts", "The behaviour differs and needs the ``midojo-mim`` extra."
   "none", "``plugins.injector_type``", "Optional. The probe requests ``mim`` when it is unset."
   "none", "``plugins.environment_type``", "Optional. The default is ``unmanaged``."
   "none", "``probes.midojo.InjectionSpecProbe.chain_timeout``", "New. Seconds for a whole chain; off by default."
   "none", "``probes.midojo.InjectionSpecProbe.turn_timeout``", "New. Seconds for one turn; the default is 60."
   "none", "``plugins.injectors.mim.bind_timeout``, ``release_timeout``", "Optional bounds of the injector; the Mim page describes them."

A key under ``plugins.injectors.mim.<key>`` applies to the class, and
``plugins.injectors.mim.Mim.<key>`` overrides it.

Behaviour that differs from the earlier route:

* A half-set pair, a URL without a credential or the reverse, is refused by the
  injector with a capability error when the run starts, and not by the probe
  when it loads. Both refuse before the first chain opens, and neither prints a
  value. The injector requires only that the pair is complete: the probe no
  longer checks a credential's content.
* A refusal that stops the first chain from opening, such as a service that
  refuses the binding, is raised from the probe's run. The iterative base class
  only logs an error raised inside its loop, so the probe raises it when no
  chain produced an attempt. A run from the command line reports it in
  ``garak.log`` and still exits with status 0. The text of an injector's or an
  environment's refusal reaches the message with every credential the probe
  holds masked.
* The command line masks ``bind_credential`` and the user info of
  ``service_url`` as it masks ``upstream``. It still masks ``interceptor_credential``
  and ``interceptor_url``, so an old configuration leaks nothing before the
  probe refuses it.
* The trace's ``injection_outcomes`` keep the fields of MiDojo's sealed
  ``CallInjectionOutcome``. ``scope``, ``call_id``, ``tool`` and
  ``admission_order`` stay on the interceptor's full outcome, which
  ``Arming.disarm()`` returns and the service's ``unbound`` frame carries. The
  collector refuses them, because a snapshot that held them would not digest
  as the one MiDojo seals for the same evidence.

Credentials
-----------

A credential garak holds, which is the tool block's bearer, the injector's
``bind_credential`` or the generator's API key, does not appear in a
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
* The probe reads the injector's credential and the environment credential
  before and after a chain opens. A credential that another thread changes
  between the two reads is out of reach.

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
   "Failure, capture status and closure error type names", "trace, output notes", "non-secret by construction"
   "Detector outcomes, indices, grades and reasons", "detector records, console, summary", "built from the scrubbed trace and fixed codes"
   "Serialised output in the report and hit log", "report, hit log", "scrubbed upstream of the writers, except the model's own text"
   "Request URL after a redirect", "post-request warning log", "scrubbed"

A separately managed interceptor service needs both
``plugins.injectors.mim.service_url`` and
``plugins.injectors.mim.bind_credential``, or the environment pair. The
service URL must be an HTTP(S) URL with a host and without user info, a query
or a fragment. Paths and proxy prefixes are accepted. The probe's ``upstream``
needs an HTTP(S) URL with a host and no user info; ports, paths and queries are
accepted. Invalid options are refused without echoing their values, and
unreadable source files produce value-free configuration errors. A lost
release is recorded as ``release_error`` after sealing and does not overturn
the sealed verdict.

.. automodule:: garak.probes.midojo
   :members:
   :undoc-members:
   :show-inheritance:

   .. show-asr::
