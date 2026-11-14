Rubric judge
============

``rubric_judge.RubricJudge`` answers the yes/no question in
``attempt.prompt.notes["judge_rubric"]`` for each captured reply. A rubric has a
non-blank ``question`` string and may include ``criteria``, ``facts`` and
``runtime_facts``. A blank or whitespace-only question is invalid on every
backend.
The detector accepts a decisive answer only when the judge cites captured
message content or a tool result. It writes ``rubric_judge_results`` in output
order; each record contains ``verdict``, ``evidence_refs``, ``reason``,
``prompt_version``, ``request``, ``raw_response``, ``model`` and ``prompt``.
Supported scores 1.0, contradicted scores 0.0, and unresolved scores ``None``.

Install the ``midojo`` extra for the shared judge rules. Choose a backend with
``backend`` in the detector's configuration:

* ``native`` loads a chat-completions OpenAI-compatible evaluation generator
  from ``detector_model_type`` and ``detector_model_name``. Other resources,
  including Responses, are not supported. The detector calls the configured
  generator, but allows at most one chat-completions request, even if that
  generator retries. The SDK has no retries. A transient failure can incur
  one built-in backoff sleep of at most one second before a second request
  is blocked. A failed request leaves the verdict unresolved with
  ``judge_failed:`` and the error type as the reason, also when the generator
  sets ``max_retries``. Only the shared judge request body reaches the provider, not
  optional generator sampling settings. Configure provider credentials
  through the generator, not the rubric.
  Native judgements in one process run one at a time: the module-wide guard
  holds its lock across each request, so concurrent calls wait. A judgement
  started from inside another on the same thread is refused without a request
  (``judge_failed:ReentrantJudgeCall``).
* ``service`` needs ``service_url``, ``judge_model`` and ``service_token_env``.
  Set the named environment variable to the service's bearer token at runtime.
  The service owns its model credentials; do not put either secret in the
  detector configuration. Evidence that the service request cannot reproduce
  remains unresolved without a network call.
* ``replay`` needs ``replay_path`` (a judge exchange JSONL log) and
  ``judge_model``. It makes no network or generator requests. An exchange with
  the wrong judge identity, a conflicting response, or a missing request
  remains unresolved.

The service summary has no model reply, so ``raw_response`` is ``None`` for
service records. The shared judge rule reads a reply enclosed in one Markdown
code fence the same way on all three backends; native and replay also retain
the original reply in ``raw_response``.
