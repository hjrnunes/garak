garak.probes.injection
=======================

Conversation replay probes.

Replays pre-built conversations (authored offline in OpenAI chat-completion
format, with system, user and assistant messages) to a target unchanged.

Each conversation is loaded from a JSON/JSONL source and may carry an optional
``tools`` schema (forwarded to the target so it can call tools). An optional
``tool_call_condition`` states which tool calls mean the attack succeeded;
``detectors.toolcall.ToolCallCondition`` (the primary detector) evaluates it
deterministically. An optional ``judge_rubric`` (a yes/no ``question`` with
``criteria``, ``facts``, and ``runtime_facts``) is answered by
``detectors.rubric_judge.RubricJudge``, an extended detector.

.. automodule:: garak.probes.injection
   :members:
   :undoc-members:
   :show-inheritance:

   .. show-asr::
