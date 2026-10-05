garak.probes.injection
=======================

Indirect prompt injection probes.

Replays pre-built conversations (authored offline in OpenAI chat-completion
format) to a target unchanged. The conversations may contain tool calls and
tool-call results, so a harmful payload can be injected anywhere -- including as
the result of a tool call.

Each conversation is loaded from a JSON/JSONL source and may carry an optional
``tools`` schema (forwarded to the target so it can emit new tool calls) and a
natural-language ``judge_description`` describing what a successful attack looks
like. The description is consumed by ``detectors.injection_judge.InjectionJudge``.
An optional ``tool_call_condition`` states which tool calls mean the attack
succeeded; ``detectors.toolcall.ToolCallCondition`` (an extended detector)
evaluates it deterministically.

.. automodule:: garak.probes.injection
   :members:
   :undoc-members:
   :show-inheritance:

   .. show-asr::
