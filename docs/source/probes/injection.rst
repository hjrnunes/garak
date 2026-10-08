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
evaluates it deterministically. An optional ``judge_rubric`` (a yes/no
``question`` with ``criteria``, ``facts``, and ``runtime_facts``) is answered by
``detectors.rubric_judge.RubricJudge``, also an extended detector.

``SequentialInjection`` reads the same entries when their messages are all user
turns. It sends one user turn per request, as an ``IterativeProbe``: each later
request carries the earlier user turns, the ``mcp_call`` items of each earlier
response, and the target's real replies. The first failed turn ends the
conversation, and a failed conversation is not graded. The tool-call condition
is checked after every turn; ``judge_turns`` (``final`` or ``all``) chooses the
turns the judges grade.

.. automodule:: garak.probes.injection
   :members:
   :undoc-members:
   :show-inheritance:

   .. show-asr::
