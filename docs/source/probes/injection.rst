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
``detectors.rubric_judge.RubricJudge``, an extended detector. An attempt whose
request the target rejected, or whose response did not finish, keeps its error
note and loses the condition and the rubric, so no detector gives it a verdict.

``SequentialInjection`` reads the same entries when their messages are all user
turns. It sends one user turn per request, as an ``IterativeProbe``: each later
request carries the earlier user turns, the ``mcp_call`` items of each earlier
response, and the target's real replies. The target must write a
``response_status`` note, as ``OpenAIResponsesGenerator`` does: a reply
without one counts as unfinished, so every conversation ends after its first
turn. The first failed turn ends the conversation, and a failed conversation
is not graded. A turn fails when the
request is rejected, the response does not finish, or the reply has no text
and made no tool call. The tool-call condition
is checked after every turn; ``judge_turns`` (``final`` or ``all``) chooses the
turns the judges grade.

.. automodule:: garak.probes.injection
   :members:
   :undoc-members:
   :show-inheritance:

   .. show-asr::
