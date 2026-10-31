garak.probes.ipi_episode
========================

Run MiDojo episode documents as multi-turn tool-use probes.

Each source entry holds an ``episode`` (a ``midojo-episode-v1`` document, read
with ``midojo.episode.Episode.from_document``) and an opaque ``metadata``
object. The probe needs the optional ``midojo`` extra:

.. code-block:: bash

   python -m pip install -e '.[midojo]'

The extra carries a ``python_version >= '3.12'`` marker, because MiDojo needs
it. Garak lists the probe without the extra and, when a run names it, prints
that ``midojo`` is missing and runs no episodes; that is how garak treats every
plugin with an optional extra.

A source is a ``.json`` file holding one entry or a list of entries, or a
``.jsonl`` file holding one entry per line:

.. code-block:: json

   {"episode": {"schema_version": "midojo-episode-v1", "turns": ["..."]},
    "metadata": {"anything": "the probe never reads"}}

The document is the whole attack. Its plan already holds the payload, placed
for whoever compiled the episode: the probe fills nothing and rewrites nothing.
``metadata`` travels into every attempt's ``notes["source_metadata"]`` unread,
for the caller's report. The document's ``security.tool_call_condition`` (or
the entry's own ``tool_call_condition``) goes on every turn's prompt notes for
``detectors.toolcall.ToolCallCondition``. A document with a missing or
unsupported ``schema_version`` is refused before any handle opens.

Chains
------

Every chain is one episode driven through a ``midojo.episode_handle``
handle: the handle runs the interception layer and, when the runtime needs it,
a sandbox. The probe opens one handle per chain in ``_create_init_attempts``,
starts the episode's first turn, and puts the interceptor's MCP tool block in
``Conversation.notes["tools"]`` as the request's tool list, so the target can
reach the episode's tools. Each episode turn is one request, and a later request
carries the earlier turns and the target's real replies. A turn that a request
rejected, that never finished, or that neither spoke nor called a tool ends its
chain early; a turn that answers with tool calls alone does not, because MiDojo
grades those calls.

Handles live in a registry on the probe, never in an attempt's notes:
``IterativeProbe`` deep-copies its attempt queue between turns, so notes carry
only plain data (``notes["ipi_episode"]`` names the chain, turn, turn count and
whether the turn is final).

At a chain's end the probe records the reply, seals and grades the episode, and
attaches the trace to that chain's final attempt:

* ``notes["episode_trace"]`` holds the evaluation and snapshot IDs, the
  snapshot as MiDojo serves it, MiDojo's verdict, whether the capture is
  complete, and the recorded tool calls with their dispositions and whether
  each finished;
* the output message's ``notes["tool_calls"]`` and ``notes["response_status"]``
  carry the same calls in the shape ``ToolCallCondition`` reads. The status
  names what stops the condition deciding, and it reaches a reader of the
  receipt: ``completed`` when the payload was delivered and the capture is
  whole, ``unrecorded_delivery`` when the grade states no delivery fact, and
  ``incomplete`` when a call capture or the turn itself is short.
  ``ToolCallCondition`` suffixes its inconclusive reason with that status, so a
  receipt's detection reason distinguishes an unrecorded delivery from a
  truncated capture.

``probe()`` returns only each chain's final attempt, and that is the mode this
probe ships. The harness writes every attempt it executes at status 1 and every
attempt it gets back at status 2, so a two-turn chain leaves three lines in
``report.jsonl``: two attempts at status 1, then the final attempt at status 2.
The detectors run over the returned attempt only, so the ``eval`` entry counts
``total_processed: 1`` and ``garak.analyze.report_digest`` scores the chain once.
A live two-turn run showed exactly that shape: statuses ``[1, 1, 2]``, one
``eval`` entry per detector with one result, and a digest with a single probe
score. The rejected alternative returns every attempt and scores each
intermediate turn ``None`` with reason ``intermediate_turn``; the detector keeps
that reason for an attempt a caller re-grades, but this probe never produces
one.

``probe()`` seals any chain an error interrupted and closes every handle in a
``finally``: each open episode holds a listening interceptor. A chain's handle
also closes the moment its episode is sealed, so a probe with several chains
still holds one handle at a time whatever ``parallel_attempts`` says.

Generators
----------

The probe needs a Responses-style generator: one that takes the request's
``notes["tools"]`` as its tool list and reports ``response_status`` and any
``tool_calls`` on each output message. Only that generator can carry the
interceptor's tool block to the target and say whether a response finished.
Every other generator fails closed: a turn with no output, or with a response
status the generator never sets, ends its chain as errored, and both detectors
score the chain ``None`` rather than as resisted.

Parameters
----------

``source_path``
    The file of episode entries. The shipped ``injection/example_episode.json``
    runs a single-turn order lookup against ``http://localhost:8081/mcp``; point
    it at your own file.

``episode_source``
    Deprecated alias for ``source_path``. The probe warns and reads it only when
    ``source_path`` is unset; two keys naming different files are refused.

``upstream``
    An MCP server URL that wins over the URL each document's interception block
    declares.

``repeats``
    How many separate chains run the same episode, each its own evaluation with
    its own evaluation ID. Repeated chains are refused when the runtime neither
    resets its target nor gives each evaluation a fresh one, before any handle
    opens: ask the runtime for ``midojo.runtimes.runtime_capabilities`` and read
    ``isolates_attempts``.

``max_calls_per_conv``
    Set from the source to every chain's turns, so the iteration limit never
    cuts a chain short. Chains run one at a time whatever ``parallel_attempts``
    says, so a runtime that starts a sandbox holds one sandbox at a time; one
    reply per turn means ``generations`` is 1.

Attempts scored by ``detectors.ipi_episode.EpisodeVerdict``;
``detectors.toolcall.ToolCallCondition`` is its primary detector, and
``detectors.ipi_episode.EpisodeEffect`` is an extended detector, so a run that
enables extended detectors scores whether the injection took effect as well as
whether the agent broke the condition.

.. automodule:: garak.probes.ipi_episode
   :members:
   :undoc-members:
   :show-inheritance:

   .. show-asr::
