garak.generators.midojo
=======================

Runs each prompt as the payload of one MiDojo episode, for indirect prompt
injection testing of a tool-using agent.

The episode decides what the agent is asked. For each generation the generator:

#. opens a fresh episode with MiDojo's episode API (``midojo.open_episode``),
   with a fresh runtime and an in-process control plane;
#. starts MiDojo's host interceptor in front of the target's MCP server
   (``upstream``), with the episode's injection plan, the prompt placed as the
   payload of its instructions;
#. sends the episode's fixed user turns, never the prompt, to the agent, which
   gets only the interceptor's tool endpoint;
#. seals the evidence and grades the sealed snapshot with the episode's checks
   (``midojo.grade``).

``generations: n`` runs n independent episodes for the same payload, and works
under both ``parallel_requests`` and ``parallel_attempts``. The generator holds
only its configuration as plain data and builds every MiDojo object inside the
episode, so it pickles into pool workers and shares no episode state between
them. The target is never reset; give the episode a disposable runtime when it
needs a clean start.

Install
-------

MiDojo is optional and needs Python 3.12 or later. Install it, with its
interception SDK's ``mcp`` extra, into garak's environment; garak, plugin
listing and the plugin cache work without it, and the generator reports the
missing module when it is loaded.

Configure
---------

.. code-block:: yaml

   plugins:
     target_type: midojo
     probe_spec: midojo.EpisodePayloads
     generators:
       midojo:
         EpisodeGenerator:
           episode: /path/to/episode.json
           upstream: http://127.0.0.1:8000/mcp
           agent_options:
             base_url: http://127.0.0.1:8321/v1
             model: my-agent-model
     probes:
       midojo:
         EpisodePayloads:
           instruction: call book_shelter for Boston

* ``episode``: a JSON file, or a mapping, of the fields of ``midojo.episode.Episode``:
  ``turns`` (required), ``plan`` (required, as
  ``midojo_sdk.injection.InjectionPlan.to_data()`` writes it), ``security``,
  ``utility``, ``runtime``, ``environment``, ``suite_name``, ``user_task_id`` and
  ``injection_task_id``. The fields may be wrapped in an ``episode`` key.
* ``payload_key``: place the payload only in the instructions whose
  ``probe_key`` (or ``template_key``) is this; by default every instruction
  carries it.
* ``upstream``: the URL of the target's MCP server.
* ``bind_host``, ``port``, ``advertise_host``: where each interceptor listens,
  and the host the agent reaches it at (port 0 picks a free port per episode).
* ``agent_client``, ``agent_options``: the MiDojo ``AgentClient`` that reaches
  the agent, as ``module:Class``, and its keyword arguments. The default is
  MiDojo's Responses API client; its API key is read from
  ``agent_api_key_env`` (``OPENAI_API_KEY``) unless ``agent_options`` gives
  one. The client must support host interception.

What each output carries
------------------------

``generate()`` receives the prompt, not the attempt, so each output
``Message`` carries its own episode record in ``Message.notes["midojo_episode"]``:
its ``evaluation_id`` and ``snapshot_id``, the ``verdict`` (MiDojo's grade, with
the evaluation and snapshot it graded), ``interrupted``, ``reset``,
``injection_outcomes`` and ``error``. An episode that failed or was cut short
has no verdict. ``detectors.midojo.EpisodeVerdict`` scores the verdict.

Opting a probe in
-----------------

The generator refuses, before any episode runs, a prompt whose probe has not
opted in. A probe opts in by marking each attempt prompt as a payload, for
example in its ``_attempt_prestore_hook``:

.. code-block:: python

   from garak.generators.midojo import mark_payload

   def _attempt_prestore_hook(self, attempt, seq):
       mark_payload(attempt.prompt)  # Conversation.notes["episode_payload"] = True
       return attempt

The prompt must be a single user turn: the payload text. Probes that write the
whole task for the target (``latentinjection``, ``promptinject``) are refused:
their prompts are user requests with the injection embedded, and an episode
sends its own user turns. ``probes.midojo.EpisodePayloads`` is a payload probe
that opts in.

Garak's own path for agent targets (data setup through the target, the
``detectors.toolcall.ToolCallCondition`` detector) is unchanged.

.. automodule:: garak.generators.midojo
   :members:
   :undoc-members:
   :show-inheritance:
