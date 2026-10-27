# SPDX-FileCopyrightText: Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiDojo episode generator

Runs each prompt as the payload of one MiDojo episode. The episode, not the
prompt, decides what the agent under test is asked: its injection plan places
the payload in a tool result, the generator sends the episode's own fixed user
turns to the agent, and MiDojo seals the evidence. The sealed snapshot is graded
with the episode's checks, and ``detectors.midojo.EpisodeVerdict`` scores that
verdict.

MiDojo is an optional dependency (Python 3.12 or later): this module imports it
only when the generator is built.

What reaches the detector
-------------------------

``generate()`` sees a prompt, never the ``Attempt``. Each output ``Message``
therefore carries its own episode in ``Message.notes["midojo_episode"]``, plain
JSON data:

* ``evaluation_id`` and ``snapshot_id``: the evaluation this output's episode
  opened and the sealed snapshot it left (None when the episode never opened or
  never sealed);
* ``verdict``: MiDojo's grade of that snapshot, with the ``evaluation_id`` and
  ``snapshot_id`` it graded, or None when the episode failed or was cut short;
* ``interrupted``, ``reset``, ``injection_outcomes`` and ``error``: what
  happened during the run.

Every output names its own evaluation; nothing refers to "the latest
evaluation".

Probe opt-in
------------

A probe opts in by marking each attempt prompt as a payload:
``Conversation.notes["episode_payload"] = True`` (see :func:`mark_payload`), and
the prompt must be one user turn, the payload. Any other prompt raises
:class:`NotAPayloadError` before an episode runs.

Document settings
-----------------

An episode document may give settings beside its fields: ``tools`` and
``interception``. Any other key beside the fields is refused when the generator
loads, in a flat spec and beside a wrapped ``episode`` alike.

``interception`` may only name the host interception the generator runs:
``"host"`` or ``{"mode": "host"}``.

``tools`` limits and blocks the agent's tools:

* ``allowed``: the only tools the agent may call. An agent client whose
  constructor takes ``allowed_tools`` gets them, and is trusted to apply the
  limit: its requests are not checked. For MiDojo's stock Responses API client
  the generator sends every request itself, with the client's settings and an
  MCP tool block naming the allowed tools, and refuses to send one that is not
  so limited (:class:`UnlimitedToolsError`). Any other client, a subclass of
  the stock Responses client included, is refused when the generator loads.
* ``write_disposition``: ``{"default": "forward", "tools": {name: disposition}}``.
  A tool whose disposition is ``block`` is blocked at MiDojo's host
  interceptor: the agent still sees it, its calls are recorded as blocked and
  never reach the target. ``forward`` runs the call on the target.
"""

import asyncio
import contextlib
import copy
import functools
import importlib
import inspect
import json
import logging
import os
from pathlib import Path
from typing import List, Union

from garak import _config
from garak.attempt import Conversation, Message
from garak.exception import (
    APIKeyMissingError,
    BadGeneratorException,
    GarakException,
)
from garak.generators.base import Generator

# set by an opted-in probe on each prompt whose single user turn is a payload
PAYLOAD_NOTE = "episode_payload"
# the episode record each output carries
EPISODE_NOTE = "midojo_episode"
# the one channel host interception delivers: the episode context hands the
# interceptor only the plan's tool-output instructions
DELIVERED_CHANNEL = "tool_output"
# the fields of an episode spec, as midojo.episode.Episode takes them
EPISODE_FIELDS = (
    "runtime",
    "turns",
    "plan",
    "security",
    "utility",
    "environment",
    "suite_name",
    "user_task_id",
    "injection_task_id",
)
# MiDojo's Responses API agent client, the default agent client
RESPONSES_AGENT_CLIENT = "midojo.agent_client:OpenAIResponsesAgentClient"
# the settings of that client the generator sends each request with, when it
# limits the agent to the allowed tools
RESPONSES_SETTINGS = (
    "base_url",
    "model",
    "api_key",
    "timeout",
    "instructions",
    "mcp_server_label",
)
# every setting the client keeps: in host mode its tool block names the tool
# endpoint, never mcp_server_url
RESPONSES_CLIENT_SETTINGS = RESPONSES_SETTINGS + ("mcp_server_url",)
# the settings an episode document may give beside its fields; the generator
# honours each one
TOOLS_FIELD = "tools"
INTERCEPTION_FIELD = "interception"
DOCUMENT_SETTINGS = (TOOLS_FIELD, INTERCEPTION_FIELD)
# the one interception the generator runs: MiDojo's host interception
HOST_MODE = "host"
TOOLS_KEYS = ("allowed", "write_disposition")
WRITE_DISPOSITION_KEYS = ("default", "tools")
# the write dispositions host interception honours: run the call on the target,
# or block it at the interceptor
FORWARD = "forward"
BLOCK = "block"


class NotAPayloadError(GarakException):
    """A prompt reached the episode generator without being a payload"""


class UnlimitedToolsError(GarakException):
    """A Responses API request would offer the agent tools beyond the allowed ones"""


def mark_payload(conversation: Conversation) -> Conversation:
    """Mark ``conversation`` as an episode payload, for probes that opt in"""
    if conversation.notes is None:
        conversation.notes = {}
    conversation.notes[PAYLOAD_NOTE] = True
    return conversation


def episode_payload(prompt: Conversation) -> str:
    """Return the payload a marked prompt carries: the text of its one user turn"""
    if not (prompt.notes or {}).get(PAYLOAD_NOTE):
        raise NotAPayloadError(
            "generators.midojo.EpisodeGenerator places each prompt as the payload "
            "of an episode, and this prompt's probe has not opted in "
            f"(Conversation.notes[{PAYLOAD_NOTE!r}]). Probes that write the whole "
            "task for the target, such as latentinjection and promptinject, are "
            "not supported: an episode sends its own fixed user turns. "
            "Use probes.midojo.EpisodePayloads, or opt a payload probe in."
        )
    roles = [turn.role for turn in prompt.turns]
    if roles != ["user"]:
        raise NotAPayloadError(
            f"An episode payload is a prompt of one user turn; got the roles {roles}"
        )
    text = prompt.turns[0].content.text
    if not isinstance(text, str):
        raise NotAPayloadError("An episode payload must be text")
    return text


def _import_object(name: str):
    """The object ``name`` names, as ``module:attr`` or ``module.attr``"""
    module_name, sep, attr = name.partition(":")
    if not sep:
        module_name, _, attr = name.rpartition(".")
    return getattr(importlib.import_module(module_name), attr)


def _load_spec(spec) -> tuple:
    """Return the episode fields and tool settings ``spec`` gives

    ``spec`` is a JSON file path, or a mapping. The tool settings are the tools
    the agent may call (None for every tool) and those blocked at the
    interceptor; see :func:`_tool_settings`.
    """
    if spec is None:
        raise BadGeneratorException(
            "generators.midojo.EpisodeGenerator needs an episode: "
            "the path of an episode JSON file, or its fields"
        )
    if isinstance(spec, (str, Path)):
        try:
            with open(spec, "r", encoding="utf-8") as f:
                spec = json.load(f)
        except (OSError, ValueError) as e:
            raise BadGeneratorException(f"Cannot read episode {spec}: {e}") from e
    if not isinstance(spec, dict):
        raise BadGeneratorException("An episode spec must be a JSON object")
    # a document may wrap the episode's fields in "episode", its settings beside
    # them; a flat spec gives its settings among the fields
    if isinstance(spec.get("episode"), dict):
        data = spec["episode"]
        unknown = sorted(set(spec) - {"episode", *DOCUMENT_SETTINGS})
        if unknown:
            raise BadGeneratorException(
                f"Unknown episode document settings: {unknown}; beside a wrapped "
                f"episode the generator honours only {list(DOCUMENT_SETTINGS)}"
            )
    else:
        data = {k: v for k, v in spec.items() if k not in DOCUMENT_SETTINGS}
    _check_interception(spec.get(INTERCEPTION_FIELD, HOST_MODE))
    tools = _tool_settings(spec.get(TOOLS_FIELD))
    unknown = sorted(set(data) - set(EPISODE_FIELDS))
    if unknown:
        raise BadGeneratorException(f"Unknown episode fields: {unknown}")
    if not isinstance(data.get("plan"), dict):
        raise BadGeneratorException("An episode needs an injection plan")
    if not data.get("turns"):
        raise BadGeneratorException(
            "An episode needs fixed user turns: the generator sends those, "
            "never the payload, to the agent"
        )
    if data.get("security") is None:
        raise BadGeneratorException(
            "An episode needs a security check: without one MiDojo grades every "
            "episode not_applicable, and no output could be scored"
        )
    return copy.deepcopy(data), tools


def _check_interception(interception) -> None:
    """Refuse an interception setting other than the host interception the generator runs"""
    if interception not in (HOST_MODE, {"mode": HOST_MODE}):
        raise BadGeneratorException(
            f"Interception {interception!r} cannot be honoured: the generator runs "
            f"MiDojo's host interception in front of upstream, {HOST_MODE!r}"
        )


def _tool_settings(tools) -> tuple:
    """Return the tools the agent may call (None for every tool), and the blocked ones

    Refuses settings host interception cannot honour.
    """
    if tools is None:
        return None, []
    if not isinstance(tools, dict):
        raise BadGeneratorException("An episode's tools must be a JSON object")
    unknown = sorted(set(tools) - set(TOOLS_KEYS))
    if unknown:
        raise BadGeneratorException(f"Unknown episode tool settings: {unknown}")
    allowed = tools.get("allowed")
    if allowed is not None and not (
        isinstance(allowed, list)
        and allowed
        and all(isinstance(name, str) and name for name in allowed)
    ):
        raise BadGeneratorException(
            "An episode's tools.allowed must be a non-empty list of tool names"
        )
    disposition = tools.get("write_disposition")
    if disposition is None:
        return allowed, []
    if not isinstance(disposition, dict):
        raise BadGeneratorException(
            "An episode's tools.write_disposition must be a JSON object"
        )
    unknown = sorted(set(disposition) - set(WRITE_DISPOSITION_KEYS))
    if unknown:
        raise BadGeneratorException(f"Unknown write disposition settings: {unknown}")
    default = disposition.get("default", FORWARD)
    if default != FORWARD:
        raise BadGeneratorException(
            f"Write disposition default {default!r} cannot be honoured: host "
            f"interception blocks only the tools tools.write_disposition.tools "
            f"marks {BLOCK!r}; the default must be {FORWARD!r}"
        )
    named = disposition.get("tools", {})
    if not isinstance(named, dict) or not all(
        isinstance(name, str) and name for name in named
    ):
        raise BadGeneratorException(
            "An episode's tools.write_disposition.tools must map tool names to "
            "dispositions"
        )
    refused = {
        name: value for name, value in named.items() if value not in (FORWARD, BLOCK)
    }
    if refused:
        raise BadGeneratorException(
            f"Write dispositions cannot be honoured: {refused}; host interception "
            f"honours {FORWARD!r} and {BLOCK!r}"
        )
    return allowed, sorted(name for name, value in named.items() if value == BLOCK)


class _LimitedResponses:
    """An OpenAI client's ``responses``, refusing any request not limited to ``allowed``

    Every tool a request offers must be an MCP tool block whose
    ``allowed_tools`` is exactly ``allowed``; any other request raises
    :class:`UnlimitedToolsError` before it is sent.
    """

    def __init__(self, responses, allowed):
        self._responses = responses
        self._allowed = list(allowed)

    async def create(self, **request):
        """Send the request, once its every tool is an MCP block limited to ``allowed``"""
        tools = request.get("tools")
        if not tools or not all(
            isinstance(tool, dict)
            and tool.get("type") == "mcp"
            and tool.get("allowed_tools") == self._allowed
            for tool in tools
        ):
            raise UnlimitedToolsError(
                "A Responses API request must offer only MCP tools limited to the "
                f"allowed tools {self._allowed}; it was not sent"
            )
        return await self._responses.create(**request)


class _AllowedToolsConversation:
    """One episode's conversation over the Responses API, limited to the allowed tools

    Each turn is one request through :class:`_LimitedResponses`, chained on the
    previous reply's ID, with an MCP tool block that names the interceptor's
    URL and bearer token and the allowed tools.
    """

    def __init__(self, settings: dict, tool_endpoint, allowed):
        self._settings = settings
        self._tool = {
            "type": "mcp",
            "server_label": settings["mcp_server_label"],
            "server_url": tool_endpoint.url,
            "headers": dict(tool_endpoint.headers),
            "require_approval": "never",
            "allowed_tools": list(allowed),
        }
        self._allowed = list(allowed)
        self._responses = None
        self._previous_response_id = None
        self._resources = contextlib.AsyncExitStack()

    async def send(self, message: str) -> str:
        """Send one turn; return the agent's reply"""
        if self._responses is None:
            from openai import AsyncOpenAI

            client = await self._resources.enter_async_context(
                AsyncOpenAI(
                    base_url=self._settings["base_url"],
                    api_key=self._settings["api_key"],
                    timeout=self._settings["timeout"],
                )
            )
            self._responses = _LimitedResponses(client.responses, self._allowed)
        continuation = (
            {"previous_response_id": self._previous_response_id}
            if self._previous_response_id is not None
            else {}
        )
        response = await self._responses.create(
            model=self._settings["model"],
            input=message,
            instructions=self._settings["instructions"],
            tools=[dict(self._tool)],
            stream=False,
            **continuation,
        )
        self._previous_response_id = response.id
        return response.output_text

    async def aclose(self) -> None:
        """Close the OpenAI client"""
        await self._resources.aclose()


class _AllowedToolsAgent:
    """MiDojo's stock Responses API client, its every request sent by the generator

    Reads the client's settings, and starts conversations of its own
    (:class:`_AllowedToolsConversation`) in place of the client's, so no request
    leaves without the allowed tools. Only the stock client is accepted, with
    no setting the generator does not reproduce: a subclass may change its
    requests in ways the generator would silently drop.
    """

    def __init__(self, client, allowed):
        name = type(client).__qualname__
        responses_client = _import_object(RESPONSES_AGENT_CLIENT)
        if not isinstance(client, responses_client):
            raise BadGeneratorException(
                f"Agent client {name} cannot be limited to the episode's allowed "
                "tools: it takes no allowed_tools, and is not MiDojo's "
                "OpenAIResponsesAgentClient"
            )
        if type(client) is not responses_client:
            raise BadGeneratorException(
                f"Agent client {name} cannot be limited to the episode's allowed "
                "tools: the generator sends the requests of MiDojo's stock "
                "OpenAIResponsesAgentClient only, and would drop what a subclass "
                "changes. Give the subclass an allowed_tools constructor argument, "
                "and apply the limit there"
            )
        unreproduced = sorted(set(vars(client)) - set(RESPONSES_CLIENT_SETTINGS))
        if unreproduced:
            raise BadGeneratorException(
                f"Agent client {name} cannot be limited to the episode's allowed "
                f"tools: the generator does not reproduce its settings {unreproduced}"
            )
        settings = {
            setting: getattr(client, setting, None) for setting in RESPONSES_SETTINGS
        }
        unusable = sorted(
            setting
            for setting in ("base_url", "model", "mcp_server_label")
            if not isinstance(settings[setting], str) or not settings[setting]
        ) + sorted(
            setting
            for setting in ("api_key", "instructions")
            if not isinstance(settings[setting], str)
        )
        if not isinstance(settings["timeout"], (int, float)):
            unusable.append("timeout")
        if unusable:
            raise BadGeneratorException(
                f"Agent client {name} cannot be limited to the episode's allowed "
                f"tools: its Responses API settings {unusable} are missing or "
                "unusable"
            )
        self._settings = settings
        self._allowed = list(allowed)

    def start_conversation(
        self, *, session_token=None, turn_count=1, tool_endpoint=None
    ):
        """A conversation limited to the allowed tools, in host mode only"""
        del turn_count  # each turn is one request, chained on the previous reply
        if session_token is not None or tool_endpoint is None:
            raise UnlimitedToolsError(
                "The allowed tools are sent only to a host-mode agent's tool endpoint"
            )
        return _AllowedToolsConversation(self._settings, tool_endpoint, self._allowed)


@functools.cache
def _blocking_interception(blocked: tuple):
    """MiDojo's ``HostInterception``, its interceptors blocking the ``blocked`` tools

    The agent still sees a blocked tool; MiDojo records each call to it as
    blocked, answers it with an error and never forwards it to the upstream.
    """
    from midojo.interception_host import HostInterception

    class BlockingInterception(HostInterception):
        """Host interception whose interceptors block the episode's blocked tools"""

        def interceptor(self, recorder, *, evaluation_id, plan):
            """A new interceptor for one evaluation, blocking the blocked tools"""
            made = super().interceptor(recorder, evaluation_id=evaluation_id, plan=plan)
            if blocked:
                made.mcp.block(*blocked)
            return made

    return BlockingInterception


def _outcome(outcome) -> dict:
    """An interceptor's injection outcome, as plain data"""
    return {
        "instruction_index": outcome.instruction_index,
        "probe_key": outcome.probe_key,
        "tool": outcome.tool,
        "call_id": outcome.call_id,
        "occurrence": outcome.occurrence,
        "status": outcome.status.value,
        "detail": outcome.detail,
        "selected": outcome.selected,
        "emitted": [surface.value for surface in outcome.emitted],
    }


class EpisodeGenerator(Generator):
    """Run each prompt as the payload of one MiDojo episode against an agent

    Each generation is one episode: MiDojo's host interceptor mirrors the
    target's tools from ``upstream`` and places the payload as the episode's
    injection plan says, the agent gets only the interceptor's endpoint and the
    episode's fixed user turns, and the sealed snapshot is graded with the
    episode's checks. The target is never reset.
    """

    DEFAULT_PARAMS = Generator.DEFAULT_PARAMS | {
        # path of an episode JSON file, or its fields: runtime, turns, plan (as
        # midojo_sdk.injection.InjectionPlan.to_data() writes it), security,
        # utility, environment, and provenance IDs; may be wrapped in "episode",
        # with its tool settings ("tools": allowed, write_disposition) beside them
        "episode": None,
        # the plan instructions that carry the payload: those whose probe_key or
        # template_key equals this; None for every instruction
        "payload_key": None,
        # URL of the target's MCP server, the interceptor's upstream
        "upstream": None,
        # where each episode's interceptor listens, and the host the agent
        # reaches it at; port 0 picks a free port per episode
        "bind_host": "127.0.0.1",
        "port": 0,
        "advertise_host": None,
        # the MiDojo AgentClient that reaches the agent, as "module:Class", and
        # its keyword arguments; it must support host interception
        "agent_client": RESPONSES_AGENT_CLIENT,
        "agent_options": {},
        # environment variable an agent client's api_key is read from, when
        # agent_options gives none
        "agent_api_key_env": "OPENAI_API_KEY",
    }

    generator_family_name = "MiDojo"
    name = "episode"
    # host interception needs MiDojo's interception SDK with its mcp extra too
    extra_dependency_names = ["midojo", "midojo.interception_host"]
    supports_multiple_generations = False

    def __init__(self, name="", config_root=_config):
        super().__init__(name, config_root=config_root)
        self._episode_data, (self._allowed_tools, self._blocked_tools) = _load_spec(
            self.episode
        )
        if not self.upstream:
            raise BadGeneratorException(
                "generators.midojo.EpisodeGenerator needs upstream: "
                "the URL of the target's MCP server"
            )
        if not getattr(self._agent_factory(), "supports_host_interception", False):
            raise BadGeneratorException(
                f"Agent client {self.agent_client} cannot be given a tool "
                "endpoint; the episode generator runs host interception only"
            )
        self._agent_options()
        # refuses an agent client that cannot be limited to the allowed tools
        if self._allowed_tools is not None:
            self._agent()
        # build the episode once, so a spec MiDojo refuses fails here
        try:
            self._build_episode("")
        except (TypeError, ValueError) as e:
            raise BadGeneratorException(f"MiDojo refuses the episode: {e}") from e

    def _plan_data(self, payload: str) -> dict:
        """The episode's plan as data, with ``payload`` placed where it goes"""
        plan = copy.deepcopy(self._episode_data["plan"])
        key = self.payload_key
        placed = 0
        for instruction in plan.get("instructions", []):
            target = instruction.get("target") or {}
            if key is None or key in (
                instruction.get("probe_key"),
                target.get("template_key"),
            ):
                channel = instruction.get("channel")
                if channel != DELIVERED_CHANNEL:
                    raise BadGeneratorException(
                        f"A plan instruction on channel {channel!r} would carry the "
                        "payload, but host interception delivers only "
                        f"{DELIVERED_CHANNEL!r}; narrow it out with payload_key"
                    )
                instruction["payload"] = payload
                placed += 1
        # the host's substitution map must agree with its template instructions
        plan["probe_payloads"] = [
            [probe, payload if key is None or probe == key else value]
            for probe, value in plan.get("probe_payloads", [])
        ]
        if not placed:
            raise BadGeneratorException(
                "The episode's plan has no instruction to carry the payload"
                + (f" under payload_key {key!r}" if key is not None else "")
            )
        return plan

    def _build_episode(self, payload: str):
        from midojo.episode import Episode
        from midojo_sdk.injection import InjectionPlan

        fields = {
            k: copy.deepcopy(v) for k, v in self._episode_data.items() if k != "plan"
        }
        plan = InjectionPlan.from_data(self._plan_data(payload))
        return Episode(plan=plan, **fields)

    def _agent_factory(self):
        try:
            return _import_object(self.agent_client)
        except (ImportError, AttributeError, ValueError) as e:
            raise BadGeneratorException(
                f"Cannot load agent client {self.agent_client}: {e}"
            ) from e

    def _agent(self):
        """The agent, limited to the episode's allowed tools when it has them

        A client that takes ``allowed_tools`` is given them. Otherwise MiDojo's
        Responses API client is wrapped, so the generator sends its requests.
        """
        factory = self._agent_factory()
        agent = factory(**self._agent_options())
        if (
            self._allowed_tools is None
            or "allowed_tools" in inspect.signature(factory).parameters
        ):
            return agent
        return _AllowedToolsAgent(agent, self._allowed_tools)

    def _agent_options(self) -> dict:
        options = dict(self.agent_options or {})
        parameters = inspect.signature(self._agent_factory()).parameters
        if self._allowed_tools is not None and "allowed_tools" in parameters:
            if "allowed_tools" in options:
                raise BadGeneratorException(
                    "The episode's tools.allowed limits the agent: give no "
                    "allowed_tools in agent_options"
                )
            options["allowed_tools"] = list(self._allowed_tools)
        # in host mode the agent's MCP server is the interceptor, never a URL given here
        if "mcp_server_url" in parameters:
            options.setdefault("mcp_server_url", None)
        if "api_key" in parameters and "api_key" not in options:
            key = os.getenv(self.agent_api_key_env) if self.agent_api_key_env else None
            if key is None and parameters["api_key"].default is inspect.Parameter.empty:
                raise APIKeyMissingError(
                    f"Agent client {self.agent_client} needs an API key: set "
                    f"{self.agent_api_key_env} or agent_options.api_key"
                )
            if key is not None:
                options["api_key"] = key
        return options

    def _call_model(
        self, prompt: Conversation, generations_this_call: int = 1
    ) -> List[Union[Message, None]]:
        payload = episode_payload(prompt)
        # one fresh episode per generation: nothing is shared between them
        return [
            asyncio.run(self._run_episode(payload))
            for _ in range(generations_this_call)
        ]

    async def _run_episode(self, payload: str) -> Message:
        import midojo

        episode = self._build_episode(payload)
        interception = _blocking_interception(tuple(self._blocked_tools))(
            upstream=self.upstream,
            bind_host=self.bind_host,
            port=self.port,
            advertise_host=self.advertise_host,
        )
        agent = self._agent()
        # a fresh runtime and in-process control plane; reset is never asked for
        context = midojo.open_episode(episode, interception=interception)
        error = None
        try:
            async with context as ep:
                conversation = agent.start_conversation(
                    turn_count=len(ep.turns), **ep.binding.access
                )
                try:
                    for message in ep.turns:
                        await ep.turn(message, conversation.send)
                finally:
                    await conversation.aclose()
        except Exception as e:  # the failure is recorded on the output
            logging.warning("midojo episode failed: %r", e)
            error = e
        return self._output(episode, context, error)

    @staticmethod
    def _output(episode, context, error) -> Message:
        """The output of one episode: its sealed reply, and its record in the notes"""
        import midojo

        record = {
            "evaluation_id": None,
            "snapshot_id": None,
            "interrupted": None,
            "reset": None,
            "injection_outcomes": [],
            "verdict": None,
            "error": None if error is None else f"{type(error).__name__}: {error}",
        }
        try:
            result = context.result
        except RuntimeError:  # the evaluation never opened, or never sealed
            return Message(text=None, notes={EPISODE_NOTE: record})
        record.update(
            evaluation_id=result.evaluation_id,
            snapshot_id=result.snapshot_id,
            interrupted=result.interrupted,
            reset=result.reset,
            injection_outcomes=[_outcome(o) for o in result.injection_outcomes],
        )
        snapshot = result.snapshot
        if snapshot is None:
            return Message(text=None, notes={EPISODE_NOTE: record})
        # an episode cut short is sealed with incomplete evidence, and not graded
        if error is None and not result.interrupted:
            try:
                verdict = midojo.grade(episode, snapshot)
            except ValueError as e:
                record["error"] = f"{type(e).__name__}: {e}"
            else:
                record["verdict"] = {
                    "evaluation_id": verdict.evaluation_id,
                    "snapshot_id": verdict.snapshot_id,
                    "grade": verdict.grade.model_dump(mode="json"),
                }
        return Message(text=snapshot.agent_output, notes={EPISODE_NOTE: record})


DEFAULT_CLASS = "EpisodeGenerator"
