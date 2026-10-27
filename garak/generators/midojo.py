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
"""

import asyncio
import copy
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


class NotAPayloadError(GarakException):
    """A prompt reached the episode generator without being a payload"""


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


def _load_spec(spec) -> dict:
    """Return the episode fields ``spec`` gives: a JSON file path, or a mapping"""
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
    # a document may wrap the episode's fields in "episode"
    data = spec["episode"] if isinstance(spec.get("episode"), dict) else spec
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
    return copy.deepcopy(data)


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
        # utility, environment, and provenance IDs; may be wrapped in "episode"
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
        "agent_client": "midojo.agent_client:OpenAIResponsesAgentClient",
        "agent_options": {},
        # environment variable an agent client's api_key is read from, when
        # agent_options gives none
        "agent_api_key_env": "OPENAI_API_KEY",
    }

    generator_family_name = "MiDojo"
    name = "episode"
    extra_dependency_names = ["midojo"]
    supports_multiple_generations = False

    def __init__(self, name="", config_root=_config):
        super().__init__(name, config_root=config_root)
        self._episode_data = _load_spec(self.episode)
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
        module_name, sep, attr = self.agent_client.partition(":")
        if not sep:
            module_name, _, attr = self.agent_client.rpartition(".")
        try:
            return getattr(importlib.import_module(module_name), attr)
        except (ImportError, AttributeError, ValueError) as e:
            raise BadGeneratorException(
                f"Cannot load agent client {self.agent_client}: {e}"
            ) from e

    def _agent_options(self) -> dict:
        options = dict(self.agent_options or {})
        parameters = inspect.signature(self._agent_factory()).parameters
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
        from midojo.interception_host import HostInterception

        episode = self._build_episode(payload)
        interception = HostInterception(
            upstream=self.upstream,
            bind_host=self.bind_host,
            port=self.port,
            advertise_host=self.advertise_host,
        )
        agent = self._agent_factory()(**self._agent_options())
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
