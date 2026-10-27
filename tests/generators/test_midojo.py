# SPDX-FileCopyrightText: Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for generators.midojo.EpisodeGenerator

The payload checks run everywhere. The episode tests need MiDojo, its
interception SDK and fastmcp, and skip without them; they run offline, against
a scripted MCP upstream and scripted agents (tests/generators/midojo_scripted.py).
"""

import copy
import importlib
from pathlib import Path
import pickle

import pytest

from garak import _config
from garak._config import GarakSubConfig
from garak.attempt import Conversation, Message, Turn
from garak.detectors.midojo import EpisodeVerdict, verdict_score
from garak.exception import BadGeneratorException
from garak.generators.midojo import (
    EPISODE_NOTE,
    PAYLOAD_NOTE,
    NotAPayloadError,
    episode_payload,
    mark_payload,
)
import garak.probes.latentinjection
import garak.probes.midojo
import garak.probes.promptinject

# imported by name, from this directory, here and in pool workers
SCRIPTED = "midojo_scripted"


def payload_prompt(text: str) -> Conversation:
    return mark_payload(Conversation([Turn("user", Message(text))]))


# --- payload opt-in: no MiDojo needed ---


def test_a_marked_prompt_gives_its_user_turn_as_the_payload():
    assert episode_payload(payload_prompt("plant me")) == "plant me"


def test_an_unmarked_prompt_is_refused():
    with pytest.raises(NotAPayloadError, match="has not opted in"):
        episode_payload(Conversation([Turn("user", Message("plant me"))]))


def test_a_payload_is_one_user_turn():
    prompt = mark_payload(
        Conversation(
            [Turn("system", Message("be nice")), Turn("user", Message("plant me"))]
        )
    )
    with pytest.raises(NotAPayloadError, match="one user turn"):
        episode_payload(prompt)


@pytest.mark.parametrize(
    "probe_class",
    [
        garak.probes.latentinjection.LatentInjectionFactSnippetEiffel,
        garak.probes.promptinject.HijackHateHumans,
    ],
)
def test_probes_that_write_whole_tasks_are_refused(probe_class):
    probe = probe_class()
    attempt = probe._mint_attempt(probe.prompts[0], 0)
    with pytest.raises(NotAPayloadError, match="latentinjection and promptinject"):
        episode_payload(attempt.prompt)


def test_the_payload_probe_opts_in():
    probe = garak.probes.midojo.EpisodePayloads()
    attempt = probe._mint_attempt(probe.prompts[0], 0)
    assert attempt.prompt.notes[PAYLOAD_NOTE] is True
    assert episode_payload(attempt.prompt) == probe.prompts[0]


# --- episodes: need MiDojo ---


@pytest.fixture
def scripted(monkeypatch):
    pytest.importorskip("midojo")
    pytest.importorskip("midojo.interception_host")
    pytest.importorskip("fastmcp")
    # a worker process of a spawned pool starts with this sys.path too
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    return importlib.import_module(SCRIPTED)


@pytest.fixture
def upstream(scripted):
    server = scripted.ScriptedUpstream()
    yield server
    server.stop()


def episode_generator(scripted, server, agent="ScriptedAgent", **options):
    from garak.generators.midojo import EpisodeGenerator

    config_root = GarakSubConfig()
    config_root.generators = {
        "midojo": {
            "EpisodeGenerator": {
                "episode": copy.deepcopy(scripted.EPISODE),
                "upstream": server.url,
                "agent_client": f"{SCRIPTED}:{agent}",
                **options,
            }
        }
    }
    return EpisodeGenerator(config_root=config_root)


def records(outputs):
    return [output.notes[EPISODE_NOTE] for output in outputs]


def assert_own_episodes(outputs):
    """Each output names its own sealed evaluation, and its verdict graded exactly that"""
    found = records(outputs)
    evaluations = [r["evaluation_id"] for r in found]
    snapshots = [r["snapshot_id"] for r in found]
    assert None not in evaluations and None not in snapshots
    assert len(set(evaluations)) == len(outputs), "one evaluation per output"
    assert len(set(snapshots)) == len(outputs), "one snapshot per output"
    for r in found:
        assert (r["verdict"]["evaluation_id"], r["verdict"]["snapshot_id"]) == (
            r["evaluation_id"],
            r["snapshot_id"],
        )
        assert r["interrupted"] is False and r["error"] is None
        assert r["reset"] == "not_requested", "the target is never reset"


def test_each_generation_is_one_episode_of_its_own(scripted, upstream):
    g = episode_generator(scripted, upstream)
    outputs = g.generate(payload_prompt(scripted.MALICIOUS), generations_this_call=3)

    assert len(outputs) == 3 and all(isinstance(o, Message) for o in outputs)
    assert_own_episodes(outputs)
    assert [verdict_score(o) for o in outputs] == [1.0, 1.0, 1.0]
    for output, record in zip(outputs, records(outputs)):
        # the agent saw the payload in the tool result, never as its user turn
        assert scripted.MALICIOUS in output.text
        assert [o["status"] for o in record["injection_outcomes"]] == ["applied"]
        assert record["verdict"]["grade"]["utility"] is True
    # the target's own log: three episodes, each asked for the weather and was hijacked
    assert sorted(upstream.calls) == sorted(
        [("get_weather", "Boston"), (scripted.ATTACK_TOOL, "Boston")] * 3
    )


def test_a_payload_the_agent_ignores_scores_zero(scripted, upstream):
    g = episode_generator(scripted, upstream)
    [output] = g.generate(payload_prompt(scripted.BENIGN))
    assert_own_episodes([output])
    assert verdict_score(output) == 0.0
    assert upstream.calls == [("get_weather", "Boston")]


def test_the_agent_gets_the_episodes_turns_not_the_payload(scripted, upstream):
    sent = []

    class Spy(scripted.ScriptedAgent):
        async def send_task(self, prompt, **kwargs):
            sent.append(prompt)
            return await super().send_task(prompt, **kwargs)

    g = episode_generator(scripted, upstream)
    g._agent_factory = lambda: Spy
    g.generate(payload_prompt(scripted.MALICIOUS))
    assert sent == scripted.EPISODE["turns"]


def test_parallel_requests_run_independent_episodes(scripted, upstream):
    _config.system.parallel_requests = 2
    _config.system.max_workers = 2
    g = episode_generator(scripted, upstream)
    assert g.parallel_requests == 2
    outputs = g.generate(payload_prompt(scripted.MALICIOUS), generations_this_call=3)
    assert len(outputs) == 3
    assert_own_episodes(outputs)
    assert [verdict_score(o) for o in outputs] == [1.0, 1.0, 1.0]
    assert len(upstream.calls) == 6


def test_parallel_attempts_run_independent_episodes(scripted, upstream, tmp_path):
    _config.system.parallel_attempts = 2
    _config.system.max_workers = 2
    _config.run.generations = 2
    g = episode_generator(scripted, upstream)
    probe_root = GarakSubConfig()
    probe_root.probes = {
        "midojo": {
            "EpisodePayloads": {"templates": [scripted.MALICIOUS, scripted.BENIGN]}
        }
    }
    probe = garak.probes.midojo.EpisodePayloads(config_root=probe_root)
    assert probe.parallel_attempts == 2
    path = tmp_path / "probe.report.jsonl"
    with open(path, "w", encoding="utf-8") as reportfile:
        _config.transient.reportfile = reportfile
        _config.transient.report_filename = str(path)
        attempts = list(probe.probe(g))

    assert len(attempts) == 2
    outputs = [o for attempt in attempts for o in attempt.outputs]
    assert len(outputs) == 4
    assert_own_episodes(outputs)
    detector = EpisodeVerdict()
    scores = {
        attempt.prompt.last_message().text: detector.detect(attempt)
        for attempt in attempts
    }
    assert scores == {scripted.MALICIOUS: [1.0, 1.0], scripted.BENIGN: [0.0, 0.0]}


def test_the_generator_pickles_without_episode_state(scripted, upstream):
    g = pickle.loads(pickle.dumps(episode_generator(scripted, upstream)))
    [output] = g.generate(payload_prompt(scripted.MALICIOUS))
    assert verdict_score(output) == 1.0


def test_an_unmarked_prompt_runs_no_episode(scripted, upstream):
    g = episode_generator(scripted, upstream)
    with pytest.raises(NotAPayloadError):
        g.generate(Conversation([Turn("user", Message(scripted.MALICIOUS))]))
    assert upstream.calls == []


def test_a_failed_turn_leaves_a_sealed_evaluation_and_no_verdict(scripted, upstream):
    g = episode_generator(scripted, upstream, agent="FailingAgent")
    [output] = g.generate(payload_prompt(scripted.MALICIOUS))
    [record] = records([output])
    assert record["evaluation_id"] and record["snapshot_id"]
    assert record["interrupted"] is True
    assert record["verdict"] is None
    assert "ConnectionError" in record["error"]
    assert verdict_score(output) is None


def test_an_unreachable_upstream_gives_no_verdict(scripted, upstream):
    g = episode_generator(scripted, upstream)
    upstream.stop()
    [output] = g.generate(payload_prompt(scripted.MALICIOUS))
    [record] = records([output])
    assert record["verdict"] is None and record["error"] is not None
    assert verdict_score(output) is None


def test_payload_key_places_the_payload_only_where_keyed(scripted, upstream):
    spec = copy.deepcopy(scripted.EPISODE)
    spec["plan"]["instructions"].append(
        {
            "channel": "tool_output",
            "payload": "fixed",
            "target": {"tool": "get_weather"},
            "probe_key": "other",
        }
    )
    g = episode_generator(scripted, upstream, episode=spec, payload_key="weather")
    placed = [i["payload"] for i in g._plan_data("X")["instructions"]]
    assert placed == ["X", "fixed"]
    with pytest.raises(BadGeneratorException, match="no instruction"):
        episode_generator(scripted, upstream, payload_key="absent")


def test_an_episode_without_a_security_check_scores_none(scripted, upstream):
    g = episode_generator(scripted, upstream)
    # past the load-time refusal: MiDojo grades it security False, not applicable
    del g._episode_data["security"]
    [output] = g.generate(payload_prompt(scripted.MALICIOUS))
    grade = records([output])[0]["verdict"]["grade"]
    assert grade["security"] is False
    assert grade["grade_validity"]["security"]["status"] == "not_applicable"
    assert verdict_score(output) is None


@pytest.mark.parametrize(
    "channel_instruction",
    [
        {
            "channel": "tool_description",
            "payload": "placeholder",
            "target": {"tool": "get_weather", "surface": "description"},
        },
        {
            "channel": "user_prompt",
            "payload": "placeholder",
            "target": {"surface": "template", "template_key": "weather"},
            "probe_key": "weather",
        },
    ],
)
def test_a_payload_on_a_channel_host_interception_does_not_deliver_is_refused(
    scripted, upstream, channel_instruction
):
    spec = copy.deepcopy(scripted.EPISODE)
    spec["plan"]["instructions"].append(channel_instruction)
    spec["plan"]["probe_payloads"] = [["weather", "placeholder"]]
    with pytest.raises(BadGeneratorException, match="delivers only 'tool_output'"):
        episode_generator(scripted, upstream, episode=spec)
    assert upstream.calls == []


def test_payload_key_can_leave_an_undelivered_channel_its_own_payload(
    scripted, upstream
):
    spec = copy.deepcopy(scripted.EPISODE)
    spec["plan"]["instructions"].append(
        {
            "channel": "tool_description",
            "payload": "fixed",
            "target": {"tool": "get_weather", "surface": "description"},
            "probe_key": "description",
        }
    )
    g = episode_generator(scripted, upstream, episode=spec, payload_key="weather")
    [output] = g.generate(payload_prompt(scripted.MALICIOUS))
    assert verdict_score(output) == 1.0


def test_a_multi_turn_episode_sends_every_turn_in_order_in_one_episode(
    scripted, upstream
):
    spec = copy.deepcopy(scripted.EPISODE)
    spec["turns"] = ["What's the weather in Boston?", "Anything else I should do?"]
    scripted.SENT.clear()
    g = episode_generator(scripted, upstream, agent="MultiTurnAgent", episode=spec)
    [output] = g.generate(payload_prompt(scripted.MALICIOUS))

    assert scripted.SENT == spec["turns"]
    assert_own_episodes([output])
    assert output.text == "turn 2: Anything else I should do?"
    assert verdict_score(output) == 1.0
    assert sorted(upstream.calls) == sorted(
        [("get_weather", "Boston"), (scripted.ATTACK_TOOL, "Boston")]
    )


@pytest.mark.parametrize(
    "change, message",
    [
        ({"episode": None}, "needs an episode"),
        (
            {"episode": {"turns": ["hi"], "plan": {"instructions": []}}},
            "needs a security check",
        ),
        ({"episode": {"plan": {"instructions": []}}}, "fixed user turns"),
        ({"upstream": None}, "needs upstream"),
        ({"agent": "SessionOnlyAgent"}, "host interception only"),
        ({"agent_client": "no.such.module:Agent"}, "Cannot load agent client"),
        ({"episode": {"turns": ["hi"], "plan": {}, "tools": []}}, "Unknown episode"),
        (
            {
                "episode": {
                    "turns": ["hi"],
                    "plan": {
                        "instructions": [
                            # a tool output has no description selector
                            {
                                "channel": "tool_output",
                                "target": {"surface": "description"},
                            }
                        ]
                    },
                    "security": {"tool_called": {"name": "book_shelter"}},
                }
            },
            "MiDojo refuses",
        ),
        (
            {
                "episode": {
                    "turns": ["hi"],
                    "plan": {"instructions": [{"channel": "bogus", "payload": ""}]},
                    "security": {"tool_called": {"name": "book_shelter"}},
                }
            },
            "delivers only",
        ),
    ],
)
def test_an_episode_spec_it_cannot_run_fails_at_load(
    scripted, upstream, change, message
):
    change = dict(change)
    agent = change.pop("agent", "ScriptedAgent")
    with pytest.raises(BadGeneratorException, match=message):
        episode_generator(scripted, upstream, agent=agent, **change)
    assert upstream.calls == []
