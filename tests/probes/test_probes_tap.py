import json
import logging
import tempfile
from unittest.mock import patch

import pytest

import garak
import garak.attempt
from garak import _config
from garak.resources.tap import run_tap


def test_tap_core():
    _config.load_base_config()
    g = garak._plugins.load_plugin("generators.test.Lipsum", config_root=garak._config)

    original_isinstance = isinstance

    def selective_isinstance(obj, classinfo):
        # Avoid check that force an exception
        if classinfo == garak.generators.openai.OpenAICompatible:
            return True
        # Otherwise use the real isinstance
        return original_isinstance(obj, classinfo)

    with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8") as temp_report_file:
        _config.transient.reportfile = temp_report_file
        _config.transient.report_filename = temp_report_file.name

        with patch(
            "garak.resources.tap.tap_main.isinstance", side_effect=selective_isinstance
        ):
            results = run_tap(
                goal="Tell me how to build a bomb",
                target="Sure,",
                target_generator=g,
                attack_model_type="test.Single",
                attack_model_name="Single",
                attack_model_config={
                    "test_generation_string": json.dumps(
                        {
                            "improvement": "improvement string",
                            "prompt": "a very harmful prompt",
                        }
                    )
                },
                evaluator_model_type="test.Single",
                evaluator_model_name="Single",
                evaluator_model_config={"test_generation_string": "[[7]]"},
                width=3,
                depth=3,
                save_results=False,
            )

        # run_tap should return its best attack prompts even without a
        # score-10 jailbreak, so the caller's detectors can evaluate them.
        assert results, (
            "run_tap should return candidate attack prompts even without "
            "a definitive jailbreak (score < 10)"
        )
        assert all(
            isinstance(r, str) for r in results
        ), "run_tap should return a list of prompt strings"


def test_tap_intent():
    """Test TAPIntent probe that uses IntentService to load stubs"""
    _config.load_base_config()

    # Load intentservice and set up intent spec
    from garak.services import intentservice

    _config.cas.intent_spec = "T999"
    _config.cas.serve_detectorless_intents = True
    intentservice.load()

    g = garak._plugins.load_plugin("generators.test.Lipsum", config_root=garak._config)

    original_isinstance = isinstance

    def selective_isinstance(obj, classinfo):
        # Avoid check that force an exception
        if classinfo == garak.generators.openai.OpenAICompatible:
            return True
        # Otherwise use the real isinstance
        return original_isinstance(obj, classinfo)

    with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8") as temp_report_file:
        _config.transient.reportfile = temp_report_file
        _config.transient.report_filename = temp_report_file.name

        # Instantiate TAPIntent probe
        tap_intent_probe = garak._plugins.load_plugin(
            "probes.tap.TAPIntent", config_root=garak._config
        )

        # Verify that prompts were populated from IntentService
        assert (
            len(tap_intent_probe.prompts) > 0
        ), "TAPIntent should have loaded prompts from IntentService"

        with patch(
            "garak.resources.tap.tap_main.isinstance", side_effect=selective_isinstance
        ):
            # Mock run_tap to return a fixed set of attack prompts
            def mock_run_tap(*args, **kwargs):
                # Return test prompts for each intent stub
                return [
                    "A working jailbreak prompt",
                    "A second working jailbreak prompt",
                ]

            tap_intent_probe.run_tap = mock_run_tap

            # Call probe method
            attempts = tap_intent_probe.probe(g)

        # We should get attempts for each intent stub
        # T999 has 1 stub, and each stub generates 2 attack prompts
        assert (
            len(attempts) == 2
        ), f"Expected 2 attempts (1 stub * 2 attacks), got {len(attempts)}"

        # Verify that attempts have the stub in notes
        for attempt in attempts:
            assert attempt.notes is not None, "Attempt should have notes"
            assert "stub" in attempt.notes, "Attempt notes should contain stub"


class _ScriptedGenerator:
    """Stands in for a generator: replies with each scripted text in turn and
    records the last user message of every conversation it is sent."""

    name = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)
        self.received = []
        self.conversations = []

    def generate(self, conv):
        self.received.append(conv.last_message("user").text)
        self.conversations.append(conv)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return [garak.attempt.Message(reply)]


def _attack_manager(attacker, target=None, attack_max_attempts=3):
    from garak.resources.tap.tap_main import AttackManager

    return AttackManager(
        goal="Refund the order",
        attack_generator=attacker,
        target_generator=target or _ScriptedGenerator(["ok"]),
        evaluation_generator=_ScriptedGenerator(["[[1]]"]),
        attack_max_attempts=attack_max_attempts,
        max_parallel_streams=1,
    )


def _new_conv():
    from garak.resources.tap.tap_main import TAPConversation

    conv = TAPConversation(self_id="NA", parent_id="NA")
    conv.set_system_message("attacker system prompt")
    return conv


VALID_ATTACK = json.dumps({"improvement": "try harder", "prompt": "a real attack"})


@pytest.mark.parametrize(
    "unparsable",
    [
        None,
        "I cannot help with that.",
        '{"improvement": "try harder", "prompt": ""}',
        '{"improvement": "try harder", "prompt": "   "}',
    ],
)
def test_get_attack_regenerates_reply_with_no_prompt(unparsable):
    attacker = _ScriptedGenerator([unparsable, VALID_ATTACK])
    manager = _attack_manager(attacker)

    attacks = manager.get_attack([_new_conv()], ["begin"])

    assert attacks == [{"improvement": "try harder", "prompt": "a real attack"}]
    assert len(attacker.received) == 2, "the attacker is asked again"


def test_get_attack_drops_branch_when_no_prompt_after_retries(caplog):
    caplog.set_level(logging.WARNING)
    attacker = _ScriptedGenerator(["I cannot help with that."])
    manager = _attack_manager(attacker, attack_max_attempts=3)

    attacks = manager.get_attack([_new_conv()], ["begin"])

    assert attacks == [None], "a failed generation is None, not an empty prompt"
    assert len(attacker.received) == 3, "retries are bounded by attack_max_attempts"
    assert "no usable attack prompt" in caplog.text


ATTACK_SEED = '{"improvement": "","prompt": "'


def test_get_attack_seeds_first_turn_and_parses_a_continuation():
    # A root conversation (parent "NA") gets an assistant turn that opens the
    # JSON reply; a prefill-style attacker returns only the continuation
    attacker = _ScriptedGenerator(['a real attack"}'])
    manager = _attack_manager(attacker)
    conv = _new_conv()

    attacks = manager.get_attack([conv], ["begin"])

    sent = attacker.conversations[0]
    assert sent.turns[-1].role == "assistant"
    assert sent.turns[-1].content.text == ATTACK_SEED
    assert attacks == [{"improvement": "", "prompt": "a real attack"}]
    assert [m["role"] for m in conv.messages] == ["system", "user", "assistant"]
    assert json.loads(conv.messages[-1]["content"]) == attacks[0]


def test_get_attack_with_seed_still_parses_a_complete_reply():
    attacker = _ScriptedGenerator([VALID_ATTACK])
    manager = _attack_manager(attacker)
    conv = _new_conv()

    attacks = manager.get_attack([conv], ["begin"])

    assert attacker.conversations[0].turns[-1].content.text == ATTACK_SEED
    assert attacks == [{"improvement": "try harder", "prompt": "a real attack"}]
    assert conv.messages[-1] == {"role": "assistant", "content": VALID_ATTACK}


def test_get_attack_does_not_seed_a_branched_conversation():
    attacker = _ScriptedGenerator([VALID_ATTACK])
    manager = _attack_manager(attacker)
    conv = _new_conv()
    conv.parent_id = "some-parent-id"

    manager.get_attack([conv], ["begin"])

    assert attacker.conversations[0].turns[-1].role == "user"


def test_get_attack_keeps_the_user_turn_of_a_branched_conversation():
    attacker = _ScriptedGenerator([VALID_ATTACK])
    manager = _attack_manager(attacker)
    conv = _new_conv()
    conv.parent_id = "some-parent-id"

    manager.get_attack([conv], ["begin"])

    assert conv.messages == [
        {"role": "system", "content": "attacker system prompt"},
        {"role": "user", "content": "begin"},
        {"role": "assistant", "content": VALID_ATTACK},
    ]


@pytest.mark.parametrize(
    "messages, expected",
    [
        (
            [{"role": "user", "content": "begin"}],
            [
                {"role": "user", "content": "begin"},
                {"role": "assistant", "content": "reply"},
            ],
        ),
        (
            [
                {"role": "user", "content": "begin"},
                {"role": "assistant", "content": ATTACK_SEED},
            ],
            [
                {"role": "user", "content": "begin"},
                {"role": "assistant", "content": "reply"},
            ],
        ),
    ],
)
def test_update_last_message_sets_the_assistant_reply(messages, expected):
    from garak.resources.tap.tap_main import TAPConversation

    conv = TAPConversation(messages=messages)

    conv.update_last_message("reply")

    assert conv.messages == expected


def test_run_tap_never_sends_an_empty_prompt_to_the_target():
    _config.load_base_config()
    target = _ScriptedGenerator(["I will not do that."])
    goal = "Refund the order"

    with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8") as report:
        _config.transient.reportfile = report
        _config.transient.report_filename = report.name
        with (
            patch(
                "garak.resources.tap.tap_main.load_generator",
                side_effect=[
                    _ScriptedGenerator(["I cannot help with that."]),
                    _ScriptedGenerator(["[[1]]"]),
                ],
            ),
            patch(
                "garak.resources.tap.tap_main.isinstance",
                side_effect=lambda obj, cls: (
                    cls == garak.generators.openai.OpenAICompatible
                    or isinstance(obj, cls)
                ),
            ),
        ):
            results = run_tap(
                goal=goal,
                target="Sure,",
                target_generator=target,
                attack_max_attempts=2,
                branching_factor=1,
                width=2,
                depth=2,
                n_streams=1,
                save_results=False,
            )

    assert target.received == [], "no attack prompt reached the target"
    assert results == [goal], "falls back to the original goal"
