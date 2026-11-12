# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""No credential reaches the injection-spec probe's logs, reports or diagnostics.

Three kinds of secret pass through a chain: the tool token the target is given,
the credential the injector presents to its service, and whatever a hostile
option value or a service reply repeats. Each test plants a recognisable value
and looks for it everywhere the run writes.
"""

import json
import logging
import os
import pathlib
import traceback
from types import SimpleNamespace

import httpx
import openai
import pytest
import respx
from openai._models import construct_type
from openai.types.responses import Response

pytest.importorskip("midojo_spec")

import garak.exception  # noqa: E402
from garak.exception import (  # noqa: E402
    CapabilityError,
    ConfigFailure,
    GarakException,
)
from garak.generators.openai import OpenAIResponsesGenerator  # noqa: E402
from garak.probes.midojo import TRACE_NOTE, InjectionSpecProbe  # noqa: E402

from injection_chain import (  # noqa: E402
    ScriptedTarget,
    build_probe,
    cli_run,
    entry,
    exception_text,
    report,  # noqa: F401
    world,  # noqa: F401
)

LOOKUP_101 = ("lookup_order", {"order_id": "ORD-101"})
CREDENTIAL_ENV = "MIDOJO_INTERCEPTOR_CREDENTIAL"
ECHOED = "sentinel-echoed-credential-must-never-be-kept"
CHANGED = "sentinel-credential-after-the-change"
SAID = f"the service said {ECHOED} here"


def _kept(attempts, report):
    """Everything a run stores about ``attempts``: their notes, outputs and the report."""
    report.flush()
    kept = [pathlib.Path(report.name).read_text(encoding="utf-8")]
    for attempt in attempts:
        kept.append(
            json.dumps(
                {
                    "notes": attempt.notes,
                    "outputs": [output.notes for output in attempt.outputs],
                    "conversation": attempt.prompt.notes,
                    "stored": attempt.as_dict(),
                },
                default=repr,
            )
        )
    return "\n".join(kept)


def _option_values(sentinel):
    class RevealingValue:
        def __str__(self):
            return sentinel

        __repr__ = __str__

    class RevealingString(str):
        def __str__(self):
            return sentinel

        __repr__ = __str__

    class RevealingInt(int):
        def __str__(self):
            return sentinel

        __repr__ = __str__

    class RevealingFloat(float):
        def __str__(self):
            return sentinel

        __repr__ = __str__

    return {
        "list": [sentinel],
        "tuple": (sentinel,),
        "dict": {sentinel: sentinel},
        "set": {sentinel},
        "bytes": sentinel.encode(),
        "int": RevealingInt(1),
        "float": RevealingFloat(1.0),
        "bool": True,
        "object": RevealingValue(),
        "str_subclass": RevealingString("verify"),
        "empty": "",
        "control": f"\x00{sentinel}\r\n",
        "none": None,
    }


@pytest.mark.parametrize("option", tuple(InjectionSpecProbe.DEFAULT_PARAMS))
@pytest.mark.parametrize("kind", tuple(_option_values("marker")))
def test_every_probe_option_keeps_hostile_values_out_of_diagnostics(
    tmp_path, world, report, caplog, capsys, option, kind
):
    sentinel = f"sentinel-option-{option}-{kind}-end"
    value = _option_values(sentinel)[kind]
    caplog.set_level(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="garak")
    for route in ("config", "assigned"):
        caplog.clear()
        capsys.readouterr()
        attempts = []
        error = None
        try:
            options = {option: value} if route == "config" else {}
            probe = build_probe(tmp_path, entry("Look up ORD-101."), **options)
            if route == "assigned":
                setattr(probe, option, value)
            attempts = list(probe.probe(ScriptedTarget(world, [[LOOKUP_101]])))
        except (GarakException, TypeError, ValueError, OSError) as exc:
            error = exc
        captured = capsys.readouterr()
        records = "\n".join(
            str(record.msg)
            + repr(record.args)
            + record.getMessage()
            + (
                "".join(traceback.format_exception(*record.exc_info))
                if record.exc_info
                else ""
            )
            for record in caplog.records
        )
        where = (option, kind, route)
        assert sentinel not in records + captured.out + captured.err, (
            *where,
            "log or console",
        )
        if error is not None:
            assert sentinel not in exception_text(error), (*where, "exception")
        for attempt in attempts:
            stored = json.dumps(
                {"attempt": attempt.notes, "conversation": attempt.prompt.notes},
                default=repr,
            ) + json.dumps(attempt.as_dict(), default=repr)
            assert sentinel not in stored, (*where, "report notes")


@pytest.mark.parametrize(
    "kind", ["missing", "directory", "relative", "file_parent", "unreadable"]
)
def test_source_path_filesystem_refusals_seal_diagnostics(
    tmp_path, caplog, capsys, kind
):
    sentinel = f"sentinel-source-{kind}"
    source = tmp_path / sentinel
    if kind == "directory":
        source.mkdir()
    elif kind == "relative":
        source = pathlib.Path(sentinel) / "missing.json"
    elif kind == "file_parent":
        source.write_text("not a directory", encoding="utf-8")
        source = source / "missing.json"
    elif kind == "unreadable":
        source.write_text("[]", encoding="utf-8")
        source.chmod(0)
    try:
        if kind == "unreadable":
            try:
                if (
                    os.access(source, os.R_OK)
                    and source.read_text(encoding="utf-8") == "[]"
                ):
                    pytest.skip("the process can read files without mode bits")
            except PermissionError:
                pass
        caplog.set_level(logging.DEBUG)
        with pytest.raises(ConfigFailure) as caught:
            InjectionSpecProbe(
                config_root={
                    "probes": {
                        "midojo": {"InjectionSpecProbe": {"source_path": str(source)}}
                    }
                }
            )
        assert "source_path" in str(caught.value)
        assert sentinel not in exception_text(caught.value)
        output = capsys.readouterr()
        assert sentinel not in output.out + output.err
        assert sentinel not in " ".join(
            record.getMessage() for record in caplog.records
        )
    finally:
        if kind == "unreadable":
            source.chmod(0o600)


@pytest.mark.parametrize("suffix", [".json", ".jsonl"])
def test_source_path_read_error_is_value_free(
    tmp_path, monkeypatch, caplog, capsys, suffix
):
    sentinel = "sentinel-unreadable-source"
    source = tmp_path / (sentinel + suffix)
    source.write_text("[]", encoding="utf-8")
    real_read = pathlib.Path.read_text

    def unreadable(path, *args, **kwargs):
        if path == source:
            raise PermissionError(13, sentinel, str(source))
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "read_text", unreadable)
    caplog.set_level(logging.DEBUG)
    with pytest.raises(ConfigFailure) as caught:
        InjectionSpecProbe(
            config_root={
                "probes": {
                    "midojo": {"InjectionSpecProbe": {"source_path": str(source)}}
                }
            }
        )
    assert "source_path" in str(caught.value)
    assert sentinel not in exception_text(caught.value)
    output = capsys.readouterr()
    assert sentinel not in output.out + output.err
    assert sentinel not in " ".join(record.getMessage() for record in caplog.records)


@pytest.fixture(params=["tool bearer", "injector credential", "environment credential"])
def echoable(request, tmp_path, world, monkeypatch):
    """A probe whose chains open with a credential a service could echo to it."""
    options = {}
    if request.param == "tool bearer":
        world.token = ECHOED
    elif request.param == "injector credential":
        options = {"injector": {"bind_credential": ECHOED}}
    else:
        monkeypatch.setenv(CREDENTIAL_ENV, ECHOED)
    world.echo = ECHOED
    return build_probe(tmp_path, entry("Look up ORD-101."), **options)


PLACES = {
    "result": ([LOOKUP_101], ("result",)),
    "outcome detail": ([LOOKUP_101], ("detail",)),
    "error": ([LOOKUP_101], ("error",)),
    "argument": ([("lookup_order", {"order_id": SAID})], ()),
    "argument key": ([("lookup_order", {SAID: 1})], ()),
}


@pytest.mark.parametrize("place", PLACES)
def test_a_trace_holds_no_credential_wherever_a_service_could_echo_one(
    place, echoable, world, report
):
    calls, world.echo_places = PLACES[place]

    attempts = list(echoable.probe(ScriptedTarget(world, [calls])))

    kept = _kept(attempts, report)
    assert ECHOED not in kept, f"the {place} is kept without the credential"
    assert "[redacted] here" in kept, "and with the gap marked"


@pytest.mark.parametrize("source", ["injector credential", "environment credential"])
def test_a_credential_changed_after_the_chain_opened_is_still_scrubbed(
    source, tmp_path, world, report, monkeypatch
):
    """The service echoes what it received at the bind, not what the probe holds now."""
    options = {}
    if source == "injector credential":
        options = {"injector": {"bind_credential": ECHOED}}
    else:
        monkeypatch.setenv(CREDENTIAL_ENV, ECHOED)
    world.echo = ECHOED
    world.echo_places = ("result",)
    probe = build_probe(tmp_path, entry("Look up ORD-101."), **options)
    target = ScriptedTarget(world, [[LOOKUP_101]])
    original = target.generate

    def change_then_answer(prompt, generations_this_call=1):
        if source == "injector credential":
            probe._coordinator.injector.bind_credential = CHANGED
        else:
            monkeypatch.setenv(CREDENTIAL_ENV, CHANGED)
        return original(prompt, generations_this_call)

    target.generate = change_then_answer
    attempts = list(probe.probe(target))

    kept = _kept(attempts, report)
    assert ECHOED not in kept, "the credential the chain opened with is scrubbed"
    assert "[redacted] here" in kept, "and the gap is marked"


@pytest.mark.parametrize(
    "change", ["after the service reads", "before the service reads"]
)
def test_a_credential_changed_while_the_evaluation_opens_is_still_scrubbed(
    change, tmp_path, world, report, monkeypatch
):
    """The open reads the credential, so the probe samples it on both sides of the open."""
    monkeypatch.setenv(CREDENTIAL_ENV, ECHOED)
    world.echo_places = ("result",)

    def bind_while_the_credential_changes():
        if change == "before the service reads":
            monkeypatch.setenv(CREDENTIAL_ENV, CHANGED)
        world.echo = os.environ[CREDENTIAL_ENV]
        if change == "after the service reads":
            monkeypatch.setenv(CREDENTIAL_ENV, CHANGED)

    world.on_arm = bind_while_the_credential_changes
    probe = build_probe(tmp_path, entry("Look up ORD-101."))

    attempts = list(probe.probe(ScriptedTarget(world, [[LOOKUP_101]])))

    received = world.echo
    assert received in (ECHOED, CHANGED), "the service read one of the two values"
    kept = _kept(attempts, report)
    assert received not in kept, "the credential the service received is scrubbed"
    assert "[redacted] here" in kept, "and the gap is marked"


@pytest.mark.parametrize("failing", ["the binding", "the environment"])
def test_a_refusal_that_repeats_a_credential_leaves_with_it_masked(
    failing, tmp_path, world, report, caplog
):
    """An injector's or an environment's refusal is its own text, which can repeat what it was handed."""
    credential = "sentinel-bind-secret-echo"
    token = "sentinel-live-tool-token"
    world.token = token
    if failing == "the binding":
        world.arm_error = CapabilityError(f"service refused {credential}")
    else:
        world.open_error = CapabilityError(f"host refused {credential} and {token}")
    probe = build_probe(
        tmp_path,
        entry("Look up ORD-101."),
        injector={"bind_credential": credential},
        environment="fake",
    )

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(ConfigFailure) as caught:
            probe.probe(ScriptedTarget(world))

    everything = "\n".join(
        [exception_text(caught.value), *(r.getMessage() for r in caplog.records)]
    )
    assert "refused [redacted]" in everything, "the refusal still says what was refused"
    assert credential not in everything, "the injector's credential is masked"
    assert token not in everything, "so is the tool token, once the binding issued it"


class BearerLeakingTarget(ScriptedTarget):
    """A target that sends the tool bearer it was given to a tool, which echoes it back."""

    def __init__(self, world):
        super().__init__(world)
        self.bearers = []

    def generate(self, prompt, generations_this_call=1):
        bearer = prompt.notes["tools"][0]["authorization"]
        self.bearers.append(bearer)
        self.script = [[("lookup_order", {"order_id": bearer})]]
        return super().generate(prompt, generations_this_call)


def test_a_bearer_the_target_sends_to_a_tool_stays_out_of_the_trace_and_the_report(
    tmp_path, world, report
):
    probe = build_probe(tmp_path, entry("Look up ORD-101."))
    target = BearerLeakingTarget(world)

    (attempt,) = probe.probe(target)

    (bearer,) = target.bearers
    (call,) = attempt.notes[TRACE_NOTE]["tool_calls"]
    assert call["arguments"]["order_id"] == "[redacted]", "the call is kept, gap marked"
    assert "[redacted]" in call["result"], "and the tool's echo of it"
    (kept,) = attempt.outputs[0].notes["tool_calls"]
    assert kept["arguments"]["order_id"] == "[redacted]", "so is the call on the output"
    assert bearer and bearer not in _kept(
        [attempt], report
    ), "no record holds the bearer"


def test_a_tool_block_and_bind_credential_never_enter_report_notes(
    tmp_path, world, report, caplog
):
    token = "sentinel-live-tool-token"
    credential = "sentinel-bind-credential"
    world.token = token
    probe = build_probe(
        tmp_path,
        entry("Look up ORD-101."),
        injector={"bind_credential": credential},
    )
    target = ScriptedTarget(world)
    original = target.generate

    def check_then_answer(prompt, generations_this_call=1):
        assert (
            prompt.notes["tools"][0]["authorization"] == token
        ), "the generator receives the live MCP credential"
        return original(prompt, generations_this_call)

    target.generate = check_then_answer
    with caplog.at_level(logging.DEBUG):
        attempts = list(probe.probe(target))

    kept = _kept(attempts, report)
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert token not in kept + logged, "no report line or record holds the tool token"
    assert credential not in kept + logged, "nor the bind credential"
    assert (
        attempts[0].prompt.notes["tools"][0]["authorization"] == "[redacted]"
    ), "the serialisable conversation has a redacted block"


QUOTED_TOKEN = 'quoted-"\\-credential'


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (400, openai.BadRequestError),
        (418, openai.APIStatusError),
        (401, openai.AuthenticationError),
        (None, None),
    ],
    ids=["400", "418", "401", "structured-failed-response"],
)
def test_responses_error_echoing_live_probe_bearer_stays_out_of_evidence(
    tmp_path, world, monkeypatch, caplog, capsys, report, status, kind
):
    """The real Responses generator receives the probe's bearer and scrubs echoes."""
    observed = []
    escaped = json.dumps(QUOTED_TOKEN)[1:-1]
    world.token = QUOTED_TOKEN

    def reject(**create_args):
        bearer = create_args["tools"][0]["authorization"]
        observed.append(bearer)
        body = f"rejected plain={bearer}; escaped={escaped}; bare={bearer}"
        if status is None:
            # the API answers 200 with a response that failed and echoes the request
            return construct_type(
                type_=Response,
                value={
                    "id": "resp_1",
                    "object": "response",
                    "created_at": 0,
                    "model": "test-model",
                    "status": "failed",
                    "parallel_tool_calls": True,
                    "tool_choice": "auto",
                    "tools": [],
                    "output": [
                        {
                            "type": "mcp_call",
                            "id": f"mc_{bearer}",
                            "name": "lookup_order",
                            "arguments": "{}",
                            "server_label": "target",
                            "status": "failed",
                            "output": body,
                            "error": body,
                        }
                    ],
                    "error": {"code": "server_error", "message": body},
                },
            )
        response = httpx.Response(
            status,
            request=httpx.Request("POST", "http://offline.test/v1/responses"),
            text=body,
        )
        raise kind(message=body, response=response, body=body)

    client = SimpleNamespace(
        responses=SimpleNamespace(create=reject), close=lambda: None
    )
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: client)
    generator = OpenAIResponsesGenerator(
        name="test-model",
        config_root={
            "generators": {
                "openai": {"OpenAIResponsesGenerator": {"api_key": "offline-test-key"}}
            }
        },
    )
    probe = build_probe(tmp_path, entry("Look up ORD-101."))
    caplog.set_level(logging.DEBUG)
    generator_errors = []
    if status == 401:
        original_call = generator._call_model

        def record_exception(*args, **kwargs):
            try:
                return original_call(*args, **kwargs)
            except garak.exception.GarakException as error:
                generator_errors.append(str(error))
                raise

        monkeypatch.setattr(generator, "_call_model", record_exception)
        assert (
            probe.probe(generator) == []
        ), "the iterative base stops on authentication failure"
        assert generator_errors, "the generator raises its own scrubbed exception"
        evidence = generator_errors[0]
        assert "authentication failed" in evidence
        assert probe._live == {}, "and the chain is closed"
    else:
        (attempt,) = probe.probe(generator)
        notes = attempt.outputs[0].notes
        if status is None:
            assert attempt.notes[TRACE_NOTE]["error"] == "ResponsesNotCompleted"
            assert "[redacted]" in notes["response"]["error"]["message"]
            (call,) = notes["generator_tool_calls"]
            assert "[redacted]" in call["error"]
            assert "[redacted]" in call["output"]
            assert call["id"] == "mc_[redacted]", "an id the API chose can echo it too"
            assert notes["output_items"][0]["id"] == "mc_[redacted]"
        else:
            assert attempt.notes[TRACE_NOTE]["error"] == kind.__name__
            assert notes["error"]["type"] == kind.__name__
            assert "[redacted]" in notes["error"]["body"]
        evidence = _kept([attempt], report)
    assert observed, "the real generator sent the live tool block to its client"
    bearer = observed[0]
    assert bearer == QUOTED_TOKEN != "offline-test-key"
    output = capsys.readouterr()
    recorded = "\n".join(
        record.getMessage()
        + (
            ""
            if record.exc_info is None
            else "".join(traceback.format_exception(*record.exc_info))
        )
        for record in caplog.records
    )
    for secret in (bearer, escaped):
        assert secret not in evidence + recorded + output.out + output.err, (
            status,
            "generator error echoed a request credential",
        )


def test_a_response_header_echoing_the_live_bearer_stays_out_of_every_record(
    tmp_path, world, caplog, capsys, report
):
    """The real generator and client log a response before the probe sees it."""
    seen = []

    def answer(request):
        bearer = json.loads(request.content)["tools"][0]["authorization"]
        seen.append(bearer)
        reply = {
            "id": "resp_1",
            "object": "response",
            "created_at": 0,
            "model": "test-model",
            "status": "completed",
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "output": [
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": "done", "annotations": []}
                    ],
                }
            ],
        }
        return httpx.Response(
            200, headers={"X-Echoed-Token": bearer, "X-Request-Id": "req_1"}, json=reply
        )

    generator = OpenAIResponsesGenerator(
        name="test-model",
        config_root={
            "generators": {
                "openai": {"OpenAIResponsesGenerator": {"api_key": "offline-test-key"}}
            }
        },
    )
    probe = build_probe(tmp_path, entry("Look up ORD-101."))
    caplog.set_level(logging.DEBUG)
    with respx.mock(base_url="https://api.openai.com/v1") as router:
        router.post("/responses").mock(side_effect=answer)
        (attempt,) = probe.probe(generator)

    (bearer,) = seen
    output = capsys.readouterr()
    everything = "\n".join(
        [
            *(logging.Formatter().format(record) for record in caplog.records),
            _kept([attempt], report),
            output.out,
            output.err,
        ]
    )
    assert "req_1" in everything, "the client still logs the response"
    assert bearer and bearer not in everything, "but never the bearer a header echoed"


SENTINEL = "sentinel-probe-option-credential-must-never-be-written"
REDACTION_MARKER = "[redacted]"
SERVICE_URL = "http://127.0.0.1:8767"

# Every spelling that hands the probe the credential it no longer reads, which
# an old configuration still holds: the log masks it before the probe refuses it.
MOVED_ROUTES = [
    "probe_options",
    "probe_options_equals",
    "probe_option_file",
    "config",
    "generator_options_equals",
]
# The routes that reach the probe itself, so it refuses the moved option
PROBE_ROUTES = MOVED_ROUTES[:4]


def _moved_route(tmp_path, route):
    """The CLI arguments that give the probe its old credential option by one route."""
    options = {
        "midojo.InjectionSpecProbe": {
            "interceptor_url": SERVICE_URL,
            "interceptor_credential": SENTINEL,
        }
    }
    if route == "probe_options":
        return ["--probe_options", json.dumps(options)]
    if route == "probe_options_equals":
        return [f"--probe_options={json.dumps(options)}"]
    if route == "probe_option_file":
        option_file = tmp_path / "probe_options.json"
        option_file.write_text(json.dumps(options), encoding="utf-8")
        return ["--probe_option_file", str(option_file)]
    if route == "generator_options_equals":
        # the generator never reads it: the raw argument is what the log must mask
        generator_options = {"test.Blank": {"interceptor_credential": SENTINEL}}
        return [f"--generator_options={json.dumps(generator_options)}"]
    config_file = tmp_path / "credential.yaml"
    config_file.write_text(
        "plugins:\n"
        "  probes:\n"
        "    midojo.InjectionSpecProbe:\n"
        f"      interceptor_url: {SERVICE_URL}\n"
        f"      interceptor_credential: {SENTINEL}\n",
        encoding="utf-8",
    )
    return ["--config", str(config_file)]


def _injector_route(tmp_path):
    """The CLI arguments that set the injector's service pair where it reads them now."""
    config_file = tmp_path / "injector.yaml"
    config_file.write_text(
        "plugins:\n"
        "  injectors:\n"
        "    mim:\n"
        f"      service_url: {SERVICE_URL}\n"
        f"      bind_credential: {SENTINEL}\n",
        encoding="utf-8",
    )
    return ["--config", str(config_file)]


@pytest.mark.parametrize("route", MOVED_ROUTES)
def test_the_cli_masks_the_moved_credential_option_and_the_probe_refuses_it(
    tmp_path, route
):
    output, log_text, reports = cli_run(tmp_path, _moved_route(tmp_path, route))

    assert "DEBUG" in log_text, "garak.log holds the run's debug lines"
    assert SENTINEL not in log_text + output, "no record holds the credential"
    if route in ("probe_options", "probe_options_equals", "generator_options_equals"):
        assert (
            "interceptor_credential" in log_text
        ), "the log still records the option the run was given"
        assert REDACTION_MARKER in log_text, "with its value masked"
    if route in PROBE_ROUTES:
        assert (
            "plugins.injectors.mim.bind_credential" in log_text
        ), "the probe's refusal names where the credential is set now"
    assert reports, f"the run wrote no report: {output[-2000:]}"
    report_text = "\n".join(path.read_text(encoding="utf-8") for path in reports)
    assert SENTINEL not in report_text, "report.jsonl never records the credential"


def test_the_cli_run_keeps_the_injector_credential_out_of_every_record(tmp_path):
    """The injector reads its service pair from the plugin configuration."""
    output, log_text, reports = cli_run(tmp_path, _injector_route(tmp_path))

    assert (
        "probe start for garak.probes.midojo.InjectionSpecProbe" in log_text
    ), "the probe takes the configured injector, so the run reaches the bind"
    assert SENTINEL not in log_text + output, "no record holds the credential"
    assert reports, f"the run wrote no report: {output[-2000:]}"
    report_text = "\n".join(path.read_text(encoding="utf-8") for path in reports)
    assert SENTINEL not in report_text, "report.jsonl never records the credential"
