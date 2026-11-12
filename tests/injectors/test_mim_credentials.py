"""Credential boundaries through the CLI, bindings and conversation notes."""

import asyncio
from contextlib import contextmanager
from dataclasses import asdict
import json
import logging
import pickle
from pathlib import Path

import pytest

from garak import _config, _plugins, cli
from garak.attempt import Conversation
from garak.evidence.collector import Collector
from garak.exception import CapabilityError
from garak.injectors.mim import Mim
from test_mim import _plan, live

_CREDENTIAL = "sentinel-binding-credential-must-stay-private"


@contextmanager
def _debug_file(path: Path):
    """Capture the same root DEBUG records that garak.log receives outside pytest."""
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        yield
    finally:
        root.removeHandler(handler)
        handler.close()


def _configured(tmp_path: Path) -> Path:
    """Write a non-network CLI configuration with a sentinel bind credential."""
    path = tmp_path / "binding.json"
    path.write_text(
        json.dumps(
            {
                "plugins": {
                    "injectors": {
                        "mim": {
                            "service_url": "http://127.0.0.1:9",
                            "bind_credential": _CREDENTIAL,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def _exception_text(error: BaseException) -> str:
    """Include the message and every chained cause or context."""
    messages = []
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        messages.append(repr(error))
        error = error.__cause__ or error.__context__
    return "\\n".join(messages)


def test_cli_run_omits_bind_credential(tmp_path, caplog, capsys):
    """Neither the report nor its HTML summary or CLI output contains the key."""
    config = _configured(tmp_path)
    log_file = tmp_path / "garak.log"
    with caplog.at_level(logging.DEBUG), _debug_file(log_file):
        cli.main(
            [
                "--target_type",
                "test",
                "--target_name",
                "Blank",
                "-p",
                "test.Blank",
                "-g",
                "1",
                "--report_prefix",
                "_garak_test_mim_credentials",
                "--config",
                str(config),
            ]
        )
    report = Path(_config.transient.report_filename)
    records = {
        "log": caplog.text,
        "garak.log": log_file.read_text(encoding="utf-8"),
        "console": "".join(capsys.readouterr()),
        "report": report.read_text(encoding="utf-8"),
        "HTML summary": report.with_suffix(".html").read_text(encoding="utf-8"),
    }
    assert "start_run setup" in records["report"], "the run wrote its report"
    for name, content in records.items():
        assert _CREDENTIAL not in content, f"{name} contains a bind credential"


def test_cli_list_config_redacts_bind_credential(tmp_path, caplog, capsys):
    """Listing the active configuration names but does not print the secret."""
    log_file = tmp_path / "garak.log"
    with caplog.at_level(logging.DEBUG), _debug_file(log_file):
        cli.main(["--list_config", "--config", str(_configured(tmp_path))])
    printed = "".join(capsys.readouterr())
    assert (
        _CREDENTIAL not in printed + caplog.text + log_file.read_text()
    ), "listing masks the key across console and logs"
    assert cli.REDACTED in printed, "listing shows the masked setting"


def test_configured_injector_pickles_but_live_objects_do_not(live):
    """Only the configuration-only component can cross a worker boundary."""
    owner, upstream_url = live
    root = {
        "plugins": {
            "injectors": {
                "mim": {
                    "service_url": owner._url,
                    "bind_credential": owner._credential,
                }
            }
        }
    }
    injector = _plugins.load_plugin("injectors.mim", config_root=root)
    restored = pickle.loads(pickle.dumps(injector))
    assert isinstance(restored, Mim), "a loaded injector can be unpickled"
    assert restored.service_url == owner._url, "service URL survives serialisation"
    assert (
        restored.bind_credential == owner._credential
    ), "configured credential survives serialisation"
    run = restored.start_run()
    try:
        arming = run.arm(
            Collector("pickle-boundary"), upstream_url=upstream_url, plan=_plan()
        )
        for value in (run, arming, arming.tool_block):
            with pytest.raises(TypeError, match="cannot be pickled"):
                pickle.dumps(value)
        assert arming.disarm()[0], "the binding releases after pickle checks"
    finally:
        run.close()


def test_real_binding_credential_boundary(live, caplog, tmp_path):
    """The service accepts only its bind key; evidence and notes exclude it."""
    pytest.importorskip("midojo_mim")
    from fastmcp import Client
    from midojo_mim.service import InterceptorService
    from websockets.asyncio.client import connect
    from websockets.exceptions import InvalidStatus

    upstream_run, upstream_url = live
    service = InterceptorService(_CREDENTIAL)
    url = upstream_run._submit(service.start())
    root = {
        "plugins": {
            "injectors": {
                "mim": {
                    "service_url": url,
                    "bind_credential": _CREDENTIAL,
                }
            }
        }
    }
    injector = _plugins.load_plugin("injectors.mim", config_root=root)
    remote = injector.start_run()
    log_file = tmp_path / "garak.log"
    with (
        caplog.at_level(logging.DEBUG),
        caplog.at_level(logging.DEBUG, logger="websockets"),
        caplog.at_level(logging.DEBUG, logger="fastmcp"),
        _debug_file(log_file),
    ):
        try:
            collector = Collector("credential-boundary")
            arming = remote.arm(collector, upstream_url=upstream_url, plan=_plan())
            block = arming.tool_block

            async def call():
                async with Client(block.tool_url, auth=block.tool_token) as target:
                    return await target.call_tool("weather", {"city": "York"})

            assert upstream_run._submit(call()).content, "tool call completed"
            assert arming.disarm()[0], "service acknowledged release"
            collector.close(0)
            evidence = json.dumps(collector.seal("").to_json())
            for value in (remote, arming, block):
                assert _CREDENTIAL not in repr(value), "repr excludes bind key"
                assert _CREDENTIAL not in str(value), "str excludes bind key"
                assert block.tool_token not in repr(value), "repr excludes tool token"
            assert set(asdict(block)) == {
                "tool_url",
                "tool_token",
                "tools",
            }, "tool block has no bind credential"
            assert _CREDENTIAL not in repr(asdict(block)), "block has no bind key"
            assert _CREDENTIAL not in evidence, "sealed evidence omits bind key"
            assert block.tool_token not in evidence, "sealed evidence omits tool token"

            notes = {
                "tools": [
                    {
                        "server_url": block.tool_url,
                        "authorization": block.tool_token,
                    }
                ]
            }
            conversation = Conversation(notes=notes)
            for value in (
                repr(notes),
                repr(conversation),
                json.dumps(asdict(conversation)),
            ):
                assert _CREDENTIAL not in value, "conversation omits bind key"
                assert (
                    value.count(block.tool_token) == 1
                ), "tool token occurs only where the caller placed it"

            async def missing_header():
                with pytest.raises(InvalidStatus) as refused:
                    await connect(
                        url.replace("http://", "ws://") + "/v1/bindings",
                        subprotocols=["midojo-binding.v1"],
                        proxy=None,
                    )
                return refused.value.response.status_code

            assert (
                upstream_run._submit(missing_header()) == 403
            ), "missing Authorization is refused"
            with pytest.raises(CapabilityError, match="bind_credential") as missing:
                Mim(
                    config_root={
                        "injectors": {
                            "mim": {
                                "service_url": url,
                            }
                        }
                    }
                ).start_run()
            assert _CREDENTIAL not in _exception_text(
                missing.value
            ), "no secret in refusal or its causes"

            wrong = Mim(
                config_root={
                    "injectors": {
                        "mim": {
                            "service_url": url,
                            "bind_credential": "sentinel-wrong-key",
                        }
                    }
                }
            ).start_run()
            try:
                with pytest.raises(CapabilityError, match="upgrade refused") as refusal:
                    wrong.arm(
                        Collector("wrong-key"), upstream_url=upstream_url, plan=_plan()
                    )
                assert "sentinel-wrong-key" not in _exception_text(
                    refusal.value
                ), "wrong key is not echoed by any chained exception"
            finally:
                wrong.close()

            for name, target in (
                ("invalid_plan", upstream_url),
                ("upstream_unavailable", "http://127.0.0.1:9/mcp"),
            ):
                plan = _plan("unknown_tool") if name == "invalid_plan" else _plan()
                with pytest.raises(CapabilityError, match=name) as refusal:
                    remote.arm(Collector(name), upstream_url=target, plan=plan)
                assert _CREDENTIAL not in _exception_text(
                    refusal.value
                ), "bind refusal and its causes omit credential"
                assert block.tool_token not in _exception_text(
                    refusal.value
                ), "bind refusal and its causes omit tool token"
        finally:
            remote.close()
            upstream_run._submit(service.stop())

    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert _CREDENTIAL not in logs, "DEBUG log omits bind key"
    assert block.tool_token not in logs, "DEBUG log omits tool token"
    assert block.tool_token[:8] not in logs, "truncated frames omit token prefixes"
    assert _CREDENTIAL not in log_file.read_text(
        encoding="utf-8"
    ), "garak.log omits bind key"
    assert block.tool_token not in log_file.read_text(
        encoding="utf-8"
    ), "garak.log omits tool token"
