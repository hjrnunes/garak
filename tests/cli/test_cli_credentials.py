# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The logged command line never holds a credential given in a plugin option."""

import argparse
import json
import logging
from pathlib import Path

import pytest

from garak import _config, cli

KEY = "sentinel-api-key-must-not-be-logged"
AUTHORIZATION = "sentinel-tool-authorization-must-not-be-logged"
HEADER = "sentinel-tool-header-must-not-be-logged"
SECRETS = (KEY, AUTHORIZATION, HEADER)

OPTIONS = {
    "openai": {
        "OpenAIResponsesGenerator": {
            "api_key": KEY,
            "uri": "http://localhost:8321/v1/",
            "tools": [
                {
                    "type": "mcp",
                    "server_label": "banking",
                    "server_url": "http://localhost:8888/sse",
                    "authorization": AUTHORIZATION,
                    "headers": {"Authorization": f"Bearer {HEADER}", "X-Trace": "t1"},
                }
            ],
        }
    }
}


def test_redaction_masks_credentials_at_any_depth_and_keeps_the_rest():
    redacted = cli.redact_credentials(OPTIONS)

    generator = redacted["openai"]["OpenAIResponsesGenerator"]
    tool = generator["tools"][0]
    assert generator["api_key"] == cli.REDACTED
    assert tool["authorization"] == cli.REDACTED
    assert tool["headers"]["Authorization"] == cli.REDACTED, "header names fold case"
    assert generator["uri"] == "http://localhost:8321/v1/", "other options stay"
    assert tool["headers"]["X-Trace"] == "t1", "other headers stay"
    assert (
        OPTIONS["openai"]["OpenAIResponsesGenerator"]["api_key"] == KEY
    ), "the options in use are not changed"


@pytest.mark.parametrize("prefix", ["", "--generator_options=", "-o="])
def test_redaction_masks_a_json_string_in_every_option_spelling(prefix):
    redacted = cli.redact_credentials(prefix + json.dumps(OPTIONS))

    assert not any(secret in redacted for secret in SECRETS)
    assert cli.REDACTED in redacted
    assert redacted.startswith(prefix), "the option name before the = is kept"


@pytest.mark.parametrize("prefix", ["", "--generator_options=", "-o="])
@pytest.mark.parametrize(
    "body",
    [
        f'{{"api_key": "{KEY}"',
        f'{{"tools": [{{"Authorization": "{KEY}"}}',
        f'[{{"x-api-key": "{KEY}',
    ],
)
def test_redaction_masks_unparsable_json_that_names_a_credential(prefix, body):
    redacted = cli.redact_credentials(prefix + body)

    assert KEY not in redacted, "a half-typed option is logged before its parse error"
    assert redacted == f"{prefix}{cli.REDACTED}", "the option name before the = is kept"


@pytest.mark.parametrize(
    "text",
    [
        "--probe_option_file=options=1.json",
        "plain words, with api_key in them",
        '{"temperature": 0.2}',
        "{not json",
        "[1, 2, 3]",
    ],
)
def test_redaction_leaves_text_without_a_credential_unchanged(text):
    assert cli.redact_credentials(text) == text


def test_redaction_walks_namespaces_lists_and_tuples():
    options = json.dumps(OPTIONS)
    namespace = argparse.Namespace(generator_options=options, generations=1)

    redacted = cli.redact_credentials(
        (namespace, ["--generator_options", options], ("generator_options", options))
    )

    assert not any(secret in repr(redacted) for secret in SECRETS)
    assert redacted[0].generations == 1, "other arguments stay"
    assert isinstance(redacted[2], tuple) and isinstance(redacted[1], list)
    assert namespace.generator_options == options, "the arguments in use are kept"


@pytest.mark.parametrize("spelling", ["token", "equals"])
def test_the_cli_log_never_records_a_plugin_option_credential(spelling, caplog):
    options = json.dumps(OPTIONS)
    if spelling == "token":
        arguments = ["--generator_options", options]
    else:
        arguments = [f"--generator_options={options}"]

    with caplog.at_level(logging.DEBUG):
        cli.main(["--list_generators", *arguments])

    assert "args - raw argument string received" in caplog.text
    assert "non-config params" in caplog.text
    assert not any(
        secret in caplog.text for secret in SECRETS
    ), "no logged argument holds a credential"
    assert cli.REDACTED in caplog.text, "the log still shows the option was given"


URL_KEYS = ("interceptor_url", "upstream")
HOST = "localhost:8767/proxy?keep=value"
# a URL the probe will refuse, still logged first: its user info is the secret
USERINFO_FORMS = [
    ("https://", "reader:sentinel-pass"),
    ("HTTP://", "reader:sentinel-pass"),
    ("9http://", "reader:sentinel-pass"),
    ("://", "reader:sentinel-pass"),
    ("//", "reader:sentinel-pass"),
    ("", "reader:sentinel-pass"),
    ("https://", "reader:sentinel/pass"),
    ("https://", "reader:sentinel?pass#end"),
    ("https://", "reader:sentinel@pass"),
    ("https://", "sentinel-only-user"),
]


@pytest.mark.parametrize("scheme,userinfo", USERINFO_FORMS)
def test_url_userinfo_is_masked_in_every_form_a_url_can_be_mistyped(scheme, userinfo):
    masked = cli._redact_url_userinfo(f"{scheme}{userinfo}@{HOST}")

    assert masked == f"{scheme}{cli.REDACTED}@{HOST}", "only the user info goes"


@pytest.mark.parametrize(
    "url",
    [
        f"http://{HOST}",
        "ws://localhost:8767/mcp#section",
        "localhost:8767",
        "",
    ],
)
def test_a_url_without_user_info_is_unchanged(url):
    assert cli._redact_url_userinfo(url) == url


def test_masked_user_info_stays_masked():
    once = cli._redact_url_userinfo(f"https://reader:pass@{HOST}")

    assert cli._redact_url_userinfo(once) == once, "masking twice changes nothing"


@pytest.mark.parametrize("key", URL_KEYS)
@pytest.mark.parametrize("route", ["dict", "json", "equals"])
@pytest.mark.parametrize("scheme,userinfo", USERINFO_FORMS)
def test_a_malformed_url_option_is_masked_whichever_way_it_is_given(
    key, route, scheme, userinfo
):
    url = f"{scheme}{userinfo}@{HOST}"
    options = {"midojo.InjectionSpecProbe": {key: url, "other_url": url}}
    given = {
        "dict": options,
        "json": json.dumps(options),
        "equals": f"--probe_options={json.dumps(options)}",
    }[route]

    redacted = cli.redact_credentials(given)

    masked = f"{scheme}{cli.REDACTED}@{HOST}"
    rendered = json.dumps(redacted) if route == "dict" else redacted
    assert json.dumps(masked)[1:-1] in rendered, "the named option is masked"
    assert rendered.count(json.dumps(userinfo)[1:-1]) == 1, "only that option"


def _route_args(route, tmp_path):
    """Return CLI arguments that give generator options with credentials by a route."""
    options = {"test": {"Blank": OPTIONS["openai"]["OpenAIResponsesGenerator"]}}
    if route == "option":
        return ["--generator_options", json.dumps(options)]
    if route == "option_file":
        options_file = tmp_path / "options.json"
        options_file.write_text(json.dumps(options), encoding="utf-8")
        return ["--generator_option_file", str(options_file)]
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        json.dumps({"plugins": {"generators": options}}), encoding="utf-8"
    )
    return ["--config", str(config_file)]


def _run_with_credentials(route, tmp_path):
    """Run the CLI with generator options that carry credentials, by the given route."""
    route_args = _route_args(route, tmp_path)
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
            f"_garak_test_cli_credentials_{route}",
            *route_args,
        ]
    )


@pytest.mark.parametrize("route", ["option", "option_file", "config"])
def test_a_run_keeps_plugin_option_credentials_out_of_its_records(
    route, tmp_path, caplog, capsys
):
    with caplog.at_level(logging.DEBUG):
        _run_with_credentials(route, tmp_path)

    report = Path(_config.transient.report_filename)
    records = {
        "garak.log": caplog.text,
        "console": "".join(capsys.readouterr()),
        "report": report.read_text(encoding="utf-8"),
        "html summary": report.with_suffix(".html").read_text(encoding="utf-8"),
    }
    assert "start_run setup" in records["report"], "the run wrote its report"
    for name, text in records.items():
        assert not any(
            secret in text for secret in SECRETS
        ), f"the {name} holds no credential"


@pytest.mark.parametrize("route", ["option", "option_file", "config"])
def test_list_config_prints_no_plugin_option_credential(route, tmp_path, capsys):
    cli.main(["--list_config", *_route_args(route, tmp_path)])

    printed = capsys.readouterr().out
    assert "cli args:" in printed and "_config:" in printed, "the config is listed"
    assert not any(
        secret in printed for secret in SECRETS
    ), "the console listing holds no credential"
    assert cli.REDACTED in printed, "the listing still shows a credential was set"
