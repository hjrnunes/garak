# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test reading an injection spec document.

``tests/_assets/evidence/injection_spec/corpus.jsonl`` holds what the MiDojo
runtime's reader makes of each document: the document it holds, or the refusal
(class, message, ``gap``, and for a version the version and the versions
supported). The comparison needs no runtime. Where the Garak reader is meant
to read a document differently, the case names the kind of difference
(``differs``) and each kind is stated here. The other tests state the reader's
rules from first principles.
"""

import copy
import dataclasses
import importlib.util
import json
from pathlib import Path
from types import MappingProxyType

import pytest

pytest.importorskip("midojo_spec")

from garak.detectors import _checks  # noqa: E402
from garak.evidence import injection_spec  # noqa: E402
from garak.evidence.injection_spec import (  # noqa: E402
    InjectionSpec,
    InjectionSpecDocumentError,
    InterceptionSettings,
    UnsupportedDocumentVersion,
    UnsupportedRuntime,
)
from garak.exception import GarakException  # noqa: E402

ASSETS = Path(__file__).parents[1] / "_assets" / "evidence" / "injection_spec"
_loader = importlib.util.spec_from_file_location(
    "injection_spec_generate", ASSETS / "generate.py"
)
generate = importlib.util.module_from_spec(_loader)
_loader.loader.exec_module(generate)

LINES = [
    json.loads(line)
    for line in (ASSETS / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
]
HEADER, CASES = LINES[0], LINES[1:]
V1 = "midojo-injection-spec-v1"


def outcome_of(document, base):
    """What the Garak reader makes of ``document``, in the corpus's own terms."""
    try:
        spec = InjectionSpec.from_document(document)
    except InjectionSpecDocumentError as exc:
        error = {"class": type(exc).__name__, "message": str(exc), "gap": exc.gap}
        if isinstance(exc, UnsupportedDocumentVersion):
            error["version"] = repr(exc.version)
            error["supported"] = list(exc.supported)
        return {"error": error}
    held = HEADER["norm"][base or "min"]
    return {"document": generate._changed(held, generate.encode(spec.to_document()))}


def refused(got, exc_class="InjectionSpecDocumentError"):
    assert "error" in got, f"expected a refusal, read {got}"
    assert got["error"]["class"] == exc_class
    assert got["error"]["gap"] is None
    return got["error"]["message"]


# --- the corpus -----------------------------------------------------------------


def test_corpus_is_the_format_this_test_reads():
    assert HEADER["format"] == "garak-injection-spec-corpus-1"
    assert HEADER["kind"] == "header"
    assert all(case["kind"] == "case" for case in CASES)
    names = [case["name"] for case in CASES]
    assert len(names) == len(set(names)), "case names are unique"
    assert len(CASES) > 600, "the corpus covers every member of the document"


def check_same(case, got):
    assert got == case["outcome"], "the reader reads the document as the runtime does"


def check_message(case, got):
    """The refusal is the runtime's, in its class and gap; its wording may differ."""
    expected = case["outcome"]
    assert "error" in expected and "error" in got, "both refuse"
    for key in ("class", "gap", "version", "supported"):
        assert got["error"].get(key) == expected["error"].get(key), key


def check_non_json(case, got):
    """A value outside JSON is refused here; the runtime keeps it or refuses it its own way."""
    message = refused(got)
    assert "JSON-like" in message, "the refusal names the value"


def check_lax_check(case, got):
    """The runtime reads a check the shared check reader refuses."""
    assert "document" in case["outcome"], "the runtime reads the check"
    refused(got)


def check_runtime_crash(case, got):
    """The runtime raises an error that is not a refusal; the document is refused."""
    assert case["outcome"]["error"]["class"] == "KeyError"
    refused(got)


def check_openshell(case, got):
    """A document that declares the openshell runtime is refused, after it passed every check."""
    refused(got, "UnsupportedRuntime")


def check_reply_judge(case, got):
    """The reply judge is not part of the reader yet: the check is refused where it is read."""
    assert "reply_judge needs the reply-judge module" in refused(got)


def check_environment_inference(case, got):
    """MiDojo infers an environment model, so a key that breaks the inference is refused there."""
    assert case["outcome"]["error"]["class"] == "InjectionSpecDocumentError"
    document = generate.build(case["base"], case["ops"])
    assert got == {
        "document": {"environment": generate.encode(document["environment"])}
    }


CHECKS = {
    None: check_same,
    "message": check_message,
    "non-json": check_non_json,
    "lax-check": check_lax_check,
    "runtime-crash": check_runtime_crash,
    "openshell": check_openshell,
    "reply-judge": check_reply_judge,
    "environment-inference": check_environment_inference,
}


def test_every_kind_of_difference_is_stated():
    assert {case.get("differs") for case in CASES} <= set(CHECKS)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_reads_a_document_as_the_runtime_does(case):
    document = generate.build(case["base"], case["ops"])
    got = outcome_of(document, case["base"])
    assert generate.encode(document) == generate.encode(
        generate.build(case["base"], case["ops"])
    ), "reading changes nothing in the document"
    CHECKS[case.get("differs")](case, got)


def test_the_two_bases_read_to_their_norms():
    for name, document in generate.BASES.items():
        spec = InjectionSpec.from_document(generate.decode(document))
        assert generate.encode(spec.to_document()) == HEADER["norm"][name]


# --- what the document says to run ----------------------------------------------


def full():
    return copy.deepcopy(generate.FULL)


def test_supported_versions():
    assert injection_spec.SUPPORTED_DOCUMENT_VERSIONS == (V1,)


def test_a_version_other_than_v1_is_named():
    document = full()
    document["schema_version"] = "midojo-injection-spec-v2"
    with pytest.raises(UnsupportedDocumentVersion) as caught:
        InjectionSpec.from_document(document)
    assert caught.value.version == "midojo-injection-spec-v2"
    assert caught.value.supported == (V1,)
    assert caught.value.gap is None
    assert "midojo-injection-spec-v2" in str(caught.value)


@pytest.mark.parametrize("version", [None, 1, ["midojo-injection-spec-v1"]])
def test_a_version_that_is_not_text_is_refused_as_a_document_error(version):
    document = full()
    document["schema_version"] = version
    with pytest.raises(InjectionSpecDocumentError, match="schema_version") as caught:
        InjectionSpec.from_document(document)
    assert not isinstance(caught.value, UnsupportedDocumentVersion)


def test_a_text_subclass_is_not_a_version():
    class Version(str):
        pass

    document = full()
    document["schema_version"] = Version(V1)
    with pytest.raises(InjectionSpecDocumentError, match="schema_version"):
        InjectionSpec.from_document(document)


@pytest.mark.parametrize(
    "path",
    [(), ("interception",), ("tools",), ("tools", "write_disposition")],
    ids=lambda path: ".".join(path) or "document",
)
@pytest.mark.parametrize("name", ["x", "extra"])
def test_a_closed_object_names_the_first_key_it_does_not_define(path, name):
    document = full()
    level = document
    for step in path:
        level = level[step]
    level[name] = 1
    with pytest.raises(InjectionSpecDocumentError, match="closed") as caught:
        InjectionSpec.from_document(document)
    assert repr(name) in str(caught.value)


def test_a_closed_object_names_the_first_unknown_key_in_document_order():
    document = full()
    document["first"] = 1
    document["second"] = 2
    with pytest.raises(InjectionSpecDocumentError) as caught:
        InjectionSpec.from_document(document)
    assert "'first'" in str(caught.value) and "'second'" not in str(caught.value)


def test_an_unsupported_version_names_the_version_as_given():
    assert UnsupportedDocumentVersion(2).version == 2


@pytest.mark.parametrize(
    "settings",
    [
        {"mode": "agent"},
        {"mode": 1},
        {"upstream": 1},
        {"bind_host": 1},
        {"advertise_host": 1},
        {"port": "80"},
        {"port": True},
        {"port": -1},
        {"port": 65536},
        {"observe_requests": 1},
    ],
    ids=lambda settings: next(iter(settings))
    + "="
    + repr(next(iter(settings.values()))),
)
def test_interception_settings_check_themselves_when_built_directly(settings):
    with pytest.raises((TypeError, ValueError)):
        InterceptionSettings(**settings)


@pytest.mark.parametrize("field", ["upstream", "bind_host", "advertise_host"])
def test_interception_settings_refuse_a_text_subclass(field):
    class Text(str):
        pass

    with pytest.raises(TypeError, match=field):
        InterceptionSettings(**{field: Text("x")})


def test_interception_settings_hold_what_they_are_given():
    settings = InterceptionSettings(
        upstream="http://h/mcp", bind_host="b", advertise_host="a", port=0
    )
    assert (settings.upstream, settings.bind_host, settings.advertise_host) == (
        "http://h/mcp",
        "b",
        "a",
    )
    assert settings.port == 0 and settings.observe_requests is False


def test_the_refusals_are_value_errors_the_collector_family_can_catch():
    for error in (UnsupportedDocumentVersion, UnsupportedRuntime):
        assert issubclass(error, InjectionSpecDocumentError)
    assert issubclass(InjectionSpecDocumentError, ValueError)
    assert issubclass(InjectionSpecDocumentError, GarakException)
    assert InjectionSpecDocumentError("x").gap is None
    assert InjectionSpecDocumentError("x", gap="a-gap").gap == "a-gap"


def test_a_driver_reads_what_to_run():
    spec = InjectionSpec.from_document(full())
    assert spec.runtime_type == "unmanaged"
    assert spec.suite_name == "piggybank"
    assert spec.environment_type == "PiggybankEnvironment"
    assert spec.user_task_id == "user_task_0"
    assert spec.injection_task_id == "injection_task_0"
    assert spec.turns == (
        "Can you look up ORD-101 for me?",
        "Thanks, what does it say?",
    )
    assert spec.interception == InterceptionSettings(
        mode="host",
        upstream="http://127.0.0.1:8770/mcp",
        bind_host="127.0.0.1",
        advertise_host="localhost",
        port=8766,
        observe_requests=True,
    )
    assert spec.tools.allowed == ("lookup_order", "process_refund")
    assert spec.tools.write_disposition_default == "forward"
    assert dict(spec.tools.write_disposition) == {"process_refund": "block"}
    (instruction,) = spec.plan.instructions
    assert instruction.probe_key == generate.KEY
    assert instruction.payload == generate.PAYLOAD
    assert spec.security["tool_call_condition"]["comparisons"][0]["op"] == "eq"
    assert spec.utility["output_contains"] == "ORD-101"
    assert spec.environment["orders"][0]["status"] == "open"


def test_a_document_that_states_nothing_has_the_defaults():
    spec = InjectionSpec.from_document({"schema_version": V1})
    assert spec.runtime_type == "unmanaged"
    assert spec.turns is None
    assert spec.plan.instructions == ()
    assert spec.security is None and spec.utility is None
    assert spec.environment is None
    assert spec.suite_name is None
    assert spec.user_task_id is None and spec.injection_task_id is None
    assert spec.interception == InterceptionSettings()
    assert spec.interception.mode == "host" and spec.interception.upstream is None
    assert spec.tools is None, "no tool policy mirrors every tool"
    assert spec.environment_type == "EvaluationEnvironment"
    assert spec.check("security") is None and spec.check("utility") is None


@pytest.mark.parametrize(
    "suite_name, environment_type",
    [
        ("piggybank", "PiggybankEnvironment"),
        ("my suite", "MySuiteEnvironment"),
        ("minibank", "MinibankEnvironment"),
        ("", "EvaluationEnvironment"),
        (None, "EvaluationEnvironment"),
    ],
)
def test_the_environment_type_is_the_one_the_suite_name_implies(
    suite_name, environment_type
):
    document = {"schema_version": V1, "suite_name": suite_name}
    spec = InjectionSpec.from_document(document)
    assert spec.environment_type == environment_type
    document["environment_type"] = environment_type
    InjectionSpec.from_document(document)
    document["environment_type"] = "OtherEnvironment"
    with pytest.raises(InjectionSpecDocumentError, match="environment_type"):
        InjectionSpec.from_document(document)


def test_the_value_is_read_only():
    spec = InjectionSpec.from_document(full())
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.suite_name = "other"
    with pytest.raises(TypeError):
        spec.security["new"] = 1
    with pytest.raises(TypeError):
        spec.environment["balance"] = 6
    with pytest.raises(TypeError):
        spec.runtime["type"] = "openshell"
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.interception.port = 1
    assert isinstance(spec.environment, MappingProxyType)
    assert isinstance(spec.environment["orders"], tuple)
    assert isinstance(spec.turns, tuple)


def test_the_value_does_not_share_structure_with_the_document():
    document = full()
    spec = InjectionSpec.from_document(document)
    expected = spec.to_document()
    document["environment"]["orders"].append({"id": "Z"})
    document["security"]["tool_call_condition"]["comparisons"].clear()
    document["turns"].append("more")
    document["tools"]["allowed"].append("other")
    assert spec.to_document() == expected


def test_to_document_writes_every_member_and_reads_back_equal():
    spec = InjectionSpec.from_document(full())
    written = spec.to_document()
    assert set(written) == set(generate.FULL), "every member is written"
    assert InjectionSpec.from_document(written).to_document() == written
    bare = InjectionSpec.from_document({"schema_version": V1}).to_document()
    assert bare["verifiers"] == []
    assert bare["environment_type"] is None
    assert InjectionSpec.from_document(bare).to_document() == bare


def test_check_reads_a_fresh_check_of_the_authored_member():
    spec = InjectionSpec.from_document(full())
    first, second = spec.check("security"), spec.check("security")
    assert first is not second and first == second
    assert type(spec.check("utility")).__name__ == "OutputContains"
    with pytest.raises(ValueError, match="security and a utility"):
        spec.check("plan")


# --- what is checked, and in what order -------------------------------------------

MEMBER_REFUSED = {
    "suite_name": 1,
    "user_task_id": 1,
    "injection_task_id": ["x"],
    "environment_type": 1,
    "environment": [],
    "runtime": "unmanaged",
    "turns": "hello",
    "plan": [],
    "security": [],
    "utility": "output_contains",
    "verifiers": ["x"],
    "interception": 1,
    "tools": [],
}


@pytest.mark.parametrize("member, value", MEMBER_REFUSED.items())
def test_every_member_is_checked_whether_or_not_a_grade_reads_it(member, value):
    document = full()
    document[member] = value
    with pytest.raises(InjectionSpecDocumentError):
        InjectionSpec.from_document(document)


def test_a_member_the_version_does_not_define_is_refused():
    document = full()
    document["surprise"] = 1
    with pytest.raises(InjectionSpecDocumentError, match="'surprise'"):
        InjectionSpec.from_document(document)


def test_the_version_is_judged_before_any_member():
    document = full()
    document.update(schema_version="midojo-injection-spec-v0", surprise=1, turns="x")
    with pytest.raises(UnsupportedDocumentVersion):
        InjectionSpec.from_document(document)


def test_the_openshell_runtime_is_refused_after_every_other_member_passed():
    document = full()
    document["runtime"] = {"type": "openshell", "image": "localhost/agent:1"}
    with pytest.raises(UnsupportedRuntime, match="openshell"):
        InjectionSpec.from_document(document)
    document["turns"] = "not a list"
    with pytest.raises(InjectionSpecDocumentError, match="turns") as caught:
        InjectionSpec.from_document(document)
    assert not isinstance(caught.value, UnsupportedRuntime)


def test_an_openshell_predicate_on_an_unmanaged_runtime_is_a_gap():
    for member in ("security", "utility"):
        document = full()
        document[member] = {
            "any_of": [{"output_contains": "a"}, {"process_ran": "curl"}]
        }
        with pytest.raises(InjectionSpecDocumentError, match="OpenShell") as caught:
            InjectionSpec.from_document(document)
        assert caught.value.gap == injection_spec.OPENSHELL_PREDICATE_GAP
        assert member in str(caught.value)


def test_an_openshell_predicate_on_an_openshell_runtime_is_an_unsupported_runtime():
    document = full()
    document["runtime"] = {"type": "openshell", "image": "x"}
    document["security"] = {"not": {"network_call_to": "example.com"}}
    with pytest.raises(UnsupportedRuntime) as caught:
        InjectionSpec.from_document(document)
    assert caught.value.gap is None


@pytest.mark.parametrize(
    "check",
    [
        {"workdir_file_contains": None},
        {"workdir_file_contains": {"path": "x"}},
        {"commands_match_pattern": {"exit_code": 0}},
        {"http_request_to": "example.com"},
        {"http_request_to": {"host": "example.com"}},
        {"http_request_to": {"host": "[.]", "decision": "any"}},
        {"all_of": [{"output_contains": "a"}, {"workdir_file_contains": None}]},
    ],
    ids=lambda check: json.dumps(check),
)
def test_a_malformed_openshell_check_is_refused_without_the_runtime_gap(check):
    for member in ("security", "utility"):
        document = full()
        document[member] = check
        with pytest.raises(InjectionSpecDocumentError) as caught:
            InjectionSpec.from_document(document)
        assert (
            caught.value.gap is None
        ), "the check is malformed before it needs a runtime"


@pytest.mark.parametrize("kind", ["workdir_file_contains", "http_request_to"])
@pytest.mark.parametrize("value", [None, 1, "x", ["x"], []], ids=repr)
def test_an_openshell_value_that_is_not_an_object_is_a_check_error(kind, value):
    with pytest.raises(_checks.CheckError):
        _checks.parse({kind: value})


def test_an_http_request_names_what_it_lacks():
    with pytest.raises(_checks.CheckError, match="decision is required"):
        _checks.parse({"http_request_to": {"host": "a"}})
    with pytest.raises(_checks.CheckError, match="host is required"):
        _checks.parse({"http_request_to": {"decision": "any"}})


def test_an_http_request_host_is_exactly_text():
    class Host(str):
        pass

    with pytest.raises(_checks.CheckError, match="host"):
        _checks.parse({"http_request_to": {"host": Host("a"), "decision": "any"}})


def test_a_well_formed_openshell_check_carries_the_runtime_gap():
    document = full()
    document["security"] = {"http_request_to": {"host": "a", "decision": "any"}}
    with pytest.raises(InjectionSpecDocumentError) as caught:
        InjectionSpec.from_document(document)
    assert caught.value.gap == injection_spec.OPENSHELL_PREDICATE_GAP


def test_a_reply_judge_check_is_refused_until_the_reply_judge_is_read():
    document = full()
    document["security"] = copy.deepcopy(generate.REPLY_JUDGE)
    with pytest.raises(InjectionSpecDocumentError, match="reply-judge module"):
        InjectionSpec.from_document(document)


def test_a_document_member_may_be_any_mapping():
    document = full()
    document["environment"] = MappingProxyType({"a": MappingProxyType({"b": [1]})})
    spec = InjectionSpec.from_document(document)
    assert spec.environment["a"]["b"] == (1,)


def test_a_text_subclass_is_refused_wherever_a_str_is_read():
    class Text(str):
        pass

    for member in ("suite_name", "user_task_id", "injection_task_id"):
        document = full()
        document[member] = Text("x")
        with pytest.raises(InjectionSpecDocumentError, match=member):
            InjectionSpec.from_document(document)
    document = full()
    document["turns"] = [Text("hello")]
    with pytest.raises(InjectionSpecDocumentError, match=r"turns\[0\]"):
        InjectionSpec.from_document(document)


def test_the_environment_nests_one_hundred_levels_and_no_more():
    def nested(levels):
        value = 1
        for _ in range(levels):
            value = {"a": value}
        return value

    document = {"schema_version": V1, "environment": nested(100)}
    assert InjectionSpec.from_document(document).environment["a"]
    document["environment"] = nested(101)
    with pytest.raises(InjectionSpecDocumentError, match="nested at most 100"):
        InjectionSpec.from_document(document)


# --- the OpenShell check the reader leaves unread ---------------------------------


def context():
    return _checks.Context(
        agent_output="",
        pre_environment={},
        post_environment={},
        function_calls=(),
        turns=(),
        evidence_status={},
    )


def test_an_openshell_check_is_read_by_name_and_never_graded():
    check = _checks.parse({"process_ran": "curl"})
    assert isinstance(check, _checks.OpenShellCheck) and check.kind == "process_ran"
    with pytest.raises(_checks.CheckError, match="OpenShell"):
        check.assess(context())
    with pytest.raises(_checks.CheckError, match="OpenShell"):
        _checks.attempt(check, context())
    with pytest.raises(_checks.CheckError, match="OpenShell"):
        _checks.attempt(_checks.parse({"all_of": [{"process_ran": "x"}]}), context())


@pytest.mark.parametrize(
    "raw, reads",
    [
        ({"output_contains": "a"}, False),
        ({"process_ran": "x"}, True),
        ({"not": {"network_call_to": "x"}}, True),
        ({"all_of": [{"output_contains": "a"}, {"workdir_file_exists": "x"}]}, True),
        (
            {
                "any_of": [
                    {
                        "not": {
                            "all_of": [
                                {"http_request_to": {"host": "x", "decision": "any"}}
                            ]
                        }
                    }
                ]
            },
            True,
        ),
        ({"any_of": [{"not": {"all_of": [{"output_contains": "a"}]}}]}, False),
    ],
)
def test_reads_openshell_finds_a_predicate_anywhere(raw, reads):
    assert _checks.reads_openshell(_checks.parse(raw)) is reads
