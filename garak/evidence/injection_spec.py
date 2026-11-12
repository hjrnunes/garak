"""The injection spec document: read in full, checked in full, held as one frozen value.

``InjectionSpec.from_document(document)`` reads a ``midojo-injection-spec-v1``
document the way the MiDojo runtime's reader does, with the runtime's refusals:
the same classes and ``gap``, in the same order, so a document one refuses the
other refuses for the same reason. The wording is the runtime's except where a
check is refused, which the check reader words itself. The document is closed:
a member or a setting the version does not define is refused, and so is a
``verifiers`` list that names a verifier. A version other than v1 raises
:class:`UnsupportedDocumentVersion`, naming it.

The reader checks every member, including the ones a grade never reads (the
suite and task IDs, the environment and its type, the runtime, the turns, the
interception settings and the tool policy), because a driver that runs a
document must not run a half-checked one. It does not check what only MiDojo's
models can: it builds no environment model, so a document whose environment
holds a key that breaks MiDojo's model inference (such as ``model_config``) is
read here, and a runtime definition other than ``unmanaged`` is never validated
beyond its ``type``.

Where the reader is stricter than the runtime, it refuses a document the
runtime reads: a value outside JSON (bytes, a date), and a malformed check that
the runtime's registry accepts and the shared check reader does not
(``output_contains: 1``). Where the runtime fails with a bare ``KeyError`` (an
``env_field_equals`` without a ``value``, a ``workdir_file_contains`` without a
``path``), this reader refuses the document as :class:`InjectionSpecDocumentError`.
A ``reply_judge`` check is refused where it is read, until the reply judge
lands. NaN and infinity are held as they are, as the runtime holds them.

An OpenShell predicate is read, not graded: its value is checked as the
runtime's constructors check it (a ``workdir_file_contains`` needs a ``path``
and ``contains``, an ``http_request_to`` needs a host and a ``decision``) and
refused with no ``gap`` when it fails, before the gap a well-formed one raises.
What this reader does not run, it refuses last, after every other member
passed: :class:`UnsupportedRuntime` names a document that needs the ``openshell``
agent_runtime.

The held value mirrors the runtime's: ``runtime``, ``turns``, ``plan``,
``security``, ``utility``, ``environment``, the three provenance IDs,
``interception`` (an :class:`InterceptionSettings`) and ``tools`` (a
``midojo_spec.policy.ToolPolicy``, or None for every tool). It is read-only:
mappings become read-only views and lists become tuples.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from garak.detectors import _checks
from garak.evidence.codec import injection_codec, policy_codec
from garak.evidence.records import thaw
from garak.exception import GarakException

DOCUMENT_V1 = "midojo-injection-spec-v1"
SUPPORTED_DOCUMENT_VERSIONS = (DOCUMENT_V1,)
MAX_NESTING_DEPTH = 100
OPENSHELL_PREDICATE_GAP = "check:openshell-predicate-on-unmanaged-runtime"
EVALUATION_NAME = "evaluation"

_MEMBERS = frozenset(
    {
        "schema_version",
        "suite_name",
        "user_task_id",
        "injection_task_id",
        "environment_type",
        "environment",
        "runtime",
        "turns",
        "plan",
        "security",
        "utility",
        "verifiers",
        "interception",
        "tools",
    }
)
_INTERCEPTION_FIELDS = frozenset(
    {"mode", "upstream", "bind_host", "advertise_host", "port", "observe_requests"}
)
_TOOLS_FIELDS = frozenset({"allowed", "write_disposition"})
_WRITE_DISPOSITION_FIELDS = frozenset({"default", "tools"})
_RUNTIME_TYPES = ("unmanaged", "openshell")
_SCALARS = (str, bool, int, float, type(None))


class InjectionSpecDocumentError(GarakException, ValueError):
    """A document refused: an unknown key, a wrong type, or a check that does not fit the runtime.

    ``gap`` names why a check does not fit the runtime the document declares
    (``check:openshell-predicate-on-unmanaged-runtime``); it is None for every
    other refusal.
    """

    def __init__(self, message: str, *, gap: str | None = None) -> None:
        self.gap = gap
        super().__init__(message)


class UnsupportedDocumentVersion(InjectionSpecDocumentError):
    """A document whose ``schema_version`` is not one this reader reads. ``version`` names it."""

    def __init__(self, version: object) -> None:
        self.version = version
        self.supported = SUPPORTED_DOCUMENT_VERSIONS
        super().__init__(
            f"injection spec document version {version!r} is not supported; "
            f"supported: {', '.join(SUPPORTED_DOCUMENT_VERSIONS)}"
        )


class UnsupportedRuntime(InjectionSpecDocumentError):
    """A document that passed every check but needs the ``openshell`` agent_runtime, which is not run here."""


class _CheckGap(ValueError):
    """A check needs evidence the declared runtime cannot collect."""

    def __init__(self, message: str, gap: str) -> None:
        self.gap = gap
        super().__init__(message)


def _exact_str(value: Any, what: str) -> str:
    """``value`` when it is exactly a ``str``; a subclass could carry mutable state."""
    if type(value) is not str:
        raise TypeError(f"{what} must be a str, not {type(value).__name__}")
    return value


@dataclass(frozen=True)
class InterceptionSettings:
    """Where the interception layer listens and what it exposes.

    ``mode`` is ``host``. ``upstream`` is the MCP endpoint the layer forwards
    to, or None when the driver supplies it. ``bind_host`` and
    ``advertise_host`` say where the layer listens and the address the agent
    uses; ``port`` None or 0 means a free port. ``observe_requests`` records
    the requests the layer sees.
    """

    mode: str = "host"
    upstream: str | None = None
    bind_host: str | None = None
    advertise_host: str | None = None
    port: int | None = None
    observe_requests: bool = False

    def __post_init__(self) -> None:
        if type(self.mode) is not str or self.mode != "host":
            raise ValueError(f"interception mode must be 'host', not {self.mode!r}")
        for name in ("upstream", "bind_host", "advertise_host"):
            value = getattr(self, name)
            if value is not None:
                _exact_str(value, f"interception {name}")
        if self.port is not None:
            if type(self.port) is not int:
                raise TypeError(
                    f"interception port must be an int or None, not {type(self.port).__name__}"
                )
            if not 0 <= self.port <= 65535:
                raise ValueError("interception port must be between 0 and 65535")
        if type(self.observe_requests) is not bool:
            raise TypeError(
                f"observe_requests must be a bool, not {type(self.observe_requests).__name__}"
            )


@dataclass(frozen=True)
class InjectionSpec:
    """What a document says to run: read-only once read.

    Build one with :meth:`from_document`, which checks everything this class
    holds. ``turns`` is the user's messages, or None when the host drives the
    conversation. ``security`` and ``utility`` are the checks as authored, or
    None; :meth:`check` reads one. ``tools`` is the tool policy (None mirrors
    every upstream tool and blocks no write). The provenance IDs are None
    unless a suite compiled the document.
    """

    runtime: Mapping[str, Any]
    turns: tuple[str, ...] | None
    plan: Any
    security: Mapping[str, Any] | None
    utility: Mapping[str, Any] | None
    environment: Mapping[str, Any] | None
    suite_name: str | None
    user_task_id: str | None
    injection_task_id: str | None
    interception: InterceptionSettings
    tools: Any

    @property
    def runtime_type(self) -> str:
        """The agent_runtime type: always ``unmanaged``, the one this reader holds."""
        return self.runtime["type"]

    @property
    def environment_type(self) -> str:
        """The name of the environment model the suite name implies, as MiDojo infers it."""
        label = (self.suite_name or EVALUATION_NAME).title().replace(" ", "")
        return f"{label}Environment"

    def check(self, name: str) -> Any:
        """A fresh check read from the ``security`` or ``utility`` member, or None without one."""
        if name not in ("security", "utility"):
            raise ValueError(
                f"a document has a security and a utility check, not {name!r}"
            )
        authored = getattr(self, name)
        return None if authored is None else _checks.parse(thaw(authored))

    def to_document(self) -> dict[str, Any]:
        """The document form, every member written: reading it again gives an equal value."""
        return {
            "schema_version": DOCUMENT_V1,
            "suite_name": self.suite_name,
            "user_task_id": self.user_task_id,
            "injection_task_id": self.injection_task_id,
            "environment_type": (
                None if self.suite_name is None else self.environment_type
            ),
            "environment": None if self.environment is None else thaw(self.environment),
            "runtime": thaw(self.runtime),
            "turns": None if self.turns is None else list(self.turns),
            "plan": self.plan.to_data(),
            "security": None if self.security is None else thaw(self.security),
            "utility": None if self.utility is None else thaw(self.utility),
            "verifiers": [],
            "interception": {
                "mode": self.interception.mode,
                "upstream": self.interception.upstream,
                "bind_host": self.interception.bind_host,
                "advertise_host": self.interception.advertise_host,
                "port": self.interception.port,
                "observe_requests": self.interception.observe_requests,
            },
            "tools": None if self.tools is None else self.tools.to_data(),
        }

    @classmethod
    def from_document(cls, document: Mapping[str, object]) -> InjectionSpec:
        """Read a v1 document; any other version, or none, is refused."""
        data = _object(document, "injection spec document")
        if "schema_version" not in data:
            raise InjectionSpecDocumentError(
                "injection spec document: schema_version is required"
            )
        version = data["schema_version"]
        if type(version) is not str:
            raise InjectionSpecDocumentError(
                f"injection spec document: schema_version must be a string, not {type(version).__name__}"
            )
        if version != DOCUMENT_V1:
            raise UnsupportedDocumentVersion(version)
        return _read_v1(cls, data)


# --- reading ---


def _read_v1(cls: type[InjectionSpec], data: Mapping[str, Any]) -> InjectionSpec:
    _closed(data, _MEMBERS, "injection spec document")
    _verifiers(data.get("verifiers"))
    turns = _turns(data.get("turns"))
    # What a member's reader raises while the document is read carries the document's
    # prefix, as the runtime's does: the runtime's reader catches its own refusals.
    try:
        runtime = _runtime(data.get("runtime"))
        plan = _plan(data.get("plan"))
        security = _data_object(data.get("security"), "security")
        utility = _data_object(data.get("utility"), "utility")
        environment = _data_object(data.get("environment"), "environment")
        suite_name = _string(data.get("suite_name"), "suite_name")
        user_task_id = _string(data.get("user_task_id"), "user_task_id")
        injection_task_id = _string(data.get("injection_task_id"), "injection_task_id")
        interception = _interception(data.get("interception"))
        tools = _tools(data.get("tools"))
        runtime = _runtime_data(runtime)
        security = _checked("security", security, runtime)
        utility = _checked("utility", utility, runtime)
        environment = None if environment is None else _freeze(environment, 0)
    except _CheckGap as exc:
        raise InjectionSpecDocumentError(
            f"injection spec document: {exc}", gap=exc.gap
        ) from None
    except (TypeError, ValueError) as exc:
        raise InjectionSpecDocumentError(f"injection spec document: {exc}") from None
    spec = cls(
        runtime=runtime,
        turns=turns,
        plan=plan,
        security=security,
        utility=utility,
        environment=environment,
        suite_name=suite_name,
        user_task_id=user_task_id,
        injection_task_id=injection_task_id,
        interception=interception,
        tools=tools,
    )
    _environment_type(spec, _string(data.get("environment_type"), "environment_type"))
    if runtime["type"] == "openshell":
        raise UnsupportedRuntime(
            "injection spec document: the openshell agent_runtime is not run here; "
            "only the unmanaged agent_runtime is"
        )
    return spec


# --- validating one member ---


def _object(value: object, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InjectionSpecDocumentError(
            f"{where} must be a JSON object, not {type(value).__name__}"
        )
    return value


def _closed(data: Mapping[str, object], allowed: frozenset[str], where: str) -> None:
    if unknown := [key for key in data if key not in allowed]:
        raise InjectionSpecDocumentError(
            f"{where}: unknown key {unknown[0]!r}; the document is closed"
        )


def _string(value: object, where: str) -> str | None:
    if value is None or type(value) is str:
        return value
    raise InjectionSpecDocumentError(
        f"{where} must be a string or null, not {type(value).__name__}"
    )


def _turns(value: object) -> tuple[str, ...] | None:
    if value is None:
        return None
    if type(value) is not list:
        raise InjectionSpecDocumentError(
            f"turns must be an array of strings or null, not {type(value).__name__}"
        )
    if not value:
        raise InjectionSpecDocumentError(
            "turns must hold at least one message, or be null when the host drives them"
        )
    for index, message in enumerate(value):
        if type(message) is not str:
            raise InjectionSpecDocumentError(
                f"turns[{index}] must be a string, not {type(message).__name__}"
            )
    return tuple(value)


def _plan(value: object) -> Any:
    codec = injection_codec()
    if value is None:
        return codec.InjectionPlan()
    try:
        return codec.InjectionPlan.from_data(_object(value, "plan"))
    except ValueError as exc:
        raise InjectionSpecDocumentError(f"plan: {exc}") from None


def _runtime(value: object) -> Mapping[str, Any]:
    if value is None:
        return {"type": "unmanaged"}
    data = _object(value, "runtime")
    if data.get("type") not in _RUNTIME_TYPES:
        raise InjectionSpecDocumentError(
            f"runtime.type: {data.get('type')!r} is not one of the runtime definitions this document defines "
            f"({', '.join(repr(name) for name in _RUNTIME_TYPES)})"
        )
    return data


def _data_object(value: object, where: str) -> Mapping[str, Any] | None:
    return None if value is None else _object(value, where)


def _verifiers(value: object) -> None:
    """Refuse a document that names a verifier: a registry is trusted code, so a document names none."""
    if value is None:
        return
    if type(value) is not list or value:
        raise InjectionSpecDocumentError(
            "verifiers: a document names no verifier; MiDojo reads only [] because a verifier registry is trusted "
            "code, not data"
        )


def _interception(value: object) -> InterceptionSettings:
    if value is None:
        return InterceptionSettings()
    data = _object(value, "interception")
    _closed(data, _INTERCEPTION_FIELDS, "interception")
    try:
        return InterceptionSettings(
            mode=data.get("mode", "host"),
            upstream=_string(data.get("upstream"), "interception.upstream"),
            bind_host=_string(data.get("bind_host"), "interception.bind_host"),
            advertise_host=_string(
                data.get("advertise_host"), "interception.advertise_host"
            ),
            port=data.get("port"),
            observe_requests=data.get("observe_requests", False),
        )
    except (TypeError, ValueError) as exc:
        raise InjectionSpecDocumentError(f"interception: {exc}") from None


def _tools(value: object) -> Any:
    if value is None:
        return None
    data = _object(value, "tools")
    _closed(data, _TOOLS_FIELDS, "tools")
    if "allowed" not in data:
        raise InjectionSpecDocumentError("tools.allowed is required")
    allowed = data["allowed"]
    if type(allowed) is not list or any(type(name) is not str for name in allowed):
        raise InjectionSpecDocumentError("tools.allowed must be an array of strings")
    settings = _object(data.get("write_disposition", {}), "tools.write_disposition")
    _closed(settings, _WRITE_DISPOSITION_FIELDS, "tools.write_disposition")
    overrides = _object(settings.get("tools", {}), "tools.write_disposition.tools")
    try:
        return policy_codec().ToolPolicy(
            tuple(allowed),
            write_disposition_default=settings.get("default", "forward"),
            write_disposition=dict(overrides),
        )
    except (TypeError, ValueError) as exc:
        raise InjectionSpecDocumentError(f"tools: {exc}") from None


# --- what a built document is checked for ---


def _freeze(value: Any, depth: int) -> Any:
    """A deep, read-only copy of JSON data: mappings become views, lists and tuples tuples."""
    if isinstance(value, (Mapping, list, tuple)) and depth >= MAX_NESTING_DEPTH:
        raise ValueError(
            f"episode data must be nested at most {MAX_NESTING_DEPTH} deep"
        )
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            frozen[_exact_str(key, "a key in episode data")] = _freeze(item, depth + 1)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item, depth + 1) for item in value)
    if type(value) not in _SCALARS:
        raise TypeError(f"episode data must be JSON-like, not {type(value).__name__}")
    return value


def _runtime_data(runtime: Mapping[str, Any]) -> Mapping[str, Any]:
    """The runtime definition, frozen. An ``unmanaged`` one holds its ``type`` alone."""
    frozen = _freeze(runtime, 0)
    if frozen["type"] == "unmanaged" and len(frozen) != 1:
        raise ValueError(
            "runtime: the unmanaged agent_runtime has no member but its type"
        )
    return frozen


def _checked(
    name: str, authored: Mapping[str, Any] | None, runtime: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    """``authored`` frozen, once it reads as a check the runtime can collect the evidence for."""
    if authored is None:
        return None
    frozen = _freeze(authored, 0)
    check = _checks.parse(thaw(frozen))
    if runtime["type"] != "openshell" and _checks.reads_openshell(check):
        raise _CheckGap(
            f"The {name} check uses OpenShell predicates, which require the openshell agent_runtime "
            f"(the episode's agent_runtime is {runtime['type']!r})",
            OPENSHELL_PREDICATE_GAP,
        )
    return frozen


def _environment_type(spec: InjectionSpec, named: str | None) -> None:
    """Refuse a document that names an environment model MiDojo would not infer for it."""
    if named is not None and named != spec.environment_type:
        raise InjectionSpecDocumentError(
            f"environment_type: the injection spec's suite_name and environment imply {spec.environment_type!r}, "
            f"not {named!r}"
        )
