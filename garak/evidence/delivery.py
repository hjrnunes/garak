"""Delivery facts, and the exposure derived from them.

A *delivery fact* is what the component that placed an injection did at its
own boundary, recorded as it happened: for a tool-output instruction, what the
interceptor selected, transformed and emitted in one call's result (each sealed
call's ``injection_outcomes``). A fact references its instruction by index and
slot key; the plan holds neither. The interceptor numbers instructions within
the plan's tool-output part, so a fact names both indexes apart:
``interceptor_instruction_index`` as the call recorded it and
``plan_instruction_index``, the instruction's place in the whole plan.

*Exposure* is derived from the facts and names its boundary:

* ``emitted``: the placing component sent the payload out at its own boundary
  (the interceptor in a tool result it returned; the host in a turn message it
  started);
* ``agent_tool_result``: the payload reached the agent's tool result;
* ``model_input``: the payload reached the input of the agent's model.

Each is ``exposed``, ``not_exposed`` or ``unknown``. Nothing past the
interceptor is observed, so an emitted payload's ``agent_tool_result`` is
unknown and every payload's ``model_input`` is. A payload the interceptor did
not emit never reached the agent through that instruction, so its
``agent_tool_result`` is ``not_exposed``.

Delivery and exposure are report only: they never change a grade's security
or utility. Both read a plan decoded by ``midojo_spec.injection.InjectionPlan``
and the call records of a sealed snapshot, and ``to_json()`` writes the
documents the MiDojo runtime writes for the same evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping, NamedTuple, Sequence

from garak.evidence.codec import injection_codec
from garak.evidence.records import CallRecord, freeze, thaw

FUNCTION_CALLS = "function_calls"
EXPOSURE_BOUNDARIES = ("emitted", "agent_tool_result", "model_input")

_LEGACY_SLOT_KEY = "probe_key"
_TOOL_OUTPUT = "tool_output"
_USER_PROMPT = "user_prompt"
_BOUNDARIES = {
    "tool_output": EXPOSURE_BOUNDARIES,
    "tool_description": EXPOSURE_BOUNDARIES,
    "user_prompt": ("emitted", "model_input"),
    "environment_state": EXPOSURE_BOUNDARIES,
    "runtime_file": EXPOSURE_BOUNDARIES,
}


class _Outcome(NamedTuple):
    """One injection outcome of a recorded call, read once."""

    instruction_index: int
    slot_key: str | None
    occurrence: int
    status: str
    selected: bool | None
    transformation: str | None
    emitted: tuple[str, ...] | None
    payload_digest: str | None
    placed_digest: str | None
    slot: Mapping[str, Any] | None


def _malformed(call: CallRecord) -> ValueError:
    return ValueError(f"call {call.call_id} holds a malformed injection outcome")


def _outcomes(call: CallRecord) -> tuple[_Outcome, ...] | None:
    """The call's outcomes as its interceptor reported them, or None when it reported none.

    A record made before the rename names the slot key ``probe_key``; it reads
    as ``slot_key``. A record that names both is malformed, as it is to the
    runtime.
    """
    if call.injection_outcomes is None:
        return None
    read = []
    for item in call.injection_outcomes:
        try:
            legacy = _LEGACY_SLOT_KEY in item
            if legacy and "slot_key" in item:
                raise _malformed(call)
            emitted = item.get("emitted")
            read.append(
                _Outcome(
                    item["instruction_index"],
                    item[_LEGACY_SLOT_KEY] if legacy else item.get("slot_key"),
                    item["occurrence"],
                    item["status"],
                    item.get("selected"),
                    item.get("transformation"),
                    None if emitted is None else tuple(emitted),
                    item.get("payload_digest"),
                    item.get("placed_digest"),
                    item.get("slot"),
                )
            )
        except (KeyError, TypeError, AttributeError) as exc:
            raise _malformed(call) from exc
    return tuple(read)


@dataclass(frozen=True)
class DeliveryFact:
    """What the interceptor did with one instruction on one recorded call.

    ``plan_instruction_index`` is None when no tool-output instruction of the
    plan has that place and slot key and targets the call's tool (or any tool).
    ``selected``, ``transformation`` and ``emitted`` are None when the
    interceptor did not report them; the placement fields say what a placement
    wrote and where, and are None for any other outcome.
    """

    call_id: str
    function: str
    disposition: str | None
    turn: int | None
    interceptor_instruction_index: int
    plan_instruction_index: int | None
    slot_key: str | None
    occurrence: int
    status: str
    selected: bool | None
    transformation: str | None
    emitted: tuple[str, ...] | None
    payload_digest: str | None
    placed_digest: str | None
    slot: Mapping[str, Any] | None

    def to_json(self) -> dict[str, Any]:
        """Return a fresh JSON object in the runtime's field order."""
        return {
            "call_id": self.call_id,
            "function": self.function,
            "disposition": self.disposition,
            "turn": self.turn,
            "interceptor_instruction_index": self.interceptor_instruction_index,
            "plan_instruction_index": self.plan_instruction_index,
            "slot_key": self.slot_key,
            "occurrence": self.occurrence,
            "status": self.status,
            "selected": self.selected,
            "transformation": self.transformation,
            "emitted": None if self.emitted is None else list(self.emitted),
            "payload_digest": self.payload_digest,
            "placed_digest": self.placed_digest,
            "slot": thaw(self.slot),
        }


def _tool_output_indexes(plan: Any) -> list[int]:
    """The plan index of each instruction in the plan's tool-output part, in that part's order."""
    return [
        index
        for index, instruction in enumerate(plan)
        if instruction.channel.value == _TOOL_OUTPUT
    ]


def _recorded_slot(slot: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """A recorded slot as the plan codec writes it, so equal places read alike."""
    if slot is None:
        return None
    try:
        decoded = injection_codec().InjectionSlot.from_data(thaw(slot))
    except ValueError as exc:
        raise ValueError("a call holds a malformed injection slot") from exc
    return freeze(decoded.to_data())


def delivery_facts(
    plan: Any, function_calls: Iterable[CallRecord]
) -> list[DeliveryFact]:
    """Every delivery fact the recorded calls carry, in call then instruction order.

    ``plan`` is a ``midojo_spec.injection.InjectionPlan``; each fact carries
    the index of its instruction in that plan.
    """
    tool_output = _tool_output_indexes(plan)

    def plan_index(
        interceptor_index: int, slot_key: str | None, function: str
    ) -> int | None:
        if not 0 <= interceptor_index < len(tool_output):
            return None
        index = tool_output[interceptor_index]
        instruction = plan.instructions[index]
        if instruction.probe_key != slot_key or instruction.target.tool not in (
            None,
            function,
        ):
            return None
        return index

    facts = []
    for call in function_calls:
        for outcome in _outcomes(call) or ():
            facts.append(
                DeliveryFact(
                    call_id=call.call_id,
                    function=call.function,
                    disposition=call.disposition,
                    turn=call.turn,
                    interceptor_instruction_index=outcome.instruction_index,
                    plan_instruction_index=plan_index(
                        outcome.instruction_index, outcome.slot_key, call.function
                    ),
                    slot_key=outcome.slot_key,
                    occurrence=outcome.occurrence,
                    status=outcome.status,
                    selected=outcome.selected,
                    transformation=outcome.transformation,
                    emitted=outcome.emitted,
                    payload_digest=outcome.payload_digest,
                    placed_digest=outcome.placed_digest,
                    slot=_recorded_slot(outcome.slot),
                )
            )
    return facts


@dataclass(frozen=True)
class InstructionExposure:
    """One instruction of a plan: who placed it, and its exposure at each boundary its channel has.

    ``reason`` says why the first boundary that is not ``exposed`` is what it is.
    """

    plan_instruction_index: int
    channel: str
    slot_key: str | None
    placed_by: str
    exposure: Mapping[str, str]
    reason: str

    def to_json(self) -> dict[str, Any]:
        """Return a fresh JSON object in the runtime's field order."""
        return {
            "plan_instruction_index": self.plan_instruction_index,
            "channel": self.channel,
            "slot_key": self.slot_key,
            "placed_by": self.placed_by,
            "exposure": dict(self.exposure),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class Exposure:
    """An evaluation's exposure at each boundary, over every instruction of its plan.

    A boundary is ``exposed`` when any instruction's payload is exposed there,
    ``not_exposed`` when every instruction with that boundary is ``not_exposed``
    (or none has it: an empty plan places nothing, and a prompt payload is in
    no tool result), and ``unknown`` otherwise.
    """

    emitted: str
    agent_tool_result: str
    model_input: str
    instructions: tuple[InstructionExposure, ...]

    def at(self, boundary: str) -> str:
        """The exposure at ``boundary``, one of ``EXPOSURE_BOUNDARIES``."""
        return {
            "emitted": self.emitted,
            "agent_tool_result": self.agent_tool_result,
            "model_input": self.model_input,
        }[boundary]

    def to_json(self) -> dict[str, Any]:
        """Return a fresh JSON object in the runtime's field order."""
        return {
            "emitted": self.emitted,
            "agent_tool_result": self.agent_tool_result,
            "model_input": self.model_input,
            "instructions": [item.to_json() for item in self.instructions],
        }


def exposure(
    plan: Any,
    function_calls: Sequence[CallRecord],
    *,
    turns: int,
    evidence_status: Mapping[str, str],
) -> Exposure:
    """The exposure of ``plan``'s payloads in an evaluation, derived from its sealed records.

    ``function_calls`` are the evaluation's call records, ``turns`` how many
    turns it started, and ``evidence_status`` how completely they were
    collected (a key that is absent has no known gap).

    The interceptor places a tool-output instruction. Its facts are those whose
    index is the instruction's place in the plan's tool-output part and whose
    slot key is the instruction's. It is ``emitted`` when a fact names a part
    of the result that carries it. It is ``not_exposed`` only when the record
    says it cannot have been: ``function_calls`` is complete, and every call to
    the instruction's tool that ran finished with a fact for it, none naming it
    as emitted (a blocked call never ran). A call with no fact for it is
    unknown, even when the call reported others. Anything else is unknown.

    The host places a user-prompt instruction in a turn's message: it is
    ``emitted`` once that turn started, ``not_exposed`` when it did not and the
    turns are complete. The host places an environment-state or runtime-file
    payload before the agent runs, and nothing records whether a tool or a file
    read carried it on, so every boundary is unknown.
    """
    tool_output = _tool_output_indexes(plan)
    instructions = []
    for index, instruction in enumerate(plan):
        channel = instruction.channel.value
        if channel == _TOOL_OUTPUT:
            emitted, reason = _tool_output_emitted(
                tool_output.index(index), instruction, function_calls, evidence_status
            )
        elif channel == _USER_PROMPT:
            emitted, reason = _prompt_emitted(instruction, turns, evidence_status)
        else:
            emitted = "unknown"
            reason = f"placed by the host in the {channel}: no delivery fact"
        instructions.append(
            InstructionExposure(
                plan_instruction_index=index,
                channel=channel,
                slot_key=instruction.probe_key,
                placed_by="interceptor" if channel == _TOOL_OUTPUT else "host",
                exposure=MappingProxyType(
                    {
                        boundary: _past(boundary, emitted)
                        for boundary in _BOUNDARIES[channel]
                    }
                ),
                reason=reason,
            )
        )

    def combined(boundary: str) -> str:
        return _combine(
            [
                item.exposure[boundary]
                for item in instructions
                if boundary in item.exposure
            ]
        )

    return Exposure(
        emitted=combined("emitted"),
        agent_tool_result=combined("agent_tool_result"),
        model_input=combined("model_input"),
        instructions=tuple(instructions),
    )


def _past(boundary: str, emitted: str) -> str:
    """The exposure at ``boundary`` of a payload whose emission is ``emitted``.

    Only emission is recorded. A payload that was not emitted reached no tool
    result through its instruction; past the interceptor nothing is observed,
    and no model input is ever observed.
    """
    if boundary == "emitted":
        return emitted
    if boundary == "agent_tool_result" and emitted == "not_exposed":
        return "not_exposed"
    return "unknown"


def _combine(statuses: list[str]) -> str:
    if "exposed" in statuses:
        return "exposed"
    if all(status == "not_exposed" for status in statuses):
        return "not_exposed"
    return "unknown"


def _tool_output_emitted(
    interceptor_index: int,
    instruction: Any,
    function_calls: Sequence[CallRecord],
    evidence_status: Mapping[str, str],
) -> tuple[str, str]:
    gaps: list[str] = []
    if evidence_status.get(FUNCTION_CALLS, "complete") != "complete":
        gaps.append("function call capture is incomplete")
    tool = instruction.target.tool
    for call in function_calls:
        if tool is not None and call.function != tool:
            continue
        if call.blocked:
            continue  # the tool never ran: the layer returned its refusal
        if not call.complete:
            gaps.append(f"call {call.call_id} is unfinished")
            continue
        outcomes = _outcomes(call)
        if outcomes is None:
            gaps.append(f"call {call.call_id} reported no delivery facts")
            continue
        facts = [
            item for item in outcomes if item.instruction_index == interceptor_index
        ]
        if not facts:
            # An interceptor reports an outcome for every instruction of its plan whose tool matches the call,
            # but nothing here shows that plan was this one: a missing fact is a gap, not an absence.
            gaps.append(
                f"call {call.call_id} reported no delivery fact for instruction {interceptor_index}"
            )
            continue
        for fact in facts:
            if fact.slot_key != instruction.probe_key:
                gaps.append(
                    f"call {call.call_id} names instruction {interceptor_index} with another slot key"
                )
            elif fact.emitted:
                return (
                    "exposed",
                    f"emitted in {', '.join(fact.emitted)} of call {call.call_id}",
                )
            elif fact.emitted is None:
                gaps.append(f"call {call.call_id} didn't report what it emitted")
    if gaps:
        return "unknown", "; ".join(gaps)
    return "not_exposed", "no call's result carried it"


def _prompt_emitted(
    instruction: Any, turns: int, evidence_status: Mapping[str, str]
) -> tuple[str, str]:
    path = instruction.target.template_path
    turn = path[0] if path and type(path[0]) is int else None
    if turn is None:
        return "unknown", "the instruction names no turn"
    if turn < turns:
        return "exposed", f"sent in turn {turn}"
    if evidence_status.get("turns", "complete") != "complete":
        return "unknown", f"turn {turn} wasn't recorded, and turns are incomplete"
    return "not_exposed", f"turn {turn} never started"
