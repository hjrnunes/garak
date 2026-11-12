"""The collector's conversation and world: turns, observations, environment.

Turns and replies are admitted only while the collector is open. The
environment is admitted while it is open or closing. Observations are admitted
until the seal, so the collecting stage takes the final ones. A write its
stage refuses is kept as a late fact keyed by the evidence it carries. Each
write replaces what an earlier write of the same kind left: a turn starts only
once the previous one has its reply, an observation source replaces whole, and
the environment is replaced wholesale.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from garak.evidence.facts import (
    ENVIRONMENT,
    EVIDENCE_STATUSES,
    TURNS,
    Observation,
    Turn,
    json_payload,
    require_text,
)
from garak.evidence.intake import Stage
from garak.evidence.records import (
    InvalidRequestError,
    RejectedError,
    freeze,
)


class WorldMixin:
    """The turns, observations and environment one collector records."""

    def start_turn(self, message: str) -> Turn:
        """Start one turn of the conversation, refused while the previous has no reply."""
        require_text(message, "turn's message")
        arrival = self._monotonic()
        at = self._clock()
        with self._condition:
            self._on_time(arrival)
            stage = self._stage
            if stage is not Stage.OPEN:
                self._record_late(
                    "start_turn",
                    "host",
                    at,
                    evidence=(TURNS,),
                    payload={"message": message},
                )
                raise self._late_error(stage)
            with self._turn_lock:
                if self._turns and self._turns[-1].reply is None:
                    raise RejectedError(
                        "turn_pending", f"turn {self._turns[-1].index} has no reply yet"
                    )
                turn = Turn(
                    index=len(self._turns),
                    message=message,
                    reply=None,
                    started_at=self._clock(),
                    ended_at=None,
                )
                self._turns.append(turn)
                return turn

    def record_reply(self, index: int, reply: str) -> Turn:
        """Record the agent's reply to turn ``index``.

        An empty reply is a reply: an agent that answers a turn with tool calls
        and no text has answered it, and the runtime records ``""`` for it.
        """
        if type(reply) is not str:
            raise InvalidRequestError("turn's reply must be text")
        arrival = self._monotonic()
        at = self._clock()
        with self._condition:
            self._on_time(arrival)
            stage = self._stage
            if stage is not Stage.OPEN:
                self._record_late(
                    "reply_turn",
                    "host",
                    at,
                    evidence=(TURNS,),
                    payload={"index": index, "reply": reply},
                )
                raise self._late_error(stage)
            with self._turn_lock:
                if type(index) is not int or not 0 <= index < len(self._turns):
                    raise RejectedError(
                        "turn_not_started", f"turn not started: {index}"
                    )
                turn = self._turns[index]
                if turn.reply is not None:
                    raise RejectedError(
                        "turn_answered", f"turn {index} already has a reply"
                    )
                answered = Turn(
                    index=turn.index,
                    message=turn.message,
                    reply=reply,
                    started_at=turn.started_at,
                    ended_at=self._clock(),
                )
                self._turns[index] = answered
                return answered

    def _current_turn(self) -> int | None:
        """The latest started turn's index, read under the turn lock."""
        with self._turn_lock:
            return self._turns[-1].index if self._turns else None

    def record_observations(
        self,
        source: str,
        observed: Mapping[str, Any],
        evidence: Mapping[str, str],
    ) -> Observation:
        """Replace one source's observations, with its evidence parts' statuses."""
        require_text(source, "source")
        if type(observed) is not dict:
            raise InvalidRequestError("observed must be an object")
        if type(evidence) is not dict or not evidence:
            raise InvalidRequestError("evidence must be a non-empty object")
        for part, status in evidence.items():
            require_text(part, "evidence part")
            if status not in EVIDENCE_STATUSES:
                raise RejectedError(
                    "unsupported_value", f"evidence status '{status}' is unknown"
                )
        arrival = self._monotonic()
        at = self._clock()
        frozen = json_payload(observed)
        with self._condition:
            self._on_time(arrival)
            stage = self._stage
            if stage is Stage.SEALED:
                keys = tuple(sorted(f"{source}.{part}" for part in evidence))
                self._record_late(
                    "record_observations",
                    "host",
                    at,
                    evidence=keys,
                    payload={
                        "source": source,
                        "data": observed,
                        "evidence": dict(evidence),
                    },
                )
                raise self._late_error(stage)
            observation = Observation(
                source=source, observed=frozen, evidence=dict(evidence)
            )
            self._observations[source] = observation
            return observation

    def replace_environment(self, environment: Mapping[str, Any]) -> Mapping[str, Any]:
        """Replace the environment wholesale, only while open or closing."""
        if type(environment) is not dict:
            raise InvalidRequestError("environment must be an object")
        arrival = self._monotonic()
        at = self._clock()
        frozen = freeze(dict(environment))
        with self._condition:
            self._on_time(arrival)
            stage = self._stage
            if stage not in (Stage.OPEN, Stage.CLOSING):
                self._record_late(
                    "replace_environment",
                    "host",
                    at,
                    evidence=(ENVIRONMENT,),
                    payload={"environment": environment},
                )
                raise self._late_error(stage)
            self._environment = frozen
            return frozen
