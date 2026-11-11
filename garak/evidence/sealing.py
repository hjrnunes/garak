"""The seal: fixing a collector's snapshot and the status it seals.

The seal is allowed only from ``collecting``; sealing again with the same
output returns the same snapshot, and another output is kept as a late fact.
The status it fixes folds together the observations' own statuses, the call
capture's, the capture failures, the links read from their bindings, the late
facts kept before the seal and the keys the caller named.
"""

from __future__ import annotations

from garak.evidence.facts import (
    AGENT_OUTPUT,
    FUNCTION_CALLS,
    SESSION_PROVENANCE,
    require_text,
)
from garak.evidence.intake import Stage
from garak.evidence.links import LOST, STALLED
from garak.evidence.records import RejectedError
from garak.evidence.seal import Snapshot


class SealingMixin:
    """The seal and the status it fixes, on a collector that has collected."""

    def seal(self, agent_output: str, *, incomplete: tuple[str, ...] = ()) -> Snapshot:
        """Fix the immutable snapshot; allowed only from ``collecting``.

        Sealing again with the same output returns the same snapshot; with
        another output the write is kept as a late fact.
        """
        if type(agent_output) is not str:
            raise RejectedError("unsupported_value", "agent's output must be a str")
        for key in incomplete:
            require_text(key, "incomplete key")
        arrival = self._monotonic()
        at = self._clock()
        with self._condition:
            self._on_time(arrival)
            if self._stage is Stage.SEALED:
                assert self._snapshot is not None
                if self._snapshot.agent_output == agent_output:
                    return self._snapshot
                self._record_late(
                    "seal",
                    "host",
                    at,
                    evidence=(AGENT_OUTPUT,),
                    payload={"agent_output": agent_output},
                )
                raise self._late_error(self._stage)
            if self._stage is not Stage.COLLECTING:
                raise RejectedError(
                    "closure_pending",
                    "closure has not finished waiting for admitted calls",
                )
            snapshot = Snapshot(
                evaluation_id=self.evaluation_id,
                agent_output=agent_output,
                pre_environment=self._pre_environment,
                environment=self._environment,
                function_calls=self._ledger.calls(),
                turns=tuple(self._turns),
                observations={
                    source: observation.observed
                    for source, observation in self._observations.items()
                },
                evidence_status=self._sealed_status(named=incomplete),
                sealed_at=self._clock(),
                capture_failures=tuple(self._capture_failures.values()),
                run_id=self.run_id,
                user_task_id=self.user_task_id,
                injection_task_id=self.injection_task_id,
            )
            self._snapshot = snapshot
            self._stage = Stage.SEALED
            self._condition.notify_all()
            return snapshot

    def _sealed_status(self, named: tuple[str, ...] = ()) -> dict[str, str]:
        """The status the seal fixes; before the seal, the one it would fix now.

        The caller holds the state lock. ``named`` adds the caller's own
        incomplete keys, as the seal's ``incomplete`` argument gives them.
        """
        if self._snapshot is not None:
            return dict(self._snapshot.evidence_status)
        statuses: dict[str, str] = {}
        for observation in self._observations.values():
            statuses.update(observation.flattened())
        statuses.update(self._capture_status())
        if self._capture_failures:
            statuses[FUNCTION_CALLS] = "incomplete"
        now = self._monotonic()
        for link in self._links.values():
            if link.state(now) in (LOST, STALLED):
                statuses[FUNCTION_CALLS] = "incomplete"
        for fact in self._late_facts:
            for key in fact.evidence_keys():
                statuses[key] = "incomplete"
        for key in named:
            statuses[key] = "incomplete"
        return statuses

    def _capture_status(self) -> dict[str, str]:
        """Why the call capture alone cannot be claimed complete.

        An interception layer in the agent may fail open, so no record at all
        is an incomplete capture, never a clean one. A call that layer started
        whose finish never arrived is the same loss: only its result is
        missing. A host-started call does not mark the capture: the host sends
        the seal and the finish alike.
        """
        calls = self._ledger.calls()
        if self.interception == "agent" and not calls:
            return {FUNCTION_CALLS: "incomplete"}
        if any(
            not call.complete and call.provenance in SESSION_PROVENANCE
            for call in calls
        ):
            return {FUNCTION_CALLS: "incomplete"}
        return {}
