"""Parent-process ownership of injector runs, evaluation chains and turns.

The coordinator is a library object. A harness owns it for the duration of a
run; no probe, worker, or process receives a live coordinator. Deadlines apply
to each turn separately. A caller that needs a total chain bound cancels the
chain from its own timer.
"""

from __future__ import annotations

import logging
import math
import threading
import weakref
from dataclasses import dataclass, field
from typing import Callable

from garak import _config, _plugins
from garak.evidence.collector import Collector, Stage
from garak.evidence.facts import ENVIRONMENT, FUNCTION_CALLS, TURNS
from garak.exception import CapabilityError, GarakException

_EXPECTED_CLEANUP = (RuntimeError, OSError, ValueError, CapabilityError, ExceptionGroup)


class TurnTimeout(GarakException):
    """The per-turn deadline expired and the chain was closed."""


class _Step:
    """Capture a failure of any type without a broad exception handler.

    Context-manager exit receives the active exception, including an unknown
    subclass, so every later teardown step can run. Re-raise the original
    object after the last release rather than converting its type or text.
    """

    def __init__(self, errors: list[BaseException]):
        self.errors = errors

    def __enter__(self) -> _Step:
        return self

    def __exit__(self, kind, error, traceback) -> bool:
        if error is not None:
            self.errors.append(error.with_traceback(traceback))
            return True
        return False


def _failure_object(errors: list[BaseException]) -> BaseException | None:
    """Select the original unexpected error, or one aggregated cleanup error."""
    if not errors:
        return None
    unexpected = next(
        (error for error in errors if not isinstance(error, _EXPECTED_CLEANUP)), None
    )
    if unexpected is not None:
        if len(errors) > 1:
            try:
                unexpected.add_note(
                    f"{len(errors) - 1} additional coordinator cleanup failures"
                )
            except (AttributeError, TypeError):
                pass
        return unexpected
    if len(errors) == 1:
        return errors[0]
    return ExceptionGroup("coordinator cleanup failed", errors)


def _raise_failures(errors: list[BaseException], original=None) -> None:
    """Raise a cleanup failure, preserving any original exception as its cause."""
    failure = _failure_object(errors)
    if failure is not None:
        raise failure from original


def _finish_or_note(errors, original=None, result=None, details=None) -> None:
    """Keep an active exception; expose cleanup types without foreign text."""
    if not errors:
        return
    failures = details or errors
    names = [type(failure).__name__ for failure in failures]
    if result is not None:
        result["cleanup_failures"] = names
    if original is not None:
        suffix = "failure" if len(names) == 1 else "failures"
        note = f"{len(names)} coordinator cleanup {suffix}: {', '.join(names)}"
        try:
            original.add_note(note)
        except (AttributeError, TypeError):
            pass
        return
    _raise_failures(errors)


def _positive(value: float, name: str) -> float:
    if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


@dataclass(frozen=True)
class ContextRequest:
    """Selected component names and declared capability requirements."""

    injector: str | None = None
    environment: str | None = None
    required_generator: frozenset[str] = field(default_factory=frozenset)
    required_injector: frozenset[str] = field(default_factory=frozenset)
    required_environment: frozenset[str] = field(default_factory=frozenset)


def _plugins_config(config_root):
    root = _config if config_root is None else config_root
    if isinstance(root, dict):
        return root.get("plugins", root)
    return getattr(root, "plugins", root)


def _selection(config, key, fallback):
    value = config.get(key) if isinstance(config, dict) else getattr(config, key, None)
    return value or fallback


def select_components(
    request: ContextRequest, *, config_root=None
) -> tuple[object | None, object]:
    """Load configured components, then request defaults, then safe fallbacks."""
    config = _plugins_config(config_root)
    injector_name = _selection(config, "injector_type", request.injector or "none")
    environment_name = _selection(
        config, "environment_type", request.environment or "unmanaged"
    )
    injector = (
        None
        if injector_name == "none"
        else _plugins.load_plugin(
            f"injectors.{injector_name}",
            config_root=_config if config_root is None else config_root,
        )
    )
    environment = _plugins.load_plugin(
        f"environments.{environment_name}",
        config_root=_config if config_root is None else config_root,
    )
    return injector, environment


def check_capabilities(request, generator, injector, environment) -> None:
    """Refuse every missing declared capability before opening any resource."""
    for label, component, required in (
        ("generator", generator, request.required_generator),
        ("injector", injector, request.required_injector),
        ("environment", environment, request.required_environment),
    ):
        missing = required - getattr(component, "capabilities", frozenset())
        if missing:
            raise CapabilityError(
                f"{label} lacks capabilities: {', '.join(sorted(missing))}"
            )


class _Bounded:
    """Run a blocking foreign operation on a daemon and wait at most timeout."""

    def __init__(
        self, function: Callable, timeout: float, late: Callable | None = None
    ):
        self.value = None
        self.errors: list[BaseException] = []
        self.done = threading.Event()
        self.function = function
        self.timeout = timeout
        self.late = late
        self._lock = threading.Lock()
        self._abandoned = False
        self._late_taken = False

    def _run(self):
        try:
            with _Step(self.errors):
                self.value = self.function()
        finally:
            with self._lock:
                self.done.set()
                late = self._take_late()
            if late is not None:
                self._release_late(late)

    @staticmethod
    def _release_late(late):
        failures = []
        with _Step(failures):
            late()
        if failures:
            logging.debug("late release failed: %s", type(failures[0]).__name__)

    def _take_late(self):
        ready = self._abandoned and self.done.is_set() and self.value is not None
        if ready and self.late is not None and not self._late_taken:
            self._late_taken = True
            return lambda: self.late(self.value)
        return None

    def get(self):
        """Return the result, propagate its error, or report a bounded loss."""
        worker = threading.Thread(target=self._run, daemon=True)
        worker.start()
        if not self.done.wait(self.timeout):
            with self._lock:
                self._abandoned = True
                late = self._take_late()
            if late is not None:
                threading.Thread(
                    target=self._release_late, args=(late,), daemon=True
                ).start()
            raise TimeoutError("coordinator resource operation exceeded its bound")
        _raise_failures(self.errors)
        return self.value


class RunCoordinator:
    """Own one injector run and all chain scopes, across re-entrant harness calls."""

    def __init__(
        self,
        injector=None,
        environment=None,
        *,
        collector_factory: Callable = Collector,
        max_live: int = 1,
        turn_timeout: float = 60,
        start_timeout: float = 15,
        bind_timeout: float = 15,
        setup_timeout: float = 60,
        close_timeout: float = 10,
        drain_timeout: float = 2,
        request: ContextRequest | None = None,
        generator=None,
    ):
        if type(max_live) is not int or max_live < 1:
            raise ValueError("max_live must be positive")
        self.turn_timeout = _positive(turn_timeout, "turn_timeout")
        self.start_timeout = _positive(start_timeout, "start_timeout")
        self.bind_timeout = _positive(bind_timeout, "bind_timeout")
        self.setup_timeout = _positive(setup_timeout, "setup_timeout")
        self.close_timeout = _positive(close_timeout, "close_timeout")
        self.drain_timeout = _positive(drain_timeout, "drain_timeout")
        self.injector = injector
        self.environment = environment
        self.collector_factory = collector_factory
        self.request = request or ContextRequest()
        self.generator = generator
        self.max_live = max_live
        self._lock = threading.RLock()
        self._depth = 0
        self._run = None
        self._chains: set[ChainScope] = set()
        self._chains_seen: weakref.WeakSet[ChainScope] = weakref.WeakSet()
        self._cleanup_errors: tuple[BaseException, ...] = ()
        self._open_count = 0
        self._closed = False

    def __enter__(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("coordinator is closed")
            self._depth += 1
        return self

    def __exit__(self, kind, error, traceback):
        with self._lock:
            self._depth -= 1
            outermost = self._depth == 0
        if outermost:
            failures: list[BaseException] = []
            with _Step(failures):
                self.close()
            _finish_or_note(failures, error, details=self._cleanup_errors or failures)
        return False

    def chain(
        self,
        evaluation_id: str,
        *,
        upstream_url: str | None = None,
        plan=None,
        policy=None,
        faces=("mcp",),
        host_tools=(),
        repeated: bool = False,
        reset: bool = False,
        request: ContextRequest | None = None,
    ) -> ChainScope:
        """Create an unopened chain; entering it performs capability checks."""
        return ChainScope(
            self,
            evaluation_id,
            binding_args={
                "upstream_url": upstream_url,
                "plan": plan,
                "policy": policy,
                "faces": faces,
                "host_tools": host_tools,
            },
            repeated=repeated,
            reset=reset,
            request=request or self.request,
        )

    def _start(self):
        with self._lock:
            if self.injector is not None and self._run is None:
                self._run = _Bounded(
                    self.injector.start_run,
                    self.start_timeout,
                    late=lambda run: run.close(),
                ).get()
            return self._run

    def close(self):
        """Close every open chain, then the injector run, exactly once."""
        if self._run is not None and threading.current_thread() is getattr(
            self._run, "_thread", None
        ):
            raise RuntimeError("coordinator cannot close on the injector loop")
        with self._lock:
            if self._closed:
                return
            self._closed = True
            chains = tuple(self._chains)
        errors: list[BaseException] = []
        run_errors: list[BaseException] = []
        try:
            for chain in chains:
                chain_errors: list[BaseException] = []
                with _Step(chain_errors):
                    chain._finish()
                errors.extend(chain._closure_errors or chain_errors)
        finally:
            if self._run is not None:
                with _Step(run_errors):
                    _Bounded(self._run.close, self.close_timeout).get()
            errors.extend(run_errors)
            self._cleanup_errors = tuple(errors)
            if run_errors:
                names = [type(failure).__name__ for failure in run_errors]
                for chain in self._chains_seen:
                    chain.result["cleanup_failures"].extend(names)
        _raise_failures(errors)

    def __reduce__(self):
        raise TypeError("a live coordinator cannot be pickled")


class ChainScope:
    """Own one collector, one binding, and one environment scope."""

    def __init__(self, owner, evaluation_id, *, binding_args, repeated, reset, request):
        self.owner = owner
        self.evaluation_id = evaluation_id
        self._binding_args = binding_args
        self.repeated = repeated
        self.reset = reset
        self.request = request
        self.collector = None
        self.arming = None
        self.environment_scope = None
        self.tool_block = None
        self.result = {"snapshot": None, "failed_turn": None, "cleanup_failures": []}
        self._lock = threading.RLock()
        self._done = threading.Event()
        self._entered = False
        self._claimed = False
        self._closing = False
        self._sealed = False
        self._closure_failure: BaseException | None = None
        self._closure_errors: tuple[BaseException, ...] = ()
        self._timed_out = False
        self._active_turn = None

    @property
    def closed(self):
        """Whether every cleanup step has finished."""
        return self._done.is_set()

    def __enter__(self):
        owner = self.owner
        with self._lock:
            if self._closing:
                raise RuntimeError("chain is closed")
            if self._claimed or self._entered:
                raise RuntimeError("chain is already entered")
        if owner.environment is None:
            raise CapabilityError("environment capability is required")
        check_capabilities(
            self.request, owner.generator, owner.injector, owner.environment
        )
        owner.environment.check_attempt(repeated=self.repeated)
        if self.reset and "reset" not in getattr(
            owner.environment, "capabilities", frozenset()
        ):
            raise CapabilityError("environment lacks target reset capability")
        failures: list[BaseException] = []
        with self._lock:
            if self._closing:
                raise RuntimeError("chain is closed")
            if self._claimed or self._entered:
                raise RuntimeError("chain is already entered")
            with owner._lock:
                if owner._closed:
                    raise RuntimeError("coordinator is closed")
                if owner._open_count >= owner.max_live:
                    raise CapabilityError(f"max_live cap {owner.max_live} reached")
                if owner._open_count:
                    owner.environment.check_attempt(repeated=True)
                owner._open_count += 1
                self._claimed = True
                owner._chains.add(self)
                owner._chains_seen.add(self)
            with _Step(failures):
                self.collector = owner.collector_factory(self.evaluation_id)
                run = owner._start()
                if run is not None:
                    self.arming = _Bounded(
                        lambda: run.arm(self.collector, **self._binding_args),
                        owner.bind_timeout,
                        late=lambda arming: arming.disarm(),
                    ).get()
                    self.tool_block = self.arming.tool_block
                self.environment_scope = _Bounded(
                    lambda: owner.environment.open(self, reset=self.reset),
                    owner.setup_timeout,
                    late=lambda scope: scope.close(),
                ).get()
                self._entered = True
        if failures:
            cleanup: list[BaseException] = []
            with _Step(cleanup):
                self._finish()
            _finish_or_note(
                cleanup, failures[0], self.result, details=self._closure_errors
            )
            _raise_failures(failures)
        return self

    def __exit__(self, kind, error, traceback):
        failures: list[BaseException] = []
        with _Step(failures):
            self._finish()
        _finish_or_note(failures, error, self.result, details=self._closure_errors)
        return False

    def turn(self, message: str, *, timeout: float | None = None) -> TurnScope:
        """Start a turn with its own deadline; the chain persists across turns."""
        return TurnScope(self, message, timeout or self.owner.turn_timeout)

    def cancel(self, reason: str = "cancelled") -> None:
        """Close on the calling thread, including when called by a watchdog."""
        del reason  # Caller text may include credentials; never record it.
        self._finish()

    def _finish(self):
        run = self.owner._run
        if run is not None and threading.current_thread() is getattr(
            run, "_thread", None
        ):
            raise RuntimeError("coordinator cannot close on the injector loop")
        with self._lock:
            if self._closing:
                waiting = not self._done.is_set()
            else:
                waiting = False
                self._closing = True
                active = self._active_turn
                if active is not None:
                    active._timer.cancel()
        if waiting:
            if not self._done.wait(
                self.owner.drain_timeout + self.owner.close_timeout * 5
            ):
                raise TimeoutError("coordinator chain closure exceeded its bound")
            if self._closure_failure is not None:
                raise self._closure_failure
            return
        if self._done.is_set():
            if self._closure_failure is not None:
                raise self._closure_failure
            return
        errors: list[BaseException] = []
        try:
            if self.collector is not None:
                with _Step(errors):
                    self.collector.begin_closure(self.owner.drain_timeout)
            if self.arming is not None:
                with _Step(errors):
                    released, _ = _Bounded(
                        self.arming.disarm, self.owner.close_timeout
                    ).get()
                    if not released:
                        raise CapabilityError("injector release was not acknowledged")
            if self.collector is not None:
                with _Step(errors):
                    self.collector.drain()
            if self.environment_scope is not None:
                with _Step(errors):
                    observed = _Bounded(
                        self.environment_scope.observe, self.owner.close_timeout
                    ).get()
                    if observed:
                        self.collector.record_observations(
                            "environment", observed, {"state": "complete"}
                        )
                with _Step(errors):
                    _Bounded(
                        self.environment_scope.close, self.owner.close_timeout
                    ).get()
        finally:
            self._closure_errors = tuple(errors)
            self._closure_failure = _failure_object(errors)
            self.result["cleanup_failures"] = [
                type(failure).__name__ for failure in errors
            ]
            with self.owner._lock:
                self.owner._chains.discard(self)
                if self._claimed:
                    self.owner._open_count -= 1
                    self._claimed = False
            self._done.set()
        if self._closure_failure is not None:
            raise self._closure_failure

    def seal(self, agent_output: str = "", *, incomplete: tuple[str, ...] = ()):
        """Release the chain, then seal its immutable evidence snapshot.

        ``incomplete`` names evidence the caller knows was cut short, such as
        ``TURNS``, ``FUNCTION_CALLS`` and ``AGENT_OUTPUT`` for a chain that
        ended before its agent finished. The names join what the coordinator
        marks itself (``TURNS`` after a failed turn; ``FUNCTION_CALLS`` and
        ``ENVIRONMENT`` after a failed closure) and never replace it. The first
        seal fixes the snapshot: a later call returns it and ignores a
        different ``agent_output`` or ``incomplete``. A closure failure is
        sticky, so after one every call raises it again, and the snapshot the
        first call made stays in ``result["snapshot"]``.
        """
        if type(incomplete) is not tuple or any(
            type(name) is not str or not name for name in incomplete
        ):
            raise TypeError("incomplete must be a tuple of evidence names")
        failures: list[BaseException] = []
        with _Step(failures):
            self._finish()
        can_seal = (
            self.collector is not None
            and not self._sealed
            and (not failures or self.collector.stage is Stage.COLLECTING)
        )
        if can_seal:
            with _Step(failures):
                own = (TURNS,) if self.result["failed_turn"] else ()
                if self._closure_failure is not None:
                    own += (FUNCTION_CALLS, ENVIRONMENT)
                self.result["snapshot"] = self.collector.seal(
                    agent_output,
                    incomplete=tuple(dict.fromkeys((*own, *incomplete))),
                )
                self._sealed = True
        _raise_failures(failures)
        return self.result["snapshot"]

    def __reduce__(self):
        raise TypeError("a live chain cannot be pickled")


class TurnScope:
    """One collector turn, with an independent watchdog deadline."""

    def __init__(self, chain: ChainScope, message: str, timeout: float):
        self.chain = chain
        self.message = message
        self.timeout = _positive(timeout, "turn timeout")
        self.index = None
        self._timer = threading.Timer(timeout, self._expire)
        self._timer.daemon = True
        self._watchdog_errors: list[BaseException] = []
        self._watchdog_done = threading.Event()

    def __enter__(self):
        with self.chain._lock:
            if self.chain._closing or self.chain._active_turn is not None:
                raise RuntimeError("chain is closed or has an active turn")
            self.index = self.chain.collector.start_turn(self.message).index
            self.chain._active_turn = self
            self._timer.start()
        return self

    def _expire(self):
        with self.chain._lock:
            self.chain._timed_out = True
            self.chain.result["failed_turn"] = {
                "index": self.index,
                "kind": "timeout",
                "message": "turn timed out",
            }
        try:
            with _Step(self._watchdog_errors):
                self.chain.cancel("deadline")
        finally:
            self._watchdog_done.set()

    def reply(self, text: str):
        """Record the agent's text while this turn still admits replies."""
        if self.chain._timed_out:
            raise TurnTimeout("turn deadline expired")
        return self.chain.collector.record_reply(self.index, text)

    def __exit__(self, kind, error, traceback):
        self._timer.cancel()
        with self.chain._lock:
            self.chain._active_turn = None
            if error is not None and not self.chain._timed_out:
                self.chain.result["failed_turn"] = {
                    "index": self.index,
                    "kind": type(error).__name__,
                    "message": "turn failed",
                }
            timed_out = self.chain._timed_out
        if timed_out:
            if not self._watchdog_done.wait(
                self.chain.owner.drain_timeout + self.chain.owner.close_timeout * 5
            ):
                if error is not None:
                    _finish_or_note(
                        [
                            TimeoutError(
                                "coordinator deadline cleanup exceeded its bound"
                            )
                        ],
                        error,
                        self.chain.result,
                    )
                    return False
                raise TurnTimeout("turn deadline expired") from TimeoutError(
                    "coordinator deadline cleanup exceeded its bound"
                )
            errors = self._watchdog_errors
            if error is not None:
                _finish_or_note(
                    errors,
                    error,
                    self.chain.result,
                    details=self.chain._closure_errors,
                )
                return False
            cause = (
                errors[0]
                if len(errors) == 1
                else (
                    BaseExceptionGroup("coordinator deadline cleanup failed", errors)
                    if errors
                    else error
                )
            )
            raise TurnTimeout("turn deadline expired") from cause
        return False

    def __reduce__(self):
        raise TypeError("a live turn cannot be pickled")
