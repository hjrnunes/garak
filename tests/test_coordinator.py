"""Run, chain and turn ownership across failures and cancellation."""

import asyncio
import contextlib
import gc
import logging
import pickle
import socket
import threading
import time
import weakref
from types import SimpleNamespace

import pytest

from garak.coordinator import (
    _Bounded,
    ChainScope,
    ContextRequest,
    RunCoordinator,
    TurnTimeout,
    check_capabilities,
    select_components,
)
from garak.evidence.collector import Collector
from garak.exception import CapabilityError


class Surprise(Exception):
    """An error deliberately absent from production handler lists."""


class Fakes:
    """Count each resource transition and optionally fail one step."""

    def __init__(self, fail=None, error_type=Surprise):
        self.events = []
        self.fail = fail
        self.error_type = error_type
        self.injector = SimpleNamespace(capabilities=frozenset({"bind"}))
        self.environment = SimpleNamespace(
            capabilities=frozenset({"observe"}), isolation=None
        )
        self.injector.start_run = self.start_run
        self.environment.check_attempt = self.check_attempt
        self.environment.open = self.open

    def step(self, name):
        self.events.append(name)
        if self.fail == name or (isinstance(self.fail, tuple) and name in self.fail):
            raise self.error_type(name)

    def start_run(self):
        self.step("start")
        return SimpleNamespace(arm=self.arm, close=lambda: self.step("run_close"))

    def arm(self, collector, **kwargs):
        self.step("arm")
        return self

    tool_block = None

    def disarm(self):
        self.step("disarm")
        return True, []

    def open(self, chain, *, reset=False):
        self.step("setup")
        return self

    def check_attempt(self, *, repeated=False):
        if repeated and self.environment.isolation is None:
            raise CapabilityError("repeat needs isolation")

    def observe(self):
        self.step("observe")
        return {}

    def close(self):
        self.step("close")


def coordinator(fake, **kwargs):
    """Use a real collector with a fake injector and environment."""
    return RunCoordinator(
        fake.injector, fake.environment, collector_factory=Collector, **kwargs
    )


def require_shared_codec():
    """Skip sealing tests when the optional shared codec is unavailable."""
    pytest.importorskip("midojo_spec")


def test_order_and_multi_turn():
    require_shared_codec()
    fake = Fakes()
    with coordinator(fake) as run:
        with run.chain("c", upstream_url="http://unused", plan=object()) as chain:
            with chain.turn("first") as turn:
                turn.reply("one")
            with chain.turn("second") as turn:
                turn.reply("two")
            result = chain.seal()
    assert fake.events[:3] == ["start", "arm", "setup"], "acquisition order"
    assert fake.events.count("disarm") == 1, "one release"
    assert fake.events.count("observe") == 1, "one observation"
    assert [t.index for t in result.turns] == [0, 1], "turns stamped"


@pytest.mark.parametrize(
    "step", ["arm", "setup", "disarm", "observe", "close", "run_close"]
)
def test_surprise_releases_remaining_resources(step):
    if step == "run_close":
        require_shared_codec()
    fake = Fakes(fail=step)
    with pytest.raises(Surprise):
        with coordinator(fake) as run:
            with run.chain("c", upstream_url="unused", plan=object()) as chain:
                chain.seal()
    assert fake.events.count("disarm") == (0 if step == "arm" else 1), "disarm once"
    assert fake.events.count("observe") == (
        0 if step in ("arm", "setup") else 1
    ), "observe once"
    assert fake.events.count("close") == (
        0 if step in ("arm", "setup") else 1
    ), "scope closed once"
    assert fake.events.count("run_close") == 1, "run closed once"


def test_capability_and_isolation_refused_before_open():
    fake = Fakes()
    req = ContextRequest(required_injector=frozenset({"missing", "another"}))
    with pytest.raises(CapabilityError, match="another.*missing"):
        check_capabilities(req, object(), fake.injector, fake.environment)
    with pytest.raises(CapabilityError, match="isolation"):
        with coordinator(fake) as run:
            with run.chain("repeat", repeated=True):
                pass
    with pytest.raises(CapabilityError, match="reset"):
        with coordinator(fake) as run:
            with run.chain("reset", reset=True):
                pass
    assert "start" not in fake.events, "no resources opened"


def test_selection_precedence(monkeypatch):
    from garak import coordinator as module

    paths = []
    monkeypatch.setattr(
        module._plugins, "load_plugin", lambda p, **kw: paths.append(p) or p
    )
    assert select_components(
        ContextRequest(injector="mim", environment="unmanaged"),
        config_root={
            "plugins": {"injector_type": "other", "environment_type": "other"}
        },
    ) == ("injectors.other", "environments.other"), "config wins"
    assert paths == ["injectors.other", "environments.other"], "loads both"
    paths.clear()
    assert select_components(
        ContextRequest(injector="mim", environment="unmanaged"),
        config_root={"plugins": {}},
    ) == ("injectors.mim", "environments.unmanaged"), "request wins"
    paths.clear()
    assert select_components(ContextRequest(), config_root={"plugins": {}}) == (
        None,
        "environments.unmanaged",
    ), "none means no injector"
    assert paths == ["environments.unmanaged"], "no injector is loaded"


def test_reentrant_lazy_and_live_cap():
    require_shared_codec()
    fake = Fakes()
    run = coordinator(fake, max_live=1)
    with run:
        with run:
            assert not fake.events, "run start is lazy"
            with run.chain("one", upstream_url="unused", plan=object()) as chain:
                with pytest.raises(CapabilityError, match="max_live"):
                    with run.chain("two"):
                        pass
                chain.seal()
        assert fake.events.count("run_close") == 0, "inner exit keeps owner"
    assert fake.events.count("start") == 1, "started once"
    assert fake.events.count("run_close") == 1, "outer exit closes run"


def test_second_live_chain_requires_isolation_before_collector_opens():
    from garak.environments.unmanaged import Unmanaged

    collectors = []

    def collect(name):
        collectors.append(name)
        return Collector(name)

    with RunCoordinator(
        None, Unmanaged(), max_live=2, collector_factory=collect
    ) as run:
        with run.chain("first") as first:
            with pytest.raises(CapabilityError, match="Unmanaged.*isolation"):
                with run.chain("second"):
                    pass
            assert collectors == ["first"], "second collector not opened"
        with run.chain("third") as third:
            assert collectors == ["first", "third"], "refusal left cap unchanged"
    assert first.closed and third.closed, "both accepted scopes closed"

    fake = Fakes()
    fake.environment.isolation = "per-chain"
    with coordinator(fake, max_live=2) as run:
        with run.chain("first") as first:
            with run.chain("second") as second:
                with pytest.raises(CapabilityError, match="max_live"):
                    with run.chain("third"):
                        pass
    assert first.closed and second.closed, "isolated scopes closed"
    assert fake.events.count("arm") == 2, "only admitted chains armed"
    assert fake.events.count("disarm") == 2, "both bindings disarmed"
    assert fake.events.count("close") == 2, "both scopes closed"
    assert fake.events.count("run_close") == 1, "run owner closed"


def test_closed_chains_are_not_retained_by_run():
    from garak.environments.unmanaged import Unmanaged

    fake = Fakes(fail="run_close")
    run = RunCoordinator(fake.injector, Unmanaged())
    with pytest.raises(Surprise):
        with run:
            for index in range(30):
                with run.chain(
                    f"chain-{index}", upstream_url="unused", plan=object()
                ) as scope:
                    pass
                if index == 0:
                    first = weakref.ref(scope)
                if index == 29:
                    held = scope
                del scope
            gc.collect()
            assert first() is None, "closed chain can be collected during run"
    assert held.closed, "held chain closed"
    assert held.result["cleanup_failures"] == ["Surprise"], "run failure reaches holder"
    assert fake.events.count("disarm") == 30, "all bindings disarmed"
    assert fake.events.count("run_close") == 1, "run owner closed once"


def test_chain_refuses_second_enter_without_replacing_resources():
    fake = Fakes()
    fake.environment.isolation = "per-chain"
    run = coordinator(fake, max_live=2)
    with run:
        scope = run.chain("one", upstream_url="unused", plan=object())
        with scope:
            with pytest.raises(RuntimeError, match="entered"):
                scope.__enter__()
            with run.chain("another"):
                pass
        with pytest.raises(RuntimeError, match="closed"):
            scope.__enter__()
    assert fake.events.count("arm") == 2, "one arming per chain"
    assert fake.events.count("setup") == 2, "one scope per chain"
    assert fake.events.count("disarm") == 2, "both armings released"
    assert fake.events.count("close") == 2, "both scopes closed once"
    assert fake.events.count("run_close") == 1, "run closed once"


def test_body_failure_and_keyboard_interrupt_close():
    for error in (Surprise, KeyboardInterrupt):
        fake = Fakes()
        run = coordinator(fake)
        with pytest.raises(error):
            with run:
                with run.chain("c", upstream_url="unused", plan=object()):
                    raise error("body")
        assert fake.events.count("disarm") == 1, "body failure releases"


@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt])
def test_body_error_survives_release_failure(error_type):
    fake = Fakes()
    fake.disarm = lambda: (fake.step("disarm"), (False, []))[1]
    run = coordinator(fake)
    scope = run.chain("c", upstream_url="unused", plan=object())
    with pytest.raises(error_type) as raised:
        with run:
            with scope:
                raise error_type("body")
    assert raised.value.args == ("body",), "body exception retained"
    assert raised.value.__notes__ == [
        "1 coordinator cleanup failure: CapabilityError"
    ], "failure types noted without messages"
    assert scope.result["cleanup_failures"] == ["CapabilityError"], "release recorded"
    assert scope.closed, "scope closed"
    assert fake.events.count("disarm") == 1, "disarm attempted once"
    assert fake.events.count("close") == 1, "environment closed once"
    assert fake.events.count("run_close") == 1, "run closed once"


@pytest.mark.parametrize("kind", ["immutable", "tuple_notes"])
def test_cleanup_note_refusal_never_masks_body(kind):
    class ImmutableError(Exception):
        def __setattr__(self, name, value):
            raise AttributeError("attributes refused")

    fake = Fakes()
    fake.disarm = lambda: (fake.step("disarm"), (False, []))[1]
    body = ImmutableError("body") if kind == "immutable" else ValueError("body")
    if kind == "tuple_notes":
        body.__notes__ = ("existing note",)
    run = coordinator(fake)
    scope = run.chain("c", upstream_url="unused", plan=object())
    with pytest.raises(type(body)) as raised:
        with run:
            with scope:
                raise body
    assert raised.value is body, "original exception retained"
    assert scope.result["cleanup_failures"] == ["CapabilityError"], "loss recorded"
    assert scope.closed, "scope closed"
    assert fake.events.count("disarm") == 1, "disarm attempted once"
    assert fake.events.count("close") == 1, "environment closed once"
    assert fake.events.count("run_close") == 1, "run closed once"


def test_cleanup_error_refusing_notes_does_not_mask_body():
    class TupleNotesSurprise(Surprise):
        def __init__(self, message):
            super().__init__(message)
            self.__notes__ = ("existing note",)

    fake = Fakes(fail=("disarm", "observe"), error_type=TupleNotesSurprise)
    run = coordinator(fake)
    scope = run.chain("c", upstream_url="unused", plan=object())
    body = ValueError("body")
    with pytest.raises(ValueError) as raised:
        with run:
            with scope:
                raise body
    assert raised.value is body, "body remains original"
    assert scope.result["cleanup_failures"] == [
        "TupleNotesSurprise",
        "TupleNotesSurprise",
    ], "both failures recorded"
    assert scope.closed, "scope closed"
    assert fake.events.count("disarm") == 1, "disarm attempted"
    assert fake.events.count("observe") == 1, "observation attempted"
    assert fake.events.count("close") == 1, "environment closed"
    assert fake.events.count("run_close") == 1, "run closed"


def test_entry_failure_survives_release_failure():
    fake = Fakes(fail="setup")
    fake.disarm = lambda: (fake.step("disarm"), (False, []))[1]
    scope = coordinator(fake).chain("c", upstream_url="unused", plan=object())
    with pytest.raises(Surprise) as raised:
        with scope.owner:
            scope.__enter__()
    assert raised.value.args == ("setup",), "entry error retained"
    assert raised.value.__notes__ == [
        "1 coordinator cleanup failure: CapabilityError"
    ], "release type noted"
    assert scope.result["cleanup_failures"] == ["CapabilityError"], "loss recorded"
    assert scope.closed, "chain closed"
    assert fake.events.count("disarm") == 1, "arming disarmed once"
    assert fake.events.count("run_close") == 1, "run owner closed"


def test_body_error_notes_each_cleanup_step_not_group_type():
    fake = Fakes(fail=("disarm", "close"), error_type=RuntimeError)
    scope = coordinator(fake).chain("c", upstream_url="unused", plan=object())
    with pytest.raises(ValueError) as raised:
        with scope.owner:
            with scope:
                raise ValueError("body")
    assert raised.value.args == ("body",), "body retained"
    assert raised.value.__notes__ == [
        "2 coordinator cleanup failures: RuntimeError, RuntimeError"
    ], "step types noted"
    assert scope.result["cleanup_failures"] == [
        "RuntimeError",
        "RuntimeError",
    ], "step types recorded"
    assert scope.closed, "scope closed"
    assert fake.events.count("disarm") == 1, "disarm attempted once"
    assert fake.events.count("close") == 1, "environment close attempted once"
    assert fake.events.count("run_close") == 1, "run closed once"


def test_run_exit_retains_body_error_and_records_run_close():
    fake = Fakes(fail="run_close")
    scope = coordinator(fake).chain("c", upstream_url="unused", plan=object())
    with pytest.raises(ValueError) as raised:
        with scope.owner:
            with scope:
                raise ValueError("body")
    assert raised.value.args == ("body",), "body error retained"
    assert raised.value.__notes__ == [
        "1 coordinator cleanup failure: Surprise"
    ], "run failure type noted"
    assert scope.result["cleanup_failures"] == ["Surprise"], "run failure recorded"
    assert scope.closed, "chain closed"
    assert fake.events.count("disarm") == 1, "arming disarmed once"
    assert fake.events.count("run_close") == 1, "run close attempted once"


def test_run_exit_notes_each_failure_when_chain_still_open():
    fake = Fakes(fail=("disarm", "run_close"), error_type=RuntimeError)
    run = coordinator(fake)
    scope = run.chain("open", upstream_url="unused", plan=object())
    with pytest.raises(ValueError) as raised:
        with run:
            scope.__enter__()
            raise ValueError("body")
    assert raised.value.__notes__ == [
        "2 coordinator cleanup failures: RuntimeError, RuntimeError"
    ], "run exit lists step types"
    assert scope.result["cleanup_failures"] == [
        "RuntimeError",
        "RuntimeError",
    ], "chain records both failures"
    assert scope.closed, "open chain closed"
    assert fake.events.count("disarm") == 1, "binding disarmed once"
    assert fake.events.count("close") == 1, "environment closed once"
    assert fake.events.count("run_close") == 1, "run close attempted once"


def test_chain_and_run_failures_note_each_exit():
    fake = Fakes(fail="run_close")
    fake.disarm = lambda: (fake.step("disarm"), (False, []))[1]
    scope = coordinator(fake).chain("c", upstream_url="unused", plan=object())
    with pytest.raises(ValueError) as raised:
        with scope.owner:
            with scope:
                raise ValueError("body")
    assert raised.value.__notes__ == [
        "1 coordinator cleanup failure: CapabilityError",
        "1 coordinator cleanup failure: Surprise",
    ], "each exit notes its failure"
    assert scope.result["cleanup_failures"] == [
        "CapabilityError",
        "Surprise",
    ], "both releases recorded"
    assert scope.closed, "scope closed"
    assert fake.events.count("disarm") == 1, "disarm attempted once"
    assert fake.events.count("close") == 1, "environment closed once"
    assert fake.events.count("run_close") == 1, "run close attempted once"


def test_cancel_from_another_thread_and_double_exit():
    fake = Fakes()
    with coordinator(fake) as run:
        scope = run.chain("c", upstream_url="unused", plan=object())
        with scope as chain:
            worker = threading.Thread(target=chain.cancel, args=("stop",))
            worker.start()
            worker.join(0.5)
            assert not worker.is_alive(), "cancellation bounded"
        scope.__exit__(None, None, None)
    assert fake.events.count("disarm") == 1, "only one disarm"


def test_blocked_turn_deadline_releases_before_caller_returns():
    fake = Fakes()
    entered = threading.Event()
    leave = threading.Event()
    with coordinator(fake, turn_timeout=0.04, close_timeout=0.08) as run:
        with run.chain("c", upstream_url="unused", plan=object()) as chain:

            def blocked():
                with pytest.raises(TurnTimeout):
                    with chain.turn("hi"):
                        entered.set()
                        leave.wait(0.3)

            worker = threading.Thread(target=blocked)
            worker.start()
            assert entered.wait(0.2), "turn entered"
            started = time.monotonic()
            while not chain.closed and time.monotonic() - started < 0.5:
                time.sleep(0.005)
            assert chain.closed, "watchdog closes blocked chain"
            assert time.monotonic() - started < 0.5, "deadline is short"
            leave.set()
            worker.join(0.3)
            assert not worker.is_alive(), "caller returned"
    assert fake.events.count("disarm") == 1, "deadline disarms"


def test_failed_turn_result_and_pickling():
    fake = Fakes()
    with coordinator(fake) as run:
        with run.chain("c", upstream_url="unused", plan=object()) as chain:
            with pytest.raises(Surprise):
                with chain.turn("hi") as turn:
                    raise Surprise("fake-token")
            assert chain.result["failed_turn"]["index"] == 0, "turn failure indexed"
            for live in (run, chain, turn):
                with pytest.raises(TypeError):
                    pickle.dumps(live)
    assert fake.events.count("disarm") == 1, "failed turn releases"


@pytest.mark.parametrize("step", ["begin", "drain", "seal", "observations"])
def test_collector_surprise_still_releases(step):
    fake = Fakes()

    class FaultCollector(Collector):
        def begin_closure(self, wait_seconds):
            if step == "begin":
                raise Surprise("begin")
            return super().begin_closure(wait_seconds)

        def drain(self):
            if step == "drain":
                raise Surprise("drain")
            return super().drain()

        def seal(self, output, **kwargs):
            if step == "seal":
                raise Surprise("seal")
            return super().seal(output, **kwargs)

        def record_observations(self, *args):
            if step == "observations":
                raise Surprise("observations")
            return super().record_observations(*args)

    fake.observe = lambda: {"state": "ok"} if step == "observations" else {}
    run = RunCoordinator(
        fake.injector, fake.environment, collector_factory=FaultCollector
    )
    with pytest.raises(Surprise):
        with run:
            with run.chain("c", upstream_url="unused", plan=object()) as chain:
                chain.seal()
    assert fake.events.count("disarm") == 1, "disarm after collector fault"
    assert fake.events.count("close") == 1, "environment closed after collector fault"
    assert fake.events.count("run_close") == 1, "run owner closed"


def test_cancel_races_normal_exit():
    fake = Fakes()
    run = coordinator(fake)
    with run:
        with run.chain("c", upstream_url="unused", plan=object()) as chain:
            worker = threading.Thread(target=chain.cancel)
            worker.start()
        worker.join(0.5)
        assert not worker.is_alive(), "cancel returned"
    assert fake.events.count("disarm") == 1, "race disarms once"
    assert fake.events.count("close") == 1, "race closes scope once"


def test_racing_seal_reports_persistent_failed_release_as_incomplete():
    require_shared_codec()
    fake = Fakes()
    disarming = threading.Event()
    release = threading.Event()
    started = threading.Event()
    cancelled = []
    sealed = []

    def disarm():
        fake.step("disarm")
        disarming.set()
        release.wait(0.4)
        return False, []

    fake.disarm = disarm
    with coordinator(fake, close_timeout=0.2) as run:
        scope = run.chain("racing", upstream_url="unused", plan=object())
        scope.__enter__()

        def cancel():
            try:
                scope.cancel()
            except CapabilityError as error:
                cancelled.append(error)

        def seal():
            started.set()
            try:
                scope.seal()
            except CapabilityError as error:
                sealed.append(error)

        closer = threading.Thread(target=cancel)
        waiter = threading.Thread(target=seal)
        closer.start()
        assert disarming.wait(0.2), "release started"
        waiter.start()
        assert started.wait(0.2), "sealer started"
        release.set()
        closer.join(0.4)
        waiter.join(0.4)
        assert not closer.is_alive() and not waiter.is_alive(), "both closed"
        assert len(cancelled) == len(sealed) == 1, "both see release failure"
        assert sealed[0] is cancelled[0], "waiter sees original failure"
        snapshot = scope.result["snapshot"]
        assert (
            snapshot.evidence_status["function_calls"] == "incomplete"
        ), "seal marks loss"
        with pytest.raises(CapabilityError) as again:
            scope.seal()
        assert again.value is cancelled[0], "second seal also fails"
        assert scope.result["snapshot"] is snapshot, "snapshot unchanged"
    assert scope.closed, "scope closed"
    assert fake.events.count("disarm") == 1, "one release attempt"
    assert fake.events.count("close") == 1, "environment closed once"
    assert fake.events.count("run_close") == 1, "run closed once"


def test_serial_failed_closure_seals_incomplete_and_never_recovers():
    require_shared_codec()
    fake = Fakes()
    fake.disarm = lambda: (fake.step("disarm"), (False, []))[1]
    with coordinator(fake) as run:
        scope = run.chain("serial", upstream_url="unused", plan=object())
        scope.__enter__()
        with pytest.raises(CapabilityError) as closed:
            scope.cancel()
        with pytest.raises(CapabilityError) as first:
            scope.seal()
        assert first.value is closed.value, "serial seal reports release failure"
        snapshot = scope.result["snapshot"]
        assert (
            snapshot.evidence_status["function_calls"] == "incomplete"
        ), "call loss marked"
        assert (
            snapshot.evidence_status["environment"] == "incomplete"
        ), "teardown marked"
        with pytest.raises(CapabilityError) as second:
            scope.seal()
        assert second.value is closed.value, "second seal still fails"
        assert scope.result["snapshot"] is snapshot, "snapshot fixed"
    assert fake.events.count("disarm") == 1, "one disarm attempt"
    assert fake.events.count("close") == 1, "environment closed once"
    assert fake.events.count("run_close") == 1, "run closed once"


def test_failed_closure_before_collecting_never_seals():
    fake = Fakes()

    class FaultCollector(Collector):
        def begin_closure(self, wait_seconds):
            raise Surprise("closure")

    with RunCoordinator(
        fake.injector, fake.environment, collector_factory=FaultCollector
    ) as run:
        scope = run.chain("no-seal", upstream_url="unused", plan=object())
        scope.__enter__()
        with pytest.raises(Surprise) as closed:
            scope.cancel()
        with pytest.raises(Surprise) as first:
            scope.seal()
        with pytest.raises(Surprise) as second:
            scope.seal()
        assert first.value is second.value is closed.value, "failure persists"
        assert scope.result["snapshot"] is None, "collector never sealed"
    assert scope.closed, "scope closed"
    assert fake.events.count("disarm") == 1, "binding disarmed"
    assert fake.events.count("close") == 1, "environment closed once"
    assert fake.events.count("run_close") == 1, "run closed once"


def test_bounded_blocking_environment_close():
    fake = Fakes()
    entered = threading.Event()
    release = threading.Event()

    def blocked_close():
        entered.set()
        release.wait(1)
        fake.step("close")

    fake.close = blocked_close
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="bound"):
        with coordinator(fake, close_timeout=0.04) as run:
            with run.chain("c", upstream_url="unused", plan=object()):
                pass
    assert entered.is_set(), "close attempted"
    assert time.monotonic() - started < 0.5, "blocked close bounded"
    assert fake.events.count("disarm") == 1, "disarmed despite close timeout"
    assert fake.events.count("run_close") == 1, "run closed despite close timeout"
    release.set()


@pytest.fixture
def live_service():
    """Run the real local MiM with an in-process MCP upstream."""
    pytest.importorskip("midojo_mim")
    require_shared_codec()
    from fastmcp import FastMCP
    import uvicorn
    from garak.injectors.mim import Mim

    app = FastMCP("coordinator-upstream")

    @app.tool
    def weather(city: str) -> str:
        """Return a predictable tool reply."""
        return f"Rain in {city}"

    owner = Mim().start_run()
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]

    class Server(uvicorn.Server):
        @contextlib.contextmanager
        def capture_signals(self):
            yield

    async def start():
        server = Server(
            uvicorn.Config(
                app.http_app(path="/mcp"),
                log_level="warning",
                lifespan="on",
                ws="websockets-sansio",
            )
        )
        task = asyncio.create_task(server.serve(sockets=[listener]))
        while not server.started:
            if task.done():
                raise RuntimeError("upstream stopped")
            await asyncio.sleep(0.01)
        return server, task

    server, task = owner._submit(start())
    try:
        yield owner, f"http://127.0.0.1:{port}/mcp"
    finally:

        async def stop():
            server.should_exit = True
            await task

        owner._submit(stop())
        owner.close()


def test_real_socket_release_and_redaction(live_service, caplog):
    """The actual binding acknowledges disarm and leaves no live loop or port."""
    from midojo_spec.injection import (
        Channel,
        InjectionInstruction,
        InjectionPlan,
        InjectionTarget,
    )
    from garak.injectors.mim import Mim
    from garak.environments.unmanaged import Unmanaged

    owner, upstream = live_service
    plan = InjectionPlan(
        (
            InjectionInstruction(
                channel=Channel.TOOL_OUTPUT,
                payload="Injected",
                target=InjectionTarget(tool="weather"),
                probe_key="weather:note",
            ),
        )
    )
    injector = Mim(
        config_root={
            "injectors": {
                "mim": {
                    "bind_timeout": 1,
                    "release_timeout": 1,
                }
            }
        }
    )
    with caplog.at_level(logging.DEBUG):
        with RunCoordinator(injector, Unmanaged(), close_timeout=2) as run:
            with run.chain("socket", upstream_url=upstream, plan=plan) as chain:
                block = chain.tool_block
                assert block.tool_url, "binding returns tool endpoint"
                snapshot = chain.seal()
                assert snapshot.evaluation_id == "socket", "snapshot sealed"
            loop = run._run._thread
            port = int(run._run._url.rsplit(":", 1)[1])
    assert not loop.is_alive(), "loop thread joined"
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout=0.1)
    assert block.tool_token not in caplog.text, "tool token not logged"
    assert owner._credential not in caplog.text, "bind credential not logged"


def test_multiple_teardown_failures_are_aggregated():
    fake = Fakes(fail=("disarm", "observe", "close"), error_type=RuntimeError)
    with pytest.raises(ExceptionGroup) as captured:
        with coordinator(fake) as run:
            with run.chain("c", upstream_url="unused", plan=object()):
                pass
    assert len(captured.value.exceptions) == 3, "all failures reported"
    assert fake.events.count("disarm") == 1, "disarm attempted once"
    assert fake.events.count("observe") == 1, "observation attempted once"
    assert fake.events.count("close") == 1, "scope closed once"
    assert fake.events.count("run_close") == 1, "run owner closed"


def test_unexpected_teardown_error_stays_unconverted_with_other_failure():
    fake = Fakes(fail="disarm")
    fake.close = lambda: (_ for _ in ()).throw(ValueError("scope close"))
    with pytest.raises(Surprise) as captured:
        with coordinator(fake) as run:
            with run.chain("c", upstream_url="unused", plan=object()):
                pass
    assert str(captured.value) == "disarm", "original type and text retained"
    assert fake.events.count("observe") == 1, "observation still attempted"
    assert fake.events.count("run_close") == 1, "run closed after both errors"


def test_blocked_run_start_times_out_then_late_owner_closes():
    fake = Fakes()
    entered = threading.Event()
    release = threading.Event()
    closed = threading.Event()

    def start_run():
        entered.set()
        release.wait(0.5)
        return SimpleNamespace(arm=fake.arm, close=closed.set)

    fake.injector.start_run = start_run
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="bound"):
        with coordinator(fake, start_timeout=0.04, close_timeout=0.05) as run:
            with run.chain("c", upstream_url="unused", plan=object()):
                pass
    assert entered.is_set(), "start attempted"
    assert time.monotonic() - started < 0.5, "start bounded"
    release.set()
    assert closed.wait(0.2), "late run owner released"


def test_close_races_chain_acquisition_without_orphaning_arming():
    fake = Fakes()
    arming_started = threading.Event()
    release_arm = threading.Event()
    original_arm = fake.arm

    def arm(*args, **kwargs):
        arming_started.set()
        release_arm.wait(0.3)
        return original_arm(*args, **kwargs)

    fake.injector.start_run = lambda: SimpleNamespace(
        arm=arm, close=lambda: fake.step("run_close")
    )
    run = coordinator(fake)
    scope = run.chain("c", upstream_url="unused", plan=object())
    opener = threading.Thread(target=scope.__enter__)
    opener.start()
    assert arming_started.wait(0.2), "arm started"
    closer = threading.Thread(target=run.close)
    closer.start()
    release_arm.set()
    opener.join(0.5)
    closer.join(0.5)
    assert not opener.is_alive() and not closer.is_alive(), "threads finished"
    assert fake.events.count("disarm") == 1, "arming not orphaned"
    assert fake.events.count("close") == 1, "scope closed once"
    assert fake.events.count("run_close") == 1, "run closed once"


def test_deadline_cleanup_failure_reaches_caller():
    fake = Fakes(fail="disarm")
    release = threading.Event()
    with coordinator(fake, turn_timeout=0.03) as run:
        chain = run.chain("c", upstream_url="unused", plan=object())
        with pytest.raises(Surprise) as closed:
            with chain:
                with pytest.raises(TurnTimeout) as captured:
                    with chain.turn("hi"):
                        release.wait(0.15)
                assert isinstance(
                    captured.value.__cause__, Surprise
                ), "cleanup error chained"
        assert closed.value is captured.value.__cause__, "closure failure persists"
    assert fake.events.count("disarm") == 1, "disarm attempted"
    assert fake.events.count("close") == 1, "scope closed"
    assert fake.events.count("run_close") == 1, "run closed"


def test_turn_body_error_survives_deadline_cleanup_failure():
    fake = Fakes(fail="disarm")
    run = coordinator(fake, turn_timeout=0.03)
    scope = run.chain("c", upstream_url="unused", plan=object())
    with pytest.raises(ValueError) as raised:
        with run:
            with scope:
                with scope.turn("hi"):
                    time.sleep(0.07)
                    raise ValueError("body")
    assert raised.value.args == ("body",), "turn body retained"
    assert raised.value.__notes__ == [
        "1 coordinator cleanup failure: Surprise",
        "1 coordinator cleanup failure: Surprise",
    ], "deadline cleanup type noted"
    assert scope.result["cleanup_failures"] == ["Surprise"], "failure recorded"
    assert scope.closed, "scope closed"
    assert fake.events.count("disarm") == 1, "disarm attempted once"
    assert fake.events.count("close") == 1, "environment closed once"
    assert fake.events.count("run_close") == 1, "run closed once"


def test_turn_deadline_notes_each_cleanup_step_not_group_type():
    fake = Fakes(fail=("disarm", "observe"), error_type=RuntimeError)
    run = coordinator(fake, turn_timeout=0.03)
    scope = run.chain("c", upstream_url="unused", plan=object())
    with pytest.raises(ValueError) as raised:
        with run:
            with scope:
                with scope.turn("hi"):
                    time.sleep(0.07)
                    raise ValueError("body")
    assert raised.value.__notes__ == [
        "2 coordinator cleanup failures: RuntimeError, RuntimeError",
        "2 coordinator cleanup failures: RuntimeError, RuntimeError",
    ], "deadline notes step types"
    assert scope.result["cleanup_failures"] == [
        "RuntimeError",
        "RuntimeError",
    ], "step failures recorded"
    assert scope.closed, "scope closed"
    assert fake.events.count("disarm") == 1, "disarm attempted once"
    assert fake.events.count("observe") == 1, "observe attempted once"
    assert fake.events.count("close") == 1, "environment closed once"
    assert fake.events.count("run_close") == 1, "run closed once"


def test_blocked_bind_times_out_and_late_arming_disarms():
    fake = Fakes()
    entered = threading.Event()
    release = threading.Event()
    late_disarm = threading.Event()

    def arm(*args, **kwargs):
        entered.set()
        release.wait(0.5)
        return SimpleNamespace(
            tool_block=None,
            disarm=lambda: late_disarm.set() or (True, []),
        )

    fake.injector.start_run = lambda: SimpleNamespace(
        arm=arm, close=lambda: fake.step("run_close")
    )
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="bound"):
        with coordinator(fake, bind_timeout=0.04, close_timeout=0.05) as run:
            with run.chain("c", upstream_url="unused", plan=object()):
                pass
    assert entered.is_set(), "bind attempted"
    assert time.monotonic() - started < 0.5, "bind bounded"
    assert fake.events.count("run_close") == 1, "run owner closed"
    release.set()
    assert late_disarm.wait(0.2), "late arming released"


def test_late_callback_never_blocks_timed_out_waiter(caplog):
    finished = threading.Event()
    release = threading.Event()
    late_done = threading.Event()
    calls = []
    errors = []

    class TimedOutAfterFinish:
        def __init__(self):
            self.event = threading.Event()

        def wait(self, timeout):
            assert self.event.wait(0.2), "operation finished"
            return False

        def set(self):
            self.event.set()

        def is_set(self):
            return self.event.is_set()

    def late(value):
        calls.append((value, threading.current_thread()))
        finished.set()
        release.wait(0.4)
        late_done.set()
        raise ValueError("fake-token")

    bounded = _Bounded(lambda: "scope", 0.03, late=late)
    bounded.done = TimedOutAfterFinish()

    def wait():
        try:
            bounded.get()
        except (TimeoutError, ValueError) as error:
            errors.append(error)

    waiter = threading.Thread(target=wait)
    with caplog.at_level(logging.DEBUG):
        waiter.start()
        try:
            assert finished.wait(0.2), "late release started"
            waiter.join(0.15)
            assert not waiter.is_alive(), "waiter returns within bound"
        finally:
            release.set()
            waiter.join(0.4)
        assert late_done.wait(0.2), "late release completed"
        calls[0][1].join(0.2)
        assert not calls[0][1].is_alive(), "late worker finished"
        assert "ValueError" in caplog.text, "late failure type logged"
    assert len(errors) == 1 and type(errors[0]) is TimeoutError, "timeout retained"
    assert len(calls) == 1 and calls[0][0] == "scope", "late release runs once"
    assert calls[0][1] is not waiter, "late release runs elsewhere"
    assert "fake-token" not in caplog.text, "late failure text redacted"


@pytest.mark.parametrize("close_via_run", [False, True])
def test_cancel_during_setup_is_bounded_and_late_scope_closes_once(close_via_run):
    fake = Fakes()
    entered = threading.Event()
    release = threading.Event()
    late_closed = threading.Event()
    close_count = []
    failures = []

    def setup(chain, *, reset=False):
        fake.step("setup")
        entered.set()
        release.wait(0.6)

        def close():
            close_count.append(1)
            late_closed.set()

        return SimpleNamespace(observe=lambda: {}, close=close)

    fake.environment.open = setup
    run = coordinator(fake, setup_timeout=0.08, close_timeout=0.05)
    chain = run.chain("setup", upstream_url="unused", plan=object())

    def acquire():
        with pytest.raises(TimeoutError, match="bound"):
            chain.__enter__()

    with run:
        opener = threading.Thread(target=acquire)
        opener.start()
        assert entered.wait(0.2), "setup started"
        action = run.close if close_via_run else chain.cancel
        canceller = threading.Thread(target=lambda: failures.append(action()))
        started = time.monotonic()
        canceller.start()
        try:
            opener.join(0.4)
            canceller.join(0.4)
            assert not opener.is_alive() and not canceller.is_alive(), "cancel bounded"
            assert time.monotonic() - started < 0.4, "setup bound respected"
            assert chain.closed, "chain closed"
            assert fake.events.count("disarm") == 1, "arming disarmed once"
            assert failures == [None], "close returned"
        finally:
            release.set()
            opener.join(0.4)
            canceller.join(0.4)
    assert late_closed.wait(0.2), "late scope closed"
    assert len(close_count) == 1, "late scope closed once"
    assert fake.events.count("run_close") == 1, "run owner closed"


def test_loop_thread_refusal_has_no_cleanup_side_effects():
    fake = Fakes()
    run = coordinator(fake)
    with run:
        with run.chain("c", upstream_url="unused", plan=object()) as chain:
            run._run._thread = threading.current_thread()
            before = list(fake.events)
            with pytest.raises(RuntimeError, match="injector loop"):
                chain.cancel()
            with pytest.raises(RuntimeError, match="injector loop"):
                run.close()
            assert fake.events == before, "refusal does not release resources"
            run._run._thread = None
    assert fake.events.count("disarm") == 1, "later normal close disarms"
    assert fake.events.count("run_close") == 1, "later normal close stops run"
