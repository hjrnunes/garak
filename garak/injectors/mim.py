"""Remote MCP interception through the optional midojo-binding-v1 service."""

from __future__ import annotations

import asyncio
from concurrent.futures import CancelledError
import logging
import math
import os
import secrets
import threading
import uuid
from urllib.parse import urlsplit

from garak.evidence.facts import LateFactError
from garak.evidence.records import InvalidRequestError, RejectedError
from garak.exception import CapabilityError
from garak.injectors.base import Arming, Injector, ToolBlock

DEFAULT_CLASS = "Mim"
_FRAME_LIMIT = 16 * 1024 * 1024
_OUTCOME_FIELDS = frozenset(
    {
        "instruction_index",
        "slot_key",
        "occurrence",
        "status",
        "detail",
        "selected",
        "transformation",
        "emitted",
    }
)
_PLACEMENT_FIELDS = frozenset({"payload_digest", "placed_digest", "slot"})
_PLACED_STATUSES = frozenset({"applied", "partial", "fallback"})
_CLEANUP_ERRORS = (RuntimeError, OSError, TimeoutError, ValueError, CancelledError)


def _bind_url(service_url: str) -> str:
    """Validate a service endpoint and retain its reverse-proxy path prefix."""
    try:
        if not isinstance(service_url, str):
            raise ValueError("invalid service URL")
        parts = urlsplit(service_url)
        if (
            parts.scheme not in ("http", "https")
            or not parts.hostname
            or parts.username is not None
            or parts.password is not None
            or parts.query
            or parts.fragment
        ):
            raise ValueError("invalid authority")
        _ = parts.port
    except (TypeError, ValueError):
        raise CapabilityError(
            "Mim injector needs a valid http(s) service_url"
        ) from None
    scheme = "wss" if parts.scheme == "https" else "ws"
    path = parts.path.rstrip("/")
    if not path.endswith("/v1/bindings"):
        path += "/v1/bindings"
    return f"{scheme}://{parts.netloc}{path}"


def _evidence_outcome(outcome: dict) -> dict:
    """Keep the outcome fields supported by the collector's evidence codec."""
    fields = _OUTCOME_FIELDS
    if outcome["status"] in _PLACED_STATUSES:
        fields = fields | _PLACEMENT_FIELDS
    return {key: value for key, value in outcome.items() if key in fields}


class _RedactSecrets(logging.Filter):
    """Scrub sensitive WebSocket DEBUG records before any handler sees them."""

    def __init__(self, *secrets_to_hide: str):
        super().__init__()
        self.secrets = set(secrets_to_hide)

    def add(self, secret: str) -> None:
        """Protect the tool token as soon as bound names it."""
        self.secrets.add(secret)

    def clean(self, message: str) -> str:
        """Hide known credentials and as-yet-unseen tokens in bound frames."""
        if '"tool_token"' in message:
            return "[redacted binding frame]"
        for secret in self.secrets:
            if secret:
                message = message.replace(secret, "[redacted]")
        return message

    def filter(self, record: logging.LogRecord) -> bool:
        """Replace secrets in message text and formatting arguments."""
        record.msg, record.args = self.clean(record.getMessage()), ()
        return True


class Mim(Injector):
    """Configure a binding service without starting it or importing optional packages."""

    extra_dependency_names = ["midojo_spec", "midojo_mim"]
    service_url: str | None = None
    bind_credential: str | None = None
    bind_timeout: float = 10.0
    release_timeout: float = 10.0

    def start_run(self) -> MimRun:
        """Acquire a private loop and optionally a local interceptor listener."""
        for name in ("bind_timeout", "release_timeout"):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise CapabilityError(
                    f"Mim injector requires a finite, positive {name} capability"
                )
        if bool(self.service_url) != bool(self.bind_credential):
            raise CapabilityError(
                "Mim injector requires both service_url and bind_credential"
            )
        url = self.service_url or os.getenv("MIDOJO_INTERCEPTOR_URL")
        credential = self.bind_credential or os.getenv("MIDOJO_INTERCEPTOR_CREDENTIAL")
        if bool(url) != bool(credential):
            raise CapabilityError(
                "Mim injector requires both service_url and bind_credential"
            )
        return MimRun(url, credential, self.bind_timeout, self.release_timeout)


class MimRun:
    """One run's loop and service; call arm, disarm and close from other threads."""

    extra_dependency_names = ["midojo_spec", "midojo_mim"]

    def __init__(
        self,
        url: str | None,
        credential: str | None,
        bind_timeout: float,
        release_timeout: float,
    ):
        from garak.configurable import Configurable

        try:
            Configurable._load_deps(self)
        except ModuleNotFoundError:
            raise CapabilityError(
                "Mim injector lacks midojo-mim capability; pip install garak[midojo-mim]"
            ) from None
        from midojo_spec import link
        from midojo_mim.service import InterceptorService

        self.wire = link
        self._service_type = InterceptorService
        self._url = url
        self._credential = credential or secrets.token_urlsafe(32)
        self._bind_timeout = bind_timeout
        self._release_timeout = release_timeout
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._serve, name="garak-mim", daemon=True
        )
        self._thread.start()
        self._ready.wait()
        self._service = None
        self._armings: list[_MimArming] = []
        self._closed = False
        if url is None:
            try:
                try:
                    self._service = self._service_type(self._credential)
                    self._url = self._submit(self._service.start())
                except _CLEANUP_ERRORS:
                    self._close_after_failed_start()
                    raise CapabilityError(
                        "Mim injector cannot start its local service"
                    ) from None
            finally:
                if self._url is None and not self._closed:
                    self._close_after_failed_start()

    def _close_after_failed_start(self) -> None:
        """Release a failed start without masking known cleanup failures."""
        try:
            self.close()
        except (*_CLEANUP_ERRORS, ExceptionGroup) as error:
            logging.debug("Mim injector cleanup after failed start: %s", error)

    def _serve(self) -> None:
        """Run the one event loop until the owner stops it."""
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()
        self._loop.close()

    def _submit(self, coroutine):
        """Execute an operation on the run's loop, without leaking transport errors."""
        if threading.current_thread() is self._thread:
            coroutine.close()
            raise RuntimeError(
                "Mim injector run cannot be used from its own event loop"
            )
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result()

    def arm(
        self,
        collector,
        *,
        upstream_url: str,
        plan,
        policy=None,
        faces=("mcp",),
        host_tools=(),
    ) -> Arming:
        """Bind one MCP face and open its capture link after the upgrade succeeds."""
        if any(
            instruction.channel != "tool_output" for instruction in plan.instructions
        ):
            raise CapabilityError(
                "Mim injector lacks non-tool-output placement capability"
            )
        if faces != ("mcp",):
            raise CapabilityError("Mim injector lacks non-mcp face capability")
        if host_tools:
            raise CapabilityError("Mim injector lacks host tool capability")
        if self._closed:
            raise CapabilityError("Mim injector run is closed; cannot arm a binding")
        arming = self._submit(self._bind(collector, upstream_url, plan, policy))
        self._armings.append(arming)
        return arming

    async def _bind(self, collector, upstream_url, plan, policy):
        """Upgrade, bind and publish one protected tool block."""
        from websockets.asyncio.client import connect
        from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidURI

        wire = self.wire
        websocket_url = _bind_url(self._url)
        logger = logging.getLogger(f"{__name__}.websocket.{uuid.uuid4().hex}")
        redactor = _RedactSecrets(self._credential)
        logger.addFilter(redactor)
        try:
            socket = await asyncio.wait_for(
                connect(
                    websocket_url,
                    subprotocols=[wire.SUBPROTOCOL],
                    additional_headers={"Authorization": f"Bearer {self._credential}"},
                    proxy=None,
                    max_size=_FRAME_LIMIT,
                    logger=logger,
                ),
                self._bind_timeout,
            )
        except (InvalidURI, ValueError):
            raise CapabilityError(
                "Mim injector needs a valid http(s) service_url"
            ) from None
        except (InvalidHandshake, OSError, TimeoutError):
            raise CapabilityError(
                "Mim injector binding upgrade refused or unavailable"
            ) from None
        try:
            await socket.send(
                wire.encode_message(
                    "bind",
                    api=wire.API,
                    upstream_url=upstream_url,
                    plan=plan.to_data(),
                    policy=(
                        None
                        if policy is None
                        else policy.to_data() if hasattr(policy, "to_data") else policy
                    ),
                    faces=["mcp"],
                    host_tools=[],
                )
            )
            message = wire.decode_message(
                await asyncio.wait_for(socket.recv(), self._bind_timeout)
            )
            if message.op == "rejected":
                raise CapabilityError(
                    redactor.clean(
                        f"Mim injector bind refused {message['kind']}: {message['detail']}"
                    )
                )
            if message.op != "bound" or set(message.fields) != set(wire.BOUND_FIELDS):
                raise CapabilityError(
                    "Mim injector binding lacks a valid bound response"
                )
            tool_url = message["tool_url"]
            token = message["tool_token"]
            tools = message["tools"]
            try:
                _bind_url(tool_url)
            except CapabilityError:
                raise CapabilityError(
                    "Mim injector binding lacks a valid bound response"
                ) from None
            if (
                not isinstance(token, str)
                or not token
                or not isinstance(tools, (list, tuple))
                or not all(isinstance(tool, str) and tool for tool in tools)
            ):
                raise CapabilityError(
                    "Mim injector binding lacks a valid bound response"
                )
            block = ToolBlock(tool_url, token, tuple(tools))
            redactor.add(block.tool_token)
            name = f"mim:{uuid.uuid4()}"
            arming = _MimArming(self, socket, collector, block, name)
            arming._task = asyncio.create_task(arming._dispatch())
            return arming
        except (ConnectionClosed, TimeoutError, wire.LinkFault, KeyError, TypeError):
            raise CapabilityError(
                "Mim injector binding response unavailable or malformed"
            ) from None
        finally:
            if "arming" not in locals():
                await socket.close()

    async def _finish_service_shutdown(self) -> None:
        """Release a listener if the service's normal stop failed early."""
        server = getattr(self._service, "_server", None)
        serving = getattr(self._service, "_serving", None)
        listener = getattr(self._service, "_listener", None)
        if server is not None:
            server.should_exit = True
        try:
            if serving is not None:
                await asyncio.wait_for(serving, self._release_timeout)
        finally:
            if listener is not None:
                listener.close()

    def close(self) -> None:
        """Release resources, then report failures; unexpected errors remain unchanged."""
        if threading.current_thread() is self._thread:
            raise RuntimeError(
                "Mim injector run cannot be used from its own event loop"
            )
        if self._closed:
            return
        failures = []
        try:
            try:
                for arming in self._armings:
                    try:
                        arming.disarm()
                    except _CLEANUP_ERRORS as error:
                        failures.append(error)
            finally:
                if self._service is not None:
                    try:
                        try:
                            self._submit(self._service.stop())
                        except _CLEANUP_ERRORS as error:
                            failures.append(error)
                    finally:
                        try:
                            self._submit(self._finish_service_shutdown())
                        except _CLEANUP_ERRORS as cleanup_error:
                            failures.append(cleanup_error)
        finally:
            self._closed = True
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join()
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise ExceptionGroup("Mim injector cleanup failed", failures)

    def __reduce__(self):
        """Do not pass a live run across a worker boundary."""
        raise TypeError("a live injector run cannot be pickled")


class _MimArming(Arming):
    """One socket, capture link and dispatcher for a chain."""

    def __init__(self, run, socket, collector, block, name):
        self._run = run
        self._socket = socket
        self.tool_block = block
        self._link = collector.open_link(
            name, bound=lambda: socket.state.name == "OPEN"
        )
        self._channel = collector.channel("interceptor", self._link)
        self._next_id = 0
        self._task = None
        self._done = asyncio.Event()
        self._released = False
        self._outcomes = []
        self._disarmed = False

    async def _dispatch(self):
        """Read and answer sequential frames, pinging only while the reader runs."""
        wire = self._run.wire
        from websockets.exceptions import ConnectionClosed

        try:
            while True:
                try:
                    raw = await asyncio.wait_for(
                        socket_receive(self._socket), wire.HEARTBEAT_INTERVAL_SECONDS
                    )
                except TimeoutError:
                    await self._socket.send(wire.encode_message("ping"))
                    self._link.beat()
                    continue
                try:
                    message = wire.decode_message(raw)
                except wire.LinkFault as fault:
                    frame = wire.decode_frame(raw)
                    if frame.id != self._next_id:
                        raise wire.LinkFault(
                            f"frame {frame.id} expected {self._next_id}"
                        ) from fault
                    self._next_id += 1
                    answer = self._answer(frame)
                    await self._socket.send(wire.encode_answer(answer))
                    continue
                if message.op == "unbound" and set(message.fields) == set(
                    wire.UNBOUND_FIELDS
                ):
                    self._outcomes = message["injection_outcomes"]
                    self._released = True
                    self._link.release()
                    return
                raise wire.LinkFault(f"unexpected control message {message.op}")
        except (ConnectionClosed, OSError, RuntimeError, wire.LinkFault):
            self._link.lose("Mim injector binding stream lost or invalid")
        finally:
            self._done.set()
            await self._socket.close()

    def _answer(self, frame):
        """Apply a frame or return the collector's precise rejection."""
        wire = self._run.wire
        args = frame.args
        try:
            if frame.op == "start_call":
                self._channel.start_call(
                    args["call_id"],
                    function=args["function"],
                    args=args["args"],
                    disposition=args["disposition"],
                )
            elif frame.op == "finish_call_detailed":
                fields = {
                    name: args[name]
                    for name in ("function", "args", "result", "disposition")
                }
                fields["error"] = args.get("error")
                fields["blocked"] = fields["disposition"] in ("denied", "blocked")
                if "injection_outcomes" in args:
                    fields["structured_content"] = args.get("structured_content")
                    fields["injection_outcomes"] = [
                        _evidence_outcome(outcome)
                        for outcome in args["injection_outcomes"]
                    ]
                self._channel.finish_call(args["call_id"], **fields)
            elif frame.op == "report_capture_failure":
                self._channel.report_capture_failure(
                    args["call_id"], args["function"], args["phase"], args["reason"]
                )
            else:
                raise wire.LinkFault(f"unsupported op {frame.op}")
        except LateFactError as exc:
            return wire.Answer(
                frame.id, rejected=wire.Rejection(exc.kind, exc.detail, True)
            )
        except RejectedError as exc:
            return wire.Answer(frame.id, rejected=wire.Rejection(exc.kind, exc.detail))
        except (InvalidRequestError, KeyError, TypeError, ValueError) as exc:
            return wire.Answer(
                frame.id, rejected=wire.Rejection(wire.INVALID_REQUEST, str(exc))
            )
        return wire.Answer(frame.id, ok=None)

    def disarm(self) -> tuple[bool, list]:
        """Wait for the service's unbound acknowledgement, once."""
        if not self._disarmed:
            self._disarmed = True
            self._run._submit(self._release())
        return self._released, list(self._outcomes)

    async def _release(self):
        """Request release and mark an unacknowledged response as lost."""
        from websockets.exceptions import ConnectionClosed

        if not self._done.is_set():
            try:
                await self._socket.send(self._run.wire.encode_message("unbind"))
                await asyncio.wait_for(self._done.wait(), self._run._release_timeout)
            except (ConnectionClosed, OSError, RuntimeError, TimeoutError):
                self._link.lose("Mim injector release was not acknowledged")
                self._task.cancel()
        if not self._released:
            self._link.lose("Mim injector binding ended without release")
        await self._socket.close()


async def socket_receive(socket):
    """Receive one text frame; a non-text frame is a protocol fault."""
    raw = await socket.recv()
    if not isinstance(raw, str):
        raise RuntimeError("Mim injector received a non-text binding frame")
    return raw
