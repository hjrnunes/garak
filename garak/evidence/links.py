"""Capture links: the liveness of the seam a channel records through.

A link names the binding between an interceptor and the collector. Its state
is read from the binding itself at the seal: the beat clock, the loss flag and
the peer's ``bound()`` answer, never a dispatcher's own account of the loss.
That is the rule the stalled-dispatcher defect broke: a dispatcher that never
reports a loss must not hold the seal open, and a service that already expired
the binding must not let the seal claim a complete capture.

Links are diagnostics, not evidence: a lost or stalled link marks
``function_calls`` incomplete in the snapshot's evidence status, and the
collector's ``links`` summary never enters the digest.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

LinkState = str
LIVE = "live"
RELEASED = "released"
LOST = "lost"
STALLED = "stalled"


@dataclass
class CaptureLink:
    """One capture seam's liveness, written by the frames that cross it.

    ``beat()`` records any frame or ping; ``release()`` an acknowledged
    release; ``lose(reason)`` the stream ending without one. ``state(now)``
    reads the binding. A link's first end stands: a release after a loss, or a
    loss after a release, is ignored. An expired ``bound()`` is a loss while
    the link was not released, and a beat older than ``stall_after`` is a
    stall. A link with no ``bound()`` predicate is bound for as long as it was
    not released.

    ``bound()`` must answer without blocking, because the seal calls it. A
    predicate that raises ``OSError`` or ``RuntimeError`` cannot confirm the
    binding, so the link reads as lost; any other error is a defect in the
    predicate and propagates.
    """

    name: str
    stall_after: float | None = None
    bound: Callable[[], bool] | None = None
    _last_beat: float = 0.0
    _released: bool = False
    _lost: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def beat(self, at: float | None = None) -> None:
        """Record a frame or a ping crossing the link; an older beat never replaces a newer one."""
        with self._lock:
            self._last_beat = max(
                self._last_beat, at if at is not None else _monotonic()
            )

    def release(self) -> None:
        """Record the peer's acknowledged release: the link is done and clean.

        A release after a loss is ignored: a stream that already ended without
        one cannot be acknowledged afterwards.
        """
        with self._lock:
            if self._lost is None:
                self._released = True

    def lose(self, reason: str) -> None:
        """Record the stream ending without an acknowledged release.

        A loss after a release is ignored: the release already ended the link.
        """
        with self._lock:
            if not self._released:
                self._lost = reason

    def state(self, now: float | None = None) -> LinkState:
        """Read the link's state from the binding, at ``now`` (the beat clock's)."""
        with self._lock:
            if self._released:
                return RELEASED
            if self._lost is not None:
                return LOST
            if self.bound is not None and not self._is_bound():
                return LOST
            if self.stall_after is not None:
                moment = now if now is not None else _monotonic()
                if moment - self._last_beat > self.stall_after:
                    return STALLED
            return LIVE

    def _is_bound(self) -> bool:
        """The ``bound()`` answer; a predicate that cannot reach its peer says no."""
        try:
            return bool(self.bound())
        except (OSError, RuntimeError):
            return False

    def summary(self) -> dict[str, Any]:
        """A diagnostics-only description of the link; never part of the digest."""
        with self._lock:
            return {
                "name": self.name,
                "released": self._released,
                "lost": self._lost,
                "stall_after": self.stall_after,
            }

    def __reduce__(self) -> Any:
        """Refuse pickling: a live link holds a lock and shared timing state."""
        raise TypeError("a capture link cannot be pickled")

    def __copy__(self) -> CaptureLink:
        """Refuse copying: a live link holds a lock and shared timing state."""
        raise TypeError("a capture link cannot be copied")

    def __deepcopy__(self, memo: dict[int, Any]) -> CaptureLink:
        """Refuse copying: a live link holds a lock and shared timing state."""
        raise TypeError("a capture link cannot be copied")


def _monotonic() -> float:
    import time

    return time.monotonic()
