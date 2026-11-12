"""An externally owned target with no isolation or runtime observations."""

from garak.environments.base import Environment, EnvironmentScope
from garak.exception import CapabilityError

DEFAULT_CLASS = "Unmanaged"


class _UnmanagedScope(EnvironmentScope):
    """A chain whose target belongs entirely to its caller."""

    def __init__(self):
        self._closed = False

    def observe(self) -> dict:
        """Return no observations: this component cannot observe the target."""
        return {}

    def close(self) -> None:
        """Finish the chain without resetting its externally owned target."""
        self._closed = True


class Unmanaged(Environment):
    """Leave a target's setup, disposal and observation to its caller."""

    capabilities = frozenset()

    def __init__(self, config_root=None):
        super().__init__(config_root=config_root)
        if self.isolation is not None or self.observations != ():
            raise CapabilityError(
                "Unmanaged environment lacks declared isolation and observation capabilities"
            )

    def open(self, chain, *, reset: bool = False) -> EnvironmentScope:
        """Open one chain; refuse a reset that the target cannot perform."""
        if reset:
            raise CapabilityError("Unmanaged environment lacks target reset capability")
        return _UnmanagedScope()
