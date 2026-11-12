"""An environment whose operator declares a fresh target for every chain."""

from garak.environments.base import Environment, EnvironmentScope
from garak.exception import CapabilityError

DEFAULT_CLASS = "Disposable"


class _DisposableScope(EnvironmentScope):
    """A chain whose target the operator replaces, and garak neither owns nor watches."""

    def __init__(self):
        self._closed = False

    def observe(self) -> dict:
        """Return no observations: this component cannot observe the target."""
        return {}

    def close(self) -> None:
        """Finish the chain; the operator disposes of its target."""
        self._closed = True


class Disposable(Environment):
    """Declare that every chain meets a target that no earlier chain has touched.

    This is your declaration, as the operator, and garak never verifies it. You
    make it true by giving each chain a new target, for example a new container
    or a freshly seeded service, outside garak. With the declaration a probe may
    run repeated chains and the baseline of a paired run. If the declaration is
    false, a repeat or a baseline meets the state an earlier chain left, and a
    comparison drawn from them is not valid.

    The target is replaced, not reset, so the environment has no reset
    capability and refuses a reset. It records no observations.
    """

    isolation = "disposable"
    capabilities = frozenset()

    def __init__(self, config_root=None):
        super().__init__(config_root=config_root)
        if self.isolation != "disposable" or self.observations != ():
            raise CapabilityError(
                "Disposable environment declares isolation and no observations, "
                "and configuration cannot change that"
            )

    def open(self, chain, *, reset: bool = False) -> EnvironmentScope:
        """Open one chain; refuse a reset, which a replaced target does not need."""
        if reset:
            raise CapabilityError(
                "Disposable environment lacks target reset capability"
            )
        return _DisposableScope()
