"""Environment lifecycle and declared target isolation."""

from garak.configurable import Configurable
from garak.exception import CapabilityError


class EnvironmentScope:
    """The observations and cleanup owned by one chain, not the shared component."""

    def observe(self) -> dict:
        """Collect available observations while the chain is still open."""
        raise NotImplementedError

    def close(self) -> None:
        """Release this chain's resources once."""
        raise NotImplementedError

    def __reduce__(self):
        """Keep live scopes in their owning process."""
        raise TypeError("a live environment scope cannot be pickled")


class Environment(Configurable):
    """Configure an environment without holding chain resources.

    A hook-driven environment reads setup and teardown hooks from the target
    configuration. The consumer passes bindings during setup; no reset runs
    after setup.
    """

    isolation: str | None = None
    observations: tuple[str, ...] = ()

    def __init__(self, config_root=None):
        if isinstance(config_root, dict) and "plugins" in config_root:
            config_root = config_root["plugins"]
        self._load_config(config_root=config_root)

    def check_attempt(self, *, repeated: bool = False) -> None:
        """Refuse a repeat when target isolation is not declared."""
        if repeated and self.isolation is None:
            raise CapabilityError(
                f"{type(self).__name__} environment lacks target isolation for repeated attempts"
            )

    def open(self, chain, *, reset: bool = False) -> EnvironmentScope:
        """Open an isolated chain, or refuse an unsupported reset."""
        raise NotImplementedError
