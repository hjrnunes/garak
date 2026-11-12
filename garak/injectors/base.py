"""Injector configuration and chain binding lifecycle."""

from dataclasses import dataclass, field

from garak.configurable import Configurable


@dataclass(frozen=True)
class ToolBlock:
    """The agent's tool endpoint, private bearer token and advertised tools."""

    tool_url: str
    tool_token: str = field(repr=False)
    tools: tuple[str, ...] = ()

    def __reduce__(self):
        """Keep a live tool credential in its owning process."""
        raise TypeError("a live tool block cannot be pickled")


class Arming:
    """One chain's armed binding, independent of the cached injector."""

    def disarm(self) -> tuple[bool, list]:
        """Acknowledge release and return its outcomes; a lost link is not released."""
        raise NotImplementedError

    def __reduce__(self):
        """Keep a live binding in its owning process."""
        raise TypeError("a live injector binding cannot be pickled")


class Injector(Configurable):
    """Configuration-only component; start a run before arming a chain."""

    def __init__(self, config_root=None):
        if isinstance(config_root, dict) and "plugins" in config_root:
            config_root = config_root["plugins"]
        self._load_config(config_root=config_root)

    def start_run(self):
        """Acquire run resources and return their owner."""
        raise NotImplementedError
