"""A disposable environment is the operator's declaration of a fresh target per chain."""

import pickle

import pytest

from garak import _plugins
from garak.environments.disposable import Disposable
from garak.exception import CapabilityError


def test_load_and_allow_repeated_attempts():
    """The component loads through the plugin provider and declares isolation."""
    environment = _plugins.load_plugin("environments.disposable")

    assert isinstance(environment, Disposable), "component loads"
    assert environment.isolation == "disposable", "isolation is declared"
    environment.check_attempt(repeated=True)
    environment.check_attempt()


def test_it_cannot_reset_a_target():
    """A disposable target is replaced, not reset, so a reset is refused."""
    environment = Disposable()

    assert "reset" not in environment.capabilities, "no reset capability"
    with pytest.raises(CapabilityError, match="Disposable.*reset"):
        environment.open("chain", reset=True)


def test_scope_lifecycle():
    """A scope records no observations and cannot leave its process."""
    scope = Disposable().open("chain")

    assert scope.observe() == {}, "no observations"
    scope.close()
    scope.close()
    with pytest.raises(TypeError, match="pickled"):
        pickle.dumps(scope)


@pytest.mark.parametrize(
    "setting", [{"isolation": "shared"}, {"isolation": None}, {"observations": ("a",)}]
)
def test_config_cannot_change_what_it_declares(setting):
    """Configuration cannot turn the declaration into another claim."""
    root = {"environments": {"disposable": setting}}

    with pytest.raises(CapabilityError, match="Disposable"):
        Disposable(config_root=root)
