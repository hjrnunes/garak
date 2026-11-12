"""Unmanaged chains have neither reset nor repeated-attempt isolation."""

import pickle

import pytest

from garak import _plugins
from garak.environments.unmanaged import Unmanaged
from garak.exception import CapabilityError


def test_load_and_refuse_isolation():
    """Load the component through the plugin provider and refuse unsupported work."""
    environment = _plugins.load_plugin("environments.unmanaged")
    assert isinstance(environment, Unmanaged), "component loads"
    with pytest.raises(CapabilityError, match="Unmanaged.*reset"):
        environment.open("chain", reset=True)
    with pytest.raises(CapabilityError, match="Unmanaged.*isolation"):
        environment.check_attempt(repeated=True)


def test_nested_plugin_root_refuses_false_isolation():
    """A nested configuration cannot grant an unmanaged target isolation."""
    root = {"plugins": {"environments": {"unmanaged": {"isolation": "disposable"}}}}
    assert (
        _plugins.load_plugin(
            "environments.unmanaged", config_root=root, break_on_fail=False
        )
        is False
    ), "nested isolation claim is refused"


def test_loader_preserves_isolation_capability_error():
    """The loader reports the component and unsupported capability directly."""
    root = {"environments": {"unmanaged": {"isolation": "disposable"}}}
    with pytest.raises(CapabilityError, match="Unmanaged.*isolation"):
        _plugins.load_plugin("environments.unmanaged", config_root=root)


def test_scope_lifecycle():
    """An unmanaged scope records no observations and cannot leave its process."""
    scope = Unmanaged().open("chain")
    assert scope.observe() == {}, "no observations"
    scope.close()
    scope.close()
    with pytest.raises(TypeError, match="pickled"):
        pickle.dumps(scope)
