"""Component discovery does not load optional interception packages."""

import subprocess
import sys


def test_environment_import_stays_light():
    """Listing an environment leaves optional service modules unloaded."""
    code = (
        "import sys; from garak import _plugins; "
        "_plugins.load_plugin('environments.unmanaged'); "
        "assert not {'midojo_spec', 'midojo_mim', 'fastmcp', 'uvicorn'} "
        "& set(sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr.decode()
