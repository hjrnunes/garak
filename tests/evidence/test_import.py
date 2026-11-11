"""Optional codec imports must not prevent ordinary module discovery."""

import subprocess
import sys


def test_import_without_optional_dependency():
    """Import the collector without its extra, but fail clearly when encoding."""
    script = """
import sys
sys.modules['midojo_spec'] = None
import garak.evidence.codec
import garak.evidence.records
import garak.evidence.ledger
from garak.exception import GarakException
try:
    garak.evidence.codec.encode({})
except GarakException as error:
    assert 'pip install garak[midojo-spec]' in str(error)
else:
    raise AssertionError('missing extra was not reported')
"""
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, f"optional import failed: {done.stderr}"
