"""Lossless encoding of JSON call bodies and validation by the shared codecs."""

from garak.exception import GarakException

INSTALL_HINT = "Install the shared evidence codecs with pip install garak[midojo]"


class MissingSnapshotCodecError(GarakException):
    """The optional shared codecs are needed to compare and validate call bodies."""


def snapshot_codec():
    """Load the optional snapshot codec when evidence intake first needs it."""
    try:
        from midojo_spec import snapshot
    except ImportError as exc:
        raise MissingSnapshotCodecError(INSTALL_HINT) from exc
    return snapshot


def injection_codec():
    """Load the optional injection codec when an outcome's slot first needs it."""
    try:
        from midojo_spec import injection
    except ImportError as exc:
        raise MissingSnapshotCodecError(INSTALL_HINT) from exc
    return injection


def recorder_codec():
    """Load the optional recorder vocabulary when a capture reason is cut."""
    try:
        from midojo_spec import recorder
    except ImportError as exc:
        raise MissingSnapshotCodecError(INSTALL_HINT) from exc
    return recorder


def encode(value: object) -> str:
    """Encode JSON data without conflating numeric types, signs, or key order."""
    codec = snapshot_codec()
    return codec.dumps(codec.canonical(value))
