# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One POST to a judge service over ``http.client``, and the URL it is sent to.

The request goes only to the URL it was given: ``http.client`` reads no proxy
or ``.netrc`` from the environment and follows no redirect, so a bearer token
cannot leave for another host. The URL's user info is dropped, not sent (a
client that turns it into a ``Basic`` credential would overwrite the bearer
token). A ``gzip`` or ``deflate`` answer is decoded as ``httpx`` does, and
neither the answer as sent nor what it decodes to is more than
:data:`MAX_ANSWER_BYTES` (a longer one is :class:`AnswerTooLarge`). An answer in
any other content coding (``br`` and ``zstd`` among them) is not read at all,
whatever its bytes are: :class:`UnreadableCoding`. ``httpx`` decodes ``br`` and
``zstd`` when their packages are installed, so there the same answer can be a
verdict; here it never is. A timeout bounds each socket
operation (connecting, sending, each read), not the call as a whole. HTTPS
verifies against the trust store Python's ``ssl`` module loads by default.

:func:`parse_url` cleans a URL as ``httpx`` does (lower-case scheme and host,
default port dropped, path and query percent-escaped, dot segments removed,
internationalized host names in their ASCII form by the ``idna`` package
``httpx`` uses), so the endpoint a record names is the same text the earlier
clients gave. A URL it cannot read is None, and so is one ``httpx`` reads but
nothing can be sent to (a port that is not ASCII digits up to 65535).
"""

from __future__ import annotations

import http.client
import ipaddress
import re
import string
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import NamedTuple

from garak import __version__

MAX_ANSWER_BYTES = 8 * 1024 * 1024
USER_AGENT = f"garak/{__version__}"
TRANSPORT_ERRORS = (http.client.HTTPException, OSError, ValueError)

_MAX_URL_LENGTH = 65536
_DEFAULT_PORTS = {"ftp": 21, "http": 80, "https": 443, "ws": 80, "wss": 443}
_UNRESERVED = string.ascii_letters + string.digits + "-._~"
_URL = re.compile(
    r"(?:(?P<scheme>[a-zA-Z][a-zA-Z0-9+.-]*):)?"
    r"(?://(?P<authority>[^/?#]*))?"
    r"(?P<path>[^?#]*)"
    r"(?:\?(?P<query>[^#]*))?"
    r"(?:#(?P<fragment>.*))?"
)
_AUTHORITY = re.compile(r"(?:(?P<userinfo>.*)@)?(?P<host>\[.*\]|[^:@]*):?(?P<port>.*)?")
_IPV4_STYLE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$")


def _printable_ascii_except(*excluded: int) -> str:
    return "".join(chr(code) for code in range(0x20, 0x7F) if code not in excluded)


# What the URL standard leaves unescaped in a query, and in a path (which also
# escapes ` { and }). Each keeps "%": an escape, or a "%" with none, is as written.
_QUERY_SAFE = _printable_ascii_except(0x20, 0x22, 0x23, 0x3C, 0x3E)
_PATH_SAFE = _printable_ascii_except(0x20, 0x22, 0x23, 0x3C, 0x3E, 0x60, 0x7B, 0x7D)
_HOST_SAFE = "!$&'()*+,;=" + '"`{}%|\\'


class UnsupportedProtocol(ValueError):
    """The URL's scheme is not ``http`` or ``https``."""


class DecodingError(ValueError):
    """The answer names a content coding (``gzip``, ``deflate``) its bytes are not valid in."""


class UnreadableCoding(Exception):  # noqa: N818 - named for what it says
    """The answer is in a content coding this module does not decode; ``status`` is the one it came with."""

    def __init__(self, status: int) -> None:
        self.status = status
        super().__init__("the answer is in a content coding that is not decoded")


class AnswerTooLarge(Exception):  # noqa: N818 - named for what it says
    """The answer is longer than :data:`MAX_ANSWER_BYTES`; ``status`` is the one it came with."""

    def __init__(self, status: int) -> None:
        self.status = status
        super().__init__(f"the answer is longer than {MAX_ANSWER_BYTES} bytes")


class Answer(NamedTuple):
    """What a service answered: its status, and the body when the status is a success (else empty)."""

    status: int
    body: bytes


@dataclass(frozen=True)
class Endpoint:
    """A URL read for sending: where to connect, what to ask for, and the text a record names it by."""

    scheme: str
    host: str
    port: int | None
    path: str
    query: str | None

    @property
    def text(self) -> str:
        """The URL without its user info, query or fragment, cleaned."""
        netloc = f"[{self.host}]" if ":" in self.host else self.host
        if self.port is not None:
            netloc += f":{self.port}"
        return f"{self.scheme}://{netloc}{self.path}"

    @property
    def target(self) -> str:
        """The request target: the path, then the query when there is one."""
        query = "" if self.query is None else f"?{self.query}"
        return f"{self.path or '/'}{query}"


def parse_url(url: str) -> Endpoint | None:
    """The endpoint ``url`` names, or None when it is not an absolute URL with a host and a valid port.

    A fragment and user info are read and dropped. A scheme other than
    ``http`` and ``https`` is read (it fails when sent: :class:`UnsupportedProtocol`).
    """
    try:
        return _parse(url)
    except ValueError:  # a bad address, host name or port, or text UTF-8 cannot hold
        return None


def _parse(url: str) -> Endpoint:
    if len(url) > _MAX_URL_LENGTH or any(
        char.isascii() and not char.isprintable() for char in url
    ):
        raise ValueError("not a URL")
    found = _URL.match(url)
    assert found is not None  # every part is optional
    scheme = (found["scheme"] or "").lower()
    if not scheme or found["authority"] is None:
        raise ValueError("not an absolute URL")
    authority = _AUTHORITY.match(found["authority"])
    assert authority is not None
    host = _encode_host(authority["host"] or "")
    if not host:
        raise ValueError("no host")
    path = found["path"]  # the authority ends at a /, so a path begins with one
    query = found["query"]
    return Endpoint(
        scheme=scheme,
        host=host,
        port=_port(authority["port"], scheme),
        path=_quote(_without_dot_segments(path), _PATH_SAFE),
        query=None if query is None else _quote(query, _QUERY_SAFE),
    )


def _port(port: str | None, scheme: str) -> int | None:
    if not port:
        return None
    if not (port.isascii() and port.isdigit()) or int(port) > 65535:
        raise ValueError("not a port")
    return None if int(port) == _DEFAULT_PORTS.get(scheme) else int(port)


def _encode_host(host: str) -> str:
    if not host:
        return ""
    if _IPV4_STYLE.match(host):
        ipaddress.IPv4Address(host)
        return host
    if host.startswith("[") and host.endswith("]"):
        ipaddress.IPv6Address(host[1:-1])
        return host[1:-1]
    if host.isascii():
        return _quote(host.lower(), _HOST_SAFE)
    try:
        import idna  # a dependency of httpx, which the OpenAI client needs
    except ImportError:
        raise ValueError("no IDNA codec to read an internationalized host") from None
    return idna.encode(host.lower()).decode("ascii")  # an IDNAError is a ValueError


def _without_dot_segments(path: str) -> str:
    components = path.split("/")
    if "." not in components and ".." not in components:
        return path
    kept: list[str] = []
    for component in components:
        if component == ".":
            continue
        if component == "..":
            if len(kept) > 1:  # the first component is the root's empty one
                kept.pop()
        else:
            kept.append(component)
    return "/".join(kept)


def _quote(text: str, safe: str) -> str:
    """``text`` percent-escaped but for unreserved and ``safe`` characters."""
    unescaped = _UNRESERVED + safe
    return "".join(
        char if char in unescaped else "".join(f"%{b:02X}" for b in char.encode())
        for char in text
    )


def post(
    endpoint: Endpoint,
    body: bytes,
    headers: Mapping[str, str],
    *,
    timeout: float,
) -> Answer:
    """POST ``body`` to ``endpoint`` and read the answer.

    The body is read only when the status is a success, and the connection
    is closed whatever happens. An :class:`UnsupportedProtocol`,
    :class:`UnreadableCoding` (raised before any of the body is read),
    :class:`AnswerTooLarge`, or any error of ``TRANSPORT_ERRORS`` (a timeout
    is a ``TimeoutError``, a body its coding does not hold a
    :class:`DecodingError`) ends the call.
    """
    if endpoint.scheme not in ("http", "https"):
        raise UnsupportedProtocol(f"unknown protocol {endpoint.scheme!r}")
    secure = endpoint.scheme == "https"
    # The port is always given: with none, http.client reads an IPv6 host's last group as one.
    # An explicit 0 is a port to connect to (it fails), never the default.
    port = endpoint.port if endpoint.port is not None else (443 if secure else 80)
    connection_class = (
        http.client.HTTPSConnection if secure else http.client.HTTPConnection
    )
    connection = connection_class(endpoint.host, port, timeout=timeout)
    try:
        connection.request(
            "POST",
            endpoint.target,
            body=body,
            headers={"User-Agent": USER_AGENT, **headers},
        )
        response = connection.getresponse()
        status = response.status
        if not 200 <= status < 300:
            return Answer(status, b"")
        stages = _stages(response.getheader("Content-Encoding"), status)
        chunks: list[bytes] = []
        sent = decoded = 0
        # One read is one chunk of a chunked answer, as the chunks httpx decodes one by one.
        while chunk := response.read1(65536):
            sent += len(chunk)
            if sent > MAX_ANSWER_BYTES:
                raise AnswerTooLarge(status)
            for number, stage in enumerate(stages, start=1):
                last = number == len(stages)
                chunk = stage.decode(
                    chunk, MAX_ANSWER_BYTES - decoded if last else None
                )
            decoded += len(chunk)
            chunks.append(chunk)
        return Answer(status, b"".join(chunks))
    finally:
        connection.close()


class _Stage:
    """One content coding, undone a chunk at a time as ``httpx``'s decoders undo it.

    A ``deflate`` answer is a zlib stream, or, as some servers send it, a raw
    one: when the first chunk is not zlib, the answer is read as raw, and a
    later chunk that is not valid in either is a :class:`DecodingError`. What a
    chunk decodes to is no more than ``room`` bytes (:data:`MAX_ANSWER_BYTES`
    for a coding another undoes after), so a small answer cannot expand without
    limit.
    """

    def __init__(self, coding: str, status: int) -> None:
        self.deflate = coding == "deflate"
        self.status = status
        self.first = True
        self.inflater = zlib.decompressobj(
            zlib.MAX_WBITS if self.deflate else zlib.MAX_WBITS | 16
        )

    def decode(self, data: bytes, room: int | None = None) -> bytes:
        room = MAX_ANSWER_BYTES if room is None else room
        first, self.first = self.first, False
        try:
            inflated = self.inflater.decompress(data, room + 1)
        except zlib.error:
            if not (self.deflate and first):
                raise DecodingError(
                    "the answer is not valid in the coding it names"
                ) from None
            self.inflater = zlib.decompressobj(-zlib.MAX_WBITS)
            return self.decode(data, room)
        if len(inflated) > room:
            raise AnswerTooLarge(self.status)
        return inflated


def _stages(header: str | None, status: int) -> list[_Stage]:
    """The stages that undo the content codings ``header`` names, the one applied last first.

    A coding other than ``gzip``, ``deflate`` and ``identity`` is
    :class:`UnreadableCoding`, even beside codings that are read, and even when
    the bytes would parse as they are: an answer this does not decode is never
    taken for a verdict. ``httpx`` reads ``br`` and ``zstd`` too, when their
    packages are installed.
    """
    codings = [part.strip().lower() for part in (header or "").split(",")]
    codings = [c for c in codings if c not in ("", "identity")]
    if any(c not in ("gzip", "deflate") for c in codings):
        raise UnreadableCoding(status)
    return [_Stage(c, status) for c in reversed(codings)]
