# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test the one POST a judge request makes, and the URL it is sent to."""

import gzip
import http.client
import json
import socket
import sys
import types
import zlib

import pytest

from garak.detectors import _judge_http as transport
from garak.detectors._judge_http import Answer, Endpoint, parse_url

BODY = json.dumps({"id": "sha256:" + "ab" * 32}).encode()
HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}
LONGEST_PATH = "a" * (65536 - len("http://h/"))  # the URL is then 65536 characters


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def endpoint_of(url: str) -> Endpoint:
    found = parse_url(url)
    assert found is not None, url
    return found


# --- the URL ------------------------------------------------------------------------------

# Each URL with the endpoint a record names it by, and the request target: what httpx made of it.
URLS = [
    ("http://127.0.0.1:8080/judge", "http://127.0.0.1:8080/judge", "/judge"),
    ("HTTP://Example.COM:80/Judge", "http://example.com/Judge", "/Judge"),
    ("https://example.com:443/a", "https://example.com/a", "/a"),
    ("https://host:80/", "https://host:80/", "/"),
    ("http://host:443/", "http://host:443/", "/"),
    ("http://host:0080/", "http://host/", "/"),
    ("http://h:/", "http://h/", "/"),
    ("ftp://h:21/", "ftp://h/", "/"),
    ("ws://h:80/x", "ws://h/x", "/x"),
    ("http://host", "http://host", "/"),
    ("http://host?x=1", "http://host", "/?x=1"),
    ("http://host/?", "http://host/", "/?"),
    ("http://host/p#", "http://host/p", "/p"),
    ("http://user:pw@host:81/p?q#f", "http://host:81/p", "/p?q"),
    ("http://a@b@c/", "http://c/", "/"),
    (
        "https://example.com/a b/ü?x=1 2&y=ü#frag",
        "https://example.com/a%20b/%C3%BC",
        "/a%20b/%C3%BC?x=1%202&y=%C3%BC",
    ),
    ("http://host/a%20b/%zz", "http://host/a%20b/%zz", "/a%20b/%zz"),
    ("http://host/%7efoo", "http://host/%7efoo", "/%7efoo"),
    ("http://host/{a}", "http://host/%7Ba%7D", "/%7Ba%7D"),
    ("http://host/a`b?c`d", "http://host/a%60b", "/a%60b?c`d"),
    ("http://host/p;x=1", "http://host/p;x=1", "/p;x=1"),
    ("http://host/p|q", "http://host/p|q", "/p|q"),
    ("http://host/ünï", "http://host/%C3%BCn%C3%AF", "/%C3%BCn%C3%AF"),
    ('http://host/p?a=b c&d="e"', "http://host/p", "/p?a=b%20c&d=%22e%22"),
    ("http://host/p?a=%41&b=%zz", "http://host/p", "/p?a=%41&b=%zz"),
    ("http://host/a/../b/./c", "http://host/b/c", "/b/c"),
    ("http://host/a/../../b", "http://host/b", "/b"),
    ("http://host/.", "http://host", "/"),
    ("http://host/a/..", "http://host", "/"),
    ("http://[::1]:9/p", "http://[::1]:9/p", "/p"),
    ("http://[::1]/", "http://[::1]/", "/"),
    ("http://[fe80::1%25en0]/", "http://[fe80::1%25en0]/", "/"),
    ("http://bücher.example/ü", "http://xn--bcher-kva.example/%C3%BC", "/%C3%BC"),
    ("http://HOST.Example./x", "http://host.example./x", "/x"),
    ("http://1.2.3/", "http://1.2.3/", "/"),
    ("http://ho st/", "http://ho%20st/", "/"),
    ("http://host\\p/", "http://host\\p/", "/"),
    ("http://-bad-.example/", "http://-bad-.example/", "/"),
    ("http://BÜCHER.Example/", "http://xn--bcher-kva.example/", "/"),
    ("http://faß.example/", "http://xn--fa-hia.example/", "/"),
    ("http://ςa.example/", "http://xn--a-xmb.example/", "/"),
    ("http://a。b/", "http://a.b/", "/"),
    ("http://host:65535/", "http://host:65535/", "/"),
    ("http://host/p?a>b<c", "http://host/p", "/p?a%3Eb%3Cc"),
    ('http://host/p?a"b c', "http://host/p", "/p?a%22b%20c"),
    ("http://host/p?a?b/c", "http://host/p", "/p?a?b/c"),
    ("http://host/100%", "http://host/100%", "/100%"),
    ("http://host/?q=100%", "http://host/", "/?q=100%"),
    ("http://h/" + LONGEST_PATH, "http://h/" + LONGEST_PATH, "/" + LONGEST_PATH),
]


@pytest.mark.parametrize(("url", "text", "target"), URLS)
def test_a_url_is_cleaned_as_httpx_cleans_it(url, text, target):
    found = endpoint_of(url)
    assert (found.text, found.target) == (text, target)


@pytest.mark.parametrize(("url", "text", "target"), URLS)
def test_the_cleaned_endpoint_is_what_httpx_gives(url, text, target):
    httpx = pytest.importorskip("httpx")
    parsed = httpx.URL(url)
    cleaned = parsed.copy_with(username=None, password=None, query=None, fragment=None)
    assert str(cleaned) == text, "the record names the endpoint as httpx cleaned it"
    assert parsed.raw_path.decode() == target, "and asks for what httpx asked for"


@pytest.mark.parametrize(
    "url",
    [
        "not a url",
        "/p",
        "//host/p",
        "http:///p",
        "http://",
        "http:/p",
        "mailto:a@b",
        "  http://host/  ",
        "",
        "http://h:abc/",
        "http://h:99999/",
        "http://h: 80/",
        "http://h:+80/",
        "http://h:8_0/",
        "http://host/a\tb",
        "http://host/\x00",
        "http://host/a\nb",
        "http://host/\x7f",
        "http://host/\ud800",
        "http://h\ud800st/",
        "http://[::1/",
        "http://[::g]/",
        "http://[1.2.3.4]/",
        "http://1.2.3.999/",
        "http://host/" + "a" * 70000,
        "http://h/" + LONGEST_PATH + "a",
        "http://h:65536/",
        "http://\u00e9" + "a" * 70 + ".example/",
        "http://a\u200db.example/",
        "http://\uff45xample.com/",
        "http://host:80path",
    ],
)
def test_a_url_that_is_not_an_absolute_url_with_a_host_is_none(url):
    assert parse_url(url) is None


def test_an_internationalized_host_is_unreadable_without_the_idna_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "idna", None)  # an import of it then fails
    assert parse_url("http://bücher.example/") is None
    assert parse_url("http://example.com/") is not None


def test_a_url_httpx_reads_but_nothing_can_be_sent_to_is_none():
    httpx = pytest.importorskip("httpx")
    for url in ("not a url", "http:///p", "http://h:99999/", "http://h:\u0663/"):
        assert parse_url(url) is None
        httpx.URL(url)  # httpx reads each one, and fails only when it connects


def test_an_url_httpx_refuses_is_none():
    httpx = pytest.importorskip("httpx")
    for url in (
        "http://h:abc/",
        "http://host/a\tb",
        "http://host/\x00",
        "http://a\u200db.example/",
        "http://\uff45xample.com/",
    ):
        with pytest.raises(httpx.InvalidURL):
            httpx.URL(url)
        assert parse_url(url) is None


# --- the POST --------------------------------------------------------------------------------


def test_an_answer_is_its_status_and_its_body(fake_judge):
    answer = transport.post(
        endpoint_of(fake_judge.url("/supported")), BODY, HEADERS, timeout=5
    )
    assert answer.status == 200
    assert json.loads(answer.body)["verdict"] == "supported"


def test_the_request_has_the_body_the_headers_and_a_user_agent(fake_judge):
    transport.post(
        endpoint_of(fake_judge.url("/supported?a=1&b=%202")),
        BODY,
        {**HEADERS, "Authorization": "Bearer abc"},
        timeout=5,
    )
    [received] = fake_judge.requests
    assert (received.path, received.query) == ("/supported", "a=1&b=%202")
    assert received.body == json.loads(BODY)
    assert received.headers["content-type"] == "application/json"
    assert received.headers["accept"] == "application/json"
    assert received.headers["authorization"] == "Bearer abc"
    assert received.headers["user-agent"] == transport.USER_AGENT
    assert received.headers["user-agent"].startswith("garak/")
    assert received.headers["host"] == fake_judge.host_port
    assert received.headers["content-length"] == str(len(BODY))


def test_an_answer_that_is_not_a_success_has_no_body_and_is_not_read(fake_judge):
    answer = transport.post(
        endpoint_of(fake_judge.url("/500")), BODY, HEADERS, timeout=5
    )
    assert answer == Answer(500, b"")
    big = transport.post(
        endpoint_of(fake_judge.url("/big-failure")), BODY, HEADERS, timeout=5
    )
    assert big == Answer(500, b""), "a failure's body is never held, however long"


def test_any_success_status_has_its_body(fake_judge):
    answer = transport.post(
        endpoint_of(fake_judge.url("/created")), BODY, HEADERS, timeout=5
    )
    assert answer.status == 201 and json.loads(answer.body)["verdict"] == "supported"


@pytest.mark.parametrize(
    "path",
    [
        "/gzip",
        "/gzip-capitalised",
        "/deflate",
        "/deflate-raw",
        "/deflate-then-gzip",
        "/deflate-split",
        "/gzip-split",
        "/identity-coding",
    ],
)
def test_an_answer_in_a_content_coding_httpx_reads_is_decoded(fake_judge, path):
    answer = transport.post(endpoint_of(fake_judge.url(path)), BODY, HEADERS, timeout=5)
    assert answer.status == 200
    assert json.loads(answer.body)["verdict"] == "supported"


@pytest.mark.parametrize(
    "path", ["/gzip-not-gzip", "/deflate-not-deflate", "/deflate-raw-split"]
)
def test_an_answer_that_is_not_in_the_coding_it_names_is_a_decoding_error(
    fake_judge, path
):
    with pytest.raises(transport.DecodingError) as raised:
        transport.post(endpoint_of(fake_judge.url(path)), BODY, HEADERS, timeout=5)
    assert type(raised.value).__name__ == "DecodingError", "the name a record shows"
    assert issubclass(transport.DecodingError, transport.TRANSPORT_ERRORS)


def test_a_truncated_coded_answer_gives_what_was_decoded_as_httpx_does(fake_judge):
    answer = transport.post(
        endpoint_of(fake_judge.url("/gzip-truncated")), BODY, HEADERS, timeout=5
    )
    assert answer.status == 200 and answer.body
    with pytest.raises(ValueError):
        json.loads(answer.body)


def test_a_coded_answer_is_decoded_no_further_than_one_byte_past_the_limit(monkeypatch):
    asked_for = []
    real = zlib.decompressobj

    class Spy:
        def __init__(self, wbits):
            self.inflater = real(wbits)

        def decompress(self, data, max_length=0):
            asked_for.append(max_length)
            return self.inflater.decompress(data, max_length)

    monkeypatch.setattr(transport.zlib, "decompressobj", Spy)
    assert transport._Stage("gzip", 200).decode(gzip.compress(b"{}")) == b"{}"
    assert asked_for == [transport.MAX_ANSWER_BYTES + 1], "the decoder is bounded"


PLAIN = b'{"verdict": "supported"}'
RAW = zlib.compressobj(wbits=-zlib.MAX_WBITS)
RAW_DEFLATE = RAW.compress(PLAIN) + RAW.flush()


def test_a_deflate_answer_is_read_as_raw_only_when_its_first_chunk_is_not_zlib():
    # As httpx's decoder does: the first chunk decides, and a later chunk that is invalid is an error.
    whole = transport._Stage("deflate", 200)
    assert whole.decode(RAW_DEFLATE) == PLAIN

    one_byte_first = transport._Stage("deflate", 200)
    one_byte_first.decode(RAW_DEFLATE[:1])  # too short to show it is not zlib
    with pytest.raises(transport.DecodingError):
        one_byte_first.decode(RAW_DEFLATE[1:])

    raw_after_first = transport._Stage("deflate", 200)
    first = raw_after_first.decode(RAW_DEFLATE[:3])  # not zlib: raw from here on
    assert first + raw_after_first.decode(RAW_DEFLATE[3:]) == PLAIN


def test_codings_are_undone_last_applied_first_a_chunk_at_a_time():
    stages = transport._stages("deflate, gzip", 200)
    assert [stage.deflate for stage in stages] == [False, True]
    wire = gzip.compress(zlib.compress(b"x" * 1000))
    decoded = b""
    for start in range(0, len(wire), 7):
        chunk = wire[start : start + 7]
        for stage in stages:
            chunk = stage.decode(chunk)
        decoded += chunk
    assert decoded == b"x" * 1000


@pytest.mark.parametrize(
    "coding", ["br", "zstd", "BR", " Zstd ", "compress", "x-gzip", "x-unknown"]
)
def test_a_coding_that_is_not_decoded_is_refused_alone_and_beside_one_that_is(coding):
    for header in (coding, f"{coding}, gzip", f"gzip, {coding}", f"identity, {coding}"):
        with pytest.raises(transport.UnreadableCoding) as raised:
            transport._stages(header, 201)
        assert raised.value.status == 201, header


@pytest.mark.parametrize(
    ("header", "deflates"),
    [
        (None, []),
        ("", []),
        (" ", []),
        ("identity", []),
        ("Identity", []),
        ("gzip,", [False]),
        (", deflate", [True]),
        ("identity, deflate", [True]),
    ],
)
def test_no_coding_and_the_codings_that_are_decoded_are_not_refused(header, deflates):
    assert [s.deflate for s in transport._stages(header, 200)] == deflates


@pytest.mark.parametrize("path", ["/br", "/zstd", "/unknown-coding", "/gzip-then-br"])
def test_an_answer_in_a_coding_that_is_not_decoded_is_refused_whatever_its_bytes(
    fake_judge, path
):
    # Each of these is valid JSON as it is sent: it is refused before it is read.
    with pytest.raises(transport.UnreadableCoding) as raised:
        transport.post(endpoint_of(fake_judge.url(path)), BODY, HEADERS, timeout=5)
    assert raised.value.status == 200
    assert not isinstance(raised.value, transport.TRANSPORT_ERRORS)


def test_an_answer_in_a_coding_that_is_not_decoded_is_not_read(fake_judge, monkeypatch):
    def refused(self, *args):
        raise AssertionError("the body was read")

    monkeypatch.setattr(transport.http.client.HTTPResponse, "read1", refused)
    with pytest.raises(transport.UnreadableCoding):
        transport.post(endpoint_of(fake_judge.url("/zstd")), BODY, HEADERS, timeout=5)


@pytest.mark.parametrize("path", ["/gzip-bomb", "/gzip-bomb-in-chunks"])
def test_a_small_answer_that_decodes_to_more_than_the_limit_is_refused(
    fake_judge, path
):
    with pytest.raises(transport.AnswerTooLarge) as raised:
        transport.post(endpoint_of(fake_judge.url(path)), BODY, HEADERS, timeout=30)
    assert raised.value.status == 200


def test_a_chunk_decodes_to_no_more_than_the_room_it_is_given():
    stage = transport._Stage("gzip", 200)
    with pytest.raises(transport.AnswerTooLarge) as raised:
        stage.decode(gzip.compress(b"x" * 100), 99)
    assert raised.value.status == 200
    assert transport._Stage("gzip", 200).decode(gzip.compress(b"x" * 100), 100) == (
        b"x" * 100
    ), "exactly the room is read"


@pytest.mark.parametrize(
    ("path", "status"), [("/redirect", 302), ("/multiple-choices", 300)]
)
def test_a_redirect_is_returned_and_never_followed(fake_judge, path, status):
    answer = transport.post(endpoint_of(fake_judge.url(path)), BODY, HEADERS, timeout=5)
    assert answer == Answer(status, b"")
    assert [r.path for r in fake_judge.requests] == [path]


def test_a_header_the_caller_gives_replaces_the_default_user_agent(fake_judge):
    transport.post(
        endpoint_of(fake_judge.url("/supported")),
        BODY,
        {**HEADERS, "User-Agent": "judge-client/1"},
        timeout=5,
    )
    assert fake_judge.requests[0].headers["user-agent"] == "judge-client/1"


def spy_on_connections(monkeypatch, *, refuse):
    """Record each connection made through ``http.client``, and with ``refuse`` connect nowhere."""
    made = []

    def spy(base):
        class Spy(base):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.made_with = (args, kwargs)
                self.closed = 0
                made.append(self)

            def connect(self):
                if refuse:
                    raise ConnectionRefusedError("no connection in this test")
                super().connect()

            def close(self):
                self.closed += 1
                super().close()

        return Spy

    # http.client's classes name each other as globals: only the transport's view of the module is swapped.
    client = types.SimpleNamespace(
        HTTPConnection=spy(http.client.HTTPConnection),
        HTTPSConnection=spy(http.client.HTTPSConnection),
    )
    monkeypatch.setattr(transport, "http", types.SimpleNamespace(client=client))
    return made


@pytest.mark.parametrize(
    ("url", "host", "port"),
    [
        ("http://h/p", "h", 80),
        ("http://[::1]/p", "::1", 80),
        ("https://h/p", "h", 443),
        ("https://[::1]/p", "::1", 443),
        ("http://h:81/p", "h", 81),
        ("https://h:80/p", "h", 80),
        ("http://h:0/p", "h", 0),
        ("https://[::1]:0/p", "::1", 0),
    ],
)
def test_the_port_is_always_given_so_an_ipv6_host_is_not_cut_at_its_last_group(
    monkeypatch, url, host, port
):
    made = spy_on_connections(monkeypatch, refuse=True)
    with pytest.raises(ConnectionRefusedError):
        transport.post(endpoint_of(url), BODY, HEADERS, timeout=7)
    [connection] = made
    assert connection.made_with == ((host, port), {"timeout": 7})


@pytest.mark.parametrize("path", ["/supported", "/500", "/big"])
def test_the_connection_is_closed_however_the_call_ends(fake_judge, monkeypatch, path):
    made = spy_on_connections(monkeypatch, refuse=False)
    try:
        transport.post(endpoint_of(fake_judge.url(path)), BODY, HEADERS, timeout=30)
    except transport.AnswerTooLarge:
        pass
    [connection] = made
    assert connection.closed >= 1


def test_the_connection_is_closed_when_it_fails(monkeypatch):
    made = spy_on_connections(monkeypatch, refuse=True)
    with pytest.raises(ConnectionRefusedError):
        transport.post(endpoint_of("http://h/p"), BODY, HEADERS, timeout=5)
    assert made[0].closed >= 1


def test_an_answer_longer_than_the_limit_is_refused_with_its_status(fake_judge):
    with pytest.raises(transport.AnswerTooLarge) as raised:
        transport.post(endpoint_of(fake_judge.url("/big")), BODY, HEADERS, timeout=30)
    assert raised.value.status == 200
    assert str(transport.MAX_ANSWER_BYTES) in str(raised.value)


def test_an_answer_of_exactly_the_limit_is_read_and_one_byte_more_is_not(
    fake_judge, monkeypatch
):
    url = endpoint_of(fake_judge.url("/supported"))
    length = len(transport.post(url, BODY, HEADERS, timeout=5).body)
    monkeypatch.setattr(transport, "MAX_ANSWER_BYTES", length)
    assert len(transport.post(url, BODY, HEADERS, timeout=5).body) == length
    monkeypatch.setattr(transport, "MAX_ANSWER_BYTES", length - 1)
    with pytest.raises(transport.AnswerTooLarge):
        transport.post(url, BODY, HEADERS, timeout=5)


def test_a_timeout_is_a_timeout_error(fake_judge):
    with pytest.raises(TimeoutError):
        transport.post(endpoint_of(fake_judge.url("/slow")), BODY, HEADERS, timeout=0.2)


def test_a_refused_connection_is_an_os_error():
    with pytest.raises(ConnectionRefusedError):
        transport.post(
            endpoint_of(f"http://127.0.0.1:{free_port()}/judge"),
            BODY,
            HEADERS,
            timeout=5,
        )


def test_an_ipv6_host_is_connected_to_as_a_host_and_not_read_as_a_port():
    with pytest.raises(OSError):  # refused, or no route: never a malformed URL
        transport.post(endpoint_of("http://[::1]:9/judge"), BODY, HEADERS, timeout=5)


def test_a_scheme_other_than_http_and_https_is_unsupported():
    with pytest.raises(transport.UnsupportedProtocol, match="unknown protocol 'ftp'"):
        transport.post(endpoint_of("ftp://127.0.0.1/judge"), BODY, HEADERS, timeout=5)


def test_https_to_a_server_that_does_not_speak_tls_fails_the_handshake(fake_judge):
    with pytest.raises(OSError):
        transport.post(
            endpoint_of(fake_judge.url("/supported").replace("http:", "https:")),
            BODY,
            HEADERS,
            timeout=5,
        )
    assert fake_judge.requests == []


def test_no_proxy_in_the_environment_is_used(fake_judge, monkeypatch):
    for name in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.setenv(name, f"http://127.0.0.1:{free_port()}")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    answer = transport.post(
        endpoint_of(fake_judge.url("/supported")), BODY, HEADERS, timeout=5
    )
    assert answer.status == 200 and len(fake_judge.requests) == 1


def test_no_netrc_credential_is_sent(fake_judge, monkeypatch, tmp_path):
    netrc = tmp_path / ".netrc"
    netrc.write_text("machine 127.0.0.1 login netrc-user password netrc-secret\n")
    netrc.chmod(0o600)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("NETRC", str(netrc))
    transport.post(endpoint_of(fake_judge.url("/supported")), BODY, HEADERS, timeout=5)
    assert "authorization" not in fake_judge.requests[0].headers


def test_user_info_in_the_url_is_not_sent_in_place_of_the_bearer_token(fake_judge):
    url = f"http://user:secret@{fake_judge.host_port}/supported"
    transport.post(
        endpoint_of(url), BODY, {**HEADERS, "Authorization": "Bearer abc"}, timeout=5
    )
    transport.post(endpoint_of(url), BODY, HEADERS, timeout=5)
    with_token, without = fake_judge.requests
    assert with_token.headers["authorization"] == "Bearer abc"
    assert "authorization" not in without.headers
