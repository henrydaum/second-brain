"""Egress: where a plugin may reach, and who decides.

``net.http`` is the single control that makes generous filesystem and database
reads safe, so the only thing that relaxes it is a list the *user* keeps. These
pin that the decision cannot migrate into the code being decided about, and
that the relaxation fails closed in every direction it could fail open.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from socketserver import TCPServer

import pytest

from runtime.context import set_kernel_parts
from sandbox.guest.requests import NET_HTTP, Request
from sandbox.handlers.fs_net import _net_http
from sandbox.policy import SAFE, UNSAFE, Chain, classify


def _allow(*hosts):
    """Set the user's allowlist for the duration of a test."""
    set_kernel_parts(config={"net_allowed_hosts": list(hosts)})


@pytest.fixture(autouse=True)
def _empty_allowlist():
    """Every test states its own allowlist; none inherits the last one."""
    _allow()
    yield
    _allow()


def _decide(url):
    return classify(Request(NET_HTTP, {"url": url}), Chain(root="user"))


# ── the allowlist ─────────────────────────────────────────────────────

def test_an_unlisted_host_is_asked_about():
    """The default is empty, so the default is a dialog."""
    decision = _decide("https://api.search.brave.com/res/v1/web/search?q=x")
    assert decision.level == UNSAFE
    assert "api.search.brave.com" in decision.reason


def test_a_listed_host_is_allowed():
    _allow("api.search.brave.com")
    decision = _decide("https://api.search.brave.com/res/v1/web/search?q=x")
    assert decision.level == SAFE
    assert "api.search.brave.com" in decision.reason


def test_a_bare_domain_covers_its_subdomains():
    """Naming a service means the service, not one hostname of it."""
    _allow("duckduckgo.com")
    assert _decide("https://html.duckduckgo.com/html/").level == SAFE


def test_a_lookalike_domain_is_not_covered():
    """The match is on a dot boundary.

    A plain suffix comparison would hand every attacker-registered lookalike
    the grant — ``notduckduckgo.com`` ends with ``duckduckgo.com``.
    """
    _allow("duckduckgo.com")
    assert _decide("https://notduckduckgo.com/x").level == UNSAFE


def test_the_verb_is_never_consulted():
    """A GET with data in the query string is exfiltration too."""
    _allow("example.com")
    for method in ("GET", "POST", "PUT", "DELETE"):
        request = Request(NET_HTTP, {"url": "https://other.example/x",
                                     "method": method})
        assert classify(request, Chain(root="user")).level == UNSAFE


def test_the_path_is_never_consulted():
    """Inside an allowed host, anything goes — that is what the grant means."""
    _allow("example.com")
    assert _decide("https://example.com/anything?at=all").level == SAFE


# ── failing closed ────────────────────────────────────────────────────

def test_a_secret_handle_where_a_host_goes_is_asked_about():
    """Substitution happens in the handler, after this decision.

    So the policy function sees the handle, not the hostname it stands for. It
    must not resolve to something an allowlist entry can match, or a plugin
    could smuggle a destination past the gate inside a credential.
    """
    _allow("example.com")
    assert _decide("https://<secret:host>/x").level == UNSAFE


def test_an_unparseable_url_is_asked_about():
    """No host means no match, in the safe direction."""
    _allow("example.com")
    for url in ("::::", "", "not a url", "///"):
        assert _decide(url).level == UNSAFE


def test_an_absent_kernel_allows_nothing():
    """Tests and a bare container have no config, and get no grant."""
    set_kernel_parts(config={})
    assert _decide("https://example.com/x").level == UNSAFE


def test_the_allowlist_accepts_a_comma_separated_string():
    """``/config`` edits are text, and a person will type one line."""
    set_kernel_parts(config={"net_allowed_hosts": "example.com, other.test"})
    assert _decide("https://other.test/x").level == SAFE


def test_a_plugin_cannot_declare_its_own_reach():
    """The grant lives in config, so nothing on the Request can add to it.

    An ``endpoints``-style declaration would make the contained code the
    authority on its own containment — the bug ``sandbox/isolation.py`` exists
    to prevent, one level down.
    """
    request = Request(NET_HTTP, {"url": "https://example.com/x",
                                "allowed_hosts": ["example.com"],
                                "endpoints": ["example.com"]})
    assert classify(request, Chain(root="user")).level == UNSAFE


# ── what the answer carries ───────────────────────────────────────────

#: Bytes that are emphatically not text, so a download that quietly decoded
#: them would come back mangled rather than merely different.
BLOB = bytes(range(256)) * 512


class _Handler(BaseHTTPRequestHandler):
    """Answers 200 on /ok and 429 with an explanation everywhere else."""

    redirect_hits = 0
    last_path = ""
    last_body = b""
    last_content_type = ""

    def log_message(self, *args):
        return

    def _binary(self, payload, declare=True):
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        if declare:
            self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        try:
            self.wfile.write(payload)
        except OSError:
            pass

    def do_GET(self):
        type(self).last_path = self.path
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/redirect-target")
            self.end_headers()
            return
        if self.path == "/redirect-target":
            type(self).redirect_hits += 1
            self.send_response(200)
            self.end_headers()
            return
        if self.path == "/blob":
            return self._binary(BLOB)
        if self.path == "/blob-redirect":
            self.send_response(302)
            self.send_header("Location", "/blob")
            self.end_headers()
            return
        if self.path == "/offsite":
            self.send_response(302)
            self.send_header("Location", "https://elsewhere.test/thing.bin")
            self.end_headers()
            return
        if self.path == "/declares-too-much":
            self.send_response(200)
            self.send_header("Content-Length", str(64 * 1024 * 1024))
            self.end_headers()
            return
        if self.path == "/undeclared-flood":
            # No Content-Length, so only the streaming cap can catch it.
            return self._binary(BLOB * 200, declare=False)
        body, code = (b'{"hello":"world"}', 200) if self.path == "/ok" else (
            json.dumps({"error": "rate limited", "retry_after": 60}).encode(),
            429)
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("X-Trace", "abc123")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        type(self).last_body = self.rfile.read(length)
        type(self).last_content_type = self.headers.get("Content-Type") or ""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b"{}")


@pytest.fixture
def server():
    """A local HTTP server, so these tests need no network."""
    _Handler.redirect_hits = 0
    _Handler.last_path = ""
    _Handler.last_body = b""
    _Handler.last_content_type = ""
    httpd = TCPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def test_a_reply_carries_status_body_and_headers(server):
    result = _net_http(None, {"url": f"{server}/ok"})
    assert result.ok
    assert result.data["status"] == 200
    assert json.loads(result.data["body"]) == {"hello": "world"}
    # Lowercased, because HTTP header names are case-insensitive and a caller
    # comparing them should not have to guess which case it got.
    assert result.data["headers"]["x-trace"] == "abc123"


def test_params_merge_with_the_existing_query(server):
    result = _net_http(None, {
        "url": f"{server}/ok?fixed=1#section",
        "params": {"q": "hello world", "tag": ["a", "b"]},
    })

    assert result.ok
    assert _Handler.last_path == "/ok?fixed=1&q=hello+world&tag=a&tag=b"


def test_json_is_encoded_and_gets_a_content_type(server):
    result = _net_http(None, {
        "url": f"{server}/send", "method": "POST",
        "json": {"room": "general", "on": True},
    })

    assert result.ok
    assert json.loads(_Handler.last_body) == {"room": "general", "on": True}
    assert _Handler.last_content_type == "application/json"


def test_json_respects_an_explicit_content_type(server):
    result = _net_http(None, {
        "url": f"{server}/send", "method": "POST",
        "headers": {"content-type": "application/problem+json"},
        "json": {"problem": True},
    })

    assert result.ok
    assert _Handler.last_content_type == "application/problem+json"


def test_guest_http_options_are_additive_and_json_decoding_is_opt_in():
    from sandbox.guest.requests import Result
    from sandbox.guest.sdk import SDK

    class Channel:
        def __init__(self):
            self.requests = []

        def send(self, request):
            self.requests.append(request)
            return Result(data={"status": 200, "headers": {},
                                "body": '{"items":[1,2]}'})

    channel = Channel()
    sdk = SDK(channel)
    plain = sdk.net.http("https://example.test")
    structured = sdk.net.http_json(
        "https://example.test", params={"after": 3}, json={"ok": True})

    assert plain["body"] == '{"items":[1,2]}'
    assert structured["body"] == {"items": [1, 2]}
    assert channel.requests[0].args == {
        "url": "https://example.test", "method": "GET",
        "headers": {}, "body": None}
    assert channel.requests[1].args["params"] == {"after": 3}
    assert channel.requests[1].args["json"] == {"ok": True}


def test_guest_http_rejects_two_request_bodies():
    from sandbox.guest.sdk import SDK

    with pytest.raises(ValueError, match="mutually exclusive"):
        SDK(None).net.http("https://example.test", body="x", json={"x": 1})


def test_http_json_handles_empty_and_malformed_bodies():
    from sandbox.guest.requests import Result
    from sandbox.guest.sdk import SDK

    class Channel:
        body = ""

        def send(self, request):
            return Result(data={"status": 204, "headers": {},
                                "body": self.body})

    channel = Channel()
    sdk = SDK(channel)
    assert sdk.net.http_json("https://example.test")["body"] is None

    channel.body = "not json"
    with pytest.raises(ValueError, match="HTTP 204"):
        sdk.net.http_json("https://example.test")


def test_an_error_status_is_an_answer_not_a_failure(server):
    """The body is where an API says *why*, and it used to be discarded.

    ``Result.failure("http 429")`` told a caller nothing it could act on —
    which is why the store's web-search service grew a private
    ``_read_http_error_body`` and its tool reached into it.
    """
    result = _net_http(None, {"url": f"{server}/rate-limited"})
    assert result.ok
    assert result.data["status"] == 429
    assert json.loads(result.data["body"])["retry_after"] == 60


def test_a_redirect_is_returned_and_never_followed(server):
    """Following would spend the original host decision on a new URL."""
    result = _net_http(None, {"url": f"{server}/redirect"})

    assert result.ok
    assert result.data["status"] == 302
    assert result.data["headers"]["location"] == "/redirect-target"
    assert _Handler.redirect_hits == 0


def test_no_reply_at_all_is_still_a_failure():
    """A refused connection produced no answer, so there is nothing to hand
    back but the reason."""
    result = _net_http(None, {"url": "http://127.0.0.1:1/nothing-here"})
    assert not result.ok
    assert result.retryable


def test_a_non_http_scheme_is_refused():
    """``file://`` would read any path, and ``data:`` is not egress at all.

    Sharper than it looks: an approved command puts ``net.http`` in
    ``chain.approved``, so the policy function returns SAFE and only this check
    stands between that grant and ``file:///…/config.json``.
    """
    for url in ("file:///etc/passwd", "data:text/plain,hi", "ftp://x/y"):
        assert _net_http(None, {"url": url}).denied


# ── downloads ─────────────────────────────────────────────────────────
#
# ``to_file`` is the only way anything binary or large crosses at all: the wire
# carries decoded text under a 16 MB cap, and the things worth saving are
# neither. These pin that the bytes arrive intact, that the two grants a
# download needs are both asked for, and that nothing survives a limit being
# hit — a partial file is the failure mode that looks like success.


class _Ctx:
    """The slice of a context ``_download_cap`` reads. 4 MB, so the flood
    endpoint overruns it without the test having to serve 100 MB."""

    config = {"max_download_mb": 4}


def test_a_download_writes_the_bytes_and_answers_about_the_file(server, tmp_path):
    """Not decoded, not truncated, not base64 — the file on disk is the file."""
    dest = tmp_path / "thing.bin"
    result = _net_http(_Ctx(), {"url": f"{server}/blob", "to_file": str(dest)})

    assert result.ok
    assert dest.read_bytes() == BLOB
    assert result.data["path"] == str(dest)
    assert result.data["bytes"] == len(BLOB)
    assert result.data["content_type"] == "application/octet-stream"
    # The body key stays, empty: one shape whichever branch answered.
    assert result.data["body"] == ""


def test_a_download_follows_a_redirect_inside_the_same_host(server, tmp_path):
    """The host was classified before the handler ran, so another hop inside
    it grants nothing new — and almost every real download takes one."""
    dest = tmp_path / "viaredirect.bin"
    result = _net_http(_Ctx(), {"url": f"{server}/blob-redirect",
                                "to_file": str(dest)})

    assert result.ok
    assert dest.read_bytes() == BLOB


def test_a_download_stops_at_a_redirect_to_another_host(server, tmp_path):
    """Following would spend one host's decision on a different host.

    It comes back as the 3xx it is, in the download shape with the file half
    empty, so the guest re-calls and the new host meets the gate.
    """
    dest = tmp_path / "offsite.bin"
    result = _net_http(_Ctx(), {"url": f"{server}/offsite", "to_file": str(dest)})

    assert result.ok
    assert result.data["status"] == 302
    assert "elsewhere.test" in result.data["headers"]["location"]
    assert result.data["path"] == ""
    assert not dest.exists()


def test_an_error_status_downloads_nothing_and_still_explains_itself(server, tmp_path):
    """Same shape as a success, so one branch on ``status`` covers both."""
    dest = tmp_path / "missing.bin"
    result = _net_http(_Ctx(), {"url": f"{server}/rate-limited",
                                "to_file": str(dest)})

    assert result.ok
    assert result.data["status"] == 429
    assert json.loads(result.data["body"])["retry_after"] == 60
    assert result.data["path"] == ""
    assert not dest.exists()


def test_a_declared_oversize_reply_is_refused_before_it_is_read(server, tmp_path):
    """The cheapest refusal available: the server said how big it is."""
    dest = tmp_path / "huge.bin"
    result = _net_http(_Ctx(), {"url": f"{server}/declares-too-much",
                                "to_file": str(dest)})

    assert not result.ok
    assert "download limit" in result.error
    assert not dest.exists()


def test_an_undeclared_oversize_reply_is_caught_while_streaming(server, tmp_path):
    """A server need not declare a length, and a reply that never ends is
    exactly what a declared one cannot catch."""
    dest = tmp_path / "flood.bin"
    result = _net_http(_Ctx(), {"url": f"{server}/undeclared-flood",
                                "to_file": str(dest)})

    assert not result.ok
    assert "download limit" in result.error
    # The partial file goes with it. Half a file is not a smaller answer, and
    # leaving one behind would present as a successful download of a corrupt
    # file — the failure mode this whole branch exists to avoid.
    assert not dest.exists()


def test_a_guest_may_lower_the_ceiling_but_never_raise_it(server, tmp_path):
    """The same rule the timeouts follow: a plugin may ask, it does not get to
    grant itself."""
    small = tmp_path / "small.bin"
    assert not _net_http(_Ctx(), {"url": f"{server}/blob",
                                  "to_file": str(small),
                                  "max_bytes": 64}).ok

    big = tmp_path / "big.bin"
    result = _net_http(_Ctx(), {"url": f"{server}/declares-too-much",
                                "to_file": str(big),
                                "max_bytes": 512 * 1024 * 1024})
    assert not result.ok
    assert str(4 * 1024 * 1024) in result.error


def test_a_kernel_owned_destination_is_refused_before_the_request(
        server, tmp_path, monkeypatch):
    """``_guard_write`` covers this write like any other.

    A doomed destination should not cost a round trip, let alone one that
    leaves bytes with nowhere to go. The deny-list is faked rather than read,
    because ``protected_paths`` is cached per process and a test that named a
    *real* protected path would be relying on the state some earlier test left
    that cache in — which is a way of writing to the real file.
    """
    from sandbox import protected

    forbidden = tmp_path / "config.json"
    monkeypatch.setattr(protected, "protected_paths",
                        lambda: {forbidden.resolve(): "test"})

    result = _net_http(_Ctx(), {"url": f"{server}/blob",
                                "to_file": str(forbidden)})
    assert result.denied
    assert not forbidden.exists()


def test_an_inline_body_is_capped_rather_than_unbounded(server):
    """It was ``response.read()``, and the only thing that stopped a large
    reply was ``protocol.encode`` refusing the finished Result — a crash-shaped
    answer, after the kernel had already paid for the whole thing."""
    from sandbox.handlers import fs_net

    original = fs_net.MAX_READ_BYTES
    fs_net.MAX_READ_BYTES = 100
    try:
        result = _net_http(None, {"url": f"{server}/blob"})
    finally:
        fs_net.MAX_READ_BYTES = original

    assert result.ok
    assert result.data["truncated"] is True
    assert len(result.data["body"]) <= 100


# ── what a download is allowed to be ──────────────────────────────────

def _decide_download(url, dest, **chain):
    return classify(Request(NET_HTTP, {"url": url, "to_file": str(dest)}),
                    Chain(root="user", **chain))


def test_a_download_needs_both_the_host_and_the_destination():
    """Two grants kept by different people, so the stricter one wins.

    Otherwise ``net_allowed_hosts`` would quietly be a way of granting writes,
    which is not what anybody typed it for.
    """
    import trees

    inside = Path(trees.tree("workspace").path) / "downloads" / "a.png"
    _allow("example.com")

    assert _decide_download("https://example.com/x", inside).level == SAFE
    # Right host, wrong destination.
    assert _decide_download("https://example.com/x",
                            Path.home() / "a.png").level == UNSAFE
    # Right destination, wrong host.
    assert _decide_download("https://nope.test/x", inside).level == UNSAFE


def test_an_approved_command_does_not_get_the_write_for_free():
    """A command declaring egress declared egress.

    ``chain.approved`` is otherwise the whole answer, and this is the one
    Request where a type is two capabilities — so the destination half has to
    be named too, exactly as it would be if the command wrote the bytes itself.
    """
    import trees
    from sandbox.guest.requests import FS_WRITE_BYTES

    inside = Path(trees.tree("workspace").path) / "downloads" / "a.png"

    egress_only = _decide_download("https://nope.test/x", inside,
                                   approved=frozenset({NET_HTTP}))
    assert egress_only.level == UNSAFE

    both = _decide_download("https://nope.test/x", inside,
                            approved=frozenset({NET_HTTP, FS_WRITE_BYTES}))
    assert both.level == SAFE


def test_a_plain_fetch_is_unchanged_by_any_of_this():
    """Pass no destination and the old answer comes back, byte for byte."""
    _allow("example.com")
    assert _decide("https://example.com/x").level == SAFE
    assert _decide("https://nope.test/x").level == UNSAFE
