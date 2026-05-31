"""Tests for the frontend-mediated authorization helper."""
from runtime.interactive_auth import authorize_via_frontend, extract_oauth_code


# ── extract_oauth_code ───────────────────────────────────────────────

def test_extract_from_full_redirect_url():
    code, state = extract_oauth_code("http://localhost/?code=abc123&state=xyz")
    assert code == "abc123" and state == "xyz"


def test_extract_from_query_string():
    code, state = extract_oauth_code("code=abc&state=s")
    assert code == "abc" and state == "s"


def test_extract_bare_code():
    code, state = extract_oauth_code("  rawcode  ")
    assert code == "rawcode" and state is None


def test_extract_empty():
    assert extract_oauth_code("") == (None, None)


# ── authorize_via_frontend ───────────────────────────────────────────

class _FakeReq:
    def __init__(self, value, resolved=True, cancelled=False):
        self.id = "req1"
        self.value = value
        self.metadata = {"cancelled": cancelled}
        self._resolved = resolved

    def wait(self, timeout=None):
        return self._resolved


class _FakeRuntime:
    def __init__(self, req):
        self._req = req
        self.last = None
        self.actions = []

    def request_input(self, session_key, title, prompt, type="boolean"):
        self.last = {"session_key": session_key, "title": title, "prompt": prompt, "type": type}
        return self._req

    def handle_action(self, *a, **k):
        self.actions.append((a, k))


def test_returns_pasted_value_and_requests_string():
    rt = _FakeRuntime(_FakeReq("http://localhost/?code=zzz"))
    out = authorize_via_frontend(rt, "chat", "https://auth", instructions="go")
    assert out == "http://localhost/?code=zzz"
    assert rt.last["type"] == "string"
    assert "https://auth" in rt.last["prompt"]


def test_no_session_returns_none():
    rt = _FakeRuntime(_FakeReq("x"))
    assert authorize_via_frontend(rt, None, "https://auth") is None


def test_no_runtime_returns_none():
    assert authorize_via_frontend(None, "chat", "https://auth") is None


def test_timeout_returns_none_and_cancels():
    rt = _FakeRuntime(_FakeReq("x", resolved=False))
    assert authorize_via_frontend(rt, "chat", "https://auth", timeout=0.01) is None
    assert rt.actions, "should have attempted to cancel the dangling request"


def test_cancelled_returns_none():
    rt = _FakeRuntime(_FakeReq("x", cancelled=True))
    assert authorize_via_frontend(rt, "chat", "https://auth") is None
