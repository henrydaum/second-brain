"""Tests for MCP OAuth support. Skipped if the mcp package isn't installed."""
import asyncio

import pytest

pytest.importorskip("mcp")

from plugins.services.helpers.mcp_oauth import (  # noqa: E402
    FileTokenStorage,
    build_oauth_provider,
    make_oauth_handlers,
)


# ── handlers (the testable core of the OAuth bridge) ─────────────────

def test_handlers_parse_pasted_redirect_url():
    redirect, callback = make_oauth_handlers(lambda url: "http://localhost/?code=C&state=S")

    async def go():
        await redirect("https://auth/url")
        return await callback()

    assert asyncio.run(go()) == ("C", "S")


def test_handlers_pass_auth_url_to_prompt():
    seen = {}

    def prompt(url):
        seen["url"] = url
        return "code=X"

    redirect, callback = make_oauth_handlers(prompt)

    async def go():
        await redirect("https://auth/here")
        await callback()

    asyncio.run(go())
    assert seen["url"] == "https://auth/here"


def test_handlers_raise_when_user_aborts():
    redirect, callback = make_oauth_handlers(lambda url: None)

    async def go():
        await redirect("u")
        await callback()

    with pytest.raises(RuntimeError):
        asyncio.run(go())


# ── token storage ────────────────────────────────────────────────────

def test_token_storage_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr("paths.DATA_DIR", tmp_path, raising=False)
    from mcp.shared.auth import OAuthToken

    storage = FileTokenStorage("server one")

    async def go():
        assert await storage.get_tokens() is None
        await storage.set_tokens(OAuthToken(access_token="tok", token_type="Bearer"))
        return await storage.get_tokens()

    got = asyncio.run(go())
    assert got is not None and got.access_token == "tok"


def test_build_provider_does_not_raise():
    # Construction should be pure config (no network), so this verifies our
    # OAuthClientMetadata is accepted by the SDK.
    provider = build_oauth_provider("https://server/mcp", "srv", None, lambda url: None)
    assert provider is not None
