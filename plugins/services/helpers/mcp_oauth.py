"""OAuth 2.1 support for remote (HTTP) MCP servers.

Isolated in its own module so [service_mcp.py](../service_mcp.py) stays importable
without the ``mcp`` package — this module is only imported when an OAuth-capable
server is actually being connected, at which point ``mcp`` is already present.

The MCP SDK's ``OAuthClientProvider`` drives the protocol (discovery, dynamic
client registration, PKCE, token refresh). We supply three things:

- ``FileTokenStorage`` — caches the registered client + tokens per server under
  ``DATA_DIR/mcp_oauth/<server>.json`` so authorization happens once, then
  refreshes silently.
- ``redirect_handler`` — receives the authorization URL (we stash it).
- ``callback_handler`` — runs the user-facing prompt (off the event loop via a
  worker thread) and returns the ``(code, state)`` the user pastes back.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path

from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken

from runtime.interactive_auth import extract_oauth_code

logger = logging.getLogger("MCPOAuth")

_SAFE = re.compile(r"[^a-zA-Z0-9_.-]")


class FileTokenStorage(TokenStorage):
    """Per-server token + client-registration cache on disk."""

    def __init__(self, server_name: str):
        from paths import DATA_DIR
        self._path = DATA_DIR / "mcp_oauth" / f"{_SAFE.sub('_', server_name)}.json"

    def _read(self) -> dict:
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _write_key(self, key: str, value) -> None:
        data = self._read()
        data[key] = value
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    async def get_tokens(self) -> OAuthToken | None:
        raw = self._read().get("tokens")
        return OAuthToken.model_validate(raw) if raw else None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        self._write_key("tokens", tokens.model_dump(mode="json"))

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        raw = self._read().get("client")
        return OAuthClientInformationFull.model_validate(raw) if raw else None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        self._write_key("client", client_info.model_dump(mode="json"))


def make_oauth_handlers(prompt_fn):
    """Return ``(redirect_handler, callback_handler)`` for an OAuth provider.

    ``prompt_fn(auth_url) -> str | None`` is a *blocking* callable that shows the
    URL to the user and returns the pasted-back code/redirect URL (or None to
    abort). It runs on a worker thread so it never blocks the event loop.
    Extracted from ``build_oauth_provider`` so the parsing/abort logic is
    testable without constructing a full provider.
    """
    holder: dict[str, str | None] = {"url": None}

    async def redirect_handler(auth_url: str) -> None:
        holder["url"] = auth_url

    async def callback_handler() -> tuple[str, str | None]:
        pasted = await asyncio.to_thread(prompt_fn, holder["url"])
        if not pasted:
            raise RuntimeError("MCP authorization was cancelled or timed out.")
        code, state = extract_oauth_code(pasted)
        if not code:
            raise RuntimeError("No authorization code found in the pasted response.")
        return code, state

    return redirect_handler, callback_handler


def build_oauth_provider(server_url: str, server_name: str, scope: str | None, prompt_fn):
    """Construct an ``OAuthClientProvider`` for a server.

    ``prompt_fn`` has the contract described in ``make_oauth_handlers``.
    """
    redirect_handler, callback_handler = make_oauth_handlers(prompt_fn)

    metadata = OAuthClientMetadata(
        client_name="Second Brain",
        # Loopback redirect — we capture the code via paste-back, not a server.
        redirect_uris=["http://localhost:33418/callback"],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        scope=scope or None,
    )

    return OAuthClientProvider(
        server_url=server_url,
        client_metadata=metadata,
        storage=FileTokenStorage(server_name),
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
    )
