"""Frontend-mediated authorization.

Some auth flows (MCP OAuth, and potentially others) need to show the user a
URL and collect a value back — but the user may be on Telegram, not at the
machine running Second Brain. This bridges that gap by reusing the runtime's
typed-input request mechanism: push the auth link to the active conversation,
block for the user's pasted-back reply, and return it.

``request_input(type="string")`` pushes a ``PHASE_APPROVING_REQUEST`` the
frontends already render as a free-text prompt, so a plain reply (the pasted
code or redirect URL) resolves it — no frontend changes needed.

IMPORTANT: the caller must not hold the session lock while waiting here, or the
user's answer (which needs that lock) can never arrive. See ``MCPService`` for
how OAuth-capable connections are kept off the locked command-dispatch path.
"""

from __future__ import annotations

import logging
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger("InteractiveAuth")


def extract_oauth_code(text: str) -> tuple[str | None, str | None]:
    """Pull ``(code, state)`` from a pasted authorization code or redirect URL.

    Accepts either a bare code, a full redirect URL
    (``http://localhost/?code=...&state=...``), or a raw query string.
    """
    text = (text or "").strip()
    if not text:
        return None, None
    # A URL or query string carrying ?code=… — parse it out.
    if "code=" in text or "://" in text or text.startswith("?"):
        query = urlparse(text).query or text.lstrip("?")
        params = parse_qs(query)
        code = (params.get("code") or [None])[0]
        state = (params.get("state") or [None])[0]
        if code:
            return code, state
    # Otherwise treat the whole thing as the code itself.
    return text, None


def authorize_via_frontend(
    runtime,
    session_key: str | None,
    auth_url: str,
    *,
    title: str = "Authorization required",
    instructions: str = "",
    timeout: float = 600.0,
) -> str | None:
    """Push ``auth_url`` to the active conversation and return the pasted reply.

    Returns the user's text (a code or redirect URL) on success, or ``None`` if
    there is no conversation to ask in, or the user cancels/times out.
    """
    if runtime is None or not session_key or not hasattr(runtime, "request_input"):
        return None

    prompt = f"{instructions}\n\n{auth_url}".strip() if instructions else auth_url
    try:
        req = runtime.request_input(session_key, title, prompt, type="string")
    except Exception as e:
        logger.warning("Could not request input for authorization: %s", e)
        return None

    if not req.wait(timeout=timeout):
        # Don't leave the conversation parked on a dead approval frame.
        try:
            runtime.handle_action(session_key, "cancel", None)
        except Exception:
            pass
        return None
    if req.metadata.get("cancelled"):
        return None

    value = req.value
    return value.strip() if isinstance(value, str) else None
