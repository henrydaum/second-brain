"""Magic-link auth for the web demo.

Tokens are random URL-safe strings persisted in `web_auth_tokens`. Verification
marks the token used (single-use) and the caller (frontend route handler) is
responsible for binding the email to an account row and setting the cookie.

Email is sent via the existing Gmail service (plugins/services/service_gmail.py).
Henry already has Google OAuth wired up there, so we reuse it instead of adding
a second transport.
"""

from __future__ import annotations

import logging
import re
import secrets
import time

logger = logging.getLogger("web_auth")

TOKEN_TTL_SECONDS = 30 * 60  # 30 minutes
_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")


def is_email(value: str) -> bool:
    return bool(value) and bool(_EMAIL_RE.match(value.strip()))


def normalize_email(value: str) -> str:
    return (value or "").strip().lower()


def mint_token(db, email: str) -> str:
    """Insert a new single-use auth token for the email. Returns the token."""
    token = secrets.token_urlsafe(24)
    now = time.time()
    with db.lock:
        db.conn.execute(
            "INSERT INTO web_auth_tokens (token, email, created_at, used_at) VALUES (?, ?, ?, NULL)",
            (token, email, now),
        )
        db.conn.commit()
    return token


def verify_token(db, token: str) -> str | None:
    """Consume a token if valid and unused. Returns the email or None."""
    if not token:
        return None
    now = time.time()
    cutoff = now - TOKEN_TTL_SECONDS
    with db.lock:
        row = db.conn.execute(
            "SELECT email, created_at, used_at FROM web_auth_tokens WHERE token = ?",
            (token,),
        ).fetchone()
        if not row or row["used_at"] is not None:
            return None
        if row["created_at"] < cutoff:
            return None
        db.conn.execute("UPDATE web_auth_tokens SET used_at = ? WHERE token = ?", (now, token))
        db.conn.commit()
        return row["email"]


def send_magic_link(gmail_service, to_email: str, link_url: str, from_address: str | None = None) -> bool:
    """Send the magic-link email. Returns True if delivered."""
    if gmail_service is None:
        logger.warning("[web_auth] No Gmail service available; magic link not sent. Link=%s", link_url)
        return False
    subject = "Sign in to Second Brain"
    body = (
        "Tap the link below to sign in. It expires in 30 minutes and can only be used once.\n\n"
        f"{link_url}\n\n"
        "If you didn't request this, you can ignore this email."
    )
    try:
        msg_id = gmail_service.send_message(to=to_email, subject=subject, body=body, from_address=from_address or "me")
        return bool(msg_id)
    except Exception as e:
        logger.exception("[web_auth] Magic-link send failed: %s", e)
        return False
