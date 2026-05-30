"""Checkout orchestration for credit-pack purchases."""

import logging
import secrets
import time
import uuid

from billing import stripe
from config.config_data import DEFAULT_WEB_CREDITS
from plugins.helpers import web_auth

logger = logging.getLogger("billing.storefront")


def create_checkout(frontend, session_id: str, account_id: str, ip: str) -> dict:
    base, token = frontend._base_url(), secrets.token_urlsafe(24)
    db = getattr(frontend.runtime, "db", None)
    if db is not None:
        with db.lock:
            db.conn.execute("INSERT INTO web_auth_tokens (token, email, created_at, used_at) VALUES (?, ?, ?, NULL)", (token, "__pending_checkout__", time.time()))
            db.conn.commit()
    snap = frontend.account_info(session_id, ip, account_id)
    pack = frontend._credits().pack() if frontend._credits() else DEFAULT_WEB_CREDITS["pack"]
    return stripe.create_checkout_session(
        secret_key=str(frontend.config.get("stripe_secret_key") or ""),
        price_id=str(pack.get("stripe_price_id") or ""),
        price_cents=int(pack.get("price_cents") or DEFAULT_WEB_CREDITS["pack"]["price_cents"]),
        success_url=f"{base}/?checkout=success&claim={token}",
        cancel_url=f"{base}/?checkout=cancel",
        email_hint=snap.get("email") or None,
        metadata={"session_id": session_id, "account_id": account_id or "", "claim_token": token,
                  "anon_user_id": frontend._anon_user_id(session_id, ip), "ip_hash": frontend._ip_hash(ip)},
    )


def handle_webhook(frontend, payload: bytes, sig_header: str) -> dict:
    event = stripe.verify_webhook(str(frontend.config.get("stripe_secret_key") or ""), str(frontend.config.get("stripe_webhook_secret") or ""), payload, sig_header)
    db = getattr(frontend.runtime, "db", None)
    if db is None:
        return {"ok": True, "ignored": "no_db"}
    if event.get("type") != "checkout.session.completed":
        return {"ok": True, "ignored": event.get("type")}
    data = (event.get("data") or {}).get("object") or {}
    meta = data.get("metadata") or {}
    email = web_auth.normalize_email(data.get("customer_email") or (data.get("customer_details") or {}).get("email") or "")
    amount, grant = int(data.get("amount_total") or 0), int((frontend._credits().pack() if frontend._credits() else DEFAULT_WEB_CREDITS["pack"])["credits"])
    if not email:
        logger.warning("[stripe] checkout.session.completed missing email; event=%s", event.get("id"))
        return {"ok": True, "ignored": "no_email"}
    with db.lock:
        try:
            db.conn.execute("INSERT INTO web_payments (stripe_event_id, email, amount_cents, credits_granted, ts) VALUES (?, ?, ?, ?, ?)", (event.get("id"), email, amount, grant, time.time()))
        except Exception:
            db.conn.rollback()
            return {"ok": True, "duplicate": True}
        row = db.conn.execute("SELECT user_id, account_id FROM web_users WHERE email = ?", (email,)).fetchone()
        if row:
            uid, aid = row["user_id"], row["account_id"] or str(uuid.uuid4())
            db.conn.execute("UPDATE web_users SET account_id = ? WHERE user_id = ?", (aid, uid))
        else:
            uid, aid = meta.get("anon_user_id") or "", str(uuid.uuid4())
            upgraded = uid and db.conn.execute("UPDATE web_users SET email = ?, account_id = ? WHERE user_id = ? AND email IS NULL", (email, aid, uid)).rowcount
            if not upgraded:
                uid = str(uuid.uuid4())
                db.conn.execute("INSERT INTO web_users (user_id, session_id, ip_hash, created_at, last_seen, purchased_credits, email, account_id) VALUES (?, ?, ?, ?, ?, 0, ?, ?)", (uid, meta.get("session_id") or "", meta.get("ip_hash") or "", time.time(), time.time(), email, aid))
        if meta.get("claim_token"):
            db.conn.execute("UPDATE web_auth_tokens SET email = ? WHERE token = ? AND used_at IS NULL", (email, meta["claim_token"]))
        db.conn.commit()
    frontend._claim_canvas_actions(meta.get("anon_user_id") or "", aid)
    if frontend._credits():
        frontend._credits().grant(db, uid, grant, "purchase")
    try:
        link = f"{frontend._base_url()}/auth/verify?token={web_auth.mint_token(db, email)}"
        gmail = getattr(frontend.runtime, "services", {}).get("gmail")
        if gmail:
            web_auth.send_magic_link(gmail, email, link, from_address=str(frontend.config.get("web_email_from") or "me"))
        else:
            logger.warning("[stripe] No gmail service; magic link for %s: %s", email, link)
    except Exception:
        logger.exception("[stripe] failed to send post-purchase magic link")
    return {"ok": True}


def claim_checkout(frontend, token: str) -> str | None:
    db = getattr(frontend.runtime, "db", None)
    if db is None or not token:
        return None
    with db.lock:
        cur = db.conn.execute("UPDATE web_auth_tokens SET used_at = ? WHERE token = ? AND used_at IS NULL AND email IS NOT NULL AND email != '__pending_checkout__'", (time.time(), token))
        if (cur.rowcount or 0) != 1:
            db.conn.commit()
            return None
        email = db.conn.execute("SELECT email FROM web_auth_tokens WHERE token = ?", (token,)).fetchone()
        acct = db.conn.execute("SELECT account_id FROM web_users WHERE email = ?", (email["email"],)).fetchone() if email else None
        db.conn.commit()
    return acct["account_id"] if acct else None
