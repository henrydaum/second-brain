"""Tiny localhost web frontend for the demo build."""

from __future__ import annotations

import json
import logging
import mimetypes
import hashlib
import random
import secrets
import time
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

from plugins.helpers import stripe_client, web_auth

from canvas import actions as canvas_actions
from canvas.render import pool_hash as _pool_hash, render_canvas as _new_render_canvas
from config import config_manager
from plugins.BaseFrontend import BaseFrontend, FrontendCapabilities
from plugins.helpers.palettes import get_palette, list_palettes
from plugins.skills.helpers.skill_store import anonymize_owner_in_dir
from paths import DATA_DIR, SANDBOX_SKILLS


def _anonymize_skill_owner(owner_values):
    return anonymize_owner_in_dir(SANDBOX_SKILLS, owner_values)


def _read_skill_via(runtime, slug: str):
    registry = getattr(runtime, "skill_registry", None)
    return registry.get_record(slug) if registry is not None else None

logger = logging.getLogger("WebFrontend")
WEB_ROOT = Path(__file__).with_name("web")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FAVICON_PATH = PROJECT_ROOT / "icon.ico"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
WEB_PROFILE = "artist"

class WebFrontend(BaseFrontend):
    """Static website plus JSON chat bridge backed by ConversationRuntime."""

    name = "web"
    description = "Local browser demo frontend."
    capabilities = FrontendCapabilities(supports_buttons=True, supports_message_edit=True, supports_rich_text=True, supports_proactive_push=True)
    config_settings = [
        ("Web Host", "web_host", "Host interface for the demo web server.", "127.0.0.1", {"type": "text"}),
        ("Web Port", "web_port", "Port for the demo web server.", 8765, {"type": "integer"}),
        ("Web Global 5h Turns", "web_global_5h_turn_limit", "Public web chat-turn budget per 5 hours.", 600, {"type": "integer"}),
        ("Web Global Weekly Turns", "web_global_week_turn_limit", "Public web chat-turn budget per 7 days.", 6000, {"type": "integer"}),
        ("Web Session 5h Turns", "web_session_5h_turn_limit", "Per-browser chat-turn budget per 5 hours.", 40, {"type": "integer"}),
        ("Web Session Weekly Turns", "web_session_week_turn_limit", "Per-browser chat-turn budget per 7 days.", 160, {"type": "integer"}),
        ("Web IP 5h Turns", "web_ip_5h_turn_limit", "Per-IP chat-turn budget per 5 hours (cookie-clearing backstop).", 60, {"type": "integer"}),
        ("App Base URL", "app_base_url", "Public origin of the web demo, used in Stripe redirects and magic links.", "http://127.0.0.1:8765", {"type": "text"}),
        ("Price (cents)", "web_price_cents", "Stripe checkout price in cents.", 299, {"type": "integer"}),
        ("Credit Pack Size", "web_credit_pack_size", "Messages granted per purchase.", 2000, {"type": "integer"}),
        ("Stripe Secret Key", "stripe_secret_key", "Stripe API secret key (sk_test_... or sk_live_...).", "", {"type": "text"}),
        ("Stripe Webhook Secret", "stripe_webhook_secret", "Stripe webhook signing secret (whsec_...).", "", {"type": "text"}),
        ("Stripe Price ID", "stripe_price_id", "Stripe Price ID for the credit pack.", "", {"type": "text"}),
        ("Magic-link From", "web_email_from", "Send-as address for magic-link emails (must be a Gmail send-as alias or 'me').", "me", {"type": "text"}),
    ]

    def __init__(self):
        super().__init__()
        self._server = None
        self._outbox: dict[str, list[dict]] = {}
        self._lock = threading.RLock()
        self._usage_lock = threading.RLock()

    def session_key(self, ctx=None) -> str:
        return f"web:{ctx or 'demo'}"

    def start(self) -> None:
        host = str(self.config.get("web_host") or "127.0.0.1")
        port = int(self.config.get("web_port") or 8765)
        self._server = _Server((host, port), _Handler, self)
        logger.info("Web demo listening at http://%s:%s", host, port)
        self._start_conversation_sweeper()
        self._server.serve_forever()

    def _start_conversation_sweeper(self) -> None:
        """Periodically purge demo conversations older than 24h that aren't
        currently bound to a live session. Conversations are intentionally
        ephemeral on the web demo — this catches the ones left behind when
        users close the tab without hitting "New chat"."""
        def loop():
            interval = 3600.0  # hourly
            max_age = 86400.0  # 24h
            while self._server is not None:
                try:
                    self._sweep_stale_conversations(max_age)
                except Exception:
                    logger.exception("conversation sweeper failed")
                time.sleep(interval)
        t = threading.Thread(target=loop, name="web-conversation-sweeper", daemon=True)
        t.start()

    def _sweep_stale_conversations(self, max_age: float) -> int:
        db = getattr(self.runtime, "db", None)
        if db is None:
            return 0
        cutoff = time.time() - max_age
        live_ids = {getattr(s, "conversation_id", None) for s in self.runtime.sessions.values()}
        live_ids.discard(None)
        with db.lock:
            rows = db.conn.execute(
                "SELECT id FROM conversations WHERE category = 'Demo' AND COALESCE(updated_at, created_at) < ?",
                (cutoff,),
            ).fetchall()
        stale = [int(r["id"]) for r in rows if int(r["id"]) not in live_ids]
        for cid in stale:
            try:
                db.delete_conversation(cid)
            except Exception:
                logger.exception("sweeper: delete_conversation failed cid=%s", cid)
        if stale:
            logger.info("conversation sweeper purged %d stale demo conversation(s)", len(stale))
        return len(stale)

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
        self.unbind()

    def chat(self, session_id: str, message: str, ip: str = "", account_id: str = "") -> list[dict]:
        key = self.session_key(session_id)
        text = (message or "").strip()
        if text.startswith("/"):
            return self.new_chat(session_id) if text == "/new" else [{"type": "error", "content": "Slash commands are disabled on the public demo. Use the chat or the New button."}]
        self._ensure_conversation(key)
        if self.has_pending_approval(key):
            return [{"type": "error", "content": "Use the approval buttons to answer this permission request."}]
        ok, error, paywall = self._record_usage(session_id, ip, account_id)
        if not ok:
            if paywall:
                return [{"type": "paywall", **paywall, "message": error}]
            return [{"type": "error", "content": error}]
        self.submit_text(key, text)
        events = self._drain(key)
        # Append fresh account snapshot so the client can keep the chip updated.
        snap = self._account_snapshot(session_id, ip, account_id)
        if snap:
            events.append({"type": "account", "account": snap})
        return events

    def approve(self, session_id: str, value: bool) -> list[dict]:
        key = self.session_key(session_id)
        self._ensure_conversation(key)
        self.submit_text(key, "yes" if value else "no")
        return self._drain(key)

    def new_chat(self, session_id: str) -> list[dict]:
        key = self.session_key(session_id)
        # Ephemeral conversations: wipe the previous transcript before opening a
        # fresh one. The 24h sweeper picks up anything left behind when users
        # close the tab without hitting "New chat".
        session = self.runtime.sessions.get(key)
        prev_cid = getattr(session, "conversation_id", None) if session else None
        self.runtime.close_session(key)
        if prev_cid is not None:
            try:
                self.runtime.db.delete_conversation(prev_cid)
            except Exception:
                logger.exception("new_chat: delete_conversation failed cid=%s", prev_cid)
        # Detach the session from any canvas; for_session will mint a fresh
        # one on the next render.
        cr = self._canvas_runtime()
        if cr is not None:
            cr.unbind_session(key)
        self._ensure_conversation(key)
        return [{"type": "canvas_reset"}, *self._drain(key)]

    def _ensure_conversation(self, key: str) -> None:
        self._ensure_web_profile()
        session = self.runtime.get_session(key)
        if session.conversation_id is not None:
            self._apply_web_scope(key)
            return
        cid = self.runtime.create_conversation("Web demo conversation", kind="user", category="Demo")
        if cid:
            self.runtime.load_conversation(key, cid, agent_profile="default")
            self._apply_web_scope(key)

    def _ensure_web_profile(self) -> None:
        profiles = self.config.setdefault("agent_profiles", {})
        profile = profiles.setdefault(WEB_PROFILE, {})

    def _apply_web_scope(self, key: str) -> None:
        session = self.runtime.sessions.get(key)
        if session:
            session.profile_override = WEB_PROFILE
            session.active_agent_profile = WEB_PROFILE
        self.runtime.add_system_prompt_extra(key, "artist", "Website safety: browser users cannot run slash commands or edit runtime configuration. Use the skill workflow for canvas work: search_skills, then execute_skill or create_skill plus execute_skill. Do not call sharing or gallery tools.")

    def _push(self, session_key: str, item: dict) -> None:
        with self._lock:
            self._outbox.setdefault(session_key, []).append(item)

    def _drain(self, session_key: str) -> list[dict]:
        with self._lock:
            return self._outbox.pop(session_key, [])

    def render_messages(self, session_key: str, messages: list[str]) -> None:
        for msg in messages:
            if msg:
                self._push(session_key, {"type": "message", "role": "assistant", "content": msg})

    def render_attachments(self, session_key: str, paths: list[str]) -> None:
        """Surface attached images without touching canvas state.

        Tools that mutate the canvas already produce a hero_image event via
        their own machinery (execute_skill / manage_layers); here we just
        relay any other image attachments so the UI can show them.
        """
        snap = self._new_canvas_snap(session_key) or {}
        for path in paths:
            p = Path(path)
            if not _is_public_image(p):
                continue
            self._push(session_key, _image_event(p, _canvas_payload_full(self.runtime, session_key, snap)))

    def render_form_field(self, session_key: str, form: dict) -> None:
        self._push(session_key, {"type": "form", "form": form})

    def render_approval_request(self, session_key: str, req) -> None:
        self._push(session_key, {"type": "approval", "title": getattr(req, "title", "Approval requested"), "body": getattr(req, "body", ""), "choices": ["yes", "no"]})

    def render_buttons(self, session_key: str, buttons: list[dict]) -> None:
        self._push(session_key, {"type": "buttons", "buttons": buttons})

    def render_error(self, session_key: str, error: dict) -> None:
        text = (error or {}).get("message") or str(error)
        with self._lock:
            last = (self._outbox.get(session_key) or [{}])[-1]
        if last.get("content") != text:
            self._push(session_key, {"type": "error", "content": text})

    def render_typing(self, session_key: str, on: bool) -> None:
        self._push(session_key, {"type": "typing", "on": bool(on)})

    def render_tool_status(self, session_key: str, payload: dict) -> None:
        name = payload.get("tool_name") or payload.get("command_name") or payload.get("name") or "tool"
        evt = {
            "type": "tool_status",
            "call_id": payload.get("call_id"),
            "name": name,
            "status": payload.get("status", "running"),
            "ok": payload.get("ok"),
            "error": payload.get("error"),
        }
        if payload.get("progress") is not None:
            evt["progress"] = payload["progress"]
        self._push(session_key, evt)

    def _chain_progress_cb(self, key: str):
        """Build an on_step callback that emits tool_status progressed events.
        Only emits when the chain has >1 step, so single-skill renders stay quiet."""
        call_id = uuid.uuid4().hex
        def cb(done: int, total: int) -> None:
            if total <= 1:
                return
            self.render_tool_status(key, {
                "name": "render",
                "status": "finished" if done >= total else "progressed",
                "call_id": call_id,
                "progress": {"done": done, "total": total},
            })
        return cb

    def _live_session_keys(self) -> list[str]:
        return [k for k in getattr(self.runtime, "sessions", {}) if k.startswith("web:")]

    def _anon_user_id(self, session_id: str, ip: str) -> str:
        """Anonymous fallback id. Stable across a browser session (no cookie clear)."""
        return hashlib.sha256(f"{ip}|{session_id}".encode()).hexdigest()[:24]

    def _ip_hash(self, ip: str) -> str:
        return hashlib.sha256(ip.encode()).hexdigest()[:16] if ip else ""

    def _resolve_account(self, db, account_id: str) -> dict | None:
        """Return the web_users row keyed by account_id, or None."""
        if not account_id or db is None:
            return None
        with db.lock:
            row = db.conn.execute(
                "SELECT user_id, email, account_id, tier, credits FROM web_users WHERE account_id = ?",
                (account_id,),
            ).fetchone()
        return dict(row) if row else None

    def _record_usage(self, session_id: str, ip: str, account_id: str = "") -> tuple[bool, str, dict | None]:
        """Returns (allowed, error_msg, paywall_payload_or_None).
        paywall_payload is set only when the user has hit a personal cap and an
        offer should be shown (vs. global demo cap → plain error)."""
        db = getattr(self.runtime, "db", None)
        if db is None:
            return True, "", None
        now = time.time()
        five_h, week = now - 5 * 3600, now - 7 * 24 * 3600
        ip_hash = self._ip_hash(ip)

        with self._usage_lock, db.lock:
            account = None
            if account_id:
                row = db.conn.execute(
                    "SELECT user_id, email, tier, credits FROM web_users WHERE account_id = ?",
                    (account_id,),
                ).fetchone()
                if row:
                    account = dict(row)
            uid = account["user_id"] if account else self._anon_user_id(session_id, ip)
            if not account:
                db.conn.execute(
                    "INSERT INTO web_users (user_id, session_id, ip_hash, created_at, last_seen) "
                    "VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(user_id) DO UPDATE SET last_seen = excluded.last_seen, "
                    "  ip_hash = excluded.ip_hash, "
                    "  session_id = COALESCE(web_users.session_id, excluded.session_id)",
                    (uid, session_id, ip_hash, now, now),
                )
            else:
                db.conn.execute("UPDATE web_users SET last_seen = ?, ip_hash = ? WHERE user_id = ?", (now, ip_hash, uid))

            # Tier shortcuts: unlimited never blocks; paid consumes credits.
            tier = (account or {}).get("tier") or "free"
            if tier == "unlimited":
                db.conn.execute("INSERT INTO web_usage_events (user_id, ts) VALUES (?, ?)", (uid, now))
                db.conn.commit()
                return True, "", None
            if tier == "paid":
                credits = int((account or {}).get("credits") or 0)
                if credits > 0:
                    db.conn.execute("UPDATE web_users SET credits = credits - 1 WHERE user_id = ?", (uid,))
                    db.conn.execute("INSERT INTO web_usage_events (user_id, ts) VALUES (?, ?)", (uid, now))
                    db.conn.commit()
                    return True, "", None
                # Out of credits → paywall again
                db.conn.commit()
                return False, "You're out of messages. Buy another pack to continue.", self._paywall_payload()

            # Free tier: rolling windows + IP backstop.
            db.conn.execute("DELETE FROM web_usage_events WHERE ts < ?", (week,))
            q = self._quota_status(db, uid, ip_hash)

            global_caps = (
                (q["g5"], q["g5_lim"], "The public demo is busy right now. Try again a little later."),
                (q["gw"], q["gw_lim"], "The public demo hit its weekly budget. Try again later."),
            )
            for used, limit, msg in global_caps:
                if limit > 0 and used >= limit:
                    db.conn.commit()
                    return False, msg, None  # global cap → plain error, no paywall

            personal_caps = (
                (q["s5"], q["s5_lim"], "You've hit your 5-hour demo limit."),
                (q["sw"], q["sw_lim"], "You've hit your weekly demo limit."),
                (q["ip5"], q["ip5_lim"], "This network has hit its 5-hour demo limit."),
            )
            for used, limit, msg in personal_caps:
                if limit > 0 and used >= limit:
                    db.conn.commit()
                    return False, msg, self._paywall_payload()

            db.conn.execute("INSERT INTO web_usage_events (user_id, ts) VALUES (?, ?)", (uid, now))
            db.conn.commit()
        return True, "", None

    def _paywall_payload(self) -> dict:
        return {
            "price_cents": _int(self.config.get("web_price_cents"), 299),
            "credits": _int(self.config.get("web_credit_pack_size"), 2000),
        }

    # ----- account / billing / auth -----

    def _quota_status(self, db, uid: str, ip_hash: str) -> dict:
        """All counters + limits the free tier needs. Single source of truth."""
        now = time.time()
        five_h, week = now - 5 * 3600, now - 7 * 24 * 3600
        g5 = db.conn.execute("SELECT COUNT(*) AS n FROM web_usage_events WHERE ts >= ?", (five_h,)).fetchone()["n"]
        gw = db.conn.execute("SELECT COUNT(*) AS n FROM web_usage_events WHERE ts >= ?", (week,)).fetchone()["n"]
        s5 = db.conn.execute("SELECT COUNT(*) AS n FROM web_usage_events WHERE user_id = ? AND ts >= ?", (uid, five_h)).fetchone()["n"]
        sw = db.conn.execute("SELECT COUNT(*) AS n FROM web_usage_events WHERE user_id = ? AND ts >= ?", (uid, week)).fetchone()["n"]
        ip5 = 0
        if ip_hash:
            ip5 = db.conn.execute(
                "SELECT COUNT(*) AS n FROM web_usage_events e "
                "JOIN web_users u ON u.user_id = e.user_id "
                "WHERE u.ip_hash = ? AND e.ts >= ?",
                (ip_hash, five_h),
            ).fetchone()["n"]
        return {
            "g5": g5, "gw": gw, "s5": s5, "sw": sw, "ip5": ip5,
            "g5_lim": _int(self.config.get("web_global_5h_turn_limit"), 600),
            "gw_lim": _int(self.config.get("web_global_week_turn_limit"), 6000),
            "s5_lim": _int(self.config.get("web_session_5h_turn_limit"), 40),
            "sw_lim": _int(self.config.get("web_session_week_turn_limit"), 160),
            "ip5_lim": _int(self.config.get("web_ip_5h_turn_limit"), 60),
        }

    def _free_tier_remaining(self, db, uid: str, ip_hash: str) -> int:
        """Personal cap remaining for the free tier — min of (session-5h, session-week, IP-5h)."""
        q = self._quota_status(db, uid, ip_hash)
        return max(0, min(q["s5_lim"] - q["s5"], q["sw_lim"] - q["sw"], q["ip5_lim"] - q["ip5"]))

    def _next_refill_seconds(self, db, uid: str) -> int | None:
        """Free tier: seconds until the oldest event in the 5h window expires (freeing one slot).
        None if the user has no events in the window (nothing to refill)."""
        now = time.time()
        five_h_ago = now - 5 * 3600
        row = db.conn.execute(
            "SELECT MIN(ts) AS oldest FROM web_usage_events WHERE user_id = ? AND ts >= ?",
            (uid, five_h_ago),
        ).fetchone()
        if not row or row["oldest"] is None:
            return None
        return max(0, int(row["oldest"] + 5 * 3600 - now))

    def _account_snapshot(self, session_id: str, ip: str, account_id: str) -> dict:
        """Single-field quota surface: `messages_remaining` is an int, or null for unlimited.
        `messages_max` is the upper bound (free → session 5h limit, paid → credit pack size,
        unlimited → null). `next_refill_seconds` is populated only for free users at 0
        remaining so the out-of-messages UI can show a wait time."""
        db = getattr(self.runtime, "db", None)
        free_max = _int(self.config.get("web_session_5h_turn_limit"), 40)
        pack_max = _int(self.config.get("web_credit_pack_size"), 2000)
        if db is None:
            return {"signed_in": False, "tier": "free", "messages_remaining": None, "messages_max": free_max, "next_refill_seconds": None}
        ip_hash = self._ip_hash(ip)
        with db.lock:
            row = None
            if account_id:
                row = db.conn.execute(
                    "SELECT user_id, email, tier, credits FROM web_users WHERE account_id = ?",
                    (account_id,),
                ).fetchone()
            if row:
                tier = row["tier"] or "free"
                refill: int | None = None
                if tier == "unlimited":
                    remaining: int | None = None
                    max_val: int | None = None
                elif tier == "paid":
                    remaining = int(row["credits"] or 0)
                    max_val = pack_max
                else:
                    remaining = self._free_tier_remaining(db, row["user_id"], ip_hash)
                    max_val = free_max
                    if remaining == 0:
                        refill = self._next_refill_seconds(db, row["user_id"])
                return {"signed_in": True, "email": row["email"], "tier": tier, "messages_remaining": remaining, "messages_max": max_val, "next_refill_seconds": refill}
            anon_uid = self._anon_user_id(session_id, ip)
            remaining = self._free_tier_remaining(db, anon_uid, ip_hash)
            refill = self._next_refill_seconds(db, anon_uid) if remaining == 0 else None
        return {"signed_in": False, "tier": "free", "messages_remaining": remaining, "messages_max": free_max, "next_refill_seconds": refill}

    def account_info(self, session_id: str, ip: str, account_id: str) -> dict:
        return self._account_snapshot(session_id, ip, account_id)

    def _base_url(self) -> str:
        return str(self.config.get("app_base_url") or "http://127.0.0.1:8765").rstrip("/")

    def create_checkout(self, session_id: str, account_id: str, ip: str) -> dict:
        """Mint a Stripe Checkout Session and return its URL."""
        base = self._base_url()
        claim_token = secrets.token_urlsafe(24)
        # Stash the claim token so /auth/claim can exchange it after Stripe redirects back.
        # Reuse web_auth_tokens but with a synthetic "pending-checkout" email.
        db = getattr(self.runtime, "db", None)
        if db is not None:
            with db.lock:
                db.conn.execute(
                    "INSERT INTO web_auth_tokens (token, email, created_at, used_at) VALUES (?, ?, ?, NULL)",
                    (claim_token, "__pending_checkout__", time.time()),
                )
                db.conn.commit()
        email_hint = None
        snap = self._account_snapshot(session_id, ip, account_id)
        if snap.get("email"):
            email_hint = snap["email"]
        meta = {
            "session_id": session_id,
            "account_id": account_id or "",
            "claim_token": claim_token,
            "anon_user_id": self._anon_user_id(session_id, ip),
            "ip_hash": self._ip_hash(ip),
        }
        result = stripe_client.create_checkout_session(
            secret_key=str(self.config.get("stripe_secret_key") or ""),
            price_id=str(self.config.get("stripe_price_id") or ""),
            success_url=f"{base}/?checkout=success&claim={claim_token}",
            cancel_url=f"{base}/?checkout=cancel",
            email_hint=email_hint,
            metadata=meta,
        )
        return result

    def handle_stripe_webhook(self, payload: bytes, sig_header: str) -> dict:
        secret_key = str(self.config.get("stripe_secret_key") or "")
        webhook_secret = str(self.config.get("stripe_webhook_secret") or "")
        event = stripe_client.verify_webhook(secret_key, webhook_secret, payload, sig_header)
        event_id = event.get("id")
        etype = event.get("type")
        db = getattr(self.runtime, "db", None)
        if db is None:
            return {"ok": True, "ignored": "no_db"}
        if etype != "checkout.session.completed":
            return {"ok": True, "ignored": etype}
        data = (event.get("data") or {}).get("object") or {}
        meta = data.get("metadata") or {}
        email = web_auth.normalize_email(data.get("customer_email") or (data.get("customer_details") or {}).get("email") or "")
        amount_cents = int(data.get("amount_total") or 0)
        credits_pack = _int(self.config.get("web_credit_pack_size"), 2000)
        if not email:
            logger.warning("[stripe] checkout.session.completed missing email; event=%s", event_id)
            return {"ok": True, "ignored": "no_email"}

        with db.lock:
            # Idempotency: if we already processed this event, bail.
            try:
                db.conn.execute(
                    "INSERT INTO web_payments (stripe_event_id, email, amount_cents, credits_granted, ts) VALUES (?, ?, ?, ?, ?)",
                    (event_id, email, amount_cents, credits_pack, time.time()),
                )
            except Exception:
                db.conn.rollback()
                return {"ok": True, "duplicate": True}

            row = db.conn.execute("SELECT user_id, account_id, tier, credits FROM web_users WHERE email = ?", (email,)).fetchone()
            if row:
                aid = row["account_id"] or str(uuid.uuid4())
                new_tier = "unlimited" if (row["tier"] == "unlimited") else "paid"
                db.conn.execute(
                    "UPDATE web_users SET account_id = ?, tier = ?, credits = COALESCE(credits, 0) + ? WHERE user_id = ?",
                    (aid, new_tier, credits_pack, row["user_id"]),
                )
            else:
                # Promote the buyer's anonymous row (from metadata) into an account,
                # or create a fresh one if we can't find theirs.
                anon_uid = meta.get("anon_user_id") or ""
                aid = str(uuid.uuid4())
                upgraded = False
                if anon_uid:
                    cur = db.conn.execute(
                        "UPDATE web_users SET email = ?, account_id = ?, tier = 'paid', credits = COALESCE(credits, 0) + ? "
                        "WHERE user_id = ? AND email IS NULL",
                        (email, aid, credits_pack, anon_uid),
                    )
                    upgraded = (cur.rowcount or 0) > 0
                if not upgraded:
                    new_uid = str(uuid.uuid4())
                    db.conn.execute(
                        "INSERT INTO web_users (user_id, session_id, ip_hash, created_at, last_seen, tier, credits, email, account_id) "
                        "VALUES (?, ?, ?, ?, ?, 'paid', ?, ?, ?)",
                        (new_uid, meta.get("session_id") or "", meta.get("ip_hash") or "", time.time(), time.time(), credits_pack, email, aid),
                    )

            # Bind the claim_token to this email so /auth/claim can pick it up.
            claim_token = meta.get("claim_token") or ""
            if claim_token:
                db.conn.execute("UPDATE web_auth_tokens SET email = ? WHERE token = ? AND used_at IS NULL", (email, claim_token))
            db.conn.commit()

        # Fire-and-forget magic-link email so the user can return on any device.
        try:
            link_token = web_auth.mint_token(db, email)
            link = f"{self._base_url()}/auth/verify?token={link_token}"
            gmail = getattr(self.runtime, "services", {}).get("gmail")
            if gmail:
                web_auth.send_magic_link(gmail, email, link, from_address=str(self.config.get("web_email_from") or "me"))
            else:
                logger.warning("[stripe] No gmail service; magic link for %s: %s", email, link)
        except Exception:
            logger.exception("[stripe] failed to send post-purchase magic link")

        return {"ok": True}

    def claim_checkout(self, claim_token: str) -> str | None:
        """Exchange a post-checkout claim token for the buyer's account_id.
        Returns account_id or None."""
        db = getattr(self.runtime, "db", None)
        if db is None or not claim_token:
            return None
        with db.lock:
            row = db.conn.execute(
                "SELECT email, used_at, created_at FROM web_auth_tokens WHERE token = ?",
                (claim_token,),
            ).fetchone()
            if not row or row["used_at"] is not None:
                return None
            email = row["email"]
            if not email or email == "__pending_checkout__":
                # Webhook hasn't been processed yet.
                return None
            db.conn.execute("UPDATE web_auth_tokens SET used_at = ? WHERE token = ?", (time.time(), claim_token))
            acct = db.conn.execute("SELECT account_id FROM web_users WHERE email = ?", (email,)).fetchone()
            db.conn.commit()
            return acct["account_id"] if acct else None

    def request_magic_link(self, email: str) -> dict:
        email = web_auth.normalize_email(email)
        if not web_auth.is_email(email):
            return {"ok": False, "error": "Please enter a valid email address."}
        db = getattr(self.runtime, "db", None)
        if db is None:
            return {"ok": False, "error": "Server not ready."}
        token = web_auth.mint_token(db, email)
        link = f"{self._base_url()}/auth/verify?token={token}"
        gmail = getattr(self.runtime, "services", {}).get("gmail")
        if gmail is None:
            logger.warning("[auth] No gmail service; magic link for %s: %s", email, link)
            return {"ok": True, "delivered": False}
        sent = web_auth.send_magic_link(gmail, email, link, from_address=str(self.config.get("web_email_from") or "me"))
        return {"ok": True, "delivered": bool(sent)}

    def verify_magic_link(self, token: str) -> str | None:
        """Returns account_id on success, or None.
        If the email doesn't yet have an account row, create one (free tier)."""
        db = getattr(self.runtime, "db", None)
        if db is None:
            return None
        email = web_auth.verify_token(db, token)
        if not email or email == "__pending_checkout__":
            return None
        with db.lock:
            row = db.conn.execute("SELECT user_id, account_id FROM web_users WHERE email = ?", (email,)).fetchone()
            if row and row["account_id"]:
                return row["account_id"]
            aid = str(uuid.uuid4())
            if row:
                db.conn.execute("UPDATE web_users SET account_id = ? WHERE user_id = ?", (aid, row["user_id"]))
            else:
                new_uid = str(uuid.uuid4())
                db.conn.execute(
                    "INSERT INTO web_users (user_id, session_id, ip_hash, created_at, last_seen, tier, credits, email, account_id) "
                    "VALUES (?, '', '', ?, ?, 'free', 0, ?, ?)",
                    (new_uid, time.time(), time.time(), email, aid),
                )
            db.conn.commit()
            return aid

    def redeem_promo(self, code: str, account_id: str) -> dict:
        db = getattr(self.runtime, "db", None)
        if db is None:
            return {"ok": False, "error": "Server not ready."}
        code = (code or "").strip()
        if not code:
            return {"ok": False, "error": "Enter a code."}
        if not account_id:
            return {"ok": False, "error": "Sign in first to redeem a promo code.", "need_auth": True}
        with db.lock:
            row = db.conn.execute("SELECT code, kind, credits, max_uses, uses FROM web_promo_codes WHERE code = ?", (code,)).fetchone()
            if not row:
                return {"ok": False, "error": "Code not found."}
            if (row["uses"] or 0) >= (row["max_uses"] or 1):
                return {"ok": False, "error": "This code has already been used."}
            user = db.conn.execute("SELECT user_id, tier, credits FROM web_users WHERE account_id = ?", (account_id,)).fetchone()
            if not user:
                return {"ok": False, "error": "Account not found."}
            if row["kind"] == "unlimited":
                db.conn.execute("UPDATE web_users SET tier = 'unlimited' WHERE user_id = ?", (user["user_id"],))
                granted = "unlimited"
            elif row["kind"] == "credits":
                amt = int(row["credits"] or 0)
                new_tier = "paid" if user["tier"] != "unlimited" else "unlimited"
                db.conn.execute(
                    "UPDATE web_users SET credits = COALESCE(credits, 0) + ?, tier = ? WHERE user_id = ?",
                    (amt, new_tier, user["user_id"]),
                )
                granted = f"{amt} credits"
            else:
                return {"ok": False, "error": "Unknown code kind."}
            db.conn.execute("UPDATE web_promo_codes SET uses = uses + 1 WHERE code = ?", (code,))
            db.conn.commit()
        return {"ok": True, "granted": granted}

    def delete_account(self, account_id: str, session_id: str, confirm_email: str) -> dict:
        """Permanent erasure: account row, auth tokens, payment history, private
        archive, and conversations on the current session. Skills and shared
        gallery items are kept but author/owner is anonymized."""
        db = getattr(self.runtime, "db", None)
        if db is None:
            return {"ok": False, "error": "Server not ready."}
        if not account_id:
            return {"ok": False, "error": "Not signed in."}
        typed = (confirm_email or "").strip().lower()
        if not typed:
            return {"ok": False, "error": "Type your email to confirm."}
        with db.lock:
            row = db.conn.execute(
                "SELECT user_id, email FROM web_users WHERE account_id = ?",
                (account_id,),
            ).fetchone()
        if not row:
            return {"ok": False, "error": "Account not found."}
        email = str(row["email"] or "").strip()
        if email.lower() != typed:
            return {"ok": False, "error": "Email doesn't match the account."}

        # Conversations bound to the current browser session.
        key = self.session_key(session_id)
        try:
            session = self.runtime.sessions.get(key)
            cid = getattr(session, "conversation_id", None) if session else None
            if cid is not None:
                try:
                    db.delete_conversation(cid)
                except Exception:
                    logger.exception("delete_account: delete_conversation failed")
            with self._lock:
                self._outbox.pop(key, None)
            if session is not None:
                try:
                    self.runtime.sessions.pop(key, None)
                except Exception:
                    pass
        except Exception:
            logger.exception("delete_account: session teardown failed")

        # Anonymize skill authorship for this account.
        owner_values = {email, account_id}
        try:
            skills_changed = _anonymize_skill_owner(owner_values)
        except Exception:
            logger.exception("delete_account: skill anonymize failed")
            skills_changed = 0

        # Drop DB rows last so the email/account_id are still available above.
        # NOTE: user_canvas_actions rows (share/save/download/remix) keyed
        # under this account's user_id are NOT currently scrubbed — see
        # the gap note in the migration summary.
        with db.lock:
            db.conn.execute("DELETE FROM web_auth_tokens WHERE email = ?", (email,))
            db.conn.execute("DELETE FROM web_payments WHERE email = ?", (email,))
            db.conn.execute("DELETE FROM web_users WHERE account_id = ?", (account_id,))
            db.conn.commit()

        logger.info("Account deleted (skills_anonymized=%s)", skills_changed)
        return {"ok": True}

    def _owner_id(self, session_id: str, ip: str, account_id: str) -> str:
        """Stable identity for the saved archive: account_id when signed in,
        otherwise the anonymous fallback derived from session+IP."""
        return account_id or self._anon_user_id(session_id, ip)

    def save_canvas(self, session_id: str, ip: str, account_id: str, title: str = "") -> list[dict]:
        """Record that this user saved the current canvas (pool-based).

        No file copy. The canvas state already lives in canvas_pools; the
        image already lives in canvas_renders/. "Save" is purely the act of
        adding it to this user's collection (user_canvas_actions row), and
        bumping skill popularity scores.
        """
        key = self.session_key(session_id)
        cr = self._canvas_runtime()
        db = getattr(self.runtime, "db", None)
        if cr is None or db is None:
            return [{"type": "error", "content": "Save failed: canvas runtime unavailable."}]
        cs = cr.for_session(key)
        if not cs.canvas.layers:
            return [{"type": "error", "content": "Nothing to save yet — make something first."}]
        snap = self._new_canvas_snap(key) or {}
        if not snap.get("path"):
            return [{"type": "error", "content": "Save failed: render produced no image."}]
        ph = snap.get("pool_hash") or _pool_hash(cs.canvas)
        owner = self._owner_id(session_id, ip, account_id)
        canvas_actions.record_user_action(
            db, user_id=owner, pool_hash=ph, action="save",
            layers=cs.canvas.layers, image_path=snap["path"],
            meta={"title": title} if title else None,
        )
        label = f'"{title}"' if title else "canvas"
        return [
            {"type": "saved", "item": {"pool_hash": ph, "url": snap.get("path") and _file_url(Path(snap["path"]))}},
            {"type": "status", "content": f"Saved {label} to your collection."},
        ]

    def history(self, session_id: str) -> list[dict]:
        """Return user/assistant text messages for this session, oldest first.

        Lets the client rehydrate the chat view on page (re)load so navigating
        away and back doesn't drop the transcript.
        """
        key = self.session_key(session_id)
        session = self.runtime.sessions.get(key)
        if session is None:
            return []
        from runtime.token_stripper import strip_model_tokens
        out: list[dict] = []
        for msg in session.history or []:
            role = msg.get("role")
            content = msg.get("content")
            if role not in ("user", "assistant") or not isinstance(content, str):
                continue
            if role == "assistant":
                content = strip_model_tokens(content)[0]
            if not content.strip():
                continue
            out.append({"role": role, "content": content})
        return out

    def cancel(self, session_id: str) -> list[dict]:
        """Set the cancel flag on the in-flight agent turn for this session."""
        key = self.session_key(session_id)
        if key not in self.runtime.sessions:
            return []
        result = self.runtime.cancel_session(key)
        events = self._drain(key)
        if result and getattr(result, "messages", None):
            for msg in result.messages:
                events.append({"type": "status", "content": msg})
        return events

    def canvas_payload(self, session_id: str) -> dict:
        """Return the current canvas (new system), rendering on-demand if needed."""
        key = self.session_key(session_id)
        snap = self._new_canvas_snap(key) or {}
        return _canvas_payload_full(self.runtime, key, snap)

    def palettes_payload(self) -> list[dict]:
        return [p.to_dict() for p in list_palettes()]

    def set_palette(self, session_id: str, palette_id: str) -> list[dict]:
        key = self.session_key(session_id)
        return self._new_canvas_action_events(
            key, "set_palette", {"palette_id": palette_id}, fail_prefix="Palette replay failed",
        )

    def set_skill_control(self, session_id: str, chain_index: int, name: str, value, _action: str = "") -> list[dict]:
        """Update one control on a chain entry, then re-render.

        The legacy ``_action`` parameter (e.g. "randomize") is intentionally
        unused under the new canvas system; reseed-on-control belongs in a
        future renderer hook, not here. Kept in the signature so the HTTP
        handler can keep passing it positionally without changes.
        """
        del _action  # intentionally unused; see docstring
        key = self.session_key(session_id)
        return self._new_canvas_action_events(
            key, "set_control",
            {"chain_index": chain_index, "name": name, "value": value},
            fail_prefix="Control replay failed",
        )

    def delete_layer(self, session_id: str, chain_index: int) -> list[dict]:
        """Remove one entry from the chain and re-render. Deleting index 0 clears the canvas."""
        key = self.session_key(session_id)
        events = self._new_canvas_action_events(
            key, "remove_layer", {"chain_index": chain_index},
            fail_prefix="Delete layer failed",
        )
        # If the deletion left an empty chain, surface the canvas_reset event
        # the UI expects instead of a hero_image-less response.
        if events and events[0].get("type") == "hero_image" and not (events[0].get("canvas") or {}).get("path"):
            return [{"type": "canvas_reset"}]
        return events

    def share(self, session_id: str, title: str, artist: str, *, ip: str = "", account_id: str = "") -> list[dict]:
        """Share the user's current canvas.

        Pool-hash model: the canvas is already in canvas_pools (written on
        render). We just record the share action + return a URL pointing at
        /share/{pool_hash}. No file copy, no separate share_links row.
        """
        key = self.session_key(session_id)
        cr = self._canvas_runtime()
        db = getattr(self.runtime, "db", None)
        if cr is None or db is None:
            return [{"type": "error", "content": "Share failed: canvas runtime unavailable."}]
        cs = cr.for_session(key)
        if not cs.canvas.layers:
            return [{"type": "error", "content": "Nothing to share yet — make something first."}]
        # Ensure a render exists so canvas_pools has the row.
        snap = self._new_canvas_snap(key) or {}
        if not snap.get("path"):
            return [{"type": "error", "content": "Share failed: render produced no image."}]
        ph = snap.get("pool_hash") or _pool_hash(cs.canvas)
        owner = self._owner_id(session_id, ip, account_id)
        canvas_actions.record_user_action(
            db, user_id=owner, pool_hash=ph, action="share",
            layers=cs.canvas.layers, image_path=snap["path"],
            meta={"title": title, "artist": artist},
        )
        share_url = f"{self._base_url()}/share/{ph}"
        qr_url = f"/share/{ph}/qr.png"
        return [
            {"type": "share_link", "share_id": ph, "url": share_url, "qr_url": qr_url, "kind": "pool"},
            {"type": "message", "role": "assistant",
             "content": f'Shared "{title}" by {artist}.' if title else "Shared."},
        ]

    def remix(self, session_id: str, pool_hash: str = "", share_id: str = "", path: str = "", **_unused) -> list[dict]:
        """Open a remix of another canvas in the current session.

        ``pool_hash`` is the canonical key. ``share_id`` and ``path`` are
        accepted as aliases so existing UI callers (share-deep-link
        handler, gallery item buttons) keep working — they all carry the
        same value (the pool_hash) under different names.
        """
        key = self.session_key(session_id)
        ph = pool_hash or share_id or path
        if not ph:
            raise ValueError("remix requires a pool_hash (or share_id / path alias).")
        return self._remix_from_pool(key, ph)

    def _remix_from_pool(self, key: str, pool_hash: str) -> list[dict]:
        """Pool-hash remix: clone canvas_pools entry into a new canvas_id."""
        cr = self._canvas_runtime()
        db = getattr(self.runtime, "db", None)
        if cr is None or db is None:
            return [{"type": "error", "content": "Remix failed: canvas runtime unavailable."}]
        new_cs = cr.remix(pool_hash)
        if new_cs is None:
            return [{"type": "error", "content": "That canvas link is invalid or no longer available."}]
        cr.bind_session(key, new_cs.canvas_id)
        snap = self._new_canvas_snap(key) or {}
        # Record against the SOURCE pool_hash so popularity attributes to
        # the original look, not the user's fresh editing handle.
        owner = self._anon_user_id(key, "")
        canvas_actions.record_user_action(
            db, user_id=owner, pool_hash=pool_hash, action="remix",
            layers=new_cs.canvas.layers, image_path=snap.get("path"),
        )
        img_url = _file_url(Path(snap["path"])) if snap.get("path") else None
        return [
            {"type": "hero_image", "url": img_url,
             "name": Path(snap["path"]).name if snap.get("path") else None,
             "canvas": _canvas_payload_full(self.runtime, key, snap)},
            {"type": "message", "role": "assistant",
             "content": "Remix loaded. Tell me how to mutate it."},
        ]

    def download(self, session_id: str) -> list[dict]:
        """Fire-and-forget signal: user clicked Download for the current canvas."""
        key = self.session_key(session_id)
        cr = self._canvas_runtime()
        db = getattr(self.runtime, "db", None)
        if cr is None or db is None:
            return []
        cs = cr.for_session(key)
        if not cs.canvas.layers:
            return []
        snap = self._new_canvas_snap(key) or {}
        ph = snap.get("pool_hash") or _pool_hash(cs.canvas)
        # Use the anonymous user_id; downloads typically aren't gated by login.
        owner = self._anon_user_id(session_id, "")
        canvas_actions.record_user_action(
            db, user_id=owner, pool_hash=ph, action="download",
            layers=cs.canvas.layers, image_path=snap.get("path"),
        )
        return []

    def regenerate(self, session_id: str) -> list[dict]:
        """Re-render the current chain with a fresh seed."""
        key = self.session_key(session_id)
        # Record the intent on the canvas state machine (history event), then
        # render with force_new_seed=True so the renderer mints a fresh seed
        # and writes a new file into the pool folder.
        cr = self._canvas_runtime()
        if cr is None:
            return [{"type": "error", "content": "Regenerate failed: canvas runtime not available"}]
        cs = cr.for_session(key)
        if not cs.canvas.layers:
            return []
        cr.handle_action(cs.canvas_id, "regenerate", {})
        snap = self._new_canvas_snap(key, force_new_seed=True)
        if not snap or not snap.get("path"):
            return [{"type": "error", "content": "Regenerate failed: render produced no image"}]
        return [{
            "type": "hero_image",
            "url": _file_url(Path(snap["path"])),
            "name": Path(snap["path"]).name,
            "canvas": _canvas_payload_full(self.runtime, key, snap),
        }]

    def pool_share_payload(self, pool_hash: str) -> dict | None:
        """Resolve a pool_hash to the data needed to render its share page.

        Returns ``{"pool_hash", "state", "image_path", "layers"}`` or None
        if the pool_hash isn't in ``canvas_pools`` (i.e. nothing was ever
        rendered for it).
        """
        from canvas import persistence as canvas_persistence
        from canvas.render import existing_seeds, folder_for
        from canvas.canvas import Canvas

        db = getattr(self.runtime, "db", None)
        if db is None:
            return None
        state = canvas_persistence.load_pool(db, pool_hash)
        if state is None:
            return None
        # Build a transient Canvas just to ask the renderer for the folder.
        snap_canvas = Canvas.from_dict(state)
        seeds = existing_seeds(snap_canvas)
        if not seeds:
            return None
        # Pick the most-recently-rendered file (stable "current" thumbnail).
        folder = folder_for(snap_canvas)
        files = sorted(folder.glob("*.webp"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            return None
        return {
            "pool_hash": pool_hash,
            "state": state,
            "image_path": str(files[0]),
            "layers": list(state.get("layers") or []),
        }

    def record_link_open(self, pool_hash: str, ip: str = "", account_id: str = "", payload: dict | None = None) -> None:
        """Count a public share-page view as a pool-scored skill signal."""
        db = getattr(self.runtime, "db", None)
        payload = payload or self.pool_share_payload(pool_hash)
        if db is None or not payload:
            return
        canvas_actions.record_user_action(
            db, user_id=self._owner_id(f"share:{pool_hash}", ip, account_id),
            pool_hash=pool_hash, action="link_open",
            layers=payload.get("layers") or [], image_path=payload.get("image_path"),
        )

    # ── pool-hash listings, share links, QR codes ─────────────────────

    def get_link(self, session_id: str, ip: str, account_id: str,
                 kind: str = "current", path: str = "",
                 title: str = "", artist: str = "") -> dict:
        """Return a /share/{pool_hash} URL for the current canvas or for a
        ``pool_hash`` passed via ``path``. Does NOT record a share action —
        get_link is "give me the URL", not "share". Saving a share goes
        through ``/api/share``.

        Title/artist are accepted for backward compat with the legacy UI
        signature but are no longer attached at link-mint time; they're
        only meaningful when an actual share action is recorded.
        """
        del title, artist  # ignored under the pool-hash model
        cr = self._canvas_runtime()
        db = getattr(self.runtime, "db", None)
        if cr is None or db is None:
            raise RuntimeError("share links require the canvas runtime and DB")
        if kind == "current":
            key = self.session_key(session_id)
            cs = cr.for_session(key)
            if not cs.canvas.layers:
                raise ValueError("Nothing to share yet — make something first.")
            snap = self._new_canvas_snap(key) or {}
            ph = snap.get("pool_hash") or _pool_hash(cs.canvas)
        elif kind in ("gallery", "archive", "pool"):
            ph = (path or "").strip()
            if not ph:
                raise ValueError("get_link requires a pool_hash via 'path'.")
        else:
            raise ValueError(f"unknown link kind: {kind!r}")
        del account_id, ip  # currently unused; kept for signature compat
        base = self._base_url()
        return {
            "share_id": ph,
            "url": f"{base}/share/{ph}",
            "qr_url": f"/share/{ph}/qr.png",
            "kind": kind,
        }

    def share_qr_png(self, pool_hash: str) -> bytes | None:
        """Render a PNG QR code for the canvas's share URL, or None if unknown."""
        if not self.pool_share_payload(pool_hash):
            return None
        try:
            import io
            import qrcode
            url = f"{self._base_url()}/share/{pool_hash}"
            img = qrcode.make(url)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except Exception:
            logger.exception("share_qr_png failed for pool_hash=%s", pool_hash)
            return None

    def gallery(self, session_id: str, limit: int = 24, offset: int = 0) -> dict:
        """Publicly shared canvases, newest first.

        Backed by user_canvas_actions: every "share" action is a vote.
        We deduplicate by pool_hash and present each unique configuration
        once, with metadata (title/artist) from its most recent share row.
        """
        del session_id  # gallery is global, not per-session
        db = getattr(self.runtime, "db", None)
        if db is None:
            return {"items": [], "total": 0}
        return self._listing(db, action="share", limit=limit, offset=offset, owner=None)

    def archive_listing(self, session_id: str, ip: str, account_id: str,
                        limit: int = 24, offset: int = 0) -> dict:
        """Canvases this user has saved (i.e. added to their own collection)."""
        owner = self._owner_id(session_id, ip, account_id)
        db = getattr(self.runtime, "db", None)
        if db is None:
            return {"items": [], "total": 0}
        return self._listing(db, action="save", limit=limit, offset=offset, owner=owner)

    def _listing(self, db, *, action: str, limit: int, offset: int,
                 owner: str | None) -> dict:
        """Shared implementation for gallery / archive listings."""
        owner_clause = "AND user_id = ?" if owner else ""
        sql_count = (
            f"SELECT COUNT(DISTINCT pool_hash) AS n FROM user_canvas_actions "
            f"WHERE action = ? {owner_clause}"
        )
        # SQL placeholder order is (action, [owner]). Match it exactly.
        sql_count_params: tuple = (action, owner) if owner else (action,)
        # Sub-query picks the most recent row per pool_hash so meta_json
        # follows the latest user-supplied title/artist for that look.
        sql_page = f"""
            SELECT a.pool_hash, a.meta_json, a.ts
              FROM user_canvas_actions a
              JOIN (
                SELECT pool_hash, MAX(ts) AS max_ts FROM user_canvas_actions
                 WHERE action = ? {owner_clause}
                 GROUP BY pool_hash
              ) latest
                ON latest.pool_hash = a.pool_hash AND latest.max_ts = a.ts
             WHERE a.action = ? {owner_clause}
             ORDER BY a.ts DESC
             LIMIT ? OFFSET ?
        """
        page_params = (
            ((action, owner) if owner else (action,))
            + ((action, owner) if owner else (action,))
            + (limit, offset)
        )
        with db.lock:
            total = int(db.conn.execute(sql_count, sql_count_params).fetchone()["n"] or 0)
            rows = db.conn.execute(sql_page, page_params).fetchall()
        items: list[dict] = []
        for row in rows:
            ph = row["pool_hash"]
            payload = self.pool_share_payload(ph)
            if payload is None:
                continue
            meta = {}
            try:
                if row["meta_json"]:
                    meta = json.loads(row["meta_json"])
            except (TypeError, ValueError):
                meta = {}
            items.append({
                "pool_hash": ph,
                "path": ph,  # JS uses `path` as the remix identifier
                "url": _file_url(Path(payload["image_path"])),
                "title": str(meta.get("title") or "untitled"),
                "artist": str(meta.get("artist") or "anonymous"),
            })
        return {"items": items, "total": total}

    # ── new canvas system: helpers ────────────────────────────────────

    def _canvas_runtime(self):
        """Return the new CanvasRuntime, or None if not wired."""
        services = getattr(self.runtime, "services", None) or {}
        return services.get("canvas")

    def _new_canvas_snap(self, session_key: str, *, force_new_seed: bool = False) -> dict | None:
        """Build the frontend canvas-payload from the new CanvasRuntime.

        Renders on demand. If the chain is empty, returns an empty-canvas
        shape (no render). If the pool already has a cached render, this is
        essentially free.
        """
        cr = self._canvas_runtime()
        if cr is None:
            return None
        cs = cr.for_session(session_key)
        if not cs.canvas.layers:
            return {
                "path": None,
                "chain": [],
                "size": cs.canvas.size,
                "palette_id": cs.canvas.palette_id,
                "canvas_id": cs.canvas_id,
            }
        skill_registry = getattr(self.runtime, "skill_registry", None)
        if skill_registry is None:
            return None
        try:
            rr = _new_render_canvas(
                cs,
                skill_loader=skill_registry.get_record,
                force_new_seed=force_new_seed,
                db=getattr(self.runtime, "db", None),
            )
        except Exception as e:
            logger.exception("new canvas render failed for session=%s", session_key)
            return {"path": None, "chain": list(cs.canvas.layers), "error": str(e),
                    "size": cs.canvas.size, "palette_id": cs.canvas.palette_id, "canvas_id": cs.canvas_id}
        return {
            "path": str(rr.image_path),
            "chain": list(cs.canvas.layers),
            "size": cs.canvas.size,
            "palette_id": cs.canvas.palette_id,
            "canvas_id": cs.canvas_id,
            "pool_hash": rr.pool_hash,
            "seed": rr.seed,
            "cache_hit": rr.cache_hit,
        }

    def _new_canvas_action_events(self, key: str, action_type: str, payload: dict, *, fail_prefix: str) -> list[dict]:
        """Mutate the new canvas, render, return [{type:hero_image|error}] events."""
        cr = self._canvas_runtime()
        if cr is None:
            return [{"type": "error", "content": f"{fail_prefix}: canvas runtime not available"}]
        cs = cr.for_session(key)
        try:
            result = cr.handle_action(cs.canvas_id, action_type, dict(payload))
        except Exception as e:
            logger.exception("%s failed", fail_prefix)
            return [{"type": "error", "content": f"{fail_prefix}: {e}"}]
        if not getattr(result, "ok", True):
            err = getattr(result, "error", None)
            msg = err.message if err is not None else (result.message or fail_prefix)
            return [{"type": "error", "content": f"{fail_prefix}: {msg}"}]
        snap = self._new_canvas_snap(key) or {}
        if "error" in snap:
            return [{"type": "error", "content": f"{fail_prefix}: {snap['error']}"}]
        if not snap.get("path"):
            # Empty chain after the action (e.g. remove_layer on the last layer).
            return [{
                "type": "hero_image",
                "url": None,
                "name": None,
                "canvas": _canvas_payload_full(self.runtime, key, snap),
            }]
        return [{
            "type": "hero_image",
            "url": _file_url(Path(snap["path"])),
            "name": Path(snap["path"]).name,
            "canvas": _canvas_payload_full(self.runtime, key, snap),
        }]

class _Server(ThreadingHTTPServer):
    def __init__(self, addr, handler, frontend):
        super().__init__(addr, handler)
        self.frontend = frontend


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/health":
            return self._json({"ok": True})
        if path == "/api/events":
            sid = str(parse_qs(parsed.query).get("session_id", ["demo"])[0])[:80]
            return self._json({"ok": True, "events": self.server.frontend._drain(self.server.frontend.session_key(sid))})
        if path == "/api/history":
            sid = str(parse_qs(parsed.query).get("session_id", ["demo"])[0])[:80]
            return self._json({"ok": True, "history": self.server.frontend.history(sid)})
        if path == "/api/canvas":
            sid = str(parse_qs(parsed.query).get("session_id", ["demo"])[0])[:80]
            return self._json({"ok": True, "canvas": self.server.frontend.canvas_payload(sid)})
        if path == "/api/palettes":
            return self._json({"ok": True, "palettes": self.server.frontend.palettes_payload()})
        if path == "/api/gallery":
            qs = parse_qs(parsed.query); sid = str(qs.get("session_id", ["demo"])[0])[:80]
            try: limit = max(1, min(96, int(qs.get("limit", ["24"])[0])))
            except (TypeError, ValueError): limit = 24
            try: offset = max(0, int(qs.get("offset", ["0"])[0]))
            except (TypeError, ValueError): offset = 0
            res = self.server.frontend.gallery(sid, limit=limit, offset=offset)
            return self._json({"ok": True, **res})
        if path == "/api/archive":
            qs = parse_qs(parsed.query); sid = str(qs.get("session_id", ["demo"])[0])[:80]
            try: limit = max(1, min(96, int(qs.get("limit", ["24"])[0])))
            except (TypeError, ValueError): limit = 24
            try: offset = max(0, int(qs.get("offset", ["0"])[0]))
            except (TypeError, ValueError): offset = 0
            res = self.server.frontend.archive_listing(
                sid, self.client_address[0], self._cookie_uid(),
                limit=limit, offset=offset,
            )
            return self._json({"ok": True, **res})
        if path.startswith("/share/"):
            tail = path[len("/share/"):]
            share_id, _, sub = tail.partition("/")
            pool_payload = self.server.frontend.pool_share_payload(share_id) if share_id else None
            if pool_payload is None:
                return self.send_error(404)
            if sub == "":
                self.server.frontend.record_link_open(share_id, self.client_address[0], self._cookie_uid(), pool_payload)
                return self._redirect(f"/?share={quote(share_id, safe='')}")
            if sub in {"image.png", "image.webp", "image"}:
                img = pool_payload.get("image_path")
                if img is None:
                    return self.send_error(404)
                return self._raw_file(Path(img), "image/webp")
            if sub == "qr.png":
                raw = self.server.frontend.share_qr_png(share_id)
                if raw is None:
                    return self.send_error(404)
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Cache-Control", "public, max-age=86400")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
            return self.send_error(404)
        if path == "/api/account":
            qs = parse_qs(parsed.query)
            sid = str(qs.get("session_id", ["demo"])[0])[:80]
            return self._json({"ok": True, "account": self.server.frontend.account_info(sid, self.client_address[0], self._cookie_uid())})
        if path == "/files":
            qs = parse_qs(parsed.query)
            sid = str(qs.get("session_id", [self._cookie_sid()])[0])[:80]
            owner = self.server.frontend._owner_id(sid, self.client_address[0], self._cookie_uid())
            return self._local_file(qs.get("path", [""])[0], sid, owner)
        if path == "/auth/verify":
            qs = parse_qs(parsed.query)
            token = str(qs.get("token", [""])[0])
            account_id = self.server.frontend.verify_magic_link(token)
            if account_id:
                return self._redirect_with_uid("/account", account_id)
            return self._html("<!doctype html><meta charset=utf-8><title>Sign in</title><style>body{font-family:system-ui;max-width:480px;margin:80px auto;padding:0 24px;color:#222}</style><h1>Link expired</h1><p>That sign-in link is invalid or already used. Request a new one from the home page.</p><p><a href=\"/\">Back home</a></p>", 400)
        if path == "/auth/claim":
            qs = parse_qs(parsed.query)
            token = str(qs.get("token", [""])[0])
            account_id = self.server.frontend.claim_checkout(token)
            if account_id:
                return self._redirect_with_uid("/account?welcome=1", account_id)
            return self._redirect("/?checkout=pending")
        if path == "/auth/logout":
            return self._redirect_clear_uid("/")
        if path == "/favicon.ico":
            return self._raw_file(FAVICON_PATH, "image/x-icon")
        rel = "index.html" if path in {"", "/"} else path.lstrip("/")
        if rel == "account":
            rel = "account.html"
        if rel == "privacy":
            rel = "privacy.html"
        return self._file(WEB_ROOT / rel)

    def do_POST(self):
        # Stripe webhook receives raw JSON we must NOT decode before signature check.
        if self.path == "/stripe/webhook":
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length)
            sig = self.headers.get("Stripe-Signature") or ""
            try:
                result = self.server.frontend.handle_stripe_webhook(raw, sig)
                return self._json(result)
            except Exception as e:
                logger.exception("Stripe webhook failed")
                return self._json({"ok": False, "error": str(e)}, 400)
        body = self._body()
        sid = str(body.get("session_id") or "demo")[:80]
        try:
            if self.path == "/api/chat":
                events = self.server.frontend.chat(sid, str(body.get("message") or ""), self.client_address[0], self._cookie_uid())
                return self._json({"ok": True, "events": events})
            if self.path == "/api/checkout":
                try:
                    result = self.server.frontend.create_checkout(sid, self._cookie_uid(), self.client_address[0])
                    return self._json({"ok": True, **result})
                except RuntimeError as e:
                    return self._json({"ok": False, "error": str(e)}, 400)
            if self.path == "/api/auth/request":
                return self._json(self.server.frontend.request_magic_link(str(body.get("email") or "")))
            if self.path == "/api/promo/redeem":
                return self._json(self.server.frontend.redeem_promo(str(body.get("code") or ""), self._cookie_uid()))
            if self.path == "/api/account/delete":
                return self._json(self.server.frontend.delete_account(
                    self._cookie_uid(), sid, str(body.get("confirm_email") or "")))
            if self.path == "/api/new":
                return self._json({"ok": True, "events": self.server.frontend.new_chat(sid)})
            if self.path == "/api/cancel":
                return self._json({"ok": True, "events": self.server.frontend.cancel(sid)})
            if self.path == "/api/approval":
                return self._json({"ok": True, "events": self.server.frontend.approve(sid, bool(body.get("value")))})
            if self.path == "/api/share":
                return self._json({"ok": True, "events": self.server.frontend.share(
                    sid, str(body.get("title") or "untitled"), str(body.get("artist") or "anonymous"),
                    ip=self.client_address[0], account_id=self._cookie_uid())})
            if self.path == "/api/remix":
                return self._json({"ok": True, "events": self.server.frontend.remix(
                    sid,
                    pool_hash=str(body.get("pool_hash") or ""),
                    share_id=str(body.get("share_id") or ""),
                    path=str(body.get("path") or ""),
                )})
            if self.path == "/api/archive_remix":
                # Backward-compat alias: same as /api/remix; gallery card buttons
                # post here with `path` carrying the pool_hash.
                return self._json({"ok": True, "events": self.server.frontend.remix(
                    sid, path=str(body.get("path") or ""),
                )})
            if self.path == "/api/get_link":
                try:
                    res = self.server.frontend.get_link(
                        sid, self.client_address[0], self._cookie_uid(),
                        kind=str(body.get("kind") or "current"),
                        path=str(body.get("path") or ""),
                        title=str(body.get("title") or ""),
                        artist=str(body.get("artist") or ""),
                    )
                    return self._json({"ok": True, **res})
                except (ValueError, RuntimeError) as e:
                    return self._json({"ok": False, "error": str(e)}, 400)
            if self.path == "/api/save":
                return self._json({"ok": True, "events": self.server.frontend.save_canvas(sid, self.client_address[0], self._cookie_uid(), str(body.get("title") or ""))})
            if self.path == "/api/palette":
                return self._json({"ok": True, "events": self.server.frontend.set_palette(sid, str(body.get("palette_id") or ""))})
            if self.path == "/api/download":
                return self._json({"ok": True, "events": self.server.frontend.download(sid)})
            if self.path == "/api/regenerate":
                return self._json({"ok": True, "events": self.server.frontend.regenerate(sid)})
            if self.path == "/api/skill_control":
                return self._json({"ok": True, "events": self.server.frontend.set_skill_control(
                    sid,
                    int(body.get("chain_index") or 0),
                    str(body.get("name") or ""),
                    body.get("value"),
                    str(body.get("action") or ""),
                )})
            if self.path == "/api/layer_delete":
                return self._json({"ok": True, "events": self.server.frontend.delete_layer(sid, int(body.get("chain_index") or 0))})
        except Exception as e:
            logger.exception("Web request failed")
            return self._json({"ok": False, "events": [{"type": "error", "content": str(e)}]}, 500)
        self.send_error(404)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length) or b"{}")

    def _json(self, data: dict, status: int = 200):
        raw = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _file(self, path: Path):
        try:
            root, target = WEB_ROOT.resolve(), path.resolve()
            if root not in target.parents and target != root:
                raise FileNotFoundError
            raw = target.read_bytes()
        except FileNotFoundError:
            return self.send_error(404)
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(str(target))[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _cookie_sid(self) -> str:
        return self._cookie("sb_sid")[:80]

    def _cookie_uid(self) -> str:
        return self._cookie("sb_uid")[:80]

    def _cookie(self, name: str) -> str:
        raw = self.headers.get("Cookie") or ""
        for part in raw.split(";"):
            k, _, v = part.strip().partition("=")
            if k == name:
                return unquote(v)
        return ""

    def _redirect(self, location: str, *, extra_headers: list[tuple[str, str]] = ()):
        self.send_response(303)
        self.send_header("Location", location)
        for k, v in extra_headers:
            self.send_header(k, v)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _redirect_with_uid(self, location: str, account_id: str):
        cookie = f"sb_uid={quote(account_id)}; Path=/; SameSite=Lax; Max-Age=31536000; HttpOnly"
        self._redirect(location, extra_headers=[("Set-Cookie", cookie)])

    def _redirect_clear_uid(self, location: str):
        cookie = "sb_uid=; Path=/; SameSite=Lax; Max-Age=0; HttpOnly"
        self._redirect(location, extra_headers=[("Set-Cookie", cookie)])

    def _html(self, body: str, status: int = 200):
        raw = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _raw_file(self, path: Path, content_type: str):
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return self.send_error(404)
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _local_file(self, raw_path: str, session_id: str = "", owner_id: str = ""):
        path = Path(unquote(raw_path))
        if not _is_user_accessible_image(path, self.server.frontend.session_key(session_id) if session_id else "", owner_id):
            return self.send_error(404)
        raw = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(str(path))[0] or "image/png")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, fmt, *args):
        logger.debug(fmt, *args)


def _is_public_image(path: Path) -> bool:
    try:
        target, root = path.resolve(), DATA_DIR.resolve()
        return target.is_file() and target.suffix.lower() in IMAGE_EXTS and (target == root or root in target.parents)
    except Exception:
        return False


def _is_user_accessible_image(path: Path, session_key: str = "", owner_id: str = "") -> bool:
    """Any image under DATA_DIR is fair game once the canvas system owns it
    (renders live in DATA_DIR/canvas_renders/). session_key / owner_id are
    accepted for signature compat but no longer used to gate access; the
    old gallery / archive folder layouts are gone."""
    del session_key, owner_id
    return _is_public_image(path)


def _file_url(path: Path) -> str:
    # mtime cache-buster: the composite path is stable across recompositions,
    # so without this the browser keeps serving the old image.
    try:
        v = int(path.stat().st_mtime)
    except Exception:
        v = 0
    return f"/files?path={quote(str(path.resolve()), safe='')}&v={v}"


def _image_event(path: Path, canvas_payload: dict) -> dict:
    return {"type": "hero_image", "url": _file_url(path), "name": path.name, "canvas": canvas_payload}


def _canvas_payload(state: dict | None) -> dict:
    if not state:
        return {}
    if not state.get("path"):
        return {k: v for k, v in state.items() if k != "path"}
    p = Path(state["path"])
    return {**state, "url": _file_url(p), "name": p.name}


def _canvas_payload_full(runtime, session_key: str, state: dict | None) -> dict:
    """Canvas payload plus per-entry control schemas for the website panel."""
    base = _canvas_payload(state)
    if not state:
        return base
    chain = state.get("chain") or state.get("last_chain") or []
    panels = []
    layers = []
    palette_shown = False
    for idx, step in enumerate(chain):
        skill = _read_skill_via(runtime, step.get("slug") or "")
        slug = step.get("slug") or ""
        name = skill.name if skill else slug
        layers.append({"chain_index": idx, "slug": slug, "skill_name": name, "kind": step.get("kind") or (skill.kind if skill else "")})
        if not skill or not skill.controls:
            continue
        # The canvas has one palette. Keep the palette swatch on the first
        # panel that has one; strip it from later panels so chains don't
        # show a redundant swatch for every layer that touches color.
        schema = []
        for c in skill.controls:
            if c.get("type") == "palette":
                if palette_shown:
                    continue
                palette_shown = True
            schema.append(c)
        if not schema:
            continue
        values = dict(step.get("controls") or {})
        if not any(c.get("type") == "palette" for c in schema):
            values.pop("palette", None)
        panels.append({
            "chain_index": idx,
            "slug": skill.slug,
            "skill_name": skill.name,
            "schema": schema,
            "values": values,
            "seed": int(step.get("seed") or 0),
        })
    base["controls_panels"] = panels
    base["layers"] = layers
    return base


from plugins.skills.helpers.skill_controls import coerce_control_value as _coerce_control_value  # noqa: F401  (re-export for any external callers)


def _int(value, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default
