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

from plugins.helpers import share_links, stripe_client, web_auth

from PIL import Image

from canvas.render import render_canvas as _new_render_canvas
from config import config_manager
from plugins.BaseFrontend import BaseFrontend, FrontendCapabilities
from plugins.skills.helpers import skill_scoring
from plugins.helpers.palettes import get_palette, list_palettes
from plugins.skills.helpers.skill_runner import replay_chain
from plugins.skills.helpers.skill_store import anonymize_owner_in_dir
from paths import SANDBOX_SKILLS


def __anonymize_skill_owner(owner_values):
    return anonymize_owner_in_dir(SANDBOX_SKILLS, owner_values)


def _read_skill_via(runtime, slug: str):
    registry = getattr(runtime, "skill_registry", None)
    return registry.get_record(slug) if registry is not None else None
from plugins.tools.helpers import layered_canvas as lc
from plugins.helpers.gallery import GALLERY_DIR, anonymize_shared, archive_dir, archive_rows, canvas, delete_archive, gallery_rows, migrate_archive, read_json, reset_canvas, save_to_archive, set_current, share_current
from paths import DATA_DIR

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
        reset_canvas(key)
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
        state = canvas(session_key)
        current_composite = Path(state["path"]).resolve() if state and state.get("path") else None
        for path in paths:
            p = Path(path)
            if not _is_public_image(p):
                continue
            # Compose tools already updated canvas state and wrote the composite;
            # don't re-call set_current on our own composite (that would wipe layers).
            if current_composite and p.resolve() == current_composite:
                self._push(session_key, _image_event(p, _canvas_payload_full(self.runtime, session_key, state)))
                continue
            meta = read_json(p.with_suffix(".json"))
            set_current(session_key, p, bool(meta.get("original")), meta)
            self._push(session_key, _image_event(p, _canvas_payload_full(self.runtime, session_key, canvas(session_key))))

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

        # Anonymize public artifacts authored by this account.
        owner_values = {email, account_id}
        try:
            skills_changed = _anonymize_skill_owner(owner_values)
        except Exception:
            logger.exception("delete_account: skill anonymize failed")
            skills_changed = 0
        try:
            shared_changed = anonymize_shared(owner_values)
        except Exception:
            logger.exception("delete_account: shared anonymize failed")
            shared_changed = 0

        # Delete private archive directory.
        archive_root = str(archive_dir(account_id).resolve())
        try:
            delete_archive(account_id)
        except Exception:
            logger.exception("delete_account: archive delete failed")

        # Drop DB rows last so the email/account_id are still available above.
        with db.lock:
            db.conn.execute("DELETE FROM web_auth_tokens WHERE email = ?", (email,))
            db.conn.execute("DELETE FROM web_payments WHERE email = ?", (email,))
            db.conn.execute("DELETE FROM web_users WHERE account_id = ?", (account_id,))
            # Anonymize gallery/ephemeral shares this user created so their
            # name doesn't survive on rows we keep; archive-kind rows now
            # point at deleted files, so drop them entirely.
            db.conn.execute(
                "UPDATE canvas_shares SET owner_id = NULL, artist = 'anonymous' "
                "WHERE owner_id = ? OR (artist IS NOT NULL AND lower(artist) = ?)",
                (account_id, email.lower()),
            )
            db.conn.execute(
                "DELETE FROM canvas_shares WHERE kind = 'archive' AND image_path LIKE ?",
                (archive_root + "%",),
            )
            db.conn.commit()

        logger.info("Account deleted (skills_anonymized=%s shared_anonymized=%s)",
                    skills_changed, shared_changed)
        return {"ok": True}

    def _owner_id(self, session_id: str, ip: str, account_id: str) -> str:
        """Stable identity for the saved archive: account_id when signed in,
        otherwise the anonymous fallback derived from session+IP."""
        return account_id or self._anon_user_id(session_id, ip)

    def save_canvas(self, session_id: str, ip: str, account_id: str, title: str = "") -> list[dict]:
        """Persist the current composite into the owner's private archive and
        boost skill scores for the chain that produced it."""
        key = self.session_key(session_id)
        snapshot = canvas(key) or {}
        src = snapshot.get("path")
        if not src:
            return [{"type": "error", "content": "Nothing to save yet — make something first."}]
        chain = list(snapshot.get("chain") or [])
        owner = self._owner_id(session_id, ip, account_id)
        dest, meta = save_to_archive(src, owner, title=title, chain=chain)
        db = getattr(self.runtime, "db", None)
        skill_scoring.record_event(db, "save", chain, str(dest.resolve()))
        events: list[dict] = [{"type": "saved", "item": _archive_url(meta)}]
        link_event = self._mint_share_link(
            db, kind="archive", image_path=str(dest.resolve()),
            title=meta.get("title"), artist="you",
            chain=chain, session_key=key, owner_id=owner,
        )
        if link_event:
            events.append(link_event)
        events.append({"type": "status", "content": f'Saved "{meta["title"]}" to your archive.'})
        return events

    def archive_listing(self, session_id: str, ip: str, account_id: str, limit: int = 24, offset: int = 0) -> dict:
        owner = self._owner_id(session_id, ip, account_id)
        all_rows = archive_rows(owner)
        total = len(all_rows)
        page = all_rows[offset : offset + limit]
        return {"items": [_archive_url(r) for r in page], "total": total}

    def archive_remix(self, session_id: str, ip: str, account_id: str, path: str) -> list[dict]:
        owner = self._owner_id(session_id, ip, account_id)
        p = Path(unquote(path)).resolve()
        if not _is_archive_image(p, owner):
            raise ValueError("That archive entry is not available to remix.")
        return self._remix(self.session_key(session_id), p)

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
        key = self.session_key(session_id)
        snapshot = canvas(key) or {}
        chain = list(snapshot.get("chain") or [])
        dest, meta = share_current(key, title, artist)
        self._sync_gallery_file(dest)
        db = getattr(self.runtime, "db", None)
        skill_scoring.record_event(db, "share", chain, str(dest.resolve()))
        events: list[dict] = [{"type": "shared", "item": _gallery_url(meta)}]
        link_event = self._mint_share_link(
            db, kind="gallery", image_path=str(dest.resolve()),
            title=meta.get("title"), artist=meta.get("artist"),
            chain=chain, session_key=key,
            owner_id=self._owner_id(session_id, ip, account_id),
        )
        if link_event:
            events.append(link_event)
        events.append({"type": "message", "role": "assistant", "content": f'Shared "{meta["title"]}" by {meta["artist"]}.'})
        return events

    def gallery(self, session_id: str, limit: int = 24, offset: int = 0) -> dict:
        db = getattr(self.runtime, "db", None)
        all_rows = list(gallery_rows(db))
        total = len(all_rows)
        page = all_rows[offset : offset + limit]
        return {"items": [_gallery_url(r) for r in page], "total": total}

    def remix(self, session_id: str, path: str = "", share_id: str = "") -> list[dict]:
        key = self.session_key(session_id)
        if share_id:
            db = getattr(self.runtime, "db", None)
            share = share_links.lookup_share(db, share_id) if db else None
            if not share:
                raise ValueError("That share link is invalid or has expired.")
            p = Path(share["image_path"]).resolve()
            if not p.is_file():
                raise ValueError("That shared canvas is no longer available.")
            return self._remix(key, p, share=share)
        p = Path(unquote(path)).resolve()
        if not _is_gallery_image(p):
            raise ValueError("That gallery image is not available to remix.")
        return self._remix(key, p)

    # --- Share links ---------------------------------------------------

    def _mint_share_link(
        self, db, *, kind: str, image_path: str,
        title: str | None, artist: str | None,
        chain: list, session_key: str, owner_id: str = "",
    ) -> dict | None:
        if db is None:
            return None
        state = lc.get_state(session_key) or {}
        try:
            sid = share_links.create_share(
                db, kind=kind, image_path=image_path,
                title=title, artist=artist, chain=chain,
                palette_id=state.get("palette_id"),
                size=int(state.get("size") or lc.DEFAULT_SIZE),
                owner_id=owner_id or None,
            )
        except Exception:
            logger.exception("create_share failed (kind=%s)", kind)
            return None
        base = self._base_url()
        return {
            "type": "share_link",
            "share_id": sid,
            "url": share_links.build_share_url(base, sid),
            "qr_url": share_links.build_qr_url(base, sid),
            "kind": kind,
        }

    def get_link(
        self, session_id: str, ip: str, account_id: str,
        kind: str, path: str = "", title: str = "", artist: str = "",
    ) -> dict:
        """Mint (or return existing) share link for the current canvas, a
        gallery item, or an archive item."""
        key = self.session_key(session_id)
        db = getattr(self.runtime, "db", None)
        if db is None:
            raise RuntimeError("share links require the database to be available")
        owner = self._owner_id(session_id, ip, account_id)
        if kind == "current":
            snapshot = canvas(key) or {}
            src = snapshot.get("path")
            if not src:
                raise ValueError("Nothing to share yet — make something first.")
            chain = list(snapshot.get("chain") or [])
            sid = share_links.create_share(
                db, kind="ephemeral", image_path="__placeholder__",
                title=title or None, artist=artist or None, chain=chain,
                palette_id=(lc.get_state(key) or {}).get("palette_id"),
                size=int((lc.get_state(key) or {}).get("size") or lc.DEFAULT_SIZE),
                owner_id=owner,
            )
            # Snapshot the live canvas under the share's own filename so it
            # survives further edits in this session.
            snap = share_links.snapshot_current_canvas(src, sid)
            with db.lock:
                db.conn.execute(
                    "UPDATE canvas_shares SET image_path = ? WHERE share_id = ?",
                    (str(snap.resolve()), sid),
                )
                db.conn.commit()
        elif kind in ("gallery", "archive"):
            target = Path(unquote(path)).resolve()
            if kind == "gallery":
                if not _is_gallery_image(target):
                    raise ValueError("That gallery image is not available.")
            else:
                if not _is_archive_image(target, owner):
                    raise ValueError("That archive entry is not available.")
            meta = read_json(target.with_suffix(".json"))
            chain = list(meta.get("chain") or [])
            sid = share_links.find_or_create_for_path(
                db, kind=kind, image_path=str(target),
                title=meta.get("title"), artist=meta.get("artist") or ("you" if kind == "archive" else "anonymous"),
                chain=chain,
                palette_id=meta.get("palette_id"),
                size=meta.get("size"),
                owner_id=owner,
            )
        else:
            raise ValueError(f"unknown link kind: {kind!r}")
        base = self._base_url()
        return {
            "share_id": sid,
            "url": share_links.build_share_url(base, sid),
            "qr_url": share_links.build_qr_url(base, sid),
            "kind": kind,
        }

    def share_landing_data(self, share_id: str) -> dict | None:
        """Resolve a share_id to render the landing page."""
        db = getattr(self.runtime, "db", None)
        if db is None:
            return None
        share = share_links.lookup_share(db, share_id)
        if not share:
            return None
        image_path = Path(share["image_path"])
        share["image_exists"] = image_path.is_file()
        share["image_url"] = _file_url(image_path) if share["image_exists"] else ""
        share_links.bump_view_count(db, share_id)
        return share

    def share_qr_png(self, share_id: str) -> bytes | None:
        db = getattr(self.runtime, "db", None)
        if db is None or not share_links.lookup_share(db, share_id):
            return None
        return share_links.generate_qr_png(share_links.build_share_url(self._base_url(), share_id))

    def share_image_path(self, share_id: str) -> Path | None:
        """Resolve the canvas image for a share_id. Authorization is the
        share_id itself — anyone with the link sees the image."""
        db = getattr(self.runtime, "db", None)
        if db is None:
            return None
        share = share_links.lookup_share(db, share_id)
        if not share:
            return None
        p = Path(share["image_path"])
        return p if p.is_file() else None

    def _remix(self, key: str, p: Path, *, share: dict | None = None) -> list[dict]:
        # Share rows carry their own chain. Sidecar JSON (gallery/archive)
        # carries chain + title/artist. For ephemeral shares there is no
        # sidecar — fall back to share.chain and synth a meta dict.
        if share is not None:
            meta = {
                "title": share.get("title") or "shared canvas",
                "artist": share.get("artist") or "anonymous",
                "chain": list(share.get("chain") or []),
            }
        else:
            meta = read_json(p.with_suffix(".json"))
        reset_canvas(key)
        source_chain = list(meta.get("chain") or [])
        chain_restored = False
        restore_note = ""
        if source_chain:
            missing = [s.get("slug") for s in source_chain if not _read_skill_via(self.runtime, s.get("slug") or "")]
            if not missing:
                try:
                    state = lc.get_state(key)
                    out = lc.image_path(key)
                    replay_chain(
                        source_chain,
                        palette=get_palette(state.get("palette_id")),
                        size=int(state.get("size") or lc.DEFAULT_SIZE),
                        output_image_path=out,
                        workdir=out.parent,
                        skill_loader=lambda slug: _read_skill_via(self.runtime, slug),
                        on_step=self._chain_progress_cb(key),
                    )
                    with Image.open(out) as img:
                        lc.commit_image(key, img.convert("RGBA"), "remix", None)
                    new_state = lc.get_state(key)
                    new_state["last_chain"] = source_chain
                    lc.replace_state(key, new_state)
                    chain_restored = True
                except Exception as e:
                    logger.exception("remix chain replay failed: %s", e)
                    restore_note = " (couldn't rebuild layers — opening as a flat image)"
            else:
                restore_note = " (couldn't rebuild layers — opening as a flat image)"
        if not chain_restored:
            set_current(key, p, False, {"kind": "remix", **meta})
        skill_scoring.record_event(getattr(self.runtime, "db", None), "remix", source_chain, str(p))
        c = canvas(key)
        img_path = Path(c["path"]) if c and c.get("path") else p
        return [_image_event(img_path, _canvas_payload_full(self.runtime, key, c)), {"type": "message", "role": "assistant", "content": f"Remix loaded.{restore_note} Tell me how to mutate it."}]

    def download(self, session_id: str) -> list[dict]:
        """Fire-and-forget signal: user clicked Download for the current canvas."""
        key = self.session_key(session_id)
        snapshot = canvas(key) or {}
        chain = list(snapshot.get("chain") or [])
        if not chain:
            return []
        skill_scoring.record_event(getattr(self.runtime, "db", None), "download", chain, snapshot.get("path"))
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
                cs, skill_loader=skill_registry.get_record, force_new_seed=force_new_seed,
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

    def _sync_gallery_file(self, path: Path) -> None:
        dirs = [str(p) for p in (self.config.get("sync_directories") or [])]
        if str(GALLERY_DIR) not in dirs:
            self.config["sync_directories"] = dirs + [str(GALLERY_DIR)]
            config_manager.save(self.config)
        db = getattr(self.runtime, "db", None)
        if db:
            from plugins.services.helpers.parser_registry import get_modality
            db.upsert_file(str(path), path.name, path.suffix.lower(), get_modality(path.suffix.lower()), path.stat().st_mtime)
            orch = getattr(self.runtime, "_orchestrator_ref", None) or getattr(self.runtime, "services", {}).get("orchestrator")
            if orch:
                orch.on_file_discovered(str(path), path.suffix.lower(), get_modality(path.suffix.lower()))


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
            res = self.server.frontend.archive_listing(sid, self.client_address[0], self._cookie_uid(), limit=limit, offset=offset)
            return self._json({"ok": True, **res})
        if path.startswith("/share/"):
            tail = path[len("/share/"):]
            share_id, _, sub = tail.partition("/")
            if not share_links.is_valid_share_id(share_id):
                return self.send_error(404)
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
            if sub in {"image.png", "image.webp", "image"}:
                # URL kept extension-agnostic so old links to .png still
                # work after the WebP migration; we serve whatever the
                # actual file is with the right Content-Type.
                image_path = self.server.frontend.share_image_path(share_id)
                if image_path is None:
                    return self.send_error(404)
                ext = image_path.suffix.lower()
                mime = "image/webp" if ext == ".webp" else "image/png"
                return self._raw_file(image_path, mime)
            if sub == "":
                share = self.server.frontend.share_landing_data(share_id)
                if share is None:
                    return self._html(_share_missing_html(share_id), 404)
                return self._html(_share_landing_html(share, self.server.frontend._base_url()))
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
                anon = self.server.frontend._anon_user_id(self._cookie_sid(), self.client_address[0])
                try: migrate_archive(anon, account_id)
                except Exception: logger.exception("archive migrate (magic-link) failed")
                return self._redirect_with_uid("/account", account_id)
            return self._html("<!doctype html><meta charset=utf-8><title>Sign in</title><style>body{font-family:system-ui;max-width:480px;margin:80px auto;padding:0 24px;color:#222}</style><h1>Link expired</h1><p>That sign-in link is invalid or already used. Request a new one from the home page.</p><p><a href=\"/\">Back home</a></p>", 400)
        if path == "/auth/claim":
            qs = parse_qs(parsed.query)
            token = str(qs.get("token", [""])[0])
            account_id = self.server.frontend.claim_checkout(token)
            if account_id:
                anon = self.server.frontend._anon_user_id(self._cookie_sid(), self.client_address[0])
                try: migrate_archive(anon, account_id)
                except Exception: logger.exception("archive migrate (claim) failed")
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
                    sid, path=str(body.get("path") or ""), share_id=str(body.get("share_id") or ""))})
            if self.path == "/api/get_link":
                return self._json({"ok": True, **self.server.frontend.get_link(
                    sid, self.client_address[0], self._cookie_uid(),
                    kind=str(body.get("kind") or ""),
                    path=str(body.get("path") or ""),
                    title=str(body.get("title") or ""),
                    artist=str(body.get("artist") or ""),
                )})
            if self.path == "/api/save":
                return self._json({"ok": True, "events": self.server.frontend.save_canvas(sid, self.client_address[0], self._cookie_uid(), str(body.get("title") or ""))})
            if self.path == "/api/archive_remix":
                return self._json({"ok": True, "events": self.server.frontend.archive_remix(sid, self.client_address[0], self._cookie_uid(), str(body.get("path") or ""))})
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


def _is_archive_image(path: Path, owner_id: str) -> bool:
    try:
        target = path.resolve()
        owner_root = archive_dir(owner_id).resolve()
        return target.is_file() and target.suffix.lower() in IMAGE_EXTS and owner_root in target.parents
    except Exception:
        return False


def _archive_url(row: dict) -> dict:
    return {**row, "url": _file_url(Path(row["path"]))}


def _is_gallery_image(path: Path) -> bool:
    try:
        target, root = path.resolve(), GALLERY_DIR.resolve()
        return target.is_file() and target.suffix.lower() in IMAGE_EXTS and root in target.parents
    except Exception:
        return False


def _is_user_accessible_image(path: Path, session_key: str, owner_id: str = "") -> bool:
    """Public shared gallery, the requester's own canvas dir, or their archive."""
    try:
        target = path.resolve()
        if not (target.is_file() and target.suffix.lower() in IMAGE_EXTS):
            return False
        if _is_gallery_image(target):
            return True
        if session_key:
            own = lc.image_path(session_key).parent.resolve()
            if own in target.parents or target.parent == own:
                return True
        if owner_id and _is_archive_image(target, owner_id):
            return True
        return False
    except Exception:
        return False


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


def _gallery_url(row: dict) -> dict:
    return {**row, "url": _file_url(Path(row["path"]))}


def _share_landing_html(share: dict, base_url: str) -> str:
    """/share/{id} is a redirect into the SPA so the visitor lands on the
    canvas, not on a metadata page. The HTML body redirects via JS + meta
    refresh; the <head> keeps OG tags so link unfurls (Slack/Discord/Twitter)
    still preview the canvas. Crawlers read the meta, humans get redirected."""
    import html as _html
    sid = share["share_id"]
    title = share.get("title") or "Untitled"
    artist = share.get("artist") or "anonymous"
    url = share_links.build_share_url(base_url, sid)
    image_url = f"/share/{sid}/image.png" if share.get("image_exists") else ""
    og_image = f"{base_url.rstrip('/')}{image_url}" if image_url else ""
    target = f"/?share={_html.escape(sid, quote=True)}"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_html.escape(title)} — Second Brain</title>
  <meta http-equiv="refresh" content="0; url={target}">
  <link rel="canonical" href="/">
  <meta property="og:title" content="{_html.escape(title)}">
  <meta property="og:description" content="A canvas by {_html.escape(artist)} on Second Brain.">
  {f'<meta property="og:image" content="{_html.escape(og_image)}">' if og_image else ''}
  <meta property="og:type" content="article">
  <meta property="og:url" content="{_html.escape(url)}">
  <meta name="twitter:card" content="summary_large_image">
  <link rel="icon" type="image/x-icon" href="/favicon.ico">
  <script>location.replace({json.dumps(target)});</script>
</head>
<body style="background:#0b0d13;color:#888;font-family:system-ui;margin:0;padding:48px 24px;text-align:center;font-size:14px;">
  Opening canvas… <a href="{target}" style="color:#888;">Continue</a>
</body>
</html>"""


def _share_missing_html(share_id: str) -> str:
    import html as _html
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Link not found</title><link rel="stylesheet" href="/style.css?v=12"></head>
<body class="share-landing missing"><main class="share-landing-shell"><section class="share-landing-meta"><h1>Link not found</h1><p>No canvas exists for <code>{_html.escape(share_id)}</code>. It may have been removed.</p><p><a href="/">← Back to Second Brain</a></p></section></main></body></html>"""


def _int(value, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default
