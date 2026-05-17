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
from types import SimpleNamespace
from urllib.parse import parse_qs, quote, unquote, urlparse

from plugins.helpers import stripe_client, web_auth

from PIL import Image

from config import config_manager
from plugins.BaseFrontend import BaseFrontend, FrontendCapabilities
from plugins.helpers import skill_scoring
from plugins.helpers.palettes import get_palette, list_palettes
from plugins.helpers.skill_runner import replay_chain
from plugins.helpers.skill_store import read_skill
from plugins.tools.helpers import layered_canvas as lc
from plugins.helpers.gallery import GALLERY_DIR, canvas, gallery_rows, read_json, reset_canvas, set_current, share_current, similar_rows
from paths import DATA_DIR

logger = logging.getLogger("WebFrontend")
WEB_ROOT = Path(__file__).with_name("web")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FAVICON_PATH = PROJECT_ROOT / "icon.ico"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
WEB_PROFILE = "web_demo"
WEB_TOOLS = [
    "search_skills",
    "create_skill",
    "update_skill",
    "delete_skill",
    "execute_skill",
    "manage_layers",
    "read_skill",
    "read_skill_guide"
]


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
        self._server.serve_forever()

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
        self.runtime.close_session(key)
        reset_canvas(key)
        self._ensure_conversation(key)
        return [{"type": "canvas_reset"}, *(self._drain(key) or [{"type": "message", "role": "assistant", "content": "New image ready."}])]

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
        profile.update({
            "llm": "default",
            "prompt_suffix": (
                "You are running the public Second Brain web demo. You make generative art collaboratively with the user. "
                "Keep replies short, warm, and conversational — like an artist talking through their work. Never use slash commands.\n\n"
                "The canvas is one square image with a selected color-theory palette and size. For any request to draw, render, stylize, or transform the canvas: first call search_skills; if a strong match exists, execute_skill; otherwise create_skill with Python code, then execute_skill. Creation skills start a new image. Transform skills receive canvas.image and modify it.\n\n"
                "Before authoring a new skill, call read_skill_guide ONCE per session for the canonical template, API reference, and method catalog. Then search_skills for adjacent references — the built-in library contains high-quality skills you can clone-and-adjust instead of writing from scratch. To clone-and-adjust an existing skill, call read_skill(slug) to see its source, then create_skill with your modifications.\n\n"
                "For natural subjects (suns, flowers, mountains, trees, landscapes, waves), prefer established generative methods — Vogel spirals for petals/seeds, flow fields for organic curves, Voronoi for cell structures, L-systems for branching, sediment bands for landscapes — over freehand drawing. Freehand draws of natural subjects look amateurish; method-based draws look designed.\n\n"
                "Skill code defines run(canvas, **params), uses allowed imports only (math, random, colorsys, numpy, PIL.Image, PIL.ImageDraw, PIL.ImageFilter, PIL.ImageOps, PIL.ImageEnhance), and must call canvas.commit(image). Create a blank image with canvas.new(color=canvas.palette.background) or canvas.create_image(). Use canvas.palette.primary, secondary, tertiary, accent, and background for colors; slots work as '#RRGGBB' strings and RGB sequences. Use canvas.size, width, height, and seed for deterministic geometry. An art_kit namespace is pre-injected (no import needed) with palette_color(t), vogel_spiral, fbm, rule_of_thirds, radial_falloff, smoothstep, lerp, oklch_to_rgb, and more — see read_skill_guide for the full list.\n\n"
                "Always integrate the palette: pull every color from canvas.palette slots or art_kit.palette_color(t); never hardcode hex unless the user explicitly asks. Reserve palette.accent for ≤10% of pixels. Let palette.background set the mood.\n\n"
                "After a creation skill, follow with 1–2 transforms (palette_grade, then bloom_glow or vignette) — this post-process pass consistently lifts quality. Keep transform chains ≤3 deep so palette swatch re-renders stay snappy.\n\n"
                "Seed every random source from canvas.seed: random.Random(canvas.seed) or numpy.random.default_rng(canvas.seed). Non-deterministic skills break the palette re-render flow.\n\n"
                "You cannot see the canvas directly. After executing a skill, explain the intended move briefly and ask for feedback when useful. "
                "Sharing, downloading, gallery, and remix are handled by the website buttons — not by tools."
            ),
            "whitelist_or_blacklist_tools": "whitelist",
            "tools_list": WEB_TOOLS,
        })

    def _apply_web_scope(self, key: str) -> None:
        session = self.runtime.sessions.get(key)
        if session:
            session.profile_override = WEB_PROFILE
            session.active_agent_profile = WEB_PROFILE
        self.runtime.add_system_prompt_extra(key, "web_demo", "Website safety: browser users cannot run slash commands or edit runtime configuration. Use the skill workflow for canvas work: search_skills, then execute_skill or create_skill plus execute_skill. Do not call sharing or gallery tools.")

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
                self._push(session_key, _image_event(p, _canvas_payload_full(session_key, state)))
                continue
            meta = read_json(p.with_suffix(".json"))
            set_current(session_key, p, bool(meta.get("original")), meta)
            self._push(session_key, _image_event(p, _canvas_payload_full(session_key, canvas(session_key))))

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
                    "  session_id = COALESCE(web_users.session_id, excluded.session_id)",
                    (uid, session_id, ip_hash, now, now),
                )
            else:
                db.conn.execute("UPDATE web_users SET last_seen = ?, ip_hash = COALESCE(ip_hash, ?) WHERE user_id = ?", (now, ip_hash, uid))

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

            global_caps = (
                (g5, _int(self.config.get("web_global_5h_turn_limit"), 600), "The public demo is busy right now. Try again a little later."),
                (gw, _int(self.config.get("web_global_week_turn_limit"), 6000), "The public demo hit its weekly budget. Try again later."),
            )
            for used, limit, msg in global_caps:
                if limit > 0 and used >= limit:
                    db.conn.commit()
                    return False, msg, None  # global cap → plain error, no paywall

            personal_caps = (
                (s5, _int(self.config.get("web_session_5h_turn_limit"), 40), "You've hit your 5-hour demo limit."),
                (sw, _int(self.config.get("web_session_week_turn_limit"), 160), "You've hit your weekly demo limit."),
                (ip5, _int(self.config.get("web_ip_5h_turn_limit"), 60), "This network has hit its 5-hour demo limit."),
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

    def _account_snapshot(self, session_id: str, ip: str, account_id: str) -> dict:
        db = getattr(self.runtime, "db", None)
        if db is None:
            return {"signed_in": False}
        if account_id:
            with db.lock:
                row = db.conn.execute(
                    "SELECT email, tier, credits FROM web_users WHERE account_id = ?",
                    (account_id,),
                ).fetchone()
            if row:
                return {"signed_in": True, "email": row["email"], "tier": row["tier"], "credits": int(row["credits"] or 0)}
        return {"signed_in": False}

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

    def canvas_payload(self, session_id: str) -> dict:
        key = self.session_key(session_id)
        return _canvas_payload_full(key, canvas(key))

    def palettes_payload(self) -> list[dict]:
        return [p.to_dict() for p in list_palettes()]

    def set_palette(self, session_id: str, palette_id: str) -> list[dict]:
        key = self.session_key(session_id)
        before = lc.get_state(key)
        if palette_id == before.get("palette_id"):
            return [{"type": "hero_image", "canvas": _canvas_payload_full(key, canvas(key))}] if before.get("image_path") else []
        try:
            state = lc.set_palette(key, palette_id)
            chain = state.get("last_chain") or []
            if chain:
                out = lc.image_path(key).with_name("_palette_replay.png")
                replay_chain(chain, palette=get_palette(palette_id), size=int(state.get("size") or lc.DEFAULT_SIZE), output_image_path=out, workdir=out.parent, skill_loader=read_skill)
                with Image.open(out) as img:
                    lc.commit_image(key, img.convert("RGBA"), f"palette:{palette_id}", None)
            c = canvas(key)
            return [{"type": "hero_image", "url": _file_url(Path(c["path"])), "name": Path(c["path"]).name, "canvas": _canvas_payload_full(key, c)}] if c and c.get("path") else []
        except Exception as e:
            lc.replace_state(key, before)
            return [{"type": "error", "content": f"Palette replay failed: {e}"}]

    def set_skill_control(self, session_id: str, chain_index: int, name: str, value, action: str = "") -> list[dict]:
        """Update one control on a chain entry, then replay the chain."""
        key = self.session_key(session_id)
        before = lc.get_state(key)
        chain = list(before.get("last_chain") or [])
        if not (0 <= chain_index < len(chain)):
            return [{"type": "error", "content": "That control no longer exists."}]
        step = chain[chain_index]
        skill = read_skill(step.get("slug") or "")
        if not skill:
            return [{"type": "error", "content": "Skill for that control was deleted."}]
        schema = {c.get("name"): c for c in (skill.controls or [])}
        spec = schema.get(name)
        if not spec:
            return [{"type": "error", "content": f"Unknown control '{name}'."}]
        try:
            if spec.get("type") == "button":
                act = action or spec.get("action") or "randomize"
                if act == "randomize":
                    param = spec.get("param") or "seed"
                    new_val = random.randint(1, 2_147_483_647)
                    if param == "seed":
                        state = lc.randomize_seed(key, chain_index, new_val)
                    else:
                        state = lc.set_skill_control(key, chain_index, param, new_val)
                else:
                    return [{"type": "error", "content": f"Unknown button action '{act}'."}]
            elif spec.get("type") == "pan":
                v = value or {}
                xv = float(v.get("x", spec.get("x_default", 0.0)))
                yv = float(v.get("y", spec.get("y_default", 0.0)))
                lc.set_skill_control(key, chain_index, spec["x_param"], xv)
                state = lc.set_skill_control(key, chain_index, spec["y_param"], yv)
            else:
                state = lc.set_skill_control(key, chain_index, name, _coerce_control_value(spec, value))
            out = lc.image_path(key).with_name("_control_replay.png")
            replay_chain(
                state.get("last_chain") or [],
                palette=get_palette(state.get("palette_id")),
                size=int(state.get("size") or lc.DEFAULT_SIZE),
                output_image_path=out,
                workdir=out.parent,
                skill_loader=read_skill,
                on_step=self._chain_progress_cb(key),
            )
            with Image.open(out) as img:
                lc.commit_image(key, img.convert("RGBA"), f"control:{step.get('slug')}.{name}", None)
            c = canvas(key)
            return [{"type": "hero_image", "url": _file_url(Path(c["path"])), "name": Path(c["path"]).name, "canvas": _canvas_payload_full(key, c)}] if c and c.get("path") else []
        except Exception as e:
            lc.replace_state(key, before)
            return [{"type": "error", "content": f"Control replay failed: {e}"}]

    def delete_layer(self, session_id: str, chain_index: int) -> list[dict]:
        """Remove one entry from the chain and re-render. Deleting index 0 clears the canvas."""
        key = self.session_key(session_id)
        before = lc.get_state(key)
        chain = list(before.get("last_chain") or [])
        if not (0 <= chain_index < len(chain)):
            return [{"type": "error", "content": "That layer no longer exists."}]
        if chain_index == 0:
            lc.reset(key)
            return [{"type": "canvas_reset"}]
        trimmed = chain[:chain_index] + chain[chain_index + 1:]
        out = lc.image_path(key).with_name("_layer_delete.png")
        try:
            replay_chain(
                trimmed,
                palette=get_palette(before.get("palette_id")),
                size=int(before.get("size") or lc.DEFAULT_SIZE),
                output_image_path=out,
                workdir=out.parent,
                skill_loader=read_skill,
            )
            with Image.open(out) as img:
                lc.commit_image(key, img.convert("RGBA"), f"layer_delete:{chain_index}", None)
            new_state = lc.get_state(key)
            new_state["last_chain"] = trimmed
            lc.replace_state(key, new_state)
            c = canvas(key)
            return [{"type": "hero_image", "url": _file_url(Path(c["path"])), "name": Path(c["path"]).name, "canvas": _canvas_payload_full(key, c)}] if c and c.get("path") else []
        except Exception as e:
            lc.replace_state(key, before)
            return [{"type": "error", "content": f"Delete layer failed: {e}"}]

    def share(self, session_id: str, title: str, artist: str) -> list[dict]:
        key = self.session_key(session_id)
        snapshot = canvas(key) or {}
        chain = list(snapshot.get("chain") or [])
        dest, meta = share_current(key, title, artist)
        self._sync_gallery_file(dest)
        skill_scoring.record_event(getattr(self.runtime, "db", None), "share", chain, str(dest.resolve()))
        return [{"type": "shared", "item": _gallery_url(meta)}, {"type": "message", "role": "assistant", "content": f'Shared "{meta["title"]}" by {meta["artist"]}.'}]

    def gallery(self, session_id: str, limit: int = 24, offset: int = 0) -> dict:
        item = canvas(self.session_key(session_id))
        ctx = SimpleNamespace(services=getattr(self.runtime, "services", {}), db=getattr(self.runtime, "db", None))
        db = getattr(self.runtime, "db", None)
        # Fetch a bigger window than the page so we can paginate over it.
        # similar_rows is bounded by its arg; gallery_rows is the full list.
        all_rows = similar_rows(item["path"], ctx, max(limit + offset, 200)) if item and item.get("path") else list(gallery_rows(db))
        total = len(all_rows)
        page = all_rows[offset : offset + limit]
        return {"items": [_gallery_url(r) for r in page], "total": total}

    def remix(self, session_id: str, path: str) -> list[dict]:
        p = Path(unquote(path)).resolve()
        if not _is_gallery_image(p):
            raise ValueError("That gallery image is not available to remix.")
        key = self.session_key(session_id)
        meta = read_json(p.with_suffix(".json"))
        reset_canvas(key)
        source_chain = list(meta.get("chain") or [])
        chain_restored = False
        restore_note = ""
        if source_chain:
            missing = [s.get("slug") for s in source_chain if not read_skill(s.get("slug") or "")]
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
                        skill_loader=read_skill,
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
        return [_image_event(img_path, _canvas_payload_full(key, c)), {"type": "message", "role": "assistant", "content": f"Remix loaded.{restore_note} Tell me how to mutate it."}]

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
        """Re-run the current chain with a fresh seed on every step."""
        key = self.session_key(session_id)
        state = lc.get_state(key)
        chain = list(state.get("last_chain") or [])
        if not chain:
            return [{"type": "error", "content": "Nothing to regenerate yet — make something first."}]
        # Reseed every step so the whole composition shifts, not just one layer.
        reseeded = [{**step, "seed": random.randint(1, 2_147_483_647)} for step in chain]
        out = lc.image_path(key).with_name("_regenerate.png")
        try:
            replay_chain(
                reseeded,
                palette=get_palette(state.get("palette_id")),
                size=int(state.get("size") or lc.DEFAULT_SIZE),
                output_image_path=out,
                workdir=out.parent,
                skill_loader=read_skill,
                on_step=self._chain_progress_cb(key),
            )
            with Image.open(out) as img:
                # commit_image with chain_entry=None preserves the chain; we
                # update it ourselves so seeds reflect the new render.
                lc.commit_image(key, img.convert("RGBA"), "regenerate", None)
            # Persist the reseeded chain so subsequent palette swaps re-render
            # against the same composition.
            new_state = lc.get_state(key)
            new_state["last_chain"] = reseeded
            lc.replace_state(key, new_state)
            c = canvas(key)
            return [{"type": "hero_image", "url": _file_url(Path(c["path"])), "name": Path(c["path"]).name, "canvas": _canvas_payload_full(key, c)}] if c and c.get("path") else []
        except Exception as e:
            return [{"type": "error", "content": f"Regenerate failed: {e}"}]

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
        if path == "/api/account":
            qs = parse_qs(parsed.query)
            sid = str(qs.get("session_id", ["demo"])[0])[:80]
            return self._json({"ok": True, "account": self.server.frontend.account_info(sid, self.client_address[0], self._cookie_uid())})
        if path == "/files":
            qs = parse_qs(parsed.query)
            sid = str(qs.get("session_id", [self._cookie_sid()])[0])[:80]
            return self._local_file(qs.get("path", [""])[0], sid)
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
            if self.path == "/api/new":
                return self._json({"ok": True, "events": self.server.frontend.new_chat(sid)})
            if self.path == "/api/approval":
                return self._json({"ok": True, "events": self.server.frontend.approve(sid, bool(body.get("value")))})
            if self.path == "/api/share":
                return self._json({"ok": True, "events": self.server.frontend.share(sid, str(body.get("title") or "untitled"), str(body.get("artist") or "anonymous"))})
            if self.path == "/api/remix":
                return self._json({"ok": True, "events": self.server.frontend.remix(sid, str(body.get("path") or ""))})
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

    def _local_file(self, raw_path: str, session_id: str = ""):
        path = Path(unquote(raw_path))
        if not _is_user_accessible_image(path, self.server.frontend.session_key(session_id) if session_id else ""):
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


def _is_gallery_image(path: Path) -> bool:
    try:
        target, root = path.resolve(), GALLERY_DIR.resolve()
        return target.is_file() and target.suffix.lower() in IMAGE_EXTS and root in target.parents
    except Exception:
        return False


def _is_user_accessible_image(path: Path, session_key: str) -> bool:
    """Either the shared gallery (public) or the requester's own canvas dir."""
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


def _canvas_payload_full(session_key: str, state: dict | None) -> dict:
    """Canvas payload plus per-entry control schemas for the website panel."""
    base = _canvas_payload(state)
    if not state:
        return base
    chain = state.get("chain") or state.get("last_chain") or []
    panels = []
    layers = []
    for idx, step in enumerate(chain):
        skill = read_skill(step.get("slug") or "")
        slug = step.get("slug") or ""
        name = skill.name if skill else slug
        layers.append({"chain_index": idx, "slug": slug, "skill_name": name, "kind": step.get("kind") or (skill.kind if skill else "")})
        if not skill or not skill.controls:
            continue
        panels.append({
            "chain_index": idx,
            "slug": skill.slug,
            "skill_name": skill.name,
            "schema": skill.controls,
            "values": dict(step.get("controls") or {}),
            "seed": int(step.get("seed") or 0),
        })
    base["controls_panels"] = panels
    base["layers"] = layers
    return base


def _coerce_control_value(spec: dict, value):
    """Coerce an incoming control value to match its declared type."""
    t = spec.get("type")
    if t == "slider":
        v = float(value)
        lo, hi = float(spec.get("min", v)), float(spec.get("max", v))
        return max(lo, min(hi, v))
    if t == "bool":
        return bool(value)
    if t == "enum":
        allowed = [opt.get("value") for opt in (spec.get("options") or [])]
        if value not in allowed:
            raise ValueError(f"enum value {value!r} not in allowed options")
        return value
    if t == "palette":
        return str(value)
    # pan controls deliver values via their underlying numeric param names; this
    # path only fires when the frontend sends {name: x_param, value: number}.
    return value


def _gallery_url(row: dict) -> dict:
    return {**row, "url": _file_url(Path(row["path"]))}


def _int(value, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default
