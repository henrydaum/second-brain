"""Tiny localhost web frontend for the demo build."""

from __future__ import annotations

import json
import logging
import mimetypes
import hashlib
import random
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, quote, unquote, urlparse

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
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
WEB_PROFILE = "web_demo"
WEB_TOOLS = [
    "search_skills",
    "create_skill",
    "update_skill",
    "delete_skill",
    "execute_skill",
    "read_skill",
    "read_skill_guide",
    "request_user_feedback",
]
USAGE_PATH = DATA_DIR / "web_usage.json"


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

    def chat(self, session_id: str, message: str, ip: str = "") -> list[dict]:
        key = self.session_key(session_id)
        text = (message or "").strip()
        if text.startswith("/"):
            return self.new_chat(session_id) if text == "/new" else [{"type": "error", "content": "Slash commands are disabled on the public demo. Use the chat or the New button."}]
        self._ensure_conversation(key)
        if self.has_pending_approval(key):
            return [{"type": "error", "content": "Use the approval buttons to answer this permission request."}]
        ok, error = self._record_usage(session_id, ip)
        if not ok:
            return [{"type": "error", "content": error}]
        self.submit_text(key, text)
        return self._drain(key)

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
                self._push(session_key, _image_event(p, state))
                continue
            meta = read_json(p.with_suffix(".json"))
            set_current(session_key, p, bool(meta.get("original")), meta)
            self._push(session_key, _image_event(p, canvas(session_key)))

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
        name = payload.get("tool_name") or payload.get("command_name") or "tool"
        self._push(session_key, {
            "type": "tool_status",
            "call_id": payload.get("call_id"),
            "name": name,
            "status": payload.get("status", "running"),
            "ok": payload.get("ok"),
            "error": payload.get("error"),
        })

    def _live_session_keys(self) -> list[str]:
        return [k for k in getattr(self.runtime, "sessions", {}) if k.startswith("web:")]

    def _record_usage(self, session_id: str, ip: str) -> tuple[bool, str]:
        now = time.time()
        five_h, week = now - 5 * 3600, now - 7 * 24 * 3600
        sid = hashlib.sha256(f"{ip}|{session_id}".encode()).hexdigest()[:24]
        with self._usage_lock:
            data = _read_usage()
            data["global"] = [t for t in data.get("global", []) if t >= week]
            sessions = data.setdefault("sessions", {})
            sessions[sid] = [t for t in sessions.get(sid, []) if t >= week]
            g5 = sum(t >= five_h for t in data["global"])
            s5 = sum(t >= five_h for t in sessions[sid])
            limits = (
                (g5, _int(self.config.get("web_global_5h_turn_limit"), 600), "The public demo is busy right now. Try again a little later."),
                (len(data["global"]), _int(self.config.get("web_global_week_turn_limit"), 6000), "The public demo hit its weekly budget. Try again later."),
                (s5, _int(self.config.get("web_session_5h_turn_limit"), 40), "This browser has hit its 5-hour demo limit. Try again later."),
                (len(sessions[sid]), _int(self.config.get("web_session_week_turn_limit"), 160), "This browser has hit its weekly demo limit."),
            )
            for used, limit, msg in limits:
                if limit > 0 and used >= limit:
                    data["sessions"] = {k: v for k, v in sessions.items() if v}
                    _write_usage(data)
                    return False, msg
            data["global"].append(now); sessions[sid].append(now); _write_usage(data)
        return True, ""

    def canvas_payload(self, session_id: str) -> dict:
        return _canvas_payload(canvas(self.session_key(session_id)))

    def palettes_payload(self) -> list[dict]:
        return [p.to_dict() for p in list_palettes()]

    def set_palette(self, session_id: str, palette_id: str) -> list[dict]:
        key = self.session_key(session_id)
        before = lc.get_state(key)
        if palette_id == before.get("palette_id"):
            return [{"type": "hero_image", "canvas": _canvas_payload(canvas(key))}] if before.get("image_path") else []
        try:
            state = lc.set_palette(key, palette_id)
            chain = state.get("last_chain") or []
            if chain:
                out = lc.image_path(key).with_name("_palette_replay.png")
                replay_chain(chain, palette=get_palette(palette_id), size=int(state.get("size") or lc.DEFAULT_SIZE), output_image_path=out, workdir=out.parent, skill_loader=read_skill)
                with Image.open(out) as img:
                    lc.commit_image(key, img.convert("RGBA"), f"palette:{palette_id}", None)
            c = canvas(key)
            return [{"type": "hero_image", "url": _file_url(Path(c["path"])), "name": Path(c["path"]).name, "canvas": _canvas_payload(c)}] if c and c.get("path") else []
        except Exception as e:
            lc.replace_state(key, before)
            return [{"type": "error", "content": f"Palette replay failed: {e}"}]

    def share(self, session_id: str, title: str, artist: str) -> list[dict]:
        key = self.session_key(session_id)
        snapshot = canvas(key) or {}
        chain = list(snapshot.get("chain") or [])
        dest, meta = share_current(key, title, artist)
        self._sync_gallery_file(dest)
        skill_scoring.record_event(getattr(self.runtime, "db", None), "share", chain, str(dest.resolve()))
        return [{"type": "shared", "item": _gallery_url(meta)}, {"type": "message", "role": "assistant", "content": f'Shared "{meta["title"]}" by {meta["artist"]}.'}]

    def gallery(self, session_id: str, limit: int = 24) -> list[dict]:
        item = canvas(self.session_key(session_id))
        ctx = SimpleNamespace(services=getattr(self.runtime, "services", {}), db=getattr(self.runtime, "db", None))
        db = getattr(self.runtime, "db", None)
        rows = similar_rows(item["path"], ctx, limit) if item and item.get("path") else gallery_rows(db)[:limit]
        return [_gallery_url(r) for r in rows]

    def remix(self, session_id: str, path: str) -> list[dict]:
        p = Path(unquote(path)).resolve()
        if not _is_gallery_image(p):
            raise ValueError("That gallery image is not available to remix.")
        key = self.session_key(session_id)
        meta = read_json(p.with_suffix(".json"))
        reset_canvas(key); set_current(key, p, False, {"kind": "remix", **meta})
        # Attribute the remix to the source image's chain (stored in its sidecar).
        source_chain = list(meta.get("chain") or [])
        skill_scoring.record_event(getattr(self.runtime, "db", None), "remix", source_chain, str(p))
        return [_image_event(p, canvas(key)), {"type": "message", "role": "assistant", "content": "Remix loaded. Tell me how to mutate it."}]

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
            return [{"type": "hero_image", "url": _file_url(Path(c["path"])), "name": Path(c["path"]).name, "canvas": _canvas_payload(c)}] if c and c.get("path") else []
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
        path = urlparse(self.path).path
        if path == "/api/health":
            return self._json({"ok": True})
        if path == "/api/events":
            sid = str(parse_qs(urlparse(self.path).query).get("session_id", ["demo"])[0])[:80]
            return self._json({"ok": True, "events": self.server.frontend._drain(self.server.frontend.session_key(sid))})
        if path == "/api/canvas":
            sid = str(parse_qs(urlparse(self.path).query).get("session_id", ["demo"])[0])[:80]
            return self._json({"ok": True, "canvas": self.server.frontend.canvas_payload(sid)})
        if path == "/api/palettes":
            return self._json({"ok": True, "palettes": self.server.frontend.palettes_payload()})
        if path == "/api/gallery":
            qs = parse_qs(urlparse(self.path).query); sid = str(qs.get("session_id", ["demo"])[0])[:80]
            return self._json({"ok": True, "items": self.server.frontend.gallery(sid)})
        if path == "/files":
            return self._local_file(parse_qs(urlparse(self.path).query).get("path", [""])[0])
        rel = "index.html" if path in {"", "/"} else path.lstrip("/")
        return self._file(WEB_ROOT / rel)

    def do_POST(self):
        body = self._body()
        sid = str(body.get("session_id") or "demo")[:80]
        try:
            if self.path == "/api/chat":
                events = self.server.frontend.chat(sid, str(body.get("message") or ""), self.client_address[0])
                return self._json({"ok": True, "events": events})
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

    def _local_file(self, raw_path: str):
        path = Path(unquote(raw_path))
        if not _is_public_image(path):
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


def _file_url(path: Path) -> str:
    # mtime cache-buster: the composite path is stable across recompositions,
    # so without this the browser keeps serving the old image.
    try:
        v = int(path.stat().st_mtime)
    except Exception:
        v = 0
    return f"/files?path={quote(str(path.resolve()), safe='')}&v={v}"


def _image_event(path: Path, state: dict | None = None) -> dict:
    return {"type": "hero_image", "url": _file_url(path), "name": path.name, "canvas": _canvas_payload(state)}


def _canvas_payload(state: dict | None) -> dict:
    if not state:
        return {}
    if not state.get("path"):
        return {k: v for k, v in state.items() if k != "path"}
    p = Path(state["path"])
    return {**state, "url": _file_url(p), "name": p.name}


def _gallery_url(row: dict) -> dict:
    return {**row, "url": _file_url(Path(row["path"]))}


def _read_usage() -> dict:
    try:
        return json.loads(USAGE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"global": [], "sessions": {}}


def _write_usage(data: dict) -> None:
    USAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    USAGE_PATH.write_text(json.dumps(data), encoding="utf-8")


def _int(value, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default
