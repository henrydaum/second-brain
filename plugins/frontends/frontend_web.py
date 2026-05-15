"""Tiny localhost web frontend for the demo build."""

from __future__ import annotations

import json
import logging
import mimetypes
import hashlib
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

from plugins.BaseFrontend import BaseFrontend, FrontendCapabilities
from plugins.tools.helpers.fractal_gallery import read_json, set_current
from paths import DATA_DIR

logger = logging.getLogger("WebFrontend")
WEB_ROOT = Path(__file__).with_name("web")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
WEB_PROFILE = "web_demo"
WEB_TOOLS = ["render_mandelbrot", "render_julia", "render_burning_ship", "render_tricorn", "render_phoenix", "render_newton_fractal", "render_barnsley_fern", "render_sierpinski", "render_mandelbulb", "render_formula_fractal", "render_strange_attractor", "render_flow_field", "render_cellular_automata", "render_reaction_diffusion", "render_lenia_like", "render_pixel_mosaic", "render_files", "save_fractal_art", "find_similar_images"]
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
        self._ensure_conversation(key)
        return self._drain(key) or [{"type": "message", "role": "assistant", "content": "New demo conversation ready."}]

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
                "You are running the public Second Brain web demo. Keep replies concise, visual, and safe. "
                "Do not tell users to use slash commands. Use render_mandelbrot for fractal requests, then rely on the image attachment to update the large showcase pane. "
                "Choose freely from the visual tools: Mandelbrot, Julia, Burning Ship, Tricorn, Phoenix, Newton, Barnsley fern, Sierpinski, Mandelbulb, safe custom formula fractals, strange attractors, flow fields, cellular automata, reaction-diffusion, Lenia-like automata, and pixel mosaics. "
                "After rendering, ask whether they want to share it. If yes, ask for an optional title and name, then call save_fractal_art. Use find_similar_images when they want to see related shared art."
            ),
            "whitelist_or_blacklist_tools": "whitelist",
            "tools_list": WEB_TOOLS,
        })

    def _apply_web_scope(self, key: str) -> None:
        session = self.runtime.sessions.get(key)
        if session:
            session.profile_override = WEB_PROFILE
            session.active_agent_profile = WEB_PROFILE
        self.runtime.add_system_prompt_extra(key, "web_demo", "Website safety: browser users cannot run slash commands or edit runtime configuration. The visible demo tools are for generating, sharing, and comparing fractal images. Never save/share art without calling save_fractal_art and letting the permission dialog confirm it.")

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
        for path in paths:
            p = Path(path)
            if _is_public_image(p):
                set_current(session_key, p, bool(read_json(p.with_suffix(".json")).get("original")))
                self._push(session_key, {"type": "hero_image", "url": f"/files?path={quote(str(p.resolve()), safe='')}", "name": p.name})

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
        self._push(session_key, {"type": "status", "content": f"{name}: {payload.get('status', 'running')}"})

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
