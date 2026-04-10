"""
HTTP API server for Second Brain.

Presents the same chat interface as the GUI: plain text goes to the
agent, ``/slash`` commands control the system.  External callers like
OpenClaw just send text and get text back.

Endpoints:
    POST /chat                — send ``{"message": "..."}``; get a response
    GET  /files?path=...      — serve a file from an indexed sync directory

Uses stdlib only — no Flask/FastAPI dependency.
"""

import json
import logging
import mimetypes
import re
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote, quote

from Stage_1.registry import get_modality
from Stage_3.agent import Agent
from Stage_3.system_prompt import build_system_prompt
from frontend.shared.commands import CommandEntry, CommandRegistry, register_core_commands
from frontend.shared.dispatch import route_input
from frontend.shared.token_stripper import strip_model_tokens

logger = logging.getLogger("API")

CHUNK_SIZE = 64 * 1024  # 64 KB read chunks for file streaming


class _Handler(BaseHTTPRequestHandler):
    """Thin REST handler: chat, file serving, and command autocomplete."""

    def log_message(self, fmt, *args):
        logger.debug(fmt % args)

    # ------ auth ------

    def _check_auth(self) -> bool:
        token = self.server.api_token
        if not token:
            return True
        auth = self.headers.get("Authorization", "")
        if auth == f"Bearer {token}":
            return True
        self._send_json(401, {"error": "Unauthorized"})
        return False

    # ------ helpers ------

    def _base_url(self) -> str:
        host = self.headers.get("Host", f"127.0.0.1:{self.server.server_port}")
        return f"http://{host}"

    def _send_json(self, code: int, obj):
        body = json.dumps(obj, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    def _is_path_allowed(self, file_path: Path) -> bool:
        resolved = file_path.resolve()
        for sd in self.server.config.get("sync_directories", []):
            if resolved.is_relative_to(Path(sd).resolve()):
                return True
        return False

    def _ensure_agent(self):
        """Lazily create the agent on first chat message. Returns the agent or None."""
        if self.server.agent is not None:
            return self.server.agent

        llm = self.server.services.get("llm")
        if llm is None or not llm.loaded:
            return None

        self.server.agent = Agent(
            llm,
            self.server.tool_registry,
            self.server.config,
            system_prompt=lambda: build_system_prompt(
                self.server.db,
                self.server.orchestrator,
                self.server.tool_registry,
                self.server.services,
            ),
            on_message=self.server.on_agent_message,
            approve_command=lambda cmd, justification: True,
        )
        return self.server.agent

    # ------ routes ------

    def do_GET(self):
        if not self._check_auth():
            return

        # GET /files?path=...
        if self.path.startswith("/files"):
            m = re.search(r"[?&]path=([^&]+)", self.path)
            if not m:
                self._send_json(400, {"error": "Missing 'path' query parameter"})
                return

            file_path = Path(unquote(m.group(1)))
            if not file_path.is_file():
                self._send_json(404, {"error": "File not found"})
                return
            if not self._is_path_allowed(file_path):
                self._send_json(403, {"error": "Path is outside allowed sync directories"})
                return

            content_type, _ = mimetypes.guess_type(str(file_path))
            content_type = content_type or "application/octet-stream"
            file_size = file_path.stat().st_size

            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(file_size))
            self.send_header("Content-Disposition", f'inline; filename="{file_path.name}"')
            self.end_headers()

            with open(file_path, "rb") as f:
                while chunk := f.read(CHUNK_SIZE):
                    self.wfile.write(chunk)
            return

        self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        if not self._check_auth():
            return

        # POST /chat
        if self.path == "/chat":
            try:
                body = self._read_body()
            except json.JSONDecodeError as e:
                self._send_json(400, {"error": f"Invalid JSON: {e}"})
                return

            message = body.get("message", "").strip()
            if not message:
                self._send_json(400, {"error": "Missing 'message' field"})
                return

            agent = self._ensure_agent()

            # For chat messages (non-commands), check that the agent is ready
            if not message.startswith("/") and agent is None:
                self._send_json(503, {"error": "LLM service not available. Try /load llm"})
                return

            with self.server.chat_lock:
                try:
                    result = route_input(message, self.server.registry, agent)
                except Exception as e:
                    logger.error(f"/chat error: {e}")
                    self._send_json(500, {"error": str(e)})
                    return

            # Build response
            base_url = self._base_url()
            attachments = []
            for p in result.attachments:
                modality = get_modality(Path(p).suffix)
                att = {"path": p, "modality": modality}
                att["url"] = f"{base_url}/files?path={quote(p, safe='')}"
                attachments.append(att)

            # Strip thinking tokens — API callers get clean text only
            clean_text, _ = strip_model_tokens(result.text)

            self._send_json(200, {
                "type": result.type,
                "response": clean_text,
                "attachments": attachments,
            })
            return

        self._send_json(404, {"error": "Not found"})


def start_api_server(tool_registry, db, config, services, orchestrator,
                     ctrl=None, root_dir=None) -> HTTPServer:
    """
    Start the API server on a daemon thread. Returns the HTTPServer instance.

    Parameters:
        ctrl:     Controller instance (needed for slash commands).
        root_dir: Project root path (needed for /reload command).
    """
    port = config.get("api_port", 5123)
    token = config.get("api_token", "")

    try:
        server = HTTPServer(("127.0.0.1", port), _Handler)
    except OSError as e:
        logger.error(f"API server failed to bind on port {port}: {e}. Check api_port in config.")
        raise
    server.tool_registry = tool_registry
    server.db = db
    server.config = config
    server.services = services
    server.orchestrator = orchestrator
    server.api_token = token
    server.agent = None  # created lazily on first chat message
    server.chat_lock = threading.Lock()  # agent.chat() is not thread-safe
    server.conversation_id = None  # current conversation DB id

    def _on_agent_message(msg: dict):
        """Persist every agent message to the conversations DB."""
        role = msg.get("role", "")
        content = msg.get("content") or ""
        tool_call_id = msg.get("tool_call_id")
        tool_name = msg.get("name")

        if server.conversation_id is None:
            title = content[:80].replace("\n", " ").strip() if role == "user" else "New conversation"
            server.conversation_id = db.create_conversation(title)

        if msg.get("tool_calls"):
            content = json.dumps({"content": content, "tool_calls": msg["tool_calls"]})

        db.save_message(server.conversation_id, role, content,
                        tool_call_id=tool_call_id, tool_name=tool_name)

    server.on_agent_message = _on_agent_message

    # Build command registry for the API (same commands as GUI/REPL)
    registry = CommandRegistry()
    if ctrl and root_dir is not None:
        register_core_commands(
            registry, ctrl, services, tool_registry, root_dir,
            get_agent=lambda: server.agent,
        )

    # Override /clear to also start a new conversation in the DB
    def _api_clear(_arg):
        server.conversation_id = None
        if server.agent:
            server.agent.reset()
        return "(conversation history cleared)"

    registry.register(CommandEntry("clear", "Clear agent conversation history",
                                   handler=_api_clear))
    server.registry = registry

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info(f"API server listening on http://127.0.0.1:{port}")
    return server
