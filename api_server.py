"""
HTTP API server for Second Brain.

Exposes an agent chat endpoint and file serving over REST so
external systems (e.g. OpenClaw) can operate Second Brain directly.

Endpoints:
    GET  /tools              — list all tool schemas (OpenAI format)
    GET  /tools/{name}       — single tool schema
    POST /tools/{name}       — call a tool directly (JSON body = kwargs)
    POST /agent/run          — send a message to the built-in agent (JSON body = {"message": "..."})
    GET  /files?path=...     — serve a file from an indexed sync directory

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

logger = logging.getLogger("API")

CHUNK_SIZE = 64 * 1024  # 64 KB read chunks for file streaming


class _Handler(BaseHTTPRequestHandler):
    """Thin REST handler dispatching to ToolRegistry and the built-in Agent."""

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
        """Build the base URL for attachment links."""
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
        """Check that file_path lives inside a configured sync_directory."""
        resolved = file_path.resolve()
        for sd in self.server.config.get("sync_directories", []):
            if resolved.is_relative_to(Path(sd).resolve()):
                return True
        return False

    # ------ routes ------

    def do_GET(self):
        if not self._check_auth():
            return

        # GET /tools
        if self.path == "/tools":
            schemas = self.server.tool_registry.get_all_schemas()
            self._send_json(200, schemas)
            return

        # GET /tools/{name}
        m = re.fullmatch(r"/tools/([^/]+)", self.path)
        if m:
            schema = self.server.tool_registry.get_schema(m.group(1))
            if schema is None:
                self._send_json(404, {"error": f"Unknown tool: {m.group(1)}"})
            else:
                self._send_json(200, schema)
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

        # POST /tools/{name}
        m = re.fullmatch(r"/tools/([^/]+)", self.path)
        if m:
            tool_name = m.group(1)
            try:
                kwargs = self._read_body()
            except json.JSONDecodeError as e:
                self._send_json(400, {"error": f"Invalid JSON: {e}"})
                return

            result = self.server.tool_registry.call(tool_name, **kwargs)
            self._send_json(200, result.to_dict(base_url=self._base_url()))
            return

        # POST /agent/run
        if self.path == "/agent/run":
            try:
                body = self._read_body()
            except json.JSONDecodeError as e:
                self._send_json(400, {"error": f"Invalid JSON: {e}"})
                return

            message = body.get("message", "").strip()
            if not message:
                self._send_json(400, {"error": "Missing 'message' field"})
                return

            llm = self.server.services.get("llm")
            if llm is None or not llm.loaded:
                self._send_json(503, {"error": "LLM service not available"})
                return

            if self.server.agent is None:
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
                )

            collected_paths = []

            def _collect(tool_name, result):
                if result.gui_display_paths:
                    collected_paths.extend(result.gui_display_paths)

            self.server.agent.on_tool_result = _collect

            try:
                response = self.server.agent.chat(message)
            except Exception as e:
                logger.error(f"/agent/run error: {e}")
                self._send_json(500, {"error": str(e)})
                return

            base_url = self._base_url()
            attachments = []
            for p in collected_paths:
                modality = get_modality(Path(p).suffix)
                att = {"path": p, "modality": modality}
                if base_url:
                    att["url"] = f"{base_url}/files?path={quote(p, safe='')}"
                attachments.append(att)

            self._send_json(200, {"response": response, "attachments": attachments})
            return

        self._send_json(404, {"error": "Not found"})


def start_api_server(tool_registry, db, config, services, orchestrator) -> HTTPServer:
    """
    Start the API server on a daemon thread. Returns the HTTPServer instance.
    """
    port = config.get("api_port", 5123)
    token = config.get("api_token", "")

    # Auto-approve dangerous commands when called through the API
    # (OpenClaw already has root-level system access)
    tool_registry.on_approve_command = lambda cmd, justification: True

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
    server.agent = None  # created on first /agent/run call, then reused

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info(f"API server listening on http://127.0.0.1:{port}")
    return server
