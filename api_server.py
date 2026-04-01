"""
HTTP API server for Second Brain.

Exposes tool dispatch, REPL commands, and file serving over REST so
external systems (e.g. OpenClaw) can operate Second Brain directly.

Endpoints:
    GET  /tools              — list all tool schemas (OpenAI format)
    GET  /tools/{name}       — single tool schema
    POST /tools/{name}       — call a tool (JSON body = kwargs)
    POST /repl               — run a REPL command (JSON body = {"command": "...", "arg": "..."})
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
from gui.commands import CommandRegistry, register_core_commands

logger = logging.getLogger("API")

CHUNK_SIZE = 64 * 1024  # 64 KB read chunks for file streaming


class _Handler(BaseHTTPRequestHandler):
    """Thin REST handler dispatching to ToolRegistry and CommandRegistry."""

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

        # POST /repl
        if self.path == "/repl":
            try:
                body = self._read_body()
            except json.JSONDecodeError as e:
                self._send_json(400, {"error": f"Invalid JSON: {e}"})
                return

            command = body.get("command", "").strip()
            if not command:
                self._send_json(400, {"error": "Missing 'command' field"})
                return

            arg = body.get("arg", "").strip()

            output = self.server.command_registry.dispatch(command, arg)
            self._send_json(200, {"output": output or ""})
            return

        self._send_json(404, {"error": "Not found"})


def start_api_server(tool_registry, db, config, services, orchestrator,
                     ctrl, root_dir) -> HTTPServer:
    """
    Start the API server on a daemon thread. Returns the HTTPServer instance.
    """
    port = config.get("api_port", 5123)
    token = config.get("api_token", "")

    # Build command registry (same commands as REPL)
    command_registry = CommandRegistry()
    register_core_commands(command_registry, ctrl, services, tool_registry, root_dir)

    # Auto-approve dangerous commands when called through the API
    # (OpenClaw already has root-level system access)
    tool_registry.on_approve_command = lambda cmd, justification: True

    server = HTTPServer(("127.0.0.1", port), _Handler)
    server.tool_registry = tool_registry
    server.command_registry = command_registry
    server.db = db
    server.config = config
    server.services = services
    server.orchestrator = orchestrator
    server.api_token = token

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info(f"API server listening on http://127.0.0.1:{port}")
    return server
