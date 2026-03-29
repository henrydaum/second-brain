"""
HTTP API server for Second Brain.

Exposes tool dispatch and agent chat over REST so external systems
(e.g. OpenClaw) can call Second Brain tools directly.

Endpoints:
    GET  /tools              — list all tool schemas
    GET  /tools/{name}       — single tool schema
    POST /tools/{name}       — call a tool (JSON body = kwargs)
    POST /chat               — full agent chat loop (JSON body = {"message": "..."})

Uses stdlib only — no Flask/FastAPI dependency.
"""

import json
import logging
import re
import threading
from functools import partial
from http.server import HTTPServer, BaseHTTPRequestHandler

from Stage_3.agent import Agent
from Stage_3.system_prompt import build_system_prompt

logger = logging.getLogger("API")


class _Handler(BaseHTTPRequestHandler):
    """Thin REST handler dispatching to ToolRegistry and Agent."""

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
            self._send_json(200, result.to_dict())
            return

        # POST /chat
        if self.path == "/chat":
            try:
                body = self._read_body()
            except json.JSONDecodeError as e:
                self._send_json(400, {"error": f"Invalid JSON: {e}"})
                return

            message = body.get("message", "")
            if not message:
                self._send_json(400, {"error": "Missing 'message' field"})
                return

            # Create a fresh agent for each request (stateless)
            srv = self.server
            llm = srv.services.get("llm")
            if llm is None or not llm.loaded:
                self._send_json(503, {"error": "LLM service not loaded"})
                return

            agent = Agent(
                llm, srv.tool_registry, srv.config,
                system_prompt=lambda: build_system_prompt(
                    srv.db, srv.orchestrator, srv.tool_registry, srv.services
                ),
            )
            try:
                response = agent.chat(message)
                self._send_json(200, {"response": response})
            except Exception as e:
                logger.error(f"/chat error: {e}")
                self._send_json(500, {"error": str(e)})
            return

        self._send_json(404, {"error": "Not found"})


def start_api_server(tool_registry, db, config, services, orchestrator) -> HTTPServer:
    """
    Start the API server on a daemon thread. Returns the HTTPServer instance.
    """
    port = config.get("api_port", 5123)
    token = config.get("api_token", "")

    server = HTTPServer(("127.0.0.1", port), _Handler)
    server.tool_registry = tool_registry
    server.db = db
    server.config = config
    server.services = services
    server.orchestrator = orchestrator
    server.api_token = token

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info(f"API server listening on http://127.0.0.1:{port}")
    return server
