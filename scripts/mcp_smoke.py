"""Manual smoke test for the MCP client service against a real stdio server.

Spawns a tiny Python MCP server (no Node/npx needed) and drives it through
``MCPService`` exactly as the app would — verifying connection, tool discovery,
registration into a registry, a live tool call round-trip, and clean unload.

Run:  python scripts/mcp_smoke.py
Requires:  pip install mcp
"""

import sys
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plugins.services.service_mcp import MCPService  # noqa: E402

# A minimal MCP server exposing two tools. Written to a temp file and launched
# as a subprocess over stdio — the same transport real MCP servers use.
SERVER_SRC = textwrap.dedent('''
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("smoke")

    @mcp.tool()
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    @mcp.tool()
    def echo(text: str) -> str:
        """Echo text back."""
        return f"echo: {text}"

    if __name__ == "__main__":
        mcp.run()  # stdio transport by default
''')


class FakeRegistry:
    """Stand-in for the real ToolRegistry."""

    def __init__(self):
        self.tools = {}

    def register(self, tool):
        self.tools[tool.name] = tool

    def unregister(self, name):
        self.tools.pop(name, None)


def main() -> int:
    server_path = Path(tempfile.gettempdir()) / "sb_mcp_smoke_server.py"
    server_path.write_text(SERVER_SRC, encoding="utf-8")

    config = {
        "mcp_servers": {
            "smoke": {"command": sys.executable, "args": [str(server_path)]},
        },
        "mcp_tool_timeout": 30,
        "mcp_connect_timeout": 30,
    }

    registry = FakeRegistry()
    svc = MCPService(config)
    svc.bind_runtime(tool_registry=registry)

    ok = True
    try:
        print("1. load() — spawns the stdio server and lists its tools")
        loaded = svc.load()
        print(f"   loaded={loaded}  connected={svc.is_connected('smoke')}")
        print(f"   registered tools: {svc.registered_tools()}")
        ok &= loaded and svc.is_connected("smoke")
        ok &= set(svc.registered_tools()) == {"mcp__smoke__add", "mcp__smoke__echo"}

        print("2. call mcp__smoke__add(a=2, b=3)")
        add_tool = registry.tools.get("mcp__smoke__add")
        r1 = add_tool.run(None, a=2, b=3)
        print(f"   success={r1.success}  summary={r1.llm_summary!r}  data={r1.data!r}")
        ok &= r1.success and "5" in (r1.llm_summary or "")

        print("3. call mcp__smoke__echo(text='hi')")
        r2 = registry.tools["mcp__smoke__echo"].run(None, text="hi")
        print(f"   success={r2.success}  summary={r2.llm_summary!r}")
        ok &= r2.success and "echo: hi" in (r2.llm_summary or "")

        print("4. unload() — disconnects and unregisters")
        svc.unload()
        print(f"   loaded={svc.loaded}  registry tools={list(registry.tools)}")
        ok &= (not svc.loaded) and not registry.tools
    finally:
        try:
            if svc.loaded:
                svc.unload()
        finally:
            server_path.unlink(missing_ok=True)

    print("\n" + ("SMOKE TEST PASSED ✓" if ok else "SMOKE TEST FAILED ✗"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
