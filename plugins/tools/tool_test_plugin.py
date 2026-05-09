import subprocess
import sys
from pathlib import Path

from paths import ROOT_DIR
from plugins.BaseTool import BaseTool, ToolResult
from plugins.helpers.plugin_paths import plugin_info, resolve_plugin_path
from plugins.plugin_discovery import load_single_plugin, unload_plugin


class TestPlugin(BaseTool):
    name = "test_plugin"
    description = (
        "Validate a plugin source file and run the project pytest suite. Use this "
        "while authoring plugins to get naming, folder, import, contract, and test feedback. "
        "This tool does not register or unregister live plugins."
    )
    parameters = {
        "type": "object",
        "properties": {
            "plugin_path": {"type": "string", "description": "Path to the plugin file to validate."},
        },
        "required": ["plugin_path"],
    }
    requires_services = []
    max_calls = 5
    background_safe = False

    def run(self, context, **kwargs) -> ToolResult:
        path, err = resolve_plugin_path((kwargs.get("plugin_path") or "").strip())
        if err:
            return ToolResult.failed(err)
        info, err = plugin_info(path)
        if err:
            return ToolResult.failed(err)
        if not path.exists():
            return ToolResult.failed(f"Plugin file not found: {path}")

        loaded_name, load_error = _try_load(info.plugin_type, path, getattr(context, "config", {}) or {})
        pytest_result = _run_pytest(getattr(context, "config", {}) or {})
        ok = load_error is None and pytest_result["returncode"] == 0
        summary = [
            f"Plugin path: {path}",
            f"Plugin type: {info.plugin_type}",
            f"Load check: {'ok: ' + loaded_name if load_error is None else 'failed: ' + load_error}",
            f"Pytest: {'passed' if pytest_result['returncode'] == 0 else 'failed'}",
        ]
        if pytest_result["summary"]:
            summary.append(pytest_result["summary"])
        return ToolResult(
            success=ok,
            error="" if ok else "Plugin test failed.",
            data={"plugin_path": str(path), "plugin_type": info.plugin_type, "load_error": load_error, "pytest": pytest_result},
            llm_summary="\n".join(summary),
        )


def _try_load(plugin_type: str, path: Path, config: dict) -> tuple[str | None, str | None]:
    tool_registry = _ToolRegistry()
    orchestrator = _TaskRegistry()
    command_registry = _CommandRegistry()
    frontend_manager = _FrontendManager()
    services = {}
    name, error = load_single_plugin(
        plugin_type, path,
        tool_registry=tool_registry,
        orchestrator=orchestrator,
        services=services,
        config=dict(config),
        command_registry=command_registry,
        frontend_manager=frontend_manager,
    )
    unload_plugin(
        plugin_type, name or "",
        tool_registry=tool_registry,
        orchestrator=orchestrator,
        services=services,
        source_path=str(path),
        command_registry=command_registry,
        frontend_manager=frontend_manager,
    )
    return name, error


def _run_pytest(config: dict) -> dict:
    timeout = int(config.get("plugin_test_timeout", 120))
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests"],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
        return {"returncode": proc.returncode, "summary": _tail(output), "timed_out": False}
    except subprocess.TimeoutExpired as e:
        output = ((e.stdout or "") + "\n" + (e.stderr or "")).strip()
        return {"returncode": 124, "summary": f"pytest timed out after {timeout}s\n{_tail(output)}".strip(), "timed_out": True}


def _tail(text: str, lines: int = 40) -> str:
    return "\n".join((text or "").splitlines()[-lines:])


class _ToolRegistry:
    def __init__(self):
        self.tools = {}

    def register(self, tool):
        self.tools[tool.name] = tool

    def unregister(self, name):
        self.tools.pop(name, None)


class _TaskRegistry:
    def __init__(self):
        self.tasks = {}

    def register_task(self, task):
        self.tasks[task.name] = task

    def unregister_task(self, name):
        self.tasks.pop(name, None)


class _CommandRegistry:
    def __init__(self):
        self._commands = {}

    def register(self, command):
        self._commands[command.name] = command

    def unregister(self, name):
        self._commands.pop(name, None)


class _FrontendManager:
    def __init__(self):
        self.adapters = {}

    def register(self, cls):
        self.adapters[cls.name] = cls()
        return None

    def unregister(self, name):
        self.adapters.pop(name, None)
