import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from paths import ROOT_DIR
from plugins.BaseTool import BaseTool, ToolResult
from plugins.helpers.plugin_paths import plugin_info, resolve_plugin_path
from plugins.plugin_discovery import load_single_plugin, unload_plugin


class TestPlugin(BaseTool):
    name = "test_plugin"
    description = (
        "Run purpose-built diagnostics for a plugin source file, then run the broad pytest "
        "regression suite. Use this while authoring plugins to get naming, folder, import, "
        "contract, and improvement suggestions. This tool does not register or unregister live plugins."
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

        loaded_name, load_error, diagnostics = _try_load(info.plugin_type, path, getattr(context, "config", {}) or {})
        pytest_result = _run_pytest(getattr(context, "config", {}) or {})
        diagnostic_errors = [d for d in diagnostics if d["level"] == "error"]
        ok = load_error is None and not diagnostic_errors and pytest_result["returncode"] == 0
        summary = [
            f"Plugin path: {path}",
            f"Plugin type: {info.plugin_type}",
            f"Load check: {'ok: ' + loaded_name if load_error is None else 'failed: ' + load_error}",
            "Diagnostics:",
        ]
        summary.extend(_format_diagnostics(diagnostics))
        summary.append(f"Pytest regression suite: {'passed' if pytest_result['returncode'] == 0 else 'failed'}")
        summary.append("Note: pytest checks whether the app still works; plugin diagnostics are the purpose-built signal for this file.")
        if pytest_result["summary"]:
            summary.append(pytest_result["summary"])
        return ToolResult(
            success=ok,
            error="" if ok else "Plugin test failed.",
            data={"plugin_path": str(path), "plugin_type": info.plugin_type, "load_error": load_error, "diagnostics": diagnostics, "pytest": pytest_result},
            llm_summary="\n".join(summary),
        )


def _try_load(plugin_type: str, path: Path, config: dict) -> tuple[str | None, str | None, list[dict]]:
    tool_registry = _ToolRegistry()
    orchestrator = _TaskRegistry()
    command_registry = _CommandRegistry()
    frontend_manager = _FrontendManager()
    services = {}
    state = SimpleNamespace(tool_registry=tool_registry, orchestrator=orchestrator, command_registry=command_registry, frontend_manager=frontend_manager, services=services)
    name, error = load_single_plugin(
        plugin_type, path,
        tool_registry=tool_registry,
        orchestrator=orchestrator,
        services=services,
        config=dict(config),
        command_registry=command_registry,
        frontend_manager=frontend_manager,
    )
    diagnostics = _diagnose(plugin_type, name, state) if error is None else []
    unload_plugin(
        plugin_type, name or "",
        tool_registry=tool_registry,
        orchestrator=orchestrator,
        services=services,
        source_path=str(path),
        command_registry=command_registry,
        frontend_manager=frontend_manager,
    )
    return name, error, diagnostics


def _diag(level: str, check: str, message: str, suggestion: str = "") -> dict:
    return {"level": level, "check": check, "message": message, "suggestion": suggestion}


def _format_diagnostics(items: list[dict]) -> list[str]:
    if not items:
        return ["- ok: Load diagnostics did not find additional issues."]
    lines = []
    for item in items:
        line = f"- {item['level']}: {item['check']}: {item['message']}"
        if item.get("suggestion"):
            line += f" Suggestion: {item['suggestion']}"
        lines.append(line)
    return lines


def _diagnose(plugin_type: str, loaded_name: str | None, state) -> list[dict]:
    checks = []
    items = _loaded_items(plugin_type, loaded_name, state)
    if not items:
        return [_diag("error", "registration", "The plugin loader reported success, but no registered object was found.", "Check the plugin name and registration path.")]
    for name, obj in items:
        checks += _common_checks(name, obj)
        checks += globals()[f"_diagnose_{plugin_type}"](obj)
    return checks or [_diag("ok", "contract", "No contract issues found.")]


def _loaded_items(plugin_type: str, loaded_name: str | None, state) -> list[tuple[str, object]]:
    if plugin_type == "tool":
        return list(state.tool_registry.tools.items())
    if plugin_type == "task":
        return list(state.orchestrator.tasks.items())
    if plugin_type == "command":
        return list(state.command_registry._commands.items())
    if plugin_type == "service":
        return list(state.services.items())
    if plugin_type == "frontend":
        return list(state.frontend_manager.adapters.items())
    return []


def _common_checks(name: str, obj) -> list[dict]:
    checks = []
    if not isinstance(name, str) or not name.strip():
        checks.append(_diag("error", "name", "Plugin registered with an empty name.", "Set a stable non-empty name."))
    if not getattr(obj, "_source_path", ""):
        checks.append(_diag("error", "source_path", "Plugin did not retain its source path.", "Let plugin_discovery set _source_path during load."))
    settings = getattr(obj, "config_settings", [])
    if not isinstance(settings, list) or any(not isinstance(x, tuple) or len(x) != 5 for x in settings):
        checks.append(_diag("error", "config_settings", "config_settings must be a list of 5-item tuples.", "Use (title, variable_name, description, default, type_info)."))
    return checks


def _diagnose_tool(tool) -> list[dict]:
    from plugins.BaseTool import BaseTool
    checks = []
    if not (tool.description or "").strip():
        checks.append(_diag("warning", "description", "Tool description is empty.", "Explain what the tool does, when to use it, and key limits."))
    params = getattr(tool, "parameters", None)
    if not isinstance(params, dict) or params.get("type") != "object" or not isinstance(params.get("properties", {}), dict):
        checks.append(_diag("error", "parameters", "Tool parameters must be an object JSON schema with properties.", "Use {'type': 'object', 'properties': {...}, 'required': [...]}."))
    elif any(req not in params.get("properties", {}) for req in params.get("required", [])):
        checks.append(_diag("error", "parameters.required", "A required parameter is missing from properties.", "Keep required names in sync with properties."))
    if tool.__class__.run is BaseTool.run:
        checks.append(_diag("error", "run", "Tool does not override run().", "Implement run(self, context, **kwargs) and return ToolResult."))
    return checks


def _diagnose_task(task) -> list[dict]:
    from plugins.BaseTask import BaseTask
    checks = []
    trigger = getattr(task, "trigger", "path")
    if trigger not in ("path", "event"):
        checks.append(_diag("error", "trigger", f"Unknown trigger '{trigger}'.", "Use trigger='path' or trigger='event'."))
    if trigger == "event":
        if task.__class__.run_event is BaseTask.run_event:
            checks.append(_diag("error", "run_event", "Event task does not override run_event().", "Implement run_event(run_id, payload, context)."))
        if not getattr(task, "trigger_channels", []):
            checks.append(_diag("error", "trigger_channels", "Event task has no trigger_channels.", "Declare at least one event bus channel."))
    elif task.__class__.run is BaseTask.run:
        checks.append(_diag("error", "run", "Path task does not override run().", "Implement run(paths, context)."))
    for attr in ("modalities", "reads", "writes", "requires_services"):
        if not isinstance(getattr(task, attr, None), list):
            checks.append(_diag("error", attr, f"{attr} must be a list.", f"Set {attr} = [] when unused."))
    if getattr(task, "writes", []) and not (getattr(task, "output_schema", "") or "").strip():
        checks.append(_diag("warning", "output_schema", "Task writes tables but has no output_schema.", "Add CREATE TABLE SQL for the tables in writes."))
    if trigger == "path" and not getattr(task, "reads", []) and not getattr(task, "modalities", []):
        checks.append(_diag("warning", "modalities", "Root path task has no modalities.", "Declare modalities so file discovery can root the task."))
    return checks


def _diagnose_service(service) -> list[dict]:
    from plugins.BaseService import BaseService
    checks = []
    if not isinstance(service, BaseService):
        checks.append(_diag("error", "base_class", "build_services returned a non-BaseService object.", "Return BaseService instances from build_services(config)."))
    if not (getattr(service, "model_name", "") or service.__class__.__name__).strip():
        checks.append(_diag("warning", "model_name", "Service has no display name.", "Set model_name to a human-readable name."))
    if not isinstance(getattr(service, "shared", True), bool):
        checks.append(_diag("error", "shared", "shared must be a boolean.", "Use shared=True for one shared instance or shared=False for per-call clients."))
    if service.__class__._load is BaseService._load:
        checks.append(_diag("error", "_load", "Service does not implement _load().", "Initialize resources in _load() and return True/False."))
    if service.__class__.unload is BaseService.unload:
        checks.append(_diag("error", "unload", "Service does not implement unload().", "Release resources safely in unload()."))
    return checks


def _diagnose_command(command) -> list[dict]:
    from plugins.BaseCommand import BaseCommand
    from state_machine.conversation import FormStep
    checks = []
    if not (command.description or "").strip():
        checks.append(_diag("warning", "description", "Command description is empty.", "Add a short user-facing description for help views."))
    if command.__class__.run is BaseCommand.run:
        checks.append(_diag("error", "run", "Command does not override run().", "Implement run(args, context)."))
    try:
        form = command.form({}, SimpleNamespace())
        if not isinstance(form, list) or any(not isinstance(step, FormStep) for step in form):
            checks.append(_diag("error", "form", "form() must return a list of FormStep objects.", "Return [] when the command has no form."))
    except Exception as e:
        checks.append(_diag("warning", "form", f"form() raised with empty args and diagnostic context: {e}", "If the form needs runtime context, guard missing context or keep this warning in mind during manual testing."))
    return checks


def _diagnose_frontend(frontend) -> list[dict]:
    from plugins.BaseFrontend import BaseFrontend, FrontendCapabilities
    checks = []
    if not (frontend.description or "").strip():
        checks.append(_diag("warning", "description", "Frontend description is empty.", "Describe the transport in one short sentence."))
    if not isinstance(getattr(frontend, "capabilities", None), FrontendCapabilities):
        checks.append(_diag("error", "capabilities", "capabilities must be a FrontendCapabilities instance.", "Set capabilities = FrontendCapabilities(...)."))
    for method in ("start", "stop", "session_key", "render_messages", "render_attachments", "render_form_field", "render_approval_request", "render_buttons", "render_error"):
        if getattr(frontend.__class__, method) is getattr(BaseFrontend, method):
            checks.append(_diag("error", method, f"Frontend does not override {method}().", f"Implement {method} for this transport."))
    return checks


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
