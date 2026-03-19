"""
System prompt builder.

Assembles a concise system prompt for the agent that includes both a static
explanation of how Second Brain works and dynamic runtime state (tables,
task status, services, file inventory). Built fresh each time the user enters
chat mode so the LLM always sees current state.
"""

from pathlib import Path

from Stage_1.registry import get_supported_extensions, _MODALITY_MAP

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SKIP_DIRS = {"__pycache__", ".git", ".venv", "venv"}


def build_system_prompt(db, orchestrator, tool_registry, services: dict) -> str:
    sections = [
        _identity(),
        _architecture(),
        _database_tables(db),
        _pipeline_status(db, orchestrator),
        _services_status(services),
        _available_tools(tool_registry),
        _file_inventory(db),
        _source_files(),
        _authoring_guidance(),
    ]
    return "\n\n".join(s for s in sections if s)


# ── Static sections ──────────────────────────────────────────────────

def _identity() -> str:
    return (
        "You are the Second Brain assistant — an AI embedded in a local file intelligence system. "
        "You have tools to search and query a database of user files. "
        "Be concise and always cite which files your answers come from."
    )


def _architecture() -> str:
    return (
        "## How Second Brain works\n"
        "The system watches local directories and processes every file through a four-stage pipeline:\n"
        "\n"
        "Stage 0 — Services: shared resources (LLM, text/image embedders, OCR) with a load/unload lifecycle. "
        "Parsers, tasks, and tools can use services; the orchestrator waits until they're loaded.\n"
        "\n"
        "Stage 1 — Parsers: convert any supported file into a standardised ParseResult. "
        "Each file extension maps to a default modality (text, image, audio, video, tabular, container). "
        "Some files contain multiple modalities (e.g. a PDF with embedded images); the parser reports "
        "these via also_contains so the pipeline can process them too.\n"
        "\n"
        "Stage 2 — Tasks: background workers that run on every file. Each task declares the database "
        "tables it reads from and writes to. The orchestrator derives a dependency DAG from these "
        "declarations automatically — no explicit wiring. When an upstream task completes, downstream "
        "tasks that read its output are enqueued if all of their dependencies are met. "
        "Task results get written to their SQLite output tables.\n"
        "\n"
        "Stage 3 — Tools: the on-demand query layer (where you operate). Tools accept arguments, "
        "query the database or call other tools, and return structured results. Your available tools are provided via function calling. The tool call limit is per-message, not per-session. If you run out of one tool, you can call another.\n"
        "\n"
        "Both tasks and tools receive a SecondBrainContext giving them access to the database, config, "
        "services, the parser, and (for tools) the ability to call other tools."
    )


def _authoring_guidance() -> str:
    return (
        "## Extending the system\n"
        "You can help the user design new tasks and tools. The key interfaces (BaseTask, BaseTool, "
        "TaskResult, ToolResult, SecondBrainContext) will be provided in full when needed via "
        "dedicated creation tools. For now, understand that tasks declare reads/writes/modalities "
        "and tools declare name/description/parameters as a JSON schema."
    )


# ── Dynamic sections ─────────────────────────────────────────────────

def _database_tables(db) -> str:
    try:
        result = db.query(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        names = [row[0] for row in result["rows"]]
    except Exception:
        names = []

    if not names:
        return "## Database\nNo tables yet."

    return "## Database tables (for more info, call the sql_query tool)\n" + ", ".join(names)


def _pipeline_status(db, orchestrator) -> str:
    lines = ["## Task pipeline"]

    # DAG
    dag = orchestrator.dependency_pipeline_graph()
    if dag:
        lines.append(dag)

    # Per-task status counts
    stats = db.get_system_stats().get("tasks", {})
    if stats:
        lines.append("")
        lines.append("Status (P=pending, D=done, F=failed):")
        for name, counts in sorted(stats.items()):
            paused = " [PAUSED]" if name in orchestrator.paused else ""
            lines.append(
                f"  {name}: P:{counts['PENDING']} D:{counts['DONE']} F:{counts['FAILED']}{paused}"
            )

    return "\n".join(lines)


def _services_status(services: dict) -> str:
    if not services:
        return ""
    parts = []
    for name, svc in services.items():
        status = "loaded" if getattr(svc, "loaded", False) else "unloaded"
        parts.append(f"{name} ({status})")
    return "## Services\n" + ", ".join(parts)


def _available_tools(tool_registry) -> str:
    if not tool_registry:
        return ""
    enabled = [t for t in tool_registry.tools.values() if t.agent_enabled]
    disabled = [t for t in tool_registry.tools.values() if not t.agent_enabled]
    lines = ["## Your tools"]
    if enabled:
        lines.append("These are the ONLY tools you can call via function calling:")
        for tool in enabled:
            desc = (tool.description or "").split("\n")[0]
            lines.append(f"- **{tool.name}**: {desc}")
    else:
        lines.append("No tools are currently enabled.")
    if disabled:
        lines.append("")
        lines.append(
            "The following tools exist but are DISABLED (agent_enabled=False). "
            "You cannot call them directly. They may be used internally by other tools."
        )
        for tool in disabled:
            lines.append(f"- ~~{tool.name}~~")
    return "\n".join(lines)


def _file_inventory(db) -> str:
    file_stats = db.get_system_stats().get("files", {})
    total = sum(file_stats.values()) if file_stats else 0

    lines = ["## File inventory"]

    if file_stats:
        parts = [f"{count} {mod}" for mod, count in sorted(file_stats.items())]
        lines.append(", ".join(parts) + f" ({total} total)")
    else:
        lines.append("No files indexed yet.")

    # Supported extensions
    exts = sorted(get_supported_extensions())
    if exts:
        lines.append("Supported extensions: " + " ".join(exts))

    return "\n".join(lines)


def _source_files() -> str:
    names = sorted({
        p.stem for p in _PROJECT_ROOT.rglob("*")
        if p.suffix in (".py", ".pyw")
        and not any(part in _SKIP_DIRS for part in p.parts)
    })
    if not names:
        return ""
    return "## Source code (use read_source_code to view)\n" + ", ".join(names)
