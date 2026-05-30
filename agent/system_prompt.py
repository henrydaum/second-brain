"""System prompt assembly.

Produces a single merged system message. MiniMax (the production provider)
rejects requests with more than one system message or any system message
not at index 0, so the historical static / semi-stable / dynamic split
that fed prompt caching is collapsed here.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from plugins.services.helpers.parser_registry import get_supported_extensions
from runtime.agent_scope import AgentScope

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_STATIC_PROMPT_PATH = Path(__file__).with_name("system_prompt_static.md")
_ART_ENCYCLOPEDIA_PATH = Path(__file__).with_name("generative_art_encyclopedia.md")


def _static_prompt() -> str:
    return _STATIC_PROMPT_PATH.read_text(encoding="utf-8").strip()


def build_prompt_sections(
    db,
    orchestrator,
    tool_registry,
    services: dict,
    *,
    scope: AgentScope | None = None,
    profile_name: str = "default",
    extra_suffix: str = "",
    commands=None,
    config: dict | None = None,
    conversation_metadata: dict[str, Any] | None = None,
    prompt_extras: dict[str, Any] | None = None,
    notification_suffix: str = "",
    session_key: str | None = None,
) -> list[dict[str, str]]:
    """Build ordered system prompt messages."""
    r = tool_registry
    semi = [
        _tool_catalog(r),
        _command_catalog(commands),
        _authoring_guidance(r) if _has_tool(r, "test_plugin") else "",
        _sandbox_files() if _has_tool(r, "test_plugin") else "",
        _attachments() if _has_tool(r, "sql_query") else "",
        _database_tables(db) if _has_tool(r, "sql_query") else "",
        _generative_art_encyclopedia() if _has_tool(r, "execute_technique") else "",
        _palette_catalog(r) if _has_tool(r, "execute_technique") else "",
        _technique_workflow(r) if _has_tool(r, "execute_technique") else "",
    ]
    dynamic = [
        _current_datetime(),
        _model_status(services),
        _profile_status(profile_name, scope),
        _canvas_state(services, session_key) if _has_tool(r, "execute_technique") else "",
        # _services_status(services),
        # _pipeline_status(db, orchestrator),
        # _sync_dirs(config),
        _file_inventory(db) if _has_any(r, "read_file", "hybrid_search", "lexical_search") else "",
        # _agent_memory(),
        _conversation_metadata(conversation_metadata) if _has_tool(r, "sql_query") else "",
        _prompt_extras(prompt_extras),
        notification_suffix,
        getattr(scope, "prompt_suffix", "") if scope else "",
        extra_suffix,
    ]
    parts = [_static_prompt(), *[s for s in semi if s], *[s for s in dynamic if s]]
    return [{"role": "system", "content": "\n\n".join(p.strip() for p in parts if p).strip()}]


def _generative_art_encyclopedia() -> str:
    try:
        return _ART_ENCYCLOPEDIA_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _palette_catalog(registry=None) -> str:
    """List the available palettes with their colors so the agent can match user color requests to a palette id."""
    try:
        from plugins.helpers.palettes import list_palettes
        palettes = list_palettes()
    except Exception:
        return ""
    if not palettes:
        return ""
    lines = [
        "## Available palettes",
        "The canvas has a fixed catalog of palettes. To change the palette on a layer, the layer's technique must expose a `palette` control" + (" (use read_technique on its slug to check — look for `palette = Palette()`)" if _has_tool(registry, "read_technique") else "") + ". Then call `manage_layers` with `action=set_control`, `chain_index=<n>`, `name=\"palette\"`, `value=\"<id>\"`. Use the `id` slug below, not the display name. Match a user's color request (e.g. \"red/orange/yellow\", \"neon\", \"earthy\") to the closest palette by its kind and hex colors.",
        "",
        "Format: `id (Name) — kind: primary secondary tertiary accent / bg background`",
        "",
    ]
    for p in palettes:
        lines.append(f"- {p.id} ({p.name}) — {p.kind}: {p.primary} {p.secondary} {p.tertiary} {p.accent} / bg {p.background}")
    return "\n".join(lines)


def build_system_prompt(*args, **kwargs) -> str:
    """Compatibility wrapper for old callers that expect one system string."""
    return "\n\n".join(m["content"] for m in build_prompt_sections(*args, **kwargs) if m.get("content"))


def _system_message(title: str, content: str) -> dict[str, str]:
    return {"role": "system", "content": f"[{title}]\n{content.strip()}"}


def _has_tool(registry, name: str) -> bool:
    return bool(registry) and name in getattr(registry, "tools", {})


def _has_any(registry, *names: str) -> bool:
    return any(_has_tool(registry, n) for n in names)


def _technique_authoring_tools(registry) -> list[str]:
    return [n for n in ("create_technique", "update_technique", "delete_technique") if _has_tool(registry, n)]


def _current_datetime() -> str:
    return f"Current date and time: {datetime.now().strftime('%A, %B %d, %Y %I:%M %p')}"


def _model_status(services: dict) -> str:
    llm = (services or {}).get("llm")
    if not llm:
        return "Current model: unavailable."
    name = getattr(llm, "_active_name", None)
    inner = getattr(llm, "active", None)
    model = getattr(inner, "model_name", None) if inner else getattr(llm, "model_name", None)
    return "Current model: " + (f"{name} ({model})." if name and model else f"{name or model or 'unknown'}.")


def _profile_status(profile_name: str, scope: AgentScope | None) -> str:
    suffix = " Tool access is profile-limited." if scope and scope.has_tool_filter else ""
    return f"Active agent profile: {profile_name or 'default'}.{suffix}"


def _tool_catalog(tool_registry) -> str:
    lines = ["## Available tool catalog"]
    if not tool_registry:
        return "\n".join([*lines, "No tool registry is currently available."])
    schemas = tool_registry.get_all_schemas() if hasattr(tool_registry, "get_all_schemas") else []
    if not schemas:
        return "\n".join([*lines, "No tools are currently registered."])
    for schema in schemas:
        fn = schema.get("function", schema)
        desc = (fn.get("description") or "").strip().replace("\n", " ")
        lines.append(f"- {fn.get('name')}: {desc}" if desc else f"- {fn.get('name')}")
    return "\n".join(lines)


def _command_catalog(commands) -> str:
    lines = ["## Available slash commands"]
    entries = []
    try:
        entries = commands.visible_commands() if hasattr(commands, "visible_commands") else []
    except Exception:
        entries = []
    if entries:
        for cmd in entries:
            desc = (getattr(cmd, "description", "") or "").strip()
            hint = _form_hint(getattr(cmd, "form", None), commands)
            lines.append(f"- /{cmd.name}{(' ' + hint) if hint else ''}: {desc}" if desc else f"- /{cmd.name}{(' ' + hint) if hint else ''}")
        return "\n".join(lines)
    if isinstance(commands, dict) and commands:
        for name, spec in sorted(commands.items()):
            hint = _form_hint(getattr(spec, "form", None), None)
            lines.append(f"- /{name}{(' ' + hint) if hint else ''}")
        return "\n".join(lines)
    return "\n".join([*lines, "No slash-command catalog is available in this prompt."])


def _form_hint(form, commands=None) -> str:
    try:
        steps = form({}, commands.context(None) if commands and hasattr(commands, "context") else None) if callable(form) else (form or [])
    except Exception:
        steps = []
    return " ".join(f"<{s.name}>" if getattr(s, "required", True) else f"[{s.name}]" for s in steps)


def _plugin_contracts() -> str:
    return (
        """## Plugin contracts
Second Brain has six plugin families: tools, tasks, services, commands, frontends, and techniques.

Built-in plugins live under plugins/<family>. Sandbox plugins live in the matching DATA_DIR sandbox directory. Templates are the source of truth. To learn more about how they work, read the files directly.

Techniques are the canvas family: each technique is a `class X(BaseTechnique)` defining descriptor controls as class attributes and `def run(self, canvas)` to paint a background, apply a filter, or overlay an object onto the canvas. They run in a sandboxed subprocess with restricted imports — see templates/technique_template.py and plugins/BaseTechnique.py."""
    )


def _authoring_guidance(registry=None) -> str:
    from paths import DATA_DIR, ROOT_DIR
    can_techniques = bool(_technique_authoring_tools(registry))
    families = "tools, tasks, services, commands, frontends, and techniques" if can_techniques else "tools, tasks, services, commands, and frontends"
    technique_note = "Technique authoring should use the dedicated technique tools named in the canvas workflow." if can_techniques else "Technique authoring is outside this session's tool scope; do not claim you can create, edit, or delete techniques."
    return (
        f"""## Building plugins
You can extend Second Brain by authoring {families}.

Read the matching template in templates/, then write the plugin into {DATA_DIR}/sandbox_<family>/ with the required prefix, e.g. tool_foo.py in {DATA_DIR}/sandbox_tools/. The root directory is {ROOT_DIR}. Do not create sandbox plugins in the project root.

{technique_note}

Workflow:
1. Understand the user's intended behavior. Ask clarifying questions when a missing decision would materially change the design.
2. Read the relevant template with read_file.
3. Read a similar built-in or sandbox plugin when one exists.
4. Write the file into the correct sandbox directory using the file-editing tools.
5. Call test_plugin(plugin_path=...) after edits for naming, import, contract, and diagnostic feedback.
6. Treat pytest output as broad regression context, not proof that the plugin's behavior is correct.
7. If diagnostics, pytest, or watcher logs show a failure, edit the same file and test again.

Valid plugin files are loaded, reloaded, or unloaded as they change when plugin_watcher is loaded.
To remove a plugin from the live runtime, delete its file with the run_command tool.

Names must be unique across built-in and sandbox plugins. Config settings use (title, variable_name, description, default, type_info), are stored in plugin_config.json, and are read with context.config.get(key).

The context object is passed to every plugin and contains relevant runtime information and helper methods. Read its definition in runtime/context.py if you have questions about how to use it effectively in your plugin code."""
    )


def _technique_workflow(registry) -> str:
    can_search = _has_tool(registry, "search_techniques")
    can_read = _has_tool(registry, "read_technique")
    can_guide = _has_tool(registry, "read_technique_guide")
    can_execute = _has_tool(registry, "execute_technique")
    author_tools = _technique_authoring_tools(registry)
    lines = ["""## Canvas technique workflow
Your name is Second Brain. Second Brain makes generative, algorithmic art — not literal illustration. Treat every canvas request as a prompt for an abstract algorithmic interpretation, not a representational depiction. For example, a "flower" is a Vogel spiral of palette-blended cells, not stacked petals. Second Brain plays to their strengths: math, code, and procedural generation.

Workflow:
"""]
    steps = []
    if can_search:
        steps.append("Call `search_techniques` with the subject. Search returns only techniques visible to this session; when community techniques are disabled, community-authored techniques will not appear.")
    if can_read:
        steps.append("Use `read_technique` to inspect a promising hit or clone a nearby technique when authoring is enabled.")
    if can_execute:
        steps.append("If a strong match exists, call `execute_technique`.")
    if author_tools:
        steps.append(f"Technique authoring is enabled for this session. You may use the available authoring tools: {', '.join(author_tools)}.")
        if can_guide:
            steps.append("Call `read_technique_guide` once per session before the first new technique if you need the template, helpers, or composition guidance.")
        if _has_tool(registry, "create_technique"):
            steps.append("If no match exists, pick an algorithmic technique before writing code, then call `create_technique` with a complete BaseTechnique class and execute the returned slug.")
    else:
        steps.append("Technique authoring is not enabled for this session. Do not say you can create, edit, or delete techniques; work with search, read, execution, and layer controls that are available.")
    lines.extend(f"{i}. {s}" for i, s in enumerate(steps, 1))
    lines.append("""

Techniques (good-for hints — formulas live in the encyclopedia above):
- vogel_spiral -- flowers, sunflowers, galaxies, star fields, seed-pod patterns
- fbm / value_noise -- clouds, terrain, atmospheres, fog, organic textures, ray fields
- radial_falloff -- suns, moons, vignettes, centered radiant subjects
- flow_field -- wind, hair, currents, smoke, motion, fur, weather
- lindenmayer + turtle_segments -- trees, branches, ferns, coral, lightning, roots
- voronoi_assign -- cells, cracked glass, abstract portraits, stained glass, basalt
- wave_field -- water, ripples, sound, interference, reflections
- attractor_points (de Jong, Clifford) -- organic abstract forms, smoke, dust
- jittered_grid -- skylines, crowds, forests-from-far, tiled mosaics
- rule_of_thirds -- horizon and focal-point placement for any composition

When in doubt, prefer noise, gradients, and procedural patterns in palette tones over explicit shapes. Compose multiple techniques (e.g. fbm background + vogel foreground + radial vignette) rather than drawing literal features.

You always follow through. If you start authoring, updating, testing, or executing techniques, keep using the available tools until the canvas is rendered, a tool limit blocks you, or a specific user decision is required. Do not stop on status-only text like "let me try", "fixing now", or "I'll test this"; pair that narration with the actual tool call in the same response. Iteration is normal, but save "you can always say what to change" for the final rendered result.""")
    return "\n".join(lines)


def _canvas_state(services: dict | None, session_key: str | None) -> str:
    """Return a short summary of the user's current canvas for the agent prompt."""
    canvas_rt = (services or {}).get("canvas")
    if not session_key or canvas_rt is None or not hasattr(canvas_rt, "for_session"):
        return ""
    try:
        cs = canvas_rt.current_for_session(session_key) if hasattr(canvas_rt, "current_for_session") else None
        if cs is None:
            return ""
        layers = list(cs.canvas.layers or [])
        lines = [
            "## Current canvas",
            f"Canvas id: {cs.canvas_id}",
            f"Size: {cs.canvas.size}; Palette: {cs.canvas.palette_id}; Seed: {cs.render_seed if cs.render_seed is not None else 'not rendered yet'}",
            f"Layer count: {len(layers)}/6",
        ]
        if not layers:
            lines.append("Layers: none. The canvas is empty. Add or execute a background technique before filters or objects.")
        else:
            lines.append("Layers below are the source of truth. Use these zero-based indices for manage_layers; layer 0 is the background and later layers are filters or objects. Do not assume layers not listed here.")
            for i, layer in enumerate(layers):
                controls = layer.get("controls") or {}
                detail = "; controls " + json.dumps(controls, sort_keys=True, default=str) if controls else ""
                lines.append(f"- index {i}: slug={layer.get('slug')} kind={layer.get('kind')}{detail}")
        return "\n".join(lines)
    except Exception:
        return ""


def _sandbox_files() -> str:
    from paths import SANDBOX_COMMANDS, SANDBOX_FRONTENDS, SANDBOX_SERVICES, SANDBOX_TASKS, SANDBOX_TOOLS
    lines = []
    for sd in (SANDBOX_TOOLS, SANDBOX_TASKS, SANDBOX_SERVICES, SANDBOX_COMMANDS, SANDBOX_FRONTENDS):
        if sd.exists():
            lines.extend(f"  {p}" for p in sorted(sd.glob("*.py")) if not p.name.startswith("_"))
    return "## Sandbox plugins\n" + ("\n".join(lines) if lines else """## Sandbox plugins
None yet. When new sandbox plugins are made, they will show up here.""")


def _attachments() -> str:
    from paths import ATTACHMENT_CACHE
    return (
        f"""## Attachments
Files sent through frontends are saved to the attachment cache and indexed by the normal task pipeline. If they can be parsed into text, they will be added to the user message directly using a separate attachment parser system. You can extend this system by following the structure within attachments/parsers/.

To find recent attachments with sql_query:
SELECT path, file_name, mtime FROM files WHERE path LIKE '{ATTACHMENT_CACHE}%' ORDER BY mtime DESC LIMIT 10"""
    )


def _database_tables(db) -> str:
    try:
        names = [row[0] for row in db.query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")["rows"]]
    except Exception:
        names = []
    return "## Database tables (inspect with sql_query, if available)\n" + (", ".join(names) if names else "No tables yet.")


def _pipeline_status(db, orchestrator) -> str:
    lines = ["## Task pipeline"]
    try:
        dag = orchestrator.dependency_pipeline_graph() if orchestrator else None
        stats = db.get_system_stats().get("tasks", {}) if db else {}
    except Exception:
        dag, stats = None, {}
    if dag:
        lines.append(dag)
    if stats:
        lines += ["", "Status (P=pending, D=done, F=failed):"]
        paused = getattr(orchestrator, "paused", set()) if orchestrator else set()
        lines += [f"  {n}: P:{c['PENDING']} D:{c['DONE']} F:{c['FAILED']}{' [PAUSED]' if n in paused else ''}" for n, c in sorted(stats.items())]
    if len(lines) == 1:
        lines.append("No task status is currently available.")
    return "\n".join(lines)


def _services_status(services: dict) -> str:
    if not services:
        return "## Services\nNo services are currently registered."
    return "## Services\n" + ", ".join(f"{name} ({'loaded' if getattr(svc, 'loaded', False) else 'unloaded'})" for name, svc in sorted(services.items()))


def _sync_dirs(config: dict | None) -> str:
    dirs = (config or {}).get("sync_directories") or []
    return "## Sync directories\n" + ("\n".join(f"- {d}" for d in dirs) if dirs else "None configured.")


def _file_inventory(db) -> str:
    try:
        stats = db.get_system_stats().get("files", {}) if db else {}
    except Exception:
        stats = {}
    total = sum(stats.values()) if stats else 0
    lines = ["## File inventory", (", ".join(f"{c} {m}" for m, c in sorted(stats.items())) + f" ({total} total)") if stats else "No files indexed yet."]
    exts = sorted(get_supported_extensions())
    if exts:
        lines.append("Supported extensions: " + " ".join(exts))
    return "\n".join(lines)


def _agent_memory() -> str:
    from paths import DATA_DIR
    path = DATA_DIR / "memory.md"
    content = path.read_text() if path.exists() else "(empty)"
    return (
        f"""## Memory (from memory.md)
Path: {path}
Contains durable notes that persist across sessions. When the user asks Second Brain to remember something, write it down here. Store useful long-lived facts, preferences, project decisions, and lessons. Do not store trivial, stale, or unnecessarily sensitive details unless the user explicitly asks. Nightly, dream_memory may rewrite memory.md with reusable lessons and preferences.

Current contents:
{content}"""
    )


def _conversation_metadata(meta: dict[str, Any] | None) -> str:
    if not meta:
        return ""
    lines = "\n".join(["## Current conversation", f"Number: {meta.get('id')}", f"Category: {(meta.get('category') or '').strip() or 'Main'}", f"Title: {(meta.get('title') or '').strip() or 'New Conversation'}"])
    lines += "\nUse conversation IDs to query the 'conversations' and 'conversation_messages' tables. When a conversation gets too long, it will be compacted to save space. History prior to the compaction will still be available in the database, but won't be visible in the conversation context for new messages."
    return lines


def _prompt_extras(extras: dict[str, Any] | None) -> str:
    values = [v for v in (extras or {}).values() if isinstance(v, str) and v]
    return "\n\n".join(values)
