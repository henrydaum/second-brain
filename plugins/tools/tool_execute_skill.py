"""execute_skill: run a canvas skill (agent-facing adapter onto the RunSkill action)."""

from __future__ import annotations

import logging

from plugins.BaseTool import BaseTool, ToolResult
from plugins.skills.helpers import skill_error_log, skill_scoring
from plugins.tools.helpers import layered_canvas as lc

logger = logging.getLogger("SkillTools")


class ExecuteSkill(BaseTool):
    name = "execute_skill"
    description = "Run a stored skill on the canvas by slug. Creations start a new chain from a blank palette-background image; transforms read the current canvas and require something already on it. Chain cap is 4 layers (1 creation + up to 3 transforms). Errors include a hint line — read it and adjust before retrying."
    max_calls = 6
    parameters = {
        "type": "object",
        "properties": {"slug": {"type": "string"}, "params": {"type": "object", "default": {}}},
        "required": ["slug"],
    }
    config_settings = [
        ("Skill Execution Timeout (s)", "skill_timeout_s",
         "Wall-clock seconds before a single skill run is killed. Raise for heavy compute; lower to catch runaway loops sooner.",
         30,
         {"type": "slider", "range": (5, 180, 35), "is_float": False}),
        ("Skill Memory Cap (MB)", "skill_memory_mb",
         "Per-skill address-space limit for the sandbox subprocess. Linux-only — the cap is set via RLIMIT_AS and is ignored on Windows / macOS.",
         768,
         {"type": "slider", "range": (256, 4096, 30), "is_float": False}),
    ]

    def run(self, context, **kwargs) -> ToolResult:
        session_key = getattr(context, "session_key", None) or "local"
        slug = str(kwargs.get("slug") or "")
        params = dict(kwargs.get("params") or {})
        runtime = getattr(context, "runtime", None)
        if runtime is None:
            return ToolResult.failed("runtime not bound; cannot enact run_skill")
        session = runtime.get_session(session_key) if hasattr(runtime, "get_session") else None
        cs = getattr(session, "cs", None) if session else None
        if cs is None:
            return ToolResult.failed("no live conversation state for this session")
        try:
            result = cs.enact("run_skill", {"slug": slug, "params": params}, actor_id="agent")
        except Exception as e:
            logger.exception("execute_skill enact crashed: slug=%s", slug)
            return ToolResult.failed(str(e))
        if not result.ok:
            msg = (result.error.message if result.error else (result.message or "execute_skill failed"))
            skill_error_log.record_error(getattr(context, "db", None), slug, params,
                                         {"error_type": "RunSkillFailed", "message": msg},
                                         session_key=session_key)
            return ToolResult.failed(msg)
        snap = (result.data or {}).get("canvas") or lc.canvas(session_key) or {}
        final = snap.get("path")
        chain = snap.get("chain") or []
        skill_scoring.record_event(getattr(context, "db", None), "generate", chain, final)
        attach = [final] if final else []
        return ToolResult(
            data={"canvas": snap, "chain": chain},
            llm_summary=f"Executed skill '{slug}' on the canvas.",
            attachment_paths=attach,
        )
