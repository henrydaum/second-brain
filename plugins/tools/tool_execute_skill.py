"""execute_skill: run a canvas skill (agent-facing adapter onto the new CanvasRuntime).

Routes the agent's "add a layer + render" request through the new canvas
state machine: ``context.canvas.for_session(...)`` → ``handle_action("add_layer", ...)``
→ ``render_canvas(...)``. Old layered-canvas / cs.enact("run_skill") path is
gone from this tool; the conversation-side canvas system remains in place
but is no longer driven from here.
"""

from __future__ import annotations

import logging

from canvas.render import bus_progress, render_canvas
from plugins.BaseTool import BaseTool, ToolResult
from plugins.skills.helpers import skill_error_log, skill_scoring
from plugins.skills.helpers.skill_runner import SkillRunError

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

		canvas_rt = getattr(context, "canvas", None)
		skill_registry = getattr(context, "skill_registry", None)
		if canvas_rt is None:
			return ToolResult.failed("canvas runtime not available on context")
		if skill_registry is None:
			return ToolResult.failed("skill registry not available on context")

		# Resolve the skill so we know its kind (creation vs transform) and
		# can fail fast on unknown slugs before mutating state.
		skill_inst = skill_registry.get(slug)
		if skill_inst is None:
			return ToolResult.failed(f"unknown skill: '{slug}'")
		kind = getattr(skill_inst, "kind", None) or "creation"

		cs = canvas_rt.for_session(session_key)

		# A transform with an empty chain has nothing to read — refuse
		# before we corrupt state. Mirrors the SkillRunError the renderer
		# would raise, but at the action layer.
		if kind == "transform" and not cs.canvas.layers:
			return ToolResult.failed(
				"Transform skills require a creation first. "
				"Run a creation skill before this transform."
			)

		add_result = canvas_rt.handle_action(cs.canvas_id, "add_layer", {
			"skill_slug": slug,
			"kind": kind,
			"controls": params,
		})
		if not add_result.ok:
			msg = add_result.error.message if add_result.error else "add_layer failed"
			skill_error_log.record_error(
				getattr(context, "db", None), slug, params,
				{"error_type": "AddLayerFailed", "message": msg},
				session_key=session_key,
			)
			return ToolResult.failed(msg)

		# Render the chain. On failure, roll the layer back so the next
		# call to the agent sees the pre-failure state.
		try:
			render_result = render_canvas(
				cs,
				skill_loader=skill_registry.get_record,
				db=getattr(context, "db", None),
				on_event=bus_progress(getattr(context, "session_key", None), float((getattr(context, "config", {}) or {}).get("skill_timeout_s") or 30)),
			)
		except SkillRunError as e:
			canvas_rt.handle_action(cs.canvas_id, "remove_layer", {"chain_index": len(cs.canvas.layers) - 1})
			diag = dict(getattr(e, "diagnostic", None) or {"error_type": "SkillRunError", "message": str(e)})
			skill_error_log.record_error(getattr(context, "db", None), slug, params, diag, session_key=session_key)
			return ToolResult.failed(str(e))
		except Exception as e:
			logger.exception("execute_skill render crashed: slug=%s", slug)
			canvas_rt.handle_action(cs.canvas_id, "remove_layer", {"chain_index": len(cs.canvas.layers) - 1})
			return ToolResult.failed(str(e))

		snap = {
			"path": str(render_result.image_path),
			"chain": list(cs.canvas.layers),
			"size": cs.canvas.size,
			"palette_id": cs.canvas.palette_id,
			"canvas_id": cs.canvas_id,
			"pool_hash": render_result.pool_hash,
			"seed": render_result.seed,
			"cache_hit": render_result.cache_hit,
		}
		skill_scoring.record_event(
			getattr(context, "db", None), "generate", snap["chain"], snap["path"],
		)
		return ToolResult(
			data={"canvas": snap, "chain": snap["chain"]},
			llm_summary=f"Executed skill '{slug}' on the canvas.",
			attachment_paths=[snap["path"]],
		)
