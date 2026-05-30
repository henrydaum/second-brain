"""execute_technique: run a canvas technique (agent-facing adapter onto the new CanvasRuntime).

Routes the agent's "add a layer + render" request through the new canvas
state machine: ``context.canvas.for_session(...)`` → ``handle_action("add_layer", ...)``
→ ``render_canvas(...)``. Old layered-canvas / cs.enact("run_technique") path is
gone from this tool; the conversation-side canvas system remains in place
but is no longer driven from here.
"""

from __future__ import annotations

import logging

from events.event_bus import bus
from events.event_channels import CANVAS_CHANGED
from canvas.render import bus_progress, render_canvas
from plugins.BaseTool import BaseTool, ToolResult
from plugins.techniques.helpers import technique_error_log, technique_scoring
from plugins.techniques.helpers.technique_runner import TechniqueRunError

logger = logging.getLogger("TechniqueTools")


class ExecuteTechnique(BaseTool):
	name = "execute_technique"
	description = "Run a stored technique on the canvas by slug. Backgrounds start a new chain from a blank palette-background image; filters read the current canvas and replace it; objects read the current canvas and alpha-composite an overlay onto it. filters and objects both require something already on the canvas. Chain cap is 4 layers (1 background + up to 3 filters/objects). Errors include a hint line — read it and adjust before retrying."
	max_calls = 6
	parameters = {
		"type": "object",
		"properties": {"slug": {"type": "string"}, "params": {"type": "object", "default": {}}},
		"required": ["slug"],
	}
	config_settings = [
		("Technique Execution Timeout (s)", "technique_timeout_s",
		 "Wall-clock seconds before a single technique run is killed. Raise for heavy compute; lower to catch runaway loops sooner.",
		 30,
		 {"type": "slider", "range": (5, 180, 35), "is_float": False}),
		("Technique Memory Cap (MB)", "technique_memory_mb",
		 "Per-technique address-space limit for the sandbox subprocess. Linux-only — the cap is set via RLIMIT_AS and is ignored on Windows / macOS.",
		 768,
		 {"type": "slider", "range": (256, 4096, 30), "is_float": False}),
	]

	def run(self, context, **kwargs) -> ToolResult:
		session_key = getattr(context, "session_key", None) or "local"
		slug = str(kwargs.get("slug") or "")
		params = dict(kwargs.get("params") or {})

		canvas_rt = getattr(context, "canvas", None)
		technique_registry = getattr(context, "technique_registry", None)
		if canvas_rt is None:
			return ToolResult.failed("canvas runtime not available on context")
		if technique_registry is None:
			return ToolResult.failed("technique registry not available on context")

		# Resolve the technique so we know its kind (background vs filter) and
		# can fail fast on unknown slugs before mutating state.
		technique_inst = technique_registry.get(slug)
		if technique_inst is None:
			return ToolResult.failed(f"unknown technique: '{slug}'")
		kind = getattr(technique_inst, "kind", None) or "background"

		cs = canvas_rt.for_session(session_key)

		# A filter or object with an empty chain has nothing to read — refuse
		# before we corrupt state. Mirrors the TechniqueRunError the renderer
		# would raise, but at the action layer.
		if kind in ("filter", "object") and not cs.canvas.layers:
			return ToolResult.failed(
				f"{kind.title()} techniques require a background first. "
				f"Run a background technique before this {kind}."
			)

		credits = (getattr(context, "services", None) or {}).get("credits")
		db = getattr(context, "db", None)
		try:
			add_result, render_result = canvas_rt.render_actions(
				cs.canvas_id,
				[("add_layer", {"technique_slug": slug, "kind": kind, "controls": params})],
				lambda state: render_canvas(
					state,
					technique_loader=technique_registry.get_record,
					db=db,
					on_event=bus_progress(getattr(context, "session_key", None), float((getattr(context, "config", {}) or {}).get("technique_timeout_s") or 30)),
					worker_pool=(getattr(context, "services", None) or {}).get("technique_worker_pool"),
					authorize_uncached=credits.render_authorizer(db, session_key, allow_prompt_overrun=True) if credits and db and session_key.startswith("web:") else None,
				),
			)
			if not add_result.ok:
				msg = add_result.error.message if add_result.error else "add_layer failed"
				technique_error_log.record_error(db, slug, params, {"error_type": "AddLayerFailed", "message": msg}, session_key=session_key)
				return ToolResult.failed(msg)
		except TechniqueRunError as e:
			diag = dict(getattr(e, "diagnostic", None) or {"error_type": "TechniqueRunError", "message": str(e)})
			technique_error_log.record_error(getattr(context, "db", None), slug, params, diag, session_key=session_key)
			return ToolResult.failed(str(e))
		except Exception as e:
			logger.exception("execute_technique render crashed: slug=%s", slug)
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
			"warning": render_result.warning,
			"warning_message": render_result.warning_message,
		}
		technique_scoring.record_event(
			getattr(context, "db", None), "generate", snap["chain"], snap["path"],
		)
		bus.emit(CANVAS_CHANGED, {"session_key": session_key, "action": "add_layer", "canvas": snap})
		layer_index = len(cs.canvas.layers)  # newly-added layer's 1-based position
		total = len(cs.canvas.layers)
		cache_tag = ", cached" if render_result.cache_hit else ""
		summary = (
			f"Executed {kind} technique '{slug}' (layer {layer_index}/{total}, "
			f"seed={render_result.seed}{cache_tag})."
		)
		if render_result.warning:
			summary += (
				f"\nWARNING ({render_result.warning}): "
				f"{render_result.warning_message or 'post-render validator flagged this layer'}"
			)
		return ToolResult(
			data={"canvas": snap, "chain": snap["chain"]},
			llm_summary=summary,
			attachment_paths=[snap["path"]],
		)
