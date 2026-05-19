"""search_skills: find reusable canvas skills."""

from __future__ import annotations

import logging
from functools import partial

from plugins.BaseTool import BaseTool, ToolResult
from plugins.skills.helpers import skill_scoring

logger = logging.getLogger("SearchSkills")


class SearchSkills(BaseTool):
    name = "search_skills"
    description = "Find existing canvas skills by intent before writing a new one. The built-in library covers most common subjects (fractals, L-systems, attractors, flow fields, Voronoi, waves); a strong match lets you go search → execute in one shot. Always try this before create_skill."
    max_calls = 4
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "default": 5},
            "kind": {"type": "string", "enum": ["creation", "transform"]},
        },
        "required": ["query"],
    }

    def run(self, context, **kwargs) -> ToolResult:
        query = str(kwargs.get("query") or "")
        kind = kwargs.get("kind")
        top_k = int(kwargs.get("top_k") or 5)
        registry = getattr(context, "skill_registry", None)
        if registry is None:
            return ToolResult.failed("skill registry not available")
        try:
            score_lookup = partial(skill_scoring.get_scores, getattr(context, "db", None))
            rows = registry.search(
                query, top_k=top_k, kind=kind,
                text_embedder=context.services.get("text_embedder"),
                score_lookup=score_lookup,
            )
        except Exception as e:
            logger.exception("search_skills failed: query=%r kind=%r top_k=%r", query, kind, top_k)
            return ToolResult.failed(f"search failed ({type(e).__name__}: {e}). Check Second Brain logs for the full traceback.")
        if not rows:
            return ToolResult(data=[], llm_summary="No matching skills found.")
        summary = "Matching skills:\n" + "\n".join(
            f"- {r['slug']} ({r['kind']}, score {r['score']:.2f}): {r['description']}" for r in rows
        )
        return ToolResult(data=rows, llm_summary=summary)
