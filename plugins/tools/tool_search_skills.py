"""Semantic search over embedded canvas skills."""

from __future__ import annotations

import sqlite3

import numpy as np

from plugins.BaseTool import BaseTool, ToolResult


class SearchSkills(BaseTool):
    name = "search_skills"
    description = "Search stored canvas skills semantically by embedding a query and ranking skill name + description."
    max_calls = 6
    background_safe = True
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural-language skill search query."},
            "slug": {"type": "string", "description": "Deprecated alias for query."},
            "limit": {"type": "integer", "default": 5, "minimum": 1, "maximum": 10},
        },
    }

    def run(self, context, **kwargs) -> ToolResult:
        query = str(kwargs.get("query") or kwargs.get("slug") or "").strip()
        if not query:
            return ToolResult.failed("query is required")
        db = getattr(context, "db", None)
        embedder = (getattr(context, "services", {}) or {}).get("text_embedder")
        if db is None:
            return ToolResult.failed("database not available")
        if embedder is None:
            return ToolResult.failed("text_embedder service unavailable")
        q = _norm(embedder.encode(query))
        if q is None:
            return ToolResult.failed("text_embedder returned no embedding")
        try:
            rows = _rows(db)
        except sqlite3.OperationalError:
            return ToolResult.failed("skill embeddings are not ready yet; let embed_skills run first")
        limit = max(1, min(10, int(kwargs.get("limit") or 5)))
        scored = []
        for row in rows:
            vec = np.frombuffer(row["embedding"], dtype="<f4")
            if vec.size == q.size:
                scored.append(({k: row[k] for k in ("slug", "name", "description", "kind")}, float(np.dot(q, vec))))
        scored.sort(key=lambda item: item[1], reverse=True)
        skills = [{**meta, "score": round(score, 4)} for meta, score in scored[:limit]]
        if not skills:
            return ToolResult.failed(f"No skills found for query '{query}'.")
        names = ", ".join(s["slug"] for s in skills)
        return ToolResult(data={"skills": skills}, llm_summary=f"Top skill matches: {names}")


def _rows(db):
    with db.lock:
        cur = db.conn.execute("""
            SELECT slug, name, description, kind, embedding
            FROM skill_embeddings
            WHERE hidden = 0
        """)
        return [dict(row) for row in cur.fetchall()]


def _norm(raw):
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[0]
    if arr.size == 0:
        return None
    n = float(np.linalg.norm(arr))
    return arr / n if n else arr
