"""Semantic search over embedded canvas skills."""

from __future__ import annotations

import sqlite3
import math
from pathlib import Path

import numpy as np

from paths import ROOT_DIR
from plugins.BaseTool import BaseTool, ToolResult


class SearchSkills(BaseTool):
    name = "search_skills"
    description = "Search stored canvas skills semantically by embedding a query and ranking skill name + description."
    max_calls = 6
    background_safe = True
    config_settings = [
        ("Weigh Skill Popularity", "weigh_popularity", "Blend canvas engagement signals into search ranking.", True, {"type": "bool"}),
        ("Popularity Alpha", "popularity_alpha", "How much popularity affects search ranking.", 0.25, {"type": "slider", "range": (0.0, 1.0, 100), "is_float": True}),
    ]
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural-language skill search query."},
            "slug": {"type": "string", "description": "Deprecated alias for query."},
            "limit": {"type": "integer", "default": 5, "minimum": 1, "maximum": 10},
            "built_in_only": {"type": "boolean", "default": False, "description": "Only return built-in library skills."},
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
        candidates = []
        for row in rows:
            if kwargs.get("built_in_only") and not _built_in(row.get("path")):
                continue
            vec = np.frombuffer(row["embedding"], dtype="<f4")
            if vec.size == q.size:
                pop = _popularity(row)
                candidates.append(({k: row[k] for k in ("slug", "name", "description", "kind")}, float(np.dot(q, vec)), pop, dict(row)))
        scored = _blend(candidates, getattr(context, "config", {}) or {})
        scored.sort(key=lambda item: item[1], reverse=True)
        skills = [meta for meta, _ in scored[:limit]]
        if not skills:
            return ToolResult.failed(f"No skills found for query '{query}'.")
        lines = [f"Top skill matches for '{query}':"]
        for s in skills:
            desc = (s.get("description") or "").strip().replace("\n", " ")
            if len(desc) > 160:
                desc = desc[:157].rstrip() + "..."
            lines.append(f"- {s['slug']} ({s.get('kind') or '?'}) — {desc}" if desc else f"- {s['slug']} ({s.get('kind') or '?'})")
        lines.append("Call read_skill(slug=...) to see the full source of any promising hit.")
        return ToolResult(data={"skills": skills}, llm_summary="\n".join(lines))


def _rows(db):
    with db.lock:
        cur = db.conn.execute("""
            SELECT path, slug, name, description, kind, embedding
                 , COALESCE(shares, 0) AS shares
                 , COALESCE(downloads, 0) AS downloads
                 , COALESCE(remixes, 0) AS remixes
                 , COALESCE(saves, 0) AS saves
                 , COALESCE(link_opens, 0) AS link_opens
            FROM skill_embeddings
            LEFT JOIN skill_scores USING (slug)
            WHERE hidden = 0
        """)
        return [dict(row) for row in cur.fetchall()]


def _built_in(path) -> bool:
    try:
        return Path(path).resolve().parent == (ROOT_DIR / "plugins" / "skills").resolve()
    except Exception:
        return False


def _popularity(row) -> float:
    return sum(float(row.get(k) or 0.0) for k in ("shares", "downloads", "remixes", "saves", "link_opens"))


def _blend(candidates, config):
    alpha = max(0.0, min(1.0, float(config.get("popularity_alpha", 0.25) or 0.0)))
    use_pop = bool(config.get("weigh_popularity", True)) and alpha > 0
    logs = [math.log1p(pop) for _meta, _cos, pop, _row in candidates]
    lo, hi = (min(logs), max(logs)) if logs else (0.0, 0.0)
    out = []
    for meta, cos, pop, row in candidates:
        pscore = ((math.log1p(pop) - lo) / (hi - lo)) if use_pop and hi > lo else 0.0
        score = (1 - alpha) * cos + alpha * pscore if use_pop else cos
        out.append(({**meta, "score": round(score, 4), "cosine_score": round(cos, 4), "popularity_score": round(pscore, 4), **{k: float(row.get(k) or 0.0) for k in ("shares", "downloads", "remixes", "saves", "link_opens")}}, score))
    return out


def _norm(raw):
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[0]
    if arr.size == 0:
        return None
    n = float(np.linalg.norm(arr))
    return arr / n if n else arr
