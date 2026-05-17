"""search_skills: find reusable canvas skills."""

from __future__ import annotations

from array import array
from math import sqrt

from plugins.BaseTool import BaseTool, ToolResult
from plugins.helpers import skill_scoring, skill_store


class SearchSkills(BaseTool):
    name = "search_skills"
    description = "Search persistent canvas skills by intent. Use before creating a new drawing or transform skill."
    max_calls = 4
    parameters = {"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer", "default": 5}, "kind": {"type": "string", "enum": ["creation", "transform"]}}, "required": ["query"]}

    def run(self, context, **kwargs) -> ToolResult:
        query, kind, top_k = str(kwargs.get("query") or ""), kwargs.get("kind"), int(kwargs.get("top_k") or 5)
        # Merge DB results (indexed by the embed_skills task) with in-memory
        # results (covers built-in skills before the indexer has run). Dedupe by
        # slug, keep the higher score.
        merged: dict[str, dict] = {}
        for row in _db_search(context, query, top_k, kind):
            merged[row["slug"]] = row
        for row in skill_store.search_skills(query, top_k=top_k, kind=kind, text_embedder=context.services.get("text_embedder")):
            prev = merged.get(row["slug"])
            if prev is None or row["score"] > prev["score"]:
                merged[row["slug"]] = row
        # Boost by implicit signals — multiplier is 1.0 for zero-score skills,
        # so the cold start is unchanged.
        stats = skill_scoring.get_scores(getattr(context, "db", None), merged.keys())
        for slug, row in merged.items():
            row["cosine"] = row["score"]
            row["score"] = row["score"] * skill_scoring.search_multiplier(stats.get(slug, {}))
        rows = sorted(merged.values(), key=lambda r: r["score"], reverse=True)[:top_k]
        return ToolResult(data=rows, llm_summary=("No matching skills found." if not rows else "Matching skills:\n" + "\n".join(f"- {r['slug']} ({r['kind']}, score {r['score']:.2f}): {r['description']}" for r in rows)))


def _db_search(context, query, top_k, kind):
    emb = context.services.get("text_embedder"); db = context.db
    if not emb or not getattr(emb, "loaded", False) or not db:
        return []
    q = emb.encode([query])
    if q is None:
        return []
    qv = [float(x) for x in q[0]]
    try:
        with db.lock:
            rows = db.conn.execute("SELECT path,name,description,kind,owner,embedding FROM skill_embeddings").fetchall()
    except Exception:
        return []
    out = []
    for r in rows:
        if kind and r["kind"] != kind:
            continue
        v = array("f"); v.frombytes(r["embedding"])
        denom = (sqrt(sum(x*x for x in qv)) or 1) * (sqrt(sum(x*x for x in v)) or 1)
        slug = str(r["path"]).rsplit("\\", 1)[-1].rsplit("/", 1)[-1].rsplit(".skill.py", 1)[0]
        out.append({"slug": slug, "name": r["name"], "description": r["description"], "kind": r["kind"], "owner": r["owner"], "score": float(sum(a*b for a, b in zip(qv, v)) / denom)})
    return sorted(out, key=lambda x: x["score"], reverse=True)[:top_k]
