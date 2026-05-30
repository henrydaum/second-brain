"""Hybrid (embedding + BM25) search over canvas techniques."""

from __future__ import annotations

import re
import sqlite3
import math
from pathlib import Path

import numpy as np

from paths import ROOT_DIR
from plugins.BaseTool import BaseTool, ToolResult


# Reciprocal Rank Fusion constant. 60 is the canonical default from Cormack
# et al.; it dampens the head of each list enough that one ranker can't
# unilaterally dominate the fused order.
_RRF_K = 60


class SearchTechniques(BaseTool):
    name = "search_techniques"
    description = "Search stored canvas techniques by hybrid keyword + semantic ranking over technique name and description."
    max_calls = 6
    background_safe = True
    config_settings = [
        ("Weigh Technique Popularity", "weigh_popularity", "Blend canvas engagement signals into search ranking.", True, {"type": "bool"}),
        ("Popularity Alpha", "popularity_alpha", "Additive popularity bonus on top of the hybrid (BM25+embedding) score. 0 disables popularity entirely.", 0.1, {"type": "slider", "range": (0.0, 1.0, 100), "is_float": True}),
    ]
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural-language technique search query."},
            "slug": {"type": "string", "description": "Deprecated alias for query."},
            "limit": {"type": "integer", "default": 5, "minimum": 1, "maximum": 10},
            "built_in_only": {"type": "boolean", "default": False, "description": "Only return built-in library techniques."},
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
        limit = max(1, min(10, int(kwargs.get("limit") or 5)))
        built_in_only = bool(kwargs.get("built_in_only"))
        # Web sessions where the account has not opted into community techniques
        # see only built-ins, regardless of the tool's own parameter.
        sk = getattr(context, "session_key", None)
        if sk and isinstance(sk, str) and sk.startswith("web:"):
            runtime = getattr(context, "runtime", None)
            session = getattr(runtime, "sessions", {}).get(sk) if runtime is not None else None
            if session is not None and not bool(getattr(session, "community_techniques_enabled", False)):
                built_in_only = True
        try:
            techniques = search_techniques_semantic(
                db, embedder, query,
                limit=limit,
                built_in_only=built_in_only,
                config=getattr(context, "config", {}) or {},
            )
        except sqlite3.OperationalError:
            return ToolResult.failed("technique index is not ready yet; let embed_techniques and index_techniques run first")
        except ValueError as e:
            return ToolResult.failed(str(e))
        if not techniques:
            return ToolResult.failed(f"No techniques found for query '{query}'.")
        lines = [f"Top technique matches for '{query}':"]
        for s in techniques:
            desc = (s.get("description") or "").strip().replace("\n", " ")
            if len(desc) > 160:
                desc = desc[:157].rstrip() + "..."
            lines.append(f"- {s['slug']} ({s.get('kind') or '?'}) — {desc}" if desc else f"- {s['slug']} ({s.get('kind') or '?'})")
        lines.append("Call read_technique(slug=...) to see the full source of any promising hit.")
        return ToolResult(data={"techniques": techniques}, llm_summary="\n".join(lines))


def search_techniques_semantic(db, embedder, query: str, *, limit: int = 5, built_in_only: bool = False, config: dict | None = None) -> list[dict]:
    """Hybrid technique retrieval: BM25 keyword + embedding cosine fused via RRF,
    with an optional additive popularity bonus.

    Shared by the agent tool above and the web frontend's manual 'Search'
    button. Raises ``sqlite3.OperationalError`` if the embeddings or FTS
    tables are missing, ``ValueError`` if the embedder returns no vector.
    """
    pool_size = max(2 * limit, 20)

    embed_hits = _embedding_candidates(db, embedder, query, pool_size, built_in_only)
    bm25_hits = _bm25_candidates(db, query, pool_size, built_in_only)

    # Collect every candidate slug, preferring the most-informative row
    # available (embedding row is richest; fall back to BM25 row).
    rows_by_slug: dict[str, dict] = {}
    for slug, _rank, row in embed_hits:
        rows_by_slug.setdefault(slug, row)
    for slug, _rank, row in bm25_hits:
        rows_by_slug.setdefault(slug, row)

    embed_rank = {slug: rank for slug, rank, _ in embed_hits}
    bm25_rank = {slug: rank for slug, rank, _ in bm25_hits}

    fused: list[tuple[str, float]] = []
    for slug in rows_by_slug:
        score = 0.0
        if slug in embed_rank:
            score += 1.0 / (_RRF_K + embed_rank[slug])
        if slug in bm25_rank:
            score += 1.0 / (_RRF_K + bm25_rank[slug])
        fused.append((slug, score))

    fused = _apply_popularity(fused, rows_by_slug, config or {})
    fused.sort(key=lambda item: item[1], reverse=True)

    out = []
    for slug, score in fused[: max(1, int(limit))]:
        row = rows_by_slug[slug]
        out.append({
            "slug": slug,
            "name": row.get("name"),
            "description": row.get("description"),
            "kind": row.get("kind"),
            "score": round(score, 5),
            "embedding_rank": embed_rank.get(slug),
            "bm25_rank": bm25_rank.get(slug),
            **{k: float(row.get(k) or 0.0) for k in ("shares", "downloads", "remixes", "saves", "link_opens")},
        })
    return out


# ---------------------------------------------------------------------------
# Retrievers
# ---------------------------------------------------------------------------

def _embedding_candidates(db, embedder, query: str, k: int, built_in_only: bool) -> list[tuple[str, int, dict]]:
    """Return [(slug, rank, row)] sorted by cosine descending, keyed per-slug."""
    q = _norm(embedder.encode(query))
    if q is None:
        raise ValueError("text_embedder returned no embedding")
    rows = _embedding_rows(db)
    best_by_slug: dict[str, tuple[float, dict]] = {}
    for row in rows:
        if built_in_only and not _built_in(row.get("path")):
            continue
        vec = np.frombuffer(row["embedding"], dtype="<f4")
        if vec.size != q.size:
            continue
        cos = float(np.dot(q, vec))
        slug = row["slug"]
        existing = best_by_slug.get(slug)
        if existing is None or cos > existing[0]:
            best_by_slug[slug] = (cos, row)
    ranked = sorted(best_by_slug.items(), key=lambda kv: kv[1][0], reverse=True)[:k]
    return [(slug, i + 1, row) for i, (slug, (_cos, row)) in enumerate(ranked)]


def _bm25_candidates(db, query: str, k: int, built_in_only: bool) -> list[tuple[str, int, dict]]:
    """Return [(slug, rank, row)] sorted by BM25 ascending (best first)."""
    match = _build_fts_match(query)
    if not match:
        return []
    sql = """
        SELECT slug, name, description, kind, path, bm25(technique_fts) AS rank
        FROM technique_fts
        WHERE technique_fts MATCH ? AND hidden = 0
        ORDER BY rank
        LIMIT ?
    """
    with db.lock:
        try:
            cur = db.conn.execute(sql, (match, k))
            rows = [dict(r) for r in cur.fetchall()]
        except sqlite3.OperationalError:
            # Either the FTS5 table hasn't been created yet (index_techniques
            # hasn't run) or the query had malformed FTS5 syntax. Degrade
            # gracefully to embedding-only.
            return []
    seen: dict[str, tuple[int, dict]] = {}
    for row in rows:
        if built_in_only and not _built_in(row.get("path")):
            continue
        slug = row["slug"]
        if slug in seen:
            continue
        # Attach popularity columns so downstream code can read them uniformly.
        for col in ("shares", "downloads", "remixes", "saves", "link_opens"):
            row.setdefault(col, 0.0)
        seen[slug] = (len(seen) + 1, row)
    return [(slug, rank, row) for slug, (rank, row) in seen.items()]


def _build_fts_match(query: str) -> str:
    """Sanitize a free-text query into a permissive FTS5 MATCH expression.

    Tokens are quoted (so FTS5 operator characters can't smuggle in) and
    joined with OR for recall. Empty input returns "".
    """
    tokens = re.findall(r"[A-Za-z0-9]+", query)
    if not tokens:
        return ""
    return " OR ".join(f'"{t}"' for t in tokens)


def _embedding_rows(db):
    with db.lock:
        cur = db.conn.execute("""
            SELECT path, slug, name, description, kind, embedding
                 , COALESCE(shares, 0) AS shares
                 , COALESCE(downloads, 0) AS downloads
                 , COALESCE(remixes, 0) AS remixes
                 , COALESCE(saves, 0) AS saves
                 , COALESCE(link_opens, 0) AS link_opens
            FROM technique_embeddings
            LEFT JOIN technique_scores USING (slug)
            WHERE hidden = 0
        """)
        return [dict(row) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# Popularity
# ---------------------------------------------------------------------------

def _apply_popularity(fused: list[tuple[str, float]], rows_by_slug: dict[str, dict], config: dict) -> list[tuple[str, float]]:
    alpha = max(0.0, min(1.0, float(config.get("popularity_alpha", 0.1) or 0.0)))
    if not bool(config.get("weigh_popularity", True)) or alpha <= 0:
        return fused
    pops = {slug: _popularity(rows_by_slug[slug]) for slug, _ in fused}
    logs = [math.log1p(p) for p in pops.values()]
    if not logs:
        return fused
    lo, hi = min(logs), max(logs)
    if hi <= lo:
        return fused
    scale = 1.0 / (_RRF_K + 1)  # roughly the magnitude of a top-ranked RRF hit
    out = []
    for slug, score in fused:
        norm_pop = (math.log1p(pops[slug]) - lo) / (hi - lo)
        out.append((slug, score + alpha * scale * norm_pop))
    return out


def _popularity(row) -> float:
    return sum(float(row.get(k) or 0.0) for k in ("shares", "downloads", "remixes", "saves", "link_opens"))


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def _built_in(path) -> bool:
    try:
        return Path(path).resolve().parent == (ROOT_DIR / "plugins" / "techniques").resolve()
    except Exception:
        return False


def _norm(raw):
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[0]
    if arr.size == 0:
        return None
    n = float(np.linalg.norm(arr))
    return arr / n if n else arr
