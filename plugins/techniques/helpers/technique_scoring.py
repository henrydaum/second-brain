"""Technique scoring from implicit signals (share, download, remix).

Attribution: when an image earns a signal, the score is distributed across
the chain that produced it. Background gets 0.6, the remaining 0.4 splits
across filters/objects. Background-only chains get 1.0. The `generate`
kind is tracked as a per-technique denominator (counter only, no weight).

The store is two SQLite tables: append-only `technique_events` and rolled-up
`technique_scores`. Aggregates are kept in sync at insert time so search-time
reads are a single indexed lookup.
"""

from __future__ import annotations

import math
import time
from typing import Iterable


_KIND_FIELDS = {"share": "shares", "download": "downloads", "remix": "remixes", "save": "saves", "link_open": "link_opens"}


def _safe_float(v) -> float:
    if v is None or v == "":
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(v) -> int:
    if v is None or v == "":
        return 0
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _attribution(chain: list[dict]) -> list[tuple[str, str, float]]:
    """Return [(slug, chain_position, weight), ...] summing to 1.0."""
    if not chain:
        return []
    steps = [(str(s.get("slug") or ""), str(s.get("kind") or "")) for s in chain if s.get("slug")]
    if not steps:
        return []
    backgrounds = [(slug, kind) for slug, kind in steps if kind == "background"]
    overlays = [(slug, kind) for slug, kind in steps if kind in ("filter", "object")]
    out: list[tuple[str, str, float]] = []
    if backgrounds and overlays:
        cw = 0.6 / len(backgrounds)
        tw = 0.4 / len(overlays)
        for slug, _ in backgrounds:
            out.append((slug, "background", cw))
        for slug, kind in overlays:
            out.append((slug, kind, tw))
    else:
        # Background-only or overlay-only: split 1.0 equally.
        eq = 1.0 / len(steps)
        for slug, kind in steps:
            out.append((slug, kind or "background", eq))
    return out


def record_event(db, kind: str, chain: list[dict], image_path: str | None) -> None:
    """Log an implicit signal and update aggregates.

    `kind` is one of: 'share', 'download', 'remix', 'generate'.
    For 'generate', `chain` should be the single just-executed step
    (we want the per-technique use counter, not the full chain).
    """
    if db is None or not chain:
        return
    now = time.time()
    if kind == "generate":
        # One generation row per technique in the supplied chain (usually one).
        rows = [(str(s.get("slug") or ""), str(s.get("kind") or "background"), 1.0)
                for s in chain if s.get("slug")]
        if not rows:
            return
        with db.lock:
            for slug, position, _ in rows:
                db.conn.execute(
                    "INSERT INTO technique_events (ts, kind, slug, image_path, chain_position, weight) VALUES (?, ?, ?, ?, ?, ?)",
                    (now, "generate", slug, image_path, position, 1.0),
                )
                db.conn.execute(
                    "INSERT INTO technique_scores (slug, generations, updated_at) VALUES (?, 1, ?) "
                    "ON CONFLICT(slug) DO UPDATE SET generations = generations + 1, updated_at = excluded.updated_at",
                    (slug, now),
                )
            db.conn.commit()
        return

    field = _KIND_FIELDS.get(kind)
    if not field:
        return
    parts = _attribution(chain)
    if not parts:
        return
    with db.lock:
        for slug, position, weight in parts:
            db.conn.execute(
                "INSERT INTO technique_events (ts, kind, slug, image_path, chain_position, weight) VALUES (?, ?, ?, ?, ?, ?)",
                (now, kind, slug, image_path, position, weight),
            )
            db.conn.execute(
                f"INSERT INTO technique_scores (slug, {field}, updated_at) VALUES (?, ?, ?) "
                f"ON CONFLICT(slug) DO UPDATE SET {field} = {field} + excluded.{field}, updated_at = excluded.updated_at",
                (slug, weight, now),
            )
        db.conn.commit()


def get_scores(db, slugs: Iterable[str] | None = None) -> dict[str, dict]:
    """Return {slug: {shares, downloads, remixes, generations}} for the requested slugs
    (or every slug if None)."""
    if db is None:
        return {}
    out: dict[str, dict] = {}
    with db.lock:
        try:
            if slugs is None:
                cur = db.conn.execute("SELECT slug, shares, downloads, remixes, saves, generations FROM technique_scores")
            else:
                slugs = list(slugs)
                if not slugs:
                    return {}
                placeholders = ",".join("?" * len(slugs))
                cur = db.conn.execute(
                    f"SELECT slug, shares, downloads, remixes, saves, generations FROM technique_scores WHERE slug IN ({placeholders})",
                    slugs,
                )
            for row in cur.fetchall():
                out[row["slug"]] = {
                    "shares": _safe_float(row["shares"]),
                    "downloads": _safe_float(row["downloads"]),
                    "remixes": _safe_float(row["remixes"]),
                    "saves": _safe_float(row["saves"]),
                    "generations": _safe_int(row["generations"]),
                }
        except Exception:
            return {}
    return out


def weighted_score(stats: dict) -> float:
    """Blend of implicit signals. Tunable; kept here so it's easy to change."""
    if not stats:
        return 0.0
    return 2.0 * stats.get("shares", 0.0) + 2.0 * stats.get("remixes", 0.0) + 2.0 * stats.get("saves", 0.0) + stats.get("downloads", 0.0)


def search_multiplier(stats: dict) -> float:
    """Multiplier applied to cosine score in search ranking. Zero-score → 1.0."""
    return 1.0 + math.log1p(weighted_score(stats))


def remix_counts_by_path(db) -> dict[str, int]:
    """Aggregate remix events by source image path. Used to rank the gallery."""
    if db is None:
        return {}
    out: dict[str, int] = {}
    with db.lock:
        try:
            # Each remix click creates N rows (one per chain step) sharing a ts.
            # Count distinct ts per image_path so we get one tally per remix.
            cur = db.conn.execute(
                "SELECT image_path, COUNT(DISTINCT ts) AS n FROM technique_events "
                "WHERE kind = 'remix' AND image_path IS NOT NULL GROUP BY image_path"
            )
            for row in cur.fetchall():
                out[row["image_path"]] = int(row["n"] or 0)
        except Exception:
            return {}
    return out
