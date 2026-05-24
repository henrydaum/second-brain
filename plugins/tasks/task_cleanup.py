"""Periodic cleanup task.

Fires from the timekeeper on a schedule (default 4 AM daily). One task,
two responsibilities — kept together so the schedule has one source of
truth and config has one place to look.

1. **Canvas render cache eviction.** Walks ``DATA_DIR/canvas_renders/``
   and, if total size exceeds the configured cap, deletes the
   least-recently-used pool folders until the directory is back under
   the cap.

   "Least recently used" is per-pool:
   - Primary signal: ``MAX(ts)`` from ``user_canvas_actions`` for that
     pool_hash (share / save / download / remix / link_open).
   - Fallback: folder mtime, for pools no one has ever acted on. On
     content-addressed renders this is the moment the pool was first
     populated, since the files are never rewritten.

   Deleted pools keep their ``canvas_pools`` row — the canvas
   definition is durable. If anyone visits the share link later,
   ``pool_share_payload`` re-renders on demand with a fresh seed (same
   composition, different RNG draw).

2. **Stale ephemeral conversation pruning.** Deletes
   ``category = 'Art'`` conversations older than
   ``conversation_max_age_hours`` that are not currently bound to a
   live session. Captures tabs people closed without hitting
   "New chat".
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

from paths import DATA_DIR
from plugins.BaseTask import BaseTask, TaskResult
from events.event_channels import CLEANUP_DUE

logger = logging.getLogger("TaskCleanup")

RENDERS_DIR = DATA_DIR / "canvas_renders"


class Cleanup(BaseTask):
    name = "cleanup"
    trigger = "event"
    trigger_channels = [CLEANUP_DUE]
    writes = []
    timeout = 600

    config_settings = [
        ("Canvas Cache Cap (GB)", "canvas_cache_max_gb",
         "Maximum total size of DATA_DIR/canvas_renders/. Cleanup trims "
         "oldest pool folders until total size drops below this. "
         "Deleted pools re-render lazily when their share link is visited.",
         5.0, {"type": "slider", "range": (0.5, 100.0, 199), "is_float": True}),
        ("Conversation Max Age (hours)", "conversation_max_age_hours",
         "Ephemeral web conversations (category='Art') older than this are deleted at "
         "cleanup time, unless still bound to a live session. The cleanup cron schedule "
         "lives in 'scheduled_jobs' (editable via /schedule) — default is 4 AM daily.",
         24.0, {"type": "slider", "range": (1.0, 168.0, 167), "is_float": True}),
    ]

    def run_event(self, run_id: str, payload: dict, context) -> TaskResult:
        del run_id, payload  # the heartbeat itself is the signal
        config = context.config or {}
        try:
            _evict_canvas_cache(config, context)
        except Exception:
            logger.exception("canvas cache eviction failed")
        try:
            _prune_stale_conversations(config, context)
        except Exception:
            logger.exception("conversation pruning failed")
        return TaskResult(success=True)


# =================================================================
# canvas render cache
# =================================================================

def _evict_canvas_cache(config: dict, context) -> None:
    cap_gb = float(config.get("canvas_cache_max_gb", 5.0))
    cap_bytes = int(cap_gb * 1024 * 1024 * 1024)
    if not RENDERS_DIR.is_dir():
        return
    pools = _scan_pools(RENDERS_DIR)
    total = sum(p["size"] for p in pools)
    logger.info(
        "cleanup: cache scan pools=%d total=%.2f GB cap=%.2f GB",
        len(pools), total / 1e9, cap_gb,
    )
    if total <= cap_bytes:
        return
    access = _last_access_by_pool(getattr(context, "db", None))
    for p in pools:
        p["last_access"] = access.get(p["pool_hash"], p["mtime"])
    pools.sort(key=lambda p: p["last_access"])  # oldest first
    freed = 0
    evicted = 0
    for p in pools:
        if total - freed <= cap_bytes:
            break
        try:
            shutil.rmtree(p["path"])
        except OSError:
            logger.exception("rmtree failed pool=%s", p["pool_hash"])
            continue
        freed += p["size"]
        evicted += 1
        logger.debug(
            "evicted pool=%s size=%.2f MB last_access=%s",
            p["pool_hash"], p["size"] / 1e6,
            time.strftime("%Y-%m-%d", time.localtime(p["last_access"])),
        )
    logger.info(
        "cleanup: cache freed=%.2f GB evicted=%d remaining=%.2f GB",
        freed / 1e9, evicted, (total - freed) / 1e9,
    )


def _scan_pools(root: Path) -> list[dict]:
    """One entry per pool_hash subfolder: total bytes + folder mtime."""
    pools: list[dict] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        size = 0
        newest = 0.0
        for f in entry.iterdir():
            if not f.is_file():
                continue
            try:
                st = f.stat()
            except OSError:
                continue
            size += st.st_size
            if st.st_mtime > newest:
                newest = st.st_mtime
        if size == 0:
            # Empty folder — evict first if we need to free anything.
            newest = 0.0
        pools.append({
            "path": entry,
            "pool_hash": entry.name,
            "size": size,
            "mtime": newest,
        })
    return pools


def _last_access_by_pool(db) -> dict[str, float]:
    """``pool_hash -> MAX(ts)`` across all recorded user actions.

    Captures share/save/download/remix/link_open — anything that signals
    a human cared about this pool recently.
    """
    if db is None:
        return {}
    try:
        with db.lock:
            rows = db.conn.execute(
                "SELECT pool_hash, MAX(ts) AS last_ts "
                "FROM user_canvas_actions GROUP BY pool_hash"
            ).fetchall()
    except Exception:
        logger.exception("user_canvas_actions read failed")
        return {}
    return {r["pool_hash"]: float(r["last_ts"] or 0.0) for r in rows}


# =================================================================
# stale ephemeral conversations
# =================================================================

def _prune_stale_conversations(config: dict, context) -> None:
    """Delete category='Art' conversations older than the configured age
    that aren't currently bound to a live session.

    The 'Art' category marks web-frontend conversations as ephemeral —
    real REPL/Telegram conversations live under user-chosen categories
    (or none) and are never touched here.
    """
    db = getattr(context, "db", None)
    runtime = getattr(context, "runtime", None)
    if db is None:
        return
    max_age_hours = float(config.get("conversation_max_age_hours", 24.0))
    cutoff = time.time() - max_age_hours * 3600.0
    live_ids: set[int] = set()
    if runtime is not None:
        for s in getattr(runtime, "sessions", {}).values():
            cid = getattr(s, "conversation_id", None)
            if cid is not None:
                live_ids.add(int(cid))
    try:
        with db.lock:
            rows = db.conn.execute(
                "SELECT id FROM conversations "
                "WHERE category = 'Art' AND COALESCE(updated_at, created_at) < ?",
                (cutoff,),
            ).fetchall()
    except Exception:
        logger.exception("conversation scan failed")
        return
    stale = [int(r["id"]) for r in rows if int(r["id"]) not in live_ids]
    deleted = 0
    for cid in stale:
        try:
            db.delete_conversation(cid)
            deleted += 1
        except Exception:
            logger.exception("delete_conversation failed cid=%s", cid)
    if deleted:
        logger.info(
            "cleanup: pruned %d stale art conversation(s) older than %.1fh",
            deleted, max_age_hours,
        )
