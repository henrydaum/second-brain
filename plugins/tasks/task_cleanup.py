"""Cache eviction task.

Fires from the timekeeper on a schedule. Walks
``DATA_DIR/canvas_renders/`` and, if total size exceeds the configured
cap, deletes the least-recently-used pool folders until the directory is
back under the cap.

"Least recently used" is per-pool:
- Primary signal: ``MAX(ts)`` from ``user_canvas_actions`` for that
  pool_hash. This captures meaningful access (share / save / download /
  remix / link_open), not just file creation time.
- Fallback: folder mtime, for pools that no one has ever acted on. On
  content-addressed renders this is the moment the pool was first
  populated, since the files are never rewritten.

Deleted pools keep their ``canvas_pools`` row — the canvas definition
is durable. If anyone visits the share link later, ``pool_share_payload``
re-renders on demand with a fresh seed (same composition, different RNG
draw).
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

from paths import DATA_DIR
from plugins.BaseTask import BaseTask, TaskResult
from events.event_channels import CANVAS_CACHE_EVICT_DUE

logger = logging.getLogger("TaskCacheEvict")

RENDERS_DIR = DATA_DIR / "canvas_renders"


class CacheEvict(BaseTask):
    name = "cache_evict"
    trigger = "event"
    trigger_channels = [CANVAS_CACHE_EVICT_DUE]
    writes = []
    timeout = 600

    config_settings = [
        ("Canvas Cache Cap (GB)", "canvas_cache_max_gb",
         "Maximum total size of DATA_DIR/canvas_renders/. Daily eviction "
         "trims oldest pool folders until total size drops below this. "
         "Deleted pools re-render lazily when their share link is visited.",
         5.0, {"type": "slider", "range": (0.5, 100.0, 199), "is_float": True}),
        ("Cache Eviction Cron", "canvas_cache_evict_cron",
         "Cron expression for when to run the cache eviction sweep. "
         "Default is 4 AM daily (low-traffic window). Changes take effect "
         "after restart.",
         "0 4 * * *", {"type": "text"}),
    ]

    JOB_NAME = "canvas_cache_evict"

    def setup(self, config: dict, services: dict | None = None) -> None:
        """Auto-provision the scheduled job that fires our eviction event."""
        if not services:
            return
        timekeeper = services.get("timekeeper")
        if timekeeper is None:
            return
        cron = str(config.get("canvas_cache_evict_cron") or "0 4 * * *")
        job_def = {
            "channel": CANVAS_CACHE_EVICT_DUE,
            "cron": cron,
            "one_time": False,
            "enabled": True,
            "payload": {},
        }
        try:
            existing = timekeeper.get_job(self.JOB_NAME)
            if existing is None:
                timekeeper.create_job(self.JOB_NAME, job_def)
                logger.info("registered scheduled job '%s' cron=%s", self.JOB_NAME, cron)
            elif existing.get("cron") != cron:
                # Keep the timekeeper in sync if the user edits the cron in
                # config without touching scheduled_jobs directly.
                timekeeper.update_job(self.JOB_NAME, {"cron": cron})
                logger.info("updated scheduled job '%s' cron=%s", self.JOB_NAME, cron)
        except Exception:
            logger.exception("failed to register cache_evict scheduled job")

    def run_event(self, run_id: str, payload: dict, context) -> TaskResult:
        del run_id, payload  # nothing to consume; the heartbeat itself is the signal
        cap_gb = float((context.config or {}).get("canvas_cache_max_gb", 5.0))
        cap_bytes = int(cap_gb * 1024 * 1024 * 1024)
        if not RENDERS_DIR.is_dir():
            return TaskResult(success=True)
        pools = _scan_pools(RENDERS_DIR)
        total = sum(p["size"] for p in pools)
        logger.info(
            "cache_evict scan pools=%d total=%.2f GB cap=%.2f GB",
            len(pools), total / 1e9, cap_gb,
        )
        if total <= cap_bytes:
            return TaskResult(success=True)
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
            "cache_evict freed=%.2f GB evicted=%d remaining=%.2f GB",
            freed / 1e9, evicted, (total - freed) / 1e9,
        )
        return TaskResult(success=True)


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
            # Empty folder — evict by treating it as ancient so it goes
            # first if we end up needing to free anything.
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
