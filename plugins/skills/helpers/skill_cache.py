"""Layer-level cache + seed pool for skill rendering.

Each layer is a deterministic function of:
    (skill_slug, code_version, normalized(params+controls),
     palette_id, size, input_image_hash, seed)

So the rendered PNG can be content-addressed. The seed pool lets
``RunSkill`` sample a previously-used seed instead of always minting a
fresh one — guaranteeing a cache hit whenever the same skill is invoked
with the same context. Only the user clicking Regenerate (or a per-skill
randomize button) mints a brand new seed and grows the pool.

The cache file layout is intentionally flat (``DATA_DIR/canvas/cache/{key}.png``);
shard later if directory size becomes a problem.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from paths import DATA_DIR

logger = logging.getLogger("SkillCache")

CACHE_DIR = DATA_DIR / "canvas" / "cache"
_disk_lock = threading.RLock()
_runtime_ref: Any = None


def bind_runtime(runtime: Any) -> None:
    """Wire the runtime in so we can reach the shared DB handle."""
    global _runtime_ref
    _runtime_ref = runtime


def _db() -> Any:
    return getattr(_runtime_ref, "db", None) if _runtime_ref is not None else None


# ── key compute ────────────────────────────────────────────────────

def code_version(code: str) -> str:
    """Stable hash of a skill's source. Same slug + edited code -> new key."""
    return hashlib.sha256((code or "").encode("utf-8")).hexdigest()[:16]


def image_hash(path: Path | None) -> str:
    """sha256 of the PNG bytes, or the empty-string sentinel for None."""
    if path is None:
        return "none"
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    except OSError:
        return "missing"


def _normalize(merged_params: dict) -> str:
    return json.dumps(merged_params, sort_keys=True, default=str)


def pool_key(
    *, slug: str, code_sha: str, merged_params: dict,
    palette_id: str, size: int, input_hash: str,
) -> str:
    """Cache key MINUS the seed. Two seeds in the same pool both produce
    cached PNGs for that exact context."""
    raw = "|".join([
        slug, code_sha, _normalize(merged_params),
        str(palette_id or ""), str(int(size)), input_hash,
    ])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def cache_key(pool_key_: str, seed: int) -> str:
    """Full content-addressed key (pool_key + seed)."""
    return hashlib.sha256(f"{pool_key_}|{int(seed)}".encode("utf-8")).hexdigest()[:32]


# ── cache (PNG on disk, index in SQLite) ───────────────────────────

def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.png"


def get(key: str) -> Path | None:
    """Return the cached PNG path if it exists, else None. Touches last_used on hit."""
    db = _db()
    if db is None:
        return None
    path = _cache_path(key)
    if not path.is_file():
        # Row may exist with a missing file (e.g. user wiped cache dir).
        # Treat as a miss and let the caller regenerate.
        return None
    with db.lock:
        try:
            db.conn.execute(
                "UPDATE canvas_layer_cache SET last_used = ?, use_count = use_count + 1 WHERE cache_key = ?",
                (time.time(), key),
            )
            db.conn.commit()
        except Exception:
            logger.exception("cache touch failed")
    return path


def put(
    key: str, src_path: Path, *,
    skill_slug: str, size: int, palette_id: str, seed: int, pool_key_: str,
) -> Path:
    """Copy ``src_path`` into the cache and register it. Returns the cached path."""
    db = _db()
    dest = _cache_path(key)
    with _disk_lock:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copyfile(src_path, dest)
        except OSError:
            logger.exception("cache copy failed (src=%s)", src_path)
            return src_path
    try:
        bytes_ = dest.stat().st_size
    except OSError:
        bytes_ = 0
    if db is not None:
        now = time.time()
        with db.lock:
            try:
                db.conn.execute(
                    """INSERT OR REPLACE INTO canvas_layer_cache
                       (cache_key, pool_key, skill_slug, size, palette_id, seed,
                        file_path, bytes, created_at, last_used, use_count)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE(
                         (SELECT use_count FROM canvas_layer_cache WHERE cache_key = ?), 0))""",
                    (key, pool_key_, skill_slug, int(size), palette_id, int(seed),
                     str(dest.resolve()), int(bytes_), now, now, key),
                )
                db.conn.commit()
            except Exception:
                logger.exception("cache index write failed")
    return dest


# ── seed pool ──────────────────────────────────────────────────────

def sample_seed(pool_key_: str) -> int | None:
    """Return one previously-used seed at random, or None if the pool is empty."""
    db = _db()
    if db is None:
        return None
    with db.lock:
        try:
            rows = db.conn.execute(
                "SELECT seed FROM canvas_seed_pool WHERE pool_key = ?",
                (pool_key_,),
            ).fetchall()
        except Exception:
            logger.exception("seed pool sample failed")
            return None
    if not rows:
        return None
    seed = int(random.choice(rows)["seed"])
    return seed


def add_seed(pool_key_: str, seed: int) -> None:
    """Record a freshly-rendered seed in the pool for future sampling."""
    db = _db()
    if db is None:
        return
    now = time.time()
    with db.lock:
        try:
            db.conn.execute(
                """INSERT INTO canvas_seed_pool (pool_key, seed, last_used)
                   VALUES (?, ?, ?)
                   ON CONFLICT(pool_key, seed) DO UPDATE SET last_used = excluded.last_used""",
                (pool_key_, int(seed), now),
            )
            db.conn.commit()
        except Exception:
            logger.exception("seed pool add failed")


def mint_seed() -> int:
    """A fresh random seed for Regenerate / randomize-button paths."""
    return random.randint(1, 2_147_483_647)
