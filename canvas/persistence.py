"""SQLite persistence for CanvasState.

One row per canvas in ``canvas_states`` (canvas_id PK, state_json blob,
updated_at). State_json is the output of ``CanvasState.to_dict()``, so any
future change to the dataclass shape only needs to keep that contract
backward-compatible.

The canvas_id is also the share-link id — /share/{canvas_id} resolves
through the same primary key.
"""

from __future__ import annotations

import json
import logging
import time

from canvas.state import CanvasState

logger = logging.getLogger("CanvasPersistence")


def save(db, cs: CanvasState) -> None:
	"""Upsert one CanvasState into the ``canvas_states`` table."""
	now = time.time()
	payload = json.dumps(cs.to_dict(), separators=(",", ":"))
	with db.lock:
		db.conn.execute(
			"INSERT INTO canvas_states (canvas_id, state_json, updated_at) "
			"VALUES (?, ?, ?) "
			"ON CONFLICT(canvas_id) DO UPDATE SET "
			"  state_json = excluded.state_json, updated_at = excluded.updated_at",
			(cs.canvas_id, payload, now),
		)
		db.conn.commit()
	logger.debug("save canvas_id=%s", cs.canvas_id)


def load(db, canvas_id: str) -> CanvasState | None:
	"""Fetch and rehydrate one CanvasState, or None if not found."""
	with db.lock:
		row = db.conn.execute(
			"SELECT state_json FROM canvas_states WHERE canvas_id = ?",
			(canvas_id,),
		).fetchone()
	if not row:
		return None
	try:
		data = json.loads(row["state_json"])
	except (TypeError, ValueError):
		logger.exception("canvas_states.state_json invalid for id=%s", canvas_id)
		return None
	return CanvasState.from_dict(data)


def list_ids(db) -> list[str]:
	"""All persisted canvas ids, newest-updated first."""
	with db.lock:
		rows = db.conn.execute(
			"SELECT canvas_id FROM canvas_states ORDER BY updated_at DESC"
		).fetchall()
	return [r["canvas_id"] for r in rows]


def delete(db, canvas_id: str) -> None:
	"""Remove one canvas from persistence. No-op if absent."""
	with db.lock:
		db.conn.execute("DELETE FROM canvas_states WHERE canvas_id = ?", (canvas_id,))
		db.conn.commit()
	logger.debug("delete canvas_id=%s", canvas_id)


# =================================================================
# canvas_pools — public content identity, keyed by pool_hash
# =================================================================

def save_pool(db, *, pool_hash: str, state: dict) -> None:
	"""Idempotent insert into canvas_pools keyed by pool_hash.

	``state`` is the canvas portion (size / palette_id / layers) — the
	render-determining state, not the full CanvasState envelope. Repeat
	calls with the same hash are no-ops (INSERT OR IGNORE).
	"""
	now = time.time()
	payload = json.dumps(state, separators=(",", ":"))
	with db.lock:
		db.conn.execute(
			"INSERT OR IGNORE INTO canvas_pools (pool_hash, state_json, created_at) "
			"VALUES (?, ?, ?)",
			(pool_hash, payload, now),
		)
		db.conn.commit()
	logger.debug("save_pool pool_hash=%s", pool_hash)


def load_pool(db, pool_hash: str) -> dict | None:
	"""Return the canvas-state dict associated with ``pool_hash`` or None."""
	with db.lock:
		row = db.conn.execute(
			"SELECT state_json FROM canvas_pools WHERE pool_hash = ?",
			(pool_hash,),
		).fetchone()
	if not row:
		return None
	try:
		return json.loads(row["state_json"])
	except (TypeError, ValueError):
		logger.exception("canvas_pools.state_json invalid for pool_hash=%s", pool_hash)
		return None
