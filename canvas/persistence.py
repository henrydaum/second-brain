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
