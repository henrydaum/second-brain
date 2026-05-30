"""User-to-canvas interactions: writes to ``user_canvas_actions`` plus the
chain-weighted fan-out into ``technique_scores`` via existing scoring helper.

One call site for everything: ``record_user_action(db, user_id, pool_hash,
action, *, layers, image_path)``. Pulls double duty:
  1. Inserts a row into ``user_canvas_actions`` so "user U did X to canvas
     C" is durable history.
  2. Calls ``technique_scoring.record_event`` so the popularity counters on
     ``technique_scores`` move (shares/saves/downloads/remixes/link_opens).

Pool_hash is the *content* identity (the renderer's hash), NOT the
canvas_id editing handle. Multiple canvas_ids can resolve to the same
pool_hash; ``user_canvas_actions`` only ever sees pool_hashes.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Iterable

from plugins.techniques.helpers import technique_scoring

logger = logging.getLogger("CanvasActions")


# Actions we treat as legitimate. New ones are accepted (TEXT column),
# but only these contribute to technique_scoring via technique_scoring._KIND_FIELDS.
KNOWN_ACTIONS = {"share", "save", "download", "remix", "link_open"}


def record_user_action(
	db,
	*,
	user_id: str,
	pool_hash: str,
	action: str,
	layers: Iterable[dict] | None = None,
	image_path: str | None = None,
	meta: dict | None = None,
) -> None:
	"""Insert a row + fan out to technique_scoring.

	``layers`` is the canvas's layer list — used by ``technique_scoring`` to
	distribute the signal across background/filter techniques.

	``meta`` is optional per-action metadata (title, artist, …) stored as
	JSON in ``user_canvas_actions.meta_json``. Useful so a share can carry
	its own title without the title becoming part of the canvas identity.
	"""
	now = time.time()
	meta_json = json.dumps(meta, separators=(",", ":")) if meta else None
	with db.lock:
		db.conn.execute(
			"INSERT INTO user_canvas_actions (user_id, pool_hash, action, ts, meta_json) "
			"VALUES (?, ?, ?, ?, ?)",
			(user_id, pool_hash, action, now, meta_json),
		)
		db.conn.commit()

	if layers:
		# technique_scoring expects [{"slug": ..., "kind": ...}, ...] — that's
		# exactly the layer dict shape we already use.
		chain = [
			{"slug": layer.get("slug"), "kind": layer.get("kind")}
			for layer in layers
			if layer.get("slug")
		]
		if chain:
			technique_scoring.record_event(db, action, chain, image_path)
	logger.info(
		"record_user_action user=%s pool=%s action=%s layers=%d",
		user_id, pool_hash, action, len(list(layers) if layers else []),
	)


def list_user_canvases(db, *, user_id: str, action: str, limit: int = 100) -> list[dict]:
	"""Return pool_hashes a user has performed ``action`` on, newest first."""
	with db.lock:
		rows = db.conn.execute(
			"SELECT pool_hash, MAX(ts) AS last_ts FROM user_canvas_actions "
			"WHERE user_id = ? AND action = ? "
			"GROUP BY pool_hash ORDER BY last_ts DESC LIMIT ?",
			(user_id, action, limit),
		).fetchall()
	return [{"pool_hash": r["pool_hash"], "last_ts": r["last_ts"]} for r in rows]


def count_action(db, *, pool_hash: str, action: str) -> int:
	"""How many users performed ``action`` on ``pool_hash``."""
	with db.lock:
		row = db.conn.execute(
			"SELECT COUNT(DISTINCT user_id) AS n FROM user_canvas_actions "
			"WHERE pool_hash = ? AND action = ?",
			(pool_hash, action),
		).fetchone()
	return int(row["n"] or 0) if row else 0
