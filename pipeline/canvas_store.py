"""Canvas-domain data layer (Phase 1 of the canvas rework).

Owns the new tables introduced in Phase 0: ``skills``, ``canvases``,
``canvas_layers``, ``image_generations``, ``user_canvas_actions``. App code
should call into ``CanvasStore`` rather than touching these tables.

Phase 1 status — pure data + caching logic. The actual pixel-producing
renderer is injected as a callable so this module is unit-testable without
the skill plugin system. Phase 2 will wire in the real renderer.

Caching model:
    Each layer's output is content-addressed by
    ``(input_generation_id, skill_slug, controls_hash, size, palette_id, seed)``.
    The ``pool_key`` drops ``seed`` so we can ask "what seeds already exist
    for this exact upstream context?" cheaply.

    Sample-vs-mint rule:
        - If ``force_new_seed`` is set OR this conversation has already
          sampled this ``pool_key`` once, mint a fresh seed and render.
        - Otherwise sample a random existing seed for this ``pool_key``.
          If none exist, mint a fresh one.

    "Has this conversation already sampled this pool_key" is tracked
    in-memory on the store; it resets on process restart by design.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from plugins.skills.helpers.skill_scoring import record_event

logger = logging.getLogger("CanvasStore")


# Renderer contract — accepts the layer's resolved context and returns the
# rendered image bytes. The store is responsible for writing them to disk.
Renderer = Callable[
	[
		str,            # skill_slug
		str,            # skill_kind ('creation' | 'transform')
		Optional[Path], # input_path (None for creations)
		dict,           # controls
		int,            # seed
		Optional[int],  # size
		Optional[str],  # palette_id
	],
	bytes,
]


@dataclass
class LayerSpec:
	"""A single layer in a canvas chain. Used for create_canvas input."""
	skill_slug: str
	controls: dict = field(default_factory=dict)


def _short_id(n_bytes: int = 8) -> str:
	"""Url-safe short id. 8 bytes -> ~11 chars. Collision risk is negligible at our scale."""
	return secrets.token_urlsafe(n_bytes).rstrip("=")


def _controls_hash(controls: dict | None) -> str:
	"""Stable hash of a controls dict. Sorts keys so ordering doesn't matter."""
	payload = json.dumps(controls or {}, sort_keys=True, separators=(",", ":"))
	return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _pool_key(
	input_generation_id: int | None,
	skill_slug: str,
	controls_hash: str,
	size: int | None,
	palette_id: str | None,
) -> str:
	"""Identity of a render *up to but not including* the seed.

	Two renders with the same pool_key only differ by the random seed
	that was used; they're interchangeable from the user's perspective.
	"""
	parts = [
		str(input_generation_id or ""),
		skill_slug,
		controls_hash,
		str(size or ""),
		palette_id or "",
	]
	return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _cache_key(pool_key: str, seed: int) -> str:
	"""Identity of a specific (pool_key, seed) render."""
	return hashlib.sha256(f"{pool_key}|{seed}".encode("utf-8")).hexdigest()


class CanvasStore:
	"""Owner of the canvas-domain tables.

	Construct one per process. ``renderer`` is optional so tests that only
	exercise CRUD paths don't need to supply one. ``render_canvas`` raises
	if called without a renderer set.
	"""

	def __init__(
		self,
		db,
		*,
		file_dir: Path,
		renderer: Renderer | None = None,
		now: Callable[[], float] = time.time,
	):
		"""Initialize the canvas store."""
		self.db = db
		self.file_dir = Path(file_dir)
		self.renderer = renderer
		self._now = now
		# conversation_id -> {pool_key: seed_locked_in}. First time a
		# conversation renders a given pool_key we either sample an
		# existing seed or mint one, then *lock it in* — subsequent
		# renders in the same conversation reuse that exact seed so they
		# hit the cache. force_new_seed (the Regenerate button) replaces
		# the lock with a fresh mint. In-memory by design; resets on
		# process restart.
		self._session_pools: dict[str, dict[str, int]] = {}

	# =================================================================
	# SKILLS
	# =================================================================

	def upsert_skill(
		self,
		slug: str,
		*,
		kind: str,
		name: str | None = None,
		code: str | None = None,
		description: str | None = None,
	) -> None:
		"""Insert or update a skill row by slug."""
		now = self._now()
		with self.db.lock:
			self.db.conn.execute(
				"INSERT INTO skills (slug, name, kind, code, description, created_at, updated_at) "
				"VALUES (?, ?, ?, ?, ?, ?, ?) "
				"ON CONFLICT(slug) DO UPDATE SET "
				"  name=excluded.name, kind=excluded.kind, code=excluded.code, "
				"  description=excluded.description, updated_at=excluded.updated_at",
				(slug, name, kind, code, description, now, now),
			)
			self.db.conn.commit()
		logger.debug("upsert_skill slug=%s kind=%s", slug, kind)

	def get_skill(self, slug: str) -> dict | None:
		"""Fetch a skill row by slug."""
		with self.db.lock:
			row = self.db.conn.execute(
				"SELECT * FROM skills WHERE slug = ?", (slug,)
			).fetchone()
		return dict(row) if row else None

	# =================================================================
	# CANVASES
	# =================================================================

	def create_canvas(
		self,
		layers: list[LayerSpec],
		*,
		title: str | None = None,
		artist: str | None = None,
		size: int | None = None,
		palette_id: str | None = None,
		canvas_id: str | None = None,
	) -> str:
		"""Create a canvas with the given ordered layer chain.

		Returns the canvas id (auto-generated short id unless caller
		supplied one). Layer positions are assigned 0..N in input order.
		"""
		if not layers:
			raise ValueError("a canvas must have at least one layer")
		cid = canvas_id or _short_id()
		now = self._now()
		with self.db.lock:
			self.db.conn.execute(
				"INSERT INTO canvases (id, title, artist, size, palette_id, "
				"current_generation_id, created_at, updated_at) "
				"VALUES (?, ?, ?, ?, ?, NULL, ?, ?)",
				(cid, title, artist, size, palette_id, now, now),
			)
			for position, layer in enumerate(layers):
				self.db.conn.execute(
					"INSERT INTO canvas_layers (canvas_id, position, skill_slug, controls_json) "
					"VALUES (?, ?, ?, ?)",
					(cid, position, layer.skill_slug, json.dumps(layer.controls or {})),
				)
			self.db.conn.commit()
		logger.info("create_canvas id=%s layers=%d", cid, len(layers))
		return cid

	def get_canvas(self, canvas_id: str) -> dict | None:
		"""Fetch a canvas plus its ordered layers."""
		with self.db.lock:
			canvas_row = self.db.conn.execute(
				"SELECT * FROM canvases WHERE id = ?", (canvas_id,)
			).fetchone()
			if not canvas_row:
				return None
			layer_rows = self.db.conn.execute(
				"SELECT * FROM canvas_layers WHERE canvas_id = ? ORDER BY position",
				(canvas_id,),
			).fetchall()
		out = dict(canvas_row)
		out["layers"] = [
			{
				"id": r["id"],
				"position": r["position"],
				"skill_slug": r["skill_slug"],
				"controls": json.loads(r["controls_json"] or "{}"),
			}
			for r in layer_rows
		]
		return out

	def update_layer_controls(self, canvas_id: str, position: int, controls: dict) -> None:
		"""Replace the controls on a single layer. Bumps canvases.updated_at."""
		now = self._now()
		with self.db.lock:
			cur = self.db.conn.execute(
				"UPDATE canvas_layers SET controls_json = ? WHERE canvas_id = ? AND position = ?",
				(json.dumps(controls or {}), canvas_id, position),
			)
			if cur.rowcount == 0:
				raise KeyError(f"no layer at position {position} on canvas {canvas_id}")
			self.db.conn.execute(
				"UPDATE canvases SET updated_at = ? WHERE id = ?", (now, canvas_id)
			)
			self.db.conn.commit()
		logger.debug("update_layer_controls canvas=%s pos=%d", canvas_id, position)

	# =================================================================
	# RENDERING
	# =================================================================

	def render_canvas(
		self,
		canvas_id: str,
		conversation_id: str,
		*,
		force_new_seed: bool = False,
	) -> tuple[Path, int]:
		"""Render the canvas chain, reusing cached intermediates.

		Returns (path_to_final_image, generation_id_of_final_layer). Sets
		``canvases.current_generation_id``.
		"""
		if self.renderer is None:
			raise RuntimeError("CanvasStore.render_canvas requires a renderer")
		canvas = self.get_canvas(canvas_id)
		if not canvas:
			raise KeyError(f"unknown canvas {canvas_id!r}")
		if not canvas["layers"]:
			raise ValueError(f"canvas {canvas_id!r} has no layers")

		size = canvas.get("size")
		palette_id = canvas.get("palette_id")

		input_gen_id: int | None = None
		input_path: Path | None = None
		final_gen_id: int | None = None
		final_path: Path | None = None

		for layer in canvas["layers"]:
			gen_id, gen_path = self._render_layer(
				skill_slug=layer["skill_slug"],
				controls=layer["controls"],
				input_generation_id=input_gen_id,
				input_path=input_path,
				size=size,
				palette_id=palette_id,
				conversation_id=conversation_id,
				force_new_seed=force_new_seed,
			)
			input_gen_id = gen_id
			input_path = gen_path
			final_gen_id = gen_id
			final_path = gen_path

		assert final_gen_id is not None and final_path is not None
		with self.db.lock:
			self.db.conn.execute(
				"UPDATE canvases SET current_generation_id = ?, updated_at = ? WHERE id = ?",
				(final_gen_id, self._now(), canvas_id),
			)
			self.db.conn.commit()
		logger.info("render_canvas id=%s -> gen=%d", canvas_id, final_gen_id)
		return final_path, final_gen_id

	def _render_layer(
		self,
		*,
		skill_slug: str,
		controls: dict,
		input_generation_id: int | None,
		input_path: Path | None,
		size: int | None,
		palette_id: str | None,
		conversation_id: str,
		force_new_seed: bool,
	) -> tuple[int, Path]:
		"""Render one layer, returning (generation_id, file_path)."""
		skill = self.get_skill(skill_slug)
		if not skill:
			raise KeyError(f"unknown skill {skill_slug!r}")
		kind = skill["kind"]

		c_hash = _controls_hash(controls)
		pool_key = _pool_key(input_generation_id, skill_slug, c_hash, size, palette_id)

		seed, sampled_existing = self._resolve_seed(
			pool_key=pool_key,
			conversation_id=conversation_id,
			force_new_seed=force_new_seed,
		)

		# Lock the chosen seed in for this conversation so subsequent
		# renders of the same canvas in the same conversation hit cache.
		self._session_pools.setdefault(conversation_id, {})[pool_key] = seed

		ck = _cache_key(pool_key, seed)
		hit = self._lookup_cache(ck)
		if hit is not None:
			gen_id, file_path = hit
			self._bump_cache_use(gen_id)
			logger.debug(
				"layer cache HIT skill=%s pool=%s seed=%d sampled=%s",
				skill_slug, pool_key[:8], seed, sampled_existing,
			)
			return gen_id, file_path

		logger.debug(
			"layer cache MISS skill=%s pool=%s seed=%d",
			skill_slug, pool_key[:8], seed,
		)
		image_bytes = self.renderer(
			skill_slug, kind, input_path, controls, seed, size, palette_id,
		)
		file_path = self._write_image(ck, image_bytes)
		gen_id = self._insert_generation(
			cache_key=ck,
			pool_key=pool_key,
			skill_slug=skill_slug,
			input_generation_id=input_generation_id,
			controls=controls,
			controls_hash=c_hash,
			seed=seed,
			file_path=file_path,
			n_bytes=len(image_bytes),
		)
		return gen_id, file_path

	def _resolve_seed(
		self,
		*,
		pool_key: str,
		conversation_id: str,
		force_new_seed: bool,
	) -> tuple[int, bool]:
		"""Decide which seed to render with. Returns (seed, sampled_existing).

		Precedence:
		  1. force_new_seed → fresh mint (overrides any prior lock)
		  2. Conversation has a locked-in seed for this pool → reuse it
		  3. First time this conversation touches this pool → sample an
		     existing seed at random, else mint a fresh one.
		"""
		if force_new_seed:
			return self._mint_seed(), False
		locked = self._session_pools.get(conversation_id, {}).get(pool_key)
		if locked is not None:
			return locked, True
		with self.db.lock:
			row = self.db.conn.execute(
				"SELECT seed FROM image_generations WHERE pool_key = ? "
				"ORDER BY RANDOM() LIMIT 1",
				(pool_key,),
			).fetchone()
		if row is not None and row["seed"] is not None:
			return int(row["seed"]), True
		return self._mint_seed(), False

	@staticmethod
	def _mint_seed() -> int:
		"""Generate a fresh random 32-bit seed."""
		return random.randint(0, 2**31 - 1)

	def _lookup_cache(self, cache_key: str) -> tuple[int, Path] | None:
		"""Find an existing generation by cache_key. Verifies the file still exists."""
		with self.db.lock:
			row = self.db.conn.execute(
				"SELECT id, file_path FROM image_generations WHERE cache_key = ?",
				(cache_key,),
			).fetchone()
		if not row:
			return None
		path = Path(row["file_path"])
		if not path.exists():
			# Disk eviction without DB cleanup — treat as a miss. The
			# new render will overwrite the row's file in place.
			logger.warning("cache row exists but file missing: %s", path)
			return None
		return int(row["id"]), path

	def _bump_cache_use(self, generation_id: int) -> None:
		"""Update last_used / use_count on a cache hit."""
		now = self._now()
		with self.db.lock:
			self.db.conn.execute(
				"UPDATE image_generations SET last_used = ?, use_count = use_count + 1 WHERE id = ?",
				(now, generation_id),
			)
			self.db.conn.commit()

	def _write_image(self, cache_key: str, image_bytes: bytes) -> Path:
		"""Persist the rendered image to disk and return its path."""
		self.file_dir.mkdir(parents=True, exist_ok=True)
		path = self.file_dir / f"{cache_key}.webp"
		path.write_bytes(image_bytes)
		return path

	def _insert_generation(
		self,
		*,
		cache_key: str,
		pool_key: str,
		skill_slug: str,
		input_generation_id: int | None,
		controls: dict,
		controls_hash: str,
		seed: int,
		file_path: Path,
		n_bytes: int,
	) -> int:
		"""Insert a new image_generations row and return its id."""
		now = self._now()
		with self.db.lock:
			cur = self.db.conn.execute(
				"INSERT INTO image_generations ("
				"  cache_key, pool_key, skill_slug, input_generation_id, "
				"  controls_json, controls_hash, seed, file_path, bytes, "
				"  created_at, last_used, use_count"
				") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
				(
					cache_key, pool_key, skill_slug, input_generation_id,
					json.dumps(controls or {}), controls_hash, seed,
					str(file_path), n_bytes, now, now,
				),
			)
			self.db.conn.commit()
			return int(cur.lastrowid)

	# =================================================================
	# USER ACTIONS
	# =================================================================

	def record_user_action(
		self,
		*,
		user_id: str,
		canvas_id: str,
		action: str,
		image_path: str | None = None,
	) -> None:
		"""Record a user action on a canvas and fan out to skill scoring.

		``action`` is one of ``share``, ``save``, ``download``, ``link_open``,
		``remix``. Other strings are stored but won't update skill_scores.
		"""
		canvas = self.get_canvas(canvas_id)
		if not canvas:
			raise KeyError(f"unknown canvas {canvas_id!r}")
		now = self._now()
		with self.db.lock:
			self.db.conn.execute(
				"INSERT INTO user_canvas_actions (user_id, canvas_id, action, ts) "
				"VALUES (?, ?, ?, ?)",
				(user_id, canvas_id, action, now),
			)
			self.db.conn.commit()

		chain = [
			{"slug": layer["skill_slug"], "kind": (self.get_skill(layer["skill_slug"]) or {}).get("kind", "creation")}
			for layer in canvas["layers"]
		]
		record_event(self.db, action, chain, image_path)
		logger.info(
			"record_user_action user=%s canvas=%s action=%s layers=%d",
			user_id, canvas_id, action, len(chain),
		)

	# =================================================================
	# QUERIES
	# =================================================================

	def list_user_canvases(self, user_id: str, action: str) -> list[dict]:
		"""Canvases a given user has performed ``action`` on, newest first."""
		with self.db.lock:
			rows = self.db.conn.execute(
				"SELECT c.*, MAX(a.ts) AS last_ts FROM user_canvas_actions a "
				"JOIN canvases c ON c.id = a.canvas_id "
				"WHERE a.user_id = ? AND a.action = ? "
				"GROUP BY c.id ORDER BY last_ts DESC",
				(user_id, action),
			).fetchall()
		return [dict(r) for r in rows]
