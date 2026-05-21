"""Render a CanvasState's chain to a single WebP file.

Folder layout under DATA_DIR/canvas_renders/:

    {pool_hash}/
        {seed}.webp
        {seed}.webp
        ...

``pool_hash`` captures the render-determining canvas state — layers
(slug + kind + sorted controls), size, palette_id. Two canvases with
identical configurations share a folder, so a directory listing IS the
seed pool for that exact configuration. Edit a control → new pool_hash
→ new folder; prior renders are preserved untouched.

Bare-bones today: stateless and final-composite-only. No intermediate
layer cache, no per-session seed lock-in (that lives a layer above when
we wire frontends/conversations to canvases).
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from canvas.canvas import Canvas
from canvas.state import CanvasState
from paths import DATA_DIR
from plugins.helpers.palettes import get_palette as _default_get_palette
from plugins.skills.helpers.skill_runner import resolve_entry, run_skill

logger = logging.getLogger("CanvasRender")

RENDERS_DIR = DATA_DIR / "canvas_renders"
PREFIX_CACHE_DIR = DATA_DIR / "canvas_prefix_cache"
WEBP_QUALITY = 90
WEBP_METHOD = 6
POOL_HASH_LEN = 16


@dataclass
class RenderResult:
	"""Outcome of a ``render_canvas`` call."""

	image_path: Path
	seed: int
	pool_hash: str
	cache_hit: bool


# ── pool hash ────────────────────────────────────────────────────────

def pool_hash(canvas) -> str:
	"""Deterministic hash of the render-determining canvas state.

	Inputs: layers (slug + kind + sorted controls), size, palette_id.
	Excludes canvas_id, title/artist, history — those don't affect what
	pixels come out.
	"""
	layers = [
		{
			"slug": str(layer.get("slug") or ""),
			"kind": str(layer.get("kind") or ""),
			"controls": dict(sorted((layer.get("controls") or {}).items())),
		}
		for layer in (canvas.layers or [])
	]
	payload = {
		"layers": layers,
		"size": int(canvas.size),
		"palette_id": str(canvas.palette_id),
	}
	raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
	return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:POOL_HASH_LEN]


def folder_for(canvas) -> Path:
	"""Directory where renders for this canvas configuration live."""
	return RENDERS_DIR / pool_hash(canvas)


def existing_seeds(canvas) -> list[int]:
	"""All seeds already rendered for this exact configuration, sorted."""
	folder = folder_for(canvas)
	if not folder.is_dir():
		return []
	seeds: list[int] = []
	for p in folder.iterdir():
		if p.suffix.lower() != ".webp":
			continue
		try:
			seeds.append(int(p.stem))
		except ValueError:
			continue
	return sorted(seeds)


def _prefix_hash(canvas, count: int) -> str:
	"""Hash the first ``count`` layers with the same render-determining inputs."""
	return pool_hash(Canvas(size=canvas.size, palette_id=canvas.palette_id, layers=list(canvas.layers[:count])))


def _prefix_path(canvas, count: int, seed: int) -> Path:
	return PREFIX_CACHE_DIR / _prefix_hash(canvas, count) / f"{int(seed)}.png"


# ── render ───────────────────────────────────────────────────────────

def _mint_seed() -> int:
	"""Random 31-bit seed (matches the existing convention in skill_cache)."""
	return random.randint(1, 2_147_483_647)


def render_canvas(
	cs: CanvasState,
	*,
	skill_loader: Callable[[str], Any],
	seed: int | None = None,
	force_new_seed: bool = False,
	db: Any = None,
) -> RenderResult:
	"""Render ``cs.canvas``'s chain to a WebP file and return the result.

	Seed selection:
	  - explicit ``seed=N``: use it as-is.
	  - ``force_new_seed=True``: mint a fresh one.
	  - else: if the pool folder has at least one existing render, return
	    the most-recently-modified one (cache hit, no subprocess). Empty
	    pool → mint and render.

	``skill_loader(slug) -> Skill | None`` mirrors what
	``skill_runner.replay_chain`` accepts; caller provides the lookup.

	If ``db`` is provided, a fresh render also writes the configuration
	to ``canvas_pools`` (idempotent on pool_hash). That row is what
	``/share/{pool_hash}`` and remix resolve against — every config that
	was ever rendered is publicly addressable.
	"""
	canvas = cs.canvas
	if not canvas.layers:
		raise ValueError("nothing to render — canvas has no layers")

	folder = folder_for(canvas)
	folder.mkdir(parents=True, exist_ok=True)

	# Register the configuration in canvas_pools on every render call (not
	# just cache misses). INSERT OR IGNORE makes this idempotent, and
	# running it on hits backfills pools for canvases rendered before
	# this code path existed. Anything that needs to resolve a
	# pool_hash (share page, QR, gallery/archive listings) depends on
	# this row.
	if db is not None:
		try:
			from canvas.persistence import save_pool
			save_pool(db, pool_hash=folder.name, state=canvas.to_dict())
		except Exception:
			logger.exception("save_pool failed for pool=%s", folder.name)

	# Resolve seed + decide whether a cache short-circuit applies.
	if seed is not None:
		seed_val = int(seed)
		cs.render_seed = seed_val
	elif force_new_seed:
		seed_val = _mint_seed()
		cs.render_seed = seed_val
	elif getattr(cs, "render_seed", None) is not None:
		seed_val = int(cs.render_seed)
	else:
		# Default path: reuse the most recently modified render in the pool.
		existing = sorted(
			(p for p in folder.iterdir() if p.suffix.lower() == ".webp"),
			key=lambda p: p.stat().st_mtime,
			reverse=True,
		)
		for hit in existing:
			try:
				cached_seed = int(hit.stem)
			except ValueError:
				continue
			logger.debug(
				"render cache HIT (newest-in-pool) canvas_id=%s pool=%s seed=%d",
				cs.canvas_id, folder.name, cached_seed,
			)
			cs.render_seed = cached_seed
			_persist_seed(db, cs)
			return RenderResult(hit, cached_seed, folder.name, cache_hit=True)
		# Pool empty — mint a fresh seed.
		seed_val = _mint_seed()
		cs.render_seed = seed_val

	_persist_seed(db, cs)

	out_path = folder / f"{seed_val}.webp"
	if out_path.is_file():
		logger.debug(
			"render cache HIT (exact seed) canvas_id=%s pool=%s seed=%d",
			cs.canvas_id, folder.name, seed_val,
		)
		return RenderResult(out_path, seed_val, folder.name, cache_hit=True)

	# Cache miss: walk the chain in a temp workdir, then re-encode the
	# final PNG as WebP into the canonical path.
	fallback_palette = _default_get_palette(canvas.palette_id)
	with tempfile.TemporaryDirectory(prefix="canvas_render_") as workdir:
		workdir_path = Path(workdir)
		start_idx, current_input = _longest_prefix(canvas, seed_val, workdir_path)
		for idx, layer in enumerate(canvas.layers[start_idx:], start=start_idx):
			slug = layer.get("slug")
			skill = skill_loader(slug) if slug else None
			if skill is None:
				raise ValueError(f"chain references unknown skill: {slug!r}")
			step_png = workdir_path / f"step_{idx:02d}.png"
			params, palette = resolve_entry(layer, fallback_palette=fallback_palette)
			run_skill(
				skill,
				params=params,
				palette=palette,
				size=int(canvas.size),
				seed=int(seed_val),
				input_image_path=current_input,
				output_image_path=step_png,
			)
			current_input = step_png
			cache_path = _prefix_path(canvas, idx + 1, seed_val)
			cache_path.parent.mkdir(parents=True, exist_ok=True)
			try:
				with Image.open(step_png) as img:
					img.save(cache_path, format="PNG")
			except Exception:
				logger.exception("prefix cache write failed prefix=%s seed=%s", cache_path.parent.name, seed_val)

		# current_input is the final PNG produced by the last layer.
		with Image.open(current_input) as img:
			img.save(out_path, format="WEBP", quality=WEBP_QUALITY, method=WEBP_METHOD)

	logger.info(
		"render canvas_id=%s pool=%s seed=%d layers=%d",
		cs.canvas_id, folder.name, seed_val, len(canvas.layers),
	)
	return RenderResult(out_path, seed_val, folder.name, cache_hit=False)


def _persist_seed(db: Any, cs: CanvasState) -> None:
	if db is None:
		return
	try:
		from canvas.persistence import save
		save(db, cs)
	except Exception:
		logger.exception("save render_seed failed for canvas_id=%s", cs.canvas_id)


def _longest_prefix(canvas, seed: int, workdir_path: Path) -> tuple[int, Path | None]:
	for count in range(len(canvas.layers) - 1, 0, -1):
		p = _prefix_path(canvas, count, seed)
		if p.is_file():
			local = workdir_path / f"prefix_{count:02d}.png"
			try:
				with Image.open(p) as img:
					img.save(local, format="PNG")
				return count, local
			except Exception:
				logger.exception("prefix cache read failed path=%s", p)
	return 0, None
