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

from canvas.state import CanvasState
from paths import DATA_DIR
from plugins.helpers.palettes import get_palette as _default_get_palette
from plugins.skills.helpers.skill_runner import run_skill

logger = logging.getLogger("CanvasRender")

RENDERS_DIR = DATA_DIR / "canvas_renders"
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

	# Resolve seed + decide whether a cache short-circuit applies.
	if seed is not None:
		seed_val = int(seed)
	elif force_new_seed:
		seed_val = _mint_seed()
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
			return RenderResult(hit, cached_seed, folder.name, cache_hit=True)
		# Pool empty — mint a fresh seed.
		seed_val = _mint_seed()

	out_path = folder / f"{seed_val}.webp"
	if out_path.is_file():
		logger.debug(
			"render cache HIT (exact seed) canvas_id=%s pool=%s seed=%d",
			cs.canvas_id, folder.name, seed_val,
		)
		return RenderResult(out_path, seed_val, folder.name, cache_hit=True)

	# Cache miss: walk the chain in a temp workdir, then re-encode the
	# final PNG as WebP into the canonical path.
	palette = _default_get_palette(canvas.palette_id)
	with tempfile.TemporaryDirectory(prefix="canvas_render_") as workdir:
		workdir_path = Path(workdir)
		current_input: Path | None = None
		for idx, layer in enumerate(canvas.layers):
			slug = layer.get("slug")
			skill = skill_loader(slug) if slug else None
			if skill is None:
				raise ValueError(f"chain references unknown skill: {slug!r}")
			step_png = workdir_path / f"step_{idx:02d}.png"
			run_skill(
				skill,
				params=dict(layer.get("controls") or {}),
				palette=palette,
				size=int(canvas.size),
				seed=int(seed_val),
				input_image_path=current_input,
				output_image_path=step_png,
			)
			current_input = step_png

		# current_input is the final PNG produced by the last layer.
		with Image.open(current_input) as img:
			img.save(out_path, format="WEBP", quality=WEBP_QUALITY, method=WEBP_METHOD)

	# Cache miss → this configuration was just rendered for the first
	# time. Register it in canvas_pools so it's publicly addressable.
	if db is not None:
		try:
			from canvas.persistence import save_pool
			save_pool(db, pool_hash=folder.name, state=canvas.to_dict())
		except Exception:
			logger.exception("save_pool failed for pool=%s", folder.name)

	logger.info(
		"render canvas_id=%s pool=%s seed=%d layers=%d",
		cs.canvas_id, folder.name, seed_val, len(canvas.layers),
	)
	return RenderResult(out_path, seed_val, folder.name, cache_hit=False)
