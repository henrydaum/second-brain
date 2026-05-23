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

There is no separate prefix-cache folder. Each intermediate layer's
output IS the canonical render of the canvas truncated at that layer,
written to its own pool_hash folder and registered in canvas_pools.
Rendering a long chain therefore warm-caches every shorter prefix at
no extra cost — including for other users who construct an identical
shorter chain later. All renders are public (content-addressed by
deterministic inputs; no secret to leak).
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
WEBP_QUALITY = 90
WEBP_METHOD = 4
POOL_HASH_LEN = 16


@dataclass
class RenderResult:
	"""Outcome of a ``render_canvas`` call."""

	image_path: Path
	seed: int
	pool_hash: str
	cache_hit: bool
	# Post-render validator warning from the LAST layer's run_skill, if any.
	# Set to a short tag like "palette_drift" / "blank_canvas" / "transparent_canvas".
	# Intermediate-layer warnings are not propagated — only the final pixels matter.
	warning: str | None = None
	warning_message: str | None = None


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


def _truncated(canvas, count: int) -> Canvas:
	return Canvas(size=canvas.size, palette_id=canvas.palette_id, layers=list(canvas.layers[:count]))


def _prefix_hash(canvas, count: int) -> str:
	"""pool_hash of the canvas truncated to its first ``count`` layers."""
	return pool_hash(_truncated(canvas, count))


def _prefix_path(canvas, count: int, seed: int) -> Path:
	return RENDERS_DIR / _prefix_hash(canvas, count) / f"{int(seed)}.webp"


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
	on_event: Callable[[dict], None] | None = None,
	worker_pool: Any = None,
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
			_emit(on_event, status="cached", total_layers=len(canvas.layers), cached_layers=len(canvas.layers), seed=cached_seed, pool_hash=folder.name)
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
		_emit(on_event, status="cached", total_layers=len(canvas.layers), cached_layers=len(canvas.layers), seed=seed_val, pool_hash=folder.name)
		return RenderResult(out_path, seed_val, folder.name, cache_hit=True)

	# Cache miss: walk the chain in a temp workdir, then re-encode the
	# final PNG as WebP into the canonical path.
	fallback_palette = _default_get_palette(canvas.palette_id)
	with tempfile.TemporaryDirectory(prefix="canvas_render_") as workdir:
		workdir_path = Path(workdir)
		start_idx, current_input = _longest_prefix(canvas, seed_val, workdir_path)
		_emit(on_event, status="started", total_layers=len(canvas.layers), cached_layers=start_idx, seed=seed_val, pool_hash=folder.name)
		last_warning: dict | None = None
		try:
			for idx, layer in enumerate(canvas.layers[start_idx:], start=start_idx):
				slug = layer.get("slug")
				skill = skill_loader(slug) if slug else None
				if skill is None:
					raise ValueError(f"chain references unknown skill: {slug!r}")
				step_png = workdir_path / f"step_{idx:02d}.png"
				params, palette = resolve_entry(layer, fallback_palette=fallback_palette)
				_emit(on_event, status="layer_started", layer_index=idx + 1, total_layers=len(canvas.layers), cached_layers=start_idx, skill_slug=str(slug), seed=seed_val, pool_hash=folder.name)
				run_result = run_skill(
					skill,
					params=params,
					palette=palette,
					size=int(canvas.size),
					seed=int(seed_val),
					input_image_path=current_input,
					output_image_path=step_png,
					worker_pool=worker_pool,
				)
				# Only the final layer's warning matters for the user-visible result;
				# overwrite as we go so the last iteration wins.
				last_warning = run_result if run_result and run_result.get("warning") else None
				current_input = step_png
				# Write this layer's output as the canonical render of the canvas
				# truncated to (idx + 1) layers. Same folder structure as the full
				# chain — the full chain is just the case where (idx + 1) == N.
				# Register the truncated config so anyone authoring it directly
				# gets an instant cache hit and the share/gallery routes resolve it.
				cache_path = _prefix_path(canvas, idx + 1, seed_val)
				cache_path.parent.mkdir(parents=True, exist_ok=True)
				try:
					with Image.open(step_png) as img:
						img.save(cache_path, format="WEBP", quality=WEBP_QUALITY, method=WEBP_METHOD)
				except Exception:
					logger.exception("render write failed prefix=%s seed=%s", cache_path.parent.name, seed_val)
				if db is not None:
					try:
						from canvas.persistence import save_pool
						save_pool(db, pool_hash=cache_path.parent.name, state=_truncated(canvas, idx + 1).to_dict())
					except Exception:
						logger.exception("save_pool failed for prefix pool=%s", cache_path.parent.name)
				_emit(on_event, status="layer_finished", layer_index=idx + 1, total_layers=len(canvas.layers), cached_layers=start_idx, skill_slug=str(slug), seed=seed_val, pool_hash=folder.name)
		except Exception as e:
			_emit(on_event, status="error", total_layers=len(canvas.layers), cached_layers=start_idx, seed=seed_val, pool_hash=folder.name, error=str(e))
			raise

		_emit(on_event, status="finished", total_layers=len(canvas.layers), cached_layers=start_idx, seed=seed_val, pool_hash=folder.name)

	logger.info(
		"render canvas_id=%s pool=%s seed=%d layers=%d",
		cs.canvas_id, folder.name, seed_val, len(canvas.layers),
	)
	return RenderResult(
		out_path, seed_val, folder.name, cache_hit=False,
		warning=(last_warning or {}).get("warning"),
		warning_message=(last_warning or {}).get("warning_message"),
	)


def _persist_seed(db: Any, cs: CanvasState) -> None:
	if db is None:
		return
	try:
		from canvas.persistence import save
		save(db, cs)
	except Exception:
		logger.exception("save render_seed failed for canvas_id=%s", cs.canvas_id)


def _emit(on_event: Callable[[dict], None] | None, **payload: Any) -> None:
	if on_event is None:
		return
	try:
		on_event(payload)
	except Exception:
		logger.exception("render progress callback failed")


def bus_progress(session_key: str | None, timeout_s: float = 30.0):
	if not session_key:
		return None
	from events.event_bus import bus
	from events.event_channels import CANVAS_RENDER_STATUS
	return lambda ev: bus.emit(CANVAS_RENDER_STATUS, {"session_key": session_key, "timeout_s": timeout_s, **ev})


def _longest_prefix(canvas, seed: int, workdir_path: Path) -> tuple[int, Path | None]:
	for count in range(len(canvas.layers) - 1, 0, -1):
		p = _prefix_path(canvas, count, seed)
		if not p.is_file():
			continue
		local = workdir_path / f"prefix_{count:02d}.png"
		try:
			with Image.open(p) as img:
				img.save(local, format="PNG")
			return count, local
		except Exception:
			logger.exception("prefix cache read failed path=%s", p)
	return 0, None
