"""Tests for canvas.render.

We don't run the real skill subprocess sandbox here — we monkeypatch
``canvas.render.run_skill`` with a stub that writes a tiny PNG to the
output path. That lets us assert folder layout, seed handling, cache
behavior, and chain-input threading without paying the subprocess cost.

We also redirect ``RENDERS_DIR`` to a project-local temp dir per test so
nothing escapes into the real user data directory.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from canvas import render as canvas_render
from canvas.state import CanvasState


# =================================================================
# fixtures / fakes
# =================================================================

def _write_fake_png(output_image_path: Path, color: tuple[int, int, int, int] = (32, 64, 96, 255)) -> None:
	"""Drop a tiny PNG at the given path so the WebP re-encode step has something to read."""
	output_image_path.parent.mkdir(parents=True, exist_ok=True)
	Image.new("RGBA", (4, 4), color).save(output_image_path, format="PNG")


def _install_fake_run_skill(monkeypatch):
	"""Replace render.run_skill with a counter-equipped stub."""
	calls: list[dict] = []

	def fake_run_skill(skill, *, params, palette, size, seed, input_image_path, output_image_path, **kwargs):
		calls.append({
			"slug": skill.slug,
			"kind": skill.kind,
			"params": dict(params),
			"palette_id": palette.id,
			"size": size,
			"seed": seed,
			"input_image_path": str(input_image_path) if input_image_path else None,
			"output_image_path": str(output_image_path),
		})
		# Vary the color a touch so successive renders differ if needed.
		_write_fake_png(Path(output_image_path), color=(seed % 200, (seed // 7) % 200, 96, 255))
		return {"ok": True}

	monkeypatch.setattr(canvas_render, "run_skill", fake_run_skill)
	return calls


@pytest.fixture
def renders_dir(request, monkeypatch):
	"""Redirect canvas_renders to a project-local temp dir for the test.

	Uses a name based on the test id rather than tmp_path because pytest's
	default tmp dir is unreadable on this Windows box (pre-existing issue
	shared with the other canvas test files).
	"""
	target = Path(f".canvas_render_test_{request.node.name}")
	if target.exists():
		shutil.rmtree(target, ignore_errors=True)
	target.mkdir(parents=True, exist_ok=True)
	monkeypatch.setattr(canvas_render, "RENDERS_DIR", target)
	monkeypatch.setattr(canvas_render, "PREFIX_CACHE_DIR", target.with_name(target.name + "_prefix"))
	yield target
	shutil.rmtree(target, ignore_errors=True)
	shutil.rmtree(canvas_render.PREFIX_CACHE_DIR, ignore_errors=True)


def _skill(slug: str, kind: str = "creation"):
	"""Build a minimal Skill stand-in matching what render.py reads (slug + kind)."""
	return SimpleNamespace(slug=slug, kind=kind, code="")


def _loader(skills: dict):
	"""Build a skill_loader callable from a dict."""
	return lambda slug: skills.get(slug)


def _state_with_creation(slug: str = "fractal", **controls) -> CanvasState:
	"""Helper: CanvasState with a single creation layer."""
	cs = CanvasState()
	cs.enact("add_layer", {"skill_slug": slug, "kind": "creation", "controls": controls})
	return cs


# =================================================================
# pool_hash
# =================================================================

def test_pool_hash_is_deterministic():
	"""Same canvas state → same hash, twice in a row."""
	cs = _state_with_creation("fractal", zoom=1.0)
	assert canvas_render.pool_hash(cs.canvas) == canvas_render.pool_hash(cs.canvas)


def test_pool_hash_changes_when_controls_change():
	"""A different control value produces a different hash."""
	a = _state_with_creation("fractal", zoom=1.0)
	b = _state_with_creation("fractal", zoom=2.0)
	assert canvas_render.pool_hash(a.canvas) != canvas_render.pool_hash(b.canvas)


def test_pool_hash_ignores_canvas_id():
	"""Two canvases with identical configs share a folder regardless of id."""
	a = CanvasState(canvas_id="aaa")
	b = CanvasState(canvas_id="bbb")
	a.enact("add_layer", {"skill_slug": "fractal", "kind": "creation", "controls": {"z": 1}})
	b.enact("add_layer", {"skill_slug": "fractal", "kind": "creation", "controls": {"z": 1}})
	assert canvas_render.pool_hash(a.canvas) == canvas_render.pool_hash(b.canvas)


def test_pool_hash_is_control_order_independent():
	"""Same controls in a different dict-insertion order still hash the same."""
	a = _state_with_creation("fractal", zoom=1.0, angle=30)
	b = _state_with_creation("fractal", angle=30, zoom=1.0)
	assert canvas_render.pool_hash(a.canvas) == canvas_render.pool_hash(b.canvas)


def test_pool_hash_changes_when_palette_changes():
	"""Palette is part of the render-determining state."""
	cs = _state_with_creation("fractal")
	h0 = canvas_render.pool_hash(cs.canvas)
	cs.enact("set_palette", {"palette_id": "obsidian"})
	assert canvas_render.pool_hash(cs.canvas) != h0


# =================================================================
# render_canvas — output layout + chaining
# =================================================================

def test_render_writes_to_pool_hash_folder(monkeypatch, renders_dir):
	"""Output file lives at {pool_hash}/{seed}.webp under RENDERS_DIR."""
	_install_fake_run_skill(monkeypatch)
	cs = _state_with_creation("fractal")
	skills = {"fractal": _skill("fractal")}

	result = canvas_render.render_canvas(cs, skill_loader=_loader(skills), seed=42)
	expected = renders_dir / canvas_render.pool_hash(cs.canvas) / "42.webp"
	assert result.image_path == expected
	assert result.image_path.is_file()
	assert result.seed == 42
	assert result.pool_hash == canvas_render.pool_hash(cs.canvas)
	assert result.cache_hit is False


def test_render_chains_input_through_layers(monkeypatch, renders_dir):
	"""Layer N gets layer N-1's output as its input."""
	calls = _install_fake_run_skill(monkeypatch)
	cs = _state_with_creation("fractal")
	cs.enact("add_layer", {"skill_slug": "swirl", "kind": "transform", "controls": {"angle": 30}})
	skills = {"fractal": _skill("fractal"), "swirl": _skill("swirl", "transform")}

	canvas_render.render_canvas(cs, skill_loader=_loader(skills), seed=7)

	assert len(calls) == 2
	# Creation has no input.
	assert calls[0]["slug"] == "fractal"
	assert calls[0]["input_image_path"] is None
	# Transform sees the creation's output path as input.
	assert calls[1]["slug"] == "swirl"
	assert calls[1]["input_image_path"] == calls[0]["output_image_path"]


def test_render_passes_seed_and_size_to_skills(monkeypatch, renders_dir):
	"""Every layer is invoked with the same resolved seed and the canvas size."""
	calls = _install_fake_run_skill(monkeypatch)
	cs = _state_with_creation("fractal")
	cs.enact("add_layer", {"skill_slug": "swirl", "kind": "transform"})
	cs.enact("set_size", {"size": 768})
	skills = {"fractal": _skill("fractal"), "swirl": _skill("swirl", "transform")}

	canvas_render.render_canvas(cs, skill_loader=_loader(skills), seed=99)

	assert {c["seed"] for c in calls} == {99}
	assert {c["size"] for c in calls} == {768}


def test_render_resolves_palette_control_out_of_params(monkeypatch, renders_dir):
	"""Palette swatches change the palette object, not the skill kwargs."""
	calls = _install_fake_run_skill(monkeypatch)
	cs = _state_with_creation("fractal", palette="frost", zoom=2)

	canvas_render.render_canvas(cs, skill_loader=_loader({"fractal": _skill("fractal")}), seed=4)

	assert calls[0]["palette_id"] == "frost"
	assert calls[0]["params"] == {"zoom": 2}


def test_render_raises_on_empty_chain(monkeypatch, renders_dir):
	"""Rendering a chainless canvas is a programmer error."""
	_install_fake_run_skill(monkeypatch)
	cs = CanvasState()
	with pytest.raises(ValueError):
		canvas_render.render_canvas(cs, skill_loader=_loader({}))


def test_render_raises_on_unknown_skill(monkeypatch, renders_dir):
	"""Chain referencing a slug the loader can't resolve fails clearly."""
	_install_fake_run_skill(monkeypatch)
	cs = _state_with_creation("ghost")
	with pytest.raises(ValueError, match="ghost"):
		canvas_render.render_canvas(cs, skill_loader=_loader({}))


# =================================================================
# seed handling + caching
# =================================================================

def test_explicit_seed_is_honored(monkeypatch, renders_dir):
	"""Passing seed=N writes exactly N.webp."""
	_install_fake_run_skill(monkeypatch)
	cs = _state_with_creation("fractal")
	r = canvas_render.render_canvas(cs, skill_loader=_loader({"fractal": _skill("fractal")}), seed=123)
	assert r.seed == 123
	assert r.image_path.name == "123.webp"


def test_force_new_seed_mints_fresh_each_call(monkeypatch, renders_dir):
	"""force_new_seed=True bypasses any pool cache and writes a new file."""
	_install_fake_run_skill(monkeypatch)
	cs = _state_with_creation("fractal")
	loader = _loader({"fractal": _skill("fractal")})
	r1 = canvas_render.render_canvas(cs, skill_loader=loader, force_new_seed=True)
	r2 = canvas_render.render_canvas(cs, skill_loader=loader, force_new_seed=True)
	assert r1.seed != r2.seed
	assert r1.cache_hit is False and r2.cache_hit is False
	folder = canvas_render.folder_for(cs.canvas)
	assert sorted(folder.glob("*.webp")) == sorted({r1.image_path, r2.image_path})


def test_render_returns_cached_when_pool_has_files(monkeypatch, renders_dir):
	"""Second call with no explicit seed reuses an existing file (cache hit)."""
	calls = _install_fake_run_skill(monkeypatch)
	cs = _state_with_creation("fractal")
	loader = _loader({"fractal": _skill("fractal")})

	r1 = canvas_render.render_canvas(cs, skill_loader=loader, seed=10)
	assert r1.cache_hit is False
	calls_after_first = len(calls)
	r2 = canvas_render.render_canvas(cs, skill_loader=loader)
	assert r2.cache_hit is True
	assert len(calls) == calls_after_first   # no new subprocess invocation
	assert r2.image_path == r1.image_path


def test_first_render_locks_existing_pool_seed(monkeypatch, renders_dir):
	"""A new session adopts an existing pool seed instead of minting another."""
	_install_fake_run_skill(monkeypatch)
	cs1 = _state_with_creation("fractal")
	loader = _loader({"fractal": _skill("fractal")})
	r1 = canvas_render.render_canvas(cs1, skill_loader=loader, seed=77)
	cs2 = _state_with_creation("fractal")
	r2 = canvas_render.render_canvas(cs2, skill_loader=loader)
	assert r2.cache_hit is True
	assert r2.seed == r1.seed == cs2.render_seed


def test_appending_layer_reuses_cached_prefix(monkeypatch, renders_dir):
	"""After rendering two layers, appending a third only runs the third."""
	calls = _install_fake_run_skill(monkeypatch)
	cs = _state_with_creation("fractal")
	cs.enact("add_layer", {"skill_slug": "swirl", "kind": "transform"})
	loader = _loader({"fractal": _skill("fractal"), "swirl": _skill("swirl", "transform"), "grain": _skill("grain", "transform")})
	canvas_render.render_canvas(cs, skill_loader=loader, seed=42)
	calls.clear()
	cs.enact("add_layer", {"skill_slug": "grain", "kind": "transform"})
	r = canvas_render.render_canvas(cs, skill_loader=loader)
	assert r.seed == 42
	assert [c["slug"] for c in calls] == ["grain"]
	assert calls[0]["input_image_path"] is not None


def test_editing_last_layer_reuses_earlier_prefix(monkeypatch, renders_dir):
	"""Changing layer 3 keeps layers 1-2 cached and reruns only layer 3."""
	calls = _install_fake_run_skill(monkeypatch)
	cs = _state_with_creation("fractal")
	cs.enact("add_layer", {"skill_slug": "swirl", "kind": "transform"})
	cs.enact("add_layer", {"skill_slug": "grain", "kind": "transform", "controls": {"a": 1}})
	loader = _loader({"fractal": _skill("fractal"), "swirl": _skill("swirl", "transform"), "grain": _skill("grain", "transform")})
	canvas_render.render_canvas(cs, skill_loader=loader, seed=42)
	calls.clear()
	cs.enact("set_control", {"chain_index": 2, "name": "a", "value": 2})
	canvas_render.render_canvas(cs, skill_loader=loader)
	assert [c["slug"] for c in calls] == ["grain"]


def test_editing_first_layer_invalidates_suffix(monkeypatch, renders_dir):
	"""Changing layer 1 changes every prefix, so the full chain reruns."""
	calls = _install_fake_run_skill(monkeypatch)
	cs = _state_with_creation("fractal", zoom=1)
	cs.enact("add_layer", {"skill_slug": "swirl", "kind": "transform"})
	cs.enact("add_layer", {"skill_slug": "grain", "kind": "transform"})
	loader = _loader({"fractal": _skill("fractal"), "swirl": _skill("swirl", "transform"), "grain": _skill("grain", "transform")})
	canvas_render.render_canvas(cs, skill_loader=loader, seed=42)
	calls.clear()
	cs.enact("set_control", {"chain_index": 0, "name": "zoom", "value": 2})
	canvas_render.render_canvas(cs, skill_loader=loader)
	assert [c["slug"] for c in calls] == ["fractal", "swirl", "grain"]


def test_force_new_seed_does_not_reuse_old_prefix(monkeypatch, renders_dir):
	"""Regenerate-style renders use a fresh seed and therefore rerun all layers."""
	calls = _install_fake_run_skill(monkeypatch)
	cs = _state_with_creation("fractal")
	cs.enact("add_layer", {"skill_slug": "swirl", "kind": "transform"})
	loader = _loader({"fractal": _skill("fractal"), "swirl": _skill("swirl", "transform")})
	r1 = canvas_render.render_canvas(cs, skill_loader=loader, seed=42)
	calls.clear()
	r2 = canvas_render.render_canvas(cs, skill_loader=loader, force_new_seed=True)
	assert r2.seed != r1.seed
	assert [c["slug"] for c in calls] == ["fractal", "swirl"]


def test_explicit_seed_with_existing_file_is_a_cache_hit(monkeypatch, renders_dir):
	"""Passing the same explicit seed twice doesn't re-render."""
	calls = _install_fake_run_skill(monkeypatch)
	cs = _state_with_creation("fractal")
	loader = _loader({"fractal": _skill("fractal")})
	canvas_render.render_canvas(cs, skill_loader=loader, seed=55)
	calls_after_first = len(calls)
	r = canvas_render.render_canvas(cs, skill_loader=loader, seed=55)
	assert r.cache_hit is True
	assert len(calls) == calls_after_first


def test_existing_seeds_lists_pool(monkeypatch, renders_dir):
	"""existing_seeds returns every seed in the pool folder, sorted."""
	_install_fake_run_skill(monkeypatch)
	cs = _state_with_creation("fractal")
	loader = _loader({"fractal": _skill("fractal")})
	canvas_render.render_canvas(cs, skill_loader=loader, seed=3)
	canvas_render.render_canvas(cs, skill_loader=loader, seed=1)
	canvas_render.render_canvas(cs, skill_loader=loader, seed=2)
	assert canvas_render.existing_seeds(cs.canvas) == [1, 2, 3]


def test_existing_seeds_empty_when_folder_absent(renders_dir):
	"""No folder yet → no seeds (and no exception)."""
	cs = _state_with_creation("fractal")
	assert canvas_render.existing_seeds(cs.canvas) == []
