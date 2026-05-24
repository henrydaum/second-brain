"""End-to-end tests for the `object` skill kind.

Drives the real sandbox subprocess so we exercise the full composite path
(prior-canvas RGBA → object commits sparse-alpha → framework
alpha-composites → PNG written). Slow-ish, but the alternative is reaching
into private helpers, and the composite is the whole feature.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageChops

from canvas.state import CanvasState
from paths import ROOT_DIR
from plugins.plugin_discovery import discover_skills
from plugins.helpers.palettes import get_palette
from plugins.skills.helpers.skill_registry import SkillRegistry
from plugins.skills.helpers.skill_runner import SkillRunError, default_controls, run_skill
from plugins.skills.helpers.skill_store import _KINDS, assert_valid


# =================================================================
# Kind whitelist
# =================================================================

def test_object_is_an_accepted_kind():
	assert "object" in _KINDS


def test_canvas_state_rejects_object_at_layer_zero():
	"""push_chain_entry handles object like filter — append, not replace —
	and the canvas starts empty, so the tool layer is what refuses it.
	Here we cover the canvas-level invariant: layer 0 reorder still requires
	a background."""
	cs = CanvasState()
	cs.enact("add_layer", {"skill_slug": "bg", "kind": "background"})
	cs.enact("add_layer", {"skill_slug": "overlay", "kind": "object"})
	r = cs.enact("move_layer", {"from_index": 1, "to_index": 0})
	assert not r.ok


# =================================================================
# Sandbox composite — drives the real subprocess
# =================================================================

OBJECT_SKILL_SRC = """
from plugins.BaseSkill import BaseSkill
from PIL import ImageDraw


class HalfRedSkill(BaseSkill):
    name = "Half Red"
    description = "Paints the left half of the canvas opaque red on transparent."
    kind = "object"

    def run(self, canvas):
        img = canvas.new_layer()
        draw = ImageDraw.Draw(img, "RGBA")
        draw.rectangle((0, 0, canvas.size // 2, canvas.size), fill=(255, 0, 0, 255))
        canvas.commit(img)
"""


def _skill(src, kind):
	return SimpleNamespace(slug="test_obj", kind=kind, code=src)


def test_object_skill_composites_onto_prior_canvas(tmp_path_factory):
	# pytest's default tmp dir is unwritable in this repo's environment;
	# mirror the workaround used by test_canvas_render.
	target = Path(".canvas_object_test")
	if target.exists():
		shutil.rmtree(target, ignore_errors=True)
	target.mkdir(parents=True, exist_ok=True)
	try:
		# Seed a "prior canvas" image — solid blue.
		prior = target / "prior.png"
		Image.new("RGBA", (32, 32), (0, 0, 255, 255)).save(prior, "PNG")

		out = target / "out.png"
		run_skill(
			_skill(OBJECT_SKILL_SRC, "object"),
			params={},
			palette=get_palette("japandi"),
			size=32,
			seed=1,
			input_image_path=prior,
			output_image_path=out,
			timeout_s=20.0,
		)
		final = Image.open(out).convert("RGBA")
		# Left half: red (the object), right half: blue (the prior canvas).
		assert final.getpixel((4, 16)) == (255, 0, 0, 255)
		assert final.getpixel((28, 16)) == (0, 0, 255, 255)
	finally:
		shutil.rmtree(target, ignore_errors=True)


def test_object_skill_without_prior_canvas_is_refused():
	with pytest.raises(SkillRunError) as ei:
		run_skill(
			_skill(OBJECT_SKILL_SRC, "object"),
			params={},
			palette=get_palette("japandi"),
			size=32,
			seed=1,
			input_image_path=None,
			output_image_path=Path(".unused_obj_out.png"),
			timeout_s=10.0,
		)
	assert ei.value.diagnostic.get("error_type") == "MissingPriorCanvas"


def test_built_in_object_catalog_smokes_through_sandbox():
	target = Path(".canvas_object_catalog_test")
	if target.exists():
		shutil.rmtree(target, ignore_errors=True)
	target.mkdir(parents=True, exist_ok=True)
	try:
		registry = SkillRegistry()
		discover_skills(ROOT_DIR, registry, {})
		records = [s for s in registry.list_records() if s.kind == "object" and s.owner == "library"]
		names = {s.name for s in records}
		assert {
			"Asemic Ribbon", "Boid Swarm", "Circle Pack Medallion", "Differential Vine",
			"L-System Crest", "Phyllotaxis Pod", "Spirograph Orbit", "Voronoi Shard Halo",
			"Particle Constellation", "Wave Ribbon", "3D Menger Sponge", "Border",
		} <= names
		assert len(records) >= 16

		prior = target / "prior.png"
		Image.new("RGBA", (96, 96), (13, 24, 37, 255)).save(prior, "PNG")
		palette = get_palette("japandi")
		for i, skill in enumerate(records):
			assert_valid(skill.code)
			controls = [c for c in skill.controls if c.get("type") != "palette"]
			assert len(controls) <= 4, skill.name
			out = target / f"{skill.slug}.png"
			run_skill(
				skill,
				params=default_controls(skill),
				palette=palette,
				size=96,
				seed=1000 + i,
				input_image_path=prior,
				output_image_path=out,
				timeout_s=20.0,
			)
			final = Image.open(out).convert("RGB")
			assert ImageChops.difference(Image.open(prior).convert("RGB"), final).getbbox(), skill.name
	finally:
		shutil.rmtree(target, ignore_errors=True)


def test_color_field_and_crop_smoke_through_sandbox():
	target = Path(".canvas_basic_skill_test")
	if target.exists():
		shutil.rmtree(target, ignore_errors=True)
	target.mkdir(parents=True, exist_ok=True)
	try:
		registry = SkillRegistry()
		discover_skills(ROOT_DIR, registry, {})
		palette = get_palette("japandi")
		color_field = registry.get_record("color_field")
		crop = registry.get_record("crop")
		assert color_field is not None and crop is not None

		bg = target / "color_field.png"
		run_skill(color_field, params={"mode": "linear", "angle": 20}, palette=palette, size=96, seed=9, input_image_path=None, output_image_path=bg, timeout_s=20.0)
		assert len(set(Image.open(bg).convert("RGB").getdata())) > 1

		prior = target / "prior.png"
		img = Image.new("RGBA", (96, 96), palette.background)
		for box, color in [((0, 0, 48, 48), palette.primary), ((48, 0, 96, 48), palette.secondary), ((0, 48, 48, 96), palette.tertiary), ((48, 48, 96, 96), palette.accent)]:
			img.paste(Image.new("RGBA", (box[2] - box[0], box[3] - box[1]), color), box[:2])
		img.save(prior, "PNG")
		out = target / "crop.png"
		run_skill(crop, params={"zoom": 2.0, "cx": 0.25, "cy": 0.25}, palette=palette, size=96, seed=10, input_image_path=prior, output_image_path=out, timeout_s=20.0)
		final = Image.open(out).convert("RGB")
		assert ImageChops.difference(Image.open(prior).convert("RGB"), final).getbbox()
		assert final.getpixel((48, 48)) == Image.new("RGB", (1, 1), palette.primary).getpixel((0, 0))
	finally:
		shutil.rmtree(target, ignore_errors=True)


def test_border_grows_inward_from_canvas_edge():
	target = Path(".canvas_border_test")
	if target.exists():
		shutil.rmtree(target, ignore_errors=True)
	target.mkdir(parents=True, exist_ok=True)
	try:
		registry = SkillRegistry()
		discover_skills(ROOT_DIR, registry, {})
		palette = get_palette("japandi")
		prior = target / "prior.png"
		Image.new("RGBA", (32, 32), palette.background).save(prior, "PNG")
		out = target / "border.png"
		run_skill(registry.get_record("border"), params={"width": 4, "color": "accent"}, palette=palette, size=32, seed=1, input_image_path=prior, output_image_path=out, timeout_s=20.0)
		final = Image.open(out).convert("RGB")
		accent = Image.new("RGB", (1, 1), palette.accent).getpixel((0, 0))
		background = Image.new("RGB", (1, 1), palette.background).getpixel((0, 0))
		assert final.getpixel((0, 0)) == accent
		assert final.getpixel((3, 16)) == accent
		assert final.getpixel((4, 16)) == background
		assert final.getpixel((31, 31)) == accent
	finally:
		shutil.rmtree(target, ignore_errors=True)
