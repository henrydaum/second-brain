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
from PIL import Image

from canvas.state import CanvasState
from plugins.helpers.palettes import get_palette
from plugins.skills.helpers.skill_runner import SkillRunError, run_skill
from plugins.skills.helpers.skill_store import _KINDS


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
