from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from plugins.helpers.palettes import get_palette
from plugins.skills.helpers.art_kit import build_namespace
from plugins.skills.helpers.skill_runner import run_skill


def _palette():
	return SimpleNamespace(
		background="#101820",
		tertiary="#31525b",
		secondary="#6f9ceb",
		primary="#f2aa4c",
		accent="#f7f7f7",
	)


def test_cube_mesh_renders_into_pil_image():
	art = build_namespace(_palette())
	img = Image.new("RGBA", (128, 128), _palette().background)
	cube = art.cube_mesh(size=1.4, color=_palette().primary)

	out = art.render_3d(img, cube, camera=(2.4, 1.8, 3.0), target=(0, 0, 0), outline=_palette().background)

	assert out is img
	assert len(cube.vertices) == 8
	assert len(cube.faces) == 6
	assert len(set(img.getdata())) > 1


def test_render_3d_uses_palette_ramp_when_mesh_has_no_color():
	art = build_namespace(_palette())
	img = Image.new("RGBA", (96, 96), _palette().background)

	art.render_3d(img, art.cube_mesh(), camera=(2.5, 2.0, 3.0))

	assert len(set(img.getdata())) > 1


def test_render_3d_runs_inside_skill_sandbox():
	src = """
from plugins.BaseSkill import BaseSkill


class SandboxCubeSkill(BaseSkill):
    name = "Sandbox Cube"
    kind = "background"

    def run(self, canvas):
        img = canvas.create_image()
        cube = art_kit.cube_mesh(size=1.4, color=canvas.palette.primary)
        art_kit.render_3d(img, cube, camera=(2.4, 1.8, 3.0), outline=canvas.palette.background)
        canvas.commit(img)
"""
	out = Path(".canvas_3d_sandbox_test.png")
	try:
		run_skill(
			SimpleNamespace(slug="sandbox_cube", kind="background", code=src),
			params={}, palette=get_palette("japandi"), size=96, seed=1,
			input_image_path=None, output_image_path=out, timeout_s=20.0,
		)
		assert len(set(Image.open(out).convert("RGBA").getdata())) > 1
	finally:
		out.unlink(missing_ok=True)
