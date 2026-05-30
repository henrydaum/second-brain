import pytest
import uuid
from pathlib import Path
from types import SimpleNamespace

from plugins.techniques.helpers import technique_store
from plugins.techniques.helpers.technique_store import TechniqueValidationError, assert_valid, source_uses_palette, to_technique_record
from plugins.techniques.helpers.technique_controls import coerce_control_value


def test_module_level_run_is_rejected():
	with pytest.raises(TechniqueValidationError, match="BaseTechnique"):
		assert_valid("def run(canvas):\n    canvas.commit(canvas.new())\n")


def test_kwargs_run_signature_is_rejected():
	src = (
		"from plugins.BaseTechnique import BaseTechnique\n\n"
		"class BadTechnique(BaseTechnique):\n"
		"    name = 'Bad'\n"
		"    def run(self, canvas, **params):\n"
		"        canvas.commit(canvas.new())\n"
	)
	with pytest.raises(TechniqueValidationError, match="exactly"):
		assert_valid(src)


def test_literal_controls_are_rejected_by_validation():
	src = (
		"from plugins.BaseTechnique import BaseTechnique\n\n"
		"class BadTechnique(BaseTechnique):\n"
		"    name = 'Bad'\n"
		"    controls = []\n"
		"    def run(self, canvas):\n"
		"        canvas.commit(canvas.new())\n"
	)
	with pytest.raises(TechniqueValidationError, match="literal controls"):
		assert_valid(src)


def test_palette_choices_points_to_enum_descriptor():
	src = (
		"from plugins.BaseTechnique import BaseTechnique, Palette\n\n"
		"class BadTechnique(BaseTechnique):\n"
		"    name = 'Bad'\n"
		"    palette_slot = Palette('Slot', choices=['background'], default='background')\n"
		"    def run(self, canvas):\n"
		"        canvas.commit(canvas.new())\n"
	)
	with pytest.raises(TechniqueValidationError, match="use Enum"):
		assert_valid(src)


def test_get_controls_is_rejected():
	src = (
		"from plugins.BaseTechnique import BaseTechnique, Enum\n\n"
		"class BadTechnique(BaseTechnique):\n"
		"    name = 'Bad'\n"
		"    @classmethod\n"
		"    def get_controls(cls):\n"
		"        return {'slot': Enum(['primary'], default='primary')}\n"
		"    def run(self, canvas):\n"
		"        canvas.commit(canvas.new())\n"
	)
	with pytest.raises(TechniqueValidationError, match="get_controls"):
		assert_valid(src)


def test_stale_palette_control_is_filtered_from_technique_record():
	d = Path(".test_tmp_technique_controls"); d.mkdir(exist_ok=True)
	path = d / f"technique_blur_{uuid.uuid4().hex}.py"
	try:
		path.write_text("def run(self, canvas):\n    # palette mentioned in a comment only\n    canvas.commit(canvas.image)\n", encoding="utf-8")
		inst = SimpleNamespace(_source_path=path, name="Blur", description="", kind="filter", controls=[{"type": "palette"}])
		assert source_uses_palette(path.read_text(encoding="utf-8")) is False
		assert to_technique_record(inst).controls == []
	finally:
		if path.exists():
			path.unlink()
		try:
			d.rmdir()
		except OSError:
			pass


def test_palette_control_is_kept_when_source_uses_palette():
	assert source_uses_palette("def run(self, canvas):\n    canvas.new()\n") is True


def test_coerce_text_truncates_and_handles_none():
	spec = {"type": "text", "max_length": 5}
	assert coerce_control_value(spec, "abcdefgh") == "abcde"
	assert coerce_control_value(spec, None) == ""
	assert coerce_control_value(spec, 42) == "42"


def test_write_technique_rejects_module_level_run(monkeypatch):
	d = Path(".test_tmp_write_technique_reject"); d.mkdir(exist_ok=True)
	try:
		monkeypatch.setattr(technique_store, "SANDBOX_TECHNIQUES", d)
		with pytest.raises(TechniqueValidationError, match="BaseTechnique"):
			technique_store.write_technique(
				name="No Class", description="", kind="background",
				owner="u", code="def run(canvas):\n    canvas.commit(canvas.new())\n",
			)
	finally:
		for p in d.glob("*"):
			p.unlink()
		d.rmdir()


def test_write_technique_accepts_full_class_source(monkeypatch):
	d = Path(".test_tmp_write_technique_accept"); d.mkdir(exist_ok=True)
	src = (
		"from plugins.BaseTechnique import BaseTechnique, Slider\n\n"
		"class FreshTechnique(BaseTechnique):\n"
		"    name = 'Draft'\n"
		"    description = 'Draft'\n"
		"    kind = 'background'\n"
		"    strength = Slider(0, 1, default=0.5)\n"
		"    def run(self, canvas):\n"
		"        canvas.commit(canvas.new())\n"
	)
	try:
		monkeypatch.setattr(technique_store, "SANDBOX_TECHNIQUES", d)
		technique, path = technique_store.write_technique(
			name="Fresh", description="Fresh desc", kind="background",
			owner="owner1", code=src,
		)
		out = path.read_text(encoding="utf-8")
		assert technique.slug == "fresh"
		assert "owner = 'owner1'" in out
		assert "strength = Slider" in out
		assert "def run(self, canvas):" in out
	finally:
		for p in d.glob("*"):
			p.unlink()
		d.rmdir()


def test_technique_control_cap_is_four_non_palette_controls():
	ok_src = (
		"from plugins.BaseTechnique import BaseTechnique, Bool, Enum, Slider\n\n"
		"class FourControlTechnique(BaseTechnique):\n"
		"    name = 'Four'\n"
		"    description = 'Four controls'\n"
		"    kind = 'background'\n"
		"    a = Slider(0, 1, default=0.5)\n"
		"    b = Slider(0, 1, default=0.5)\n"
		"    c = Bool(default=False)\n"
		"    d = Enum(['x', 'y'], default='x')\n"
		"    def run(self, canvas):\n"
		"        canvas.commit(canvas.new())\n"
	)
	technique_store.assert_valid(ok_src)
	with pytest.raises(TechniqueValidationError, match="cap is 4"):
		technique_store.assert_valid(ok_src.replace(
			"    def run(self, canvas):\n",
			"    e = Slider(0, 1, default=0.5)\n"
			"    def run(self, canvas):\n",
		))
