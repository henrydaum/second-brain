import pytest
import uuid
from pathlib import Path
from types import SimpleNamespace

from plugins.skills.helpers import skill_store
from plugins.skills.helpers.skill_store import SkillValidationError, assert_valid, source_uses_palette, to_skill_record
from plugins.skills.helpers.skill_controls import coerce_control_value


def test_module_level_run_is_rejected():
	with pytest.raises(SkillValidationError, match="BaseSkill"):
		assert_valid("def run(canvas):\n    canvas.commit(canvas.new())\n")


def test_kwargs_run_signature_is_rejected():
	src = (
		"from plugins.BaseSkill import BaseSkill\n\n"
		"class BadSkill(BaseSkill):\n"
		"    name = 'Bad'\n"
		"    def run(self, canvas, **params):\n"
		"        canvas.commit(canvas.new())\n"
	)
	with pytest.raises(SkillValidationError, match="exactly"):
		assert_valid(src)


def test_literal_controls_are_rejected_by_validation():
	src = (
		"from plugins.BaseSkill import BaseSkill\n\n"
		"class BadSkill(BaseSkill):\n"
		"    name = 'Bad'\n"
		"    controls = []\n"
		"    def run(self, canvas):\n"
		"        canvas.commit(canvas.new())\n"
	)
	with pytest.raises(SkillValidationError, match="literal controls"):
		assert_valid(src)


def test_palette_choices_points_to_enum_descriptor():
	src = (
		"from plugins.BaseSkill import BaseSkill, Palette\n\n"
		"class BadSkill(BaseSkill):\n"
		"    name = 'Bad'\n"
		"    palette_slot = Palette('Slot', choices=['background'], default='background')\n"
		"    def run(self, canvas):\n"
		"        canvas.commit(canvas.new())\n"
	)
	with pytest.raises(SkillValidationError, match="use Enum"):
		assert_valid(src)


def test_get_controls_is_rejected():
	src = (
		"from plugins.BaseSkill import BaseSkill, Enum\n\n"
		"class BadSkill(BaseSkill):\n"
		"    name = 'Bad'\n"
		"    @classmethod\n"
		"    def get_controls(cls):\n"
		"        return {'slot': Enum(['primary'], default='primary')}\n"
		"    def run(self, canvas):\n"
		"        canvas.commit(canvas.new())\n"
	)
	with pytest.raises(SkillValidationError, match="get_controls"):
		assert_valid(src)


def test_stale_palette_control_is_filtered_from_skill_record():
	d = Path(".test_tmp_skill_controls"); d.mkdir(exist_ok=True)
	path = d / f"skill_blur_{uuid.uuid4().hex}.py"
	try:
		path.write_text("def run(self, canvas):\n    # palette mentioned in a comment only\n    canvas.commit(canvas.image)\n", encoding="utf-8")
		inst = SimpleNamespace(_source_path=path, name="Blur", description="", kind="filter", controls=[{"type": "palette"}])
		assert source_uses_palette(path.read_text(encoding="utf-8")) is False
		assert to_skill_record(inst).controls == []
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


def test_write_skill_rejects_module_level_run(monkeypatch):
	d = Path(".test_tmp_write_skill_reject"); d.mkdir(exist_ok=True)
	try:
		monkeypatch.setattr(skill_store, "SANDBOX_SKILLS", d)
		with pytest.raises(SkillValidationError, match="BaseSkill"):
			skill_store.write_skill(
				name="No Class", description="", kind="background",
				owner="u", code="def run(canvas):\n    canvas.commit(canvas.new())\n",
			)
	finally:
		for p in d.glob("*"):
			p.unlink()
		d.rmdir()


def test_write_skill_accepts_full_class_source(monkeypatch):
	d = Path(".test_tmp_write_skill_accept"); d.mkdir(exist_ok=True)
	src = (
		"from plugins.BaseSkill import BaseSkill, Slider\n\n"
		"class FreshSkill(BaseSkill):\n"
		"    name = 'Draft'\n"
		"    description = 'Draft'\n"
		"    kind = 'background'\n"
		"    strength = Slider(0, 1, default=0.5)\n"
		"    def run(self, canvas):\n"
		"        canvas.commit(canvas.new())\n"
	)
	try:
		monkeypatch.setattr(skill_store, "SANDBOX_SKILLS", d)
		skill, path = skill_store.write_skill(
			name="Fresh", description="Fresh desc", kind="background",
			owner="owner1", code=src,
		)
		out = path.read_text(encoding="utf-8")
		assert skill.slug == "fresh"
		assert "owner = 'owner1'" in out
		assert "strength = Slider" in out
		assert "def run(self, canvas):" in out
	finally:
		for p in d.glob("*"):
			p.unlink()
		d.rmdir()
