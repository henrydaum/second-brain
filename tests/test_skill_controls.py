import pytest
import uuid
from pathlib import Path
from types import SimpleNamespace

from plugins.skills.helpers.skill_store import SkillValidationError, source_uses_palette, to_skill_record, validate_controls


def test_palette_use_does_not_auto_add_palette_control():
	controls = validate_controls([], set(), code="def run(canvas):\n    canvas.commit(canvas.new(canvas.palette.primary))")
	assert controls == []


def test_explicit_palette_control_is_preserved():
	controls = validate_controls([{"type": "palette"}], set(), code="")
	assert controls == [{"type": "palette", "name": "palette", "label": "Palette"}]


def test_seed_is_not_a_normal_control_param():
	with pytest.raises(SkillValidationError):
		validate_controls([{"type": "slider", "name": "seed", "min": 0, "max": 1}], set(), code="")


def test_stale_palette_control_is_filtered_from_skill_record():
	d = Path(".test_tmp_skill_controls"); d.mkdir(exist_ok=True)
	path = d / f"skill_blur_{uuid.uuid4().hex}.py"
	try:
		path.write_text("def run(self, canvas):\n    # palette mentioned in a comment only\n    canvas.commit(canvas.image)\n", encoding="utf-8")
		inst = SimpleNamespace(_source_path=path, name="Blur", description="", kind="transform", controls=[{"type": "palette"}])
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
