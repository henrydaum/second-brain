import pytest
import uuid
from pathlib import Path
from types import SimpleNamespace

from plugins.skills.helpers.skill_store import SkillValidationError, source_uses_palette, to_skill_record, validate_controls
from plugins.skills.helpers.skill_controls import coerce_control_value


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
		inst = SimpleNamespace(_source_path=path, name="Blur", description="", kind="effect", controls=[{"type": "palette"}])
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


def test_text_control_validates_and_normalizes_defaults():
	out = validate_controls(
		[{"type": "text", "name": "phrase"}],
		{"phrase"},
		code="def run(self, canvas, phrase=''):\n    canvas.commit(canvas.new())\n",
	)
	assert out[0]["type"] == "text"
	assert out[0]["default"] == ""
	assert out[0]["max_length"] == 120
	assert out[0]["placeholder"] is None
	assert out[0]["label"] == "Phrase"


def test_text_control_honours_explicit_max_length_and_placeholder():
	out = validate_controls(
		[{"type": "text", "name": "phrase", "default": "hi", "max_length": 8, "placeholder": "p"}],
		{"phrase"},
		code="def run(self, canvas, phrase=''):\n    canvas.commit(canvas.new())\n",
	)
	assert out[0]["max_length"] == 8
	assert out[0]["placeholder"] == "p"
	assert out[0]["default"] == "hi"


def test_coerce_text_truncates_and_handles_none():
	spec = {"type": "text", "max_length": 5}
	assert coerce_control_value(spec, "abcdefgh") == "abcde"
	assert coerce_control_value(spec, None) == ""
	assert coerce_control_value(spec, 42) == "42"
