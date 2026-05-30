"""Tests for the descriptor-style control declaration (Tier 3).

Covers: descriptors compile to the runtime controls list, defaults are
honoured, auto-clamping happens at dispatch time, and Pan-consumed sliders
are suppressed from the UI control list but still tracked for clamping.
"""

import pytest
import tempfile
from pathlib import Path
from PIL import Image

from plugins.BaseTechnique import BaseTechnique, Slider, Bool, Enum, Pan, Text
from plugins.techniques.helpers.technique_sandbox_entry import Canvas, _dispatch_run, _apply_param_bounds


# ---------------------------------------------------------------------------
# Compilation: descriptors to runtime controls
# ---------------------------------------------------------------------------

def test_slider_descriptor_compiles_to_dict_control():
    class S(BaseTechnique):
        name = "S"
        kind = "filter"
        amount = Slider(0.0, 1.0, default=0.5)

    assert len(S.controls) == 1
    c = S.controls[0]
    assert c["type"] == "slider"
    assert c["name"] == "amount"
    assert c["min"] == 0.0
    assert c["max"] == 1.0
    assert c["default"] == 0.5
    assert "step" in c


def test_bool_and_enum_descriptors_compile():
    class S(BaseTechnique):
        name = "S"
        kind = "filter"
        flag = Bool(default=True)
        mode = Enum(["radial", "uniform"], default="uniform")

    by_type = {c["type"]: c for c in S.controls}
    assert by_type["bool"]["default"] is True
    assert by_type["enum"]["default"] == "uniform"
    assert {opt["value"] for opt in by_type["enum"]["options"]} == {"radial", "uniform"}


def test_pan_absorbs_underlying_sliders_from_ui():
    class S(BaseTechnique):
        name = "S"
        kind = "filter"
        strength = Slider(-1.0, 1.0, default=0.6)
        cx = Slider(0, 1, default=0.5)
        cy = Slider(0, 1, default=0.5)
        center = Pan(x="cx", y="cy")

    types = [c["type"] for c in S.controls]
    # cx and cy are NOT emitted as separate slider entries.
    assert types == ["slider", "pan"]
    pan = [c for c in S.controls if c["type"] == "pan"][0]
    assert pan["x_param"] == "cx"
    assert pan["y_param"] == "cy"
    assert pan["x_default"] == 0.5
    assert pan["y_default"] == 0.5
    # Bounds for cx/cy are still tracked for dispatch.
    assert "cx" in S._param_bounds
    assert "cy" in S._param_bounds
    assert S._param_bounds["cx"]["min"] == 0.0


def test_pan_with_missing_slider_raises():
    with pytest.raises(TypeError, match="Pan"):
        class S(BaseTechnique):
            name = "S"
            center = Pan(x="cx", y="cy")  # neither cx nor cy declared


def test_literal_controls_rejected():
    with pytest.raises(TypeError, match="literal controls"):
        class S(BaseTechnique):
            name = "S"
            kind = "filter"
            controls = []


def test_kwargs_run_signature_rejected_by_base_class():
    with pytest.raises(TypeError, match="exactly"):
        class S(BaseTechnique):
            name = "S"
            kind = "filter"

            def run(self, canvas, **params):
                canvas.commit(canvas.image)


def test_relaxed_owner_default():
    class S(BaseTechnique):
        name = "S"

    assert S.owner == "library"


# ---------------------------------------------------------------------------
# Dispatch: setattr + auto-clamp
# ---------------------------------------------------------------------------

def _make_canvas() -> Canvas:
    img = Image.new("RGBA", (16, 16), "#aabbcc")
    tf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tf.name)
    tf.close()
    job = {
        "palette": {
            "colors": {"background": "#000", "primary": "#fff",
                       "secondary": "#888", "tertiary": "#444", "accent": "#f00"},
            "id": "p", "name": "p", "kind": "k",
        },
        "size": 16,
        "seed": 7,
        "input_image_path": tf.name,
    }
    return Canvas(job)


def test_dispatch_sets_attributes_and_clamps():
    class S(BaseTechnique):
        name = "S"
        kind = "filter"
        gamma = Slider(0.2, 3.0, default=1.0)

        def run(self, canvas):
            canvas.commit(canvas.image)

    s = S()
    _apply_param_bounds(s, {"gamma": 99})  # way over max
    assert s.gamma == 3.0
    s2 = S()
    _apply_param_bounds(s2, {"gamma": -5})
    assert s2.gamma == 0.2
    s3 = S()
    _apply_param_bounds(s3, {})  # nothing passed
    assert s3.gamma == 1.0


def test_dispatch_calls_run_without_kwargs_when_signature_is_clean():
    captured = {}

    class S(BaseTechnique):
        name = "S"
        kind = "filter"
        gamma = Slider(0.2, 3.0, default=1.0)

        def run(self, canvas):
            captured["gamma"] = self.gamma
            canvas.commit(canvas.image)

    _dispatch_run(S(), _make_canvas(), {"gamma": 0.5})
    assert captured["gamma"] == 0.5


def test_dispatch_rejects_unknown_params():
    class S(BaseTechnique):
        name = "S"
        kind = "filter"

        def run(self, canvas):
            canvas.commit(canvas.image)

    with pytest.raises(ValueError, match="unknown control"):
        _dispatch_run(S(), _make_canvas(), {"x": 0.8})


def test_enum_dispatch_snaps_to_allowed():
    class S(BaseTechnique):
        name = "S"
        kind = "filter"
        mode = Enum(["radial", "uniform"], default="radial")

        def run(self, canvas):
            canvas.commit(canvas.image)

    s = S()
    _apply_param_bounds(s, {"mode": "bogus"})
    assert s.mode == "radial"
    s2 = S()
    _apply_param_bounds(s2, {"mode": "uniform"})
    assert s2.mode == "uniform"


def test_pan_underlying_sliders_clamped_at_dispatch():
    class S(BaseTechnique):
        name = "S"
        kind = "filter"
        cx = Slider(0, 1, default=0.5)
        cy = Slider(0, 1, default=0.5)
        center = Pan(x="cx", y="cy")

        def run(self, canvas):
            canvas.commit(canvas.image)

    s = S()
    _apply_param_bounds(s, {"cx": 1.5, "cy": -0.3})
    assert s.cx == 1.0
    assert s.cy == 0.0


# ---------------------------------------------------------------------------
# Text descriptor
# ---------------------------------------------------------------------------

def test_text_descriptor_compiles_to_dict_control():
    class S(BaseTechnique):
        name = "S"
        kind = "background"
        phrase = Text(default="hi", max_length=50, placeholder="say something")

    c = [x for x in S.controls if x["type"] == "text"][0]
    assert c["name"] == "phrase"
    assert c["default"] == "hi"
    assert c["max_length"] == 50
    assert c["placeholder"] == "say something"
    assert S._param_bounds["phrase"] == {"type": "text", "default": "hi", "max_length": 50}


def test_text_dispatch_clamps_and_coerces():
    class S(BaseTechnique):
        name = "S"
        kind = "background"
        phrase = Text(default="d", max_length=5)

        def run(self, canvas):
            canvas.commit(canvas.image)

    s = S()
    _apply_param_bounds(s, {"phrase": "abcdefghij"})
    assert s.phrase == "abcde"
    s2 = S()
    _apply_param_bounds(s2, {"phrase": None})
    assert s2.phrase == ""
    s3 = S()
    _apply_param_bounds(s3, {})
    assert s3.phrase == "d"
    s4 = S()
    _apply_param_bounds(s4, {"phrase": 42})  # non-string coerced
    assert s4.phrase == "42"
