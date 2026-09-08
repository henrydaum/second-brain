"""Shipped techniques, their agent workflow, and actual nested sandbox execution."""
import json
from pathlib import Path
import types

import pytest

from tests.test_store_image_editing import rig, load


NAMES = "load_image crop resize rotate flip brightness contrast saturation exposure gamma blur sharpen grayscale invert solid gradient line shape text duotone vignette".split()


@pytest.fixture
def photo(rig, tmp_path):
    from PIL import Image
    image = Image.new("RGBA", (12, 8))
    for y in range(8):
        for x in range(12):
            image.putpixel((x, y), (x * 20, y * 30, 80, 0 if x == 0 else 128 if x == 1 else 255))
    path = str(tmp_path / "photo.png")
    rig.kit.write_png(rig.sdk, path, image)
    return path


def catalog(rig):
    return load("image_test.canvas_catalog", rig.scripts / "canvas_catalog.py")


@pytest.mark.parametrize("name", NAMES)
def test_every_shipped_technique_renders_and_caches(rig, photo, name):
    script = "technique_" + name
    controls = dict(catalog(rig).discover(rig.sdk)["techniques"][script]["example"])
    if name == "load_image": controls["path"] = photo
    if name in ("crop", "shape"): controls.update(left=2, top=1, right=10, bottom=7)
    if name == "resize": controls.update(width=6, height=4)
    if name == "line": controls.update(points=[[1, 1], [10, 6]], width=1)
    if name == "text": controls.update(content="Hi\nX", x=1, y=0, size=6)
    spec = catalog(rig).prepare(rig.sdk, script, controls)
    cid = rig.service.create(rig.sdk, width=12, height=8)
    if spec["kind"] != "background":
        rig.service.add_layer(rig.sdk, cid, "technique_load_image", "background", {"path": photo})
    rig.service.add_layer(rig.sdk, cid, script, spec["kind"], controls)
    result = rig.renderer.main(rig.sdk, cid, seed=0)
    image = rig.kit.read_image(rig.sdk, result["path"])
    assert image.mode == "RGBA"
    assert image.width == result["width"] and image.height == result["height"]
    assert image.getchannel("A").getextrema()[1] > 0
    assert rig.renderer.main(rig.sdk, cid)["cache_hit"]


def test_crop_rotate_resize_composes_and_resumes_at_new_dimensions(rig, photo):
    from PIL import Image
    s, sdk = rig.service, rig.sdk
    cid = s.create(sdk)
    s.add_layer(sdk, cid, "technique_load_image", "background", {"path": photo})
    s.add_layer(sdk, cid, "technique_crop", "filter", {"left": 2, "top": 1, "right": 10, "bottom": 7})
    s.add_layer(sdk, cid, "technique_rotate", "filter", {"angle": 90})
    first = rig.renderer.main(sdk, cid)
    expected = rig.kit.read_image(sdk, photo).crop((2, 1, 10, 7)).transpose(Image.Transpose.ROTATE_90)
    assert rig.kit.read_image(sdk, first["path"]).tobytes() == expected.tobytes()
    assert (first["width"], first["height"]) == (6, 8)
    s.add_layer(sdk, cid, "technique_resize", "filter", {"width": 3, "height": 4})
    second = rig.renderer.main(sdk, cid)
    assert second["cached_layers"] == 3
    assert (second["width"], second["height"]) == (3, 4)
    s.undo(sdk, cid)
    assert rig.renderer.main(sdk, cid)["path"] == first["path"]


@pytest.mark.parametrize("script,controls", [
    ("technique_gamma", {"gamma": 0}), ("technique_saturation", {"factor": True}),
    ("technique_blur", {"radius": float("nan")}), ("technique_blur", {"strength": 5}),
    ("technique_crop", {"right": 0, "bottom": 4}), ("technique_resize", {"width": 1.5, "height": 3}),
    ("technique_line", {"points": [[0, 1, 2], [3, 4]]}), ("technique_load_image", {"path": ""}),
    ("technique_sharpen", {"threshold": 256}),
])
def test_invalid_controls_are_rejected(rig, script, controls):
    with pytest.raises(ValueError): catalog(rig).prepare(rig.sdk, script, controls)


def test_catalog_discovery_and_fine_adjustment_workflow(rig, photo):
    sdk, s = rig.sdk, rig.service
    add = load("image_add_tool", rig.root / "tools/tool_add_layer.py").AddLayer()
    manage = load("image_manage_tool", rig.root / "tools/tool_manage_layers.py").ManageLayers()
    search = load("image_search_tool", rig.root / "tools/tool_search_techniques.py").SearchTechniques()
    assert any(row["script"] == "technique_sharpen" for row in search.run(sdk, query="sharpness"))
    assert len(search.run(sdk)) == len(NAMES)
    assert search.run(sdk, script="technique_blur")["controls"]["radius"]["step"] == .25
    added = add.run(sdk, script="technique_load_image", controls={"path": photo})
    cid = added["data"]["canvas_id"]
    assert added["data"]["layers"][0]["controls"]["fit"] == "native"
    add.run(sdk, script="technique_saturation", controls={"factor": 1.1})
    layer_id = s.get_state(sdk, cid)["layers"][1]["id"]
    inspected = manage.run(sdk, "controls", layer_id=layer_id)
    assert inspected["current"] == {"factor": 1.1}
    assert inspected["technique"]["controls"]["factor"]["minimum"] == 0
    rig.renderer.main(sdk, cid)
    changed = manage.run(sdk, "set_control", layer_id=layer_id, name="factor", value=1.125)
    assert changed["ok"]
    assert len(s.get_state(sdk, cid)["layers"]) == 2
    assert s.get_state(sdk, cid)["layers"][1]["controls"]["factor"] == 1.125
    assert rig.renderer.main(sdk, cid)["cached_layers"] == 1
    before = s.get_state(sdk, cid)
    failed = manage.run(sdk, "set_control", layer_id=layer_id, name="facotr", value=2)
    assert not failed["ok"]
    assert s.get_state(sdk, cid) == before
    manage.run(sdk, "undo")
    assert s.get_state(sdk, cid)["layers"][1]["controls"]["factor"] == 1.1
    manage.run(sdk, "redo")
    assert s.get_state(sdk, cid)["layers"][1]["controls"]["factor"] == 1.125


def test_palette_is_opt_in_live_and_undoable(rig, photo):
    s, sdk = rig.service, rig.sdk
    cid = s.create(sdk)
    s.add_layer(sdk, cid, "technique_load_image", "background", {"path": photo})
    original = rig.renderer.main(sdk, cid)
    s.set_palette(sdk, cid, colors={"accent": "#ff0000", "primary": "#00ff00"})
    after = rig.renderer.main(sdk, cid)
    assert rig.kit.read_image(sdk, original["path"]).tobytes() == rig.kit.read_image(sdk, after["path"]).tobytes()
    s.add_layer(sdk, cid, "technique_solid", "object", {"color": "@accent"})
    red = rig.renderer.main(sdk, cid)
    assert rig.kit.read_image(sdk, red["path"]).getpixel((3, 3)) == (255, 0, 0, 255)
    s.set_palette(sdk, cid, colors={"accent": "#0000ff"})
    blue = rig.renderer.main(sdk, cid)
    assert rig.kit.read_image(sdk, blue["path"]).getpixel((3, 3)) == (0, 0, 255, 255)
    s.undo(sdk, cid)
    assert rig.renderer.main(sdk, cid)["path"] == red["path"]
    s.stop(sdk)
    s.start(sdk)
    assert s.get_state(sdk, cid)["palette_colors"]["accent"] == "#ff0000"


def test_source_files_invalidate_without_manual_dependencies(rig, photo):
    from PIL import Image
    cid = rig.service.create(rig.sdk)
    rig.service.add_layer(rig.sdk, cid, "technique_load_image", "background", {"path": photo})
    first = rig.renderer.main(rig.sdk, cid)
    rig.kit.write_png(rig.sdk, photo, Image.new("RGBA", (3, 2), "green"))
    second = rig.renderer.main(rig.sdk, cid)
    assert second["path"] != first["path"]
    assert not second["cache_hit"]
    assert (second["width"], second["height"]) == (3, 2)


@pytest.mark.parametrize("name,controls", [
    ("brightness", {"factor": 1}), ("contrast", {"factor": 1}), ("saturation", {"factor": 1}),
    ("exposure", {"stops": 0}), ("gamma", {"gamma": 1}), ("blur", {"radius": 0}),
    ("sharpen", {"amount": 0}), ("duotone", {"amount": 0}), ("vignette", {"amount": 0}),
])
def test_neutral_adjustments_preserve_pixels(rig, photo, name, controls):
    image = rig.kit.read_image(rig.sdk, photo)
    pixels = load("image_test.technique_" + name, rig.scripts / ("technique_" + name + ".py"))
    effective = catalog(rig).prepare(rig.sdk, "technique_" + name, controls)["controls"]
    result = pixels.apply(rig.sdk, image, effective, {"colors": {"secondary": "#000000", "accent": "#ffffff"}})
    assert result.tobytes() == image.tobytes()


@pytest.mark.parametrize("radius", [.5, 2, 3, 15])
def test_blur_does_not_bleed_hidden_rgb(rig, radius):
    from PIL import Image
    image = Image.new("RGBA", (21, 9), (0, 0, 255, 0))
    for y in range(9): image.putpixel((10, y), (255, 0, 0, 128))
    pixels = load("image_test.technique_blur", rig.scripts / "technique_blur.py")
    result = pixels.apply(rig.sdk, image, {"radius": radius}, {})
    visible = [result.getpixel((x, 4)) for x in range(21) if result.getpixel((x, 4))[3] > 0]
    assert visible
    assert all(pixel[:3] == (255, 0, 0) for pixel in visible)


def test_exif_native_import(rig, tmp_path):
    from PIL import Image
    image = Image.new("RGB", (7, 3), "red")
    exif = Image.Exif()
    exif[274] = 6
    path = str(tmp_path / "oriented.jpg")
    image.save(path, exif=exif)
    cid = rig.service.create(rig.sdk)
    rig.service.add_layer(rig.sdk, cid, "technique_load_image", "background", {"path": path})
    result = rig.renderer.main(rig.sdk, cid)
    assert (result["width"], result["height"]) == (3, 7)


def test_entire_shipped_recipe_runs_in_sandbox(rig, photo, tmp_path, monkeypatch):
    from sandbox import Sandbox, Chain, bridge
    from tests.support import retarget_trees
    roots = retarget_trees(monkeypatch, tmp_path)
    s, sdk = rig.service, rig.sdk
    cid = s.create(sdk)
    steps = [("load_image", "background", {"path": photo}),
             ("crop", "filter", {"left": 2, "top": 1, "right": 10, "bottom": 7}),
             ("rotate", "filter", {"angle": 90}),
             ("saturation", "filter", {"factor": 1.15}),
             ("blur", "filter", {"radius": .5}),
             ("sharpen", "filter", {"amount": 20}),
             ("text", "object", {"content": "X", "size": 5, "color": "@accent"})]
    for name, kind, controls in steps: s.add_layer(sdk, cid, "technique_" + name, kind, controls)
    # A newly authored technique participates in the real nested sandbox recipe.
    workspace = Path(roots["workspace"]) / "scripts"
    workspace.mkdir(parents=True, exist_ok=True)
    authored = workspace / "technique_custom_brightness.py"
    authored.write_text((rig.scripts / "canvas_technique_template.py").read_text())
    s.add_layer(sdk, cid, authored.stem, "filter", {"factor": 0})
    adapter = types.SimpleNamespace(exports=s.exports)
    for name in s.exports:
        setattr(adapter, name, lambda *a, _name=name, **k: getattr(s, _name)(sdk, *a, **k))
    sb = Sandbox(context=types.SimpleNamespace(services={"canvas": adapter}), approve=lambda *a, **k: True)
    previous = bridge._SANDBOX
    bridge.configure(sb)
    sb.plugin_roots = list(roots.values())
    try:
        tool_dir = rig.scripts.parent / "tools"
        tool_dir.mkdir(exist_ok=True)
        tool = tool_dir / "tool_add_layer.py"
        tool.write_text((rig.root / "tools/tool_add_layer.py").read_text())
        for radius in (3.5, "25"):
            added = sb.run(str(tool), "AddLayer",
                           kwargs={"canvas_id": cid, "script": "canvas_blur",
                                   "controls": {"radius": radius}}, chain=Chain(root="user"))
            assert added.ok, added.error
            stored = s.get_state(sdk, cid)["layers"][-1]["controls"]["radius"]
            assert type(stored) in (int, float) and stored == float(radius)
        result = sb.run(str(rig.scripts / "canvas_render.py"), "main",
                        kwargs={"canvas_id": cid, "seed": 0}, chain=Chain(root="user"))
        assert result.ok, result.error
        assert (result.data["width"], result.data["height"]) == (6, 8)
        rendered = rig.kit.read_image(sdk, result.data["path"])
        assert rendered.getpixel((3, 3))[:3] == (0, 0, 0)
    finally:
        bridge.configure(previous)
        sb.shutdown()


def test_manifest_ships_every_technique_and_dependency(rig):
    from sandbox.validator import validate_file
    manifest = json.loads((rig.root / "bundles/bundle_image_editing.json").read_text())
    files = set(manifest["files"])
    assert "tools/tool_search_techniques.py" in files
    for name in NAMES: assert f"scripts/technique_{name}.py" in files
    for path in files:
        report = validate_file(rig.root / path)
        assert report.ok, report.render()
        assert set(report.declarations.get("dependencies_files", [])) <= files


def test_discovery_reads_live_metadata_without_executing_code(rig):
    sdk = rig.sdk
    directory = Path(sdk.paths.get("workspace")) / "scripts"
    directory.mkdir(parents=True)
    source = (rig.scripts / "canvas_technique_template.py").read_text()
    path = directory / "technique_custom.py"
    path.write_text(source + "\nraise RuntimeError('must not run during discovery')\n")
    (directory / "technique_directory.py").mkdir()
    (directory / "ordinary_script.py").write_text(source)
    cat = catalog(rig)
    spec = cat.main(sdk, script=path.stem)
    assert spec["origin"] == "workspace" and spec["source_path"] == str(path)
    assert spec["controls"]["factor"]["default"] == 1
    assert len(cat.main(sdk)) == len(NAMES) + 1
    path.write_text(source.replace("'default': 1", "'default': 2"))
    assert cat.prepare(sdk, path.stem)["controls"]["factor"] == 2
    path.unlink()
    with pytest.raises(ValueError, match="unknown technique"):
        cat.main(sdk, script=path.stem)


def test_invalid_override_is_reported_and_cannot_fall_back(rig):
    sdk = rig.sdk
    directory = Path(sdk.paths.get("workspace")) / "scripts"
    directory.mkdir(parents=True)
    path = directory / "technique_blur.py"
    path.write_text("TECHNIQUE = dict(title='not literal')")
    cat = catalog(rig)
    rows = cat.main(sdk)
    failure = next(row for row in rows if row["script"] == path.stem)
    assert failure["error"] and failure["origin"] == "workspace"
    with pytest.raises(ValueError): cat.prepare(sdk, path.stem, {"radius": 2})
    path.unlink()
    assert cat.main(sdk, script=path.stem)["origin"] == "installed"


@pytest.mark.parametrize("replace,with_text", [
    ("'minimum': 0", "'minimum': 5"),
    ("def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):", "def main(sdk):"),
    ("TECHNIQUE =", "NOT_A_TECHNIQUE ="),
])
def test_bad_author_declarations_have_actionable_diagnostics(rig, replace, with_text):
    path = rig.scripts / "technique_broken.py"
    path.write_text((rig.scripts / "canvas_technique_template.py").read_text().replace(replace, with_text))
    errors = catalog(rig).discover(rig.sdk)["errors"]
    assert errors[path.stem]["source_path"] == str(path)
    assert errors[path.stem]["error"]


def test_legacy_recipe_names_resolve_to_individual_files(rig, photo):
    cat = catalog(rig)
    assert cat.prepare(rig.sdk, "canvas_blur")["script"] == "technique_blur"
    cid = rig.service.create(rig.sdk)
    rig.service.add_layer(rig.sdk, cid, "canvas_load_image", "background", {"path": photo})
    rig.service.add_layer(rig.sdk, cid, "canvas_brightness", "filter", {"factor": 1})
    result = rig.renderer.main(rig.sdk, cid)
    assert rig.kit.read_image(rig.sdk, result["path"]).tobytes() == rig.kit.read_image(rig.sdk, photo).tobytes()
    assert "technique_brightness.py" in rig.calls
    duplicate = rig.scripts / "technique_duplicate.py"
    duplicate.write_text((rig.scripts / "technique_blur.py").read_text())
    with pytest.raises(ValueError, match="ambiguous"):
        cat.prepare(rig.sdk, "canvas_blur")


def test_authoring_guide_returns_valid_template(rig):
    search = load("image_search_tool", rig.root / "tools/tool_search_techniques.py").SearchTechniques()
    guide = search.run(rig.sdk, guide=True)
    path = rig.scripts / "technique_from_template.py"
    path.write_text(guide["template"])
    assert guide["workflow"]
    assert search.run(rig.sdk, script=path.stem)["controls"]["factor"]["default"] == 1


@pytest.mark.parametrize("radius", [3.5, "3.5", "25", 25])
def test_add_layer_normalizes_numeric_controls(rig, radius):
    add = load("image_add_tool", rig.root / "tools/tool_add_layer.py").AddLayer()
    result = add.run(rig.sdk, script="canvas_blur", controls={"radius": radius})
    assert result["ok"]
    value = result["data"]["layers"][0]["controls"]["radius"]
    assert type(value) in (int, float) and value == float(radius)
    manage = load("image_manage_tool", rig.root / "tools/tool_manage_layers.py").ManageLayers()
    assert manage.run(rig.sdk, "set_control", chain_index=0, name="radius", value="2.5")["ok"]
    manage.run(rig.sdk, "set_controls", chain_index=0, controls={"radius": "4"})
    state = rig.service.for_session(rig.sdk)
    assert state["layers"][0]["controls"]["radius"] == 4


def test_control_coercion_is_recursive_and_preserves_text(rig):
    cat = catalog(rig)
    controls = {"points": [["1", "2.5"], ["3", "4"]], "width": "2"}
    result = cat.prepare(rig.sdk, "technique_line", controls)["controls"]
    assert result["points"] == [[1, 2.5], [3, 4]]
    assert controls["points"][0][0] == "1"
    assert cat.prepare(rig.sdk, "technique_rotate", {"expand": "false"})["controls"]["expand"] is False
    assert cat.prepare(rig.sdk, "technique_crop", {"right": "25", "bottom": "10"})["controls"]["right"] == 25
    assert cat.prepare(rig.sdk, "technique_text", {"content": "25"})["controls"]["content"] == "25"


@pytest.mark.parametrize("value", [True, False, "true", "false", "wide", "", "NaN", "Infinity", "-1"])
def test_bad_numeric_controls_explain_received_value(rig, value):
    with pytest.raises(ValueError) as failure:
        catalog(rig).prepare(rig.sdk, "canvas_blur", {"radius": value})
    assert "radius" in str(failure.value) and repr(value) in str(failure.value)


def test_fractional_integer_controls_are_not_truncated(rig):
    with pytest.raises(ValueError, match="fractions are not rounded"):
        catalog(rig).prepare(rig.sdk, "technique_crop", {"right": "2.5", "bottom": 10})
