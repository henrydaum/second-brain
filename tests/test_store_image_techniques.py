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
    script = "canvas_" + name
    controls = dict(catalog(rig).CATALOG[script]["example"])
    if name == "load_image": controls["path"] = photo
    if name in ("crop", "shape"): controls.update(left=2, top=1, right=10, bottom=7)
    if name == "resize": controls.update(width=6, height=4)
    if name == "line": controls.update(points=[[1, 1], [10, 6]], width=1)
    if name == "text": controls.update(content="Hi\nX", x=1, y=0, size=6)
    spec = catalog(rig).prepare(script, controls)
    cid = rig.service.create(rig.sdk, width=12, height=8)
    if spec["kind"] != "background":
        rig.service.add_layer(rig.sdk, cid, "canvas_load_image", "background", {"path": photo})
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
    s.add_layer(sdk, cid, "canvas_load_image", "background", {"path": photo})
    s.add_layer(sdk, cid, "canvas_crop", "filter", {"left": 2, "top": 1, "right": 10, "bottom": 7})
    s.add_layer(sdk, cid, "canvas_rotate", "filter", {"angle": 90})
    first = rig.renderer.main(sdk, cid)
    expected = rig.kit.read_image(sdk, photo).crop((2, 1, 10, 7)).transpose(Image.Transpose.ROTATE_90)
    assert rig.kit.read_image(sdk, first["path"]).tobytes() == expected.tobytes()
    assert (first["width"], first["height"]) == (6, 8)
    s.add_layer(sdk, cid, "canvas_resize", "filter", {"width": 3, "height": 4})
    second = rig.renderer.main(sdk, cid)
    assert second["cached_layers"] == 3
    assert (second["width"], second["height"]) == (3, 4)
    s.undo(sdk, cid)
    assert rig.renderer.main(sdk, cid)["path"] == first["path"]


@pytest.mark.parametrize("script,controls", [
    ("canvas_gamma", {"gamma": 0}), ("canvas_saturation", {"factor": True}),
    ("canvas_blur", {"radius": float("nan")}), ("canvas_blur", {"strength": 5}),
    ("canvas_crop", {"right": 0, "bottom": 4}), ("canvas_resize", {"width": 1.5, "height": 3}),
    ("canvas_line", {"points": [[0, 1, 2], [3, 4]]}), ("canvas_load_image", {"path": ""}),
    ("canvas_sharpen", {"threshold": 256}),
])
def test_invalid_controls_are_rejected(rig, script, controls):
    with pytest.raises(ValueError): catalog(rig).prepare(script, controls)


def test_catalog_discovery_and_fine_adjustment_workflow(rig, photo):
    sdk, s = rig.sdk, rig.service
    add = load("image_add_tool", rig.root / "tools/tool_add_layer.py").AddLayer()
    manage = load("image_manage_tool", rig.root / "tools/tool_manage_layers.py").ManageLayers()
    search = load("image_search_tool", rig.root / "tools/tool_search_techniques.py").SearchTechniques()
    assert any(row["script"] == "canvas_sharpen" for row in search.run(sdk, query="sharpness"))
    assert len(search.run(sdk)) == len(NAMES)
    assert search.run(sdk, script="canvas_blur")["controls"]["radius"]["step"] == .25
    added = add.run(sdk, script="canvas_load_image", controls={"path": photo})
    cid = added["data"]["canvas_id"]
    assert added["data"]["layers"][0]["controls"]["fit"] == "native"
    add.run(sdk, script="canvas_saturation", controls={"factor": 1.1})
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
    s.add_layer(sdk, cid, "canvas_load_image", "background", {"path": photo})
    original = rig.renderer.main(sdk, cid)
    s.set_palette(sdk, cid, colors={"accent": "#ff0000", "primary": "#00ff00"})
    after = rig.renderer.main(sdk, cid)
    assert rig.kit.read_image(sdk, original["path"]).tobytes() == rig.kit.read_image(sdk, after["path"]).tobytes()
    s.add_layer(sdk, cid, "canvas_solid", "object", {"color": "@accent"})
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
    rig.service.add_layer(rig.sdk, cid, "canvas_load_image", "background", {"path": photo})
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
    pixels = load("image_test.canvas_pixels", rig.scripts / "canvas_pixels.py")
    effective = catalog(rig).prepare("canvas_" + name, controls)["controls"]
    result = getattr(pixels, name)(rig.sdk, image, effective, {"colors": {"secondary": "#000000", "accent": "#ffffff"}})
    assert result.tobytes() == image.tobytes()


@pytest.mark.parametrize("radius", [.5, 2, 3, 15])
def test_blur_does_not_bleed_hidden_rgb(rig, radius):
    from PIL import Image
    image = Image.new("RGBA", (21, 9), (0, 0, 255, 0))
    for y in range(9): image.putpixel((10, y), (255, 0, 0, 128))
    pixels = load("image_test.canvas_pixels", rig.scripts / "canvas_pixels.py")
    result = pixels.blur(rig.sdk, image, {"radius": radius}, {})
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
    rig.service.add_layer(rig.sdk, cid, "canvas_load_image", "background", {"path": path})
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
    for name, kind, controls in steps: s.add_layer(sdk, cid, "canvas_" + name, kind, controls)
    adapter = types.SimpleNamespace(exports=s.exports)
    for name in s.exports:
        setattr(adapter, name, lambda *a, _name=name, **k: getattr(s, _name)(sdk, *a, **k))
    sb = Sandbox(context=types.SimpleNamespace(services={"canvas": adapter}), approve=lambda *a, **k: True)
    previous = bridge._SANDBOX
    bridge.configure(sb)
    sb.plugin_roots = list(roots.values())
    try:
        result = sb.run(str(rig.scripts / "canvas_render.py"), "main",
                        kwargs={"canvas_id": cid, "seed": 0}, chain=Chain(root="user"))
        assert result.ok, result.error
        assert (result.data["width"], result.data["height"]) == (6, 8)
    finally:
        bridge.configure(previous)
        sb.shutdown()


def test_manifest_ships_every_technique_and_dependency(rig):
    from sandbox.validator import validate_file
    manifest = json.loads((rig.root / "bundles/bundle_image_editing.json").read_text())
    files = set(manifest["files"])
    assert "tools/tool_search_techniques.py" in files
    for name in NAMES: assert f"scripts/canvas_{name}.py" in files
    for path in files:
        report = validate_file(rig.root / path)
        assert report.ok, report.render()
        assert set(report.declarations.get("dependencies_files", [])) <= files
