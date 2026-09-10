"""Shipped techniques, their agent workflow, and actual nested sandbox execution."""
import json
from pathlib import Path
import types

import pytest

from tests.test_store_image_editing import rig, load


NAMES = "load_image crop resize rotate flip brightness contrast saturation exposure gamma blur sharpen grayscale invert solid gradient line shape text duotone vignette levels color_balance threshold posterize pixelate median pad mask_shape mask_range mask_combine curves white_balance shadows_highlights vibrance affine perspective drop_shadow outline color_key dither halftone displace texture chromatic_aberration fisheye swirl kaleidoscope feedback_tunnel scanlines glitch_slice pixel_sort ascii".split()


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
    if name in ("load_image", "mask_combine", "displace"): controls["path"] = photo
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
    texture_path = str(Path(roots["workspace"]) / "test_texture.png")
    steps = [("load_image", "background", {"path": photo}),
             ("crop", "filter", {"left": 2, "top": 1, "right": 10, "bottom": 7}),
             ("rotate", "filter", {"angle": 90}),
             ("saturation", "filter", {"factor": 1.15}),
             ("blur", "filter", {"radius": .5}),
             ("sharpen", "filter", {"amount": 20}),
             ("levels", "filter", {"black": 5, "white": 250}),
             ("color_balance", "filter", {"red": .05}),
             ("median", "filter", {"radius": 1}),
             ("pixelate", "filter", {"block_size": 2}),
             ("posterize", "filter", {"levels": 4}),
             ("threshold", "filter", {"threshold": 100}),
             ("pad", "filter", {"width": 6, "height": 8}),
             ("curves", "filter", {"points": [[0,0],[.5,.6],[1,1]]}),
             ("white_balance", "filter", {"temperature": 10}),
             ("shadows_highlights", "filter", {"shadows": .2}),
             ("vibrance", "filter", {"amount": .3}),
             ("affine", "filter", {"scale_x": .9}),
             ("perspective", "filter", {"corners": [[.05,0],[1,0],[1,1],[0,1]]}),
             ("drop_shadow", "filter", {"radius": 1, "x": 1, "y": 1}),
             ("outline", "filter", {"radius": 1}),
             ("color_key", "filter", {"amount": .3}),
             ("dither", "filter", {"strength": .5}),
             ("halftone", "filter", {"cell_size": 2}),
             ("displace", "filter", {"path": texture_path, "x": 1, "y": 1}),
             ("chromatic_aberration", "filter", {"amount":.02}),
             ("fisheye", "filter", {"strength":.3}),
             ("swirl", "filter", {"turns":.2}),
             ("kaleidoscope", "filter", {"segments":4}),
             ("feedback_tunnel", "filter", {"depth":3}),
             ("scanlines", "filter", {"lines":4}),
             ("glitch_slice", "filter", {"slices":3}),
             ("pixel_sort", "filter", {"low":0,"high":1}),
             ("ascii", "filter", {"columns":3}),
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
    asked = []
    def approve_library_launch(chain, request, decision):
        asked.append((request.type, decision.reason))
        # Pillow/numpy launches legitimately ask; workspace IO never should.
        return request.type == "script.run" and "which imports" in decision.reason
    sb = Sandbox(context=types.SimpleNamespace(services={"canvas": adapter}), approve=approve_library_launch)
    previous = bridge._SANDBOX
    bridge.configure(sb)
    sb.plugin_roots = list(roots.values())
    try:
        texture = sb.run(str(rig.scripts / "technique_texture.py"), "main",
                         kwargs={"kind": "background", "input_path": None, "output_path": texture_path,
                                 "width": 6, "height": 8, "seed": 7,
                                 "palette": s.list_palettes(sdk)[0], "controls": {"scale": 4}}, chain=Chain(root="user"))
        assert texture.ok, texture.error
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
        assert all(kind == "script.run" and "which imports" in reason for kind, reason in asked)
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


def test_new_tonal_techniques_preserve_alpha_and_expected_values(rig):
    from PIL import Image
    image = Image.new("RGBA", (1, 1), (64, 128, 192, 127))
    def apply(name, controls):
        spec = catalog(rig).prepare(rig.sdk, "technique_" + name, controls)
        module = load("image_test.technique_" + name, rig.scripts / ("technique_" + name + ".py"))
        return module.apply(rig.sdk, image, spec["controls"], {"colors": {"accent": "#ff0000"}})
    assert apply("levels", {}).tobytes() == image.tobytes()
    assert apply("color_balance", {}).tobytes() == image.tobytes()
    assert apply("posterize", {"levels": 2}).getpixel((0, 0)) == (0, 255, 255, 127)
    assert apply("threshold", {"threshold": 100, "highlights": "@accent"}).getpixel((0, 0)) == (255, 0, 0, 127)
    assert apply("threshold", {"threshold": 100, "highlights": "transparent"}).getpixel((0, 0))[3] == 0
    assert apply("levels", {"black": 64, "white": 192}).getpixel((0, 0)) == (0, 128, 255, 127)
    with pytest.raises(ValueError): apply("levels", {"black": 100, "white": 50})


def test_pixelate_and_median_preserve_transparent_edges(rig):
    from PIL import Image
    image = Image.new("RGBA", (3, 3), (0, 0, 255, 0))
    for x in (0, 1):
        for y in range(3): image.putpixel((x, y), (255, 0, 0, 255))
    for name, controls in [("pixelate", {"block_size": 3}), ("median", {"radius": 1})]:
        module = load("image_test.technique_" + name, rig.scripts / ("technique_" + name + ".py"))
        result = module.apply(rig.sdk, image, controls, {})
        assert result.size == image.size
        assert all(p[:3] == (255, 0, 0) for p in [result.getpixel((x, y)) for y in range(result.height) for x in range(result.width)] if p[3])


def test_padding_places_pixels_without_resampling(rig):
    from PIL import Image
    module = load("image_test.technique_pad", rig.scripts / "technique_pad.py")
    image = Image.new("RGBA", (2, 2), (255, 0, 0, 128))
    result = module.apply(rig.sdk, image, {"width": 5, "height": 4, "x": 2, "y": 1, "color": "transparent"}, {})
    assert result.getpixel((0, 0)) == (0, 0, 0, 0)
    assert result.crop((2, 1, 4, 3)).tobytes() == image.tobytes()
    cropped = module.apply(rig.sdk, image, {"width": 1, "height": 1, "x": -1, "y": -1, "color": "transparent"}, {})
    assert cropped.getpixel((0, 0)) == (255, 0, 0, 128)


def test_render_folders_group_seeds_and_share_recipe_prefixes(rig):
    s, sdk = rig.service, rig.sdk
    cid = s.create(sdk, width=3, height=2)
    s.add_layer(sdk, cid, "technique_solid", "background", {"color": "red"})
    first = rig.renderer.main(sdk, cid, seed=1)
    second = rig.renderer.main(sdk, cid, seed=2)
    assert Path(first["path"]).parent == Path(second["path"]).parent
    assert Path(first["path"]).name == "1.png"
    assert Path(second["path"]).name == "2.png"
    assert rig.renderer.main(sdk, cid, seed=1)["cache_hit"]
    s.add_layer(sdk, cid, "technique_levels", "filter", {})
    longer = rig.renderer.main(sdk, cid, seed=1)
    assert longer["cached_layers"] == 1
    assert Path(longer["path"]).parent != Path(first["path"]).parent
    other = s.create(sdk, width=3, height=2)
    s.add_layer(sdk, other, "technique_solid", "background", {"color": "red"})
    assert rig.renderer.main(sdk, other, seed=1)["path"] == first["path"]


def test_live_canvas_prompt_refreshes_within_turn_and_is_read_only(rig):
    from types import SimpleNamespace
    from tests.test_system_prompt import _sections_with
    sdk, service = rig.sdk, rig.service
    assert service.agent_prompt_refresh == "call"
    before = service.list_canvases(sdk)
    assert "none selected" in service.agent_prompt(sdk)
    assert service.list_canvases(sdk) == before
    cid = service.get_or_create(sdk, width=7, height=9)["canvas_id"]
    service.add_layer(sdk, cid, "technique_blur", "filter", {"radius": 2})
    plugin = SimpleNamespace(name="canvas", description="", parameters={},
                             agent_prompt=lambda ctx: service.agent_prompt(sdk),
                             agent_prompt_refresh=service.agent_prompt_refresh)
    _, first = _sections_with([plugin])
    service.set_control(sdk, cid, 0, "radius", 8)
    service.set_palette(sdk, cid, colors={"accent": "#123456"})
    _, second = _sections_with([plugin])
    assert '"radius":2' in first["content"]
    assert '"radius":8' in second["content"] and "#123456" in second["content"]
    assert cid in second["content"] and '"starting_dimensions":[7,9]' in second["content"]
    state = service.get_state(sdk, cid)
    service.agent_prompt(sdk)
    assert service.get_state(sdk, cid) == state
    sdk.session.get = lambda: {"key": "other", "user_id": "other", "conversation_id": "other"}
    assert "none selected" in service.agent_prompt(sdk)
    assert cid not in service.agent_prompt(sdk)


def test_cached_recipe_can_be_remixed_after_edits_and_restart(rig, photo):
    sdk, service = rig.sdk, rig.service
    cid = service.get_or_create(sdk)["canvas_id"]
    service.add_layer(sdk, cid, "technique_load_image", "background", {"path": photo})
    result = rig.renderer.main(sdk, cid, seed=17)
    service.add_layer(sdk, cid, "technique_blur", "filter", {"radius": 3})
    service.stop(sdk)
    service.start(sdk)
    tool = load("image_manage_tool", rig.root / "tools/tool_manage_layers.py").ManageLayers()
    saved = tool.run(sdk, "cached", pool_hash=result["pool_hash"], seed=17)
    assert saved["pixels_available"] and saved["path"] == result["path"]
    assert len(saved["recipe"]["layers"]) == 1
    restored = tool.run(sdk, "remix", pool_hash=result["pool_hash"], seed=17)
    assert restored["canvas_id"] != cid and len(restored["layers"]) == 1
    assert len(service.get_state(sdk, cid)["layers"]) == 2
    assert restored["render_seed"] == 17 and not restored["undo_stack"]
    assert service.for_session(sdk)["canvas_id"] == restored["canvas_id"]
    # Evicting pixels does not delete the saved editable recipe.
    Path(result["path"]).unlink()
    assert not service.cached_render(sdk, result["pool_hash"], 17)["pixels_available"]
    assert service.remix(sdk, result["pool_hash"], 17)["layers"]
    with pytest.raises(ValueError): service.cached_render(sdk, "../bad", 17)


def test_mask_coverage_inversion_and_combination(rig, tmp_path):
    import numpy as np
    from PIL import Image
    source = Image.new("RGBA", (2, 1), (255, 255, 255, 0))
    source.putpixel((0, 0), (255, 255, 255, 128))
    def apply(name, image, controls):
        prepared = catalog(rig).prepare(rig.sdk, "technique_" + name, controls)
        module = load("image_test.technique_" + name, rig.scripts / ("technique_" + name + ".py"))
        return module.apply(rig.sdk, image, prepared["controls"], {"colors": {"accent": "#ffffff"}})
    mask = apply("mask_range", source, {})
    assert mask.getpixel((0, 0)) == (128, 128, 128, 255)
    assert mask.getpixel((1, 0)) == (0, 0, 0, 255)
    inverted = apply("mask_range", source, {"invert": True})
    assert inverted.getpixel((1, 0)) == (255, 255, 255, 255)
    path = str(tmp_path / "selection.png")
    rig.kit.write_png(rig.sdk, path, inverted)
    combined = apply("mask_combine", mask, {"path": path, "operation": "union"})
    assert combined.getpixel((1, 0)) == (255, 255, 255, 255)
    subtracted = apply("mask_combine", mask, {"path": path, "operation": "subtract"})
    assert subtracted.getpixel((0, 0)) == (1, 1, 1, 255)
    with pytest.raises(ValueError, match="matching dimensions"):
        apply("mask_combine", Image.new("RGBA", (3, 2)), {"path": path})
    # A real masked edit changes only selected coverage.
    base = Image.new("RGBA", (2, 1), "black")
    edit = Image.new("RGBA", (2, 1), "white")
    result = rig.kit.composite(base, edit, mask=mask, replace=True)
    assert result.getpixel((0, 0)) == (128, 128, 128, 255)
    assert result.getpixel((1, 0)) == (0, 0, 0, 255)


@pytest.mark.parametrize("name,controls", [
    ("curves", {}), ("white_balance", {}), ("shadows_highlights", {}),
    ("vibrance", {}), ("affine", {}), ("perspective", {}),
    ("drop_shadow", {"opacity": 0}), ("outline", {"radius": 0}),
    ("color_key", {"amount": 0}),
])
def test_new_batch_neutral_controls_preserve_pixels(rig, photo, name, controls):
    image = rig.kit.read_image(rig.sdk, photo)
    controls = catalog(rig).prepare(rig.sdk, "technique_" + name, controls)["controls"]
    module = load("image_test.technique_" + name, rig.scripts / ("technique_" + name + ".py"))
    assert module.apply(rig.sdk, image, controls, {}).tobytes() == image.tobytes()


def test_new_corrections_and_geometry(rig):
    from PIL import Image
    def apply(name, image, controls):
        controls = catalog(rig).prepare(rig.sdk, "technique_" + name, controls)["controls"]
        module = load("image_test.technique_" + name, rig.scripts / ("technique_" + name + ".py"))
        return module.apply(rig.sdk, image, controls, {})
    gray = Image.new("RGBA", (2, 2), (100, 100, 100, 77))
    warm = apply("white_balance", gray, {"temperature": 50}).getpixel((0, 0))
    assert warm[0] > warm[1] > warm[2] and warm[3] == 77
    bright = apply("shadows_highlights", gray, {"shadows": 1}).getpixel((0, 0))
    assert bright[0] > 100 and bright[3] == 77
    assert apply("vibrance", gray, {"amount": 1}).tobytes() == gray.tobytes()
    curve = apply("curves", gray, {"points": [[0, 0], [1, .5]], "channel": "red"})
    assert curve.getpixel((0, 0)) == (50, 100, 100, 77)
    with pytest.raises(ValueError, match="strictly increasing"):
        apply("curves", gray, {"points": [[0,0],[.5,.5],[.5,.8],[1,1]]})
    with pytest.raises(ValueError, match="singular"):
        apply("affine", gray, {"shear_x": 1, "shear_y": 1})
    with pytest.raises(ValueError, match="convex"):
        apply("perspective", gray, {"corners": [[0,0],[1,1],[1,0],[0,1]]})
    moved = apply("affine", Image.new("RGBA", (3, 2), "red"), {"x": 1})
    assert moved.getpixel((0, 0))[3] == 0
    assert moved.getpixel((1, 0)) == (255, 0, 0, 255)
    resized = apply("perspective", gray, {"width": 5, "height": 7})
    assert resized.size == (5, 7)


def test_alpha_effects_and_sampling_do_not_leak_hidden_colour(rig, tmp_path):
    import numpy as np
    from PIL import Image
    def apply(name, image, controls):
        controls = catalog(rig).prepare(rig.sdk, "technique_" + name, controls)["controls"]
        module = load("image_test.technique_" + name, rig.scripts / ("technique_" + name + ".py"))
        return module.apply(rig.sdk, image, controls, {})
    image = Image.new("RGBA", (5, 3))
    image.putpixel((1, 1), (255, 0, 0, 255))
    shadow = apply("drop_shadow", image, {"radius": 0, "x": 2, "y": 0, "opacity": 1})
    assert shadow.getpixel((1, 1)) == (255, 0, 0, 255)
    assert shadow.getpixel((3, 1)) == (0, 0, 0, 255)
    outline = apply("outline", image, {"radius": 1, "color": "blue"})
    assert outline.getpixel((1, 1)) == (255, 0, 0, 255)
    assert outline.getpixel((2, 1)) == (0, 0, 255, 255)
    keyed = apply("color_key", image, {"color": "red", "tolerance": 0, "softness": 0})
    assert keyed.getpixel((1, 1))[3] == 0
    edge = Image.new("RGBA", (2, 1), (0, 0, 255, 0))
    edge.putpixel((0, 0), (255, 0, 0, 128))
    sampled = rig.kit.sample_rgba(edge, np.array([[.5, -3]]), np.array([[0., 0.]]))
    assert sampled.getpixel((0, 0)) == (255, 0, 0, 64)
    assert sampled.getpixel((1, 0)) == (0, 0, 0, 0)
    path = str(tmp_path / "neutral_map.png")
    rig.kit.write_png(rig.sdk, path, Image.new("RGBA", edge.size, (128, 128, 128, 255)))
    assert apply("displace", edge, {"path": path}).tobytes() == edge.tobytes()
    with pytest.raises(ValueError, match="match"):
        apply("displace", image, {"path": path})


def test_pattern_endpoints_alpha_and_texture_seed(rig):
    from PIL import Image
    def apply(name, image, controls, seed=0):
        controls = catalog(rig).prepare(rig.sdk, "technique_" + name, controls)["controls"]
        module = load("image_test.technique_" + name, rig.scripts / ("technique_" + name + ".py"))
        palette = {"colors": {"primary": "black", "accent": "white"}}
        return module.apply(rig.sdk, image, controls, palette, **({"seed": seed} if name == "texture" else {}))
    image = Image.new("RGBA", (16, 16), (128, 128, 128, 117))
    dither = apply("dither", image, {})
    assert set(dither.getchannel("A").getextrema()) == {117}
    assert dither.getpixel((0, 0))[:3] == (255, 255, 255)
    assert dither.getpixel((0, 1))[:3] == (0, 0, 0)
    for color in ("black", "white"):
        solid = Image.new("RGBA", image.size, color)
        assert apply("halftone", solid, {}).tobytes() == solid.tobytes()
    texture = apply("texture", image, {"scale": 4}, seed=7)
    assert texture.tobytes() == apply("texture", image, {"scale": 4}, seed=7).tobytes()
    assert texture.tobytes() != apply("texture", image, {"scale": 4}, seed=8).tobytes()


@pytest.mark.parametrize("name,controls", [
    ("chromatic_aberration", {"amount": 0}), ("fisheye", {"strength": 0}),
    ("swirl", {"turns": 0}), ("feedback_tunnel", {"depth": 0}),
    ("scanlines", {"strength": 0}), ("glitch_slice", {"slices": 0}),
])
def test_glitch_neutral_settings_preserve_pixels(rig, photo, name, controls):
    image = rig.kit.read_image(rig.sdk, photo)
    spec = catalog(rig).prepare(rig.sdk, "technique_" + name, controls)
    module = load("image_test.technique_" + name, rig.scripts / ("technique_" + name + ".py"))
    assert module.apply(rig.sdk, image, spec["controls"], {}).tobytes() == image.tobytes()


def test_glitch_seed_sorting_and_scanline_density(rig, photo):
    from PIL import Image
    def apply(name, image, controls, **kwargs):
        controls = catalog(rig).prepare(rig.sdk, "technique_" + name, controls)["controls"]
        module = load("image_test.technique_" + name, rig.scripts / ("technique_" + name + ".py"))
        return module.apply(rig.sdk, image, controls, {}, **kwargs)
    image = rig.kit.read_image(rig.sdk, photo)
    first = apply("glitch_slice", image, {"shift": .4}, seed=12)
    assert first.tobytes() == apply("glitch_slice", image, {"shift": .4}, seed=12).tobytes()
    assert first.tobytes() != apply("glitch_slice", image, {"shift": .4}, seed=13).tobytes()
    row = Image.new("RGBA", (5, 1))
    pixels = [(200,200,200,255), (50,50,50,100), (255,0,0,0), (180,180,180,255), (10,10,10,255)]
    for x, pixel in enumerate(pixels): row.putpixel((x,0), pixel)
    sorted_row = apply("pixel_sort", row, {"low":0,"high":1})
    assert [sorted_row.getpixel((x,0)) for x in range(5)] == [pixels[1],pixels[0],pixels[2],pixels[4],pixels[3]]
    white = Image.new("RGBA", (3, 2), (255,255,255,100))
    scan = apply("scanlines", white, {"lines":1000,"strength":1,"width":.5})
    assert scan.getpixel((0,0)) == (128,128,128,100)
    with pytest.raises(ValueError, match="low"):
        apply("pixel_sort", row, {"low":.9,"high":.1})
    with pytest.raises(ValueError, match="ASCII"):
        apply("ascii", row, {"characters":"x"})


@pytest.mark.parametrize("name", ["fisheye", "swirl", "kaleidoscope", "feedback_tunnel", "glitch_slice", "chromatic_aberration"])
def test_glitch_warps_do_not_expose_hidden_rgb(rig, name):
    from PIL import Image
    image = Image.new("RGBA", (21, 13), (0,0,255,0))
    for y in range(3,10):
        for x in range(6,15): image.putpixel((x,y), (255,0,0,128))
    controls = catalog(rig).prepare(rig.sdk, "technique_" + name)["controls"]
    module = load("image_test.technique_" + name, rig.scripts / ("technique_" + name + ".py"))
    result = module.apply(rig.sdk, image, controls, {})
    assert result.size == image.size
    assert result.getchannel("A").getextrema()[1] > 0
    for y in range(result.height):
        for x in range(result.width):
            pixel = result.getpixel((x,y))
            if pixel[3]: assert pixel[2] == 0


def test_edit_tools_return_no_attachments_and_render_identifies_exact_png(rig, photo, tmp_path):
    import hashlib
    add = load("image_add_tool", rig.root / "tools/tool_add_layer.py").AddLayer()
    manage = load("image_manage_tool", rig.root / "tools/tool_manage_layers.py").ManageLayers()
    render = load("image_render_tool", rig.root / "tools/tool_render_canvas.py").RenderCanvas()
    result = add.run(rig.sdk, script="technique_load_image", controls={"path":photo})
    assert not result.get("attachments") and not result.get("attachment_paths")
    for opacity in (.55, "0.55"):
        result = add.run(rig.sdk, script="technique_scanlines", properties={"opacity":opacity})
        assert result["ok"] and result["data"]["layers"][-1]["opacity"] == .55
        assert not result.get("attachments") and not result.get("attachment_paths")
    inspected = manage.run(rig.sdk, "inspect")
    changed = manage.run(rig.sdk, "update", chain_index=1, properties={"opacity":"0.4"})
    assert not inspected.get("attachments") and not changed.get("attachments")
    bad = add.run(rig.sdk, script="technique_scanlines", properties={"opacity":"half"})
    assert not bad["ok"] and "'half'" in bad["error"] and "properties.opacity" in bad["error"]
    exported = str(tmp_path / "export.png")
    rendered = render.run(rig.sdk, out=exported)
    data = rendered["data"]
    assert rendered["attachments"] == [data["attachment_path"]]
    assert data["attachment_path"] != exported and data["path"] == exported
    encoded = Path(data["attachment_path"]).read_bytes()
    assert hashlib.sha256(encoded).hexdigest() == data["image_sha256"]
    assert encoded == Path(exported).read_bytes()


def test_later_object_does_not_participate_in_earlier_halftone(rig):
    s, sdk = rig.service, rig.sdk
    cid = s.create(sdk, width=8, height=8)
    s.add_layer(sdk, cid, "technique_solid", "background", {"color":"black"})
    s.add_layer(sdk, cid, "technique_halftone", "filter", {"ink":"black","paper":"white"})
    first = rig.renderer.main(sdk, cid, seed=0)
    s.add_layer(sdk, cid, "technique_solid", "object", {"color":"white"})
    later = rig.renderer.main(sdk, cid, seed=0)
    assert later["cached_layers"] == 2
    assert rig.kit.read_image(sdk, first["path"]).getpixel((0,0)) == (0,0,0,255)
    assert rig.kit.read_image(sdk, later["path"]).getpixel((0,0)) == (255,255,255,255)


@pytest.mark.parametrize("recipe", ["glitch", "trippy"])
def test_glitch_worked_recipes_use_valid_controls(rig, recipe):
    cat = catalog(rig)
    worked = cat.main(rig.sdk, recipe=recipe)
    for step in worked["steps"]:
        if step["tool"] == "add_layer":
            cat.prepare(rig.sdk, **step["args"])
    assert worked["steps"][-1]["tool"] == "render_canvas"
