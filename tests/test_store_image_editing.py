"""Image bundle behavior, using real PNGs and the current SDK surface."""
import importlib.util
import json
import shutil
import sqlite3
import sys
import types
from pathlib import Path

import pytest
import sandbox
from guest.sdk import _Path
from tests.support import store_worktree, retarget_trees


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def rig(tmp_path):
    pytest.importorskip("PIL")
    pytest.importorskip("numpy")
    root = store_worktree()
    if root is None:
        pytest.skip("store checkout unavailable")
    for name in list(sys.modules):
        if name == "image_test" or name.startswith("image_test."):
            sys.modules.pop(name, None)
    scripts = tmp_path / "installed" / "scripts"
    scripts.mkdir(parents=True)
    for source in [root / "scripts/art_kit.py", *sorted((root / "scripts").glob("canvas_*.py"))]:
        shutil.copyfile(source, scripts / source.name)
    package = types.ModuleType("image_test")
    package.__path__ = [str(scripts)]
    sys.modules[package.__name__] = package
    kit = load("image_test.art_kit", scripts / "art_kit.py")
    renderer = load("image_test.canvas_render", scripts / "canvas_render.py")
    service = load("image_test_service", root / "services/service_canvas.py").CanvasService()
    db = sqlite3.connect(":memory:", check_same_thread=False)
    db.row_factory = sqlite3.Row
    class DB:
        fail = False
        def define(self, sql):
            db.execute(sql)
        def write(self, sql, args=()):
            if self.fail:
                raise OSError("disk full")
            db.execute(sql, args)
            db.commit()
        def query(self, sql, args=()):
            return [dict(row) for row in db.execute(sql, args)]
    class FS:
        def exists(self, p): return Path(p).exists()
        def read(self, p): return Path(p).read_text(encoding="utf-8")
        def read_bytes(self, p): return Path(p).read_bytes()
        def write_bytes(self, p, data):
            Path(p).parent.mkdir(parents=True, exist_ok=True)
            Path(p).write_bytes(data)
        def move(self, src, dst):
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            Path(src).replace(dst)
        def delete(self, p): Path(p).unlink()
        def temp(self, suffix):
            import uuid
            return str(tmp_path / (uuid.uuid4().hex + suffix))
    sdk = types.SimpleNamespace(
        db=DB(), fs=FS(), path=_Path,
        paths=types.SimpleNamespace(get=lambda key: str(tmp_path / key)),
        session=types.SimpleNamespace(get=lambda: {"key": "test", "user_id": "u", "conversation_id": "c"}),
        log=lambda *a, **k: None,
    )
    sdk.services = types.SimpleNamespace(call=lambda service_name, method, *a, **k: getattr(service, method)(sdk, *a, **k))
    calls = []
    def run(path, **kwargs):
        if not Path(path).is_absolute():
            path = scripts / path
        calls.append(Path(path).name)
        module = load("image_test." + Path(path).stem, path)
        return module.main(sdk, **kwargs)
    sdk.scripts = types.SimpleNamespace(run=run)
    sdk.ok = lambda data, **extra: {"ok": True, "data": data, **extra}
    sdk.fail = lambda error: {"ok": False, "error": error}
    service.start(sdk)
    return types.SimpleNamespace(sdk=sdk, service=service, renderer=renderer, kit=kit,
                                 scripts=scripts, calls=calls, root=root)


LAYER = '''box = "image_editing"
from .art_kit import write_png
def main(sdk, kind, input_path, output_path, width, height, seed, palette, controls):
    from PIL import Image
    write_png(sdk, output_path, Image.new("RGBA", (width, height), tuple(controls["color"])))
'''


def test_state_history_and_failure_are_isolated(rig):
    s, sdk = rig.service, rig.sdk
    cid = s.create(sdk, width=1, height=10001)
    controls = {"nested": {"x": [1]}, "palette": "default"}
    s.add_layer(sdk, cid, "paint", "object", controls)
    controls["nested"]["x"].append(2)
    before = s.get_state(sdk, cid)
    assert before["layers"][0]["controls"]["nested"]["x"] == [1]
    s.set_control(sdk, cid, 0, "nested", {"x": [4]})
    assert s.undo(sdk, cid)["layers"] == before["layers"]
    assert s.redo(sdk, cid)["layers"][0]["controls"]["nested"] == {"x": [4]}
    before = s.get_state(sdk, cid)
    for w, h in [(0, 1), (1, -1), (1.5, 2), (True, 2)]:
        with pytest.raises(ValueError): s.set_dimensions(sdk, cid, w, h)
        assert s.get_state(sdk, cid) == before
    with pytest.raises(ValueError): s.move_layer(sdk, cid, 0, 9)
    assert s.get_state(sdk, cid) == before
    sdk.db.fail = True
    with pytest.raises(OSError): s.set_control(sdk, cid, 0, "x", 3)
    assert s.get_state(sdk, cid) == before


def test_session_restart_and_delete_preserves_recipe(rig):
    s, sdk = rig.service, rig.sdk
    cid = s.get_or_create(sdk)["canvas_id"]
    s.add_layer(sdk, cid, "paint", "object")
    s.add_layer(sdk, cid, "paint", "filter")
    s.add_layer(sdk, cid, "paint", "background")
    assert len(s.get_state(sdk, cid)["layers"]) == 3
    assert len(s.remove_layer(sdk, cid, 0)["layers"]) == 2
    s.stop(sdk)
    s.start(sdk)
    assert s.for_session(sdk)["canvas_id"] == cid
    assert len(s.undo(sdk, cid)["layers"]) == 3


def test_transparency_cache_script_and_asset_invalidation(rig, tmp_path):
    s, sdk, r = rig.service, rig.sdk, rig.renderer
    cid = s.create(sdk, width=3, height=2)
    empty = r.main(sdk, cid, seed=0)
    assert not empty["cache_hit"]
    assert rig.kit.read_image(sdk, empty["path"]).getpixel((0, 0)) == (0, 0, 0, 0)
    for name in ("paint", "other"):
        (rig.scripts / (name + ".py")).write_text(LAYER)
    s.add_layer(sdk, cid, "paint", "object", {"color": [255, 0, 0, 128]})
    s.add_layer(sdk, cid, "other", "object", {"color": [0, 0, 255, 128]})
    result = r.main(sdk, cid)
    assert result["seed"] == 0
    assert rig.kit.read_image(sdk, result["path"]).getpixel((0, 0)) == (85, 0, 170, 192)
    assert r.main(sdk, cid)["cache_hit"]
    assert len(rig.calls) == 2
    s.set_control(sdk, cid, 1, "color", [0, 255, 0, 128])
    assert r.main(sdk, cid)["cached_layers"] == 1
    (rig.scripts / "other.py").write_text(LAYER + "\n# revision\n")
    assert r.main(sdk, cid)["cached_layers"] == 1
    asset = tmp_path / "asset.txt"
    asset.write_text("one")
    s.update_layer(sdk, cid, 1, dependencies=[str(asset)])
    r.main(sdk, cid)
    asset.write_text("two")
    assert r.main(sdk, cid)["cached_layers"] == 1
    assert r.main(sdk, cid, force=True)["cached_layers"] == 0


def test_composition_masks_filters_and_offsets(rig):
    from PIL import Image
    base = Image.new("RGBA", (2, 1), (255, 0, 0, 255))
    overlay = Image.new("RGBA", (1, 1), (0, 0, 255, 255))
    result = rig.kit.composite(base, overlay, offset=(1, 0), opacity=.5)
    assert [result.getpixel((x, 0)) for x in range(2)] == [(255, 0, 0, 255), (128, 0, 128, 255)]
    transparent = Image.new("RGBA", (2, 1))
    result = rig.kit.composite(base, transparent, replace=True, opacity=.5)
    assert result.getpixel((0, 0)) == (255, 0, 0, 128)
    mask = Image.new("L", (2, 1), 0)
    assert rig.kit.composite(base, transparent, replace=True, mask=mask).tobytes() == base.tobytes()
    gray = Image.new("RGBA", (2, 1), (128, 128, 128, 255))
    assert rig.kit.composite(gray, gray, blend_mode="multiply").getpixel((0, 0)) == (64, 64, 64, 255)


def test_corrupt_output_is_not_cached(rig):
    s, sdk = rig.service, rig.sdk
    cid = s.create(sdk, width=2, height=2)
    (rig.scripts / "paint.py").write_text(LAYER)
    s.add_layer(sdk, cid, "paint", "object", {"color": [20, 30, 40, 255]})
    original = rig.renderer.main(sdk, cid)
    Path(original["path"]).write_bytes(b"broken")
    assert not rig.renderer.main(sdk, cid)["cache_hit"]
    (rig.scripts / "paint.py").write_text('def main(sdk, **kwargs):\n    sdk.fs.write_bytes(kwargs["output_path"], b"broken")\n')
    with pytest.raises(OSError): rig.renderer.main(sdk, cid)


def test_bundle_validates(rig):
    from sandbox.validator import validate_file
    bundle = json.loads((rig.root / "bundles/bundle_image_editing.json").read_text())
    for file in bundle["files"]:
        report = validate_file(rig.root / file)
        assert report.ok, report.render()


def test_helpers_run_in_actual_sandbox(rig, tmp_path, monkeypatch):
    from sandbox import Sandbox, Chain
    from sandbox.guest.loader import unload_box
    roots = retarget_trees(monkeypatch, tmp_path)
    script = rig.scripts / "smoke.py"
    script.write_text('''box = "image_editing"
from .art_kit import composite, write_png
def main(sdk, out):
    from PIL import Image
    image = composite(Image.new("RGBA", (1, 1)), Image.new("RGBA", (1, 1), (255, 0, 0, 128)))
    write_png(sdk, out, image)
    return list(image.getpixel((0, 0)))
''')
    out = str(roots["workspace"] / "smoke.png")
    sb = Sandbox(approve=lambda *a, **k: True)
    try:
        result = sb.run(str(script), "main", kwargs={"out": out}, chain=Chain(root="user"))
        assert result.ok, result.error
        assert result.data == [255, 0, 0, 128]
    finally:
        sb.shutdown()
        unload_box("image_editing")


def test_full_renderer_nested_sandbox(rig, tmp_path, monkeypatch):
    from sandbox import Sandbox, Chain, bridge
    roots = retarget_trees(monkeypatch, tmp_path)
    cid = rig.service.create(rig.sdk, width=3, height=2)
    workspace_scripts = roots["workspace"] / "scripts"
    workspace_scripts.mkdir(parents=True)
    (workspace_scripts / "paint.py").write_text('dependencies_files = ["scripts/art_kit.py"]\n' + LAYER)
    rig.service.add_layer(rig.sdk, cid, "paint", "object", {"color": [5, 10, 20, 128]})
    adapter = types.SimpleNamespace(exports=rig.service.exports)
    for name in rig.service.exports:
        setattr(adapter, name, lambda *a, _name=name, **k: getattr(rig.service, _name)(rig.sdk, *a, **k))
    context = types.SimpleNamespace(services={"canvas": adapter})
    sb = Sandbox(context=context, approve=lambda *a, **k: True)
    previous = bridge._SANDBOX
    bridge.configure(sb)
    # configure reads the process's plugin path catalogue; retarget this test
    # after that so preceding tests cannot leave its cached roots in use.
    sb.plugin_roots = [roots["installed"], roots["workspace"]]
    try:
        for cached in (False, True):
            result = sb.run(str(rig.scripts / "canvas_render.py"), "main",
                            kwargs={"canvas_id": cid, "seed": 0}, chain=Chain(root="user"))
            assert result.ok, result.error
            assert result.data["cache_hit"] is cached
            assert rig.kit.read_image(rig.sdk, result.data["path"]).getpixel((0, 0)) == (5, 10, 20, 128)
    finally:
        bridge.configure(previous)
        sb.shutdown()


def test_pixel_geometry_preserves_alpha(rig):
    from PIL import Image
    image = Image.new("RGBA", (2, 1), (255, 0, 0, 128))
    assert rig.kit.resize_image(image, (4, 4), "contain").getpixel((0, 0))[3] == 0
    assert rig.kit.crop_image(image, (-1, 0, 2, 1)).getpixel((0, 0))[3] == 0
    moved = rig.kit.transform_image(image, (3, 1), (1, 0, -1, 0, 1, 0))
    assert moved.getpixel((0, 0))[3] == 0
    assert moved.getpixel((1, 0)) == (255, 0, 0, 128)


def test_visibility_duplicate_and_session_separation(rig):
    s, sdk = rig.service, rig.sdk
    cid = s.get_or_create(sdk, width=1, height=1)["canvas_id"]
    (rig.scripts / "paint.py").write_text(LAYER)
    s.add_layer(sdk, cid, "paint", "object", {"color": [255, 0, 0, 255]})
    state = s.duplicate_layer(sdk, cid, 0)
    assert state["layers"][0]["id"] != state["layers"][1]["id"]
    s.update_layer(sdk, cid, 0, visible=False)
    s.update_layer(sdk, cid, 1, visible=False)
    result = rig.renderer.main(sdk, cid)
    assert rig.kit.read_image(sdk, result["path"]).getpixel((0, 0)) == (0, 0, 0, 0)
    assert not rig.calls
    sdk.session.get = lambda: {"key": "another", "user_id": "u", "conversation_id": "c"}
    assert s.for_session(sdk) is None
    assert s.get_or_create(sdk)["canvas_id"] != cid
