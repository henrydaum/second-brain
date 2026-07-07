"""Tests for the store Backups bundle (``commands/command_backups.py`` +
``commands/helpers/datadir_backup.py``).

The package lives on the ``store`` branch, so both files are materialized
from the local store ref via ``git show`` and loaded with importlib under a
hand-built namespace package (mirroring what
``plugin_discovery._ensure_external_namespaces`` does for installed trees)
so the command's relative helper import works. Skips cleanly when no store
ref is available.
"""

import importlib.util
import json
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import pytest

from pipeline.database import Database

# Import the state_machine package before runtime modules to settle the
# package-init circular import (state_machine/__init__ pulls in the runtime).
import state_machine  # noqa: F401

_REPO = Path(__file__).resolve().parents[1]
_FILES = ("commands/command_backups.py", "commands/helpers/datadir_backup.py")
_PKG = "sb_backups_under_test"


def _store_source(rel: str) -> str | None:
    for ref in ("store", "origin/store"):
        proc = subprocess.run(
            ["git", "-C", str(_REPO), "show", f"{ref}:{rel}"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", check=False)
        if proc.returncode == 0:
            return proc.stdout
    return None


@pytest.fixture(scope="module")
def package(tmp_path_factory):
    """Load (command_module, engine_module) off the store ref."""
    sources = {rel: _store_source(rel) for rel in _FILES}
    if any(src is None for src in sources.values()):
        pytest.skip("Backups package not present on a local store ref")
    root = tmp_path_factory.mktemp("backups_pkg")
    for rel, src in sources.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(src, encoding="utf-8")
    # Namespace packages so `from .helpers import datadir_backup` resolves.
    for name, sub in ((_PKG, ""), (f"{_PKG}.commands", "commands"),
                      (f"{_PKG}.commands.helpers", "commands/helpers")):
        mod = types.ModuleType(name)
        mod.__path__ = [str(root / sub)]
        sys.modules[name] = mod
    modules = {}
    for mod_name, rel in ((f"{_PKG}.commands.helpers.datadir_backup", _FILES[1]),
                          (f"{_PKG}.commands.command_backups", _FILES[0])):
        spec = importlib.util.spec_from_file_location(mod_name, root / rel)
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
        modules[mod_name.rsplit(".", 1)[-1]] = module
    return modules["command_backups"], modules["datadir_backup"]


@pytest.fixture()
def env(package, tmp_path):
    """A seeded temp DATA_DIR + real Database."""
    _, engine = package
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "config.json").write_text('{"llm_backend": "one"}', encoding="utf-8")
    (data_dir / "plugin_config.json").write_text('{"backup_keep_count": 10}', encoding="utf-8")
    (data_dir / "memory.md").write_text("remember me", encoding="utf-8")
    (data_dir / "sandbox_plugins" / "tools").mkdir(parents=True)
    (data_dir / "sandbox_plugins" / "tools" / "tool_x.py").write_text("A = 1", encoding="utf-8")
    (data_dir / "attachment_cache").mkdir()
    (data_dir / "attachment_cache" / "big.bin").write_bytes(b"x" * 32)
    (data_dir / "app.log").write_text("log", encoding="utf-8")
    (data_dir / "heartbeat").write_text("", encoding="utf-8")
    (tmp_path / "outside").mkdir()
    db = Database(str(tmp_path / "outside" / "db.sqlite"))  # db_path outside DATA_DIR
    with db.lock:
        db.conn.execute("CREATE TABLE IF NOT EXISTS t (v TEXT)")
        db.conn.execute("INSERT INTO t (v) VALUES ('original')")
        db.conn.commit()
    return engine, data_dir, db


def _rows(db):
    with db.lock:
        return [r[0] for r in db.conn.execute("SELECT v FROM t ORDER BY v")]


def test_roundtrip_restore(env):
    engine, data_dir, db = env
    result = engine.create_backup(data_dir, db, name="base")
    assert Path(result["path"]).is_file() and result["file_count"] >= 5

    # Mutate everything the backup covers.
    with db.lock:
        db.conn.execute("INSERT INTO t (v) VALUES ('after')")
        db.conn.commit()
    (data_dir / "config.json").write_text('{"llm_backend": "two"}', encoding="utf-8")
    (data_dir / "sandbox_plugins" / "tools" / "tool_x.py").unlink()
    (data_dir / "sandbox_plugins" / "tools" / "tool_new.py").write_text("B = 2", encoding="utf-8")

    restored = engine.restore_backup(data_dir, db, "base")
    assert _rows(db) == ["original"]  # db row added after backup is gone
    assert json.loads((data_dir / "config.json").read_text(encoding="utf-8")) == {"llm_backend": "one"}
    assert (data_dir / "sandbox_plugins" / "tools" / "tool_x.py").is_file()
    assert not (data_dir / "sandbox_plugins" / "tools" / "tool_new.py").exists()
    assert restored["config"] == {"llm_backend": "one"}
    assert restored["plugin_config"] == {"backup_keep_count": 10}
    assert (data_dir / "backups" / f"{restored['safety_backup']}.zip").is_file()


def test_exclusions(env):
    engine, data_dir, db = env
    engine.create_backup(data_dir, db, name="scoped")
    with zipfile.ZipFile(data_dir / "backups" / "scoped.zip") as zf:
        names = zf.namelist()
    assert not any(n.startswith(("attachment_cache", "backups")) or n in ("app.log", "heartbeat") for n in names)
    assert "database.db" in names and "manifest.json" in names


def test_tampered_member_rejected_before_mutation(env):
    engine, data_dir, db = env
    engine.create_backup(data_dir, db, name="tamper")
    path = data_dir / "backups" / "tamper.zip"
    _rewrite_member(path, "memory.md", b"evil edit")
    (data_dir / "memory.md").write_text("live", encoding="utf-8")
    with pytest.raises(engine.BackupError, match="Integrity"):
        engine.restore_backup(data_dir, db, "tamper")
    assert (data_dir / "memory.md").read_text(encoding="utf-8") == "live"  # nothing mutated


def test_corrupt_db_snapshot_rejected(env):
    engine, data_dir, db = env
    engine.create_backup(data_dir, db, name="corrupt")
    path = data_dir / "backups" / "corrupt.zip"
    garbage = b"not a sqlite file" * 10
    _rewrite_member(path, "database.db", garbage, fix_manifest_sha=True)
    with pytest.raises(engine.BackupError, match="snapshot"):
        engine.restore_backup(data_dir, db, "corrupt")
    assert _rows(db) == ["original"]  # live db untouched


def test_zip_traversal_rejected(env):
    engine, data_dir, db = env
    path = engine.backups_dir(data_dir) / "evil.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("manifest.json", json.dumps({"files": []}))
        zf.writestr("../evil.txt", "boom")
    with pytest.raises(engine.BackupError, match="Unsafe"):
        engine.restore_backup(data_dir, db, "evil")
    assert not (data_dir / "evil.txt").exists()


def test_rotation(env):
    engine, data_dir, db = env
    for i in range(4):
        engine.create_backup(data_dir, db, name=f"rot-{i}")
    assert engine.prune_backups(data_dir, 0) == []
    deleted = engine.prune_backups(data_dir, 2)
    assert deleted == ["rot-0", "rot-1"]
    assert [b["name"] for b in engine.list_backups(data_dir)] == ["rot-2", "rot-3"]


def test_name_validation(env):
    engine, data_dir, db = env
    for bad in ("../x", "a/b", "a\\b", "a b"):
        with pytest.raises(engine.BackupError):
            engine.create_backup(data_dir, db, name=bad)


class _StubRuntime:
    def __init__(self, config):
        self.config = config
        self.pushed = []

    def push_message(self, session_key, message, source=None):
        self.pushed.append(message)


class _StubRegistry:
    def __init__(self):
        self.dispatched = []

    def dispatch_dict(self, name, args, session_key=None):
        self.dispatched.append(name)


class _StubContext:
    def __init__(self, db, runtime, registry):
        self.db = db
        self.runtime = runtime
        self.command_registry = registry
        self.config = dict(runtime.config)
        self.session_key = "repl:test"
        self.user_id = 1
        self.root_dir = _REPO


def test_command_flow(package, env, monkeypatch):
    cmd_mod, engine = package
    _, data_dir, db = env
    monkeypatch.setattr(cmd_mod, "DATA_DIR", data_dir)
    cmd = cmd_mod.BackupsCommand()
    runtime = _StubRuntime({"llm_backend": "stale", "backup_keep_count": 10})
    context = _StubContext(db, runtime, _StubRegistry())

    # Form shapes per action.
    assert [s.name for s in cmd.form({}, context)] == ["action"]
    assert [s.name for s in cmd.form({"action": "create"}, context)] == ["action", "name"]
    assert [s.name for s in cmd.form({"action": "restore"}, context)] == ["action"]  # no backups yet

    out = cmd.run({"action": "create", "name": "cmd-test"}, context)
    assert "cmd-test" in out
    steps = cmd.form({"action": "restore", "backup": "cmd-test"}, context)
    assert [s.name for s in steps] == ["action", "backup", "confirm"]

    # confirm=no cancels both destructive actions.
    assert cmd.run({"action": "restore", "backup": "cmd-test", "confirm": "no"}, context) == "Restore cancelled."
    assert cmd.run({"action": "delete", "backup": "cmd-test", "confirm": "no"}, context) == "Delete cancelled."
    assert context.command_registry.dispatched == []

    out = cmd.run({"action": "restore", "backup": "cmd-test", "confirm": "yes"}, context)
    assert "Restarting" in out
    assert context.command_registry.dispatched == ["restart"]
    # Live config dict replaced in place with restored config + plugin config.
    assert runtime.config == {"llm_backend": "one", "backup_keep_count": 10}

    out = cmd.run({"action": "delete", "backup": "cmd-test", "confirm": "yes"}, context)
    assert "Deleted backup 'cmd-test'" in out
    assert cmd.run({"action": "delete", "backup": "cmd-test", "confirm": "yes"}, context).startswith("Backup delete failed")


def _rewrite_member(path: Path, member: str, content: bytes, *, fix_manifest_sha: bool = False):
    """Replace one member's bytes inside a backup zip (optionally keeping the
    manifest sha consistent, to drive past the sha gate to later checks)."""
    import hashlib
    with zipfile.ZipFile(path) as zf:
        items = {n: zf.read(n) for n in zf.namelist()}
    items[member] = content
    if fix_manifest_sha:
        manifest = json.loads(items["manifest.json"])
        for entry in manifest["files"]:
            if entry["path"] == member:
                entry["sha256"] = hashlib.sha256(content).hexdigest()
                entry["bytes"] = len(content)
        items["manifest.json"] = json.dumps(manifest).encode()
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for n, data in items.items():
            zf.writestr(n, data)
