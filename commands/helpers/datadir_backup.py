"""DATA_DIR backup/restore engine for the `/backups` command.

Pure stdlib, no kernel imports: the command passes in the data dir and the
live Database handle, so this module can be unit-tested against a temp
directory. Backups are single zip archives under ``DATA_DIR/backups/``,
each carrying a ``manifest.json`` with per-file SHA-256 provenance
(mirroring the package manager's install records).

The SQLite database is never file-copied: snapshots ride SQLite's online
backup API (so WAL sidecars and a ``db_path`` outside DATA_DIR are
irrelevant), and restore streams the snapshot back *into* the live
connection — Windows file locks never come into play.
"""

dependencies_files = []
dependencies_pip = []

import hashlib
import json
import re
import shutil
import sqlite3
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath

BACKUP_DIRNAME = "backups"
DB_MEMBER = "database.db"
MANIFEST_MEMBER = "manifest.json"
# What a backup contains, relative to DATA_DIR. Missing entries are skipped.
# attachment_cache/ (bulky, re-syncable), app.log*, heartbeat, desktop.ini
# and backups/ itself are deliberately excluded.
SCOPE_FILES = ("config.json", "plugin_config.json", "memory.md")
SCOPE_DIRS = ("memory", "sandbox_plugins", "installed_plugins", "packages")

_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class BackupError(Exception):
    """Any user-reportable backup/restore failure."""


def backups_dir(data_dir: Path) -> Path:
    """Return DATA_DIR/backups, creating it if needed."""
    path = Path(data_dir) / BACKUP_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_backup(data_dir, db, *, name=None, root_dir=None, progress=None):
    """Snapshot the db + scoped DATA_DIR tree into a single zip.

    Returns ``{"name", "path", "bytes", "file_count", "manifest_sha256"}``.
    """
    data_dir = Path(data_dir)
    progress = progress or (lambda msg: None)
    name = _validated_name(name) if name else time.strftime("backup-%Y%m%d-%H%M%S")
    target = backups_dir(data_dir) / f"{name}.zip"
    if target.exists():
        raise BackupError(f"A backup named '{name}' already exists.")

    with tempfile.TemporaryDirectory(dir=backups_dir(data_dir)) as tmp:
        tmp = Path(tmp)
        progress("Snapshotting database...")
        snapshot = tmp / DB_MEMBER
        _snapshot_db(db, snapshot)

        entries = [(DB_MEMBER, snapshot)]
        for rel in SCOPE_FILES:
            src = data_dir / rel
            if src.is_file():
                entries.append((rel, src))
        for rel in SCOPE_DIRS:
            src = data_dir / rel
            if src.is_dir():
                for f in sorted(p for p in src.rglob("*") if p.is_file()):
                    entries.append((f"{rel}/{f.relative_to(src).as_posix()}", f))

        progress(f"Archiving {len(entries)} files...")
        manifest = {
            "name": name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "app_commit": _app_commit(root_dir),
            "db_path": str(getattr(db, "db_path", "")),
            "files": [],
            "total_bytes": 0,
        }
        # Build at a .partial name and os.replace at the end so a mid-create
        # crash never leaves a plausible-looking backup zip behind.
        partial = target.with_name(target.name + ".partial")
        try:
            with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED) as zf:
                for arcname, src in entries:
                    zf.write(src, arcname)
                    size = src.stat().st_size
                    manifest["files"].append({"path": arcname, "sha256": _sha256_file(src), "bytes": size})
                    manifest["total_bytes"] += size
                zf.writestr(MANIFEST_MEMBER, json.dumps(manifest, indent=1))
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)

    progress(f"Backup written: {target.name}")
    return {
        "name": name,
        "path": str(target),
        "bytes": target.stat().st_size,
        "file_count": len(entries),
        "manifest_sha256": hashlib.sha256(json.dumps(manifest, indent=1).encode()).hexdigest(),
        "created_at": manifest["created_at"],
    }


def list_backups(data_dir) -> list[dict]:
    """Return manifest summaries for every backup zip, oldest first."""
    items = []
    for path in sorted(backups_dir(Path(data_dir)).glob("*.zip")):
        try:
            with zipfile.ZipFile(path) as zf:
                manifest = json.loads(zf.read(MANIFEST_MEMBER))
        except Exception:
            manifest = {}
        items.append({
            "name": path.stem,
            "path": str(path),
            "bytes": path.stat().st_size,
            "created_at": manifest.get("created_at", ""),
            "file_count": len(manifest.get("files", [])),
            "app_commit": manifest.get("app_commit"),
            "valid": bool(manifest),
        })
    items.sort(key=lambda i: i["created_at"] or "")
    return items


def delete_backup(data_dir, name: str) -> dict:
    """Delete one backup zip by name."""
    path = backups_dir(Path(data_dir)) / f"{_validated_name(name)}.zip"
    if not path.is_file():
        raise BackupError(f"No backup named '{name}'.")
    size = path.stat().st_size
    path.unlink()
    return {"name": name, "bytes": size}


def prune_backups(data_dir, keep: int) -> list[str]:
    """Delete the oldest backups beyond *keep*; returns deleted names."""
    if keep <= 0:
        return []
    deleted = []
    for item in list_backups(data_dir)[:-keep]:
        try:
            Path(item["path"]).unlink()
            deleted.append(item["name"])
        except FileNotFoundError:
            pass
    return deleted


def restore_backup(data_dir, db, name: str, *, root_dir=None, progress=None) -> dict:
    """Restore a backup: validate fully, safety-backup, stream db, copy files.

    Nothing is mutated until the archive fully validates (member names,
    per-file SHA-256, db integrity_check). The db streams in first via the
    online backup API; the file phase is idempotent, so a mid-restore crash
    is recovered by re-running the restore. Returns the parsed restored
    config dicts for the caller to apply to the live config.
    """
    data_dir = Path(data_dir)
    progress = progress or (lambda msg: None)
    path = backups_dir(data_dir) / f"{_validated_name(name)}.zip"
    if not path.is_file():
        raise BackupError(f"No backup named '{name}'.")

    tmp = Path(tempfile.mkdtemp(dir=backups_dir(data_dir)))
    try:
        progress("Validating archive...")
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            if MANIFEST_MEMBER not in names:
                raise BackupError(f"'{name}' has no manifest; not a Second Brain backup.")
            manifest = json.loads(zf.read(MANIFEST_MEMBER))
            _safe_extract(zf, tmp)
        for entry in manifest.get("files", []):
            member = tmp / PurePosixPath(entry["path"])
            if not member.is_file() or _sha256_file(member) != entry.get("sha256"):
                raise BackupError(f"Integrity check failed for '{entry['path']}'; backup may be corrupt.")
        _verify_snapshot(tmp / DB_MEMBER)

        progress("Taking safety backup...")
        safety = create_backup(data_dir, db, name=time.strftime("pre-restore-%Y%m%d-%H%M%S"), root_dir=root_dir)

        progress("Restoring database...")
        snap = sqlite3.connect(f"file:{(tmp / DB_MEMBER).as_posix()}?mode=ro", uri=True)
        try:
            with db.lock:
                snap.backup(db.conn)
        finally:
            snap.close()

        progress("Restoring files...")
        restored = 1
        for rel in SCOPE_FILES:
            src, dst = tmp / rel, data_dir / rel
            if src.is_file():
                shutil.copy2(src, dst)
                restored += 1
            elif dst.is_file():
                dst.unlink()
        for rel in SCOPE_DIRS:
            src, dst = tmp / rel, data_dir / rel
            if dst.is_dir():
                shutil.rmtree(dst)
            if src.is_dir():
                shutil.copytree(src, dst)
                restored += sum(1 for p in dst.rglob("*") if p.is_file())

        return {
            "name": name,
            "safety_backup": safety["name"],
            "restored_files": restored,
            "config": _read_json(tmp / "config.json"),
            "plugin_config": _read_json(tmp / "plugin_config.json"),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _validated_name(name: str) -> str:
    name = (name or "").strip()
    if not _NAME_RE.match(name):
        raise BackupError("Backup names may only use letters, digits, '.', '_' and '-'.")
    return name


def _snapshot_db(db, dest: Path) -> None:
    """Copy the live db into *dest* with SQLite's online backup API."""
    dst = sqlite3.connect(str(dest))
    try:
        with db.lock:
            db.conn.backup(dst)
        dst.commit()
    finally:
        dst.close()


def _verify_snapshot(path: Path) -> None:
    """Reject a snapshot that SQLite itself considers damaged."""
    if not path.is_file():
        raise BackupError("Backup contains no database snapshot.")
    try:
        conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        try:
            result = conn.execute("PRAGMA integrity_check").fetchone()
        finally:
            conn.close()
    except sqlite3.Error as e:
        raise BackupError(f"Database snapshot is unreadable: {e}")
    if not result or result[0] != "ok":
        raise BackupError("Database snapshot failed integrity check.")


def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    """Extract while rejecting absolute or parent-escaping member names."""
    dest = dest.resolve()
    for info in zf.infolist():
        member = info.filename.replace("\\", "/")
        if member.startswith("/") or PurePosixPath(member).is_absolute() or ".." in PurePosixPath(member).parts or (len(member) > 1 and member[1] == ":"):
            raise BackupError(f"Unsafe path in archive: {info.filename}")
        target = (dest / PurePosixPath(member)).resolve()
        if not str(target).startswith(str(dest)):
            raise BackupError(f"Unsafe path in archive: {info.filename}")
        zf.extract(info, dest)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _app_commit(root_dir) -> str | None:
    """Best-effort repo commit for provenance; None on any failure."""
    if not root_dir:
        return None
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root_dir),
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None
