"""Installing a parser or a task must take effect without a restart.

All four defects here have one shape: **a registry that changes at runtime,
read by something holding a snapshot taken at boot.** Installing
``parse_container`` teaches ``parsing`` that ``.zip`` means ``container``, and
three separate pieces of the pipeline went on believing what they were told
during startup — so the archive sat there, and the only cure anybody found was
to restart the app.

The failures compound rather than overlap, which is why each gets its own test:
the watcher never *sees* the file, the row that does exist is stale, and the
newly registered task is never asked about either.
"""

import pytest

from plugins.native.task import BaseTask
from pipeline.database import Database
from pipeline.orchestrator import Orchestrator
from pipeline.watcher import Watcher


@pytest.fixture
def db(tmp_path):
    return Database(str(tmp_path / "pipeline.db"))


class _ExtractContainer(BaseTask):
    """Stands in for the store's ``extract_container``: a modality-rooted task."""

    name = "extract_container"
    modalities = ["container"]
    reads = []
    writes = ["extracted_containers"]
    output_schema = ""


# ── The watcher's idea of what is worth looking at ───────────────────

def test_a_rescan_re_reads_which_extensions_a_parser_now_claims(db, monkeypatch):
    """The decisive one.

    ``_is_valid_file`` gates every file on ``supported_extensions``, which was
    read once in ``__init__``. ``.zip`` is not in ``parsing._NATIVE_DEFAULTS``,
    so before ``parse_container`` is installed an archive is not merely
    unparseable — it never enters the ``files`` table at all, and there is no
    row for any later fix to correct.
    """
    import pipeline.watcher as watcher_module

    monkeypatch.setattr(watcher_module, "get_supported_extensions",
                        lambda: {".md"})
    watcher = Watcher(orchestrator=None, db=db, config={})
    assert watcher._is_valid_file("/in/box.zip") is False

    # The parser is installed; the registry now answers differently.
    monkeypatch.setattr(watcher_module, "get_supported_extensions",
                        lambda: {".md", ".zip"})

    # Only the config-driven half used to be refreshed here, so this stayed
    # False until the next boot. ``rescan`` itself needs real directories and
    # a live observer; what it now does that it did not is exactly this line.
    watcher.supported_extensions = watcher_module.get_supported_extensions()
    assert watcher._is_valid_file("/in/box.zip") is True


def test_rescan_is_what_refreshes_the_extension_set(monkeypatch, db, tmp_path):
    """And it is reached by ``/packages`` after a parser install.

    Pinned separately from the behaviour above because the whole defect was
    that the refresh existed nowhere — ``rescan`` re-read the four
    config-driven fields beside it and not this one.
    """
    import pipeline.watcher as watcher_module

    monkeypatch.setattr(watcher_module, "get_supported_extensions",
                        lambda: {".md"})
    watcher = Watcher(orchestrator=None, db=db,
                      config={"sync_directories": [str(tmp_path)]})
    monkeypatch.setattr(watcher_module, "get_supported_extensions",
                        lambda: {".md", ".zip"})
    # The disk walk is the expensive half and not what is under test here.
    monkeypatch.setattr(Watcher, "_initial_scan", lambda self, dirs: None)

    watcher.rescan()

    assert ".zip" in watcher.supported_extensions


def test_a_scan_re_registers_a_file_whose_kind_changed_under_it(db, monkeypatch):
    """Untouched on disk, but the registry's answer about it has changed.

    The scan compared mtimes only, so a file already in the table was skipped
    forever — the row kept saying ``unknown`` and no modality-rooted task ever
    matched it. Nothing about the file changed; what changed is what we know.
    """
    import pipeline.watcher as watcher_module

    db.upsert_file("/in/box.zip", "box.zip", ".zip", "unknown", 100.0)
    monkeypatch.setattr(watcher_module, "get_modality", lambda ext: "container")

    known_mtime, known_modality = db.get_watched_file_state()["/in/box.zip"]
    unchanged_on_disk = abs(known_mtime - 100.0) <= 1.0
    stale = known_modality != watcher_module.get_modality(".zip")

    assert unchanged_on_disk and stale, "this is the branch mtime alone missed"


# ── The task nobody asked about ──────────────────────────────────────

def test_a_task_registered_while_running_catches_up_on_existing_files(db):
    """``_backfill_tasks`` ran only from ``start()``.

    So a task installed from the store was never enqueued against anything
    discovered before it existed. It looked broken, and restarting fixed it,
    which is the worst available combination — the bug and its workaround both
    point away from the cause.
    """
    db.upsert_file("/in/box.zip", "box.zip", ".zip", "container", 100.0)

    orchestrator = Orchestrator(db, {}, {})
    orchestrator.running = True          # what start() sets before backfilling
    orchestrator.register_task(_ExtractContainer())

    assert db.get_pending_tasks("extract_container") == [
        ("/in/box.zip", "extract_container")]


def test_registering_at_boot_leaves_the_backfill_to_start(db):
    """Boot registers every task and *then* sweeps once, so this must not.

    Harmless if it did — ``enqueue_task`` is ``INSERT OR IGNORE`` — but it
    would walk the whole files table once per task for nothing.
    """
    db.upsert_file("/in/box.zip", "box.zip", ".zip", "container", 100.0)

    orchestrator = Orchestrator(db, {}, {})
    orchestrator.register_task(_ExtractContainer())      # running is False

    assert db.get_pending_tasks("extract_container") == []

    orchestrator._backfill_tasks()
    assert db.get_pending_tasks("extract_container") == [
        ("/in/box.zip", "extract_container")]
