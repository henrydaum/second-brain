"""Changing an ignore setting must take the data out of the database.

``ignored_folders`` and ``ignored_extensions`` decide what may be indexed, and
the machinery to act on a change was already all present: a config write
triggers ``watcher.rescan()``, and a file the rules now exclude fails
``_is_valid_file``, so it drops out of ``disk_files`` and the scan's ghost
cleanup should remove it exactly as if it had been deleted from disk. It never
happened, for three separate reasons — and each of them fails *silently*: the
write succeeds, ``/config`` reports the new value, and nothing is pruned.

1. **Nothing normalized the settings.** ``config_manager._normalize_list``
   coerces the value to a list and never touches the items. So ``log`` and
   ``.LOG`` were compared against ``Path.suffix.lower()``, which always carries
   a dot and is always lowercase, and a folder entered as a full path was
   compared against individual path *components*. Both match nothing.
2. **Ghost cleanup was the last statement of an unguarded walk on an unguarded
   daemon thread.** One unreadable file ended the thread before the sweep — the
   same file on every boot, which is why restarting did not help either.
3. **Removal destroys the evidence it would need to retry.**
   ``on_file_deleted`` cleans the tables of *currently registered* tasks and
   then drops the ``files`` row regardless of what it managed to clean, so a
   file removed while its pipeline package was uninstalled stranded that
   package's rows past the reach of any later prune.

The three are independent, so each gets its own tests: fixing the normalization
alone still leaves a prune nothing reaches, and fixing the thread alone still
leaves a setting that matches no file.
"""

import os
from pathlib import Path

import pytest

from pipeline.database import Database
from pipeline.ignore_rules import IgnoreRules
from pipeline.orchestrator import Orchestrator
from pipeline.watcher import Watcher
from plugins.native.task import BaseTask


@pytest.fixture
def db(tmp_path):
    """A database carrying one task-output table, like a live install."""
    database = Database(str(tmp_path / "pipeline.db"))
    database.conn.execute(
        "CREATE TABLE extracted_text (path TEXT PRIMARY KEY, content TEXT)")
    database.conn.commit()
    return database


class _ExtractText(BaseTask):
    """Stands in for the store's ``extract_text``: it owns ``extracted_text``."""

    name = "extract_text"
    modalities = ["text"]
    reads = []
    writes = ["extracted_text"]
    output_schema = ""


class _RecordingOrchestrator:
    """Just enough orchestrator to see what the watcher asked to remove."""

    def __init__(self):
        self.discovered = []
        self.deleted = []

    def on_file_discovered(self, path, ext, modality):
        self.discovered.append(path)

    def on_file_deleted(self, path):
        self.deleted.append(path)


def _indexed(db, path, source="watched"):
    """Index a file and give it a row of derived data to lose."""
    p = Path(path)
    db.upsert_file(path=path, file_name=p.name, extension=p.suffix.lower(),
                   modality="text", mtime=1.0, source=source)
    db.conn.execute("INSERT INTO extracted_text VALUES (?, ?)", (path, "derived"))
    db.conn.commit()
    return path


def _derived(db):
    return sorted(r["path"] for r in db.conn.execute("SELECT path FROM extracted_text"))


def _cleaning_watcher(db, config):
    """A watcher wired to a real orchestrator, so a prune really cleans tables."""
    orchestrator = Orchestrator(db, {}, {})
    orchestrator.register_task(_ExtractText())
    return Watcher(orchestrator=orchestrator, db=db, config=config)


# ── What a setting means ─────────────────────────────────────────────

@pytest.mark.parametrize("entered", ["log", ".log", ".LOG", " .Log ", "LOG"])
def test_an_extension_is_read_the_way_a_person_types_it(entered):
    """The decisive normalization.

    The other side of this comparison is ``Path.suffix.lower()``, which always
    carries a dot and is always lowercase. Every spelling above is a reasonable
    thing to type into a free-text JSON list, and only one of them used to work.
    """
    rules = IgnoreRules.from_config({"ignored_extensions": [entered]})

    assert rules.ignores_extension(".log")
    assert rules.excludes("/vault/notes.log")
    assert not rules.excludes("/vault/notes.md")


def test_an_entry_that_names_nothing_is_dropped():
    """Blank entries are what a half-edited JSON list looks like."""
    rules = IgnoreRules.from_config({"ignored_extensions": ["", "  ", ".log"]})

    assert rules.extensions == frozenset({".log"})


def test_a_bare_folder_name_still_matches_any_component():
    """The original behaviour, and the shape of every default."""
    rules = IgnoreRules.from_config({"ignored_folders": ["node_modules"]})

    assert rules.excludes(str(Path("/code/app/node_modules/pkg/index.js")))
    assert not rules.excludes(str(Path("/code/app/src/index.js")))


def test_a_folder_entered_as_a_path_matches_by_location():
    """The setting is described as "folder names", but the field takes any text.

    A full path compared against individual path components can never match, so
    this was a setting that accepted the value and did nothing with it.
    """
    target = str(Path("/vault/Personal/Taxes"))
    rules = IgnoreRules.from_config({"ignored_folders": [target]})

    assert rules.excludes(str(Path("/vault/Personal/Taxes/2024/return.pdf")))
    assert not rules.excludes(str(Path("/vault/Personal/Notes/return.pdf")))


def test_a_sibling_sharing_a_prefix_is_not_swept_in():
    """``/vault/Archive`` must not claim ``/vault/Archived``.

    A bare ``startswith`` is the obvious way to write the location match and is
    wrong in the direction that deletes somebody's data.
    """
    rules = IgnoreRules.from_config(
        {"ignored_folders": [str(Path("/vault/Archive"))]})

    assert rules.excludes(str(Path("/vault/Archive/old.md")))
    assert not rules.excludes(str(Path("/vault/Archived/old.md")))


def test_the_files_own_name_is_not_read_as_a_folder():
    """``excludes`` asks the folder rules about the *parent*, deliberately.

    Applied to the whole path they would read the basename as a component too,
    so a file literally called ``node_modules`` would be excluded by the
    shipped defaults.
    """
    rules = IgnoreRules.from_config({"ignored_folders": ["node_modules"]})

    assert not rules.excludes(str(Path("/code/app/node_modules")))


def test_hidden_folders_follow_their_own_setting():
    hidden = str(Path("/vault/.git/config.md"))

    assert IgnoreRules.from_config({}).excludes(hidden)
    assert not IgnoreRules.from_config({"skip_hidden_folders": False}).excludes(hidden)


def test_a_relative_path_is_not_hidden_by_its_own_parent():
    """``Path("notes.md").parent`` is ``Path(".")``, whose one part starts with
    a dot. Read naively that makes every bare filename a hidden file."""
    assert not IgnoreRules.from_config({}).excludes("notes.md")


# ── Pruning the index ────────────────────────────────────────────────

def test_a_newly_ignored_folder_takes_its_derived_data_with_it(db):
    """The reported bug, end to end: the row *and* what was extracted from it."""
    kept = _indexed(db, str(Path("/vault/Notes/keep.md")))
    dropped = _indexed(db, str(Path("/vault/Archive/old.md")))

    watcher = _cleaning_watcher(db, {"ignored_folders": ["Archive"],
                                     "skip_hidden_folders": False})
    watcher._prune_by_rules()

    assert list(db.get_all_files()) == [kept]
    assert _derived(db) == [kept]
    assert dropped not in db.get_all_files()


def test_a_newly_ignored_extension_prunes_the_same_way(db):
    """Entered as ``LOG`` — no dot, wrong case — because that is the spelling
    that silently did nothing, and the one a person is most likely to type."""
    kept = _indexed(db, str(Path("/vault/Notes/keep.md")))
    _indexed(db, str(Path("/vault/Notes/debug.log")))

    watcher = _cleaning_watcher(db, {"ignored_extensions": ["LOG"],
                                     "skip_hidden_folders": False})
    watcher._prune_by_rules()

    assert list(db.get_all_files()) == [kept]
    assert _derived(db) == [kept]


def test_a_container_child_under_an_ignored_folder_goes_too(db):
    """Ghost cleanup cannot reach these, which is why the prune is its own loop.

    ``get_watched_file_state`` is ``source='watched'`` only, and a container's
    children live under a temp ``extract_dir`` outside every watch dir — so a
    disk diff widened to cover them would call all of them ghosts. The rule
    loop reads the whole table safely because it only deletes on a match.
    """
    child = _indexed(db, str(Path("/tmp/extract/Archive/inner.md")),
                     source="container")

    watcher = _cleaning_watcher(db, {"ignored_folders": ["Archive"],
                                     "skip_hidden_folders": False})
    watcher._prune_by_rules()

    assert child not in db.get_all_files()
    assert _derived(db) == []


def test_the_prune_runs_even_with_nowhere_left_to_watch(db, monkeypatch):
    """``rescan`` returned early when no sync directory existed, which meant an
    unplugged drive silently cancelled a prune the user had just asked for.

    Pruning is a question about config; it does not need the disk.
    """
    dropped = _indexed(db, str(Path("/vault/Archive/old.md")))
    watcher = _cleaning_watcher(db, {"ignored_folders": ["Archive"],
                                     "sync_directories": ["/nowhere/at/all"],
                                     "skip_hidden_folders": False})
    # The walk is not the subject and has nothing to walk.
    monkeypatch.setattr(Watcher, "_initial_scan", lambda self, dirs: None)

    watcher._sync_worker(valid_dirs=[])

    assert dropped not in db.get_all_files()


def test_rescan_refreshes_the_rules(db, tmp_path, monkeypatch):
    """``__init__`` and ``rescan`` each used to snapshot the same four keys.

    Two places to keep in step, and the prune is worthless if it runs against
    the settings as they were at boot.
    """
    config = {"sync_directories": [str(tmp_path)], "ignored_extensions": []}
    watcher = Watcher(orchestrator=_RecordingOrchestrator(), db=db, config=config)
    assert not watcher.rules.ignores_extension(".log")

    # A live config write mutates the same dict the watcher holds.
    config["ignored_extensions"] = ["LOG"]
    monkeypatch.setattr(Watcher, "_sync_worker", lambda self, dirs: None)
    watcher.rescan()

    assert watcher.rules.ignores_extension(".log")


# ── The scan can no longer starve the prune ──────────────────────────

def _scanning_watcher(db, monkeypatch, orchestrator):
    import pipeline.watcher as watcher_module
    monkeypatch.setattr(watcher_module, "get_supported_extensions", lambda: {".md"})
    monkeypatch.setattr(watcher_module, "get_modality", lambda ext: "text")
    return Watcher(orchestrator=orchestrator, db=db,
                   # pytest's tmp root can carry a dot-prefixed part.
                   config={"skip_hidden_folders": False})


def test_a_clean_walk_still_removes_a_ghost(db, tmp_path, monkeypatch):
    """The control. Everything below suppresses this, so it has to work first."""
    orchestrator = _RecordingOrchestrator()
    gone = str(tmp_path / "gone.md")
    db.upsert_file(path=gone, file_name="gone.md", extension=".md",
                   modality="text", mtime=1.0)

    _scanning_watcher(db, monkeypatch, orchestrator)._initial_scan([str(tmp_path)])

    assert orchestrator.deleted == [gone]


def test_a_walk_that_could_not_read_everything_deletes_nothing(db, tmp_path,
                                                               monkeypatch):
    """``os.walk`` swallows directory errors by default, so an unreadable
    subtree reads as an empty one — and an empty subtree is exactly what a
    deleted subtree looks like. Ghost cleanup would delete all of it."""
    import pipeline.watcher as watcher_module
    orchestrator = _RecordingOrchestrator()
    present = str(tmp_path / "present.md")
    db.upsert_file(path=present, file_name="present.md", extension=".md",
                   modality="text", mtime=1.0)

    def failing_walk(top, onerror=None):
        onerror(OSError(13, "Permission denied", top))
        return iter(())

    watcher = _scanning_watcher(db, monkeypatch, orchestrator)
    monkeypatch.setattr(watcher_module.os, "walk", failing_walk)
    watcher._initial_scan([str(tmp_path)])

    assert orchestrator.deleted == []


def test_an_unreadable_file_neither_ends_the_scan_nor_is_deleted(db, tmp_path,
                                                                 monkeypatch):
    """``os.path.getmtime`` was unguarded and ghost cleanup was the last thing
    in the function, so one ``PermissionError`` skipped the whole sweep — the
    same file on every boot, which is what survived a restart.

    The file exists, so it must also not be mistaken for a deleted one.
    """
    import pipeline.watcher as watcher_module
    orchestrator = _RecordingOrchestrator()
    locked = tmp_path / "locked.md"
    locked.write_text("x")
    db.upsert_file(path=str(locked), file_name="locked.md", extension=".md",
                   modality="text", mtime=1.0)

    real_getmtime = os.path.getmtime

    def picky(path):
        if str(path).endswith("locked.md"):
            raise PermissionError(13, "denied")
        return real_getmtime(path)

    watcher = _scanning_watcher(db, monkeypatch, orchestrator)
    monkeypatch.setattr(watcher_module.os.path, "getmtime", picky)
    watcher._initial_scan([str(tmp_path)])

    assert orchestrator.deleted == []


def test_a_failing_scan_does_not_stop_the_prune(db, monkeypatch):
    """The ordering that makes the fix hold: config first, disk second.

    Both used to live in one unguarded function with the prune's ancestor at
    the bottom, so anything the walk raised took the prune with it.
    """
    dropped = _indexed(db, str(Path("/vault/Archive/old.md")))

    def explode(self, dirs):
        raise OSError("the disk is on fire")

    monkeypatch.setattr(Watcher, "_initial_scan", explode)
    watcher = _cleaning_watcher(db, {"ignored_folders": ["Archive"],
                                     "skip_hidden_folders": False})
    watcher._sync_worker(valid_dirs=["/somewhere"])  # must not raise

    assert dropped not in db.get_all_files()


# ── Rows nothing else can reach ──────────────────────────────────────

def test_an_orphaned_output_row_is_swept(db):
    """``on_file_deleted`` drops the ``files`` row regardless of what it cleaned.

    A file removed while its pipeline package was uninstalled left rows behind
    with no ``files`` entry — and with no ``files`` entry the path can never be
    a ghost again, so no prune can name it. This is the only thing that can.
    """
    live = _indexed(db, str(Path("/vault/Notes/keep.md")))
    db.conn.execute("INSERT INTO extracted_text VALUES (?, ?)",
                    (str(Path("/vault/Notes/stranded.md")), "orphan"))
    db.conn.commit()

    assert db.sweep_orphaned_output_rows() == {"extracted_text": 1}
    assert _derived(db) == [live]


def test_an_empty_files_table_sweeps_nothing(db):
    """A fresh database before its first scan is indistinguishable here from one
    whose every file was removed, so the sweep refuses to guess."""
    db.conn.execute("INSERT INTO extracted_text VALUES ('/vault/a.md', 'x')")
    db.conn.commit()

    assert db.sweep_orphaned_output_rows() == {}
    assert db.conn.execute("SELECT COUNT(*) FROM extracted_text").fetchone()[0] == 1


def test_the_kernels_own_tables_are_never_swept(db):
    """``files`` and ``task_queue`` carry a path column and are not task output."""
    tables = db.path_keyed_output_tables()

    assert "files" not in tables
    assert "task_queue" not in tables
    assert "extracted_text" in tables


def test_the_kernel_denylist_still_matches_the_schema(tmp_path):
    """The sweep tells kernel tables from task output by an explicit list, and a
    new kernel table carrying a path column would silently join the sweep.

    Derived from a fresh database rather than restated, because the failure is
    deletion of rows nothing was supposed to touch — and it would show up as
    missing data long after the commit that caused it.
    """
    fresh = Database(str(tmp_path / "fresh.db"))
    with_path = {
        name for name in (
            r["name"] for r in fresh.conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")
        )
        if any(c["name"] == "path"
               for c in fresh.conn.execute(f"PRAGMA table_info({name})"))
    }

    assert with_path == set(Database._KERNEL_PATH_TABLES)
    assert fresh.path_keyed_output_tables() == []


def test_an_fts_index_is_never_deleted_from_directly(db):
    """``lexical_index`` is external-content FTS5 over ``lexical_content``,
    maintained by triggers. Deleting from the content table de-indexes it
    correctly; deleting from the index itself corrupts it."""
    db.conn.executescript("""
        CREATE TABLE lexical_content (path TEXT, chunk_index INTEGER, content TEXT);
        CREATE VIRTUAL TABLE lexical_index
            USING fts5(content, content=lexical_content, content_rowid=rowid);
    """)
    db.conn.commit()

    tables = db.path_keyed_output_tables()

    assert "lexical_content" in tables
    assert "lexical_index" not in tables


# ── Spelling a folder the way a person actually types it ─────────────

_PHOTOS = "Users/henry/My Drive/_Photos and Media/Photos"
_UNDER = str(Path("/Users/henry/My Drive/_Photos and Media/Photos/Misc/a.jpg"))


def _folders(*entries):
    return IgnoreRules.from_config({"ignored_folders": list(entries),
                                    "skip_hidden_folders": False})


def test_a_path_missing_its_leading_slash_still_matches():
    """The entry a person types when they drop the root.

    Nothing guesses the missing "/" — that would be inventing a root. An entry
    that is not absolute is simply not anchored, so its segments match wherever
    they run consecutively, which includes the absolute place that was meant.
    """
    assert _folders(_PHOTOS).excludes(_UNDER)


def test_a_trailing_separator_does_not_change_the_meaning():
    assert _folders("/" + _PHOTOS + "/").excludes(_UNDER)


def test_repeated_separators_collapse():
    assert _folders("/Users//henry//My Drive").excludes(_UNDER)


def test_the_foreign_separator_is_accepted():
    """A path pasted from the other platform still means what it looks like.

    The cost is naming a POSIX folder that contains a literal backslash, which
    is a legal filename character and not something anybody types into a
    settings field on purpose.
    """
    foreign = "\\" if os.sep == "/" else "/"
    entry = foreign + foreign.join(["Users", "henry", "My Drive"])

    assert _folders(entry).excludes(_UNDER)


@pytest.mark.skipif(os.name == "nt", reason="a leading // is a UNC host on Windows")
def test_a_doubled_leading_separator_collapses_on_posix():
    """POSIX normpath keeps exactly two leading slashes — a real distinction in
    the standard, and never one an indexed path carries."""
    assert _folders("//Users/henry/My Drive").excludes(_UNDER)


@pytest.mark.skipif(os.name != "nt", reason="UNC paths only exist on Windows")
def test_a_unc_host_survives_on_windows():
    """The two leading separators mean a host here, not a doubled slash, so
    collapsing them would rewrite the entry into a different location.

    A share *root* is still dropped, like every other root: naming one excludes
    an entire volume, and the way to stop indexing a volume is to remove it
    from ``sync_directories``, not to ignore its root.
    """
    # Built rather than written, so the literal cannot be mangled in transit.
    share = os.sep * 2 + "server" + os.sep + "share"

    assert _folders(share).folders == ()

    rule = _folders(share + os.sep + "Archive").folders[0]
    assert rule.anchored
    assert rule.segments[0].startswith(os.sep * 2)


def test_an_anchored_entry_stays_anchored():
    """Absolute means "starting at the root", not "these segments somewhere"."""
    assert not _folders(str(Path("/vault/Archive"))).excludes(
        str(Path("/elsewhere/vault/Archive/old.md")))


def test_a_relative_run_must_be_consecutive():
    """Otherwise "Photos/Misc" would claim every path holding both words."""
    assert _folders("Photos/Misc").excludes(str(Path("/a/Photos/Misc/x.jpg")))
    assert not _folders("Photos/Misc").excludes(str(Path("/a/Photos/b/Misc/x.jpg")))


def test_a_bare_name_is_just_the_one_segment_case():
    """The two rules that used to be separate. A bare name is unanchored and one
    segment long, which is why it matches any component — nothing special."""
    rule = _folders("node_modules").folders[0]

    assert rule.anchored is False
    assert len(rule.segments) == 1


def test_the_glob_spelling_of_an_extension_works():
    """``*.log`` is what people reach for first, and it named nothing."""
    rules = IgnoreRules.from_config({"ignored_extensions": ["*.log"]})

    assert rules.excludes("/vault/debug.log")


@pytest.mark.parametrize("entry", ["", "   ", ".", "/", "//", "\\", "./"])
def test_an_entry_naming_no_folder_becomes_no_rule(entry):
    """The one mistake that is total, so the one worth refusing outright.

    A root respells to a single root segment, and as an ordinary unanchored
    rule that matches the root segment *every* absolute path begins with — so
    one stray "/" left in the settings field would have excluded the whole
    index, which the prune would then delete. Everything here names no folder
    and must produce no rule at all.
    """
    rules = _folders(entry)

    assert rules.folders == ()
    assert not rules.excludes(_UNDER)
