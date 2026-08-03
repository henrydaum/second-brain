"""The attachment cache lives in the agent's own tree, and got there safely.

A file arriving over a transport is the one piece of user data the agent is
unambiguously meant to work on. While the cache sat at
``DATA_DIR/attachment_cache`` that was half true: ``policy._scratch_roots``
named the folder, so *writing* an incoming file was free, and reading, moving
or deleting one raised a dialog — the folder holding the file a person had
just handed the agent was the folder it could do least with.

Two things are pinned here. The location itself, as a fact several modules
have to agree on (the policy grant, the pipeline's priority root, the default
sync directory). And the migration off the old spelling, whose silent failure
mode is the interesting one: the folder is a ``sync_directory``, so moving it
without repointing config leaves the watcher aimed at a directory that no
longer exists and incoming attachments unindexed, neither of which surfaces as
an error anybody reads.
"""

import json

import pytest

import migrations
import trees
from tests.support import retarget_trees


@pytest.fixture
def layout(tmp_path, monkeypatch):
    """Three empty local trees at a throwaway location."""
    return retarget_trees(monkeypatch, tmp_path)


# ── Where it is ───────────────────────────────────────────────────────

def test_the_cache_is_inside_the_workspace_tree(layout):
    """The whole move, as one assertion."""
    cache = trees.attachment_cache()

    assert cache.parent == layout["workspace"]
    assert cache.is_dir()


def test_asking_for_it_creates_it(layout):
    """It is a ``sync_directory`` by default, and the watcher drops a directory
    that does not exist — so a fresh install with an uncreated cache starts
    with no valid sync directories at all."""
    assert not (layout["workspace"] / "attachments").exists()

    trees.attachment_cache()

    assert (layout["workspace"] / "attachments").is_dir()


def test_it_is_not_a_declared_root(layout):
    """A root is a shape every tree repeats; this belongs to exactly one.

    Declaring it would put an empty ``attachments/`` into ``bundled/`` and
    ``installed/`` claiming those trees might hold one, and nothing globs it.
    """
    assert "attachments" not in {root.name for root in trees.ROOTS}

    trees.attachment_cache()
    trees.materialize()

    assert not (layout["installed"] / "attachments").exists()
    assert not (layout["bundled"] / "attachments").exists()


def test_the_agent_may_work_on_an_attachment_without_asking(layout):
    """Not just write it. The old grant was write-only and this is the point.

    ``_freely_writable`` backs ``fs.write``/``move``/``delete`` alike, so one
    answer covers every verb the agent needs on a file it was handed.
    """
    from sandbox import policy

    incoming = trees.attachment_cache() / "1700000000_notes.pdf"
    incoming.write_bytes(b"%PDF-1.4\n")

    assert policy._freely_writable(incoming)
    assert "the agent's own tree" in policy._write_reason(incoming, "delete")


def test_the_cache_is_no_longer_a_scratch_root_of_its_own(layout):
    """It is covered by the authoring root, so the special case went rather
    than moving. A rule plus an exception that happens to agree is two things
    to keep in step."""
    from sandbox import policy

    workspace = layout["workspace"]
    assert set(policy._scratch_roots()) == {workspace / "temp", workspace}


# ── Getting there ─────────────────────────────────────────────────────

def _data_dir(tmp_path):
    """A DATA_DIR holding the old layout: a populated cache beside a tree."""
    root = tmp_path / "data"
    old = root / "attachment_cache"
    old.mkdir(parents=True)
    (old / "1700000000_report.pdf").write_bytes(b"old")
    (root / "workspace").mkdir()
    return root, old


def test_migration_moves_the_files(tmp_path):
    root, old = _data_dir(tmp_path)

    done = migrations.migrate(root)

    moved = root / "workspace" / "attachments" / "1700000000_report.pdf"
    assert moved.read_bytes() == b"old"
    assert not old.exists()
    assert any("attachment_cache/ -> workspace/attachments/" in line
               for line in done)


def test_migration_repoints_the_sync_directory(tmp_path):
    """The silent half. A config still naming the old path leaves the watcher
    pointed at a directory this very migration deleted."""
    root, old = _data_dir(tmp_path)
    config = root / "config.json"
    config.write_text(json.dumps({
        "sync_directories": [str(tmp_path / "Documents"), str(old)],
        "db_path": "kept",
    }), encoding="utf-8")

    migrations.migrate(root)

    saved = json.loads(config.read_text(encoding="utf-8"))
    assert saved["sync_directories"] == [
        str(tmp_path / "Documents"), str(root / "workspace" / "attachments")]
    # Everything else in the file is written back untouched.
    assert saved["db_path"] == "kept"


def test_migration_matches_a_differently_spelled_path(tmp_path):
    """Config paths are hand-edited and round-tripped through JSON, so the old
    entry can differ by a separator, a trailing slash, or Windows case."""
    root, old = _data_dir(tmp_path)
    config = root / "config.json"
    config.write_text(json.dumps({
        "sync_directories": [str(old).replace("\\", "/") + "/"],
    }), encoding="utf-8")

    migrations.migrate(root)

    saved = json.loads(config.read_text(encoding="utf-8"))
    assert saved["sync_directories"] == [
        str(root / "workspace" / "attachments")]


def test_migration_does_not_duplicate_an_already_listed_destination(tmp_path):
    root, old = _data_dir(tmp_path)
    new = root / "workspace" / "attachments"
    config = root / "config.json"
    config.write_text(json.dumps({
        "sync_directories": [str(old), str(new)],
    }), encoding="utf-8")

    migrations.migrate(root)

    saved = json.loads(config.read_text(encoding="utf-8"))
    assert saved["sync_directories"] == [str(new)]


def test_migration_keeps_a_colliding_name_and_says_so(tmp_path):
    """Never guess. Attachment names carry a unix timestamp, so a collision
    means two files that genuinely differ."""
    root, old = _data_dir(tmp_path)
    new = root / "workspace" / "attachments"
    new.mkdir(parents=True)
    (new / "1700000000_report.pdf").write_bytes(b"new")

    done = migrations.migrate(root)

    assert (new / "1700000000_report.pdf").read_bytes() == b"new"
    assert (old / "1700000000_report.pdf").read_bytes() == b"old"
    assert any(line.startswith("!") and "already holds" in line
               for line in done)


def test_migration_is_idempotent(tmp_path):
    """It runs every boot, so a second pass must be a no-op."""
    root, _old = _data_dir(tmp_path)
    (root / "config.json").write_text(json.dumps({"sync_directories": []}),
                                      encoding="utf-8")

    assert migrations.migrate(root)
    assert migrations.migrate(root) == []


def test_migration_leaves_an_unreadable_config_alone(tmp_path):
    """A boot step guessing at JSON is how a person's settings disappear."""
    root, _old = _data_dir(tmp_path)
    config = root / "config.json"
    config.write_text("{not json", encoding="utf-8")

    migrations.migrate(root)

    assert config.read_text(encoding="utf-8") == "{not json"
