"""Memory lives in the agent's own tree, and got there safely.

Memory is the one thing the system's own prompt asks the agent to keep
current about itself. While it sat at ``DATA_DIR/memory`` that request was
unanswerable without a dialog: the folder was outside every scratch root, so
saving a topic the prompt had just asked for raised an approval — and the
failure mode of a refused memory write is silence, since nothing downstream
knows a note was meant to exist.

Two things are pinned. The location, as a fact the kernel's prompt section and
the store's memory tool have to agree on. And the migration off the old
spelling, which unlike the attachment cache has no config half — the whole of
it is moving files without ever guessing at a collision.
"""

import pytest

import migrations
import trees
from tests.support import retarget_trees


@pytest.fixture
def layout(tmp_path, monkeypatch):
    """Three empty local trees at a throwaway location."""
    return retarget_trees(monkeypatch, tmp_path)


# ── Where it is ───────────────────────────────────────────────────────

def test_the_agent_may_write_memory_without_asking(layout):
    """The whole reason for the move, as one assertion.

    ``_freely_writable`` backs ``fs.write``/``move``/``delete`` alike, so the
    grant covers rewriting the index and forgetting a stale topic, not just
    creating a new one.
    """
    from sandbox import policy

    memory = layout["workspace"] / "memory"
    memory.mkdir(parents=True)
    index = memory / "MEMORY.md"
    index.write_text("- [proj](proj.md) - the project", encoding="utf-8")

    assert policy._freely_writable(index)
    assert "the agent's own tree" in policy._write_reason(index, "delete")


def test_the_prompt_reads_it_from_the_workspace(layout, monkeypatch):
    """The kernel's one memory fact, pinned against the tree it now sits in."""
    import paths
    from agent.system_prompt import _agent_memory

    monkeypatch.setattr(paths, "DATA_DIR", layout["workspace"].parent)
    memory = layout["workspace"] / "memory"
    memory.mkdir(parents=True)
    (memory / "MEMORY.md").write_text("- [proj](proj.md) - the project",
                                      encoding="utf-8")

    assert "- [proj](proj.md) - the project" in _agent_memory()


def test_it_is_not_a_declared_root(layout):
    """A root is a shape every tree repeats; this belongs to exactly one.

    Declaring it would put an empty ``memory/`` into ``bundled/`` and
    ``installed/``, claiming those trees might hold one — and a memory the
    package store ships or the app is bundled with is not a thing.
    """
    assert "memory" not in {root.name for root in trees.ROOTS}

    trees.materialize()

    assert not (layout["installed"] / "memory").exists()
    assert not (layout["bundled"] / "memory").exists()


# ── Getting there ─────────────────────────────────────────────────────

def _data_dir(tmp_path):
    """A DATA_DIR holding the old layout: memory beside the workspace tree."""
    root = tmp_path / "data"
    old = root / "memory"
    old.mkdir(parents=True)
    (old / "MEMORY.md").write_text("- [proj](proj.md) - the project",
                                   encoding="utf-8")
    (old / "proj.md").write_text("old body", encoding="utf-8")
    (root / "workspace").mkdir()
    return root, old


def test_migration_moves_the_index_and_its_topics(tmp_path):
    root, old = _data_dir(tmp_path)

    done = migrations.migrate(root)

    moved = root / "workspace" / "memory"
    assert moved.joinpath("MEMORY.md").read_text(encoding="utf-8").startswith("- [proj]")
    assert moved.joinpath("proj.md").read_text(encoding="utf-8") == "old body"
    assert not old.exists()
    assert any("memory/ -> workspace/memory/" in line for line in done)


def test_migration_keeps_a_colliding_topic_and_says_so(tmp_path):
    """Never guess. Two files under one topic name are two different notes,
    and merging them is the one outcome nobody can undo."""
    root, old = _data_dir(tmp_path)
    new = root / "workspace" / "memory"
    new.mkdir(parents=True)
    (new / "proj.md").write_text("new body", encoding="utf-8")

    done = migrations.migrate(root)

    assert (new / "proj.md").read_text(encoding="utf-8") == "new body"
    assert (old / "proj.md").read_text(encoding="utf-8") == "old body"
    assert any(line.startswith("!") and "already holds" in line
               for line in done)


def test_migration_is_idempotent(tmp_path):
    """It runs every boot, so a second pass must be a no-op."""
    root, _old = _data_dir(tmp_path)

    assert migrations.migrate(root)
    assert migrations.migrate(root) == []


def test_a_fresh_install_has_nothing_to_do(tmp_path):
    root = tmp_path / "data"
    (root / "workspace" / "memory").mkdir(parents=True)

    assert migrations.migrate(root) == []
