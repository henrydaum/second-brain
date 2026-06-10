"""Memory prompt section: folder + index model.

The kernel inlines only ``memory/MEMORY.md`` (the index) and lists topic
files by name — topic bodies stay out of the prompt and are read on demand
via the store ``memory`` tool, whose own ``agent_prompt`` carries the usage
instructions (plugin guidance stays out of the kernel).
"""

import pytest

import plugins.helpers.memory_paths as memory_paths
from agent.system_prompt import _agent_memory
from pipeline.database import DEFAULT_USER_ID


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(memory_paths, "DATA_DIR", tmp_path)
    return tmp_path


def test_empty_install_shows_empty_index(data_dir):
    text = _agent_memory()
    assert "## Memory" in text
    assert "(empty)" in text


def test_index_inlined_topics_listed_not_inlined(data_dir):
    root = data_dir / "memory"
    root.mkdir()
    (root / "MEMORY.md").write_text("- [proj](proj.md) - the project", encoding="utf-8")
    (root / "proj.md").write_text("SECRET TOPIC BODY", encoding="utf-8")
    text = _agent_memory()
    assert "- [proj](proj.md) - the project" in text
    assert "proj" in text.split("Topic files:")[-1]
    assert "SECRET TOPIC BODY" not in text


def test_no_plugin_guidance_in_kernel_section(data_dir):
    (data_dir / "memory").mkdir()
    assert "`memory` tool" not in _agent_memory()


def test_memory_root_is_per_user_ready(data_dir):
    assert memory_paths.memory_root() == data_dir / "memory"
    assert memory_paths.memory_root(DEFAULT_USER_ID) == data_dir / "memory"
    assert memory_paths.memory_root(7) == data_dir / "memory" / "users" / "7"


def test_topic_path_validates_names(data_dir):
    (data_dir / "memory").mkdir()
    assert memory_paths.topic_path("project-x").name == "project-x.md"
    assert memory_paths.topic_path("notes.md").name == "notes.md"
    for bad in ("", "..", "../evil", "a/b", "MEMORY", ".hidden", "C:\\x"):
        with pytest.raises(ValueError):
            memory_paths.topic_path(bad)


def test_list_topics_excludes_index_and_other_users(data_dir):
    root = data_dir / "memory"
    (root / "users" / "7").mkdir(parents=True)
    (root / "MEMORY.md").write_text("idx", encoding="utf-8")
    (root / "a.md").write_text("a", encoding="utf-8")
    (root / "users" / "7" / "b.md").write_text("b", encoding="utf-8")
    assert [p.stem for p in memory_paths.list_topics()] == ["a"]
    assert [p.stem for p in memory_paths.list_topics(7)] == ["b"]
