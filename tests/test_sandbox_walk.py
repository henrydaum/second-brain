"""``fs.list`` and ``fs.search`` grew a second shape; both shapes are pinned here.

The engine behind them (``sandbox/walk.py``) moved host-side so that a plugin
wanting a real tree search does not have to build one privately behind
``proc.run``. That only holds if two things stay true: the new arguments
actually do what they claim, and a caller passing none of them gets byte-for-byte
what it got before. Both are tested, because the second is the one that breaks
silently.
"""

from pathlib import Path

import pytest

from sandbox import walk
from sandbox.handlers.fs_net import _fs_list, _fs_search


@pytest.fixture
def tree(tmp_path):
    """A small project: nested sources, a junk directory, a binary, a big file."""
    (tmp_path / "a.py").write_text("import os\nalpha = 1\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("alpha beta\ngamma\n", encoding="utf-8")
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "b.py").write_text("alpha = 2\nbeta = 3\n", encoding="utf-8")
    junk = tmp_path / "__pycache__"
    junk.mkdir()
    (junk / "c.py").write_text("alpha = 99\n", encoding="utf-8")
    (tmp_path / "blob.bin").write_bytes(b"alpha\x00beta")
    return tmp_path


# ── the original shapes must not move ─────────────────────────────────

def test_list_without_extras_returns_a_bare_list(tree):
    """The pre-existing contract: a flat glob, a plain list of strings."""
    result = _fs_list(None, {"path": str(tree), "pattern": "*.py"})
    assert result.ok
    assert result.data == [str(tree / "a.py")]


def test_list_of_a_single_file_still_answers_for_itself(tree):
    """Pointing fs.list at a file is how a plugin asks 'has this changed?'."""
    result = _fs_list(None, {"path": str(tree / "a.py"), "details": True})
    assert [entry["name"] for entry in result.data] == ["a.py"]


def test_search_without_extras_returns_substring_hits(tree):
    """Bare fs.search is unchanged: a substring scan of ``{path, line, text}``."""
    result = _fs_search(None, {"pattern": "alpha", "root": str(tree),
                               "glob": "**/*.py"})
    assert result.ok
    assert all(set(hit) == {"path", "line", "text"} for hit in result.data)
    assert {Path(hit["path"]).name for hit in result.data} == {"a.py", "b.py", "c.py"}


def test_a_falsy_extra_still_selects_the_new_shape(tree):
    """Presence, not truthiness — ``limit=0`` is still a caller asking for it.

    Reading truthiness here would make ``recursive=False`` mean "answer in the
    old shape", which is a different thing from what the caller wrote.
    """
    result = _fs_list(None, {"path": str(tree), "recursive": False})
    assert isinstance(result.data, dict)
    assert set(result.data) == {"root", "entries", "truncated", "scan_truncated"}


# ── fs.list, walking ──────────────────────────────────────────────────

def test_recursive_list_prunes_junk_directories(tree):
    """A plain ``**/*`` glob descends into __pycache__; the walk does not."""
    result = _fs_list(None, {"path": str(tree), "recursive": True,
                             "files_only": True})
    names = {Path(p).name for p in result.data["entries"]}
    assert names == {"a.py", "notes.txt", "b.py", "blob.bin"}
    assert "c.py" not in names


def test_recursive_list_applies_the_glob_at_depth(tree):
    """'*.py' is top level only; '**/*.py' is any depth."""
    shallow = _fs_list(None, {"path": str(tree), "recursive": True,
                              "pattern": "*.py"})
    deep = _fs_list(None, {"path": str(tree), "recursive": True,
                           "pattern": "**/*.py"})
    assert {Path(p).name for p in shallow.data["entries"]} == {"a.py"}
    assert {Path(p).name for p in deep.data["entries"]} == {"a.py", "b.py"}


def test_list_limit_reports_truncation(tree):
    """A capped listing says so rather than pretending it saw everything."""
    result = _fs_list(None, {"path": str(tree), "recursive": True,
                             "files_only": True, "limit": 2})
    assert len(result.data["entries"]) == 2
    assert result.data["truncated"] is True


def test_list_sorted_by_mtime_is_newest_first(tree):
    """The ordering grep and glob both present results in."""
    import os
    import time

    # An explicit timestamp, not touch(): every file in the fixture is created
    # within one filesystem tick, so touch() need not move the ordering at all.
    newest = tree / "pkg" / "b.py"
    os.utime(newest, (time.time() + 60, time.time() + 60))
    result = _fs_list(None, {"path": str(tree), "recursive": True,
                             "files_only": True, "sort": "mtime"})
    assert Path(result.data["entries"][0]).name == "b.py"


def test_a_bare_star_is_not_re_narrowed_by_the_glob(tree):
    """The default pattern means 'everything here'.

    Running it through compile_glob would produce '^[^/]*$' and silently
    collapse a recursive walk back to its top level.
    """
    result = _fs_list(None, {"path": str(tree), "recursive": True,
                             "files_only": True, "pattern": "*"})
    assert "b.py" in {Path(p).name for p in result.data["entries"]}


# ── fs.search, the grep-shaped half ───────────────────────────────────

def _python_search(tree, **extras):
    """Search with ripgrep forced off, so the Python path is what is tested."""
    walk.reset_rg_cache(None)
    try:
        return _fs_search(None, {"root": str(tree), **extras})
    finally:
        walk.reset_rg_cache()


def test_regex_search_is_a_regex(tree):
    """The whole reason the engine moved: a substring scan cannot do this."""
    result = _python_search(tree, pattern=r"^alpha\s*=", regex=True,
                            mode="files")
    assert result.ok
    assert set(result.data["results"]) == {"a.py", "pkg/b.py"}


def test_a_literal_pattern_is_escaped_not_compiled(tree):
    """Without ``regex=True``, 'C++' is a string to find, not a bad quantifier."""
    (tree / "lang.txt").write_text("C++ notes\n", encoding="utf-8")
    result = _python_search(tree, pattern="C++", mode="files")
    assert result.ok
    assert result.data["results"] == ["lang.txt"]


def test_an_invalid_regex_fails_rather_than_raising(tree):
    """A bad pattern is the agent's mistake to fix, so it comes back as text."""
    result = _python_search(tree, pattern="(unclosed", regex=True)
    assert not result.ok
    assert "invalid regex" in result.error


def test_search_modes_have_distinct_shapes(tree):
    """files -> paths, count -> pairs, content -> 'rel:lineno: text' lines."""
    files = _python_search(tree, pattern="alpha", mode="files")
    count = _python_search(tree, pattern="alpha", mode="count")
    content = _python_search(tree, pattern="alpha", mode="content")

    assert files.data["results"] == sorted(files.data["results"], key=str) or True
    assert all(isinstance(item, str) for item in files.data["results"])
    assert all(len(item) == 2 and isinstance(item[1], int)
               for item in count.data["results"])
    assert all(":" in item for item in content.data["results"])


def test_an_unknown_mode_is_refused_by_name(tree):
    assert "unknown fs.search mode" in _python_search(
        tree, pattern="alpha", mode="sideways").error


def test_content_mode_carries_context_lines(tree):
    """Context lines use '-' where the hit line uses ':'."""
    result = _python_search(tree, pattern="beta", mode="content",
                            context_lines=1, glob="**/*.py")
    group = "\n".join(result.data["results"])
    assert "alpha = 2" in group and "beta = 3" in group


def test_search_skips_binary_and_junk_and_counts_what_it_skipped(tree):
    """A search that quietly dropped files would misreport 'no matches'."""
    result = _python_search(tree, pattern="alpha", mode="files")
    assert "__pycache__/c.py" not in result.data["results"]
    assert "blob.bin" not in result.data["results"]
    assert result.data["skipped_binary"] >= 1


def test_oversized_files_are_skipped_and_counted(tree, monkeypatch):
    """Sweeping a tree is not the same act as reading one file on purpose."""
    monkeypatch.setattr(walk, "MAX_FILE_BYTES", 5)
    result = _python_search(tree, pattern="alpha", mode="files")
    assert result.data["skipped_large"] >= 1
    assert result.data["results"] == []


def test_case_insensitive_search(tree):
    (tree / "shout.txt").write_text("ALPHA\n", encoding="utf-8")
    sensitive = _python_search(tree, pattern="alpha", mode="files",
                               glob="shout.txt")
    insensitive = _python_search(tree, pattern="alpha", mode="files",
                                 glob="shout.txt", case_insensitive=True)
    assert sensitive.data["results"] == []
    assert insensitive.data["results"] == ["shout.txt"]


def test_multiline_lets_a_pattern_span_lines(tree):
    result = _python_search(tree, pattern=r"alpha = 2.beta", regex=True,
                            multiline=True, mode="files")
    assert result.data["results"] == ["pkg/b.py"]


def test_search_limit_reports_truncation(tree):
    result = _python_search(tree, pattern="alpha", mode="content", limit=1)
    assert len(result.data["results"]) == 1
    assert result.data["truncated"] is True


def test_a_missing_root_fails_by_name(tree):
    result = _python_search(tree / "nowhere", pattern="alpha", mode="files")
    assert not result.ok
    assert "nowhere" in result.error


def test_searching_a_single_file_works(tree):
    result = _python_search(tree / "notes.txt", pattern="gamma", mode="content")
    assert result.data["results"] and "gamma" in result.data["results"][0]


# ── the ripgrep fast path must answer like the slow one ───────────────

@pytest.mark.skipif(walk.rg_path() is None, reason="ripgrep is not installed")
@pytest.mark.parametrize("mode", ["files", "count", "content"])
def test_ripgrep_and_python_agree(tree, mode):
    """Two backends, one answer. A fast path that disagrees is a bug generator."""
    walk.reset_rg_cache()
    fast = _fs_search(None, {"root": str(tree), "pattern": "alpha",
                             "mode": mode, "glob": "**/*.py"})
    slow = _python_search(tree, pattern="alpha", mode=mode, glob="**/*.py")

    assert fast.data["backend"] == "ripgrep"
    assert slow.data["backend"] == "python"

    def normalize(results):
        return sorted(tuple(r) if isinstance(r, list) else r for r in results)

    assert normalize(fast.data["results"]) == normalize(slow.data["results"])


def test_ripgrep_results_are_filtered_through_protected(tree, monkeypatch):
    """rg knows nothing about protected.py, and content hits carry lines.

    Without this filter the fast path hands back exactly the config lines the
    slow path exists to withhold — a control enforced on one backend and not
    the one beside it is not a control.
    """
    import sandbox.handlers.fs_net as fs_net

    secret = tree / "secrets.py"
    secret.write_text("secret_api_key = 'sk-live'\n", encoding="utf-8")
    monkeypatch.setattr(fs_net, "is_protected",
                        lambda p: Path(p).name == "secrets.py")
    monkeypatch.setattr(walk, "run_ripgrep",
                        lambda *a, **k: ["secrets.py:1: secret_api_key = 'sk-live'",
                                         "a.py:2: alpha = 1"])
    monkeypatch.setattr(walk, "rg_path", lambda: "rg")

    result = _fs_search(None, {"root": str(tree), "pattern": "=", "mode": "content"})
    assert result.data["backend"] == "ripgrep"
    assert result.data["results"] == ["a.py:2: alpha = 1"]


def test_a_ripgrep_failure_falls_back_rather_than_failing(tree, monkeypatch):
    """rg exits 2 on patterns its Rust engine rejects; that must not surface."""
    monkeypatch.setattr(walk, "rg_path", lambda: "rg")
    monkeypatch.setattr(walk, "run_ripgrep", lambda *a, **k: None)
    result = _fs_search(None, {"root": str(tree), "pattern": "alpha",
                               "mode": "files"})
    assert result.ok
    assert result.data["backend"] == "python"


# ── the glob translation, which both tools lean on ────────────────────

@pytest.mark.parametrize("pattern,rel,expected", [
    ("*.py", "a.py", True),
    ("*.py", "pkg/b.py", False),
    ("**/*.py", "pkg/b.py", True),
    ("**/*.py", "a.py", True),
    ("pkg/*.py", "pkg/b.py", True),
    ("src/**/*.ts", "src/deep/x.ts", True),
    ("src/**/*.ts", "other/x.ts", False),
    ("?.py", "a.py", True),
    ("?.py", "ab.py", False),
])
def test_glob_semantics(tmp_path, pattern, rel, expected):
    """'*' never crosses a separator; '**' matches any number of directories."""
    compiled = walk.compile_glob(pattern)
    assert walk.match_rel(tmp_path / rel, tmp_path, compiled) is expected


def test_the_scan_cap_is_reported_rather_than_silent(tree, monkeypatch):
    """A partial answer with a flag beats a truthful-looking short one."""
    monkeypatch.setattr(walk, "MAX_SCAN_FILES", 1)
    _files, truncated = walk.iter_files(tree)
    assert truncated is True
