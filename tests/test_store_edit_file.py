"""Tests for the store file tools, now that they are sandboxed.

read_file + edit_file + the file_reads helper are one package: the helper
carries the read-before-edit bookkeeping both tools share, so all three
migrated together.

The interesting property under test is a *deletion*. edit_file used to run its
own approval dialog with its own path exemptions; every effect is now a
Request, so ``sandbox/policy.py`` is the only thing that decides, and the tool
contains no authorization code at all. These drive the real bridge and the
real gate rather than a stand-in, because "the kernel decides" is only true if
the kernel is actually in the loop.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from sandbox.bridge import adapt, configure
from sandbox.facade import Sandbox
from sandbox.guest.requests import FS_DELETE, FS_WRITE

_REPO = Path(__file__).resolve().parents[1]

_FILES = ("tools/tool_edit_file.py", "tools/tool_read_file.py",
          "tools/helpers/file_reads.py")


def _store_source(rel: str) -> str | None:
    for ref in ("store", "origin/store"):
        proc = subprocess.run(
            ["git", "-C", str(_REPO), "show", f"{ref}:{rel}"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            encoding="utf-8", check=False)
        if proc.returncode == 0:
            return proc.stdout
    return None


@pytest.fixture()
def env(tmp_path):
    """The two tools loaded through the bridge, editing inside tmp_path.

    ``root_dir`` on the context is what ``paths.get("project")`` resolves to,
    so the tools' own root confinement lands on the temporary tree.
    """
    sources = {rel: _store_source(rel) for rel in _FILES}
    if any(v is None for v in sources.values()):
        pytest.skip("file tools not present on a local store ref")

    tree = tmp_path / "tree"
    (tree / "tools" / "helpers").mkdir(parents=True)
    for rel, src in sources.items():
        (tree / rel).write_text(src, encoding="utf-8")

    work = tmp_path / "work"
    work.mkdir()

    # A session that can really hold plugin state: read-before-edit lives
    # there, and a stub would silently disable the whole gate.
    state: dict = {}
    runtime = SimpleNamespace(
        sessions={"k": SimpleNamespace(plugin_state=state)},
        get_session_plugin_state=lambda key, ns: state.get(ns),
        update_session_plugin_state=(
            lambda key, ns, value: state.__setitem__(ns, value)),
    )
    context = SimpleNamespace(session_key="k", runtime=runtime,
                              root_dir=work, config={})

    asked: list = []
    verdict = {"allow": True}

    def approve(chain, request, decision) -> bool:
        asked.append((request.type, request.args.get("path")))
        return verdict["allow"]

    sandbox = Sandbox()
    sandbox.interpreter._approve = approve
    configure(sandbox)

    def _tool(stem, name):
        """The adapter the bridge built, found the way discovery finds it.

        The adapter subclasses the native base under a generated name
        (``SandboxedEditFile``), so this looks for the marker rather than
        guessing what the class ended up being called.
        """
        module = adapt(tree / "tools" / f"{stem}.py")
        assert module is not None, f"{stem} did not load as a sandboxed plugin"
        cls = next(v for v in vars(module).values()
                   if isinstance(v, type) and getattr(v, "_sandboxed", False)
                   and getattr(v, "name", "") == name)
        return cls()

    edit_tool = _tool("tool_edit_file", "edit_file")
    read_tool = _tool("tool_read_file", "read_file")

    target = work / "sample.py"
    target.write_text("alpha\nbeta\ngamma\nbeta\n", encoding="utf-8")

    def edit(**kwargs):
        kwargs.setdefault("justification", "test")
        return edit_tool.run(context, **kwargs)

    def read(path=None):
        return read_tool.run(context, path=str(path or target))

    return SimpleNamespace(edit=edit, read=read, target=target, work=work,
                           asked=asked, verdict=verdict, context=context)


# ── the tool no longer contains any authorization ─────────────────────

def test_the_tool_carries_no_approval_code():
    """The headline of the migration: policy left the plugin.

    Every exemption this used to make — scratch, the agent's plugin tree, a
    root-file warning, a config switch to turn one off — was policy living in
    a plugin, and each was a chance to disagree with the kernel. The Request
    catalogue answers all of it now, so none of it should have survived.
    """
    source = _store_source("tools/tool_edit_file.py")
    if source is None:
        pytest.skip("edit_file not present on a local store ref")
    for gone in ("approve_command", "_is_scratch", "_is_authoring",
                 "_is_root_file", "scratch_no_approval", "config_settings"):
        assert gone not in source, f"{gone} survived the migration"


def test_the_kernel_is_asked_and_can_refuse(env):
    """The gate is really in the loop, not stubbed past."""
    made = env.work / "new.py"
    env.edit(operation="create", path=str(made), content="x\n")
    assert (FS_WRITE, str(made)) in env.asked


def test_a_refusal_tells_the_model_to_stop(env):
    """A denial is not a breakage, and retrying one wastes a turn."""
    env.verdict["allow"] = False
    out = env.edit(operation="create", path=str(env.work / "nope.py"),
                   content="x\n")
    assert out.success is False
    assert "STOP" in out.error
    assert not (env.work / "nope.py").exists()


# ── behaviour that had to survive the migration ───────────────────────

def test_create_and_read_round_trip(env):
    made = env.work / "fresh.py"
    assert env.edit(operation="create", path=str(made),
                    content="hello\n").success
    assert made.read_text(encoding="utf-8") == "hello\n"
    assert "hello" in env.read(made).llm_summary


def test_create_refuses_an_existing_file(env):
    out = env.edit(operation="create", path=str(env.target), content="x")
    assert out.success is False and "already exists" in out.error


def test_read_before_edit_is_enforced(env):
    out = env.edit(operation="replace", path=str(env.target),
                   old_text="alpha", new_text="omega")
    assert out.success is False and "read_file" in out.error


def test_replace_after_reading_works(env):
    env.read()
    assert env.edit(operation="replace", path=str(env.target),
                    old_text="alpha", new_text="omega").success
    assert "omega" in env.target.read_text(encoding="utf-8")


def test_a_stale_read_is_caught(env):
    """The mtime bookkeeping has to survive crossing the boundary."""
    env.read()
    env.target.write_text("alpha\nbeta\ngamma\nbeta\nDELTA\n", encoding="utf-8")
    out = env.edit(operation="replace", path=str(env.target),
                   old_text="beta", new_text="x")
    assert out.success is False and "changed on disk" in out.error


def test_ambiguous_replace_reports_the_lines(env):
    env.read()
    out = env.edit(operation="replace", path=str(env.target),
                   old_text="beta", new_text="x")
    assert out.success is False and "lines 2, 4" in out.error


def test_replace_all(env):
    env.read()
    assert env.edit(operation="replace", path=str(env.target),
                    old_text="beta", new_text="x", replace_all=True).success
    assert "beta" not in env.target.read_text(encoding="utf-8")


def test_a_missed_replace_quotes_the_closest_text(env):
    """The self-correcting error is the tool's best feature; keep it."""
    env.read()
    out = env.edit(operation="replace", path=str(env.target),
                   old_text="alpha\nbetaX", new_text="q")
    assert out.success is False
    assert "Closest match" in out.error


def test_line_number_contamination_is_flagged(env):
    env.read()
    out = env.edit(operation="replace", path=str(env.target),
                   old_text="1: alpha\n2: beta", new_text="q")
    assert out.success is False and "line-number" in out.error


def test_append_adds_to_the_end(env):
    env.read()
    assert env.edit(operation="append", path=str(env.target),
                    content="delta\n").success
    assert env.target.read_text(encoding="utf-8").endswith("delta\n")


def test_delete_works_after_a_read(env):
    doomed = env.work / "doomed.py"
    env.edit(operation="create", path=str(doomed), content="x\n")
    assert env.edit(operation="delete", path=str(doomed)).success
    assert not doomed.exists()
    assert (FS_DELETE, str(doomed)) in env.asked


def test_root_confinement_still_applies(env, tmp_path):
    """Scope, not authorization — but it should still hold."""
    out = env.edit(operation="create", path=str(tmp_path / "outside.py"),
                   content="x")
    assert out.success is False and "outside allowed roots" in out.error


def test_justification_is_required(env):
    out = env.edit(operation="create", path=str(env.work / "j.py"),
                   content="x", justification="")
    assert out.success is False and "justification" in out.error
