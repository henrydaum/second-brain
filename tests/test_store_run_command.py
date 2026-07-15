"""Tests for the store run_command tool (``tools/tool_run_command.py``).

The package lives on the ``store`` branch, so the file is materialized from
the local store ref via ``git show`` and loaded with importlib. Skips cleanly
when no store ref is available.

Focus: the approval classifier. Read-only commands — including pipelines
composed entirely of read-only segments — auto-run; anything that can modify
state (or smuggle a second command past the whitelist via redirection,
substitution, backgrounding, or unbalanced quoting) takes the approval path.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO = Path(__file__).resolve().parents[1]
_REL = "tools/tool_run_command.py"


def _store_source() -> str | None:
    for ref in ("store", "origin/store"):
        proc = subprocess.run(
            ["git", "-C", str(_REPO), "show", f"{ref}:{_REL}"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", check=False)
        if proc.returncode == 0:
            return proc.stdout
    return None


@pytest.fixture(scope="module")
def mod(tmp_path_factory):
    """Load the tool module off the store ref."""
    src = _store_source()
    if src is None:
        pytest.skip("run_command package not present on a local store ref")
    path = tmp_path_factory.mktemp("runcmd_pkg") / "tool_run_command.py"
    path.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("tool_run_command_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ── Classifier: auto-approved read-only commands ─────────────────────

@pytest.mark.parametrize("command", [
    # search
    "rg TODO",
    "rg -n --glob *.py classify",
    "grep -r foo .",
    "findstr /s /i needle *.py",
    'find . -name "*.py"',
    "sed -n 10,40p paths.py",
    # listing / inspection
    "ls",
    "dir /b",
    "tree",
    "cat paths.py",
    "type paths.py",
    "head -50 main.pyw",
    "tail -n 20 requirements.txt",
    "wc -l requirements.txt",
    "where python",
    "pwd",
    # git
    "git status",
    "git log --oneline -20",
    "git diff HEAD~1",
    "git show --stat HEAD",
    "git branch -a",
    "git branch --show-current",
    "git blame paths.py",
    "git ls-files",
    "git rev-parse HEAD",
    # pip / versions
    "pip list",
    "pip show requests",
    "pip freeze",
    "python -m pip list",
    "python --version",
    "pip --version",
    "git --version",
    # pipelines and chains of read-only segments
    "rg foo | head -20",
    "git diff | wc -l",
    "pip freeze | findstr requests",
    "rg foo && rg bar",
    "git status; git log --oneline -5",
    "cat paths.py | grep DATA | sort | uniq",
    # discard-only redirections write nothing durable
    "rg foo 2>/dev/null",
    "dir 2>NUL",
    "git status >/dev/null 2>&1",
    'grep -rn "on_quarantine\\|strikes" --include="*.py" runtime/supervisor.py 2>/dev/null | head -30',
    # xargs is safe iff the command it wraps is
    'find . -name "*.py" | xargs grep -l foo',
    "find . -print0 | xargs -0 -n 50 grep -l pattern",
    'find . -type f -name "*.py" | xargs grep -l -i "stream\\|token" 2>/dev/null | grep -v test | head -40',
])
def test_auto_approved(mod, command):
    category, needs_approval, error = mod._classify(command)
    assert error is None
    assert needs_approval is False, f"{command!r} should auto-run, classified {category}"


# ── Classifier: commands that must take the approval path ────────────

@pytest.mark.parametrize("command", [
    # modifying / unknown commands
    "rm -rf build",
    "del /s *.pyc",
    "python -c \"print(1)\"",
    "git push",
    "git checkout .",
    "git branch -D main",
    "git branch newbranch",
    "git -C .. status",
    "git log --output=steal.txt",
    "git diff --ext-diff",
    "git grep -Oless foo",
    # pip
    "pip install requests",
    "pip install requests | cat",
    "python -m pip uninstall requests",
    "pip download requests",
    # guarded flags on read-only bases
    "find . -name '*.pyc' -delete",
    "find . -name '*.py' -exec rm {} +",
    "rg --pre=cat foo",
    "rg -z foo archive.gz",
    "sort -o clobbered.txt input.txt",
    "sed -i s/a/b/ paths.py",
    # shell constructs that can smuggle a second command
    "echo hi > out.txt",
    "echo hi > out.txt 2>&1",
    "rg foo 2>errors.txt",
    "cat x >nul.txt",
    "find . -type f | xargs rm",
    "echo x | xargs -I{} rm {}",
    "echo x | xargs -I rm echo-oops",  # placeholder named rm: inner cmd unknown
    "cat list.txt | xargs chmod 777",
    "rg foo | tee out.txt",
    "echo `whoami`",
    "echo $(rm x)",
    "sleep 30 &",
    "ls & del x",
    "echo don't && del x",  # unbalanced quote hides the chain
    # scope escapes
    "cat ../outside.txt",
    "type ..\\outside.txt",
])
def test_needs_approval(mod, command):
    category, needs_approval, error = mod._classify(command)
    assert error is None
    assert needs_approval is True, f"{command!r} must require approval, classified {category}"


def test_posix_escaped_space_path_in_bounds_auto_runs(mod, monkeypatch):
    """On a POSIX default shell, `Application\\ Support`-style escaped spaces are
    one path token; an in-bounds path must not shatter into out-of-bounds
    fragments and prompt (the Mac data-dir regression)."""
    monkeypatch.setattr(mod, "_IS_WINDOWS", False)
    root = str(Path(mod.ROOT_DIR).resolve()).replace("\\", "/")
    if " " not in root:
        pytest.skip("root path has no space to escape")
    escaped = root.replace(" ", "\\ ")
    for command in (
        f"ls {escaped}/plugins/",
        f'find {escaped} -type f -name "*.py" 2>/dev/null | xargs grep -l foo 2>/dev/null | head -20',
    ):
        category, needs_approval, error = mod._classify(command)
        assert error is None
        assert needs_approval is False, f"{command!r} should auto-run, classified {category}"


def test_posix_escaped_space_path_outside_roots_still_prompts(mod, monkeypatch):
    monkeypatch.setattr(mod, "_IS_WINDOWS", False)
    outside = Path(mod.ROOT_DIR).resolve().parent / "outside dir" / "secrets.txt"
    escaped = str(outside).replace("\\", "/").replace(" ", "\\ ")
    _, needs_approval, _ = mod._classify(f"cat {escaped}")
    assert needs_approval is True


def test_absolute_path_outside_roots_needs_approval(mod):
    outside = Path(mod.ROOT_DIR).resolve().parent / "outside.txt"
    category, needs_approval, error = mod._classify(f'cat "{outside}"')
    assert needs_approval is True


def test_powershell_structural_chars_need_approval(mod):
    # (...) subexpressions execute in PowerShell even as arguments.
    _, needs_approval, _ = mod._classify("type (calc)", "powershell")
    assert needs_approval is True
    # Plain read-only cmdlets stay auto-approved.
    _, needs_approval, _ = mod._classify("Get-Content paths.py", "powershell")
    assert needs_approval is False


def test_empty_command_blocked(mod):
    category, needs_approval, error = mod._classify("   ;  ")
    assert category == "blocked" and error


def test_categories(mod):
    assert mod._classify("rg foo")[0] == "search"
    assert mod._classify("ls")[0] == "listing"
    assert mod._classify("git status")[0] == "git_read"
    assert mod._classify("pip list")[0] == "pip_read"
    assert mod._classify("python --version")[0] == "version"
    assert mod._classify("rg foo | head -5")[0] == "compound_read"
    assert mod._classify("pip install x")[0] == "pip_modify"


# ── run(): approval wiring ───────────────────────────────────────────

def test_run_read_only_skips_approval(mod):
    tool = mod.RunCommand()
    context = SimpleNamespace(approve_command=None)  # would fail if consulted
    result = tool.run(context, command="echo hello", justification="test echo")
    assert result.success
    assert "hello" in result.data["stdout"]


def test_run_denied_command_fails_without_executing(mod, tmp_path):
    tool = mod.RunCommand()
    calls = []
    context = SimpleNamespace(approve_command=lambda cmd, why: calls.append(cmd) and False)
    target = tmp_path / "x.txt"
    result = tool.run(context, command=f'python -c "open(r\'{target}\', \'w\')"',
                      justification="test denial")
    assert not result.success
    assert calls, "approval callback was never consulted"
    assert not target.exists()


def test_run_approved_command_executes(mod):
    tool = mod.RunCommand()
    context = SimpleNamespace(approve_command=lambda cmd, why: True)
    result = tool.run(context, command='python -c "print(6*7)"', justification="test approval")
    assert result.success
    assert "42" in result.data["stdout"]


def test_package_metadata_present(mod):
    assert mod.dependencies_files == []
    assert mod.dependencies_pip == []


# ── Sticky cwd + background processes ───────────────────────────────

import time


@pytest.fixture()
def env(mod, tmp_path, monkeypatch):
    """Confined root, session bag, approving context; kills leftovers."""
    root = tmp_path / "root"
    (root / "sub").mkdir(parents=True)
    monkeypatch.setattr(mod, "ROOT_DIR", root)
    monkeypatch.setattr(mod, "_ALLOWED_ROOTS", {root.resolve()})
    monkeypatch.setattr(mod, "DATA_DIR", tmp_path)
    session = SimpleNamespace(plugin_state={})
    context = SimpleNamespace(
        approve_command=lambda cmd, why: True, approval_denial_reason="",
        runtime=SimpleNamespace(sessions={"k": session}), session_key="k")

    def call(context_override=None, **kwargs):
        kwargs.setdefault("justification", "test")
        return mod.RunCommand().run(context_override or context, **kwargs)

    box = SimpleNamespace(call=call, context=context, session=session,
                          root=root, sub=root / "sub", mod=mod)
    yield box
    for n in list(mod._PROCESSES):
        try:
            mod._bg_stop(n)
        except Exception:
            pass


def _bag(box):
    return box.session.plugin_state.get("run_command", {})


def test_cd_moves_sticky_cwd_without_approval(env):
    env.context.approve_command = None  # proves cd never consults approval
    out = env.call(command="cd sub")
    assert out.success is True
    assert _bag(env)["cwd"] == str(env.sub.resolve())
    env.context.approve_command = lambda c, w: True
    out = env.call(command="echo hi")
    assert out.data["cwd"] == str(env.sub.resolve())
    assert "(cwd:" in out.llm_summary


def test_cd_relative_outside_and_reset(env):
    env.call(command="cd sub")
    out = env.call(command="cd ..")
    assert out.success is True and _bag(env)["cwd"] == str(env.root.resolve())
    out = env.call(command="cd ..")  # parent of root — outside allowed roots
    assert out.success is False and "outside" in out.error
    env.call(command="cd sub")
    out = env.call(command="cd")
    assert out.success is True and _bag(env)["cwd"] == str(env.root.resolve())


def test_deleted_sticky_falls_back_to_root(env):
    gone = env.root / "gone"
    gone.mkdir()
    env.call(command=f"cd gone")
    gone.rmdir()
    out = env.call(command="echo hi")
    assert out.data["cwd"] == str(env.root.resolve())
    assert "cwd" not in _bag(env)


def test_explicit_cwd_kwarg_repins(env):
    out = env.call(command="echo hi", cwd="sub")
    assert out.success is True
    assert _bag(env)["cwd"] == str(env.sub.resolve())


def test_cd_without_session_fails_clearly(env):
    context = SimpleNamespace(approve_command=lambda c, w: True)
    out = env.call(context_override=context, command="cd sub")
    assert out.success is False and "session" in out.error
    out = env.call(context_override=context, command="echo hi")
    assert out.success is True  # normal commands unaffected


def test_compound_cd_not_intercepted(env):
    env.context.approve_command = None
    out = env.call(command="cd sub && echo hi")
    assert out.success is False
    assert "approval" in out.error  # took the normal approval path


def test_background_lifecycle(env):
    out = env.call(command='python -c "import time; time.sleep(30)"',
                   run_in_background=True)
    assert out.success is True
    pid_n = out.data["process_id"]
    assert out.data["backgrounded"] is True
    check = env.call(operation="check", process_id=pid_n)
    assert check.data["running"] is True
    listing = env.call(operation="list")
    assert any(p["process_id"] == pid_n for p in listing.data["processes"])
    stop = env.call(operation="stop", process_id=pid_n)
    assert stop.success is True
    assert env.call(operation="check", process_id=pid_n).success is False
    assert env.call(operation="stop", process_id=pid_n).success is False


def test_background_output_lands_in_log(env):
    out = env.call(command='python -c "print(24*7)"', run_in_background=True)
    pid_n = out.data["process_id"]
    for _ in range(100):
        check = env.call(operation="check", process_id=pid_n)
        if not check.data["running"]:
            break
        time.sleep(0.1)
    assert check.data["running"] is False and check.data["returncode"] == 0
    assert "168" in check.llm_summary


def test_check_requires_process_id(env):
    out = env.call(operation="check")
    assert out.success is False and "process_id" in out.error


def test_background_denied_spawns_nothing(env):
    env.context.approve_command = lambda cmd, why: False
    before = set(env.mod._PROCESSES)
    out = env.call(command='python -c "import time; time.sleep(30)"',
                   run_in_background=True)
    assert out.success is False
    assert set(env.mod._PROCESSES) == before


def test_background_approval_mentions_background(env):
    whys = []
    env.context.approve_command = lambda cmd, why: whys.append(why) or True
    out = env.call(command='python -c "print(1)"', run_in_background=True)
    assert out.success is True
    assert "background: yes" in whys[0]
    env.call(operation="stop", process_id=out.data["process_id"])
