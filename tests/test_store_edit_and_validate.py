"""Contracts for the two store tools the benchmark spent the most turns on.

``edit_file`` lost 102 calls across a 636-trial run to a required argument the
other arguments already implied, and its "you edited a plugin file" nudge fired
on 1,291 of 2,608 successful writes — 1,100 of them on files in ``scripts/``,
which are not plugins and which ``run_script`` already checks the same way.
``validate`` told an agent in lockdown that nothing blocked a script the mode
was about to refuse.

None of that was pinned in either direction before this file.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

import sandbox  # noqa: F401 - installs the guest package alias
from tests.support import store_source

EDIT_FILE = "tools/tool_edit_file.py"
VALIDATE = "tools/tool_validate.py"
RUN_COMMAND = "tools/tool_run_command.py"
FILE_READS = "tools/helpers/file_reads.py"
PATH_REPAIR = "tools/helpers/path_repair.py"

_PACKAGE = "_store_tools_under_test"


def _source_or_skip(relative: str) -> str:
    text = store_source(relative)
    if text is None:
        pytest.skip(f"{relative} is not present on a local store ref")
    return text


def _load(relative: str):
    """Load a store tool module, satisfying its sibling-helper imports.

    The kernel puts each declared dependency's directory on the box's import
    path, so ``from . import file_reads`` resolves to a sibling at runtime even
    though the helper ships in a subdirectory. Reproduced here by building the
    package the tool believes it is in, rather than by editing the import out —
    a loader that rewrote the source would stop testing the shipped file.
    """
    package = sys.modules.get(_PACKAGE)
    if package is None:
        package = types.ModuleType(_PACKAGE)
        package.__path__ = []
        sys.modules[_PACKAGE] = package
        for helper in (FILE_READS, PATH_REPAIR):
            name = Path(helper).stem
            module = types.ModuleType(f"{_PACKAGE}.{name}")
            module.__package__ = _PACKAGE
            exec(compile(_source_or_skip(helper), helper, "exec"), module.__dict__)
            sys.modules[f"{_PACKAGE}.{name}"] = module
            setattr(package, name, module)

    name = Path(relative).stem
    module = types.ModuleType(f"{_PACKAGE}.{name}")
    module.__package__ = _PACKAGE
    exec(compile(_source_or_skip(relative), relative, "exec"), module.__dict__)
    return module


def _exec_plain(relative: str):
    """Load a store tool that has no sibling imports."""
    namespace = {"__name__": f"test_{Path(relative).stem}"}
    exec(compile(_source_or_skip(relative), relative, "exec"), namespace)
    return namespace


# ── a minimal sdk double ──────────────────────────────────────────────

class _Path:
    @staticmethod
    def suffix(path):
        name = str(path).replace("\\", "/").rsplit("/", 1)[-1]
        return name[name.rfind("."):] if "." in name else ""

    @staticmethod
    def parent(path):
        return str(path).replace("\\", "/").rsplit("/", 1)[0]

    @staticmethod
    def normalize(path):
        return str(path).replace("\\", "/").rstrip("/").lower()

    @staticmethod
    def within(path, root):
        if not root:
            return False
        return _Path.normalize(path).startswith(_Path.normalize(root) + "/")


class _Paths:
    ROOTS = {
        "project": "C:/data/workspace",
        "workspace": "C:/data/workspace",
        "installed": "C:/data/installed",
        "bundled": "C:/app/bundled",
        "scripts": "C:/data/workspace/scripts",
    }

    def get(self, name):
        return self.ROOTS.get(name)


class _FakeSDK:
    """Enough of the SDK for the pure helpers under test."""

    def __init__(self, existing=(), mode="ask"):
        self.path = _Path()
        self.paths = _Paths()
        self._existing = {_Path.normalize(p) for p in existing}
        self._mode = mode
        self.Failed = RuntimeError

    # ``_stat`` asks fs.list whether a path is a file that exists.
    @property
    def fs(self):
        outer = self

        class _FS:
            @staticmethod
            def list(path, details=False):
                if _Path.normalize(path) in outer._existing:
                    return [{"is_dir": False, "mtime_ns": 1, "name": str(path)}]
                return []

        return _FS()

    @property
    def session(self):
        outer = self

        class _Session:
            @staticmethod
            def get(key="", details=False):
                return {"mode": outer._mode}

        return _Session()


# ── the file conforms at all ──────────────────────────────────────────

@pytest.mark.store
@pytest.mark.parametrize("relative", [EDIT_FILE, VALIDATE])
def test_the_edited_store_tools_still_conform(relative):
    """A store tool that will not load is not a tool."""
    from sandbox.validator import validate

    report = validate(_source_or_skip(relative), filename=Path(relative).name)
    errors = [f for f in report.findings if f.level == "error"]
    assert not errors, report.render()


@pytest.mark.store
def test_validate_declares_the_session_request_its_verdict_needs():
    """The lockdown verdict reads the mode, and an undeclared Request is
    refused at exactly the moment it matters."""
    from guest.requests import ALL_TYPES
    from sandbox.validator import validate

    declared = validate(_source_or_skip(VALIDATE),
                        filename="tool_validate.py").declarations
    assert "session.get" in declared["requests"]
    assert set(declared["requests"]) <= set(ALL_TYPES)


# ── edit_file: inferring the operation ────────────────────────────────

@pytest.mark.store
def test_old_text_means_replace():
    """74 of the 102 measured failures carried exactly this shape. Nothing
    else reads old_text, and replace still demands an exact match, so a wrong
    guess here cannot write anything."""
    module = _load(EDIT_FILE)
    sdk = _FakeSDK(existing=["C:/data/workspace/a.py"])

    op, err = module._operation(sdk, "C:/data/workspace/a.py",
                               {"old_text": "x", "new_text": "y"})

    assert (op, err) == ("replace", None)


@pytest.mark.store
def test_content_for_a_file_that_does_not_exist_means_create():
    """Overwrite and append of a missing file both reduce to create."""
    module = _load(EDIT_FILE)
    sdk = _FakeSDK(existing=[])

    op, err = module._operation(sdk, "C:/data/workspace/out/new.json",
                                {"content": "{}"})

    assert (op, err) == ("create", None)


@pytest.mark.store
def test_content_for_a_file_that_exists_still_has_to_ask():
    """The one that must NOT be inferred. append and overwrite carry the same
    arguments, so guessing overwrite would silently drop the file."""
    module = _load(EDIT_FILE)
    sdk = _FakeSDK(existing=["C:/data/workspace/log.txt"])

    op, err = module._operation(sdk, "C:/data/workspace/log.txt",
                                {"content": "more"})

    assert op == ""
    assert "overwrite" in err and "append" in err


@pytest.mark.store
def test_delete_is_never_inferred():
    """A destructive operation should be one the caller said out loud."""
    module = _load(EDIT_FILE)
    sdk = _FakeSDK(existing=["C:/data/workspace/gone.txt"])

    op, err = module._operation(sdk, "C:/data/workspace/gone.txt", {})

    assert op == ""
    assert err


@pytest.mark.store
def test_an_explicit_operation_always_wins():
    """Including an explicit create over a file that exists: that is a caller
    telling us it believes the file is new, and it deserves the failure."""
    module = _load(EDIT_FILE)
    sdk = _FakeSDK(existing=["C:/data/workspace/a.py"])

    assert module._operation(sdk, "C:/data/workspace/a.py",
                             {"operation": "append", "content": "x"})[0] == "append"
    assert module._operation(sdk, "C:/data/workspace/a.py",
                             {"operation": "create", "content": "x",
                              "old_text": "ignored"})[0] == "create"


@pytest.mark.store
def test_operation_is_no_longer_a_required_argument():
    """The state machine refuses a call missing a required argument before the
    tool ever runs, so inference is unreachable while it is listed."""
    module = _load(EDIT_FILE)

    assert module.EditFile.parameters["required"] == ["path"]


@pytest.mark.store
def test_the_tool_says_that_writes_create_their_parents():
    """81 denied shell mkdir calls, every one inside the agent's own workspace,
    for a directory fs.write was going to create anyway."""
    module = _load(EDIT_FILE)
    text = module.EditFile.description + str(module.EditFile.parameters)

    assert "parent" in text.lower()


# ── edit_file: who gets told to validate ──────────────────────────────

@pytest.mark.store
@pytest.mark.parametrize("path", [
    "C:/data/workspace/tools/tool_x.py",
    "C:/data/installed/services/service_y.py",
    "C:/app/bundled/commands/command_z.py",
    # Nesting does not make a script: run_script refuses this one, so its
    # author does want to be told to check it.
    "C:/data/workspace/scripts/sub/helper.py",
    # A family prefix makes the validator apply the plugin contract wherever
    # the file sits, so this is a plugin as far as the check is concerned.
    "C:/data/workspace/scripts/tool_x.py",
])
def test_a_plugin_still_gets_the_validate_nudge(path):
    """A plugin that will not load fails silently — validate is the only
    in-band signal, so anything not provably a script keeps the nudge."""
    module = _load(EDIT_FILE)

    assert module._plugin_edit_reminder(_FakeSDK(), path) == module.PLUGIN_EDIT_REMINDER


@pytest.mark.store
def test_a_script_is_pointed_at_run_script_instead():
    """run_script runs the identical validator in preflight and returns the
    same findings, so validating first only costs a turn."""
    module = _load(EDIT_FILE)

    hint = module._plugin_edit_reminder(_FakeSDK(), "C:/data/workspace/scripts/tally.py")

    assert hint == module.SCRIPT_EDIT_HINT
    assert "run_script" in hint
    assert "plugin file" not in hint


@pytest.mark.store
def test_a_file_outside_every_tree_gets_no_nudge_at_all():
    module = _load(EDIT_FILE)

    assert module._plugin_edit_reminder(_FakeSDK(), "C:/elsewhere/thing.py") == ""


@pytest.mark.store
def test_a_non_python_file_gets_no_nudge():
    module = _load(EDIT_FILE)

    assert module._plugin_edit_reminder(
        _FakeSDK(), "C:/data/workspace/scripts/notes.md") == ""


# ── validate: the verdict has to know the mode ────────────────────────

@pytest.mark.store
def test_the_disclaimer_verdict_warns_that_lockdown_will_refuse_the_launch():
    """An unmediated import makes launch unsafe, and lockdown refuses unsafe
    without asking — so "nothing blocks it" is false there, in the direction
    that wastes a run."""
    namespace = _exec_plain(VALIDATE)
    report = {"ok": True, "disclaimed": True, "unmediated": ["zipfile", "tarfile"]}

    verdict = namespace["_verdict"](report, "lockdown")

    assert "zipfile" in verdict and "tarfile" in verdict
    assert "Nothing blocks it" not in verdict


@pytest.mark.store
@pytest.mark.parametrize("mode", ["ask", "yolo"])
def test_the_disclaimer_verdict_is_unchanged_where_a_person_can_still_say_yes(mode):
    namespace = _exec_plain(VALIDATE)
    report = {"ok": True, "disclaimed": True, "unmediated": ["zipfile"]}

    assert namespace["_verdict"](report, mode) == (
        "**Loads, with a disclaimer.** Nothing blocks it; read the warnings.")


@pytest.mark.store
@pytest.mark.parametrize("mode", ["ask", "lockdown", "yolo"])
def test_the_other_two_verdicts_do_not_vary_by_mode(mode):
    """Only the disclaimer branch is a refusal in disguise. A file that will
    not load will not load in any mode, and a conforming one conforms."""
    namespace = _exec_plain(VALIDATE)

    assert namespace["_verdict"]({"ok": False}, mode).startswith("**Will not load.**")
    assert namespace["_verdict"]({"ok": True}, mode).startswith("**Conforms.**")


@pytest.mark.store
def test_validate_guidance_separates_the_case_that_needs_it_from_the_one_that_does_not():
    """The nudge is only worth a turn where nothing else would tell you."""
    namespace = _exec_plain(VALIDATE)
    guidance = namespace["Validate"].agent_prompt

    assert "run_script" in guidance
    assert "silently" in guidance


# ── run_command: the same mkdir fact, where the agent hits the wall ───

@pytest.mark.store
def test_the_lockdown_denial_says_a_directory_does_not_need_making():
    """The denial already routes the agent to the file tools; 81 denied mkdir
    calls say it needs to route them one step further."""
    source = _source_or_skip(RUN_COMMAND)

    assert "You do not need `mkdir`" in source


@pytest.mark.store
def test_a_refused_stat_does_not_escape_the_inference():
    """``_operation`` runs before the handler that translates ``Denied``, so a
    refused ``fs.list`` must not propagate out of it.

    It cannot: ``sdk.Denied`` subclasses ``sdk.Failed``, which ``_stat``
    already swallows, so a refusal reads as "not there" and the inference falls
    to ``create`` — and the write that follows is still gated and still gets
    the stop-and-ask translation. The failure is deferred to the operation that
    can actually report it, which is where it belongs.
    """
    module = _load(EDIT_FILE)
    sdk = _FakeSDK()

    class _Exploding:
        @staticmethod
        def list(path, details=False):
            raise sdk.Failed("nope")

    object.__setattr__(sdk, "_existing", set())
    type(sdk).fs = property(lambda self: _Exploding)
    try:
        op, err = module._operation(sdk, "C:/data/workspace/out/x.json",
                                    {"content": "{}"})
        assert (op, err) == ("create", None)
    finally:
        del type(sdk).fs
