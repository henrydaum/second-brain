"""What a command line did to files — the *display* question.

``sandbox.shell`` answers two questions already, both about authorization:
what commands does this line run, and is every one of them already granted.
This is a third and it is not one of those. It runs after the fact, so a
frontend can show the agent deleting a directory the way it shows the agent
writing a file.

The distinction matters because CLAUDE.md records that a ~500-line command
classifier was deliberately killed, and a table of command names is exactly
what that looked like. The difference is the failure profile. The dead one
decided *safety*, where a wrong "safe" is silent and grants something; this
decides *what to draw*, where a miss is an absent row and a false positive is
a file shown that did not change. Both are cosmetic and both are visible to
whoever is reading the panel — so a table that abstains generously is fine
here and was never fine there.

The first test is the one that keeps it true.
"""

import ast
import os
from pathlib import Path

import pytest

from sandbox import shell


def _touched(argv, cwd=None, **extra):
    args = {"argv": argv, **extra}
    if cwd:
        args["cwd"] = cwd
    return shell.files_touched(args)


# ── the separation ────────────────────────────────────────────────────

def test_no_authorization_path_reads_the_file_table():
    """``files_touched`` must never reach a decision about what may run.

    Pinned structurally rather than described, because the drift that would
    undo it is one import: a recognizer reaching for "but we already know what
    this touches" is how the dead classifier comes back, and it would look
    like a simplification at the time.
    """
    source = Path(shell.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    authorizers = {"classify_shell", "_read_only_command", "_read_only_segment",
                   "_remembered_prefix", "command_prefix", "command_prefixes"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name not in authorizers:
            continue
        called = {n.func.id for n in ast.walk(node)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        assert "files_touched" not in called, (
            f"{node.name} reads the display table; that is the old classifier")
        names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        assert "_FILE_COMMANDS" not in names, (
            f"{node.name} reads _FILE_COMMANDS; that table is display-only")


# ── what it reads ─────────────────────────────────────────────────────

@pytest.mark.parametrize("argv, paths, deleted", [
    ("rm -rf build dist", ["build", "dist"], ["build", "dist"]),
    ("rm notes.md", ["notes.md"], ["notes.md"]),
    ("rmdir empty", ["empty"], ["empty"]),
    ("mkdir -p out", ["out"], []),
    ("touch CHANGELOG.md", ["CHANGELOG.md"], []),
    # A copy reads its source and writes its destination; only one changed.
    ("cp src.py backup.py", ["backup.py"], []),
    # A move does both, and which end is which is the whole point.
    ("mv a.txt b.txt", ["b.txt", "a.txt"], ["a.txt"]),
    ("mv one two dest", ["dest", "one", "two"], ["one", "two"]),
])
def test_it_reads_the_common_file_commands(argv, paths, deleted):
    assert _touched(argv) == (paths, deleted)


def test_flags_are_not_operands():
    """``-rf`` is not a file, and neither is ``--recursive``."""
    assert _touched("rm -r --force -v target")[0] == ["target"]


def test_a_program_named_by_path_is_still_itself():
    """The list form, because a Windows path in the *string* form is lexed by
    ``shlex`` in POSIX mode, where the backslashes are escapes."""
    assert _touched("/usr/bin/rm doomed")[0] == ["doomed"]
    assert _touched(["C:\\Windows\\System32\\del.exe", "doomed"])[0] == ["doomed"]


def test_paths_resolve_against_the_requests_cwd():
    """A relative ``build/`` means nothing to whoever reads the row later."""
    paths, _ = _touched("rm -rf build", cwd="/srv/app")
    assert paths == [str(Path("/srv/app/build").absolute())]


@pytest.mark.skipif(os.name == "nt",
                    reason="`shlex` describes POSIX shells; `default` is cmd here")
def test_several_commands_on_one_line_all_count():
    """Only under a real shell — with ``shell=None`` the ``&&`` is a literal
    argument rather than a separator, and is abstained on as such."""
    paths, deleted = _touched("mkdir out && touch out/log && rm stale",
                              shell="default")
    assert paths == ["out", "out/log", "stale"] and deleted == ["stale"]


def test_a_separator_with_no_shell_to_read_it_abstains():
    assert _touched("mkdir out && rm stale") == ([], [])


# ── what it refuses ───────────────────────────────────────────────────

@pytest.mark.parametrize("argv, why", [
    ("npm install", "not in the table — thousands of files, and none named"),
    ("git checkout .", "unknowable without asking git"),
    ("python build.py", "an arbitrary program"),
    ("ls -la", "reads nothing into existence"),
])
def test_it_says_nothing_about_commands_it_does_not_know(argv, why):
    assert _touched(argv) == ([], []), why


@pytest.mark.parametrize("argv", [
    "rm *.log",                 # the shell expands it and we are not the shell
    "rm build/*",
    "rm ~/scratch",             # tilde expansion
    "rm file[12].txt",
])
def test_an_unexpanded_operand_abstains(argv):
    assert _touched(argv) == ([], [])


def test_a_glob_poisons_the_whole_line_rather_than_dropping_one_operand():
    """Reporting the half it understood would be a claim that these are *the*
    files, which is worse than saying nothing."""
    assert _touched("rm keep.txt *.log") == ([], [])


@pytest.mark.parametrize("argv", [
    "echo hello > notes.md",    # redirection: the effect is in no command name
    "rm $(cat targets.txt)",    # substitution: the line that runs is not this
    "rm `cat targets.txt`",
    "(cd tmp && rm x)",         # a subshell
    "TMPDIR=/tmp rm x",         # an assignment prefix
    'rm "unbalanced',           # the lexer refuses it
])
def test_it_inherits_the_decomposers_abstentions(argv):
    """Refused by ``_command_segments`` under a shell, and by the operand rule
    without one — where these characters are literal and a file genuinely
    named ``$(cat`` is not what anybody reading the panel would assume."""
    assert _touched(argv, shell="default") == ([], [])
    assert _touched(argv) == ([], [])


def test_an_exact_argv_list_is_read_too():
    """The list form never goes near a shell, so there is nothing to lex."""
    assert _touched(["rm", "-rf", "build"]) == (["build"], ["build"])
