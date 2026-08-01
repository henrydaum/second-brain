"""The read-only shell recognizer: what it allows, and what it must not.

The classifier this replaces failed in the invisible direction — a wrong
"unsafe" gets reported by the user, a wrong "safe" does not. So the refusals
here matter more than the allowances, and each names the specific way a
name-keyed whitelist would have got it wrong.
"""

import os

import pytest

from sandbox.policy import SAFE, UNSAFE, Chain, classify
from sandbox.guest.requests import Request


def _run(argv, **args):
    """Classify one proc.run with a fresh, ordinary chain."""
    return classify(Request(type="proc.run", args={"argv": argv, **args}),
                    Chain(root="repl").push("tool_run_command"))


def _allowed(argv, **args) -> bool:
    """Whether this invocation executes without a dialog."""
    return _run(argv, **args).level == SAFE


# ── what it is for ────────────────────────────────────────────────────

def test_the_high_frequency_read_only_verbs_are_allowed():
    """The prompts that carry no information are the entire point."""
    for argv in (["git", "status"],
                 ["git", "status", "--porcelain"],
                 ["git", "log", "--oneline", "-n", "5"],
                 ["git", "diff", "--stat"],
                 ["git", "branch", "-a"],
                 ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                 ["python", "--version"],
                 ["pwd"]):
        assert _allowed(argv), argv


def test_the_reason_names_the_command_for_the_ledger():
    """A SAFE decision still records why, and the row is read by a person."""
    assert _run(["git", "status"]).reason == "git status only reads"


def test_a_string_argv_is_split_the_way_the_handler_splits_it():
    """``shell=None`` with a string is shlex.split, not handed to a shell.

    The recognizer has to agree with ``handlers.fs_net._invocation`` about
    what will actually run, or it is vouching for a different command.
    """
    assert _allowed("git status")
    assert _allowed("git log --oneline")


def test_a_path_separator_after_a_flag_value_still_reads():
    """``--`` ends flag parsing and changes nothing about the verb."""
    assert _allowed(["git", "log", "--", "README.md"])


# ── the ways a name-keyed whitelist gets it wrong ─────────────────────

def test_the_unit_is_the_pair_because_git_is_not_read_only():
    """``git`` on a whitelist would have authorized every one of these."""
    for argv in (["git", "push"],
                 ["git", "commit", "-m", "x"],
                 ["git", "checkout", "main"],
                 ["git", "reset", "--hard"],
                 ["git", "clean", "-fd"],
                 ["git", "config", "user.name", "someone"]):
        assert not _allowed(argv), argv


def test_a_bare_word_creates_things_for_some_verbs_and_not_others():
    """Why ``positionals_ok`` exists, as two concrete bugs it prevents.

    ``git branch foo`` makes a branch and ``git remote add`` adds a remote,
    while the identical shape means "a ref to read" to ``git log``. Allowing
    free positionals everywhere would have let both writes through.
    """
    assert _allowed(["git", "log", "main"])
    assert not _allowed(["git", "branch", "release-2"])
    assert not _allowed(["git", "remote", "add", "origin", "http://x/y.git"])


def test_an_unknown_flag_abstains_rather_than_being_ignored():
    """A whitelist that skips flags it does not know is a deny list."""
    assert not _allowed(["git", "branch", "-d", "old"])
    assert not _allowed(["git", "diff", "--ext-diff"])
    assert not _allowed(["git", "status", "--totally-made-up"])


def test_things_that_look_inert_and_run_arbitrary_code_are_absent():
    """Each of these is a plausible whitelist entry and a real hazard.

    ``pytest --collect-only`` imports every test module, so module-level code
    runs; ``find`` is read-only until ``-exec``; a package install runs the
    package's own hooks. None may be added without this test being changed.
    """
    for argv in (["pytest", "--collect-only"],
                 ["find", ".", "-name", "x"],
                 ["npm", "install"],
                 ["pip", "install", "requests"],
                 ["make"],
                 ["python", "script.py"]):
        assert not _allowed(argv), argv


# ── the three structural rules ────────────────────────────────────────

def test_a_named_shell_is_read_when_the_line_holds_nothing_it_could_act_on():
    """The carve-out that makes any of this reachable in practice.

    ``tool_run_command`` defaults ``shell`` to ``"default"``, so refusing every
    named shell outright quietly disabled *both* recognizers: ``git status``
    was never auto-allowed and no command was ever offered a remembered grant.
    The tests missed it because they build Requests with no shell at all.

    ``sh -c "git status"`` runs exactly ``git`` with the argument ``status``.
    No expansion, no splitting ambiguity, nothing to be wrong about.
    """
    assert _allowed(["git", "status"], shell=None)
    for shell in ("default", "cmd", "powershell"):
        assert _allowed("git status", shell=shell), shell
        assert _allowed(["git", "status"], shell=shell), shell


def test_a_shell_line_with_anything_to_interpret_still_abstains():
    """The whitelist is of *inert* characters, so this list need not be
    complete — anything not provably inert is out by construction."""
    for line in ('echo "hi" > ~/Desktop/test.txt',   # quotes, redirect, tilde
                 "git status && git push",            # operator
                 "git status | tee log",              # pipe
                 "cat *.py",                          # glob
                 "ls ~/Desktop",                      # tilde
                 "echo $HOME",                        # substitution
                 "git status; rm -rf /",              # separator
                 "type C:\\Windows\\x.txt"):          # backslash
        assert not _allowed(line, shell="default"), line


def test_the_inert_carve_out_does_not_leak_into_a_remembered_grant():
    """A grant matches through a shell only on the same inert terms."""
    from runtime.context import set_kernel_parts

    try:
        set_kernel_parts(config={"shell_allowed_prefixes": ["git push"]})
        assert _allowed("git push origin main", shell="default")
        assert not _allowed("git push && rm -rf /", shell="default")
        assert not _allowed("git push $(whoami)", shell="default")
    finally:
        set_kernel_parts(config={"shell_allowed_prefixes": []})


def test_a_metacharacter_anywhere_abstains_without_parsing():
    """No decomposition: the presence of one is the whole answer.

    With ``shell=None`` these are literal and harmless, which is exactly why
    abstaining is cheap — an argv carrying one was written by somebody who
    expected a shell, and that mismatch is worth a dialog.
    """
    for argv in (["git", "status", "|", "rm"],
                 ["git", "status", "&&", "git", "push"],
                 ["git", "log", "$(whoami)"],
                 ["git", "status", ";", "curl", "http://x"]):
        assert not _allowed(argv), argv


def test_the_program_must_be_a_bare_name():
    """Resolving which ``git`` a path names means trusting PATH."""
    assert not _allowed(["/tmp/evil/git", "status"])
    assert not _allowed(["C:\\tools\\git", "status"])


# ── the property the whole design rests on ────────────────────────────

def test_a_recognizer_can_only_ever_widen():
    """It answers "here is a reason", never "this is unsafe".

    The dead classifier had authority to call something safe. This one has
    authority only to vouch, so every path it does not recognise falls through
    to the dialog — which is what makes it safe to write at all.
    """
    from sandbox import shell

    assert all(recognize(" ", {"argv": ["definitely-not-known"]}) is None
               for recognize in shell._SHELL_RECOGNIZERS)
    assert _run(["definitely-not-known"]).level == UNSAFE


def test_a_raising_recognizer_abstains_rather_than_failing_the_gate():
    """Widening only, so failing it closed costs a dialog and nothing else."""
    from sandbox import shell

    def boom(shown, args):
        raise RuntimeError("recognizer bug")

    original = shell._SHELL_RECOGNIZERS
    try:
        shell._SHELL_RECOGNIZERS = [boom]
        assert _run(["git", "status"]).level == UNSAFE
    finally:
        shell._SHELL_RECOGNIZERS = original


# ── the remembered half ───────────────────────────────────────────────

def _allow_prefixes(*prefixes):
    """Set the user's remembered command list for one test."""
    from runtime.context import set_kernel_parts

    set_kernel_parts(config={"shell_allowed_prefixes": list(prefixes)})


def test_a_remembered_prefix_allows_the_verb_with_any_arguments():
    """The bargain the setting states: flags and arguments are not checked."""
    try:
        _allow_prefixes("git push")
        assert _allowed(["git", "push"])
        assert _allowed(["git", "push", "origin", "main"])
        assert _allowed(["git", "push", "--force"])
        assert not _allowed(["git", "pull"]), "a different verb is a different grant"
    finally:
        _allow_prefixes()


def test_a_remembered_grant_is_still_structurally_checked():
    """The soundness argument for storing a pair instead of a string.

    A raw ``startswith`` on the rendered line would allow every one of these,
    because "git push" prefixes all of them. The prefix is re-derived from the
    structured argv instead, so a metacharacter or a shell abstains before any
    matching happens.
    """
    try:
        _allow_prefixes("git push")
        assert not _allowed(["git", "push", "&&", "rm", "-rf", "/"])
        assert not _allowed(["git", "push", "$(whoami)"])
        assert not _allowed("git push && rm -rf /", shell="default")
        assert not _allowed("git push `whoami`", shell="powershell")
        assert not _allowed("git push > ~/out.txt", shell="cmd")
        assert not _allowed(["/tmp/evil/git", "push"])
    finally:
        _allow_prefixes()


def test_the_prefix_written_down_is_the_prefix_matched():
    """One vocabulary, or a grant nobody can reason about."""
    from sandbox.shell import command_prefix

    assert command_prefix(["git", "push", "--force"]) == "git push"
    assert command_prefix(["GIT.exe", "Push"]) == "git push"
    assert command_prefix(["pytest"]) == "pytest"
    assert command_prefix(["rm", "-rf", "/"]) == "rm"
    # No unit it can describe honestly.
    assert command_prefix(["git", "push", "|", "tee"]) == ""
    assert command_prefix(["/usr/bin/git", "push"]) == ""
    assert command_prefix([]) == ""
    assert command_prefix(None) == ""


def test_a_second_word_naming_a_file_reduces_to_the_program():
    """``python train.py`` must never be storable as itself.

    Stored whole it reads like permission for one known script while granting
    whatever that file says tomorrow. Reduced, it grants more and *says* so —
    and the reduction happens in ``command_prefix``, so a hand-edited config
    entry cannot resurrect the dishonest form either.
    """
    from sandbox.shell import command_prefix

    assert command_prefix(["python", "train.py"]) == "python"
    assert command_prefix(["python", "scripts/train.py"]) == "python"
    try:
        _allow_prefixes("python train.py")
        assert not _allowed(["python", "train.py"]), "hand-edited entry matched"
    finally:
        _allow_prefixes()


# ── segmentation ──────────────────────────────────────────────────────

@pytest.fixture
def posix(monkeypatch):
    """Pretend the platform's default shell is /bin/sh.

    ``shell="default"`` is ``sh`` on POSIX and ``cmd.exe`` on Windows — one
    argument value, two lexers — so the segmentation path asks the platform
    rather than trusting the name, and these tests have to say which.
    """
    monkeypatch.setattr(os, "name", "posix")


def test_every_segment_must_be_granted_not_just_the_first(posix):
    """The whole reason chaining can be allowed at all.

    A string prefix would read "git push && rm -rf /" as starting with
    "git push" and let it run. Segmentation asks a different question — is
    every part covered — and nobody granted ``rm``.
    """
    try:
        _allow_prefixes("git push")
        assert not _allowed("git push && rm -rf /", shell="default")
        _allow_prefixes("git push", "rm")
        assert _allowed("git push && rm -rf /", shell="default")
    finally:
        _allow_prefixes()


def test_a_quoted_operator_is_not_a_separator(posix):
    """What the dead classifier's regex got wrong and a real lexer gets right."""
    from sandbox.shell import _shell_segments

    assert _shell_segments({"argv": 'git commit -m "fix && ship"',
                            "shell": "default"}) == [
        ["git", "commit", "-m", "fix && ship"]]


def test_the_line_is_split_at_every_operator(posix):
    from sandbox.shell import _shell_segments

    assert _shell_segments({"argv": "a 1 && b 2 || c; d | e",
                            "shell": "default"}) == [
        ["a", "1"], ["b", "2"], ["c"], ["d"], ["e"]]


def test_an_effect_that_lives_in_no_command_name_is_refused(posix):
    """Redirects, substitution, subshells, assignment prefixes, bad quoting.

    The redirect case is the one that started this: granting ``echo`` must not
    license writing a file anywhere, so ``echo x > ~/f`` gets no unit at all.
    The right door for it is ``fs.write``, which asks about the *path*.
    """
    from sandbox.shell import _shell_segments

    for line in ('echo "Test text file." > ~/Desktop/test.txt',
                 "cat < /etc/passwd",
                 "ls >> log",
                 "git push $(whoami)",
                 "git push `whoami`",
                 "echo ${HOME}",
                 "(cd /tmp && rm -rf x)",
                 "diff <(a) <(b)",
                 "SECRET=x git push",
                 'git commit -m "unbalanced'):
        assert _shell_segments({"argv": line, "shell": "default"}) is None, line


def test_windows_shells_keep_only_the_inert_path(posix, monkeypatch):
    """``shlex`` implements POSIX quoting; cmd and PowerShell differ.

    Mis-lexing a line is the failure that widens, so the shells whose rules
    this does not implement get the inert fast path and nothing more.
    """
    from sandbox.shell import _shell_segments

    assert _shell_segments({"argv": "a && b", "shell": "cmd"}) is None
    assert _shell_segments({"argv": "a && b", "shell": "powershell"}) is None
    monkeypatch.setattr(os, "name", "nt")
    assert _shell_segments({"argv": "a && b", "shell": "default"}) is None


def test_a_glob_in_the_program_position_names_nothing_grantable(posix):
    """``* --help`` runs whatever sorts first in the working directory.

    Elsewhere in the line a glob is ordinary argument expansion, and arguments
    are unchecked by design — so this is only about ``argv[0]``.
    """
    from sandbox.shell import command_prefix, command_prefixes

    assert command_prefix(["*", "--help"]) == ""
    assert command_prefix(["~/bin/tool"]) == ""
    assert command_prefixes({"argv": "cat *.py", "shell": "default"}) == ["cat"]


def test_read_only_holds_across_a_pipeline(posix):
    """Every segment, or it asks — same rule as the remembered half."""
    assert _allowed("git status && git diff --stat", shell="default")
    assert not _allowed("git status && git push", shell="default")


def test_proc_start_gets_the_same_recognizer_as_proc_run():
    """Keeping a long-running process is the same act as running one."""
    started = classify(
        Request(type="proc.start", args={"argv": ["git", "status"]}),
        Chain(root="repl").push("tool_run_command"))
    assert started.level == SAFE
