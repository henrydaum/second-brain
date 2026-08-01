"""The read-only shell recognizer: what it allows, and what it must not.

The classifier this replaces failed in the invisible direction — a wrong
"unsafe" gets reported by the user, a wrong "safe" does not. So the refusals
here matter more than the allowances, and each names the specific way a
name-keyed whitelist would have got it wrong.
"""

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

def test_any_shell_abstains_because_a_parser_is_what_cannot_be_reasoned_about():
    """``shell=None`` is the only shell-free path in the handler."""
    assert _allowed(["git", "status"], shell=None)
    for shell in ("default", "cmd", "powershell"):
        assert not _allowed(["git", "status"], shell=shell), shell


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
    from sandbox import policy

    assert all(recognize(" ", {"argv": ["definitely-not-known"]}) is None
               for recognize in policy._SHELL_RECOGNIZERS)
    assert _run(["definitely-not-known"]).level == UNSAFE


def test_a_raising_recognizer_abstains_rather_than_failing_the_gate():
    """Widening only, so failing it closed costs a dialog and nothing else."""
    from sandbox import policy

    def boom(shown, args):
        raise RuntimeError("recognizer bug")

    original = policy._SHELL_RECOGNIZERS
    try:
        policy._SHELL_RECOGNIZERS = [boom]
        assert _run(["git", "status"]).level == UNSAFE
    finally:
        policy._SHELL_RECOGNIZERS = original


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
        assert not _allowed(["git", "push"], shell="cmd")
        assert not _allowed(["git", "push"], shell="powershell")
        assert not _allowed("git push && rm -rf /", shell="default")
        assert not _allowed(["/tmp/evil/git", "push"])
    finally:
        _allow_prefixes()


def test_the_prefix_written_down_is_the_prefix_matched():
    """One vocabulary, or a grant nobody can reason about."""
    from sandbox.policy import command_prefix

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
    from sandbox.policy import command_prefix

    assert command_prefix(["python", "train.py"]) == "python"
    assert command_prefix(["python", "scripts/train.py"]) == "python"
    try:
        _allow_prefixes("python train.py")
        assert not _allowed(["python", "train.py"]), "hand-edited entry matched"
    finally:
        _allow_prefixes()


def test_proc_start_gets_the_same_recognizer_as_proc_run():
    """Keeping a long-running process is the same act as running one."""
    started = classify(
        Request(type="proc.start", args={"argv": ["git", "status"]}),
        Chain(root="repl").push("tool_run_command"))
    assert started.level == SAFE
