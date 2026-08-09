"""Shell commands: what one will run, and what may run unasked.

Split out of :mod:`sandbox.policy` because it stopped being a branch and became
a subject. ``classify`` decides one thing about a Request from its type,
arguments and chain; this answers a harder question that only the shell family
asks — *what commands does this line actually run* — and then applies two
recognizers to the answer.

**Everything is asked by default.** Why there is no classifier deciding
*safety* here, and why there must not be one again, is in CLAUDE.md under "And
it is where the classifier died": the old one decomposed a command line with a
regex and then ruled on each piece, which is undecidable in principle and fails
in the invisible direction — a wrong "unsafe" gets reported by the user, a
wrong "safe" does not.

What replaced it decides a *different* question. A **recognizer** returns a
reason to allow a command, or ``None`` to stay out of the way, so it can only
ever widen and a bug costs a dialog rather than a breach. Two ship:

- ``_read_only_command`` — structural. It works only because it refuses to be
  complete: deciding *every* command is Rice's theorem, deciding a few and
  abstaining on the rest is trivial.
- ``_remembered_prefix`` — what a person answered "always" to in an approval
  dialog, persisted in ``shell_allowed_prefixes`` where the policy can see it
  rather than inside the plugin it authorizes.

Both ask about **coverage**, never safety, and both derive their unit through
:func:`command_prefix` — so a grant is stored and matched in one vocabulary,
the same way :func:`render_command` is shared by the dialog and the ledger row.

Adding a third recognizer is a deliberate widening of the authorization
surface, which is what the length of this docstring is for.
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path

from .guest import requests as R
from .policy import SAFE, UNSAFE, Decision, kernel_list


# ──────────────────────────────────────────────────────────────────────
# The read-only recognizer.
# ──────────────────────────────────────────────────────────────────────
#
# This is the classifier's second attempt, and the difference is entirely in
# what it refuses to attempt. The one that died tried to decide *every*
# command: split the line at unquoted ``&&``/``||``/``;``/``|``, match each
# segment against a whitelist of program names, send redirection and
# substitution to approval. Deciding an arbitrary command line is Rice's
# theorem, and it failed in the invisible direction — a wrong "unsafe" gets
# reported, a wrong "safe" does not.
#
# Deciding *a few* commands and abstaining on everything else is not the same
# problem, and it is trivial. Three rules keep it that way:
#
#   1. **Abstain, never deny.** A recognizer returns a reason or ``None``; it
#      has no authority to say "unsafe" because nothing needs it to. A bug
#      here costs a dialog. That is the shape ``_SHELL_RECOGNIZERS`` already
#      had, and it is what makes the rest of this safe to write at all.
#
#   2. **The unit is (program, subcommand).** ``git`` is not read-only;
#      ``git status`` is. ``find`` is read-only right up until ``-exec``.
#      A whitelist keyed on the program name is wrong at the root, which is
#      most of why the first one could not be fixed.
#
#   3. **No parsing.** Any shell at all, any metacharacter anywhere, and this
#      abstains without looking further. It reads the *structured* argv and
#      not the rendered line, because the rendered line is precisely the thing
#      that cannot be reasoned about — quoting, ``$(...)``, backticks, aliases
#      and ``eval`` all beat a parser, so there is no parser.
#
# What it buys is the high-frequency, zero-information prompts: a person
# clicking through ``git status`` forty times a day is being trained not to
# read dialogs, and that training is what the one dialog that mattered will
# run into. What it deliberately never covers is anything that runs arbitrary
# code by design — ``npm install``, ``pip install``, ``pytest``, ``make``,
# ``python script.py``. Those are the minority by count and the majority by
# consequence, and they stay asked forever.
#
# **Sound against carelessness, not against a hostile repository.** Even
# ``git diff`` can invoke an external diff driver named by a repo's
# ``.gitattributes``. That is the documented threat model (see the security
# contract), and it is the line this sits on.

#: Argv elements containing any of these abstain. With ``shell=None`` the
#: handler executes the list as given, so none of them can actually *do*
#: anything — but an argv carrying one was written by somebody who expected a
#: shell, and abstaining on that costs a dialog and settles it.
_SHELL_METACHARACTERS = frozenset("|&;<>$`()\n\r")

#: ``(program, subcommand) -> (allowed_flags, positionals_ok)``. A subcommand
#: of ``""`` means the program takes none. An unknown flag abstains, which is
#: what keeps this a whitelist rather than a deny list.
#:
#: ``positionals_ok`` exists because a bare word means opposite things to
#: neighbouring verbs: to ``git log`` it is a ref or a path to read, to
#: ``git branch`` it is a branch to *create*. Allowing free positionals
#: everywhere would have let ``git remote add origin <url>`` through, which is
#: the concrete bug that makes the pair the right unit.
#:
#: Deliberately absent, each for a reason worth keeping written down:
#:   - ``pytest --collect-only`` — collection *imports* every test module, so
#:     module-level code runs. It looks inert and is not.
#:   - ``find`` — read-only until ``-exec``, and ``-exec`` is a flag.
#:   - ``git config`` — sets exactly as readily as it gets.
#:   - ``cat``/``ls`` — ``sdk.fs.read`` and ``sdk.fs.list`` do these mediated
#:     and better. A dialog is the right nudge toward the SDK.
_READ_ONLY_COMMANDS = {
    ("git", ""): (frozenset({"--version"}), False),
    ("git", "status"): (frozenset({"-s", "--short", "-b", "--branch",
                                   "--porcelain", "--long"}), False),
    ("git", "log"): (frozenset({"--oneline", "--graph", "--decorate", "--stat",
                                "--pretty", "--format", "--author", "--since",
                                "--until", "--max-count", "-n", "--no-merges",
                                "--name-only", "--name-status", "--reverse"}),
                     True),
    ("git", "diff"): (frozenset({"--stat", "--cached", "--staged", "--numstat",
                                 "--shortstat", "--name-only",
                                 "--name-status"}), True),
    ("git", "show"): (frozenset({"--stat", "--oneline", "--pretty", "--format",
                                 "--name-only"}), True),
    ("git", "branch"): (frozenset({"-a", "--all", "-v", "-vv", "--list",
                                   "--show-current"}), False),
    ("git", "remote"): (frozenset({"-v", "--verbose"}), False),
    ("git", "rev-parse"): (frozenset({"--abbrev-ref", "--short",
                                      "--show-toplevel", "--git-dir",
                                      "--is-inside-work-tree"}), True),
    # Version and identity queries: nothing to configure, nothing to write.
    ("python", ""): (frozenset({"--version", "-V"}), False),
    ("python3", ""): (frozenset({"--version", "-V"}), False),
    ("node", ""): (frozenset({"--version", "-v"}), False),
    ("npm", ""): (frozenset({"--version", "-v"}), False),
    ("pip", ""): (frozenset({"--version", "-V"}), False),
    ("pwd", ""): (frozenset(), False),
    ("whoami", ""): (frozenset(), False),
    ("hostname", ""): (frozenset(), False),
}


#: Characters a shell has nothing to say about. A **whitelist**, because the
#: interesting question is not "which characters are dangerous" — that list is
#: never finished — but "which are provably inert", which is short and closed.
#: Letters, digits, space, and the punctuation that appears in program names,
#: flags, versions and POSIX paths.
#:
#: Everything else is out, including several that look harmless: ``~`` (tilde
#: expansion), ``*`` and ``?`` (globs), quotes and ``\`` (both change how the
#: line splits), ``%`` and ``^`` (cmd.exe), ``!`` (history), ``#`` (comment).
_SHELL_INERT = frozenset(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789 -_./:@+,=")


def _exact_argv(args: dict):
    """The argv the handler will execute, or ``None`` if it cannot be known.

    Mirrors ``handlers.fs_net._invocation`` deliberately: a recognizer that
    reasons about a different command than the one that runs is worse than no
    recognizer.

    ``shell=None`` runs the argv as given, so that case is exact. A *named*
    shell (``default``, ``cmd``, ``powershell``) hands the line to something
    that parses it, and what a parser will make of an arbitrary string is the
    question this module refuses to answer — **unless the string contains
    nothing a parser could act on.** ``git pull`` under ``sh -c`` runs exactly
    ``git`` with the argument ``pull``; there is no expansion, no splitting
    ambiguity and no substitution to be wrong about, so whitespace splitting is
    what the shell itself will do.

    That carve-out is not a nicety. ``tool_run_command`` defaults ``shell`` to
    ``"default"``, so without it *every* command in practice came back ``None``
    — which quietly disabled both recognizers. ``git status`` was never
    auto-allowed and no command was ever offered a remembered grant, while the
    tests passed because they construct Requests with no shell at all.
    """
    argv = args.get("argv")
    if args.get("shell") is not None:
        line = argv if isinstance(argv, str) else " ".join(
            str(part) for part in (argv or []))
        if not line.strip() or set(line) - _SHELL_INERT:
            return None
        return line.split()
    if isinstance(argv, str):
        try:
            argv = shlex.split(argv)
        except ValueError:
            return None
    return [str(part) for part in (argv or [])]


# ──────────────────────────────────────────────────────────────────────
# Segmentation.
# ──────────────────────────────────────────────────────────────────────
#
# The dead classifier decomposed a command line with a regex and then decided
# whether each piece was *safe*. Two mistakes, and only the second one was
# fatal. Splitting is fine if you use a real lexer — ``shlex`` knows that the
# ``&&`` in ``git commit -m "fix && ship"`` is inside a quote, which is exactly
# what the regex got wrong. Deciding safety is what nothing can do.
#
# So this splits, and then asks a different question: **is every segment
# already covered by something the user granted?** Coverage is decidable given
# correct segmentation, where safety never was, and an uncovered segment simply
# asks. ``git push && rm -rf /`` needs ``rm`` granted too.
#
# POSIX shells only. ``shlex`` implements *their* quoting; ``cmd`` and
# ``powershell`` do it differently and would be mis-lexed, so they keep only
# the inert fast path in ``_exact_argv``.

#: Separators between commands. Everything else built from these characters is
#: refused, which is how redirects and subshells are excluded without keeping a
#: list of them: ``>``, ``>>``, ``<``, ``>&``, ``<(``, ``(``, ``)`` are all
#: punctuation-only tokens that are not in here.
_SHELL_OPERATORS = frozenset({"&&", "||", ";", "|", "&"})
_SHELL_PUNCTUATION = frozenset("();<>|&")


def _posix_shell(args: dict) -> bool:
    """Whether the shell that will run this is one ``shlex`` describes.

    ``"default"`` is ``/bin/sh`` on POSIX and ``cmd.exe`` on Windows — the
    same argument value meaning two different lexers, which is why this asks
    the platform rather than trusting the name.
    """
    import os

    return args.get("shell") == "default" and os.name != "nt"


def _shell_segments(args: dict):
    """The separate commands a shell line runs, or ``None`` if unknowable.

    Refused outright, because in each case the effect is not in any command
    name and no grant could honestly describe it:

    - **redirection** — ``echo x > /etc/passwd`` writes a file, and granting
      ``echo`` would license writing anywhere. This is the shape of the
      command that started the whole question, and it stays button-less on
      purpose; the right door for it is ``fs.write``, which asks about the
      *path*.
    - **substitution and expansion** — ``$(…)``, backticks, ``$VAR``. The line
      that runs is not the line that was read.
    - **subshells and grouping**, and an ``x=1 cmd`` assignment prefix, where
      the first token is not the program.
    - **anything the lexer cannot read**, such as an unbalanced quote.
    """
    if not _posix_shell(args):
        return None
    argv = args.get("argv")
    line = argv if isinstance(argv, str) else " ".join(
        str(part) for part in (argv or []))
    if not line.strip():
        return None
    try:
        lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return None

    segments, current = [], []
    for token in tokens:
        if "$" in token or "`" in token:
            return None
        if token and set(token) <= _SHELL_PUNCTUATION:
            if token not in _SHELL_OPERATORS:
                return None          # a redirect, a subshell, a grouping
            if not current:
                return None          # an operator with nothing before it
            segments.append(current)
            current = []
            continue
        current.append(token)
    if current:
        segments.append(current)
    if not segments or any("=" in segment[0] for segment in segments):
        return None
    return segments


def _command_segments(args: dict):
    """Every command this Request will run, or ``None``.

    One list for the whole family, so the read-only recognizer, the remembered
    one and the dialog's option all reason about the same decomposition.
    """
    if (argv := _exact_argv(args)):
        return [argv]
    return _shell_segments(args)


def command_prefixes(args: dict) -> list:
    """The prefixes every segment of this command would need granted.

    Empty when any segment has no unit :func:`command_prefix` can describe —
    all or nothing, since a partial grant would leave the dialog appearing
    anyway and teach the person that the button does not work.
    """
    segments = _command_segments(args)
    if not segments:
        return []
    prefixes = []
    for argv in segments:
        if not (prefix := command_prefix(argv)):
            return []
        prefixes.append(prefix)
    return list(dict.fromkeys(prefixes))


def _read_only_command(shown: str, args: dict):
    """Allow a command whose every segment only reads, or abstain.

    ``shown`` is ignored on purpose: the rendered line is for the person and
    the ledger, and reading it back to make a decision is the mistake the
    first classifier was built on.
    """
    segments = _command_segments(args)
    if not segments:
        return None
    named = []
    for argv in segments:
        if not (name := _read_only_segment(argv)):
            return None
        named.append(name)
    return ", ".join(dict.fromkeys(named)) + " only reads"


def _read_only_segment(argv):
    """One segment's name if it is a known read-only invocation, else ""."""
    if not argv:
        return ""
    if any(_SHELL_METACHARACTERS & set(part) for part in argv):
        return ""

    program = argv[0]
    # A bare name only. ``/tmp/somewhere/git`` is not the ``git`` this list is
    # talking about, and resolving which one it is means trusting PATH.
    if "/" in program or "\\" in program:
        return ""
    program = program.lower().removesuffix(".exe")

    rest = argv[1:]
    subcommand = ""
    if rest and not rest[0].startswith("-"):
        subcommand, rest = rest[0].lower(), rest[1:]

    entry = _READ_ONLY_COMMANDS.get((program, subcommand))
    if entry is None:
        return ""
    flags, positionals_ok = entry
    for part in rest:
        if part == "--" and positionals_ok:
            continue                      # ends flag parsing; changes nothing
        if part.startswith("-"):
            # ``--flag=value`` is one token; the value cannot turn a read into
            # a write, so only the name is checked.
            if part.split("=", 1)[0] not in flags:
                return ""
        elif not positionals_ok:
            return ""
    return f"{program} {subcommand}".strip()


def command_prefix(argv) -> str:
    """The unit a shell grant is written down as, or "" for none.

    **Shared by the recognizer that reads the list and the dialog option that
    writes to it**, for the reason ``render_command`` is shared by the dialog
    and the ledger: a grant stored in one vocabulary and matched in another is
    a grant nobody can reason about.

    ``(program, subcommand)``, and a raw string prefix deliberately not —
    ``"git push"`` also prefixes ``"git push && rm -rf /"``. Everything the
    read-only recognizer refuses to look at is refused here too: a named shell
    (the caller passes :func:`_exact_argv`, which is already ``None`` there),
    any metacharacter anywhere, a program that is not a bare name.

    One rule of its own: **a second word that names a file is not a verb.**
    ``python train.py`` reduces to ``python``, not to ``python train.py``,
    because the latter reads like a grant for one known script while actually
    granting whatever that file says tomorrow. Reducing keeps the label honest
    about what is being handed over — a person shown "Always allow: python"
    knows exactly how much that is, and can decline.
    """
    argv = list(argv or [])
    if not argv:
        return ""
    if any(_SHELL_METACHARACTERS & set(part) for part in argv):
        return ""
    program = argv[0]
    if "/" in program or "\\" in program:
        return ""
    # A glob or a tilde in the *program* position names something the shell
    # will pick, not something this grant could describe: ``* --help`` runs
    # whatever happens to sort first in the working directory. Elsewhere in the
    # line they are ordinary argument expansion, and arguments are unchecked by
    # design — so this is deliberately only about ``argv[0]``.
    if set(program) & set("*?[]~"):
        return ""
    program = program.lower().removesuffix(".exe")
    if not program:
        return ""
    rest = argv[1:]
    if rest and not rest[0].startswith("-"):
        word = rest[0]
        if "/" not in word and "\\" not in word and not Path(word).suffix:
            return f"{program} {word.lower()}"
    return program


def _allowed_prefixes() -> set:
    """Command prefixes the user has said may run without being asked.

    Read live on every call, like ``policy._allowed_hosts`` — revoking has to
    be as immediate as granting, and the user revokes with ``/permissions``.

    Inner whitespace is collapsed as well as trimmed, because the stored form
    is two words and ``"git  push"`` is the same grant as ``"git push"``.
    """
    return {" ".join(entry.split()).casefold()
            for entry in kernel_list("shell_allowed_prefixes")}


def _remembered_prefix(shown: str, args: dict):
    """Allow a command whose every segment the user answered "always" to.

    The *remembered* recognizer this section has been describing since it
    shipped empty: a decision persisted where the policy can see it, rather
    than inside the plugin it authorizes.

    **Every** segment, which is what makes chaining safe to allow at all:
    granting ``git push`` does not run ``git push && rm -rf /``, because ``rm``
    is a segment of its own and nobody granted it. The question asked here is
    coverage, not safety — and it is asked of a real lexer's decomposition
    rather than a regex's, which is the other half of why this can exist where
    the old classifier could not.
    """
    prefixes = command_prefixes(args)
    if not prefixes:
        return None
    allowed = _allowed_prefixes()
    if any(prefix.casefold() not in allowed for prefix in prefixes):
        return None
    return "you allowed " + ", ".join(f"`{prefix}`" for prefix in prefixes)


_SHELL_RECOGNIZERS: list = [_read_only_command, _remembered_prefix]


def render_command(args: dict) -> str:
    """The command line a shell Request is asking for, as a person reads it.

    One function because three callers need to agree: the dialog, the ledger
    row, and any future recognizer. A list and the string it would join to
    must describe the same act, or the record and the question drift apart.
    """
    argv = args.get("argv")
    if isinstance(argv, str):
        rendered = argv
    else:
        rendered = " ".join(str(part) for part in (argv or []))
    shell = args.get("shell")
    return f"{rendered} [{shell}]" if shell and shell != "default" else rendered


def classify_shell(kind: str, args: dict) -> Decision:
    """Decide about one shell Request — the entry point ``policy`` calls."""
    shown = render_command(args)
    for recognize in _SHELL_RECOGNIZERS:
        try:
            if (why := recognize(shown, args)):
                return Decision(SAFE, why)
        except Exception:
            # A recognizer that raises abstains. It can only ever widen, so
            # failing it closed costs a dialog and nothing else.
            continue
    verb = "start" if kind == R.PROC_START else "run"
    where = args.get("cwd")
    return Decision(UNSAFE,
                    f"{verb} shell command: {shown[:200]}"
                    + (f" (in {where})" if where else ""))


# ── what a command line did to files (display only) ───────────────────
#
# This answers a *third* question, and it is neither of the two above: not
# "may this run" but "what did that probably touch", asked after the fact so a
# frontend can show the agent deleting a directory the way it shows the agent
# writing a file. Nothing here reaches :func:`classify_shell`, and it must
# stay that way — the moment an authorization decision reads it, this is the
# dead classifier again. ``tests/test_shell_files.py`` pins the separation.
#
# The failure profile is what makes a table like this acceptable here and not
# there. A miss is a row the drawer does not show; a false positive is a file
# it shows that did not change. Both are cosmetic and both are visible to the
# person reading the panel, whereas a wrong "safe" was silent.

#: program → (which operands are the target, whether they are removed).
#:
#: ``all`` — every operand. ``last`` — the final operand only, the earlier ones
#: being sources that are read rather than changed.
_FILE_COMMANDS = {
    "rm": ("all", True), "rmdir": ("all", True), "del": ("all", True),
    "erase": ("all", True), "unlink": ("all", True),
    "mkdir": ("all", False), "md": ("all", False), "touch": ("all", False),
    "cp": ("last", False), "copy": ("last", False), "ln": ("last", False),
    "mv": ("last", True), "move": ("last", True),
}

#: Characters that mean an operand is not confidently a path.
#:
#: Two cases, one rule. Under a shell these are *unexpanded* — ``rm *.log``
#: names nothing until the shell expands it, and we are not the shell. With
#: ``shell=None`` there is no expansion, so ``rm $(cat x)`` really does mean
#: two files literally called ``$(cat`` and ``x)``; recording them would be
#: accurate and would read as a bug to anyone looking at the panel. Either way
#: the honest answer is to abstain, so the same set covers both.
_UNEXPANDED = frozenset("*?[]~$`&|;<>()")


def files_touched(args: dict) -> tuple[list, list]:
    """``(paths, deleted)`` for a shell Request — both empty when unknowable.

    ``paths`` is every file the line touched and ``deleted`` the subset that no
    longer exists, so a caller that only wants "which files" reads one list.
    Paths are resolved against the Request's ``cwd``, because a relative
    ``build/`` means nothing to whoever reads the row later.

    Abstains on any segment it cannot read: an unlisted program, a glob, and —
    via :func:`_command_segments` — redirection, substitution, subshells and
    anything the lexer refuses. Partial answers are not offered; a line that
    both deletes and does something unknown reports nothing at all, since
    "these are the files" is a claim and a half-true one is worse than none.
    """
    segments = _command_segments(args)
    if not segments:
        return [], []

    cwd = str(args.get("cwd") or "") or None
    paths, deleted = [], []
    for argv in segments:
        program = os.path.basename(str(argv[0])).lower()
        program = program[:-4] if program.endswith(".exe") else program
        if (spec := _FILE_COMMANDS.get(program)) is None:
            continue                      # not a file command; says nothing
        which, removes = spec
        operands = [str(a) for a in argv[1:] if not str(a).startswith("-")]
        if not operands:
            continue
        if any(set(a) & _UNEXPANDED for a in operands):
            return [], []                 # a glob: the whole line is a guess
        targets = operands if which == "all" else operands[-1:]
        sources = [] if which == "all" else operands[:-1]
        for operand in targets + (sources if removes else []):
            full = os.path.abspath(os.path.join(cwd, operand)) if cwd else operand
            paths.append(full)
            if removes and (which == "all" or operand in sources):
                deleted.append(full)
    return paths, deleted
