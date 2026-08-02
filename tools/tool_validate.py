"""Check a source file against the sandbox contract before running it.

Named ``validate`` rather than ``test_plugin`` because it stopped being about
plugins. The validator checks *guest code* — a plugin, a helper, a script — and
the ``plugin_`` in the Request name is a historical accident that is not worth
a breaking rename across every ``requests`` declaration in the store. A script
is now the most common thing checked here, since it is the thing an agent
writes most.

The tool this replaces did something else entirely: it loaded the plugin into
four fake registries, poked the loaded object for contract violations, and then
ran the whole pytest suite in a subprocess. Every part of that is now either
impossible or unnecessary. Loading arbitrary plugin code to inspect it is the
exact act the sandbox exists to mediate; the contract checks it ran by hand are
what the validator does by reading; and pytest measured whether the *app* still
worked, which was never the question being asked.

So this is now one Request and a translation layer. ``plugin.validate`` runs
the same validator the loader runs, which is what makes its verdict the real
one rather than a second opinion — if it says the file conforms, the file
loads. Nothing is imported or executed, so checking a file that would crash on
import is safe, and that is the case an authoring agent hits most.

Deliberately not a code table. The validator's findings already carry a line
number, a message and a fix; re-deriving a code by matching the message text
would be brittle and would add nothing the finding does not say. What this tool
adds is the *next step* — which document explains the rule that was broken —
plus the two structural facts the findings cannot state on their own: whether
the file will load at all, and whether it will be put in a subprocess.
"""

dependencies_files = []
dependencies_pip = []
requests = ["plugin.validate", "fs.list", "paths.get"]

from guest.bases import BaseTool

FAMILIES = ("tools", "tasks", "services", "commands", "frontends", "helpers",
            "scripts")

# Which document explains a finding, keyed by a phrase the validator writes.
# Phrases are lowercase because the message is lowercased before matching, and
# ordered because the first match wins — "kernel side" has to be tried before
# the broader "reaches the", or a kernel import gets pointed at the wrong page.
POINTERS = (
    ("does not parse", "The file is not valid Python; fix the syntax error first."),
    ("is not a request type",
     "docs/SECURITY_CONTRACT_APPENDIX.md lists every Request name. 'requests' is "
     "read by AST and checked against that list."),
    ("kernel side",
     "docs/MIGRATING_PLUGINS.md — a kernel import cannot resolve inside a box, so "
     "ask for what it gave you with a Request instead."),
    ("reaches the database",
     "docs/SDK.md — reach the database through sdk.db.query / sdk.db.write, never "
     "a connection or cursor."),
    ("reaches the",
     "docs/SDK.md — every effect is an sdk.* Request, never a direct call."),
    ("already registered",
     "Names must be unique across built-in, sandbox and installed plugins. Pick "
     "another, or edit the file that already owns this one."),
    ("must be a literal",
     "Declarations are read without running the file, so they must be plain "
     "literals — no tuple(), no comprehension, no f-string."),
    ("more than one plugin class",
     "One plugin per file. Split the extra class into its own file."),
)

DEFAULT_POINTER = "docs/SDK.md is the reference; templates/ has a worked example per family."


class Validate(BaseTool):
    """Validate."""
    name = "validate"
    description = (
        "Check a sandbox source file — a script, a plugin, or a helper — against "
        "the sandbox contract and report every problem with its line number and "
        "how to fix it. Run this after every edit, and before running a script. It "
        "reads the file only — nothing is imported, executed, registered or "
        "unregistered — so it is safe to run on code that would fail on import."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to check."},
        },
        "required": ["path"],
    }
    requires_services = []

    def agent_prompt(self, sdk) -> str:
        """The authoring workflow, plus a live listing of what is in the tree."""
        sandbox_root = sdk.paths.get("workspace")
        return (
            f"""## Writing and running your own code
You can write code into {sandbox_root}/ and run it, without asking permission for either. That is not a loophole: everything under that tree runs in a subprocess and cannot act, only ask. Every effect it performs — disk, network, database, process — is a separate request the kernel judges on its own. So writing code changes what you can *ask for*; it never changes what you are allowed to *affect*.

There are two things you can write, and the difference is whether the kernel has to register it.

**A script** is a file of `sdk` code you run once. No base class, no declarations, just functions that take `sdk`. Put it in {sandbox_root}/scripts/ and run it with run_script. Use this for anything you would otherwise have reached for the shell to do.

**A plugin** is a capability the kernel registers and calls: a tool, task, service, command or frontend. Write it into {sandbox_root}/<family>/ with the required prefix — tool_foo.py in {sandbox_root}/tools/, command_foo.py in {sandbox_root}/commands/, and so on. Write one when the thing should still be there tomorrow and be callable by name.

The one thing to understand before writing either: your code cannot act, it can only ask. Anything touching disk, network, clock or process goes through the `sdk` object every entry point receives — `sdk.fs.read(path)`, not `open(path)`; `sdk.log(...)`, not `logging`; `sdk.db.query(...)`, not a cursor. Requests return their value and raise on failure, so the code reads as straight-line Python.

Workflow:
1. Understand the intended behavior. Ask clarifying questions when a missing decision would materially change the design.
2. Read docs/SDK.md — it is the reference for what `sdk` can do, and its examples are executed by the test suite, so they are correct.
3. For a plugin, read the matching file in templates/ (tool_template.py, command_template.py, ...) for a worked example of the family.
4. Write the file into the correct directory.
5. Call validate(path=...) after every edit. Fix what it reports and call it again until it says the file conforms.
6. A conforming plugin loads automatically as soon as it is saved. A conforming script is ready to run.

Rules that are enforced rather than suggested, so code breaking one will not load:
- Import a base class from `guest.bases` — `from guest.bases import BaseTool`. Never import kernel modules (runtime, config, plugins, state_machine, agent, pipeline, events, paths): a box cannot see them.
- No `os`, `sys`, `pathlib`, `subprocess`, `requests`, `open()` or `logging`. Each has an sdk equivalent; `sdk.path.*` covers path arithmetic.
- Exactly one plugin class per file, with a unique `name`. A script has no class at all — and must not be named with a family prefix, or it will be judged as a plugin and refused.
- Declarations (`name`, `requests`, `exports`, `hooks`, ...) are read from the source without running it, so they must be plain literals.

Code importing a library that is not in the standard library still works — declare it in `dependencies_pip` — but the kernel will run that file in a separate process, because it cannot see what the library does. For a script it also means the user is asked before each run, so keep scripts to the standard library and the SDK when you can. validate tells you when this applies.

## What is in your tree
{_drafts(sdk, sandbox_root)}"""
        )

    def run(self, sdk, **kwargs):
        """Run validate."""
        # ``plugin_path`` accepted as well as ``path``: the parameter was
        # renamed with the tool, and a model that learned the old name from a
        # conversation still in context should not have its call fail on the
        # spelling of an argument that means the same thing.
        raw = (kwargs.get("path") or kwargs.get("plugin_path") or "").strip()
        if not raw:
            return sdk.fail("path is required.")

        try:
            report = sdk.plugins.validate(raw)
        except sdk.Denied as refused:
            return sdk.fail(str(refused))
        except sdk.Failed as failed:
            return sdk.fail(failed.error)

        summary = _render(report)
        # The verdict is the tool's result: a file that will not load is a
        # failed check, so the model retries instead of moving on.
        if report.get("ok"):
            return sdk.ok(report, llm_summary=summary)
        return sdk.fail(summary)


def _render(report: dict) -> str:
    """The whole answer: verdict, findings, isolation, next step."""
    findings = report.get("findings") or []
    lines = [f"Checked {report.get('path')}", "", _verdict(report), ""]

    errors = [f for f in findings if f.get("level") == "error"]
    warnings = [f for f in findings if f.get("level") == "warning"]
    notes = [f for f in findings if f.get("level") == "note"]

    if errors:
        lines.append("### Will not load — fix these")
        lines.extend(_finding(f) for f in errors)
        lines.append("")
    if warnings:
        lines.append("### Loads with a disclaimer")
        lines.extend(_finding(f) for f in warnings)
        lines.append("")
    if notes:
        lines.append("### Advisory")
        lines.extend(_finding(f) for f in notes)
        lines.append("")

    unmediated = report.get("unmediated") or []
    if unmediated:
        lines.append(
            f"This file imports {', '.join(unmediated)}, which the validator "
            "cannot see inside, so the kernel will run it in a **separate "
            "process**. That is decided by what the file imports, not by "
            "anything it declares — there is no way to opt out, and no need "
            "to. Declare each one in dependencies_pip so it gets installed.")
        lines.append("")

    if errors:
        pointers = []
        for finding in errors:
            pointer = _pointer(finding.get("message") or "")
            if pointer not in pointers:
                pointers.append(pointer)
        lines.append("Next step:")
        lines.extend(f"- {p}" for p in pointers)
    elif not findings:
        lines.append("Nothing to fix. Save the file and it will load.")

    return "\n".join(lines).rstrip()


def _verdict(report: dict) -> str:
    """One line that answers the question actually being asked."""
    if not report.get("ok"):
        return "**Will not load.** The errors below have to be fixed first."
    if report.get("disclaimed"):
        return "**Loads, with a disclaimer.** Nothing blocks it; read the warnings."
    return "**Conforms.** This file will load."


def _finding(finding: dict) -> str:
    """One problem, as a line the agent can act on."""
    line = f"- line {finding.get('line')}: {finding.get('message')}"
    fix = (finding.get("fix") or "").strip()
    return f"{line} — use {fix} instead." if fix else f"{line}."


def _pointer(message: str) -> str:
    """Where to read about the rule this message is enforcing."""
    lowered = message.lower()
    for phrase, pointer in POINTERS:
        if phrase in lowered:
            return pointer
    return DEFAULT_POINTER


def _drafts(sdk, sandbox_root) -> str:
    """Every file currently sitting in the agent's own tree.

    Scripts are listed alongside plugins deliberately: a script kept from an
    earlier conversation is the thing most worth knowing about, since the
    cheapest useful move is often to run one that already exists rather than
    write it again.
    """
    found = []
    for family in FAMILIES:
        directory = sdk.path.join(sandbox_root, family)
        try:
            entries = sdk.fs.list(directory, pattern="*.py")
        except Exception:
            continue
        found.extend(f"  {entry}" for entry in sorted(entries)
                     if not sdk.path.name(entry).startswith("_"))
    if not found:
        return "Nothing yet. Anything you write will show up here."
    return "\n".join(found)
