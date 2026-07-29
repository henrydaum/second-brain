"""Check a plugin source file before trying to load it.

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

FAMILIES = ("tools", "tasks", "services", "commands", "frontends", "helpers")

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


class TestPlugin(BaseTool):
    """Test plugin."""
    name = "test_plugin"
    description = (
        "Check a plugin source file against the sandbox contract and report every "
        "problem with its line number and how to fix it. Run this after every edit "
        "while authoring a plugin. It reads the file only — nothing is imported, "
        "executed, registered or unregistered — so it is safe to run on code that "
        "would fail on import."
    )
    parameters = {
        "type": "object",
        "properties": {
            "plugin_path": {"type": "string", "description": "Path to the plugin file to check."},
        },
        "required": ["plugin_path"],
    }
    requires_services = []
    max_calls = 5
    background_safe = True

    def agent_prompt_for(self, sdk) -> str:
        """The authoring workflow, plus a live listing of sandbox drafts."""
        sandbox_root = sdk.paths.get("sandbox_plugins")
        return (
            f"""## Building plugins
You can extend Second Brain by authoring tools, tasks, services, commands and frontends. Write them into {sandbox_root}/<family>/ with the required prefix — tool_foo.py in {sandbox_root}/tools/, command_foo.py in {sandbox_root}/commands/, and so on. You may create, edit and delete files anywhere under that tree without asking, because everything there is contained before it runs.

Plugins are sandboxed. That is the one thing to understand before writing any code: your plugin cannot act, it can only ask. Anything touching disk, network, clock or process is a request to the kernel, made through the `sdk` object every entry point receives — `sdk.fs.read(path)`, not `open(path)`; `sdk.log(...)`, not `logging`; `sdk.db.query(...)`, not a cursor. Requests return their value and raise on failure, so the code reads as straight-line Python.

Workflow:
1. Understand the intended behavior. Ask clarifying questions when a missing decision would materially change the design.
2. Read docs/SDK.md — it is the reference for what `sdk` can do, and its examples are executed by the test suite, so they are correct.
3. Read the matching file in templates/ (tool_template.py, command_template.py, ...) for a worked example of the family you are writing.
4. Write the file into the correct sandbox directory.
5. Call test_plugin(plugin_path=...) after every edit. Fix what it reports and call it again until it says the file conforms.
6. A conforming file is loaded automatically as soon as it is saved.

Rules that are enforced rather than suggested, so a plugin breaking one will not load:
- Import the base class from `guest.bases` — `from guest.bases import BaseTool`. Never import kernel modules (runtime, config, plugins, state_machine, agent, pipeline, events, paths): a box cannot see them.
- No `os`, `sys`, `pathlib`, `subprocess`, `requests`, `open()` or `logging`. Each has an sdk equivalent; `sdk.path.*` covers path arithmetic.
- Exactly one plugin class per file, with a unique `name`.
- Declarations (`name`, `requests`, `exports`, `hooks`, ...) are read from the source without running it, so they must be plain literals.

A plugin importing a library that is not in the standard library still works — declare it in `dependencies_pip` — but the kernel will run that file in a separate process, because it cannot see what the library does. test_plugin tells you when this applies.

## Sandbox plugins
{_drafts(sdk, sandbox_root)}"""
        )

    def run(self, sdk, **kwargs):
        """Run test plugin."""
        raw = (kwargs.get("plugin_path") or "").strip()
        if not raw:
            return sdk.fail("plugin_path is required.")

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
    """Every plugin file currently sitting in the agent's own tree."""
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
        return "None yet. New sandbox plugins will show up here once written."
    return "\n".join(found)
