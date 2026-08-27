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
requests = ["plugin.validate", "session.get"]

from guest.bases import BaseTool

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
        "how to fix it. Run this after every edit to a plugin: a plugin that will "
        "not load fails silently — the kernel logs it and tells the user, and you "
        "are never told at all — so this is the only way to find out. A script needs it less: "
        "run_script runs this same check in its own preflight and hands you the "
        "same errors, so write the script and run it, rather than use this tool. "
        "This tool is safe to call and never requires user permission."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to check."},
        },
        "required": ["path"],
    }
    requires_services = []

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

        summary = _render(report, _mode(sdk))
        # The verdict is the tool's result: a file that will not load is a
        # failed check, so the model retries instead of moving on.
        if report.get("ok"):
            return sdk.ok(report, llm_summary=summary)
        return sdk.fail(summary)


def _mode(sdk) -> str:
    """The conversation's security mode, or ``ask`` if it cannot be read.

    Falling back to ``ask`` rather than ``lockdown`` keeps a failure here from
    inventing a refusal that is not going to happen.
    """
    try:
        return (sdk.session.get() or {}).get("mode") or "ask"
    except Exception:
        return "ask"


def _render(report: dict, mode: str = "ask") -> str:
    """The whole answer: verdict, findings, isolation, next step."""
    findings = report.get("findings") or []
    lines = [f"Checked {report.get('path')}", "", _verdict(report, mode), ""]

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


def _verdict(report: dict, mode: str = "ask") -> str:
    """One line that answers the question actually being asked.

    The disclaimer branch has to know the mode, because otherwise it says
    something false. An unmediated import makes the kernel classify a script
    launch as unsafe, and under ``lockdown`` everything that would have been
    asked about is refused without asking — so "nothing blocks it" is exactly
    wrong there, and it is wrong in the direction that wastes a run. An agent
    reading it goes on to ``run_script``, is refused, and has to work out from
    the refusal that the file it was just told was fine was never going to
    launch.

    Nothing about the policy changes here. This is the same verdict, told to
    someone who is in a mode where the disclaimer is a refusal.
    """
    if not report.get("ok"):
        return "**Will not load.** The errors below have to be fixed first."
    if report.get("disclaimed"):
        if mode == "lockdown":
            libraries = ", ".join(report.get("unmediated") or []) or "a library"
            return (
                f"**Loads, but will not run in lockdown.** It imports "
                f"{libraries}, whose actions the kernel cannot mediate, so "
                "launching it needs approval — and lockdown refuses that "
                "without asking. Rewrite it using only the standard library "
                "and sdk.*, or say plainly that this step cannot be done in "
                "this mode.")
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
