"""Planning one plugin's migration.

The validator already knows how to say *"you called ``open()``; use
``sdk.fs.read``"*. Pointed at a plugin that has **not** been migrated yet,
that is not a list of errors — it is the checklist.

So this reads a native plugin and reports what converting it involves: which
effects it performs and the Request each becomes, what its entry point is,
what declarations to add, and how its ``run`` signature changes. Nothing is
rewritten; the point is to make the work visible before it starts.

Pair it with :mod:`sandbox.parity`, which answers the other half — whether the
rewrite still returns the same thing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .validator import ERROR, WARNING, FAMILIES, validate_file

# How ``run`` is spelled before and after, per family. The native contract
# differs per family and the sandboxed one mirrors it with ``sdk`` in place of
# ``context`` — the argument *order* changes for commands and tasks, which is
# the easiest thing to get wrong.
SIGNATURES = {
    "tool": ("run(self, context, **kwargs)", "run(self, sdk, **kwargs)"),
    "task": ("run(self, paths, context)", "run(self, sdk, paths)"),
    "command": ("run(self, args, context)", "run(self, sdk, args)"),
    "service": ("_load(self)", "start(self, sdk)  # plus stop(self, sdk)"),
    "frontend": ("start(self)", "start(self, sdk)"),
}

# Declarations worth adding while you are in the file anyway.
SUGGESTED = {
    "service": ["exports = [...]  # methods reachable via service.call"],
    "tool": ["requests = [...]  # advisory: what this expects to ask for"],
}


@dataclass
class Step:
    """One thing to change, and what to change it to."""

    line: int
    what: str
    becomes: str = ""

    def render(self) -> str:
        """One line for a checklist."""
        arrow = f"  ->  {self.becomes}" if self.becomes else ""
        return f"  line {self.line:>4}: {self.what}{arrow}"


@dataclass
class Plan:
    """What migrating one plugin involves."""

    path: str
    family: str = ""
    entry: str = ""
    steps: list = field(default_factory=list)
    disclaimers: list = field(default_factory=list)
    declarations: dict = field(default_factory=dict)
    already_migrated: bool = False

    @property
    def requests(self) -> list:
        """The distinct Requests this plugin will need."""
        found = set()
        for step in self.steps:
            for part in step.becomes.split("/"):
                part = part.strip()
                if part.startswith("sdk."):
                    found.add(part.split()[0])
        return sorted(found)

    def render(self) -> str:
        """The checklist."""
        name = Path(self.path).name
        if self.already_migrated:
            return f"{name}: already written against the SDK."

        lines = [f"{name}  ({self.family or 'helper'})"]
        if self.entry:
            lines.append(f"  entry: {self.entry}")

        before, after = SIGNATURES.get(self.family, ("", ""))
        if before:
            lines.append(f"  signature: {before}  ->  {after}")

        if self.steps:
            lines.append("")
            lines.append(f"  {len(self.steps)} effect(s) to convert:")
            lines.extend(step.render() for step in self.steps)

        if self.requests:
            lines.append("")
            lines.append(f"  Requests needed: {', '.join(self.requests)}")

        for note in self.disclaimers:
            lines.append("")
            lines.append(f"  ! {note}")

        for suggestion in SUGGESTED.get(self.family, []):
            lines.append(f"  + consider: {suggestion}")

        if not self.steps and not self.disclaimers:
            lines.append("")
            lines.append("  No direct effects — this one is mostly a rename.")
        return "\n".join(lines)


def plan(path) -> Plan:
    """Read a native plugin and report what migrating it involves."""
    path = Path(path)
    report = validate_file(path)
    declarations = dict(report.declarations)
    family = declarations.get("family", "")
    if not family:
        stem_family = path.stem.split("_")[0]
        family = stem_family if stem_family in FAMILIES else ""

    source = report.source
    migrated = "guest.bases" in source or "guest import" in source

    steps = []
    disclaimers = []
    for finding in report.findings:
        if finding.level == ERROR and finding.fix:
            steps.append(Step(finding.line, finding.message, finding.fix))
        elif finding.level == WARNING:
            disclaimers.append(finding.message)

    return Plan(
        path=str(path),
        family=family,
        entry=_entry_name(source),
        steps=steps,
        disclaimers=disclaimers,
        declarations=declarations,
        already_migrated=migrated,
    )


def _entry_name(source: str) -> str:
    """The plugin class's name, read out of the source."""
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""
    wanted = set(FAMILIES.values())
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id in wanted:
                return node.name
    return ""


def plan_tree(root, families=("tool", "task", "command")) -> list:
    """Plan every unmigrated plugin under a directory, easiest first.

    Ordered by how much there is to convert, because the first migrations
    should be the ones that prove the path rather than the ones that test it.
    """
    root = Path(root)
    plans = []
    for path in sorted(root.rglob("*.py")):
        stem_family = path.stem.split("_")[0]
        if stem_family not in families:
            continue
        found = plan(path)
        if not found.already_migrated:
            plans.append(found)
    return sorted(plans, key=lambda p: (len(p.steps), len(p.disclaimers)))
