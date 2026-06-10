"""Slash command plugin for `/skills`."""

dependencies_files = ['services/service_skills.py']
dependencies_pip = []

import re

from paths import SANDBOX_PLUGINS
from plugins.BaseCommand import BaseCommand
from state_machine.conversation import FormStep

LIST, CREATE, DELETE = "list", "create", "delete"
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")

TEMPLATE = """---
name: {name}
description: {description}
---

# {name}

<Step-by-step instructions the agent should follow when this skill applies.
Reference support files in this folder by name; the agent can read them with
a file tool. Delete this placeholder text.>
"""


class SkillsCommand(BaseCommand):
    """Slash-command handler for `/skills`."""
    name = "skills"
    description = "List, create, or delete agent skills"
    category = "Agent"

    def form(self, args, context):
        """Handle form."""
        steps = [FormStep("action", "What do you want to do with skills?", True,
                          enum=[LIST, CREATE, DELETE],
                          enum_labels=["List skills", "Create a skill", "Delete a sandbox skill"], columns=1)]
        if args.get("action") == CREATE:
            steps += [
                FormStep("skill_name", "Skill name (lowercase, digits, dashes — e.g. trip-planning).", True),
                FormStep("skill_description", "One-line description: what the skill covers and when to use it.", True),
            ]
        elif args.get("action") == DELETE:
            names = _sandbox_skill_names()
            steps.append(FormStep("skill_name", "Which sandbox skill should be deleted? (Built-in and installed skills belong to their packages.)",
                                  True, enum=names or ["(none)"], columns=1))
        return steps

    def run(self, args, context):
        """Execute `/skills` for the active session."""
        action = args.get("action") or LIST
        service = (getattr(context, "services", None) or {}).get("skills")
        if action == LIST:
            if service is None or not getattr(service, "loaded", False):
                return "Skills service is not loaded."
            skills = service.list()
            if not skills:
                return f"No skills installed. Create one with /skills, or drop a folder with a SKILL.md under {SANDBOX_PLUGINS / 'skills'}."
            return "Installed skills:\n" + "\n".join(f"- {s.name} ({s.root_name}): {s.description}" for s in skills)
        if action == CREATE:
            return _create(args)
        if action == DELETE:
            return _delete(args)
        return f"Unknown action: {action}"


def _sandbox_root():
    return SANDBOX_PLUGINS / "skills"


def _sandbox_skill_names() -> list[str]:
    root = _sandbox_root()
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "SKILL.md").is_file())


def _create(args) -> str:
    name = (args.get("skill_name") or "").strip().lower()
    description = " ".join((args.get("skill_description") or "").split())
    if not _NAME_RE.match(name):
        return "Skill names must be lowercase letters, digits, and dashes, starting with a letter or digit."
    if not description:
        return "A one-line description is required — it is all the agent sees until it loads the skill."
    folder = _sandbox_root() / name
    skill_file = folder / "SKILL.md"
    if skill_file.exists():
        return f"Skill '{name}' already exists at {skill_file}."
    folder.mkdir(parents=True, exist_ok=True)
    skill_file.write_text(TEMPLATE.format(name=name, description=description), encoding="utf-8")
    return (f"Created skill '{name}' at {skill_file}.\n"
            "Edit the SKILL.md body with the actual instructions; the index updates automatically.")


def _delete(args) -> str:
    name = (args.get("skill_name") or "").strip()
    if name in ("", "(none)"):
        return "No sandbox skills to delete."
    folder = _sandbox_root() / name
    if not (folder / "SKILL.md").is_file():
        return f"No sandbox skill named '{name}'."
    for path in sorted(folder.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()
    folder.rmdir()
    return f"Deleted sandbox skill '{name}'."
