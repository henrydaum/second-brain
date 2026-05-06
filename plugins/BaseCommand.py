"""Slash-command plugin contract."""

from __future__ import annotations

from state_machine.conversation import FormStep


class BaseCommand:
    name: str = ""
    description: str = ""
    category: str = "Other"
    hide_from_help: bool = False
    require_approval: bool = False
    approval_actor_id: str | None = None
    config_settings: list = []

    def form(self, args: dict, context) -> list[FormStep]:
        return []

    def arg_completions(self, context) -> list[str]:
        return []

    def run(self, args: dict, context) -> str | None:
        raise NotImplementedError(f"Command '{self.name}' must implement run()")
