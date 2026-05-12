"""Slash-command plugin contract."""

from __future__ import annotations

from state_machine.conversation import FormStep


class BaseCommand:
    """Base command."""
    name: str = ""
    description: str = ""
    category: str = "Other"
    hide_from_help: bool = False
    require_approval: bool = False
    approval_actor_id: str | None = None
    config_settings: list = []

    def form(self, args: dict, context) -> list[FormStep]:
        """Handle form."""
        return []

    def arg_completions(self, context) -> list[str]:
        """Handle arg completions."""
        return []

    def run(self, args: dict, context) -> str | None:
        """Execute `/BaseCommand` for the active session."""
        raise NotImplementedError(f"Command '{self.name}' must implement run()")
