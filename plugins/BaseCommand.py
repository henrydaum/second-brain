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

    # --- Agent system-prompt contribution ---
    # Static guidance injected into the agent's system prompt when this command
    # is in scope. Override agent_prompt_for() instead for dynamic text.
    agent_prompt: str = ""

    def agent_prompt_for(self, ctx) -> str:
        """Guidance for the agent system prompt, or '' to contribute nothing.

        ``ctx`` is a PromptContext (db/services/orchestrator/config/scope/...).
        Default returns the static ``agent_prompt``; override for dynamic text."""
        return self.agent_prompt

    def form(self, args: dict, context) -> list[FormStep]:
        """Handle form."""
        return []

    def arg_completions(self, context) -> list[str]:
        """Handle arg completions."""
        return []

    def run(self, args: dict, context) -> str | None:
        """Execute `/BaseCommand` for the active session."""
        raise NotImplementedError(f"Command '{self.name}' must implement run()")
