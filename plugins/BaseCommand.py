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
    approval_actions: tuple[str, ...] = ()
    approval_action_prefixes: tuple[str, ...] = ()
    approval_actor_id: str | None = None
    config_settings: list = []
    dependencies_files: list[str] = []
    dependencies_pip: list[str] = []

    # --- Agent system-prompt contribution ---
    # Guidance injected into the agent's system prompt when this command is in scope.
    # Declare a plain string, or override with ``def agent_prompt(self, ctx)``
    # when the text depends on the session (``ctx`` is a PromptContext:
    # db/services/orchestrator/config/scope/...). The collector accepts either.
    agent_prompt: str = ""

    def __init_subclass__(cls, **kwargs):
        """Internal helper to prevent subclasses sharing mutable metadata."""
        super().__init_subclass__(**kwargs)
        for attr in ("config_settings", "dependencies_files", "dependencies_pip"):
            value = getattr(cls, attr)
            if isinstance(value, list):
                setattr(cls, attr, value.copy())

    def requires_approval(self, args: dict) -> bool:
        """Whether these completed form arguments perform a privileged action."""
        if self.require_approval:
            return True
        action = str((args or {}).get("action") or "")
        return (
            action in self.approval_actions
            or any(
                action.startswith(prefix)
                for prefix in self.approval_action_prefixes
            )
        )

    def form(self, args: dict, context) -> list[FormStep]:
        """Handle form."""
        return []

    def arg_completions(self, context) -> list[str]:
        """Handle arg completions."""
        return []

    def run(self, args: dict, context) -> str | None:
        """Execute `/BaseCommand` for the active session."""
        raise NotImplementedError(f"Command '{self.name}' must implement run()")
