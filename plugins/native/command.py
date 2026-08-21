"""The native face of a slash-command adapter.

Nothing subclasses this by hand. A command is sandboxed code, and
``sandbox.bridge`` builds a subclass of this class at load whose ``run``
forwards into a box. What lives here is the half the *command registry* needs
to see: the declarations it reads and the approval question it asks.

See ``templates/command_template.py`` for what an author actually writes.
"""

from __future__ import annotations

from state_machine.conversation import FormStep


class BaseCommand:
    """What the kernel sees of a slash command."""
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
    # when the text depends on live state. ``ctx`` is a PromptContext,
    # carrying the session facts ``prompt_cues.SESSION_FACTS`` names —
    # session_key, conversation_id, user_id, profile_name, frontend_name,
    # security_mode — plus db/services/orchestrator/config/scope. The
    # collector accepts either shape.
    agent_prompt: str = ""

    # When a method-shaped contribution goes stale, and therefore which
    # block of the prompt it rides in. See ``prompt_cues.py`` for the
    # ladder; "" means the default rung.
    agent_prompt_refresh: str = ""

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

    def run(self, args: dict, context) -> str | None:
        """Execute `/BaseCommand` for the active session."""
        raise NotImplementedError(f"Command '{self.name}' must implement run()")
