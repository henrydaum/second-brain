"""
Shared input routing for all Second Brain frontends.

``route_input()`` is the single channel that GUI, REPL, and API all use
for chat-style interaction: text starting with ``/`` is dispatched as a
slash command; everything else is sent to the agent as a chat message.
"""

from dataclasses import dataclass, field

from frontend.shared.token_stripper import strip_model_tokens


@dataclass
class InputResult:
    """Value returned by :func:`route_input`.

    Attributes:
        type:        ``"command"``, ``"chat"``, or ``"error"``
        text:        Response text (command output or agent reply).
        attachments: File paths collected from tool ``gui_display_paths``
                     during agent chat. Empty for commands and errors.
    """
    type: str
    text: str = ""
    attachments: list = field(default_factory=list)


def route_input(text, registry, agent, image_paths=None):
    """Route user text to either a slash command or the agent.

    Parameters:
        text:         Raw user input string.
        registry:     A :class:`CommandRegistry` instance.
        agent:        An :class:`Agent` instance, or ``None`` if the LLM is
                      not loaded.
        image_paths:  Optional list of local image file paths to pass to the
                      agent for vision (e.g. Telegram photo uploads).

    Returns:
        An :class:`InputResult` with the response.
    """
    text = text.strip()
    if not text:
        return InputResult("error", "")

    # --- Slash command ---
    if text.startswith("/"):
        cmd_text = text[1:]
        parts = cmd_text.split(maxsplit=1)
        cmd_name = parts[0].lower() if parts else ""
        arg = parts[1].strip() if len(parts) > 1 else ""
        output = registry.dispatch(cmd_name, arg)
        return InputResult("command", output or "")

    # --- Chat message ---
    if agent is None:
        return InputResult(
            "error",
            "LLM is not loaded. Use /load llm to load it, "
            "or /services to check status.",
        )

    # Temporarily wrap on_tool_result to collect gui_display_paths
    # while preserving any existing callback (e.g. the GUI renderer).
    original_callback = agent.on_tool_result
    collected_paths = []

    def _collecting_wrapper(tool_name, result):
        if result.gui_display_paths:
            collected_paths.extend(result.gui_display_paths)
        if original_callback:
            original_callback(tool_name, result)

    agent.on_tool_result = _collecting_wrapper
    try:
        response = agent.chat(text, image_paths=image_paths)
    finally:
        agent.on_tool_result = original_callback

    clean, _ = strip_model_tokens(response or "")
    return InputResult("chat", clean, collected_paths)
