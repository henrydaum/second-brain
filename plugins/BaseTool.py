"""
Tool interface.

Tools are the on-demand action and retrieval layer of Second Brain.
A tool accepts structured input, inspects local state or external systems,
and returns a ToolResult that is useful both to frontends and to the LLM.

Unlike tasks, tools do not run automatically over every file. They are
called explicitly by the agent, the UI, or other tools and return
immediately.

Tool schemas map directly into LLM function calling:
    - name        -> function name
    - description -> function description
    - parameters  -> JSON schema for arguments

The same tool contract is used everywhere: REPL, Telegram, HTTP, and agent.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Tool")


@dataclass(init=False)
class ToolResult:
    """
    The standardized result returned by every tool.

    success:
        Whether the tool call succeeded.
    error:
        Human-readable failure reason when success is False.
    data:
        Structured payload for frontends, tables, or debugging. This is not
        sent directly to the LLM.
    llm_summary:
        Concise model-facing summary of what happened. On success, this should
        carry the facts, changes, paths, counts, or constraints the model
        needs for its next step.
    attachment_paths:
        Local file paths for frontend rendering. These are not sent directly to
        the LLM, although image paths may later be passed back on a model call.
    """
    success: bool = True
    error: str = ""
    data: Any = None
    llm_summary: str = ""
    attachment_paths: list[str] = field(default_factory=list)

    def __init__(
        self,
        success: bool = True,
        error: str = "",
        data: Any = None,
        llm_summary: str = "",
        attachment_paths: list[str] | None = None,
    ):
        """Initialize the tool result."""
        self.success = success
        self.error = error
        self.data = data
        self.llm_summary = llm_summary
        self.attachment_paths = self._normalize_attachment_paths(attachment_paths)

    @staticmethod
    def _normalize_attachment_paths(*path_lists) -> list[str]:
        """Internal helper to normalize attachment paths."""
        normalized = []
        seen = set()
        for paths in path_lists:
            if not paths:
                continue
            for path in paths:
                if path in seen:
                    continue
                seen.add(path)
                normalized.append(path)
        return normalized

    def to_dict(self, base_url: str = "") -> dict:
        """Serialize for HTTP API responses.

        Args:
            base_url: If provided, each attachment gets a fetchable ``url``
                      pointing at the ``/files`` endpoint (e.g. ``http://host:port``).
        """
        from pathlib import Path
        from urllib.parse import quote
        from plugins.services.helpers.parser_registry import get_modality

        attachments = []
        for p in self.attachment_paths:
            modality = get_modality(Path(p).suffix)
            att = {"path": p, "modality": modality}
            if base_url:
                att["url"] = f"{base_url}/files?path={quote(p, safe='')}"
            attachments.append(att)

        return {
            "success": self.success,
            "error": self.error,
            "data": self.data,
            "llm_summary": self.llm_summary,
            "attachments": attachments,
        }

    @staticmethod
    def failed(error: str) -> "ToolResult":
        """Handle failed."""
        return ToolResult(success=False, error=error)


class BaseTool:
    """
    The contract every tool implements.

    Class attributes (override these):
        name:
            Stable identifier used everywhere the tool is referenced.
        description:
            Short operational description. This is also the LLM-visible tool
            description, so it should explain what the tool does, when to use
            it, and any important limits.
        parameters:
            JSON Schema describing the input arguments.
        requires_services:
            Service names that must be loaded before the tool can run.

    Methods (override these):
        run(context, **kwargs) -> ToolResult
    """

    # --- Identity ---
    name: str = ""
    description: str = ""
    parameters: dict = {}

    # --- Service requirements ---
    requires_services: list[str] = []

    # --- Agent controls ---
    max_calls: int = 3           # Max times the agent can call this tool per message
    background_safe: bool = True # When False, refuses to run from a non-active session
    plan_mode_safe: bool = True  # When False, cannot run while the session is drafting a plan

    # --- Discovery ---
    # When False, the plugin discoverer skips this tool. Use for tools that
    # need per-call construction args and are instantiated manually instead.
    auto_register: bool = True

    # --- Config settings this plugin needs ---
    # Each entry is a tuple:
    # (title, variable_name, description, default, type_info)
    # Same format as SETTINGS_DATA in config_data.py.
    config_settings: list = []

    def __init_subclass__(cls, **kwargs):
        """Internal helper to handle init subclass."""
        super().__init_subclass__(**kwargs)
        for attr in ("parameters", "requires_services", "config_settings"):
            value = getattr(cls, attr)
            if isinstance(value, (dict, list)):
                setattr(cls, attr, value.copy())

    def run(self, context, **kwargs) -> ToolResult:
        """Execute the base tool tool."""
        raise NotImplementedError(f"Tool '{self.name}' must implement run()")

    def to_schema(self) -> dict:
        """Export the tool as an OpenAI-compatible function schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }
