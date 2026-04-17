"""
Tool interface.

Tools are the query layer — they accept arguments, read from the database
and parsers, call other tools, and return structured results.

Unlike tasks (which run in the background on every file), tools are
called on demand and return results immediately to a caller.

Tools become LLM function calls with zero translation:
    - name        -> function name
    - description -> function description
    - parameters  -> JSON schema for arguments

The same tool can be called from the REPL, Telegram, or LLM tool call.
The interface is the same regardless of the caller.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Tool")


@dataclass(init=False)
class ToolResult:
    """
    What a tool returns.

    success:     Did it work?
    error:       Error message if failed.
    data:        Rich result payload for frontend display (tables, lists, etc.).
                 Never sent to Claude directly.
    llm_summary: Text sent to the LLM. All tools must populate this.
                 Standardized human-readable format regardless of tool type.
    attachment_paths: Flat list of file paths for frontend attachment rendering.
                      Never sent to the LLM, unless the file path is an image,
                      since the LLM can receive those.
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
        gui_display_paths: list[str] | None = None,
    ):
        self.success = success
        self.error = error
        self.data = data
        self.llm_summary = llm_summary
        self.attachment_paths = self._normalize_attachment_paths(
            attachment_paths, gui_display_paths
        )

    @staticmethod
    def _normalize_attachment_paths(*path_lists) -> list[str]:
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

    @property
    def gui_display_paths(self) -> list[str]:
        """Backward-compatible alias for older tools and frontends."""
        return self.attachment_paths

    @gui_display_paths.setter
    def gui_display_paths(self, value: list[str]):
        self.attachment_paths = self._normalize_attachment_paths(value)

    def to_dict(self, base_url: str = "") -> dict:
        """Serialize for HTTP API responses.

        Args:
            base_url: If provided, each attachment gets a fetchable ``url``
                      pointing at the ``/files`` endpoint (e.g. ``http://host:port``).
        """
        from pathlib import Path
        from urllib.parse import quote
        from Stage_1.registry import get_modality

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
        return ToolResult(success=False, error=error)


class BaseTool:
    """
    The contract every tool implements.

    Class attributes (override these):
        name              Unique identifier. "keyword_search", "vector_search", etc.
        description       Natural language description. Doubles as the LLM tool description.
        parameters        JSON schema dict describing the input arguments.
        requires_services List of services that must be loaded. Same as tasks.

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
    agent_enabled: bool = True   # Whether the LLM can see and call this tool
    max_calls: int = 3           # Max times the agent can call this tool per message
    background_safe: bool = True # Whether unattended subagents may call this tool

    # --- Config settings this plugin needs ---
    # List of tuples: (title, variable_name, description, default, type_info)
    # Same format as SETTINGS_DATA in config_data.py.
    config_settings: list = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for attr in ("parameters", "requires_services", "config_settings"):
            value = getattr(cls, attr)
            if isinstance(value, (dict, list)):
                setattr(cls, attr, value.copy())

    def run(self, context, **kwargs) -> ToolResult:
        raise NotImplementedError(f"Tool '{self.name}' must implement run()")

    def to_schema(self) -> dict:
        """Export as an OpenAI-compatible function schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }


