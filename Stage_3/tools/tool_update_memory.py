"""
update_memory tool: Edit the memory.md file in the DATA_DIR.
Uses the same search/replace patching mechanism as build_plugin.
"""

from Stage_3.BaseTool import BaseTool, ToolResult
from paths import DATA_DIR
import os

MAX_MEMORY_LENGTH = 1000


class UpdateMemory(BaseTool):
    name = "update_memory"
    description = (
        "Edit the memory.md file in the DATA_DIR using search-and-replace. "
        "Provide the exact text block to find (search_block) and the text to "
        "replace it with (replace_block). If memory.md does not exist, it will "
        "be created with a placeholder. This file is automatically loaded into "
        "the system prompt memory and can be used to persist notes, reminders, "
        "or custom context across sessions."
    )
    parameters = {
        "type": "object",
        "properties": {
            "search_block": {
                "type": "string",
                "description": "Exact text block in memory.md to find and replace. Must match whitespace exactly.",
            },
            "replace_block": {
                "type": "string",
                "description": "Text to insert in place of the matched search_block.",
            },
        },
        "required": ["search_block", "replace_block"],
    }
    requires_services = []
    agent_enabled = True
    max_calls = 3

    def run(self, context, **kwargs):
        search_block = kwargs.get("search_block", "")
        replace_block = kwargs.get("replace_block", "")

        if not search_block:
            return ToolResult.failed("search_block is required and cannot be empty.")

        memory_path = os.path.join(DATA_DIR, "memory.md")

        # Create file with placeholder if it doesn't exist
        if not os.path.exists(memory_path):
            placeholder = "# Memory\n\nStart adding your notes here.\n"
            try:
                with open(memory_path, "w", encoding="utf-8") as f:
                    f.write(placeholder)
            except Exception as e:
                return ToolResult.failed(f"Failed to create memory.md: {e}")

        # Read current contents
        try:
            with open(memory_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            return ToolResult.failed(f"Failed to read memory.md: {e}")

        # Find and replace
        if search_block not in content:
            return ToolResult.failed(
                f"search_block not found in memory.md. Please ensure the text matches exactly, including whitespace."
            )

        new_content = content.replace(search_block, replace_block, 1)

        # Enforce character limit
        if len(new_content) > MAX_MEMORY_LENGTH:
            return ToolResult.failed(
                f"memory.md would exceed the {MAX_MEMORY_LENGTH}-character limit (would be {len(new_content)} chars). "
                f"Please shorten your changes to keep the file under {MAX_MEMORY_LENGTH} characters. You are encouraged to compact the entire content as needed."
            )

        # Write back
        try:
            with open(memory_path, "w", encoding="utf-8") as f:
                f.write(new_content)
        except Exception as e:
            return ToolResult.failed(f"Failed to write memory.md: {e}")

        return ToolResult(
            success=True,
            data={"path": memory_path},
            llm_summary=f"Successfully updated memory.md — replaced the specified block.",
        )
