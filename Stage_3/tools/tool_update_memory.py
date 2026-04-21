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
        "Update persistent memory in memory.md (survives across sessions) using exact search-and-replace. "
        "Call this proactively whenever you learn something worth remembering for future conversations: "
        "facts about the user, lessons about how a tool behaves, corrections the user gave you, "
        "failure modes you want to avoid next time, or useful patterns you discovered. "
        "Be rigorous — a short note now saves a repeated mistake later. "
        "Skip only transient task state that won't matter after this conversation. "
        "If memory.md does not exist, it will be created automatically."
    )
    parameters = {
        "type": "object",
        "properties": {
            "search_block": {
                "type": "string",
                "description": "Exact text block in memory.md to find and replace. Whitespace must match exactly. Pass an empty string to append replace_block to the end of memory.md instead of replacing.",
            },
            "replace_block": {
                "type": "string",
                "description": "Replacement text to insert in place of the matched search_block, or the text to append when search_block is empty.",
            },
        },
        "required": ["search_block", "replace_block"],
    }
    requires_services = ["llm"]
    agent_enabled = True
    max_calls = 3

    _COMPACT_PROMPT = (
        "You are compacting a persistent memory note for Second Brain. The current memory.md content "
        f"exceeds the {MAX_MEMORY_LENGTH}-character hard limit and must be shortened.\n\n"
        "Rewrite it to fit comfortably under the limit while preserving every distinct fact, preference, "
        "correction, and pattern worth remembering across future sessions. Merge duplicates, drop filler "
        "words, and prefer terse bullet points. Keep markdown structure if helpful. Output ONLY the new "
        "memory.md content — no preamble, no code fences, no commentary."
    )

    def _compact_via_llm(self, llm, content: str) -> str | None:
        messages = [
            {"role": "system", "content": self._COMPACT_PROMPT},
            {"role": "user", "content": content},
        ]
        try:
            response = llm.invoke(messages)
        except Exception:
            return None
        text = (response.content or "").strip()
        if text.startswith("```"):
            text = text.strip("`")
            if "\n" in text:
                text = text.split("\n", 1)[1]
            text = text.rstrip("`").strip()
        return text or None

    def run(self, context, **kwargs):
        search_block = kwargs.get("search_block", "")
        replace_block = kwargs.get("replace_block", "")

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

        # Append mode: empty search_block means append replace_block to end of file
        if not search_block:
            if not replace_block:
                return ToolResult.failed("Both search_block and replace_block are empty — nothing to do.")
            sep = "" if content.endswith("\n") or not content else "\n"
            new_content = content + sep + replace_block
            if not new_content.endswith("\n"):
                new_content += "\n"
        else:
            # Find and replace
            if search_block not in content:
                return ToolResult.failed(
                    "search_block not found in memory.md. Whitespace must match exactly. "
                    f"Current memory.md contents ({len(content)} chars):\n---\n{content}\n---"
                )
            new_content = content.replace(search_block, replace_block, 1)

        # Enforce character limit — auto-compact via LLM when exceeded
        compacted_note = ""
        if len(new_content) > MAX_MEMORY_LENGTH:
            llm = context.services.get("llm")
            if llm is None or not getattr(llm, "loaded", False):
                return ToolResult.failed(
                    f"memory.md would exceed the {MAX_MEMORY_LENGTH}-character limit "
                    f"({len(new_content)} chars) and the LLM service is unavailable to compact it."
                )
            compacted = self._compact_via_llm(llm, new_content)
            if not compacted:
                return ToolResult.failed(
                    f"memory.md would exceed the {MAX_MEMORY_LENGTH}-character limit "
                    f"({len(new_content)} chars) and automatic compaction failed."
                )
            if len(compacted) > MAX_MEMORY_LENGTH:
                return ToolResult.failed(
                    f"Automatic compaction produced {len(compacted)} chars, still over the "
                    f"{MAX_MEMORY_LENGTH}-character limit. Please shorten your update manually."
                )
            compacted_note = f" Auto-compacted from {len(new_content)} to {len(compacted)} chars via LLM."
            new_content = compacted

        # Write back
        try:
            with open(memory_path, "w", encoding="utf-8") as f:
                f.write(new_content)
        except Exception as e:
            return ToolResult.failed(f"Failed to write memory.md: {e}")

        return ToolResult(
            success=True,
            data={"path": memory_path},
            llm_summary=f"Successfully updated memory.md — replaced the specified block.{compacted_note}",
        )
