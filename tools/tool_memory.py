"""Agent memory tool.

Memory is a folder of per-topic markdown files plus a ``MEMORY.md`` index
(see ``plugins/helpers/memory_paths.py`` in the kernel). The system prompt
inlines only the index; this tool is how the agent reads topic bodies and
writes new memories. Writes are unapproved by design — they are hard
path-validated to the current user's memory folder and nothing else.
"""

dependencies_files = []
dependencies_pip = []

from plugins.BaseTool import BaseTool, ToolResult
from plugins.helpers.memory_paths import INDEX_FILENAME, list_topics, memory_root, topic_path

MAX_READ_CHARS = 20_000


class Memory(BaseTool):
    """Read and write per-topic memory files."""
    name = "memory"
    description = (
        "Read, save, append, or forget durable memory topics. Each topic is a "
        "markdown file; the MEMORY.md index in your system prompt maps topics. "
        "Read a topic before answering from it — the index is a map, not the content."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["read", "save", "append", "forget"],
                "description": "read: return a topic's content. save: create/overwrite a topic. append: add to a topic. forget: delete a topic and its index line.",
            },
            "topic": {
                "type": "string",
                "description": "Topic name (becomes <topic>.md). Letters, digits, dots, dashes, underscores, spaces.",
            },
            "content": {"type": "string", "description": "Markdown body for save/append."},
            "description": {
                "type": "string",
                "description": "One-line index hook for save/append — what this topic holds and when to read it.",
            },
        },
        "required": ["action", "topic"],
    }
    requires_services = []
    max_calls = 10
    background_safe = True

    agent_prompt = (
        """## Remembering things (memory)
The memory folder holds durable notes that persist across sessions, one markdown file per topic plus a MEMORY.md index (shown in this prompt). The index is a map, not the content: read a topic with the memory tool before answering from it. When the user asks you to remember something — or you learn a long-lived fact, preference, project decision, or lesson — save it to a fittingly named topic and give it a one-line description for the index. Update or forget topics that turn out wrong or stale. Do not store trivial, transient, or unnecessarily sensitive details unless the user explicitly asks."""
    )

    def run(self, context, **kwargs) -> ToolResult:
        """Run a memory action."""
        action = (kwargs.get("action") or "").strip().lower()
        topic = (kwargs.get("topic") or "").strip()
        user_id = getattr(context, "user_id", None)
        try:
            path = topic_path(topic, user_id)
        except ValueError as e:
            return ToolResult.failed(str(e))

        if action == "read":
            if not path.exists():
                known = ", ".join(p.stem for p in list_topics(user_id)) or "(none)"
                return ToolResult.failed(f"No memory topic '{path.stem}'. Topics: {known}")
            text = path.read_text(encoding="utf-8")
            if len(text) > MAX_READ_CHARS:
                text = text[:MAX_READ_CHARS] + "\n\n... (truncated)"
            return ToolResult(llm_summary=text)

        if action in {"save", "append"}:
            content = (kwargs.get("content") or "").strip()
            if not content:
                return ToolResult.failed(f"'{action}' needs non-empty content.")
            path.parent.mkdir(parents=True, exist_ok=True)
            if action == "append" and path.exists():
                existing = path.read_text(encoding="utf-8").rstrip()
                body = (existing + "\n\n" + content) if existing else content
            else:
                body = content
            path.write_text(body.rstrip() + "\n", encoding="utf-8")
            _upsert_index_line(user_id, path.stem, (kwargs.get("description") or "").strip(), content)
            return ToolResult(llm_summary=f"Memory topic '{path.stem}' {'appended' if action == 'append' else 'saved'}.")

        if action == "forget":
            existed = path.exists()
            path.unlink(missing_ok=True)
            _remove_index_line(user_id, path.stem)
            return ToolResult(llm_summary=f"Memory topic '{path.stem}' {'forgotten' if existed else 'did not exist (index cleaned)'}.")

        return ToolResult.failed(f"Unknown action: {action!r}. Use read, save, append, or forget.")


def _index_path(user_id):
    return memory_root(user_id) / INDEX_FILENAME


def _read_index_lines(user_id) -> list[str]:
    path = _index_path(user_id)
    if not path.exists():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_index_lines(user_id, lines: list[str]) -> None:
    path = _index_path(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n" if lines else "", encoding="utf-8")


def _entry_prefix(topic: str) -> str:
    return f"- [{topic}]({topic}.md)"


def _upsert_index_line(user_id, topic: str, description: str, content: str) -> None:
    """Create or update the topic's one-line index entry.

    Without an explicit description, a new entry falls back to the first
    content line; an existing entry's description is left alone.
    """
    lines = _read_index_lines(user_id)
    prefix = _entry_prefix(topic)
    existing = next((i for i, line in enumerate(lines) if line.startswith(prefix)), None)
    if existing is not None and not description:
        return
    hook = description or (content.splitlines()[0].strip() if content else "")[:120]
    entry = f"{prefix} - {hook}" if hook else prefix
    if existing is not None:
        lines[existing] = entry
    else:
        lines.append(entry)
    _write_index_lines(user_id, lines)


def _remove_index_line(user_id, topic: str) -> None:
    prefix = _entry_prefix(topic)
    lines = [line for line in _read_index_lines(user_id) if not line.startswith(prefix)]
    _write_index_lines(user_id, lines)
