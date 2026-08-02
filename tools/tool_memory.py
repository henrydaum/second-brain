"""Agent memory tool.

Memory lives in the agent-owned ``workspace/memory`` folder as per-topic
markdown files plus a ``MEMORY.md`` index. The system prompt inlines only the
index; this tool reads topic bodies and maintains both files through SDK
Requests. Workspace writes need no approval and cannot widen the tool's reach
into user-owned files.
"""

dependencies_files = []
dependencies_pip = []
requests = ["paths.get", "users.read", "fs.read", "fs.list", "fs.write", "fs.delete"]

import re

from guest.bases import BaseTool

INDEX_FILENAME = "MEMORY.md"
MAX_READ_CHARS = 20_000
TOPIC_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]*$")


def _exists(sdk, path):
    try:
        sdk.fs.list(path)
        return True
    except sdk.Failed:
        return False


def _memory_root(sdk):
    root = sdk.path.join(sdk.paths.get("workspace"), "memory")
    user = sdk.users.read() or {}
    user_id = int(user.get("id") or 1)
    return root if user_id == 1 else sdk.path.join(root, "users", str(user_id))


def _topic_name(raw):
    name = (raw or "").strip()
    if name.lower().endswith(".md"):
        name = name[:-3]
    if not name or not TOPIC_RE.match(name) or name.upper() == "MEMORY":
        raise ValueError(f"Invalid memory topic name: {raw!r}")
    return name


def _topic_path(sdk, root, topic):
    return sdk.path.join(root, f"{_topic_name(topic)}.md")


def _list_topics(sdk, root):
    try:
        entries = sdk.fs.list(root, pattern="*.md", details=True)
    except sdk.Failed:
        return []
    return sorted(
        sdk.path.stem(entry["name"])
        for entry in entries
        if not entry.get("is_dir") and entry.get("name") != INDEX_FILENAME
    )


def _read_index_lines(sdk, root):
    path = sdk.path.join(root, INDEX_FILENAME)
    if not _exists(sdk, path):
        return []
    return [line for line in sdk.fs.read(path).splitlines() if line.strip()]


def _write_index_lines(sdk, root, lines):
    path = sdk.path.join(root, INDEX_FILENAME)
    text = "\n".join(lines).rstrip() + "\n" if lines else ""
    sdk.fs.write(path, text)


def _entry_prefix(topic):
    return f"- [{topic}]({topic}.md)"


def _upsert_index_line(sdk, root, topic, description, content):
    lines = _read_index_lines(sdk, root)
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
    _write_index_lines(sdk, root, lines)


def _remove_index_line(sdk, root, topic):
    prefix = _entry_prefix(topic)
    lines = [line for line in _read_index_lines(sdk, root) if not line.startswith(prefix)]
    _write_index_lines(sdk, root, lines)


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

    agent_prompt = (
        """## Remembering things (memory)
The memory folder holds durable notes that persist across sessions, one markdown file per topic plus a MEMORY.md index (shown in this prompt). The index is a map, not the content: read a topic with the memory tool before answering from it. When the user asks you to remember something — or you learn a long-lived fact, preference, project decision, or lesson — save it to a fittingly named topic and give it a one-line description for the index. Update or forget topics that turn out wrong or stale. Do not store trivial, transient, or unnecessarily sensitive details unless the user explicitly asks."""
    )

    def run(self, sdk, **kwargs):
        """Run a memory action."""
        action = (kwargs.get("action") or "").strip().lower()
        root = _memory_root(sdk)
        try:
            path = _topic_path(sdk, root, kwargs.get("topic"))
        except ValueError as error:
            return sdk.fail(str(error))
        topic = sdk.path.stem(path)

        if action == "read":
            if not _exists(sdk, path):
                known = ", ".join(_list_topics(sdk, root)) or "(none)"
                return sdk.fail(f"No memory topic '{topic}'. Topics: {known}")
            text = sdk.fs.read(path)
            if len(text) > MAX_READ_CHARS:
                text = text[:MAX_READ_CHARS] + "\n\n... (truncated)"
            return sdk.ok(None, llm_summary=text)

        if action in {"save", "append"}:
            content = (kwargs.get("content") or "").strip()
            if not content:
                return sdk.fail(f"'{action}' needs non-empty content.")
            if action == "append" and _exists(sdk, path):
                existing = sdk.fs.read(path).rstrip()
                body = (existing + "\n\n" + content) if existing else content
            else:
                body = content
            sdk.fs.write(path, body.rstrip() + "\n")
            _upsert_index_line(
                sdk, root, topic,
                (kwargs.get("description") or "").strip(), content,
            )
            verb = "appended" if action == "append" else "saved"
            return sdk.ok(None, llm_summary=f"Memory topic '{topic}' {verb}.")

        if action == "forget":
            existed = _exists(sdk, path)
            if existed:
                sdk.fs.delete(path)
            _remove_index_line(sdk, root, topic)
            state = "forgotten" if existed else "did not exist (index cleaned)"
            return sdk.ok(None, llm_summary=f"Memory topic '{topic}' {state}.")

        return sdk.fail(f"Unknown action: {action!r}. Use read, save, append, or forget.")
