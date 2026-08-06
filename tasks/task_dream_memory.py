"""Nightly memory distiller.

The ``dream_memory`` event reads recent user-owned conversations and the
current user's ``workspace/memory`` folder, asks a configured LLM for a strict
JSON patch, and applies that patch without approval. Memory layout and topic
validation remain plugin-owned; no memory implementation crosses into the
kernel.
"""

dependencies_files = []
dependencies_pip = []

import json
import re
import time

from guest.bases import BaseTask

INDEX_FILENAME = "MEMORY.md"
MAX_CONVERSATIONS = 25
MAX_TRANSCRIPT_CHARS = 24_000
MAX_MEMORY_CHARS = 16_000
MAX_TOPICS = 40
TOPIC_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]*$")

SYSTEM_PROMPT = (
    "You maintain Second Brain's durable memory: a MEMORY.md index plus one "
    "markdown file per topic. Return only valid JSON describing a patch — "
    "compact standing context, not a chat summary."
)

USER_TEMPLATE = """Current memory index (MEMORY.md):
<index>
{index}
</index>

Current memory topics:
<topics>
{topics}
</topics>

Recent human-facing conversations:
<conversations>
{conversations}
</conversations>

Return JSON with exactly these keys:
{{
  "topics": {{"<topic-name>": "full markdown replacement for that topic"}},
  "forget": ["topic names to delete outright"],
  "index": {{"<topic-name>": "one-line hook: what the topic holds and when to read it"}},
  "changes": ["short bullets for additions, merges, deletions"],
  "skipped": ["short bullets for ignored transient items"]
}}

Rules:
- This is a PATCH: only include topics you are changing; untouched topics survive as-is.
- Topic names use letters, digits, dots, dashes, underscores, spaces.
- Keep each topic short, specific, and reusable; give changed topics an index hook.
- Preserve durable preferences, project facts, recurring workflows, and lessons.
- Drop duplicates, stale contradictions, temporary debug state, raw logs, one-off reminders, alerts, and status updates.
- If there is nothing new, return empty topics/forget/index.
- Do not include markdown fences or commentary outside the JSON."""


class DreamMemory(BaseTask):
    """Consolidate recent conversations into durable memory topics."""

    name = "dream_memory"
    description = "Distill recent conversations into durable memory topics."
    trigger = "event"
    trigger_channels = ["dream_memory"]
    writes = []
    timeout = 600
    event_payload_schema = {"type": "object", "properties": {}, "required": []}
    #: The schedule this task wants to exist, created by ``on_install`` and by
    #: nothing else. Four in the morning: it rewrites the whole memory folder
    #: and wants to be nowhere near a conversation while it does.
    job = {"channel": "dream_memory", "cron": "0 4 * * *", "payload": {}}

    requests = [
        "paths.get", "user.read", "fs.read", "fs.list", "fs.write",
        "fs.delete", "fs.move", "db.query", "config.read", "agent.complete",
        "service.call",
    ]
    config_settings = [
        ("Memory Dream LLM Profile", "memory_dream_llm_profile",
         "LLM profile used to rewrite memory. 'default' follows the default LLM.",
         "default", {"type": "text"}),
    ]
    agent_prompt = (
        "Nightly, dream_memory may consolidate the memory folder (MEMORY.md "
        "index + topic files) with reusable lessons and preferences."
    )

    def on_install(self, sdk):
        """Create this task's nightly schedule, once, at install.

        This was a ``default_jobs`` declaration, seeded by the orchestrator at
        every registration — so a job the user deleted was indistinguishable
        from one never installed, and came back at the next boot. ``on_install``
        runs only when somebody installs or updates this package, so a deletion
        lasts until they ask for it again.

        Read-then-skip: an existing job keeps whatever hour it has been moved
        to. Raising is reported and does not undo the install.
        """
        try:
            if sdk.services.call("timekeeper", "get_job", self.name) is None:
                sdk.services.call("timekeeper", "create_job",
                                  self.name, self.job)
                sdk.log(f"scheduled job {self.name} created")
        except sdk.Failed as error:
            raise RuntimeError(
                f"schedule {self.name!r} was not created ({error}) — this task "
                f"will not run until one is added in /schedule") from error

    def on_uninstall(self, sdk):
        """Take the schedule with it — a job with no task fires into nothing."""
        try:
            sdk.services.call("timekeeper", "remove_job", self.name)
        except sdk.Failed as error:
            sdk.log(f"could not remove the {self.name} schedule: {error}",
                    level="warning")

    def run_event(self, sdk, payload):
        """Perform one memory consolidation sweep."""
        root = _memory_root(sdk)
        state = _read_json(sdk, sdk.path.join(root, ".dream", "state.json"))
        conversations = _recent_conversations(
            sdk, float(state.get("last_success_at") or 0))
        if not conversations:
            _finish(sdk, root, "success",
                    "No recent human-facing conversations to dream over.", [], [])
            return sdk.ok({"conversations": 0})

        prompt = USER_TEMPLATE.format(
            index=_current_index(sdk, root) or "(empty)",
            topics=_current_topics(sdk, root) or "(none)",
            conversations=_format_conversations(sdk, conversations),
        )
        profile = str(
            sdk.config.read("memory_dream_llm_profile") or "default").strip()
        if profile == "default":
            profile = ""
        parsed, error = _ask_json(sdk, profile, prompt)
        if not parsed:
            _write_report(sdk, root, "failed", f"Invalid dream JSON: {error}", [], [])
            return sdk.fail(f"Invalid dream JSON: {error}")

        applied, rejected = _apply_patch(sdk, root, parsed)
        changes = applied + _string_list(parsed.get("changes"))
        skipped = rejected + _string_list(parsed.get("skipped"))
        _finish(
            sdk, root, "success",
            f"Consolidated memory from {len(conversations)} conversation(s).",
            changes, skipped,
        )
        sdk.log(
            f"memory dream applied {len(applied)} change(s) from "
            f"{len(conversations)} conversation(s)."
        )
        return sdk.ok({
            "conversations": len(conversations),
            "changes": changes,
            "skipped": skipped,
        })


def _memory_root(sdk):
    root = sdk.path.join(sdk.paths.get("workspace"), "memory")
    user = sdk.users.read() or {}
    user_id = int(user.get("id") or 1)
    return root if user_id == 1 else sdk.path.join(root, "users", str(user_id))


def _exists(sdk, path):
    try:
        sdk.fs.list(path)
        return True
    except sdk.Failed:
        return False


def _topic_name(raw):
    name = str(raw or "").strip()
    if name.lower().endswith(".md"):
        name = name[:-3]
    if not name or not TOPIC_RE.match(name) or name.upper() == "MEMORY":
        raise ValueError(f"Invalid memory topic name: {raw!r}")
    return name


def _topic_path(sdk, root, raw):
    name = _topic_name(raw)
    return sdk.path.join(root, f"{name}.md")


def _list_topics(sdk, root):
    try:
        entries = sdk.fs.list(root, pattern="*.md", details=True)
    except sdk.Failed:
        return []
    return sorted(
        (entry for entry in entries
         if not entry.get("is_dir") and entry.get("name") != INDEX_FILENAME),
        key=lambda entry: entry.get("name") or "",
    )


def _current_index(sdk, root):
    path = sdk.path.join(root, INDEX_FILENAME)
    return sdk.fs.read(path).strip() if _exists(sdk, path) else ""


def _current_topics(sdk, root):
    blocks = []
    size = 0
    for entry in _list_topics(sdk, root):
        name = sdk.path.stem(entry["name"])
        body = sdk.fs.read(entry["path"]).strip()
        block = f'<topic name="{name}">\n{body}\n</topic>'
        remaining = MAX_MEMORY_CHARS - size
        if remaining <= 0:
            break
        blocks.append(block[:remaining])
        size += len(block) + 2
    return "\n\n".join(blocks)[:MAX_MEMORY_CHARS]


def _backup(sdk, path):
    if _exists(sdk, path):
        sdk.fs.move(path, f"{path}.bak", copy=True)


def _apply_patch(sdk, root, parsed):
    applied, rejected = [], []
    topics = parsed.get("topics") if isinstance(parsed.get("topics"), dict) else {}
    forget = parsed.get("forget") if isinstance(parsed.get("forget"), list) else []
    hooks = parsed.get("index") if isinstance(parsed.get("index"), dict) else {}

    current_count = len(_list_topics(sdk, root))
    if current_count + len(topics) > MAX_TOPICS + len(forget):
        return [], [f"Patch rejected: would exceed {MAX_TOPICS} topics."]

    for raw_name, raw_body in topics.items():
        body = str(raw_body or "").strip()
        if not body:
            rejected.append(f"Skipped empty topic body: {raw_name}")
            continue
        try:
            path = _topic_path(sdk, root, raw_name)
        except ValueError:
            rejected.append(f"Skipped invalid topic name: {raw_name}")
            continue
        _backup(sdk, path)
        sdk.fs.write(path, body.rstrip() + "\n")
        applied.append(f"Updated topic: {sdk.path.stem(path)}")

    for raw_name in forget:
        try:
            path = _topic_path(sdk, root, raw_name)
        except ValueError:
            rejected.append(f"Skipped invalid forget name: {raw_name}")
            continue
        if _exists(sdk, path):
            _backup(sdk, path)
            sdk.fs.delete(path)
            applied.append(f"Forgot topic: {sdk.path.stem(path)}")

    clean_hooks = {
        str(key): str(value).strip() for key, value in hooks.items()
        if str(value).strip()
    }
    _rebuild_index(sdk, root, clean_hooks)
    return applied, rejected


def _rebuild_index(sdk, root, hooks):
    existing = {}
    index = _current_index(sdk, root)
    pattern = re.compile(
        r"^- \[(?P<name>[^\]]+)\]\([^)]+\)(?: - (?P<hook>.*))?$"
    )
    for line in index.splitlines():
        match = pattern.match(line.strip())
        if match:
            existing[match.group("name")] = (match.group("hook") or "").strip()

    lines = []
    for entry in _list_topics(sdk, root):
        name = sdk.path.stem(entry["name"])
        hook = hooks.get(name) or existing.get(name, "")
        item = f"- [{name}]({name}.md)"
        lines.append(f"{item} - {hook}" if hook else item)
    sdk.fs.write(
        sdk.path.join(root, INDEX_FILENAME),
        "\n".join(lines).rstrip() + "\n" if lines else "",
    )


def _recent_conversations(sdk, since):
    return list(sdk.db.query(
        """SELECT id, title, category, updated_at
             FROM my_conversations
            WHERE kind = 'user' AND updated_at > ?
            ORDER BY updated_at DESC
            LIMIT ?""",
        [float(since), MAX_CONVERSATIONS],
        max_rows=MAX_CONVERSATIONS,
    ) or [])


def _format_conversations(sdk, conversations):
    chunks = []
    for conversation in conversations:
        lines = [
            f"Conversation {conversation.get('id')}: "
            f"{conversation.get('title') or 'Untitled'} | "
            f"category={conversation.get('category') or 'Main'} | "
            f"updated_at={conversation.get('updated_at')}"
        ]
        messages = sdk.db.query(
            """SELECT role, content
                 FROM conversation_messages
                WHERE conversation_id = ?
                ORDER BY timestamp""",
            [conversation["id"]],
            max_rows=500,
        ) or []
        for message in messages:
            role = (message.get("role") or "").upper()
            content = _plain_content(message.get("content") or "")
            if role == "TOOL" and "error" not in content.lower():
                continue
            if role in {"SYSTEM", ""} or not content:
                continue
            lines.append(f"{role}: {content[:600]}")
        chunks.append("\n".join(lines))
    return "\n\n---\n\n".join(chunks)[:MAX_TRANSCRIPT_CHARS]


_THINKING = re.compile(
    r"<(?:think|thinking)>.*?</(?:think|thinking)>", re.DOTALL)
_STRUCTURAL = re.compile(
    r"<invoke.*?>.*?</invoke>|<tool_call.*?>.*?</tool_call>|"
    r"<(?:/)?minimax:tool_call>|<\|im_end\|>|<\|eot_id\|>", re.DOTALL)
_THINKING_TAG = re.compile(r"</?(?:think|thinking)>")


def _strip_model_tokens(text):
    clean = _THINKING.sub("", text or "")
    clean = _STRUCTURAL.sub("", clean)
    return _THINKING_TAG.sub("", clean).strip()


def _ask_json(sdk, profile, prompt):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    last_error = "model did not return parseable JSON"
    for attempt in range(3):
        try:
            response = sdk.agent.complete(messages=messages, profile=profile) or {}
        except sdk.Failed as error:
            last_error = str(error)
            continue
        content = response.get("content") or ""
        parsed = _extract_json(content)
        if parsed is not None:
            return parsed, ""
        messages = [
            {"role": "system", "content": "Repair the user's text into valid JSON only."},
            {"role": "user", "content": content},
        ]
        last_error = f"attempt {attempt + 1} was not valid JSON"
    return None, last_error


def _extract_json(text):
    text = re.sub(
        r"^```(?:json)?|```$", "", _strip_model_tokens(text).strip(),
        flags=re.IGNORECASE | re.MULTILINE,
    ).strip()
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end >= start:
        text = text[start:end + 1]
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _plain_content(content):
    try:
        data = json.loads(content)
        if isinstance(data, dict) and "tool_calls" in data:
            content = data.get("content") or ""
    except (TypeError, ValueError):
        pass
    return " ".join(str(content).split())


def _string_list(value):
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()][:20]


def _read_json(sdk, path):
    try:
        data = json.loads(sdk.fs.read(path))
        return data if isinstance(data, dict) else {}
    except (sdk.Failed, TypeError, ValueError):
        return {}


def _finish(sdk, root, status, message, changes, skipped):
    now = time.time()
    state_path = sdk.path.join(root, ".dream", "state.json")
    sdk.fs.write(state_path, json.dumps({"last_success_at": now}, indent=2))
    _write_report(sdk, root, status, message, changes, skipped)


def _write_report(sdk, root, status, message, changes, skipped):
    lines = [
        "# Memory Dream Report", "", f"- Status: {status}",
        f"- Time: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Message: {message}", "", "## Changes",
    ]
    lines.extend(f"- {item}" for item in (changes or ["None"]))
    lines.extend(["", "## Skipped"])
    lines.extend(f"- {item}" for item in (skipped or ["None"]))
    sdk.fs.write(
        sdk.path.join(root, ".dream", "report.md"),
        "\n".join(lines).rstrip() + "\n",
    )
