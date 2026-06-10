"""Nightly memory distiller.

Fired by the ``dream_memory`` event channel. It reads recent user
conversations plus the current memory folder (MEMORY.md index + per-topic
markdown files — see ``plugins/helpers/memory_paths.py`` in the kernel),
asks the configured LLM for a strict JSON *patch* (topics to upsert, topics
to forget, index hooks), and applies it without a user approval step.
A legacy single ``memory.md`` (pre-folder installs) is fed in as input the
first time, so its content migrates into topics organically.
"""

from __future__ import annotations

dependencies_files = ['services/service_litellm.py']
dependencies_pip = []

import json
import logging
import re
import shutil
import time
from typing import Any

from paths import DATA_DIR
from plugins.BaseTask import BaseTask, TaskResult
from plugins.helpers.memory_paths import INDEX_FILENAME, LEGACY_MEMORY_FILE, list_topics, memory_root, topic_path
from runtime.agent_scope import resolve_agent_llm
from runtime.token_stripper import strip_model_tokens

logger = logging.getLogger("TaskDreamMemory")

DREAM_MEMORY = "dream_memory"
"""Plugin-owned event channel for the nightly memory distiller."""

STATE_PATH = DATA_DIR / "memory_dream_state.json"
REPORT_PATH = DATA_DIR / "memory_dream_report.md"
MAX_CONVERSATIONS = 25
MAX_TRANSCRIPT_CHARS = 24000
MAX_MEMORY_CHARS = 16000
MAX_TOPICS = 40

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
- Topic names use letters, digits, dots, dashes, underscores, spaces (e.g. "user-profile", "project-second-brain").
- Keep each topic short, specific, and reusable across future sessions; give every changed topic an index hook.
- Preserve durable user preferences, project facts, recurring workflows, and hard-won lessons.
- Drop duplicates, stale contradictions, temporary debug state, raw logs, one-off reminders, alerts, and status updates.
- If there is nothing new, return empty topics/forget/index.
- Do not include markdown fences or commentary outside the JSON."""


class DreamMemory(BaseTask):
    """Dream memory."""
    name = "dream_memory"
    trigger = "event"
    trigger_channels = [DREAM_MEMORY]
    requires_services = ["llm"]
    writes = []
    timeout = 600
    event_payload_schema = {"type": "object", "properties": {}, "required": []}
    config_settings = [
        ("Memory Dream LLM Profile", "memory_dream_llm_profile",
         "Agent profile whose LLM rewrites the memory folder. 'default' follows the default LLM.",
         "default", {"type": "text"}),
    ]
    agent_prompt = "Nightly, dream_memory may consolidate the memory folder (MEMORY.md index + topic files) with reusable lessons and preferences."

    def run_event(self, run_id: str, payload: dict, context) -> TaskResult:
        """Run event."""
        db = getattr(context, "db", None)
        if db is None:
            return TaskResult.failed("No database available.")
        llm = resolve_agent_llm((context.config.get("memory_dream_llm_profile") or "default").strip() or "default", context.config, context.services)
        if llm is None or not getattr(llm, "loaded", False):
            _write_report("skipped", "LLM service is not loaded.", [], [])
            return TaskResult(success=True)

        state = _read_state()
        conversations = _recent_conversations(db, float(state.get("last_success_at") or 0))
        if not conversations:
            _write_state(time.time())
            _write_report("success", "No recent human-facing conversations to dream over.", [], [])
            return TaskResult(success=True, data={"conversations": 0})

        prompt = USER_TEMPLATE.format(
            index=_current_index() or "(empty)",
            topics=_current_topics() or "(none)",
            conversations=_format_conversations(db, conversations),
        )
        parsed, error = _ask_json(llm, prompt)
        if not parsed:
            _write_report("failed", f"Invalid dream JSON: {error}", [], [])
            return TaskResult.failed(f"Invalid dream JSON: {error}")

        applied, rejected = _apply_patch(parsed)
        _write_state(time.time())
        changes, skipped = _string_list(parsed.get("changes")), _string_list(parsed.get("skipped"))
        changes = applied + changes
        skipped = rejected + skipped
        _write_report("success", f"Consolidated memory from {len(conversations)} conversation(s).", changes, skipped)
        logger.info("Memory dream applied %d change(s) from %d conversation(s).", len(applied), len(conversations))
        return TaskResult(success=True, data={"conversations": len(conversations), "changes": changes, "skipped": skipped})


# ──────────────────────────────────────────────────────────────────────
# Memory folder I/O
# ──────────────────────────────────────────────────────────────────────

def _current_index() -> str:
    path = memory_root() / INDEX_FILENAME
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


def _current_topics() -> str:
    """Topic bodies as tagged blocks, plus the legacy memory.md until it migrates."""
    blocks = []
    if not memory_root().is_dir() and LEGACY_MEMORY_FILE.exists():
        blocks.append(f'<topic name="legacy-memory">\n{LEGACY_MEMORY_FILE.read_text(encoding="utf-8").strip()}\n</topic>')
    for path in list_topics():
        blocks.append(f'<topic name="{path.stem}">\n{path.read_text(encoding="utf-8").strip()}\n</topic>')
    return "\n\n".join(blocks)[:MAX_MEMORY_CHARS]


def _apply_patch(parsed: dict) -> tuple[list[str], list[str]]:
    """Apply the LLM's topic patch. Returns (applied, rejected) bullets."""
    applied: list[str] = []
    rejected: list[str] = []
    topics = parsed.get("topics") if isinstance(parsed.get("topics"), dict) else {}
    forget = parsed.get("forget") if isinstance(parsed.get("forget"), list) else []
    hooks = parsed.get("index") if isinstance(parsed.get("index"), dict) else {}

    if len(list_topics()) + len(topics) > MAX_TOPICS + len(forget):
        rejected.append(f"Patch rejected: would exceed {MAX_TOPICS} topics.")
        return applied, rejected

    for name, body in topics.items():
        body = str(body or "").strip()
        if not body:
            rejected.append(f"Skipped empty topic body: {name}")
            continue
        try:
            path = topic_path(str(name))
        except ValueError:
            rejected.append(f"Skipped invalid topic name: {name}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            shutil.copyfile(path, path.with_suffix(".md.bak"))
        path.write_text(body.rstrip() + "\n", encoding="utf-8")
        applied.append(f"Updated topic: {path.stem}")

    for name in forget:
        try:
            path = topic_path(str(name))
        except ValueError:
            rejected.append(f"Skipped invalid forget name: {name}")
            continue
        if path.exists():
            shutil.copyfile(path, path.with_suffix(".md.bak"))
            path.unlink()
            applied.append(f"Forgot topic: {path.stem}")

    _rebuild_index({str(k): str(v).strip() for k, v in hooks.items() if str(v).strip()})
    return applied, rejected


def _rebuild_index(hooks: dict[str, str]) -> None:
    """Regenerate MEMORY.md: one line per surviving topic, new hooks winning."""
    root = memory_root()
    existing: dict[str, str] = {}
    index_path = root / INDEX_FILENAME
    if index_path.exists():
        for line in index_path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^- \[(?P<name>[^\]]+)\]\([^)]+\)(?: - (?P<hook>.*))?$", line.strip())
            if m:
                existing[m.group("name")] = (m.group("hook") or "").strip()
    lines = []
    for path in list_topics():
        hook = hooks.get(path.stem) or existing.get(path.stem, "")
        entry = f"- [{path.stem}]({path.stem}.md)"
        lines.append(f"{entry} - {hook}" if hook else entry)
    root.mkdir(parents=True, exist_ok=True)
    index_path.write_text("\n".join(lines).rstrip() + "\n" if lines else "", encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────
# LLM + conversation plumbing
# ──────────────────────────────────────────────────────────────────────

def _ask_json(llm, prompt: str) -> tuple[dict[str, Any] | None, str]:
    """Internal helper to handle ask JSON."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    for kwargs in ({"response_format": {"type": "json_object"}}, {}, {}):
        response = llm.invoke(messages, **kwargs)
        if getattr(response, "error", None):
            continue
        parsed = _extract_json(getattr(response, "content", ""))
        if parsed:
            return parsed, ""
        messages = [
            {"role": "system", "content": "Repair the user's text into valid JSON only."},
            {"role": "user", "content": getattr(response, "content", "") or ""},
        ]
    return None, "model did not return parseable JSON"


def _extract_json(text: str) -> dict[str, Any] | None:
    """Internal helper to extract JSON."""
    text, _ = strip_model_tokens(text or "")
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.I | re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end >= start:
        text = text[start:end + 1]
    try:
        data = json.loads(text)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _recent_conversations(db, since: float) -> list[dict]:
    """Internal helper to handle recent conversations."""
    rows = db.list_conversations(MAX_CONVERSATIONS * 2)
    out = [r for r in rows if (r.get("kind") or "user") == "user" and float(r.get("updated_at") or 0) > since]
    return out[:MAX_CONVERSATIONS]


def _format_conversations(db, conversations: list[dict]) -> str:
    """Internal helper to format conversations."""
    chunks = []
    for c in conversations:
        lines = [f"Conversation {c.get('id')}: {c.get('title') or 'Untitled'} | category={c.get('category') or 'Main'} | updated_at={c.get('updated_at')}"]
        for m in db.get_conversation_messages(c["id"]):
            role, content = (m.get("role") or "").upper(), _plain_content(m.get("content") or "")
            if role == "TOOL" and "error" not in content.lower():
                continue
            if role in {"SYSTEM", ""} or not content:
                continue
            lines.append(f"{role}: {content[:600]}")
        chunks.append("\n".join(lines))
    return "\n\n---\n\n".join(chunks)[:MAX_TRANSCRIPT_CHARS]


def _plain_content(content: str) -> str:
    """Internal helper to handle plain content."""
    try:
        data = json.loads(content)
        if isinstance(data, dict) and "tool_calls" in data:
            content = data.get("content") or ""
    except Exception:
        pass
    return " ".join(str(content).split())


def _string_list(value: Any) -> list[str]:
    """Internal helper to handle string list."""
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()][:20]


def _read_state() -> dict:
    """Internal helper to read state."""
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_state(last_success_at: float) -> None:
    """Internal helper to write state."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps({"last_success_at": last_success_at}, indent=2), encoding="utf-8")


def _write_report(status: str, message: str, changes: list[str], skipped: list[str]) -> None:
    """Internal helper to write report."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# Memory Dream Report", "", f"- Status: {status}", f"- Time: {time.strftime('%Y-%m-%d %H:%M:%S')}", f"- Message: {message}", "", "## Changes"]
    lines.extend(f"- {x}" for x in (changes or ["None"]))
    lines.append("\n## Skipped")
    lines.extend(f"- {x}" for x in (skipped or ["None"]))
    REPORT_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
