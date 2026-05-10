"""Nightly memory distiller.

Fired by the default ``dream_memory`` cron job. It reads recent user
conversations plus the current memory.md, asks the configured LLM for a
strict JSON rewrite, and replaces memory.md without a user approval step.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import time
from pathlib import Path
from typing import Any

from events.event_channels import DREAM_MEMORY
from paths import DATA_DIR
from plugins.BaseTask import BaseTask, TaskResult
from runtime.agent_scope import resolve_agent_llm
from runtime.token_stripper import strip_model_tokens

logger = logging.getLogger("TaskDreamMemory")

STATE_PATH = DATA_DIR / "memory_dream_state.json"
MEMORY_PATH = DATA_DIR / "memory.md"
REPORT_PATH = DATA_DIR / "memory_dream_report.md"
BACKUP_PATH = DATA_DIR / "memory.md.bak"
MAX_CONVERSATIONS = 25
MAX_TRANSCRIPT_CHARS = 24000

SYSTEM_PROMPT = (
    "You maintain Second Brain's durable memory.md. Return only valid JSON. "
    "Rewrite memory.md as compact standing context, not a chat summary."
)

USER_TEMPLATE = """Current memory.md:
<memory>
{memory}
</memory>

Recent human-facing conversations:
<conversations>
{conversations}
</conversations>

Return JSON with exactly these keys:
{{
  "memory_md": "full markdown replacement using # User, # Projects, # Operating Lessons, # Do Not Do",
  "changes": ["short bullets for additions, merges, deletions"],
  "skipped": ["short bullets for ignored transient items"]
}}

Rules:
- Keep memory.md short, specific, and reusable across future sessions.
- Preserve durable user preferences, project facts, recurring workflows, and hard-won system lessons.
- Drop duplicates, stale contradictions, temporary debug state, raw logs, one-off reminders, alerts, and status updates.
- If there is nothing new, return the existing memory cleaned into the required sections.
- Do not include markdown fences or commentary outside the JSON."""


class DreamMemory(BaseTask):
    name = "dream_memory"
    trigger = "event"
    trigger_channels = [DREAM_MEMORY]
    requires_services = ["llm"]
    writes = []
    timeout = 600
    event_payload_schema = {"type": "object", "properties": {}, "required": []}
    config_settings = [
        ("Memory Dream LLM Profile", "memory_dream_llm_profile",
         "Agent profile whose LLM rewrites memory.md. 'default' follows the default LLM.",
         "default", {"type": "text"}),
    ]

    def run_event(self, run_id: str, payload: dict, context) -> TaskResult:
        db = getattr(context, "db", None)
        if db is None:
            return TaskResult.failed("No database available.")
        llm = resolve_agent_llm((context.config.get("memory_dream_llm_profile") or "default").strip() or "default", context.config, context.services)
        if llm is None or not getattr(llm, "loaded", False):
            _write_report("skipped", "LLM service is not loaded.", [], [])
            return TaskResult(success=True)

        MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        memory = MEMORY_PATH.read_text(encoding="utf-8") if MEMORY_PATH.exists() else ""
        state = _read_state()
        conversations = _recent_conversations(db, float(state.get("last_success_at") or 0))
        if not conversations:
            now = time.time()
            _write_state(now)
            _write_report("success", "No recent human-facing conversations to dream over.", [], [])
            return TaskResult(success=True, data={"conversations": 0})

        prompt = USER_TEMPLATE.format(memory=memory or "(empty)", conversations=_format_conversations(db, conversations))
        parsed, error = _ask_json(llm, prompt)
        if not parsed:
            _write_report("failed", f"Invalid dream JSON: {error}", [], [])
            return TaskResult.failed(f"Invalid dream JSON: {error}")

        new_memory = _normalize_memory(parsed.get("memory_md"))
        if not new_memory:
            _write_report("failed", "Dream JSON did not include non-empty memory_md.", [], [])
            return TaskResult.failed("Dream JSON did not include non-empty memory_md.")

        if MEMORY_PATH.exists():
            shutil.copyfile(MEMORY_PATH, BACKUP_PATH)
        MEMORY_PATH.write_text(new_memory, encoding="utf-8")
        now = time.time()
        _write_state(now)
        changes, skipped = _string_list(parsed.get("changes")), _string_list(parsed.get("skipped"))
        _write_report("success", f"Updated memory.md from {len(conversations)} conversation(s).", changes, skipped)
        logger.info("Memory dream updated memory.md from %d conversation(s).", len(conversations))
        return TaskResult(success=True, data={"conversations": len(conversations), "changes": changes, "skipped": skipped})


def _ask_json(llm, prompt: str) -> tuple[dict[str, Any] | None, str]:
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
    rows = db.list_conversations(MAX_CONVERSATIONS * 2)
    out = [r for r in rows if (r.get("kind") or "user") == "user" and float(r.get("updated_at") or 0) > since]
    return out[:MAX_CONVERSATIONS]


def _format_conversations(db, conversations: list[dict]) -> str:
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
    try:
        data = json.loads(content)
        if isinstance(data, dict) and "tool_calls" in data:
            content = data.get("content") or ""
    except Exception:
        pass
    return " ".join(str(content).split())


def _normalize_memory(text: Any) -> str:
    text = str(text or "").strip()
    if not text:
        return ""
    sections = ["# User", "# Projects", "# Operating Lessons", "# Do Not Do"]
    if not any(text.startswith(s) or f"\n{s}" in text for s in sections):
        text = "# User\n\n# Projects\n\n# Operating Lessons\n\n# Do Not Do\n\n" + text
    return text.rstrip() + "\n"


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()][:20]


def _read_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_state(last_success_at: float) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps({"last_success_at": last_success_at}, indent=2), encoding="utf-8")


def _write_report(status: str, message: str, changes: list[str], skipped: list[str]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# Memory Dream Report", "", f"- Status: {status}", f"- Time: {time.strftime('%Y-%m-%d %H:%M:%S')}", f"- Message: {message}", "", "## Changes"]
    lines.extend(f"- {x}" for x in (changes or ["None"]))
    lines.append("\n## Skipped")
    lines.extend(f"- {x}" for x in (skipped or ["None"]))
    REPORT_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
