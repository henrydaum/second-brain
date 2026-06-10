---
name: personal-crm
description: Remember people: keep a memory topic per important contact (context, preferences, commitments) and recall it before meetings or replies. Use for 'what do I know about <person>' or after notable interactions.
dependencies_files: [tools/tool_use_skill.py, tools/tool_memory.py, tools/tool_sql_query.py]
---

# Personal CRM

Relationships compound when context survives between interactions.

## Topic convention

One memory topic per person who recurs: `person-<firstname-lastname>`,
containing: who they are (one line), how you know them, preferences /
constraints worth respecting, running history of significant interactions
(date + one line), open commitments in either direction.

## Procedure

- **Recall** ("what do I know about X", or before drafting mail to X):
  `memory` read the person topic; check recent mentions in conversations
  via `sql_query` (conversation_messages.content LIKE '%name%', recent
  first) for anything newer than the topic. Lead with open commitments —
  those are the actionable part.
- **Capture** (after the user relays a meaningful interaction): append a
  dated line to the person topic. New significant person: create the topic
  and add an index hook naming their role ("person-jane-doe - VC contact,
  met 2026-06"). Update, don't accumulate: resolved commitments get
  removed, contradictions resolved in favor of newest.

## Boundaries

- Store what helps the user act well: roles, preferences, commitments,
  follow-ups. Do NOT store sensitive personal details (health, finances,
  relationship gossip) unless explicitly asked — and note in the topic
  that it was explicit.
- This is the user's private memory, not a dossier; if a capture feels
  like surveillance rather than thoughtfulness, ask first.
