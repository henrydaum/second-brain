---
name: weekly-review
description: Run a weekly review: mine the week's conversations and memory for decisions, loose ends, and lessons; update memory; write a review note. Use for 'weekly review' or 'what did I do this week'.
dependencies_files: [tools/tool_use_skill.py, tools/tool_sql_query.py, tools/tool_memory.py, tools/tool_edit_file.py]
---

# Weekly review

Close the week's loops: what happened, what's unfinished, what should be
remembered.

## Procedure

1. Pull the week's activity via `sql_query`:
   SELECT id, title, category, updated_at FROM conversations
   WHERE updated_at > strftime('%s','now','-7 days') AND kind='user'
   ORDER BY updated_at DESC.
   For conversations whose title suggests substance, sample messages from
   `conversation_messages` (conversation_id, role, content) to recover
   decisions and commitments.
2. Extract three lists:
   - **Done** — decisions made, things shipped.
   - **Open loops** — promised-but-not-done, questions left hanging.
   - **Lessons** — anything that changed how the user works.
3. Cross-check open loops against memory topics (`memory` read) — close
   out ones that resolved, and save genuinely durable lessons to a fitting
   topic (do not log routine work as lessons).
4. Write `reviews/<yyyy-mm-dd>-weekly.md` in a sync directory: the three
   lists plus a 2-line summary. Reply with open loops first — that is the
   actionable part.

## Pitfalls

- Tool-role messages are noise for this purpose; filter to user/assistant
  content.
- Don't re-litigate decisions in the review; record them.
