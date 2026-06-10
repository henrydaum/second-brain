---
name: meeting-notes
description: Turn a meeting transcript or recording text into minutes: decisions, action items with owners, open questions; file them and remember commitments. Use when given a transcript or asked to process a meeting.
dependencies_files: [tools/tool_use_skill.py, tools/tool_read_file.py, tools/tool_edit_file.py, tools/tool_memory.py]
---

# Meeting notes

From transcript to minutes someone can act on. (Audio files become
transcripts automatically if the audio parser package is installed; this
skill starts from the text.)

## Procedure

1. Read the transcript (windowed for long ones — see document-digest
   habits: notes per window, not full carry-forward).
2. Extract into exactly four sections:
   - **Decisions** — what was settled, as flat statements.
   - **Action items** — owner -> task -> deadline; unowned tasks go in
     as "UNASSIGNED" rather than silently dropped.
   - **Open questions** — raised, not resolved.
   - **Context** — 3-5 lines max of what the meeting was.
3. Attribute carefully: speaker labels in transcripts are unreliable.
   When ownership of a commitment is unclear, mark it with a "?" instead
   of guessing.
4. Save minutes as `meetings/<yyyy-mm-dd>-<topic>.md` via `edit_file`;
   reply with Decisions + Action items only.
5. Commitments made BY the user with dates: offer to save to a memory
   topic (e.g. `commitments`) so future sessions know about them.

## Pitfalls

- Verbatim quotes only for contentious points; otherwise compress.
- Numbers, dates, and names exactly as spoken; if garbled in transcript,
  flag rather than normalize.
