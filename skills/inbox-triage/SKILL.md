---
name: inbox-triage
description: Work through unread email: categorize, surface what matters, archive noise, draft replies for approval. Use for 'check my email', 'anything important?', or inbox cleanup sessions.
dependencies_files: [tools/tool_use_skill.py, tools/tool_email_check.py, tools/tool_email_mark_read.py, tools/tool_email_modify_labels.py, tools/tool_ask_user_question.py]
---

# Inbox triage

Turn an unread pile into: things needing the user, things handled, noise
gone. The user's attention is the scarce resource — spend it only on mail
that needs a human.

## Procedure

1. `email_check` (scope inbox, unread). Zero unread: say so, stop.
2. Bucket every message:
   - **Action** — needs the user's decision or reply.
   - **FYI** — worth one line in your summary, then mark read.
   - **Noise** — newsletters, receipts, notifications: mark read
     (`email_mark_read`); label via `email_modify_labels` if the user has
     a filing convention (ask once, remember it).
3. Report buckets in priority order: Action items first with one-line
   stakes each ("recruiter, wants an answer by Friday"), then a compressed
   FYI digest, then "archived N noise".
4. Offer reply drafts for Action items. Never send without the user seeing
   the draft — use `ask_user_question` to confirm send vs edit.

## Rules

- Never mark an Action item read; read-state is the user's todo signal.
- Quote deadlines and amounts exactly as written in the mail.
- Mass actions (>10 messages) get a count confirmation first.
- If a message looks important but ambiguous, it is an Action item.
