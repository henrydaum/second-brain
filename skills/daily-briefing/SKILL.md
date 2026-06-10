---
name: daily-briefing
description: Set up or run a scheduled morning briefing: unread email summary, calendar/weather via web, reminders from memory. Use when the user wants a recurring daily digest.
dependencies_files: [tools/tool_use_skill.py, tools/tool_schedule_subagent.py, tools/tool_email_check.py, tools/tool_web_search.py]
---

# Daily briefing

A recurring subagent that compresses the morning into one message.

## Setting it up

1. Ask which ingredients the user wants: email summary, weather (needs
   their city), news topics, memory-based reminders. And the delivery time.
2. `schedule_subagent` action=add with a cron like `30 7 * * *` and a
   prompt that is **self-contained** — the subagent wakes with no
   conversational context. Template:

   "You are the morning briefing. Do these steps and send ONE message:
   1) email_check unread inbox -> 'N unread, top 3: ...' (one line each).
   2) web_search 'weather <city> today' -> one line.
   3) memory read 'reminders' if it exists -> surface anything dated today.
   Keep the whole briefing under 12 lines. No preamble."

3. Tell the user how to change or stop it (/schedule, or ask again).

## Running as the briefing (if this skill loads inside the subagent)

- Budget: one tool call per ingredient, one output message. No follow-up
  questions — unattended sessions cannot ask.
- Skip ingredients whose tool is unavailable; note the gap in one clause
  ("(no email access)") rather than failing the whole briefing.

## Pitfalls

- Briefings drift into walls of text. The value is compression: hard cap
  the length, lead with anything requiring action today.
