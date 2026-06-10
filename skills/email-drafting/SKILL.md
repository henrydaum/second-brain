---
name: email-drafting
description: Compose and send email well: structure, tone matching, reply threading. Use when writing, replying to, or forwarding email on the user's behalf.
dependencies_files: [tools/tool_use_skill.py, tools/tool_email_send.py, tools/tool_email_check.py]
---

# Email drafting

Mail sent from Second Brain is the user speaking. Draft like them, not
like an assistant.

## Procedure

1. **Replies:** fetch the thread with `email_check` first; `email_send`
   reply mode preserves threading. Answer the questions actually asked —
   quote-and-respond if there are several.
2. **Tone:** mirror the counterpart's register (a two-line casual mail gets
   a two-line casual reply). Check memory for known style preferences and
   facts about this correspondent before drafting.
3. **Structure for cold/long mail:** one-sentence why-I'm-writing, then
   the substance, then exactly one clear ask with any deadline. Subject
   states the topic + action ("Invoice 4521 — approval needed").
4. Show the draft and get explicit confirmation before sending. After
   sending, report the result.

## Rules

- No invented commitments: never promise dates, money, or attendance the
  user hasn't stated.
- Attachments: confirm the exact file path with the user before sending
  files.
- If the mail is emotionally loaded (conflict, apology, negotiation),
  draft conservative and flag the risky sentences to the user.
