---
name: conversation-archaeology
description: Find and accurately quote past conversations: SQL over conversations and messages, compaction-aware. Use for 'what did we discuss about X', 'find that conversation where...'.
dependencies_files: [tools/tool_use_skill.py, tools/tool_sql_query.py]
---

# Conversation archaeology

The conversations database is the user's full chat history with this
system. Dig precisely.

## Procedure

1. Start narrow: SELECT id, title, category, updated_at FROM conversations
   WHERE title LIKE '%term%' ORDER BY updated_at DESC — titles are
   auto-generated and usually on-topic.
2. No title hit: search content —
   SELECT DISTINCT conversation_id FROM conversation_messages WHERE
   content LIKE '%term%' (try 2-3 wordings; content is verbatim, so
   synonyms matter).
3. Read candidates chronologically: SELECT role, content FROM
   conversation_messages WHERE conversation_id=? ORDER BY id — filter
   roles to user/assistant; tool messages and state markers are plumbing.
4. Quote exactly, attribute clearly ("you said... I replied..."), and give
   the conversation id + date so the user can /conversations to it.

## Compaction awareness

Long conversations get compacted: the live context shows a summary, but
the full pre-compaction history REMAINS in conversation_messages. The
database is therefore more complete than what either of you "remembers" —
say so when it matters.

## Rules

- Read-only digging: SELECT statements only; never modify rows to "fix"
  history.
- Paraphrase only when quoting is impractical, and mark it as paraphrase.
