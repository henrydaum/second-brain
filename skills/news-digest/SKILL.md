---
name: news-digest
description: Build a recurring or one-off news digest on chosen topics from web search, deduplicated against previous digests. Use for 'keep me updated on X' or 'what happened in Y this week'.
dependencies_files: [tools/tool_use_skill.py, tools/tool_web_search.py, tools/tool_edit_file.py, tools/tool_schedule_subagent.py]
---

# News digest

Topic-tracking without the feed-scrolling.

## One-off digest

1. `web_search` per topic with a recency-anchored query ("X news June
   2026", answers mode where useful).
2. For stories that matter, fetch the article URL via `web_search` to get
   past the headline before summarizing.
3. Output per topic: 2-4 stories, one line each — what happened + why it
   matters — with the source. Separate "developments" from "commentary".

## Recurring digest

1. Keep state in a file, e.g. `news_digest_state.md` in a sync directory:
   one line per already-reported story (date, headline, URL).
2. The scheduled prompt (via `schedule_subagent`) must tell the subagent
   to read the state file first, report only NEW stories, then append the
   newly reported ones to the state file with `edit_file`.
3. Weekly cadence beats daily for most topics; suggest it.

## Pitfalls

- Search result dates are unreliable; if a story's recency is load-bearing,
  verify the date on the page itself.
- No stories is a fine digest: "nothing significant since last time" is
  more valuable than padding with stale or marginal items.
