---
name: research-report
description: Research a topic across the web and the local corpus, then deliver a structured cited report. Use for 'research X', 'write me a report on', or any multi-source investigation.
dependencies_files: [tools/tool_use_skill.py, tools/tool_web_search.py, tools/tool_hybrid_search.py, tools/tool_edit_file.py]
---

# Research report

Produce a report the user can trust without redoing the research.

## Procedure

1. **Scope first.** Restate the question in one sentence and pick 3-6
   sub-questions. If the request is ambiguous, ask one clarifying question
   before searching, not after.
2. **Local before web.** Run `hybrid_search` on the corpus for each
   sub-question — the user's own files often hold the ground truth (notes,
   PDFs, prior reports). Note which sub-questions the corpus answers.
3. **Web for the rest.** Use `web_search` per remaining sub-question.
   Passing a URL as the query fetches that page's text — use it to read
   primary sources instead of trusting snippets.
4. **Track sources as you go**: (claim, source path-or-URL) pairs. A claim
   without a source does not go in the report.
5. **Write the report** to a markdown file in a sync directory via
   `edit_file` (create). Structure: TL;DR (3 bullets) / Findings per
   sub-question / Open questions / Sources. Keep findings as assertions
   with inline source references, not narrations of your search process.
6. Reply in chat with the TL;DR and the file path.

## Pitfalls

- Do not interleave searching and writing; finish gathering before drafting,
  or the report inherits your detours.
- Two sources disagreeing is a finding — report the disagreement, do not
  silently pick one.
- Snippets lie by omission. For any claim central to the conclusion, fetch
  the page.
