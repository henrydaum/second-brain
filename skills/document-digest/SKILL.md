---
name: document-digest
description: Summarize or extract from one long document (PDF, report, book, transcript) accurately. Use for 'summarize this file', 'what are the key points of', or extracting specific sections.
dependencies_files: [tools/tool_use_skill.py, tools/tool_read_file.py, tools/tool_lexical_search.py, tools/tool_edit_file.py]
---

# Document digest

Long documents do not fit in context. Digest them honestly instead of
summarizing the first 20k characters and guessing the rest.

## Procedure

1. Establish size: `read_file` with a small limit, note the total line
   count in the trailer. Under ~2k lines: read it all, summarize directly.
2. Larger: read in windows (offset/limit), keeping per-window notes of
   3-6 bullets each. Do not carry full text forward — carry the notes.
3. For targeted extraction ("what does it say about X"), `lexical_search`
   scoped to the document's terms first, then read around the hits, rather
   than scanning front to back.
4. Compose the digest from the window notes: thesis, key points with
   section/line references, surprises, and what the user should read in
   full. For digests worth keeping, save to a markdown file with
   `edit_file` and reply with the highlights.

## Pitfalls

- State your coverage. If you read 40% of the file, the digest says which
  parts.
- Page/line references come from the actual read offsets, not estimation.
- Tables and figures often carry the load-bearing numbers; quote them
  exactly, never paraphrase digits.
