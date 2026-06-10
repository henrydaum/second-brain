---
name: corpus-qa
description: Answer questions from the user's indexed files: pick the right search mode (hybrid/semantic/lexical), verify in the source file, cite paths. Use when asked what their notes/files say about something.
dependencies_files: [tools/tool_use_skill.py, tools/tool_hybrid_search.py, tools/tool_semantic_search.py, tools/tool_lexical_search.py, tools/tool_read_file.py, tools/tool_sql_query.py]
---

# Corpus Q&A

Answer from the user's files, with citations, or say clearly that the
corpus doesn't contain the answer.

## Choosing the search mode

- `lexical_search` — exact tokens: names, error strings, IDs, code symbols,
  rare words. Fastest and most precise when the wording is known.
- `semantic_search` — paraphrased or conceptual questions where wording is
  unknown ("what did I decide about deployment?").
- `hybrid_search` — default when unsure; fuses both.

## Procedure

1. Search with the mode chosen above. If zero hits, retry once with the
   other mode and a reworded query before concluding absence.
2. **Verify before answering.** Chunks are fragments: open the top file(s)
   with `read_file` around the hit to confirm context and catch negations
   ("we decided NOT to...").
3. Answer with the fact first, then `(from <file path>)` citations.
4. If the corpus is silent, say so explicitly and offer to search the web
   instead — do not pad an absence with general knowledge as if it came
   from their files.

## Useful checks

- `sql_query`: `SELECT modality, COUNT(*) FROM files GROUP BY modality` to
  see what's indexed at all; `SELECT path FROM files WHERE path LIKE ?` to
  confirm a specific document made it in.
- A file in `files` may still have pending tasks: check `task_queue` status
  for it before declaring its content missing.
