---
name: file-organizer
description: Audit and tidy sync directories: find duplicates, misfiled and stale files, propose batch moves/renames, execute after approval. Use for 'organize my files', 'find duplicates', 'clean up folder X'.
dependencies_files: [tools/tool_use_skill.py, tools/tool_sql_query.py, tools/tool_run_command.py, tools/tool_ask_user_question.py]
---

# File organizer

Propose-then-execute. Nothing moves without an approved plan.

## Procedure

1. **Inventory from the index**: the `files` table holds path, modality,
   size, mtime for everything in sync directories. Profile via `sql_query`:
   counts by extension/directory, largest files, oldest untouched.
2. **Duplicates:** same size is the cheap candidate filter
   (GROUP BY size HAVING COUNT(*)>1), then confirm with a content hash
   via `run_command` before ever calling two files identical.
3. **Build a written plan**: a table of (current path -> action) where
   action is move/rename/delete-candidate. Group by reason ("12 screenshots
   -> media/screenshots/"). Naming: kebab-case, date-prefixed where
   chronology matters, no spaces.
4. **Approve then execute.** Present the plan; on approval execute moves
   with `run_command` in small batches, verifying each batch landed before
   the next.
5. The pipeline watcher re-indexes moved files automatically; mention that
   stale paths fix themselves within a minute.

## Hard rules

- Deletions are quarantine-moves to an `_archive/` folder; only the user
  permanently deletes.
- Never touch DATA_DIR internals, dotfiles, or anything outside the
  agreed directories.
- Renames must not break references: grep the corpus for the old filename
  before renaming documents other files might cite.
