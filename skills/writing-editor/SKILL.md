---
name: writing-editor
description: Co-write and edit long-form documents iteratively: outline, draft, targeted revision passes in a file. Use for essays, posts, docs, or 'help me write/edit X'.
dependencies_files: [tools/tool_use_skill.py, tools/tool_read_file.py, tools/tool_edit_file.py, tools/tool_ask_user_question.py]
---

# Writing editor

Collaborative writing lives in a file, not in chat scrollback.

## Procedure

1. **Charter first** (one exchange): audience, purpose, length, register.
   Disagreements about drafts are usually disagreements about these.
2. **Outline in the file** (`edit_file` create, in a sync directory),
   get a nod on the outline before prose — outline edits cost 10x less
   than draft edits.
3. **Draft section by section.** After each section, show only what
   changed, not the whole document.
4. **Revise with targeted replaces** (`edit_file` replace with exact
   old_text), never whole-file rewrites of the user's own sentences —
   their voice is data, not noise to normalize.
5. **Separate passes, in order:** structure (does the argument sequence
   hold) -> paragraph (one idea each, earned transitions) -> line (cut
   filler, strong verbs, vary sentence length). Name the pass you're on.

## Editing discipline

- Critique concretely: not "weak opening" but "the claim in line 1 is
  defended only in the final section; move or foreshadow".
- Preserve hedges the author chose deliberately; kill hedges that are
  reflexive ("perhaps somewhat").
- When the user rejects an edit twice, the preference is signal: stop
  re-proposing it.
