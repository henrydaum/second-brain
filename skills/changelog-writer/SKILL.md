---
name: changelog-writer
description: Turn git history into a human changelog: group commits by theme, translate to user-facing language. Use for 'write a changelog', 'what changed since v1.0', release notes.
dependencies_files: [tools/tool_use_skill.py, tools/tool_run_command.py, tools/tool_edit_file.py]
---

# Changelog writer

Commits are for developers; changelogs are for users. The job is
translation, not transcription.

## Procedure

1. Establish the range: `git tag --sort=-creatordate` then
   `git log <last-tag>..HEAD --oneline` via `run_command` (or a date
   range if untagged). Note merge commits vs direct commits.
2. For substantive commits, read the real change when the message is
   vague: `git show <sha> --stat` first, full diff only when needed.
3. Group by what a USER would notice, not by file: Added / Changed /
   Fixed / Removed (Keep-a-Changelog shape). Internal refactors collapse
   to one line or vanish; a bug fix is described by symptom fixed ("no
   longer loses identity when switching accounts"), not by mechanism.
4. Order each section by impact. Call out breaking changes at the top
   with migration steps.
5. Write/prepend to CHANGELOG.md via `edit_file`; show the user the
   draft before committing anything.

## Pitfalls

- Commit messages lie by optimism; the diff is ground truth for anything
  surprising.
- One commit can be several changelog entries, several commits one entry.
- Don't fabricate version numbers — ask whether this is a tagged release
  or rolling notes.
