---
name: skill-creator
description: How to write a new skill (SKILL.md) for this system — use when asked to create, improve, or convert knowledge into a skill.
dependencies_files: tools/tool_use_skill.py
---

# Creating skills

A skill is a folder containing a `SKILL.md`: markdown instructions for one
kind of task, with a small frontmatter header. Only the frontmatter goes in
the system prompt index; the body loads on demand via `use_skill`. Write
accordingly: the description sells the skill to your future self, the body
is the playbook.

## Where to put it

Author new skills in the sandbox tree:
`<DATA_DIR>/sandbox_plugins/skills/<skill-name>/SKILL.md`
(`/skills create` scaffolds this, or write the file directly with a file
tool). Built-in skills under the store `skills/` tree belong to the repo —
do not edit those; create a sandbox skill instead, which wins on name
collision.

## Format

```markdown
---
name: lowercase-with-dashes
description: One line — what the skill covers AND when to load it.
---

# Title

Step-by-step instructions...
```

Rules:
- `name`: lowercase letters, digits, dashes. Must match how it's referenced.
- `description`: a single line, under ~200 chars. It is the ONLY thing the
  agent sees before deciding to load the skill, so include trigger words a
  task would naturally contain ("use when...").
- Body: imperative, stepwise, concrete. Name exact commands, file paths, and
  tools. State what NOT to do — failure modes you have already hit are the
  most valuable content.
- Support files: extra reference files in the same folder are fine; mention
  them by filename in the body so the agent reads them with a file tool.
- Keep one skill = one task. Two loosely related procedures are two skills.

## Retrospective skill creation (the high-value pattern)

When you finish a task that took fumbling — wrong attempts, discovered
constraints, a final working procedure — distill it:
1. Write down the working procedure only, not the detours.
2. Record the dead ends as one short "pitfalls" list.
3. Save it as a skill named after the task, with a description containing
   the words the user actually used when asking.

## Quality check before saving

- Would the description alone make you load this at the right moment?
- Could another agent follow the body with no memory of this conversation?
- Is everything in it durable, or does it contain session-specific paths,
  IDs, or temporary state? (Remove those or make them parameters.)
