---
name: store-publishing
description: How to publish or update a plugin on the Second Brain package store branch — use when moving a sandbox plugin into origin/store or editing store packages.
dependencies_files: tools/tool_use_skill.py
---

# Publishing to the package store

The store is the `store` git branch: a tree mirror of what
`installed_plugins/` would look like with everything installed
(`tools/tool_*.py`, `services/service_*.py`, ..., family `helpers/`,
plus `bundles/*.json` manifests). `/packages` installs read `origin/store`.

## Workflow

1. **Worktree**: `git worktree add ../sb-store store` (once; reuse it).
2. **Edit** files in the worktree. New plugins keep the same family layout
   and module-level literal `dependencies_files` / `dependencies_pip`.
3. **Validate**: from the repo root,
   `python scripts/package_publisher.py validate --path ../sb-store`
   — checks dependency metadata (AST-parsed) and bundle manifests.
4. **Local test before pushing**: commit in the worktree, then from the
   main repo `git fetch . store:refs/remotes/origin/store` — this points
   the local `origin/store` ref at your commit so `/packages install` and
   `build_install_plan` see it without touching the network. Test the
   install round-trip against a scratch DATA_DIR (set `LOCALAPPDATA` to a
   temp dir on Windows) so your live install is untouched.
5. **Push**: `git push origin store:store` when verified.

## Bundles

`bundles/<bundle_name>.json`:
```json
{"name": "Display Name", "description": "...", "files": ["tools/tool_x.py", "services/service_y.py"]}
```
Bundle files must be store-relative plugin/helper paths; the resolver pulls
each file's own `dependencies_files` transitively, so list entry points,
not their dependencies.

## Rules

- The store always wins on update: differing installed files are
  overwritten. Treat the branch as the newest version of everything.
- Installable paths: `.py` files in 2–3-part family paths (`<family>/x.py`,
  `<family>/helpers/x.py`) and **skill folders** (`skills/<name>/...`,
  any file type). A skill installs/uninstalls as one unit: targeting its name
  copies the whole folder. Declare plugin dependencies in the SKILL.md
  frontmatter (`dependencies_files: tools/tool_use_skill.py`) so installing
  the skill pulls the machinery it needs.
- Never commit `__pycache__` to the branch (it has a `.gitignore` now).
- All plugin PRs go through Henry — the store is trusted because it is
  reviewed, so review is the security boundary.
