---
name: sb-troubleshooting
description: Diagnose Second Brain itself: boot issues, plugins not loading, pipeline stuck, LLM/config problems — using the log, status commands, and the task queue. Use when something in this system misbehaves.
dependencies_files: [tools/tool_use_skill.py, tools/tool_read_file.py, tools/tool_sql_query.py, tools/tool_slash_command.py]
---

# Second Brain troubleshooting

Diagnose from evidence: the log, the status commands, the task queue.

## Evidence sources

1. `app.log` in DATA_DIR — `read_file` reads .log files newest-first by
   design; start with the last ~100 lines. Grep-worthy markers:
   "Could not import", "registration failed", "Service crashed during
   load", "timed out", "Traceback".
2. Status commands via `slash_command`: /services (what loaded),
   /tools, /tasks (pipeline + pause state), /packages list,
   /locations (where DATA_DIR actually is).
3. Pipeline state via `sql_query`:
   SELECT task_name, status, COUNT(*) FROM task_queue GROUP BY 1,2 —
   a pile of FAILED rows names the broken task; `task_runs` holds recent
   run errors.
4. The action ledger via `sql_query`: `action_ledger` is the kernel's
   flight recorder — one row per action, with `origin` ('user_enact' =
   frontend actions, 'agent_enact' = agent moves, 'system' = installs,
   config saves, conversation ops including *refused* ones), plus
   action_type, name, ok, error_code, args_json, call_id, duration_ms.
   It is write-optimized filler by volume: **never read it linearly, and
   only open it when you already know what you're looking for.** Always
   filter — by conversation_id, session_key, origin, ok=0, or a ts
   window — and ORDER BY id DESC LIMIT 20. Good questions it answers:
   "what exactly ran in this conversation and how long did each step
   take", "what did that package install touch" (data_json carries the
   store commit + file hashes), "which action was refused and why",
   "what happened while no one was watching" (session_key not the
   active frontend's).

## Common failures -> first move

- **Plugin missing from /tools or /services**: the load-time log line says
  why (import error, name collision, bad contract). A service that loads
  but gates its tools usually forgot self.loaded = True.
- **"Required services not available"**: /services — the dependency
  either isn't installed, isn't in autoload, or failed to load (log).
- **Pipeline idle with PENDING work**: check the task's required services
  and whether the task is paused in /tasks.
- **No LLM / refuses to chat**: /setup not run or profile broken —
  check llm_profiles via /config.
- **Installed package behaving oddly after update**: reinstall via
  /packages (store always wins); check the file actually changed.

## Rules

- Read-only first: diagnose fully before proposing config edits, and never
  edit the database to "fix" state — find the code path that owns it.
- One change at a time, re-test, then the next.
