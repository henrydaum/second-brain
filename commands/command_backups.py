"""Slash command plugin for `/backups` — DATA_DIR backup and restore."""

dependencies_files = ['commands/helpers/datadir_backup.py']
dependencies_pip = []

from paths import DATA_DIR
from plugins.BaseCommand import BaseCommand
from plugins.frontends.helpers.formatters import detail_card, md_table
from runtime.ledger import record_system
from state_machine.conversation import FormStep

# Relative import so this file works from the installed/sandbox trees too.
from .helpers import datadir_backup as engine

ACTIONS = ["create", "list", "restore", "delete"]
ACTION_LABELS = ["Create a backup", "List backups", "Restore a backup", "Delete a backup"]
CONFIRM_YES = "yes"


class BackupsCommand(BaseCommand):
    """Create, list, restore, or delete DATA_DIR backup archives."""
    name = "backups"
    description = "Create, list, restore, or delete DATA_DIR backups"
    category = "Config & System"
    config_settings = [
        ("Backups Kept", "backup_keep_count",
         "How many backup archives to keep in DATA_DIR/backups; the oldest are pruned after each create. 0 keeps everything.",
         10, {"type": "integer"}),
    ]
    agent_prompt = ("/backups snapshots and restores the app's data directory "
                    "(database, config, memory, plugins). Restoring restarts the app.")

    def form(self, args, context):
        steps = [FormStep("action", _action_prompt(), True, enum=ACTIONS, enum_labels=ACTION_LABELS, columns=2)]
        action = args.get("action")
        if action == "create":
            steps.append(FormStep("name", "Optional backup name (letters, digits, . _ -). Leave blank for a timestamp.", False))
        elif action in {"restore", "delete"}:
            names = [b["name"] for b in engine.list_backups(DATA_DIR)]
            if names:
                steps.append(FormStep("backup", f"Choose the backup to {action}.", True, enum=names, columns=1))
                steps.append(FormStep("confirm", _confirm_prompt(action, args.get("backup")), True,
                                      enum=[CONFIRM_YES, "no"], enum_labels=_confirm_labels(action)))
        return steps

    def run(self, args, context):
        action = args.get("action") or "list"
        progress = _progress_for(context)
        try:
            if action == "create":
                return _run_create(args, context, progress)
            if action == "list":
                return _format_list()
            if action == "restore":
                return _run_restore(args, context, progress)
            if action == "delete":
                return _run_delete(args, context)
            return f"Unknown action: {action}"
        except engine.BackupError as e:
            record_system(context.db, action_type=f"backup_{action}", ok=False,
                          name=args.get("backup") or args.get("name"), error_message=str(e))
            return f"Backup {action} failed: {e}"


def _run_create(args, context, progress) -> str:
    result = engine.create_backup(DATA_DIR, context.db, name=args.get("name") or None,
                                  root_dir=context.root_dir, progress=progress)
    record_system(context.db, action_type="backup_create", ok=True, name=result["name"],
                  user_id=context.user_id,
                  data={k: result[k] for k in ("path", "bytes", "file_count", "manifest_sha256")})
    keep = int(context.config.get("backup_keep_count") or 0)
    pruned = engine.prune_backups(DATA_DIR, keep)
    if pruned:
        record_system(context.db, action_type="backup_prune", ok=True, data={"deleted": pruned, "keep": keep})
    card = detail_card(f"Backup '{result['name']}'", [
        ("Path", result["path"]),
        ("Size", _human_bytes(result["bytes"])),
        ("Files", result["file_count"]),
        ("Manifest SHA-256", result["manifest_sha256"][:16] + "..."),
    ])
    if pruned:
        card += f"\n\nPruned oldest backups: {', '.join(pruned)}."
    return card


def _run_restore(args, context, progress) -> str:
    if args.get("confirm") != CONFIRM_YES:
        return "Restore cancelled."
    result = engine.restore_backup(DATA_DIR, context.db, args.get("backup", ""), root_dir=context.root_dir, progress=progress)
    # Apply the restored config to the LIVE dict (context.config is a per-call
    # copy): main.pyw's restart/shutdown paths re-save the in-memory config on
    # the way down and would otherwise clobber the just-restored config files.
    runtime = getattr(context, "runtime", None)
    live = getattr(runtime, "config", None)
    if isinstance(live, dict):
        live.clear()
        live.update(result["config"])
        live.update(result["plugin_config"])
    record_system(context.db, action_type="backup_restore", ok=True, name=result["name"],
                  user_id=context.user_id,
                  data={"safety_backup": result["safety_backup"], "restored_files": result["restored_files"]})
    summary = (f"Restored backup '{result['name']}' ({result['restored_files']} files). "
               f"Safety backup: '{result['safety_backup']}'.")
    registry = getattr(context, "command_registry", None)
    if registry is None:
        return summary + " Restart Second Brain manually to finish."
    # Consent was collected by this form's confirm step; dispatch_dict runs the
    # host command directly (the state-machine approval frame doesn't apply here).
    registry.dispatch_dict("restart", {}, session_key=context.session_key)
    return summary + " Restarting..."


def _run_delete(args, context) -> str:
    if args.get("confirm") != CONFIRM_YES:
        return "Delete cancelled."
    result = engine.delete_backup(DATA_DIR, args.get("backup", ""))
    record_system(context.db, action_type="backup_delete", ok=True, name=result["name"],
                  user_id=context.user_id, data={"bytes": result["bytes"]})
    return f"Deleted backup '{result['name']}' ({_human_bytes(result['bytes'])})."


def _action_prompt() -> str:
    backups = engine.list_backups(DATA_DIR)
    if not backups:
        return "No backups yet.\n\nChoose a backup action."
    return _table(backups) + "\n\nChoose a backup action."


def _format_list() -> str:
    backups = engine.list_backups(DATA_DIR)
    if not backups:
        return "No backups yet. Create one with `/backups create`."
    return _table(backups)


def _table(backups: list[dict]) -> str:
    rows = [(b["name"], b["created_at"] or "?", _human_bytes(b["bytes"]),
             b["file_count"] or "?", (b["app_commit"] or "")[:8]) for b in backups]
    return md_table(["Name", "Created", "Size", "Files", "App commit"], rows)


def _confirm_prompt(action: str, backup) -> str:
    target = f"'{backup}'" if backup else "the selected backup"
    if action == "delete":
        return f"Delete backup {target} permanently?"
    return (f"Restoring {target} overwrites the database, config, memory, and "
            "installed/sandbox plugins with the backup's contents, then restarts "
            "Second Brain. A safety backup is taken first. Proceed?")


def _confirm_labels(action: str) -> list[str]:
    if action == "delete":
        return ["Yes — delete it", "No — cancel"]
    return ["Yes — restore and restart", "No — cancel"]


def _progress_for(context):
    """Progress sink that reaches the user's frontend, not just stdout."""
    runtime = getattr(context, "runtime", None)
    session_key = getattr(context, "session_key", None)
    if runtime is not None and session_key:
        return lambda message: runtime.push_message(session_key, message, source="backups")
    return lambda message: print(message, flush=True)


def _human_bytes(n) -> str:
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"
