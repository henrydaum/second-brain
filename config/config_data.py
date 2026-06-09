"""
Single source of truth for all configuration settings.

Each entry: (title, variable_name, description, default, type_info)
  - title:       Human-readable label shown in frontend config views
  - variable_name: The config key stored in config.json
  - description: Help text shown below the setting
  - default:     Default value (determines type for the config creator)
  - type_info:   Dict controlling the UI widget:
                   {"type": "text"}       — single-line text field
                   {"type": "bool"}       — boolean toggle control
                   {"type": "json_list"}  — multiline text field expecting a JSON array
                   {"type": "path"}       — single filesystem path (normalized; parent must exist)
                   {"type": "path_list"}  — multiline list of folder paths (normalized; each must exist)
                   {"type": "slider", "range": (min, max, divisions), "is_float": bool}
"""

from paths import DATA_DIR, ATTACHMENT_CACHE

# The kernel ships no scheduled jobs. The tasks they used to drive
# (update_titles, dream_memory) and the timekeeper service that fires them are
# installable from the store; a plugin registers its own default jobs on install.
DEFAULT_SCHEDULED_JOBS: dict = {}

SETTINGS_DATA = [
    # --- Directories ---
    ("Sync Directories", "sync_directories",
     "Folders to monitor for new and changed files. Sub-folders are included.",
     [str(ATTACHMENT_CACHE)],
     {"type": "path_list"}),

    ("Database Path", "db_path",
     "Path to the SQLite database file. Requires app restart to take effect.",
     str(DATA_DIR / "database.db"),
     {"type": "path"}),

    ("Attachment Cache Size (GB)", "attachment_cache_size_gb",
     "Maximum size of the attachment cache folder. When exceeded, oldest files are evicted (LRU by modification time).",
     2.0,
     {"type": "slider", "range": (0.1, 20.0, 199), "is_float": True}),

    # --- File Filtering ---
    ("Ignored Extensions", "ignored_extensions",
     "File extensions to skip during sync (JSON array, e.g. [\".tmp\", \".log\"]).",
     [],
     {"type": "json_list"}),

    ("Ignored Folders", "ignored_folders",
     "Folder names to skip during sync.",
     ["node_modules", "__pycache__", ".git", ".venv", "venv"],
     {"type": "json_list"}),

    ("Skip Hidden Folders", "skip_hidden_folders",
     "Skip folders whose names start with a dot.",
     True,
     {"type": "bool"}),

    # --- Services ---
    ("Auto-load Services", "autoload_services",
     "Managed service names to load automatically on startup. Extension services auto-load when installed.",
     ["llm"],
     {"type": "json_list"}),

    # --- Frontends ---
    ("Enabled Frontends", "enabled_frontends",
     "Frontend modules to start on launch. The kernel ships only the REPL; "
     "the Telegram frontend is installable from the store. Requires app restart.",
     ["repl"],
     {"type": "json_list"}),

    # --- Processing ---
    ("Max Workers", "max_workers",
     "Maximum parallel worker threads for task processing. Takes effect on save.",
     4,
     {"type": "slider", "range": (1, 16, 15), "is_float": False}),

    ("Poll Interval", "poll_interval",
     "Seconds between orchestrator polling cycles. Takes effect on save.",
     1.0,
     {"type": "slider", "range": (0.1, 10.0, 99), "is_float": True}),

    ("Task Timeout", "task_timeout",
     "Seconds before a task is considered timed out.",
     300,
     {"type": "slider", "range": (30, 600, 57), "is_float": False}),

    ("Tool Timeout", "tool_timeout",
     "Seconds before an agent tool call is forcibly abandoned and reported to the LLM as a timeout error.",
     600,
     {"type": "slider", "range": (30, 1800, 59), "is_float": False}),

    ("Skip Permissions", "skip_permissions",
     "Tool names whose permission dialogs are automatically approved unless a permission plugin rejects them.",
     [],
     {"type": "json_list", "scope": "user"}),

    ("Restart On Crash", "restart_on_crash",
     "Relaunch Second Brain automatically if the process crashes (including hard "
     "native crashes). Clean exits (/quit, Ctrl+C) never restart. Checked at "
     "launch, so enabling it takes effect on the next start.",
     True,
     {"type": "bool"}),

    ("Stall Timeout", "stall_timeout",
     "Stall watchdog: while healthy, the app touches a heartbeat file every few "
     "seconds; if Restart On Crash is supervising and sees no heartbeat for this "
     "many seconds, it kills and relaunches the app. 0 disables stall detection. "
     "Checked at launch.",
     120,
     {"type": "slider", "range": (0, 600, 20), "is_float": False}),

    # --- Plugin Supervisor ---
    # Fully automatic: quarantines sandbox/installed plugins that repeatedly
    # crash or time out, and warns on runaway memory. Built-in kernel plugins are
    # supervised but never auto-disabled. One knob — leave it on.
    ("Plugin Supervisor", "plugin_supervisor",
     "Automatically quarantine misbehaving plugins (repeated crashes/timeouts) "
     "and warn on runaway memory. Turn off only to debug a plugin.",
     True,
     {"type": "bool"}),

    ("Reprocess Interval", "reprocess_interval",
     "Seconds between re-checking files for changes.",
     300,
     {"type": "slider", "range": (30, 3600, 119), "is_float": False}),

    ("Scheduled Jobs", "scheduled_jobs",
     "JSON object keyed by job name describing scheduled event emissions.",
     DEFAULT_SCHEDULED_JOBS,
     {"type": "json_dict", "hidden": True}),

    # --- Agent Profiles ---
    # Each profile bundles an LLM reference + optional prompt/tool scope.
    # Managed via /agent. The "default" profile is permanent and
    # follows the default LLM via the "default" sentinel.
    ("Agent Profiles", "agent_profiles",
     "Named agent profiles. Each references an LLM (by model_name or 'default') and can narrow tool access for specialized agents such as builders, researchers, or communicators.",
     {"default": {
         "llm": "default",
         "prompt_suffix": "",
         "whitelist_or_blacklist_tools": "blacklist",
         "tools_list": [],
     }},
     {"type": "json_dict", "hidden": True}),

    ("Active Agent Profile", "active_agent_profile",
     "Name of the currently active agent profile.",
     "default",
     {"type": "text", "hidden": True, "scope": "user"}),

    # --- Frontend Profiles ---
    # One profile per real frontend (keyed by frontend name). Each picks the
    # agent profile sessions on that frontend use and narrows which slash
    # commands the user may run there. A frontend with no entry is unrestricted
    # and follows the global active agent profile. Managed via /frontends.
    ("Frontend Profiles", "frontend_profiles",
     "Per-frontend access profiles. Each references an agent profile (by name or "
     "'default') and can whitelist/blacklist slash commands so a user-facing "
     "transport can expose a restricted agent and command set.",
     {},
     {"type": "json_dict", "hidden": True}),

    ("Restore Last Conversation on Startup", "startup_restore_conversation",
     "When enabled, the most recently active conversation is reloaded automatically when a frontend starts.",
     True,
     {"type": "bool", "scope": "user"}),

]
