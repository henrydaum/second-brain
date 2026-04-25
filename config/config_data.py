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
                   {"type": "slider", "range": (min, max, divisions), "is_float": bool}
"""

from paths import DATA_DIR, ATTACHMENT_CACHE

SETTINGS_DATA = [
    # --- Directories ---
    ("Sync Directories", "sync_directories",
     "Folders to monitor for new and changed files. Sub-folders are included.",
     [str(ATTACHMENT_CACHE)],
     {"type": "json_list"}),

    ("Database Path", "db_path",
     "Path to the SQLite database file. Requires app restart to take effect.",
     str(DATA_DIR / "database.db"),
     {"type": "text"}),

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
     "Service names to load automatically on startup (e.g. [\"google_drive\"]).",
     ["web_search_provider", "timekeeper", "llm", "parser"],
     {"type": "json_list"}),

    # --- Frontends ---
    ("Enabled Frontends", "enabled_frontends",
     "Frontend modules to start on launch. Options: repl, telegram. Requires app restart.",
     ["repl", "telegram"],
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

    ("Reprocess Interval", "reprocess_interval",
     "Seconds between re-checking files for changes.",
     300,
     {"type": "slider", "range": (30, 3600, 119), "is_float": False}),

    # --- Telegram ---
    ("Telegram Bot Token", "telegram_bot_token",
     "Bot token from @BotFather. Required for Telegram frontend.",
     "",
     {"type": "text"}),

    ("Telegram Allowed User ID", "telegram_allowed_user_id",
     "Your Telegram user ID (integer). Only this user can interact with the bot. "
     "Send /start to @userinfobot to find yours.",
     0,
     {"type": "text"}),

    # --- Agent Profiles ---
    # Each profile bundles an LLM reference + scope (prompt suffix, tool/table
    # restrictions). Managed via /agent. The "default" profile is permanent and
    # follows the default LLM via the "default" sentinel.
    ("Agent Profiles", "agent_profiles",
     "Named agent profiles. Each references an LLM (by model_name or 'default') and adds optional scope.",
     {"default": {
         "llm": "default",
         "prompt_suffix": "",
         "tools_allow": None,
         "tools_deny": None,
         "tables_allow": None,
         "tables_deny": None,
     }},
     {"type": "json_dict", "hidden": True}),

    ("Active Agent Profile", "active_agent_profile",
     "Name of the currently active agent profile.",
     "default",
     {"type": "text", "hidden": True}),

]
