"""
Single source of truth for all configuration settings.

Each entry: (title, variable_name, description, default, type_info)
  - title:       Human-readable label shown in the settings UI
  - variable_name: The config key stored in config.json
  - description: Help text shown below the setting
  - default:     Default value (determines type for the config creator)
  - type_info:   Dict controlling the UI widget:
                   {"type": "text"}       — single-line text field
                   {"type": "bool"}       — checkbox / switch
                   {"type": "json_list"}  — multiline text field expecting a JSON array
                   {"type": "slider", "range": (min, max, divisions), "is_float": bool}
"""

from paths import DATA_DIR

SETTINGS_DATA = [
    # --- Directories ---
    ("Sync Directories", "sync_directories",
     "Folders to monitor for new and changed files. Sub-folders are included.",
     [],
     {"type": "json_list"}),

    ("Database Path", "db_path",
     "Path to the SQLite database file. Requires app restart to take effect.",
     str(DATA_DIR / "database.db"),
     {"type": "text"}),

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
     ["web_search_provider"],
     {"type": "json_list"}),

    # --- Frontends ---
    ("Enabled Frontends", "enabled_frontends",
     "Frontend modules to start on launch. Options: gui, repl, api, mcp. Requires app restart.",
     ["gui", "repl", "api"],
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

    ("Reprocess Interval", "reprocess_interval",
     "Seconds between re-checking files for changes.",
     300,
     {"type": "slider", "range": (30, 3600, 119), "is_float": False}),

    # --- API ---
    ("API Port", "api_port",
     "Port for the REST API server (used by OpenClaw and other integrations). Requires app restart.",
     5123,
     {"type": "slider", "range": (1024, 65535, 64511), "is_float": False}),

    ("API Token", "api_token",
     "Bearer token for API authentication. Leave blank to allow unauthenticated local access.",
     "",
     {"type": "text"}),

    # --- MCP ---

    ("MCP Port", "mcp_port",
     "Port for the MCP server (used by Claude Desktop, Cursor, Hermes Agent, etc.). Requires app restart.",
     5124,
     {"type": "slider", "range": (1024, 65535, 64511), "is_float": False}),

]
