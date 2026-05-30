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

from paths import DATA_DIR, ATTACHMENT_CACHE, ROOT_DIR, SANDBOX_TECHNIQUES, TECHNIQUES_DIR

DEFAULT_WEB_CREDITS = {
    "costs": {"ai_prompt": 10, "uncached_render": 1},
    "free": {"five_hours": 60, "week": 600},
    "pack": {"credits": 1000, "price_cents": 299, "stripe_price_id": ""},
}

DEFAULT_SCHEDULED_JOBS = {
    "cleanup": {
        "channel": "cleanup_due",
        "cron": "0 4 * * *",
        "enabled": True,
        "payload": {},
    },
}

SETTINGS_DATA = [
    # --- Directories ---
    ("Sync Directories", "sync_directories",
     "Folders to monitor for new and changed files. Sub-folders are included.",
     [str(ATTACHMENT_CACHE), str(TECHNIQUES_DIR), str(ROOT_DIR / "plugins" / "techniques"), str(SANDBOX_TECHNIQUES)],
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
     "Service names to load automatically on startup.",
     ["timekeeper", "web_search_provider", "llm", "parser", "text_embedder", "image_embedder", "gmail", "technique_worker_pool", "credits"],
     {"type": "json_list"}),

    # --- Canvas rendering ---
    ("PNG Compression Level", "render_png_compress_level",
     "zlib compression level for cached canvas renders (0=no compression, 1=fastest, 6=PIL default, 9=smallest). 1 is the right choice for an interactive render path: ~40% faster encoding than 6 with only ~5% larger files. Everything in the render cache is PNG — there is no separate download format.",
     1,
     {"type": "slider", "range": (0, 9, 9), "is_float": False}),

    ("Render Memory Ceiling (MB)", "render_memory_max_mb",
     "Hard upper bound on memory granted to a single technique subprocess. The renderer scales the cap with canvas size automatically (~1.2 GB at 2048², ~3.8 GB at 4096²); this value clamps that scaling so a runaway High-resolution download can't swap the machine. On an 8 GB system, 3072 leaves headroom for the OS and a browser; raise on systems with more RAM.",
     3072,
     {"type": "slider", "range": (768, 8192, 29), "is_float": False}),

    ("Web Credits", "web_credits",
     "Public web billing: prompt and uncached-render costs, free limits, and the Stripe credit pack.",
     DEFAULT_WEB_CREDITS,
     {"type": "json_dict"}),

    # --- Frontends ---
    ("Enabled Frontends", "enabled_frontends",
     "Frontend modules to start on launch. Requires app restart.",
     ["web", "repl", "telegram"],
     {"type": "json_list"}),

    # --- Processing ---
    ("Max Workers", "max_workers",
     "Maximum parallel worker threads for task processing. Takes effect on save.",
     2,
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
     120,
     {"type": "slider", "range": (30, 1800, 59), "is_float": False}),

    ("Skip Permissions", "skip_permissions",
     "Tool names whose permission dialogs are automatically approved when plan mode is off.",
     [],
     {"type": "json_list"}),

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
         "tools_list": []
     },
     "artist": {
         "llm": "default",
         "prompt_suffix": "",
         "whitelist_or_blacklist_tools": "whitelist",
         "tools_list": [
            "search_techniques",
            "execute_technique",
            "manage_layers",
            "read_technique"
            ]
     },
     "artist_author": {
         "llm": "default",
         "prompt_suffix": "",
         "whitelist_or_blacklist_tools": "whitelist",
         "tools_list": [
            "search_techniques",
            "create_technique",
            "update_technique",
            "delete_technique",
            "execute_technique",
            "manage_layers",
            "read_technique",
            "read_technique_guide"
            ]
     }},
     {"type": "json_dict", "hidden": True}),

    ("Active Agent Profile", "active_agent_profile",
     "Name of the currently active agent profile.",
     "default",
     {"type": "text", "hidden": True}),

    ("Restore Last Conversation on Startup", "startup_restore_conversation",
     "When enabled, the most recently active conversation is reloaded automatically when a frontend starts.",
     True,
     {"type": "bool"}),

    ("Last Active Conversation", "last_active_conversation_id",
     "Conversation id last touched by a user-driven session. Restored on startup so the user reopens where they left off.",
     None,
     {"type": "integer", "hidden": True}),

]
