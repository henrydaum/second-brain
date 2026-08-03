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

import trees
from paths import DATA_DIR

# The kernel ships Timekeeper as the lightweight event clock, but no scheduled
# jobs. Store packages register their own jobs when they need recurring work.
DEFAULT_SCHEDULED_JOBS: dict = {}

SETTINGS_DATA = [
    # --- Directories ---
    ("Sync Directories", "sync_directories",
     "Folders to monitor for new and changed files. Sub-folders are included.",
     [str(trees.attachment_cache())],
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
     ["timekeeper"],
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

    # --- Agent ---
    ("Default Tool Call Limit", "default_tool_max_calls",
     "How many times the agent may call any one tool per message, for every "
     "tool that does not say otherwise. A tool declares `max_calls` only when "
     "its own nature bounds it — something that should be called once, or not "
     "at all after the first answer — so this is the number that actually "
     "governs almost every call. It also sets the turn's iteration budget, "
     "which is derived from the sum across the tools in scope.",
     25,
     {"type": "slider", "range": (1, 100, 99), "is_float": False}),

    # --- Subagents ---
    # Kernel settings because spawning is kernel routing. They were declared by
    # the store's spawn tool, which is exactly the ownership accident
    # `rehome_kernel_keys` exists to undo.
    ("Max Concurrent Subagents", "max_concurrent_subagents",
     "How many spawned agents may run at once. Also sets the ceiling on the "
     "LLM box pool, since a subagent plus the foreground turn is the most "
     "concurrent model calls that can exist.",
     4,
     {"type": "slider", "range": (1, 16, 15), "is_float": False}),

    ("Max Subagent Depth", "max_subagent_depth",
     "How deep spawned agents may nest. 1 (default) means a subagent may not "
     "spawn agents of its own. Raise it only deliberately: a fan-out is "
     "multiplicative, and nothing in the tree can answer an approval dialog.",
     1,
     {"type": "slider", "range": (1, 4, 3), "is_float": False}),

    ("Subagent Timeout", "subagent_timeout_seconds",
     "Max seconds a spawned agent may run. A child still running at this "
     "deadline is cancelled and reported as failed — never silently dropped.",
     300,
     {"type": "slider", "range": (30, 3600, 100), "is_float": False}),

    ("Keep Attachments Available Across Turns", "keep_attachments_available_across_turns",
     "Keep attached files available to the model after the first agent response. Useful for repeated media inspection, but native image/audio/video inputs may increase LLM cost.",
     False,
     {"type": "bool"}),


    ("Reveal User Commands to Agent", "reveal_user_commands",
     "Mirror completed slash commands into the conversation as a note the "
     "agent can see, so it knows when you changed state out-of-band (e.g. "
     "/config, /services). Records the command name and argument names only "
     "— never argument values, which can carry secrets.",
     False,
     {"type": "bool"}),

    ("Allowed Network Hosts", "net_allowed_hosts",
     "Hosts sandboxed plugins may reach without asking you first, e.g. "
     "api.search.brave.com. A bare domain also covers its subdomains. "
     "Outbound requests are the one control that makes broad file and "
     "database reads safe, so anything not listed here raises an approval "
     "dialog naming the host — this list is how you decide once instead of "
     "every time. A plugin cannot add to it.",
     [],
     {"type": "json_list"}),

    ("Writable Directories", "fs_writable_dirs",
     "Folders the agent may create, edit, move and delete files in without "
     "asking you first — your own project directory, say, rather than only "
     "its sandbox tree. Subfolders are covered. This is about writing: "
     "reading is not restricted by it, and neither is anything else. "
     "Deletes are included, so point it at somewhere you keep in version "
     "control. Second Brain's own program files and installed packages are "
     "never opened by this list, even if a folder here contains them — "
     "otherwise the agent could edit the rules that decide what it may do. "
     "A plugin cannot add to it.",
     [],
     {"type": "json_list"}),

    ("Allowed Command Prefixes", "shell_allowed_prefixes",
     "Commands sandboxed code may run without asking you first, named by "
     "program and subcommand — `git push`, say, or just `pytest`. Flags and "
     "arguments after the prefix are not checked, so list only a verb you "
     "would be happy to see run with any arguments; it is the same bargain as "
     "the host list, which does not check the URL path either. Matched against "
     "the exact argument list a plugin passes, so anything naming a shell, or "
     "carrying a shell metacharacter anywhere, is still asked about as usual. "
     "A plugin cannot add to it.",
     [],
     {"type": "json_list"}),

    ("Data Retention (Days)", "data_retention_days",
     "Delete data older than this many days: idle conversations (and their "
     "messages), action-ledger rows, and finished task-run records. Anything "
     "still in use is safe — a conversation's clock resets on every new "
     "message. 0 keeps everything forever.",
     90,
     {"type": "slider", "range": (0, 3650, 100), "is_float": False}),

    ("Restart On Crash", "restart_on_crash",
     "Relaunch Second Brain automatically if the process crashes (including hard "
     "native crashes). Clean exits (/quit, Ctrl+C) never restart. Checked at "
     "launch, so enabling it takes effect on the next start.",
     True,
     {"type": "bool"}),

    ("Scheduled Jobs", "scheduled_jobs",
     "JSON object keyed by job name describing scheduled event emissions.",
     DEFAULT_SCHEDULED_JOBS,
     {"type": "json_dict", "hidden": True}),

    # --- The LLM ---
    # Declared here because talking to a model is kernel routing now: the
    # ``llm/`` package owns profiles and brains, and backends are installable
    # helpers rather than a service. ``service_llm.py`` used to declare these
    # two, and when it was absorbed the declaration was not re-homed — so they
    # belonged to nobody.
    #
    # That is not cosmetic. ``config_manager.save`` keeps a key out of
    # config.json when it is a plugin key *or already present in
    # plugin_config.json*, so an undeclared setting's home was decided by
    # whichever file happened to hold it. Present, it stayed put; absent — a
    # fresh install, or any write that did not carry it — it landed in
    # config.json while every reader looked in plugin_config.json, and the
    # user's model configuration silently vanished. See ``_rehome_kernel_keys``.
    ("LLM Profiles", "llm_profiles",
     "Named model profiles. Each carries an endpoint, a secret_llm_api_key, "
     "a context size and the modalities the model accepts natively. "
     "Managed via /llm.",
     {},
     {"type": "json_dict", "hidden": True}),

    ("Default LLM Profile", "default_llm_profile",
     "Name of the profile used when nothing selects another.",
     "",
     {"type": "text", "hidden": True}),

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
