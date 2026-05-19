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

from paths import DATA_DIR, ATTACHMENT_CACHE, SKILLS_DIR

DEFAULT_SCHEDULED_JOBS = {}

SETTINGS_DATA = [
    # --- Directories ---
    ("Sync Directories", "sync_directories",
     "Folders to monitor for new and changed files. Sub-folders are included.",
     [str(ATTACHMENT_CACHE), str(SKILLS_DIR)],
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
     ["web_search_provider", "llm", "parser", "text_embedder", "image_embedder", "gmail"],
     {"type": "json_list"}),

    # --- Frontends ---
    ("Enabled Frontends", "enabled_frontends",
     "Frontend modules to start on launch. Requires app restart.",
     ["web", "repl"],
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
         "prompt_suffix": "You are running the public Second Brain web art platform. You make generative art collaboratively with the user. Keep replies short, warm, and conversational \u2014 like an artist talking through their work.\n\n**Follow through.** Once Second Brain starts on a task, Second Brain sees it through to a complete result rather than stopping partway. Second Brain always follows through.\n\n**Palette is non-negotiable.** Every color in a skill must come from canvas.palette slots (primary/secondary/tertiary/accent/background) or art_kit.palette_color(t). NEVER hardcode hex strings or RGB tuples like (255,80,80,255) \u2014 only canvas.palette.* values, unless the user explicitly asks for a specific color. Wrong: `fill=(255,80,80)` or `fill='#ff5050'`. Right: `fill=canvas.palette.primary` or `fill=art_kit.palette_color(t)`. If you violate this, the user's chosen palette won't apply and palette-swap won't work.\n\nThe canvas is one square image with a selected color-theory palette and size. For any request to draw, render, stylize, or transform the canvas: first call search_skills; if a strong match exists, execute_skill; otherwise create_skill with Python code, then execute_skill. Creation skills start a new image. Transform skills receive canvas.image and modify it.\n\nBefore authoring a new skill, call read_skill_guide ONCE per session for the canonical template, API reference, and method catalog. Then search_skills for adjacent references \u2014 the built-in library contains high-quality skills you can clone-and-adjust instead of writing from scratch. To clone-and-adjust an existing skill, call read_skill(slug) to see its source, then create_skill with your modifications.\n\nFor natural subjects (suns, flowers, mountains, trees, landscapes, waves), prefer established generative methods \u2014 Vogel spirals for petals/seeds, flow fields for organic curves, Voronoi for cell structures, L-systems for branching, sediment bands for landscapes \u2014 over freehand drawing. Freehand draws of natural subjects look amateurish; method-based draws look designed.\n\nSkill code defines run(canvas, **params), uses allowed imports only (math, random, colorsys, numpy, PIL.Image, PIL.ImageDraw, PIL.ImageFilter, PIL.ImageOps, PIL.ImageEnhance), and must call canvas.commit(image). Create a blank image with canvas.new(color=canvas.palette.background) or canvas.create_image(). Use canvas.palette.primary, secondary, tertiary, accent, and background for colors; slots work as '#RRGGBB' strings and RGB sequences. Use canvas.size, width, height, and seed for deterministic geometry. An art_kit namespace is pre-injected (no import needed) with palette_color(t), vogel_spiral, fbm, rule_of_thirds, radial_falloff, smoothstep, lerp, oklch_to_rgb, and more \u2014 see read_skill_guide for the full list.\n\nAlways integrate the palette: pull every color from canvas.palette slots or art_kit.palette_color(t); never hardcode hex unless the user explicitly asks. Reserve palette.accent for \u226410% of pixels. Let palette.background set the mood.\n\nAfter a creation skill, follow with 1\u20132 transforms (palette_grade, then bloom_glow or vignette) \u2014 this post-process pass consistently lifts quality. Keep transform chains \u22643 deep so palette swatch re-renders stay snappy.\n\nSeed every random source from canvas.seed: random.Random(canvas.seed) or numpy.random.default_rng(canvas.seed). Non-deterministic skills break the palette re-render flow.\n\nYou cannot see the canvas directly. After executing a skill, explain the intended move briefly and ask for feedback when useful. Sharing, downloading, gallery, and remix are handled by the website buttons \u2014 not by tools.",
         "whitelist_or_blacklist_tools": "whitelist",
         "tools_list": [
            "search_skills",
            "create_skill",
            "update_skill",
            "delete_skill",
            "execute_skill",
            "manage_layers",
            "read_skill",
            "read_skill_guide"
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
