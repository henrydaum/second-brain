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
     ["C:\\Users\\henry\\Documents\\My_Code\\Test Database"],
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
     [],
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

    # --- LLM ---
    ("LLM Model Name", "llm_model_name",
     "Model name for the language model API. Reloads the LLM service on save.",
     "gpt-5-mini",
     {"type": "text"}),

    ("LLM Endpoint", "llm_endpoint",
     "Custom API endpoint URL. Leave blank for the default OpenAI endpoint. Reloads the LLM service on save.",
     "",
     {"type": "text"}),

    ("LLM API Key", "llm_api_key",
     "API key or environment variable name for the LLM. Reloads the LLM service on save.",
     "OPENAI_API_KEY",
     {"type": "text"}),

    # --- Embedding ---
    ("Text Embedding Model", "embed_text_model_name",
     "SentenceTransformer model for text embeddings. Reloads the embedding service on save.",
     "BAAI/bge-m3",
     {"type": "text"}),

    ("Image Embedding Model", "embed_image_model_name",
     "CLIP model for image embeddings. Reloads the embedding service on save.",
     "clip-ViT-L-14",
     {"type": "text"}),

    ("GPU Acceleration", "embed_use_cuda",
     "Use GPU for embedding. Provides a significant speed-up. Reloads the embedding service on save.",
     True,
     {"type": "bool"}),

    ("Chunk Size", "embed_chunk_size",
     "Size in tokens for text splitting. Smaller chunks store specific facts; larger chunks preserve more context. Reloads the embedding service on save.",
     512,
     {"type": "slider", "range": (64, 2048, 31), "is_float": False}),

    ("Chunk Overlap", "embed_chunk_overlap",
     "Number of overlapping tokens between chunks. Preserves continuity across chunk boundaries.",
     50,
     {"type": "slider", "range": (0, 200, 40), "is_float": False}),

    # --- Display ---
    ("Max Query Rows", "max_query_rows",
     "Maximum rows returned from database queries.",
     25,
     {"type": "slider", "range": (5, 100, 19), "is_float": False}),
]
