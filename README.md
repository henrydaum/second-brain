# Second Brain

A local-first file intelligence pipeline that watches your directories, parses every file it finds, and makes them searchable and queryable through a self-extending LLM agent. Think of it as a personal data warehouse that builds itself — and an agent that can extend its own capabilities at runtime.

## How It Works

The system is organized into four stages, inspired by the layered transport systems in Factorio: simple, reliable pipelines at the bottom; intelligent, adaptive agents at the top.

**Stage 0 — Services** are shared model backends (LLM, embeddings, OCR, Whisper, Google Drive). They have a load/unload lifecycle so you can bring resources up and down without restarting. New services can be added at runtime by the agent.

**Stage 1 — Parsers** read files and produce standardized output. A registry maps file extensions to parser functions, so adding support for a new format is a single `register()` call. Parsers handle text, images, audio, video, tabular data, and container formats (ZIP, PDF with embedded images, etc.). Each parser also reports what *else* is in the file (`also_contains`), enabling multi-modal discovery — a PDF might yield text *and* flag that it contains images worth OCR-ing.

**Stage 2 — Pipeline** is the automated backbone. A file watcher monitors your configured directories, debounces filesystem events, and keeps a SQLite database in sync with what's on disk. When files appear or change, the orchestrator figures out which tasks apply (based on modality and dependency chains) and dispatches them to a thread pool. Eight built-in tasks — text extraction, OCR, chunking, text & image embedding, container extraction, tabular textualization, and full-text indexing — run in the background without any user intervention. The orchestrator handles batching, concurrency limits, service gating, task versioning, and dependency resolution. New tasks can be added at runtime by the agent (see Sandbox Plugins below).

**Stage 3 — Agent** is the query and authoring layer. Tools wrap database queries, search indexes, and file rendering behind a uniform interface that doubles as LLM function-calling schemas. An agent loop connects a local LLM to the tool registry, so you can ask natural-language questions about your files and get grounded answers. But the agent can also *extend itself*: a `build_plugin` tool lets the LLM create, edit, and delete new tools, tasks, and services at runtime — writing Python files to a sandbox directory and registering them immediately without a restart. A `run_command` tool gives the agent shell access to read source code, install packages, and inspect the system. Tools, tasks, and services are all auto-discovered from both the source tree (read-only) and the sandbox (agent-writable).

A shared `SecondBrainContext` object flows through every layer, giving tasks and tools access to the database, config, services, parsers, tool registry, orchestrator, and (for tools) the ability to call other tools.

## The GUI

The default interface is a Flet-based desktop app with a chat-first design. Plain text goes to the LLM agent; slash-prefixed commands (e.g. `/services`, `/load llm`) control the system. An autocomplete popup appears when typing `/`.

Features:
- **Chat with your files** — ask the agent natural-language questions and get grounded answers with source citations
- **Slash commands** — manage services, tasks, tools, and config without leaving the window
- **Tool form overlay** — `/call <tool>` opens a dynamic form auto-generated from the tool's JSON schema
- **Rich rendering** — query results display images, audio players, video players, tabular data, and text previews inline
- **Settings panel** — `/config` opens a GUI settings editor
- **Log viewer** — click the status bar to see the full log stream
- **System tray** — the window minimizes to tray on close; right-click the tray icon to show/hide or quit

A terminal REPL is also available and runs alongside the GUI by default. Configure which frontends start via the `enabled_frontends` setting (options: `gui`, `repl`).

## Project Structure

```
Second Brain/
├── main.pyw              # Entry point — starts configured frontends (gui, repl)
├── plugin_discovery.py   # Unified plugin loader (tools, tasks, services) — baked-in + sandbox
├── paths.py              # Centralized path constants (ROOT_DIR, DATA_DIR, SANDBOX_*)
├── config_data.py        # Declarative settings schema (titles, types, defaults)
├── config_manager.py     # Loads/saves config.json, merges defaults, migration
├── context.py            # SecondBrainContext — shared context for tasks & tools
├── controller.py         # Command layer between user input and the system
│
├── gui/
│   ├── app.py            # Flet GUI — chat, commands, overlays, log viewer
│   ├── commands.py       # CommandEntry dataclass + CommandRegistry
│   ├── repl.py           # Terminal REPL (runs as background thread, or standalone)
│   └── renderers.py      # Modality renderers (image/audio/video/text/tabular carousel)
│
├── Stage_0/              # Services (shared model backends)
│   ├── BaseService.py    # Service interface with load/unload lifecycle
│   └── services/
│       ├── llmService.py       # OpenAI-compatible LLM (LM Studio, OpenAI, etc.)
│       ├── embedService.py     # SentenceTransformer (text) + CLIP (image) embeddings
│       ├── ocrService.py       # Windows native OCR
│       ├── whisperService.py   # Whisper speech-to-text
│       └── driveService.py     # Google Drive sync
│
├── Stage_1/              # Parsers
│   ├── registry.py       # Extension → parser mapping, main parse() entry point
│   ├── ParseResult.py    # Standardized parser output dataclass
│   └── parsers/
│       ├── parse_text.py       # Plain text, PDF, DOCX, PPTX, code, Google Docs
│       ├── parse_image.py      # PNG, JPEG, HEIC, etc. → PIL.Image
│       ├── parse_audio.py      # WAV, MP3, FLAC, etc. → numpy array
│       ├── parse_video.py      # MP4, MKV, etc. → av.Container (lazy)
│       ├── parse_tabular.py    # CSV, XLSX, Parquet, SQLite → pandas DataFrame
│       └── parse_container.py  # ZIP, TAR, RAR, 7z → extracted child paths
│
├── Stage_2/              # Pipeline
│   ├── database.py       # SQLite: files table, task_queue, dynamic output tables
│   ├── watcher.py        # Filesystem watcher (watchdog) with debouncing & ghost cleanup
│   ├── orchestrator.py   # Task dispatcher — modality routing, deps, concurrency
│   ├── BaseTask.py       # Task interface + TaskResult dataclass
│   └── tasks/
│       ├── task_extract_text.py       # Parse text content, store in DB
│       ├── task_extract_container.py  # Unpack archives, register child files
│       ├── task_ocr_images.py         # OCR images via Windows OCR service
│       ├── task_chunk_text.py         # Split extracted text into chunks for embedding
│       ├── task_embed_text.py         # Generate text embeddings (SentenceTransformer)
│       ├── task_embed_images.py       # Generate image embeddings (CLIP)
│       ├── task_textualize_tabular.py # Convert tabular data to searchable text
│       └── task_lexical_index.py      # Build FTS5 full-text search index
│
├── Stage_3/              # Agent
│   ├── BaseTool.py       # Tool interface, ToolResult, ToolRegistry
│   ├── tool_registry.py  # Thread-safe tool registry with call dispatch
│   ├── agent.py          # LLM ↔ tool loop with conversation history
│   ├── system_prompt.py  # Dynamic system prompt builder (refreshed every message)
│   ├── SearchResult.py   # Search result dataclass
│   └── tools/
│       ├── tool_hybrid_search.py      # Fused lexical + semantic via Reciprocal Rank Fusion
│       ├── tool_lexical_search.py     # BM25 keyword search via SQLite FTS5
│       ├── tool_semantic_search.py    # Vector similarity search over embeddings
│       ├── tool_sql_query.py          # Direct SQL queries against the database
│       ├── tool_render_files.py       # Display files to user (images, audio, video, etc.)
│       ├── tool_build_plugin.py       # Create/edit/delete sandbox plugins at runtime
│       └── tool_run_command.py        # Shell access for reading code, installing packages
│
├── templates/            # Plugin templates the agent reads before authoring
│   ├── tool_template.py
│   ├── task_template.py
│   └── service_template.py
│
└── DATA_DIR/             # %LOCALAPPDATA%/Second Brain/ (created at runtime)
    ├── config.json
    ├── database.db
    └── sandbox/          # Agent-writable plugins, hot-registered on create/edit
        ├── tools/        # tool_*.py — sandbox tools
        ├── tasks/        # task_*.py — sandbox tasks
        └── services/     # *.py — sandbox services
```

## Setup

### Prerequisites

- Python 3.11+
- For LLM features: a local model server like [LM Studio](https://lmstudio.ai/) running an OpenAI-compatible endpoint, or an OpenAI API key.
- For OCR: Windows (uses the native Windows OCR engine).
- For Google Drive sync: a Google Cloud service account or OAuth credentials.

### Install

```bash
git clone <repo-url>
cd "Second Brain"
pip install -r requirements.txt
```

Key dependencies include `flet`, `pystray`, `watchdog`, `PyMuPDF (fitz)`, `python-docx`, `python-pptx`, `Pillow`, `pandas`, `sentence-transformers`, `openai`, `av` (PyAV), and `py7zr`.

### Configure

On first run, the system creates a `config.json` in the data directory (`%LOCALAPPDATA%/Second Brain/`) with sensible defaults. The main thing you need to set is `sync_directories` — the folders you want the system to watch. You can edit `config.json` directly (use `/open_data` to find it) or use `/config` in the GUI.

```json
{
    "sync_directories": ["C:/Users/you/Documents", "D:/Projects"],
    "db_path": "database.db",
    "max_workers": 4,
    "llm_model_name": "your-model-name",
    "llm_endpoint": "http://127.0.0.1:1234",
    "embed_text_model_name": "BAAI/bge-m3",
    "embed_image_model_name": "clip-ViT-L-14",
    "embed_use_cuda": true,
    "embed_chunk_size": 512,
    "embed_chunk_overlap": 50,
    "ignored_folders": ["node_modules", "__pycache__", ".git", ".venv", "venv"],
    "skip_hidden_folders": true
}
```

### Run

```bash
python main.pyw
```

The GUI launches, the system scans your configured directories, indexes every supported file, and starts watching for changes. The LLM loads automatically in the background.

To run without the GUI, edit `enabled_frontends` in config.json (or via `/config`):

```json
"enabled_frontends": ["repl"]
```

## Commands

Available as slash commands in the GUI (with autocomplete) or as plain commands in the REPL.

| Command | Description |
|---|---|
| `help` | Show all available commands |
| `services` | List all services and their loaded/unloaded status |
| `load <name>` | Load a service (e.g. `load llm`, `load text_embedder`) |
| `unload <name>` | Unload a service to free resources |
| `tasks` | List all tasks with pending/processing/done/failed counts |
| `pipeline` | Show the task dependency graph |
| `pause <name>` | Pause a task — work stays queued but won't dispatch |
| `unpause <name>` | Resume a paused task |
| `reset <name>` | Reset all entries for a task back to PENDING |
| `retry <name>` | Retry only FAILED entries for a task |
| `retry all` | Retry all FAILED entries across every task |
| `tools` | List registered tools with enabled/disabled status |
| `enable <name>` | Enable a tool for agent use |
| `disable <name>` | Disable a tool |
| `call <tool>` | Call a tool directly (opens a form in GUI, takes JSON in REPL) |
| `reload` | Re-discover all plugins (tools, tasks, services) from disk |
| `stats` | System-wide statistics |
| `config` / `settings` | Open the settings panel (GUI) |
| `open_data` | Open the data folder in Explorer |
| `open_root` | Open the project root in Explorer |
| `clear` | Clear chat conversation history |
| `quit` / `exit` | Graceful shutdown |

## Extending the System

There are two ways to add functionality: **baked-in plugins** (committed to the source tree, read-only at runtime) and **sandbox plugins** (written by the agent or by hand to `DATA_DIR/sandbox/`, mutable at runtime).

### Sandbox Plugins (Agent-Authored)

The LLM agent can create, edit, and delete plugins at runtime using the `build_plugin` tool. Ask it in natural language — e.g. *"Build me a tool that fetches the weather"* — and it will:

1. Read the appropriate template (`templates/tool_template.py`, etc.)
2. Write the plugin file to the sandbox directory
3. Validate the code (syntax, structure, import checks, name collision detection)
4. Register it immediately — no restart or `/reload` needed

The agent can also edit plugins via search/replace blocks and delete them, with proper unregistration (including service unloading to free GPU/models). Sandbox plugins are namespaced separately so they can never overwrite baked-in ones.

Sandbox directories live in `DATA_DIR/sandbox/`:
- `sandbox/tools/` — tool plugins (`tool_*.py`)
- `sandbox/tasks/` — task plugins (`task_*.py`)
- `sandbox/services/` — service plugins (`*.py`)

### Baked-In Plugins (Developer-Authored)

For permanent additions committed to the repo, drop files into the appropriate source directory.

**Parser** — create a function that takes `(path, config, services)` and returns a `ParseResult`:

```python
# Stage_1/parsers/parse_my_format.py
from Stage_1.ParseResult import ParseResult
import Stage_1.registry as registry

def parse_my_format(path, config, services=None):
    content = do_your_thing(path)
    return ParseResult(modality="text", output=content)

registry.register(".myext", "text", parse_my_format)
```

**Task** — create `task_*.py` in `Stage_2/tasks/` with a `BaseTask` subclass:

```python
# Stage_2/tasks/task_summarize.py
from Stage_2.BaseTask import BaseTask, TaskResult

class Summarize(BaseTask):
    name = "summarize"
    version = 1
    modalities = ["text"]
    depends_on = ["extract_text"]
    requires_services = ["llm"]
    output_tables = ["summaries"]
    output_schema = """
        CREATE TABLE IF NOT EXISTS summaries (
            path TEXT PRIMARY KEY,
            summary TEXT
        );
    """
    batch_size = 4

    def run(self, paths, context):
        llm = context.services["llm"]
        results = []
        for path in paths:
            text = context.db.get_extracted_text(path)
            summary = llm.invoke([{"role": "user", "content": f"Summarize: {text}"}])
            results.append(TaskResult(success=True, data=[{"path": path, "summary": summary.content}]))
        return results
```

The orchestrator handles the rest — it won't dispatch until `extract_text` is done and the `llm` service is loaded.

**Tool** — create `tool_*.py` in `Stage_3/tools/` with a `BaseTool` subclass. The schema you define becomes the LLM's function-calling interface automatically:

```python
# Stage_3/tools/tool_file_info.py
from Stage_3.BaseTool import BaseTool, ToolResult

class FileInfo(BaseTool):
    name = "file_info"
    description = "Get metadata about a file in the database."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to look up"}
        },
        "required": ["path"]
    }

    def run(self, context, **kwargs):
        info = context.db.get_file(kwargs["path"])
        return ToolResult(data=info) if info else ToolResult.failed("File not found")
```

All plugin types (baked-in and sandbox) are discovered at startup by `plugin_discovery.py`. Use `/reload` as a manual escape hatch to re-discover everything from disk.

## Supported File Types

| Modality | Extensions |
|---|---|
| Text | `.txt`, `.md`, `.py`, `.js`, `.ts`, `.html`, `.css`, `.json`, `.yaml`, `.toml`, `.xml`, `.pdf`, `.docx`, `.pptx`, `.gdoc`, and more |
| Image | `.png`, `.jpg`, `.jpeg`, `.webp`, `.tiff`, `.bmp`, `.ico`, `.heic`, `.heif` |
| Audio | `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.aac`, `.wma` |
| Video | `.mp4`, `.mkv`, `.avi`, `.mov`, `.webm`, `.wmv`, `.flv` |
| Tabular | `.csv`, `.tsv`, `.xlsx`, `.xls`, `.parquet`, `.feather`, `.sqlite`, `.db` |
| Container | `.zip`, `.tar`, `.gz`, `.7z`, `.rar` |

## License

TBD
