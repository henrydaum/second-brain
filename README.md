# The Data Refinery

A local-first file intelligence pipeline that watches your directories, parses every file it finds, and makes them searchable and queryable through an LLM-powered agent. Think of it as a personal data warehouse that builds itself.

## How It Works

The system is organized into four stages, inspired by the layered transport systems in Factorio: simple, reliable pipelines at the bottom; intelligent, adaptive agents at the top.

**Stage 0 — Services** are shared model backends (LLM, embeddings, OCR, Whisper, Google Drive). They have a load/unload lifecycle so you can bring resources up and down without restarting. Services are auto-discovered from `Stage_0/services/`.

**Stage 1 — Parsers** read files and produce standardized output. A registry maps file extensions to parser functions, so adding support for a new format is a single `register()` call. Parsers handle text, images, audio, video, tabular data, and container formats (ZIP, PDF with embedded images, etc.). Each parser also reports what *else* is in the file (`also_contains`), enabling multi-modal discovery — a PDF might yield text *and* flag that it contains images worth OCR-ing.

**Stage 2 — Pipeline** is the automated backbone. A file watcher monitors your configured directories, debounces filesystem events, and keeps a SQLite database in sync with what's on disk. When files appear or change, the orchestrator figures out which tasks apply (based on modality and dependency chains) and dispatches them to a thread pool. Eight tasks — text extraction, OCR, chunking, text & image embedding, container extraction, tabular textualization, and full-text indexing — run in the background without any user intervention. The orchestrator handles batching, concurrency limits, service gating, task versioning, and dependency resolution. Tasks are auto-discovered from `Stage_2/tasks/`.

**Stage 3 — Agent** is the query layer. Six tools wrap database queries, search indexes, and file rendering behind a uniform interface that doubles as LLM function-calling schemas. An agent loop connects a local LLM to the tool registry, so you can ask natural-language questions about your files and get grounded answers. Tools are auto-discovered from `Stage_3/tools/`.

A shared `DataRefineryContext` object flows through every layer, giving tasks and tools access to the database, config, services, parsers, and (for tools) the ability to call other tools.

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

A terminal REPL is also available via `--no-gui` and always runs in the background for debugging.

## Project Structure

```
The Data Refinery/
├── main.pyw              # Entry point — GUI + system tray (or --no-gui for REPL)
├── config_data.py        # Declarative settings schema (titles, types, defaults)
├── config_manager.py     # Loads/saves config.json, merges defaults
├── context.py            # DataRefineryContext — shared context for tasks & tools
├── controller.py         # Command layer between user input and the system
├── repl.py               # Terminal REPL (runs as background thread, or standalone)
│
├── gui/
│   ├── app.py            # Flet GUI — chat, commands, overlays, log viewer
│   ├── commands.py       # CommandEntry dataclass + CommandRegistry
│   └── renderers.py      # Modality renderers (image/audio/video/text/tabular carousel)
│
├── Stage_0/              # Services (shared model backends)
│   ├── BaseService.py    # Service interface with load/unload lifecycle
│   ├── auto_discover_services.py
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
│   ├── auto_discover_tasks.py
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
│   ├── agent.py          # LLM ↔ tool loop with conversation history
│   ├── system_prompt.py  # Dynamic system prompt builder
│   ├── SearchResult.py   # Search result dataclass
│   ├── auto_discover_tools.py
│   └── tools/
│       ├── tool_hybrid_search.py      # Fused lexical + semantic via Reciprocal Rank Fusion
│       ├── tool_lexical_search.py     # BM25 keyword search via SQLite FTS5
│       ├── tool_semantic_search.py    # Vector similarity search over embeddings
│       ├── tool_sql_query.py          # Direct SQL queries against the database
│       ├── tool_read_source_code.py   # Read Python source files
│       └── tool_render_files.py       # Display files to user (images, audio, video, etc.)
│
└── services/             # (Legacy location — services now live in Stage_0/services/)
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
cd "The Data Refinery"
pip install -r requirements.txt
```

Key dependencies include `flet`, `pystray`, `watchdog`, `PyMuPDF (fitz)`, `python-docx`, `python-pptx`, `Pillow`, `pandas`, `sentence-transformers`, `openai`, `av` (PyAV), and `py7zr`.

### Configure

On first run, the system creates a `config.json` with sensible defaults. The main thing you need to set is `sync_directories` — the folders you want the system to watch. You can edit `config.json` directly or use `/config` in the GUI.

```json
{
    "sync_directories": ["C:/Users/you/Documents", "D:/Projects"],
    "db_path": "database.db",
    "max_workers": 4,
    "llm_model_name": "gpt-5-mini",
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

For REPL-only mode (no GUI):

```bash
python main.pyw --no-gui
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
| `reload` | Hot-reload tasks and tools from disk |
| `stats` | System-wide statistics |
| `config` / `settings` | Open the settings panel (GUI) |
| `clear` | Clear chat conversation history |
| `quit` / `exit` | Graceful shutdown |

## Extending the System

All three plugin types (tasks, tools, services) support **auto-discovery with hot-reload**. Drop a new file in the right directory, run `/reload`, and it's live — no manual registration needed.

### Adding a Parser

Create a function that takes `(path, config, services)` and returns a `ParseResult`, then register it:

```python
# Stage_1/parsers/parse_my_format.py
from Stage_1.ParseResult import ParseResult
import Stage_1.registry as registry

def parse_my_format(path, config, services=None):
    content = do_your_thing(path)
    return ParseResult(modality="text", output=content)

registry.register(".myext", "text", parse_my_format)
```

### Adding a Task

Create a file named `task_*.py` in `Stage_2/tasks/` with a `BaseTask` subclass:

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

The orchestrator auto-discovers it on `/reload` and handles the rest — it won't dispatch until `extract_text` is done and the `llm` service is loaded.

### Adding a Tool

Create a file named `tool_*.py` in `Stage_3/tools/` with a `BaseTool` subclass. The schema you define becomes the LLM's function-calling interface automatically:

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
