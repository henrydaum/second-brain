# The Data Refinery

A local-first file intelligence pipeline that watches your directories, parses every file it finds, and makes them searchable and queryable through an LLM-powered agent. Think of it as a personal data warehouse that builds itself.

## How It Works

The system is organized into three stages, inspired by the layered transport systems in Factorio: simple, reliable pipelines at the bottom; intelligent, adaptive agents at the top.

**Stage 1 — Parsers** read files and produce standardized output. A registry maps file extensions to parser functions, so adding support for a new format is a single `register()` call. Parsers handle text, images, audio, video, tabular data, and container formats (ZIP, PDF with embedded images, etc.). Each parser also reports what *else* is in the file (`also_contains`), enabling multi-modal discovery — a PDF might yield text *and* flag that it contains images worth OCR-ing.

**Stage 2 — Pipeline** is the automated backbone. A file watcher monitors your configured directories, debounces filesystem events, and keeps a SQLite database in sync with what's on disk. When files appear or change, the orchestrator figures out which tasks apply (based on modality and dependency chains) and dispatches them to a thread pool. Tasks like `extract_text`, `ocr_images`, and `extract_container` run in the background without any user intervention. The orchestrator handles batching, concurrency limits, service gating, task versioning, and dependency resolution.

**Stage 3 — Agent** is the query layer. Tools wrap database queries, search indexes, and parser calls behind a uniform interface that doubles as LLM function-calling schemas. An agent loop connects a local LLM to the tool registry, so you can ask natural-language questions about your files and get grounded answers.

A shared `DataRefineryContext` object flows through every layer, giving tasks and tools access to the database, config, services, parsers, and (for tools) the ability to call other tools.

## Project Structure

```
The Data Refinery/
├── main.pyw              # Entry point — wires everything together, starts REPL
├── config.py             # Dataclass-based config (see config_manager for JSON I/O)
├── config_manager.py     # Loads/saves config.json, merges defaults
├── context.py            # DataRefineryContext — shared context for tasks & tools
├── controller.py         # Command layer between user input and the system
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
│       ├── task_ocr_images.py         # OCR images via Windows OCR service
│       ├── task_extract_container.py  # Unpack archives, register child files
│       └── task_embed_text.py         # (WIP) Embed text chunks for vector search
│
├── Stage_3/              # Agent
│   ├── BaseTool.py       # Tool interface, ToolResult, ToolRegistry
│   ├── agent.py          # LLM ↔ tool loop with conversation history
│   └── tools/
│       ├── tool_semantic_search.py  # (WIP) Vector search over embeddings
│       └── tool_lexical_search.py   # (WIP) Keyword search over extracted text
│
└── services/             # Shared model/service layer
    ├── llmService.py     # BaseLLM, LMStudioLLM, OpenAILLM (with tool calling)
    ├── embedService.py   # SentenceTransformerEmbedder (text & image)
    ├── ocrService.py     # WindowsOCR
    └── driveService.py   # Google Drive integration
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

Key dependencies include `watchdog`, `PyMuPDF (fitz)`, `python-docx`, `python-pptx`, `Pillow`, `pandas`, `sentence-transformers`, `openai`, `av` (PyAV), and `py7zr`.

### Configure

On first run, the system creates a `config.json` with sensible defaults. The main thing you need to set is `sync_directories` — the folders you want the system to watch.

```json
{
    "sync_directories": ["C:/Users/you/Documents", "D:/Projects"],
    "db_path": "database.db",
    "max_workers": 4,
    "llm_model_name": "gemma-3-4b-it@q4_k_s",
    "llm_endpoint": "http://127.0.0.1:1234",
    "embed_text_model_name": "BAAI/bge-m3",
    "embed_image_model_name": "clip-ViT-L-14",
    "embed_use_cuda": false,
    "ignored_folders": ["node_modules", "__pycache__", ".git", ".venv", "venv"],
    "skip_hidden_folders": true
}
```

### Run

```bash
python main.pyw
```

The system will scan your configured directories, index every supported file, and start watching for changes. You'll get an interactive REPL.

## REPL Commands

| Command | Description |
|---|---|
| `services` | List all services and their loaded/unloaded status |
| `load <name>` | Load a service (e.g. `load llm`, `load text_embedder`) |
| `unload <name>` | Unload a service to free resources |
| `tasks` | List all tasks with pending/running/done/failed counts |
| `pause <name>` | Pause a task — work stays queued but won't dispatch |
| `unpause <name>` | Resume a paused task |
| `reset <name>` | Reset all entries for a task back to PENDING |
| `retry <name>` | Retry only FAILED entries for a task |
| `retry all` | Retry all FAILED entries across every task |
| `stats` | System-wide statistics |
| `quit` | Graceful shutdown |

## Extending the System

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

Import the module in `Stage_1/parsers/__init__.py` (or in `registry.py` where the other parsers are imported) so it registers on startup.

### Adding a Task

Subclass `BaseTask` and declare what it works on:

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

Register it in `main.pyw`:

```python
from Stage_2.tasks.task_summarize import Summarize
orchestrator.register_task(Summarize())
```

The orchestrator handles the rest — it won't dispatch until `extract_text` is done and the `llm` service is loaded.

### Adding a Tool

Subclass `BaseTool`. The schema you define becomes the LLM's function-calling interface automatically:

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