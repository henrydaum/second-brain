# Second Brain

Second Brain is a local-first personal data engine.

It watches your folders, parses what it finds, builds a structured index in SQLite, and gives an LLM the tools to search, reason, act, schedule work, remember things, and extend the system itself.

This is not just "chat with your files." It is a private, always-on AI runtime for your data:

- a searchable file index
- a natural-language analyst
- a background scheduler
- an event-driven automation system
- a Telegram assistant
- a memory-backed agent
- a plugin platform that can author new tools, tasks, and services at runtime

If you want the short version: Second Brain is what happens when file intelligence, cron, a tool-using agent, and a local plugin runtime are built as one system instead of four separate products.

## Why It Matters

Most AI apps are stateless. Most file indexers are passive. Most automation tools are brittle. Second Brain is none of those.

It can:

- index your documents, code, PDFs, slides, spreadsheets, archives, images, audio, and video
- answer questions grounded in your own files with citations
- search by keyword, semantics, or a hybrid of both
- remember durable facts and preferences across sessions
- search the web when local knowledge is not enough
- run proactive subagents on schedules
- fire tasks from events, not just file changes
- push reminders, findings, daily briefs, and alerts into Telegram
- hot-load new tools, tasks, and services without a restart

It can be a private research assistant. It can be a file intelligence layer for your whole machine. It can be a reminder system. It can be a daily briefing engine. It can absolutely function like a personal AI calendar and operator for recurring work.

## Core Capabilities

### 1. Index Your World

Point it at one or more directories and it will continuously watch them, parse supported files, and keep the database in sync as files appear, change, or disappear.

Built-in indexing pipeline includes:

- text extraction
- OCR for images
- archive/container extraction
- chunking for embedding
- text embeddings
- image embeddings
- lexical full-text indexing
- tabular textualization for spreadsheets and data files

The result is a live knowledge base over your local files, not a one-shot import.

### 2. Search Like a Real System

Second Brain ships with multiple retrieval modes:

- `lexical_search` for exact terms and keyword-heavy queries
- `semantic_search` for meaning-based retrieval over embeddings
- `hybrid_search` for fused lexical + semantic ranking
- `sql_query` for direct inspection of the underlying SQLite database

You can ask normal-language questions, but you can also inspect the system with precision when you want to.

### 3. Run Background Subagents

Second Brain can schedule background agents to run later or run repeatedly.

That means you can create jobs like:

- "Every weekday at 8:00 AM, send me a briefing on new files in my research folder."
- "At 6:00 PM, remind me what is still unfinished."
- "Every hour, search the web for updates on a topic and send me only important changes."
- "On April 30 at 9:00 AM, review this folder and message me the top risks."

Jobs can be:

- one-time with an ISO datetime
- recurring with cron
- enabled or disabled without deleting them
- backed by input files you explicitly attach to the job

Scheduled subagents keep their own stored run history and can proactively push user-visible messages into chat.

### 4. Event-Driven Tasks

The system is no longer only file-driven.

Tasks can now be triggered by events through the internal event bus. That opens the door to workflows like:

- scheduled event emissions from the timekeeper service
- chained background runs
- approval workflows
- proactive notifications
- future external integrations that emit events into the system

Path-triggered tasks and event-triggered tasks share the same orchestration model, which makes the whole platform much more general.

### 5. Telegram Is a First-Class Frontend

Flet is gone.

Second Brain now ships with two primary frontends:

- Telegram bot
- Terminal REPL

Telegram is not an afterthought. It supports:

- slash commands
- autocomplete
- mobile-friendly responses
- file and media delivery
- interactive tool invocation
- approval prompts for sensitive actions
- proactive subagent push messages

This means your local system can act like a private mobile AI assistant without becoming a cloud SaaS product.

### 6. Durable Memory

Second Brain includes agent memory through `memory.md` in the data directory.

The agent can update that memory intentionally with `update_memory`. It is meant for durable context such as:

- preferences
- standing instructions
- durable facts
- recurring context that should shape future behavior

It is not meant for one-off reminders, transient task state, or short-lived updates that only matter in the moment.

On top of that, conversation history is stored in SQLite and can be revisited later with read-only SQL.

### 7. Web Search

Second Brain can search the public web through the built-in `web_search` tool.

It supports:

- Brave Search
- Brave Answers
- DuckDuckGo fallback when a Brave Search key is not configured

That means the agent is not trapped inside the local corpus. It can blend your private knowledge with current public information when appropriate.

### 8. Self-Extending Runtime

One of the most unusual parts of the project is that the agent can build new capabilities inside a sandbox at runtime.

If the current toolset cannot reasonably complete a task, the agent can use `build_plugin` to create, edit, or delete:

- tools
- tasks
- services

Those plugins are hot-registered immediately. No restart needed.

This means Second Brain is not just a fixed assistant. It can inspect its own architecture, generate a focused extension, and use that new capability right away.

## What You Can Use It For

- Personal search engine for your entire document corpus
- Codebase analyst over local repositories
- Research assistant that combines file search with live web search
- Daily briefings pushed to Telegram
- Reminder and recurring-task system powered by scheduled subagents
- AI calendar-like workflows using one-time and recurring jobs
- Archive and media intelligence across PDFs, images, video, audio, and spreadsheets
- Private long-term assistant with memory and conversation history
- Agentic automation that can build its own plugins when the right tool does not exist yet

## Architecture

The system is organized into four stages, with an event bus connecting long-lived components.

### Stage 0: Services

Shared backends with explicit load and unload lifecycles.

Built-in services include:

- `llm` - routed LLM service with named profiles
- `web_search_provider` - Brave Search / Brave Answers / DuckDuckGo fallback
- `timekeeper` - cron and one-time scheduling
- `ocr` - Windows OCR
- `whisper` - speech-to-text
- `text_embedder` - text embeddings
- `image_embedder` - image embeddings
- `google_drive` - Drive integration

The LLM layer supports profile routing, so you can switch models without changing the rest of the system.

### Stage 1: Parsers

Extension-driven parsers normalize raw files into structured outputs.

Supported modalities include:

- text
- image
- audio
- video
- tabular
- container

Parsers can also report `also_contains` hints, which allows multi-modal follow-up work. For example, a file can yield text and still announce that it contains images worth OCRing.

### Stage 2: Pipeline + Orchestration

This is the always-on execution layer.

It includes:

- a filesystem watcher
- a SQLite-backed task queue
- an event-trigger runner
- automatic dependency resolution from task reads/writes
- concurrency controls
- task pause, retry, reset, and timeout recovery
- downstream invalidation when upstream outputs change

There are now two kinds of work in the system:

- path-keyed tasks for files
- event-keyed tasks for runs triggered by bus events

That split is what enables both continuous file indexing and scheduled/proactive agents to coexist in one architecture.

### Stage 3: Agent + Tools

This is the reasoning and action layer.

The agent gets a dynamically rebuilt system prompt that includes:

- current date and time
- current tools
- current services
- current task pipeline state
- current file inventory
- current durable memory
- current sandbox plugins

That prompt pushes the assistant toward concise, grounded behavior: inspect the system first, prefer local evidence, and cite the files, tables, or tool results it relied on.

Built-in tools include:

| Tool | Purpose |
|---|---|
| `hybrid_search` | Best default search over indexed local files |
| `lexical_search` | Exact-term and keyword search |
| `semantic_search` | Meaning-based retrieval |
| `sql_query` | Inspect the SQLite database with read-only SQL |
| `read_file` | Read exact contents of local text files |
| `render_files` | Display local files directly in chat |
| `run_command` | Run whitelisted plugin-development commands |
| `build_plugin` | Create, edit, or delete sandbox plugins |
| `update_memory` | Update durable memory in `memory.md` |
| `web_search` | Search the public web when local data is not enough |
| `schedule_subagent` | Create and manage scheduled background subagent jobs |

## Frontends

Current frontends:

- `repl` - local terminal interface
- `telegram` - private bot frontend

Fresh configs enable both by default.

The Telegram frontend is especially useful because it makes the system feel less like a dev tool and more like a personal AI operator that can reach out to you when something matters.

## Project Structure

```text
Second Brain/
├── main.py                 # Cross-platform entry point
├── main.pyw                # Canonical startup script
├── controller.py           # Command/control surface used by frontends
├── context.py              # Shared runtime context for tools and tasks
├── plugin_discovery.py     # Built-in + sandbox discovery and hot registration
├── paths.py                # Root/data/sandbox path definitions
├── event_bus.py            # Internal pub/sub bus
├── event_channels.py       # Event channel registry
├── config_data.py          # Core config schema
├── config_manager.py       # Config + plugin-config persistence
│
├── frontend/
│   ├── repl/
│   │   └── repl.py         # Terminal frontend
│   ├── telegram/
│   │   ├── bot.py          # Telegram bot frontend
│   │   └── renderers.py    # Telegram media sending
│   └── shared/
│       ├── commands.py     # Shared slash command registry
│       ├── dispatch.py     # Shared input routing
│       └── formatters.py   # Shared formatting helpers
│
├── Stage_0/
│   ├── BaseService.py
│   └── services/
│       ├── llmService.py
│       ├── embedService.py
│       ├── ocrService.py
│       ├── whisperService.py
│       ├── webSearchService.py
│       ├── timekeeperService.py
│       └── driveService.py
│
├── Stage_1/
│   ├── registry.py
│   ├── ParseResult.py
│   └── parsers/
│
├── Stage_2/
│   ├── database.py
│   ├── watcher.py
│   ├── event_trigger.py
│   ├── orchestrator.py
│   ├── BaseTask.py
│   └── tasks/
│       ├── task_extract_text.py
│       ├── task_extract_container.py
│       ├── task_ocr_images.py
│       ├── task_chunk_text.py
│       ├── task_embed_text.py
│       ├── task_embed_images.py
│       ├── task_textualize_tabular.py
│       ├── task_lexical_index.py
│       └── task_run_subagent.py
│
├── Stage_3/
│   ├── agent.py
│   ├── BaseTool.py
│   ├── tool_registry.py
│   ├── system_prompt.py
│   ├── SearchResult.py
│   └── tools/
│       ├── tool_hybrid_search.py
│       ├── tool_lexical_search.py
│       ├── tool_semantic_search.py
│       ├── tool_sql_query.py
│       ├── tool_read_file.py
│       ├── tool_render_files.py
│       ├── tool_run_command.py
│       ├── tool_build_plugin.py
│       ├── tool_update_memory.py
│       ├── tool_web_search.py
│       └── tool_schedule_subagent.py
│
├── templates/
│   ├── tool_template.py
│   ├── task_template.py
│   └── service_template.py
│
└── DATA_DIR/
    ├── config.json
    ├── plugin_config.json
    ├── database.db
    ├── memory.md
    ├── sandbox_tools/
    ├── sandbox_tasks/
    └── sandbox_services/
```

## Setup

### Requirements

- Python 3.11+
- A configured LLM if you want agent features
- Windows if you want the built-in native OCR service
- Telegram bot token and allowed user ID if you want the Telegram frontend

### Install

```bash
git clone <repo-url>
cd "Second Brain"
pip install -r requirements.txt
```

Key dependencies include:

- `openai`
- `lmstudio`
- `sentence-transformers`
- `faster-whisper`
- `PyMuPDF`
- `python-docx`
- `python-pptx`
- `pandas`
- `watchdog`
- `python-telegram-bot`
- `croniter`
- `cron-descriptor`

### Configure

On first run, Second Brain creates its data directory automatically:

- Windows: `%LOCALAPPDATA%/Second Brain/`
- macOS: `~/Library/Application Support/Second Brain/`
- Linux: `${XDG_DATA_HOME:-~/.local/share}/Second Brain/`

The most important setting is `sync_directories`.

Minimal example:

```json
{
  "sync_directories": [
    "C:/Users/you/Documents",
    "D:/Projects"
  ],
  "enabled_frontends": ["repl", "telegram"],
  "autoload_services": ["web_search_provider", "timekeeper"],
  "telegram_bot_token": "",
  "telegram_allowed_user_id": 0,
  "llm_profiles": {
    "local": {
      "llm_model_name": "gpt-4.1-mini",
      "llm_endpoint": "http://127.0.0.1:1234/v1",
      "llm_api_key": "lm-studio",
      "llm_context_size": 128000,
      "llm_service_class": "OpenAILLM"
    }
  },
  "active_llm_profile": "local"
}
```

Notes:

- If you want tool-calling with LM Studio, point `OpenAILLM` at LM Studio's OpenAI-compatible endpoint.
- `LLMRouter` supports multiple named profiles and switching between them with `/model`.
- `timekeeper` and `web_search_provider` are good defaults to autoload because they power scheduling and web search.
- Brave Search and Brave Answers are optional and configured through plugin settings.

### Run

```bash
python main.py
```

On startup, the system:

1. loads config
2. creates sandbox directories if needed
3. initializes the database
4. discovers services, tasks, and tools
5. starts the orchestrator
6. starts the filesystem watcher
7. starts the event-trigger runner
8. launches the enabled frontends

## Commands

Available in the REPL and as slash commands in Telegram.

| Command | Description |
|---|---|
| `call <tool> {json}` | Call a tool directly |
| `cancel` | Interrupt the active agent |
| `config [key]` | Show config values |
| `configure <key> <value>` | Update config |
| `disable <tool>` | Disable a tool for agent use |
| `enable <tool>` | Enable a tool for agent use |
| `help` | Show all commands |
| `history [id]` | List or load saved conversations |
| `load <service>` | Load a service |
| `locations [tools|tasks|services]` | Inspect plugin-related file locations |
| `model ...` | Manage LLM profiles |
| `new` | Start a new conversation |
| `pause <task>` | Pause a task |
| `pipeline` | Show the path-driven dependency graph |
| `reload` | Hot-reload sandbox tasks and tools |
| `reset <task>` | Reset all entries for a path-driven task |
| `retry <task>` | Retry failed entries for a path-driven task |
| `retry all` | Retry failed entries across all path-driven tasks |
| `services` | List services and load state |
| `stats` | Show system-wide stats |
| `tasks` | List path-driven and event-driven tasks |
| `tools` | List registered tools |
| `trigger <task> [json]` | Manually fire an event-triggered task |
| `unload <service>` | Unload a service |
| `unpause <task>` | Resume a task |

## Scheduling and Calendar-Like Workflows

The easiest way to understand the new scheduler is this:

Second Brain can now operate proactively, not just reactively.

You can use `schedule_subagent` to create jobs that behave like:

- reminders
- recurring reviews
- daily briefings
- weekly planning prompts
- periodic research tasks
- "check this folder and notify me if something important changed"

That is why it is fair to describe the system as calendar-capable, even though it is not trying to be a traditional calendar UI. It can manage time-based work, recurring schedules, and proactive messaging in a way that is often more useful than a normal calendar event.

## Extending the System

Second Brain supports two extension modes:

- built-in plugins committed to the repo
- sandbox plugins that can be created live

### Sandbox Plugins

Sandbox plugins live in the mutable data directory and are safe from overwriting built-in code.

The agent can:

- create them
- edit them with exact search/replace patches
- delete them
- register them immediately

This gives you a very unusual loop:

1. ask the assistant for a new capability
2. let it author a plugin
3. approve the change
4. use the new capability immediately

### Built-In Plugins

If you want permanent source-controlled additions, add files in:

- `Stage_0/services/`
- `Stage_2/tasks/`
- `Stage_3/tools/`

Parsers live in `Stage_1/parsers/` and are registered by extension.

## Supported File Types

| Modality | Examples |
|---|---|
| Text | `.txt`, `.md`, `.py`, `.js`, `.ts`, `.html`, `.css`, `.json`, `.yaml`, `.toml`, `.xml`, `.pdf`, `.docx`, `.pptx`, `.gdoc` |
| Image | `.png`, `.jpg`, `.jpeg`, `.webp`, `.tiff`, `.bmp`, `.ico`, `.heic`, `.heif` |
| Audio | `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.aac`, `.wma` |
| Video | `.mp4`, `.mkv`, `.avi`, `.mov`, `.webm`, `.wmv`, `.flv` |
| Tabular | `.csv`, `.tsv`, `.xlsx`, `.xls`, `.parquet`, `.feather`, `.sqlite`, `.db` |
| Container | `.zip`, `.tar`, `.gz`, `.7z`, `.rar` |

## Design Philosophy

Second Brain is built around a few strong ideas:

- Local-first by default
- Structured data before vibes
- Retrieval and automation in the same runtime
- Agents should be able to act, not just answer
- Background intelligence matters
- Extensibility should be part of the product, not an afterthought

The goal is not to make a prettier chatbot.

The goal is to make a personal AI system that is actually operational.

## License

TBD
