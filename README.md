# Second Brain

Second Brain is a local-first personal data engine.

It watches your folders, parses what it finds, builds a structured index in SQLite, and gives an LLM the tools to search, reason, act, schedule work, remember things, and extend the system itself. 

Why Second Brain:
1. When the files you want to ask about are too large or too numerous for Claude, OpenAI, or Gemini; or AI companies don't have a feature you want but you think you could build yourself.
2. When OpenClaw and Hermes are too bloated for your use-case.
3. You want an AI that will automatically sync to your Google Drive files.
4. You want all of these things, and you want to have them all from your phone.

Second Brain is a private, always-on AI runtime for your data:

- a searchable file index
- a natural-language analyst
- a background scheduler
- an event-driven automation system
- a Telegram assistant
- a memory-backed agent
- a plugin platform that can author new tools, tasks, and services at runtime

**Q:** How do you see this evolving into something truly revolutionary for how we interact with information?

**A:** The deep filesystem access is very different from how other agents interact with data. Other agents can create, update, and delete individual files but they have no built-in way of aggregating data from entire hard drives. This capability is built into Second Brain, plus it syncs automatically. You can decide exactly how to process all of your files.

## Why It Matters

It can:

- index your documents, code, PDFs, slides, spreadsheets, archives, images, audio, and video
- answer questions grounded in your own files with citations
- search by keyword, semantics, or a hybrid of both
- remember durable facts and preferences across sessions
- search the web when local knowledge is not enough
- run proactive subagents on schedules
- fire tasks from events, not just file changes
- push reminders, findings, daily briefs, and alerts into Telegram
- proactively send emails and text messages
- build and load new tools, tasks, and services without a restart

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

AI can be inaccurate, so it was important to make sure Second Brain has manual search available. Every tool the AI can use, so can you. Double check your answers.

Second Brain ships with multiple retrieval modes:

- `lexical_search` for exact terms and keyword-heavy queries
- `semantic_search` for meaning-based retrieval over embeddings
- `hybrid_search` for fused lexical + semantic ranking
- `sql_query` for direct inspection of the underlying SQLite database

You can ask normal-language questions. You can also inspect the system with precision when you want to. Both are first-class.

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

The system is no longer only file-driven, either.

Tasks can be triggered by events through the internal event bus. That opens the door to workflows like:

- scheduled event emissions from the timekeeper service
- chained background runs
- approval workflows
- proactive notifications
- future external integrations that emit events into the system
- respond to emails immediately

Path-triggered tasks and event-triggered tasks share the same orchestration model. One abstraction, two kinds of trigger. That is what makes the platform genuinely general rather than a bolt-on scheduler.

### 5. Telegram As a First-Class Frontend

Second Brain now ships with two primary frontends (but it's possible to add more):

- Telegram bot
- Terminal REPL

Telegram supports:

- slash commands
- autocomplete
- mobile-friendly responses
- file and media delivery
- interactive tool invocation
- approval prompts for sensitive actions
- proactive subagent push messages

This means your local system can act like a private mobile AI assistant without becoming a cloud SaaS product. Your data never leaves your machine. The assistant reaches you, not the other way around.

The frontend code was designed to be modular and expandable. Other messaging platforms, like Discord, can be added easily using a coding agent.

### 6. Durable Memory

Second Brain includes agent memory through `memory.md` in the data directory.

The agent can update that memory intentionally with `update_memory`. It is meant for durable context such as:

- preferences
- standing instructions
- durable facts
- recurring context that should shape future behavior

It is not meant for one-off reminders, transient task state, or short-lived updates that only matter in the moment.

On top of that, conversation history is stored in SQLite and can be revisited later with read-only SQL. Nothing is thrown away unless you throw it away.

### 7. Web Search

Second Brain can search the public web through the built-in `web_search` tool.

It supports:

- Brave Search
- Brave Answers
- DuckDuckGo fallback when a Brave Search key is not configured

The agent is not trapped inside the local corpus. It can blend your private knowledge with current public information when appropriate. Local-first does not mean local-only.

### 8. Self-Extending Runtime

One of the most unusual parts of the project is that the agent can build new capabilities inside a sandbox at runtime.

If the current toolset cannot reasonably complete a task, the agent can use `build_plugin` to create, edit, or delete:

- **services** can be loaded and unloaded to help carry out complex and repetitive tasks and tools
- **tasks** create tables of statistical data from the information found in computer folders, and they can also be triggered at certain times of day (cron jobs)
- **tools** are used by LLMs to access the data resulting from tasks, as well as perform other agentic abilities like searching the web

Plugins cover all basic use-cases for an agentic system. They can be made and designed by the LLM with no code written by the user. They are hot-registered immediately. No restart needed.

This means Second Brain is not a fixed assistant. It can inspect its own architecture, generate a focused extension, and use that new capability right away. The system grows in the direction you actually use it in.

## What You Can Use It For

- Personal search engine for your entire document corpus
- Codebase analyst over local repositories
- Research assistant that combines file search with live web search
- Daily briefings pushed to Telegram
- Reminder and recurring-task system powered by scheduled subagents
- AI calendar-like workflows using one-time and recurring jobs
- Archive and media intelligence across PDFs, images, video, audio, and spreadsheets
- Private long-term assistant with memory and conversation history
- Design a personal assistant to write emails and send text messages for you
- Agentic automation that can build its own plugins when the right tool does not exist yet

## Architecture

The system is organized into four stages, with an event bus connecting long-lived components. Each stage does one thing well and hands its output to the next.

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

Parsers can also report `also_contains` hints, which allows multi-modal follow-up work. For example, a file can yield text and still announce that it contains images worth OCRing. The parser does not have to do everything in one pass.

### Stage 2: Task Pipeline + Orchestration

This is the always-on execution layer. The heart of the system.

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

That split is what allows continuous file indexing and scheduled/proactive agents to coexist inside one architecture.

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

The prompt pushes the assistant toward concise, grounded behavior: inspect the system first, prefer local evidence, cite the files, tables, or tool results it relied on. Confidence is earned by checking.

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
| `ask_subagent` | The main agent can delegate a complex task to a subagent to get a top-level answer |

## Frontends

Current frontends:

- `repl` - local terminal interface
- `telegram` - private bot frontend

Config enables both by default.

The Telegram frontend is especially useful because it makes the system feel less like a dev tool and more like a personal AI operator that can reach out to you when something matters. Works anywhere on a phone.

The frontend code is modular enough such that it is possible to create a new frontend (Discord, etc.) with minimal issues.

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
│   │   ├── telegram.py     # Telegram bot frontend
│   │   └── renderers.py    # Telegram media sending
│   └── shared/
│       ├── commands.py     # Shared slash command registry
│       ├── dispatch.py     # Shared input routing
│       └── formatters.py   # Shared formatting helpers
│
├── Stage_1/
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
│       └── tool_ask_subagent.py
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
- `sentence-transformers` (optional—only needed for local embedding)
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

The most important setting is `sync_directories`. Fill it with the folders you want to know everything about. Use /configure to set your sync_directory.

Minimal example:

```json
{
  "sync_directories": [
    "C:/Users/you/Documents",
    "C:/Users/you/AppData/Local/Second Brain/attachment_cache"
  ],
  "enabled_frontends": ["repl", "telegram"],
  "autoload_services": ["web_search_provider", "timekeeper", "llm"],
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

You will need an LLM API key. A MiniMax API key for $10/month is more than sufficient for basic operations with their M2.7 model. If you are writing complicated plugins, I recommend a stronger model like Claude Opus, GPT 5.4, or higher. Use /model to build a new model profile with your key.

Notes:

- Tool calling not available with LM Studio.
- `LLMRouter` supports multiple named profiles and switching between them with `/model`.
- `timekeeper` and `web_search_provider` are good defaults to autoload because they power scheduling and web search; `llm` is needed for basic functioning.
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
| `refresh` | Refresh the agent in case of breakage |
| `reload` | Hot-reload sandbox tasks and tools |
| `reset <task>` | Reset all entries for a path-driven task |
| `restart` | Restart the whole app |
| `retry <task>` | Retry failed entries for a path-driven task |
| `retry all` | Retry failed entries across all path-driven tasks |
| `services` | List services and load state |
| `tasks` | List path-driven and event-driven tasks |
| `tools` | List registered tools |
| `trigger <task> [json]` | Manually fire an event-triggered task with an optional JSON payload |
| `unload <service>` | Unload a service |
| `unpause <task>` | Resume a task |

## Scheduling and Calendar-Like Workflows

The easiest way to understand the new scheduler is this:

Second Brain can operate proactively, not just reactively.

You can use `schedule_subagent` to create jobs that behave like:

- reminders
- recurring reviews
- daily briefings
- weekly planning prompts
- periodic research tasks
- generate leads via email or text
- "check this folder and notify me if something important changed"

It is fair to describe the system as calendar-capable, even though it doesn't have a traditional calendar UI.

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

1. Ask the assistant for a new capability.
2. Let it author a plugin.
3. Approve the change.
4. Potentially make another edit in the core code.
5. Use the new capability immediately.

Software that can extend itself in response to use is a different kind of software.

### Built-In Plugins

If you want permanent source-controlled additions, add files in:

- `Stage_1/services/`
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
- Murphy's Law: if an LLM can mess something up, it will mess something up; fallbacks and safety wherever possible.
- Retrieval and automation in the same runtime
- Agents should be able to act, not just answer
- Background intelligence matters
- Extensibility should be part of the product, not an afterthought

The goal is not to make a prettier chatbot.

The goal is to make a personal AI system that is actually operational.

## Final Words

Most AI tools are built to be impressive in a demo and forgotten by the weekend. Second Brain is built for the opposite. It is meant to quietly keep running, watch the things you care about, and do real work while you are not looking.

A personal AI system should know your files, remember your context, respect your privacy, and grow with your use. It should be local, patient, and honest about what it does not know.

OpenClaw is great, but it's bloated. Second Brain is meant to be a lightweight and easy to learn alternative that doesn't try to do a million things out of the box.

Building your own runtime is its own kind of pleasure, because there is a sense of ownership and control. Furthermore, the patterns learned along the way—retrieval, orchestration, memory, self-extension—generalize to almost any serious agentic system somebody might want to build next.

One file at a time.

## License

TBD
