<img width="1440" height="569" alt="highreslogotypecrop" src="https://github.com/user-attachments/assets/598ab57f-ed6b-491a-9cd6-142b93b09244" />

# Second Brain

Second Brain is an attempt to make a digital brain that approximates the real thing. It's part knowledge engine, part personal operator, and part programmable automation layer.

It continuously indexes your files, remembers durable context, searches the web when local knowledge is not enough, runs tools and shell commands, and can spin up background subagents that act on schedules or events. It lives in your terminal and Telegram, so your assistant is available everywhere.

Instead of being "just a chatbot," it turns your machine into a system that can observe, search, reason, and act. Point it at your world, give it tools, and it becomes a private AI layer for research, reminders, recurring work, and everyday operations.

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

It can be your personal assistant. It can be a file intelligence layer for your whole machine. It can be a reminder system. It can be a daily briefing engine. It can absolutely function like a personal AI calendar and operator for recurring work. It's a general-intelligence system for your computer. Some might even say it can be a new operating system.

## Core Capabilities

### 1. Index Your World

Point it at one or more directories and it will continuously watch them, parse supported files, and keep the database in sync as files appear, change, or disappear.

Built-in indexing pipeline includes:

- text extraction
- OCR for images
- archive/container extraction
- text chunking (for text embedding)
- text embeddings
- image embeddings
- lexical full-text indexing
- tabular textualization for turning spreadsheets into searchable data

The result is a live knowledge base over your local files, not a one-shot import. If you don't see something you need, just ask Second Brain to build it for you and it'll create a sandboxed task to do the job. The pipeline is robust and safe, correctly handling file/folder renames, debouncing/misfires, hidden folders, and other special cases. When a file is updated, added, or removed, all data within the dependency pipeline with that file key get updated.

### 2. Search Like a Real System

You can type /call to call any tool that the LLM agent can. This is useful for manual searches and situations where precision is imperative.

Second Brain ships with multiple retrieval tools:

- `lexical_search` for exact terms and keyword-heavy queries
- `semantic_search` for meaning-based retrieval over embeddings
- `hybrid_search` for fused lexical + semantic ranking
- `sql_query` for direct inspection of the underlying SQLite database

If an LLM can mess something up, it will mess something up. That's not to say that this happens often, but Second Brain was built with that in mind. It has safety and fallbacks, including manual controls when they are called for.

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

Scheduled subagents keep their own stored run history in SQL and can proactively push user-visible messages into chat. Subagent conversation histories are not immediately available to the main chat, but you can simply ask your agent to look at the most recent runs for context using (the tool for the job is sql_query). You can also leave messages for your subagent with /message.

### 4. Event-Driven Tasks

The system is no longer only file-driven, either.

Tasks can be triggered by events through the internal event bus. That opens the door to workflows like:

- scheduled events from the timekeeper service
- chained background runs
- approval workflows
- proactive notifications
- respond to emails immediately and maintain an inbox

Path-triggered tasks and event-triggered tasks share the same orchestration layer. One abstraction, two kinds of trigger. The event bus can be triggered from anywhere in the system, and it's possible to create a new `service` for things like text messages and webhooks.

### 5. Telegram As a First-Class Frontend

Second Brain now ships with two primary frontends (but it's possible to add more):

- Telegram bot
- Terminal REPL

Telegram is free and supports:

- slash commands
- autocomplete
- mobile-friendly responses
- file and media delivery
- interactive tool invocation
- approval prompts for sensitive actions
- proactive subagent push messages

This means your local system can act like a private mobile AI assistant—an *agent*. To set up Telegram, simply message @BotFather and get the API token, then message @userinfobot to find your ID and put both in config.json manually or with /configure.

The frontend code was designed to be modular and expandable. Other messaging platforms, like Discord, can be added easily using a coding agent.

### 6. Durable Memory

Second Brain includes agent memory through `memory.md` in the data directory.

The agent can update that memory intentionally with `update_memory`. It is meant for durable context such as:

- preferences
- standing instructions
- durable facts
- recurring context that should shape future behavior

It is not meant for one-off reminders, transient task state, or short-lived updates that only matter in the moment.

On top of that, conversation history is stored in SQLite and can be revisited later with read-only SQL. Nothing is thrown away unless you throw it away; simply ask the agent to look at your most recent conversations for context.

### 7. Web Search

Second Brain can search the public web through the built-in `web_search` tool.

It supports:

- Brave Search
- Brave Answers
- DuckDuckGo fallback when a Brave Search key is not configured (API keys are free from https://brave.com/search/api/ — the $5/month free tier is more than enough, and you can set the spending limit to $5 so you never spend a cent)

The agent is not trapped inside the local corpus. It can blend your private knowledge with current public information when appropriate.

### 8. Self-Extending Runtime

One of the most unusual parts of the project is that the agent can build new capabilities inside a sandbox at runtime.

If the current toolset cannot reasonably complete a task, the agent can use `build_plugin` to create, edit, or delete plugins:

- **services** can be loaded and unloaded to help carry out complex and repetitive tasks and tools
- **tasks** extract data tables from computer folders, and they can also be triggered at certain times of day (cron jobs)
- **tools** are used by LLMs to access the data resulting from tasks, as well as perform other agentic abilities (searching the web)

Plugins cover all basic use-cases for an agentic system. They can be designed and written by an LLM with no code written by the user. They are hot-registered immediately, no restart needed.

This means Second Brain is not a fixed assistant. It can inspect its own architecture, generate a focused extension, and use that new capability right away. Modular and extensible, Second Brain is a general intelligence system, which is much like an OS.

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

System responsibilities live in clear, top-level packages:

- `plugins/` for built-in tools, tasks, services, and discovery
- `pipeline/` for watching files, queueing work, and dispatching tasks
- `agent/` for prompt construction, tool execution, history healing, and subagent runtime
- `runtime/` for controller and task/tool context wiring
- `config/` and `events/` for shared system infrastructure
- `plugins/frontends/` for REPL, Telegram, and shared slash-command helpers

### Services + Parsers

Shared backends with explicit load and unload lifecycles, implemented as built-in plugins under `plugins/services/`.

Built-in services include:

- `llm` - default-LLM router; one service per entry in `llm_profiles` is also registered (keyed by model name) so multiple LLMs can be loaded concurrently
- `web_search_provider` - Brave Search / Brave Answers / DuckDuckGo fallback
- `timekeeper` - cron and one-time scheduling
- `ocr` - Windows OCR
- `whisper` - speech-to-text
- `text_embedder` - text embeddings
- `image_embedder` - image embeddings
- `google_drive` - Drive integration

The LLM layer is split into two separate config tables:

- `llm_profiles` — connection metadata for each model (endpoint, API key, context size, backend class). Each entry is registered as its own service keyed by the model name, and managed via `/llm`.
- `agent_profiles` — named agent definitions that reference an LLM by model name (or the literal `"default"` sentinel that follows whatever LLM is currently the default) and add optional scope: a prompt suffix plus tool whitelist/blacklist filters. Managed via `/agent`.

A fresh install ships with one agent profile (`default`) that uses the default LLM and has no restrictions, so you only ever touch `/agent` if you want more than one agent.

Within the same plugin family, extension-driven parsers normalize raw files into structured outputs. Parser helpers live under `plugins/services/helpers/`, and the parser service wires them into the rest of the runtime.

Supported modalities include:

- text
- image
- audio
- video
- tabular
- container

Parsers can also report `also_contains` hints, which allows multi-modal follow-up work. For example, a file can yield text and still announce that it contains images worth OCRing. The parser does not have to do everything in one pass.

### Task Pipeline + Orchestration

This is the always-on execution layer. The heart of the system.

Core pipeline code lives under `pipeline/`, while built-in tasks live under `plugins/tasks/`.

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

That split is what allows continuous file indexing and scheduled/proactive agents to coexist inside one architecture. For example, you can create embeddings for every new file and then run a clustering algorithm once a day using those embeddings. Since clusters change slightly whenever a new file is added, it makes more sense to run it once a day, than to recalculate on every file change. Use event-driven tasks on a timer for similar situations.

### Agent + Tools

This is the reasoning and action layer.

Core agent code lives under `agent/`, and built-in tools live under `plugins/tools/`.

The agent gets a dynamically rebuilt system prompt that includes:

- current date and time
- current tools
- current services
- current task pipeline state
- current file inventory
- current durable memory
- current sandbox plugins

The prompt pushes the assistant toward concise, grounded behavior. Among other things, it tells it to cite its sources and use the right tools for the job.

The system prompt is built dynamically for each message, ensuring that the model always has the freshest information available.

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
| `ask_subagent` | The main agent can delegate a complex task to a subagent to get a high-level answer |

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
├── paths.py                # Root/data/sandbox path definitions
│
├── agent/
│   ├── agent.py            # Main reasoning loop
│   ├── history_utils.py    # Conversation repair helpers
│   ├── subagent_runtime.py # Scheduled/subagent runtime support
│   ├── system_prompt.py    # Dynamic system prompt builder
│   └── tool_registry.py    # Tool registration + execution
│
├── config/
│   ├── config_data.py      # Core config schema
│   └── config_manager.py   # Config + plugin-config persistence
│
├── events/
│   ├── event_bus.py        # Internal pub/sub bus
│   └── event_channels.py   # Event channel registry
│
├── attachments/
│   ├── attachment.py       # Attachment + AttachmentBundle dataclasses
│   ├── cache.py            # Frontend upload persistence
│   ├── registry.py         # ext -> text-blurb parser registry
│   └── parsers/            # parser_text, parser_pdf, parser_audio, ...
│
├── pipeline/
│   ├── database.py         # SQLite state + task queue
│   ├── event_trigger.py    # Bus-driven task-run enqueue
│   ├── orchestrator.py     # Task registration + dispatch
│   └── watcher.py          # Filesystem watcher
│
├── plugins/
│   ├── BaseFrontend.py
│   ├── BaseService.py
│   ├── BaseTask.py
│   ├── BaseTool.py
│   ├── plugin_discovery.py # Built-in + sandbox discovery and hot registration
│   ├── frontends/
│   │   ├── repl_frontend.py
│   │   ├── telegram_frontend.py
│   │   └── helpers/
│   │
│   ├── services/
│   │   ├── llmService.py
│   │   ├── embedService.py
│   │   ├── ocrService.py
│   │   ├── whisperService.py
│   │   ├── webSearchService.py
│   │   ├── timekeeperService.py
│   │   ├── driveService.py
│   │   ├── parserService.py
│   │   └── helpers/
│   │
│   ├── tasks/
│   │   ├── task_extract_text.py
│   │   ├── task_extract_container.py
│   │   ├── task_ocr_images.py
│   │   ├── task_chunk_text.py
│   │   ├── task_embed_text.py
│   │   ├── task_embed_images.py
│   │   ├── task_textualize_tabular.py
│   │   ├── task_lexical_index.py
│   │   └── task_run_subagent.py
│   │
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
│       ├── tool_schedule_subagent.py
│       ├── tool_ask_subagent.py
│       └── helpers/
│
├── runtime/
│   ├── context.py          # Shared runtime context for tools and tasks
│   ├── controller.py       # Command/control surface used by frontends
│   └── token_stripper.py   # Model-token cleanup
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
git clone <https://github.com/henrydaum/second-brain>
cd "Second Brain"
pip install -r requirements.txt
```

Key dependencies include:

- `openai`
- `lmstudio`
- `sentence-transformers` (heaviest import; optional; only needed for local embedding)
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
    "gpt-4.1-mini": {
      "llm_endpoint": "",
      "llm_api_key": "sk-p...oMMA",
      "llm_context_size": 0,
      "llm_service_class": "OpenAILLM"
    }
  },
  "default_llm_profile": "gpt-4.1-mini"
}
```

You will need an LLM API key. A MiniMax API key for $10/month is more than sufficient for basic operations with their M2.7 model. If you are writing complicated plugins, I recommend a stronger model like Claude Opus, GPT 5.5, or higher. You can configure multiple llm profiles with /llm.

Notes:

- Tool calling isn't available with LM Studio.
- Setting the LLM context size to 0 is recommended for automatic compaction.
- Brave Search and Brave Answers are optional for web search and configured through plugin settings.
- LLM and agent configuration is manual; you can't ask the agent to do it for you. This is to prevent the LLM from leaking your API keys, and for the sake of transparency.
- Every entry in `llm_profiles` gets registered as its own service keyed by model name, plus an `llm` router service that resolves to whatever `default_llm_profile` points at. Load and unload individual LLMs with `/load <model_name>` and `/unload <model_name>`. The default LLM is loaded automatically.

### Agent Profiles for Safe Delegation

Agent profiles let you split work across specialized agents without giving every agent the same model, prompt, and tool surface.

This matters most when one of those agents is outward-facing. If you build a communication-focused agent that writes updates, emails, summaries, or outward messages, you may not want it to have every high-power tool in the system. A scoped profile lets you reduce what that agent can call. For narrower database or folder behavior, build a purpose-specific tool that applies the exact filter you want.

A practical setup looks like:

- a builder agent with coding and file-editing tools
- a researcher agent with broader search and database access
- a communicator agent with a much smaller toolset

In other words, you can treat Second Brain less like one monolithic assistant and more like a small team of specialists, each with the model, instructions, and tool access needed for its job. Having multiple agents is easy and optional.

Each agent profile carries:

- `llm` — a model name from `llm_profiles`, or the literal string `"default"` to follow whatever LLM is currently the default at runtime
- `prompt_suffix` — extra text appended to the system prompt for this agent
- `whitelist_or_blacklist_tools` — `"whitelist"` or `"blacklist"` for tool filtering
- `tools_list` — tool names for that filter; `blacklist` plus `[]` allows all tools

Tool dependencies are auto-expanded: if you allow `hybrid_search`, the underlying `lexical_search` and `semantic_search` are also callable automatically.

Switch the active profile with `/agent switch <name>`. The switch carries the conversation history forward but applies the new scope to the next turn. The `default` profile is permanent and cannot be removed.

In Telegram, `/agent` and `/llm` open profile-list menus. Tap a profile to see its attributes and `[Set active|default]`, `[Edit]`, and `[Remove]` actions.

### Run

```bash
python main.py
```

On startup, the system:

1. loads config
2. creates sandbox directories if needed
3. initializes the database
4. discovers services, tasks, and tools
5. starts the task orchestrator
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
| `help` | Show all commands |
| `history [id]` | List or load saved conversations |
| `llm [list\|add\|edit\|remove\|show\|default]` | Manage LLM connection profiles (model, endpoint, key, context, class) |
| `agent [list\|switch\|add\|edit\|remove\|show]` | Manage scoped agent profiles (LLM reference + tool allow-deny + prompt suffix) |
| `load <service>` | Load a service |
| `locations [tools\|tasks\|services]` | Inspect plugin-related file locations |
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
| `update` | `git pull` to get the latest version of Second Brain |
| `message` | Leave a note for a scheduled subagent |

## Scheduling and Calendar-Like Workflows

Second Brain can operate proactively, not just reactively, using a built-in cron scheduler (timekeeperService).

You can use `schedule_subagent` to create jobs that behave like:

- reminders
- recurring reviews
- daily briefings
- weekly planning prompts
- periodic research tasks
- inbox checks and message triage
- "check this folder and notify me if something important changed"

It is fair to describe the system as calendar-capable, even though it doesn't have a traditional calendar UI.

Subagents in cron routines (one-time or not) have three notification modes:
1. "all" - Notify every time the job runs
2. "off" - Never notify
3. "important" - The agent will notify you when something important happens (at the discretion of the agent, depending on the job)

Subagents notify the user through the `message` tool, which is added to their tool registry depending on the notification mode. If the agent doesn't use the tool in "all" mode, the last thing they generated is sent instead. Just as the subagent can message you through the `message` tool, you can also message them through the /message command. You can leave any number of messages for them this way, and they will read them the next time they wake up.

One last thing: subagents in cron routines remember what they have done in their previous runs. Keep this in mind when designing your prompts.

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

The software is extensible and self-expanding. Technically, just the llmService and build_plugin tool are strictly necessary because they can build everything else with some careful prompting.

### Built-In Plugins

If you want permanent source-controlled additions, move sandbox plugins from the DATA_DIR to:

- `plugins/services/`
- `plugins/tasks/`
- `plugins/tools/`

Parser helpers live in `plugins/services/helpers/` and are registered by extension through the parser service. You can put extra helper functions inside the /helpers folders inside of those main plugin folders.

## Supported File Types

| Modality | Examples |
|---|---|
| Text | `.txt`, `.md`, `.py`, `.js`, `.ts`, `.html`, `.css`, `.json`, `.yaml`, `.toml`, `.xml`, `.pdf`, `.docx`, `.pptx`, `.gdoc` |
| Image | `.png`, `.jpg`, `.jpeg`, `.webp`, `.tiff`, `.bmp`, `.ico`, `.heic`, `.heif` |
| Audio | `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.aac`, `.wma` |
| Video | `.mp4`, `.mkv`, `.avi`, `.mov`, `.webm`, `.wmv`, `.flv` |
| Tabular | `.csv`, `.tsv`, `.xlsx`, `.xls`, `.parquet`, `.feather`, `.sqlite`, `.db` |
| Container | `.zip`, `.tar`, `.gz`, `.7z`, `.rar` |

## Final Words

Most AI tools are built to be impressive in a demo and forgotten by the weekend. Second Brain is built for the opposite. It is meant to quietly keep running, watch the things you care about, and do real work while you are not looking.

A personal AI system should know your files, remember your context, respect your privacy, and grow with your use. It should be local, patient, and honest about what it does not know.

OpenClaw is great, but it's bloated. Second Brain is meant to be a lightweight and easy to learn alternative that doesn't try to do a million things out of the box.

Building your own runtime is its own kind of pleasure, because there is a sense of ownership and control. Furthermore, the patterns learned along the way (tools, tasks, and services) generalize to almost any serious agentic system somebody might want to build next.

One file at a time.

## License

TBD

---

An agent by Henry Daum
