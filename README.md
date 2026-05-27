<img width="1440" height="569" alt="highreslogotypecrop" src="https://github.com/user-attachments/assets/598ab57f-ed6b-491a-9cd6-142b93b09244" />

## Atlas Cloud

<div align="center">
  <a href="https://www.atlascloud.ai/?utm_source=github&utm_medium=link&utm_campaign=electerm">
    <img src="https://github.com/electerm/electerm-resource/blob/master/static/images/atlas-cloud.png?raw=true" alt="Atlas Cloud" width="200" />
  </a>
</div>

[Atlas Cloud](https://www.atlascloud.ai/?utm_source=github&utm_medium=link&utm_campaign=electerm) provides OpenAI-compatible AI APIs and model access for AI-powered workflows in electerm.

<div align="center">
  <img src="https://github.com/electerm/electerm-resource/raw/master/static/images/electerm.gif", alt="" />
</div>

# Second Brain

Second Brain is a local-first AI runtime for your machine.

It indexes your files, remembers durable context, searches the web, runs tools, schedules cron jobs, sends Telegram updates, and lets agents extend the system while it is running. It is not a fixed chatbot wrapped around a folder search. It is a programmable conversation runtime with memory, retrieval, automation, and live plugin authoring built in.

The most important architectural shift is the conversation layer. Second Brain now routes conversations through a robust state machine: participants take actions, turns move between actors, phases suspend and resume multi-step flows, and frontends submit actions instead of owning conversation logic. Commands and frontends are plugins too, so the system can grow new user interfaces and slash-command workflows the same way it grows tools, tasks, and services.

## What It Can Do

- Index documents, code, PDFs, slides, spreadsheets, archives, images, audio, and video.
- Search local files by keyword, semantics, or hybrid ranking.
- Answer from your own corpus with citations and exact file reads.
- Keep durable memory in `memory.md`.
- Store and resume conversation history in SQLite.
- Search the public web when local knowledge is not enough.
- Run path-driven indexing tasks and event-driven background jobs.
- Schedule one-time and recurring subagents through Timekeeper cron jobs.
- Push reminders, findings, daily briefs, and alerts into Telegram.
- Use REPL and Telegram frontends out of the box.
- Author, test, and live-load new tools, tasks, services, commands, and frontends.

The result is a private AI layer for your computer: part knowledge engine, part personal operator, part automation substrate.

## Core Architecture

Second Brain is built from a few durable pieces:

- `state_machine/` contains the pure conversation primitives: participants, turns, phases, actions, forms, approvals, and serializable phase frames.
- `runtime/` owns sessions, persistence, approvals, state-machine dispatch, agent turns, and the context passed into plugins.
- `plugins/` holds every extension family: tools, tasks, services, commands, and frontends.
- `pipeline/` watches files, manages the SQLite task queue, and runs path-driven and event-driven tasks.
- `agent/` builds the dynamic system prompt, manages the tool registry, and drives LLM tool calls.
- `events/` provides the pub/sub bus used by tasks, progress updates, notifications, and runtime signals.
- `config/` owns core settings plus plugin setting persistence.

The runtime is deliberately split this way so the state machine stays pure, frontends stay transport-specific, and plugins get a stable host API instead of reaching through the whole application.

## Conversation Runtime

The conversation runtime is the heart of the current system.

`ConversationRuntime.handle_action(...)` is the adapter-facing entry point. A frontend, scheduled job, or other driver submits a labeled action such as `send_text`, `send_attachment`, `call_command`, `submit_form_text`, `answer_approval`, or `cancel`. The runtime loads the session, refreshes command and tool specs, enters the state machine, persists the marker, and drives the agent turn when the action hands priority to the agent.

The state machine models conversations the same way a turn-based game models play:

- participants have permissions and identities
- one participant has turn priority
- actions are legal or illegal depending on phase
- forms and approvals suspend the current flow
- phase frames are serializable, so interrupted flows can be restored
- attachments are carried into the next agent turn with explicit lifecycle rules

This was inspired by the same turn/phase/action model used in a turn-based card game. The important point is not the game; it is the shape. A chatbot conversation, a slash-command form, a tool approval, and a scheduled agent handoff are all stateful turn flows.

Frontends do not own that flow. `BaseFrontend` turns transport input into runtime actions, then renders `RuntimeResult`, attachments, forms, approvals, buttons, errors, and progress events. This is why the REPL and Telegram can share command behavior, approval behavior, form behavior, cancellation, status updates, and session persistence without duplicating the core conversation logic.

## Plugin System

Everything user-extensible is a plugin family:

| Family | Built-in path | Sandbox path | Contract |
|---|---|---|---|
| Tools | `plugins/tools/` | `sandbox_tools/` | LLM-callable actions via `BaseTool` |
| Tasks | `plugins/tasks/` | `sandbox_tasks/` | Pipeline and event work via `BaseTask` |
| Services | `plugins/services/` | `sandbox_services/` | Shared backends via `BaseService` |
| Commands | `plugins/commands/` | `sandbox_commands/` | User slash commands via `BaseCommand` |
| Frontends | `plugins/frontends/` | `sandbox_frontends/` | User transports via `BaseFrontend` |

Built-in plugins are source-controlled. Sandbox plugins live in the Second Brain data directory and can be created while the app is running. Valid plugins are discovered on startup; when `plugin_watcher` is in `autoload_services`, adds, edits, and deletes are synced live.

The live authoring loop is:

1. Read the relevant template in `templates/`.
2. Read a similar built-in plugin.
3. Create or edit the plugin file with `edit_file`.
4. Let `plugin_watcher` auto-load the file when it is enabled, and call `test_plugin(plugin_path=...)` for purpose-built diagnostics.
5. If testing fails, fix the same file and call `test_plugin` again.
6. To remove it durably and from the live runtime, delete the sandbox file; `plugin_watcher` unloads it when enabled.

That loop matters. Second Brain can inspect its own templates, write a focused extension, diagnose it, and use it immediately. `test_plugin` gives the plugin-specific signal; its pytest section is broad regression context, not proof that the new plugin's behavior is complete. A new command is not a special case. A new frontend is not a rewrite. They are plugins with contracts.

## File Indexing And Retrieval

Point Second Brain at folders with `sync_directories` and it keeps a live SQLite knowledge base over those files.

The built-in pipeline includes:

- file watching and debounced change detection
- parser service dispatch by extension and modality
- text extraction
- OCR for images
- speech-to-text for audio and video
- archive/container extraction
- tabular textualization
- text chunking
- text embeddings
- image embeddings
- lexical full-text indexing
- dependency invalidation when upstream file outputs change

Search tools include:

| Tool | Purpose |
|---|---|
| `hybrid_search` | Best default local search over indexed files |
| `lexical_search` | Exact terms and keyword-heavy queries |
| `semantic_search` | Meaning-based retrieval over embeddings |
| `sql_query` | Read-only inspection of the SQLite database |
| `read_file` | Exact text reads from source, docs, templates, or sandbox plugins |
| `render_files` | Return local files to the frontend |

Supported modalities:

| Modality | Examples |
|---|---|
| Text | `.txt`, `.md`, `.py`, `.js`, `.ts`, `.html`, `.css`, `.json`, `.yaml`, `.toml`, `.xml`, `.pdf`, `.docx`, `.pptx`, `.gdoc` |
| Image | `.png`, `.jpg`, `.jpeg`, `.webp`, `.tiff`, `.bmp`, `.ico`, `.heic`, `.heif` |
| Audio | `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.aac`, `.wma` |
| Video | `.mp4`, `.mkv`, `.avi`, `.mov`, `.webm`, `.wmv`, `.flv` |
| Tabular | `.csv`, `.tsv`, `.xlsx`, `.xls`, `.parquet`, `.feather`, `.sqlite`, `.db` |
| Container | `.zip`, `.tar`, `.gz`, `.7z`, `.rar` |

## Events, Cron Jobs, And Subagents

Second Brain is proactive, not just reactive.

Path-driven tasks process files. Event-driven tasks respond to bus events. Timekeeper creates one-time and recurring jobs using cron expressions. Scheduled subagents can wake up, read their conversation history, run tools, and optionally send their final result back into chat.

This supports workflows like:

- reminders and follow-ups
- daily or weekly briefings
- recurring research checks
- inbox checks and message triage
- "watch this folder and tell me what changed"
- scheduled maintenance or database cleanup
- background subagents that remember prior runs

It is calendar-capable without being trapped in a traditional calendar UI. Jobs can run silently or notify the active frontend, and subagent conversations remain available through the conversation system.

## Frontends

Built-in frontends:

- `repl` - local terminal interface
- `telegram` - private mobile chat interface

Both are plugins under `plugins/frontends/`:

- `frontend_repl.py`
- `frontend_telegram.py`

`BaseFrontend` provides the shared runtime binding, command parsing path, form and approval submission, bus subscriptions, progress rendering hooks, session helpers, and `FrontendCapabilities` model. Each frontend implements only the transport-specific parts: receiving input, deriving a session key, rendering messages, sending attachments, showing buttons, and stopping cleanly.

Telegram is useful because the local runtime can reach you anywhere: approvals, proactive reminders, file delivery, scheduled-agent results, and mobile command menus all become part of the same conversation system.

Custom frontends are first-class plugins. A Discord bot, HTTP bridge, desktop shell, or narrow operational UI can be built as a sandbox frontend, tested with `test_plugin`, and live-loaded by `plugin_watcher` when enabled.

## Setup

### Requirements

- Python 3.11+
- An LLM profile for agent features
- Windows for the built-in native OCR service, or macOS for Apple Vision OCR
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

The most important setting is `sync_directories`: the folders Second Brain should watch and index. The attachment cache is included by default so files sent through frontends can enter the same pipeline.

Minimal shape:

```json
{
  "sync_directories": [
    "C:/Users/you/Documents",
    "C:/Users/you/AppData/Local/Second Brain/attachment_cache"
  ],
  "enabled_frontends": ["repl", "telegram"],
  "autoload_services": ["web_search_provider", "timekeeper", "llm", "parser", "plugin_watcher"],
  "telegram_bot_token": "",
  "telegram_allowed_user_id": 0,
  "llm_profiles": {
    "gpt-4.1-mini": {
      "llm_endpoint": "",
      "llm_api_key": "OPENAI_API_KEY",
      "llm_context_size": 0,
      "llm_service_class": "OpenAILLM"
    }
  },
  "default_llm_profile": "gpt-4.1-mini",
  "agent_profiles": {
    "default": {
      "llm": "default",
      "prompt_suffix": "",
      "whitelist_or_blacklist_tools": "blacklist",
      "tools_list": []
    }
  }
}
```

Notes:

- Configure LLM profiles with `/llm`.
- Configure agent profiles with `/agent`.
- Configure app and plugin settings with `/config`.
- Tool calling is not available with LM Studio.
- `llm_context_size: 0` lets automatic compaction manage context.
- Brave Search and Brave Answers are optional web-search providers configured through plugin settings.
- Each `llm_profiles` entry is registered as its own service, and the `llm` router follows `default_llm_profile`.

### Run

```bash
python main.py
```

Startup does the following:

1. Loads config and plugin config.
2. Creates data, attachment, and sandbox directories.
3. Initializes SQLite.
4. Discovers services, tasks, tools, commands, and frontends.
5. Starts the task orchestrator.
6. Starts the filesystem watcher.
7. Starts the event-trigger runner.
8. Launches enabled frontends.

## Commands And Tools

Commands are user-facing plugins. They are available in the REPL and Telegram as slash commands, and they can collect forms through the state machine.

Built-in commands include:

| Command | Purpose |
|---|---|
| `/agent` | Select, switch, edit, or remove agent profiles |
| `/cancel` | Cancel the current interaction |
| `/clear` | Clear the current conversation |
| `/commands` | List available commands |
| `/config` | Select and edit config settings |
| `/conversations` | Browse, switch, and manage conversations |
| `/frontends` | Enable or disable frontend plugins |
| `/llm` | Select, edit, set default, or remove LLM profiles |
| `/locations` | Show project and plugin directories |
| `/new` | Start a conversation with default settings |
| `/schedule` | Manage Timekeeper scheduled jobs |
| `/services` | Select and load or unload services |
| `/tasks` | Pause, resume, reset, retry, or trigger tasks |
| `/tools` | Select and call tools |
| `/update` | Pull latest changes from the repo |

Built-in tools include:

| Tool | Purpose |
|---|---|
| `edit_file` | Create, overwrite, replace, append to, or delete UTF-8 text files |
| `hybrid_search` | Search local files with fused lexical and semantic ranking |
| `lexical_search` | Search local files by exact terms and keywords |
| `read_file` | Read exact text from files |
| `render_files` | Send local files back through the frontend |
| `run_command` | Run scoped terminal commands, with approval for broad actions |
| `schedule_subagent` | List, add, edit, and remove scheduled background agents |
| `semantic_search` | Search local files by embedding similarity |
| `sql_query` | Query SQLite read-only |
| `test_plugin` | Diagnose a plugin source file and summarize broad regression tests |
| `update_memory` | Update durable memory |
| `web_search` | Search the public web |

## Project Layout

```text
Second Brain/
├── main.py                 # Console entry point
├── main.pyw                # Windowed startup script
├── paths.py                # Root, data, attachment, and sandbox paths
│
├── state_machine/
│   ├── conversation.py     # Participants, callable specs, forms, phases
│   ├── action_map.py       # Action constructors and legal action routing
│   ├── action.py           # State-machine action implementations
│   ├── forms.py            # Multi-step form handling
│   └── approval.py         # Runtime approval request shape
│
├── runtime/
│   ├── conversation_runtime.py # Session gateway for frontend/automation actions
│   ├── conversation_loop.py    # Agent-turn driver
│   ├── dispatch.py             # Runtime action helpers
│   ├── persistence.py          # Conversation/session persistence
│   ├── runtime_approvals.py    # State-machine approval bridge
│   ├── runtime_config.py       # Active profile, tools, commands, prompt
│   └── session.py              # RuntimeSession and RuntimeResult
│
├── plugins/
│   ├── BaseCommand.py
│   ├── BaseFrontend.py
│   ├── BaseService.py
│   ├── BaseTask.py
│   ├── BaseTool.py
│   ├── plugin_discovery.py
│   ├── commands/
│   ├── frontends/
│   ├── services/
│   ├── tasks/
│   └── tools/
│
├── pipeline/
│   ├── database.py
│   ├── event_trigger.py
│   ├── orchestrator.py
│   └── watcher.py
│
├── agent/
│   ├── agent.py
│   ├── system_prompt.py
│   └── tool_registry.py
│
├── attachments/
├── config/
├── events/
├── templates/
│   ├── command_template.py
│   ├── frontend_template.py
│   ├── service_template.py
│   ├── task_template.py
│   └── tool_template.py
└── DATA_DIR/
    ├── config.json
    ├── plugin_config.json
    ├── database.db
    ├── memory.md
    ├── attachment_cache/
    ├── sandbox_tools/
    ├── sandbox_tasks/
    ├── sandbox_services/
    ├── sandbox_commands/
    └── sandbox_frontends/
```

## Extension Authoring Guide

Use the templates as the source of truth:

- `templates/tool_template.py`
- `templates/task_template.py`
- `templates/service_template.py`
- `templates/command_template.py`
- `templates/frontend_template.py`

Authoring rules:

- Tools expose LLM-callable capabilities and return `ToolResult`.
- Tasks are pipeline/event workers and should be idempotent where possible.
- Services own reusable backends with explicit load/unload lifecycle.
- Commands are user-facing conversation actions and can define `FormStep` flows.
- Frontends are transports; they submit runtime actions and render runtime output.
- Plugins can declare `config_settings`, which appear in config views and are stored in `plugin_config.json`.
- Sandbox plugins must follow naming conventions: `tool_*.py`, `task_*.py`, `service_*.py`, `command_*.py`, and `frontend_*.py`.

For source-controlled additions, move stable sandbox plugins into the matching built-in plugin directory. For live experimentation, keep them in the data directory, call `test_plugin`, and let `plugin_watcher` load them when it is enabled.

## Philosophy

Second Brain is built around a simple bet: an AI system should be fully customizable and easy to adapt and change, like LEGOs.

Most software products are static, whereas Second Brain can write its own code and extend itself. The Second Brain core code is so versatile and strong that it can turn into almost any shape. 

Although Second Brain is still a ways off from being like a human brain, it can still do quite a lot, and I think it is worthy of the name. The goal has always been to create something intelligent, adaptable, and useful. Those are the three things that have driven this from the start.

## License

TBD

---

An agent by Henry Daum
