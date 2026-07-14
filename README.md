<img width="1440" height="569" alt="highreslogotypecrop" src="https://github.com/user-attachments/assets/598ab57f-ed6b-491a-9cd6-142b93b09244" />

# Sponsor
<div align="center">
  <img src="https://github.com/user-attachments/assets/9e7ff971-8159-4081-b8bc-9b9ff5edd4ff#gh-light-mode-only" width="500" alt="Atlas Cloud Logo">
  <img src="https://github.com/user-attachments/assets/8497513e-09a4-4151-8b8d-ed8be782a389#gh-dark-mode-only" width="500" alt="Atlas Cloud Logo">
</div>

---

[Atlas Cloud](https://www.atlascloud.ai/?utm_source=github&utm_medium=link&utm_campaign=second-brain) is a full-modal AI inference platform that gives developers a single AI API to access video generation, image generation, and LLM APIs. Instead of managing multiple vendor integrations, you connect once and get unified access to 300+ curated models across all modalities.
Check out Atlas Cloud's new coding plan promotion for more budget-friendly API access: [https://www.atlascloud.ai/console/coding-plan](https://www.atlascloud.ai/console/coding-plan)

# Second Brain

<mark>*For an example of what Second Brain can do, visit https://second-brain.art! It's an interactive art exhibition.*</mark>

<mark>*NEW!* Second Brain now has token streaming. Telegram now has silky smooth token streaming plus rich text.</mark>

Second Brain is a local-first AI runtime for your machine, built as a **microkernel**.

The kernel is deliberately small: it boots, runs the agent turn, persists conversations in SQLite, loads and unloads plugins, keeps the lightweight Timekeeper event clock running, and gets out of the way. Everything else — file indexing and retrieval, web search, scheduling workflows, Telegram, durable memory, file-editing and shell tools, heavy file parsers — arrives as **packages** you install from the store. It is a programmable conversation runtime that you (and agents) extend while it is running. The kernel is like a brain, while plugins are like the body. The brain and body have a symbiotic relationship — it's the same way with plugins and the kernel.

A fresh install starts almost empty. Run `/setup` and install the `starter` bundle to get a working assistant in one step — see [The Kernel And The Package Store](#the-kernel-and-the-package-store) below.

Second Brain is designed to merge an LLM cleanly into a wide variety of workflows, and chat conversations are the basis of this. Second Brain routes conversations through a robust state machine: participants take actions, turns move between actors, multi-step flows follow resumable phases, and frontends submit actions instead of owning conversation logic. Everything eventually runs through the conversation state machine; it's the bedrock of the system. Think of it like the spinal cord, connecting the brain (LLM) to the body (plugins).

## What It Can Do

With the right packages installed (the `full` bundle covers most of the list below), Second Brain can:

- Index documents, code, PDFs, slides, spreadsheets, archives, images, audio, and video.
- Search local files by keyword, semantics, or hybrid ranking.
- Answer from your own corpus with citations and exact file reads.
- Develop a robust memory library.
- Store and resume conversation history in SQLite.
- Search the public web.
- Run path-driven indexing tasks and event-driven background jobs to develop a robust database.
- Schedule one-time and recurring subagents through Timekeeper cron jobs.
- Use agentskills.io compatible skills.
- Push reminders, findings, daily briefs, and alerts into Telegram.
- Author, test, and live-load new tools, tasks, services, commands, and frontends. The possibilities are endless.

This might seem like a lot. However, Second Brain can only do what you tell it to do. It will never edit your files or share information unless you explicitly enable it to. Be careful with which plugins you enable. Although all of the built-in ones are highly tested, they still carry risk — and some more than others.

## The Kernel And The Package Store

Second Brain ships as a microkernel plus a package store.

**The kernel** is what lives in this repository's main tree. It is almost pure Python and boots *fast*. It holds the runtime, runs the conversation state machine and agent turn, persists conversations, manages config, and discovers and loads plugins. It ships only the plugins it cannot run without: the LLM router, the compactor (for LLM context size management), the Parser service (with lightweight UTF-8 text parsing), Timekeeper (the event clock), the plugin watcher (live install and reload), the REPL frontend, and a small set of REPL admin commands. There are **no built-in tools or tasks** — a fresh kernel can hold a conversation, but it cannot search your files or edit code until you install packages. *Even the LLM backends are plugins! LLMs involve heavy and unstable dependencies, so to ensure kernel stability it was best to make them plugins.*

**The store** is a parallel branch (`store`) that mirrors what a fully loaded install looks like: every optional tool, task, service, command, frontend, and helper is there, plus named *bundles* that group them. You browse and install from it with `/packages`, and the kernel copies the files into your data directory and live-loads them with the Plugin Watcher.

### Getting started

A fresh install has no LLM backend and no frontend beyond the REPL. The fastest path is the onboarding wizard:

```
/setup
```

It installs a bundle with basic plugins, configures an LLM profile, and optionally sets up Telegram. If you would rather drive it by hand, install the starter bundle directly:

```
/packages install bundle_starter
```

The **starter** bundle is the recommended first install: an LLM backend (LiteLLM, which reaches most providers), the Telegram frontend, file read/edit, SQL and shell tools, ask-user-question, plugin authoring, and durable-memory plus auto-title tasks. Note: you cannot chat with an LLM in Second Brain until you install an LLM backend (LiteLLM recommended). For everything else — all file parsers, OCR, audio/video transcription, and the full indexing and search pipeline — install the larger **full** bundle:

```
/packages install bundle_full
```

Browse and manage packages anytime:

```
/packages                      # interactive: browse / install / uninstall
/packages available tools      # list installable tools (or tasks/services/commands/frontends/bundles)
/packages installed            # what you currently have
/packages install <stem>       # install one file by stem, e.g. tool_web_search or parse_pdf
/packages uninstall <stem>     # remove it, plus dependencies nothing else still needs
```

Install resolves each file's declared dependencies — other store files and pip packages — and copies them in. Uninstall removes only files and pip packages nothing else still needs, and never touches kernel requirements. You don't need to worry about managing helper files or Python packages (with one rare exception — OCR — which requires a manual pip install since it's platform/OS dependent).

The /packages command automatically maintains a clean separation between the kernel and plugins. All installed plugins and helpers will be within the installed_plugins folder of the DATA_DIR (data directory, see below). You don't have to use the /packages command to install or remove plugins. You can also simply drag and drop plugins and their helpers into this folder. It'll be automatically picked up by the Plugin Watcher service. However, if you do it this way you will have to pip install their Python dependencies as well, if you haven't got them already.

### Contributing to the store

The store is just a git branch, so adding a plugin is a pull request. Author and test your plugin as a sandbox plugin (see [Plugin System](#plugin-system) and the [Extension Authoring Guide](#extension-authoring-guide)), then open a pull request against the `store` branch that adds your `tool_*.py` / `task_*.py` / `service_*.py` / `command_*.py` / `frontend_*.py` (and any `helpers/`) under the matching family directory. Declare dependencies with the `dependencies_files` and `dependencies_pip` fields so the package manager can resolve them, and to group several files under one install, add a `bundles/<name>.json` manifest listing the store-relative files.

You can also simply send me an email at henrydaum8609@gmail.com with what you want to make :-)

## Core Architecture

Second Brain is built from a few durable pieces:

- `state_machine/` contains the pure conversation primitives: participants, turns, phases, actions, forms, approvals, and serializable phase frames.
- `runtime/` owns sessions, persistence, approvals, state-machine dispatch, agent turns, and the context passed into plugins.
- `plugins/` holds every extension family: tools, tasks, services, commands, and frontends.
- `pipeline/` watches files, manages the SQLite task queue, and runs path-driven and event-driven tasks.
- `agent/` builds the dynamic system prompt, manages the tool registry, and drives LLM tool calls.
- `events/` provides the pub/sub bus used by tasks, progress updates, notifications, and runtime signals.
- `config/` owns core settings plus plugin setting persistence.

**It's a highly complex piece of machinery about 13,000 lines of code long.** But what's important is that your Second Brain agent can understand it for you! You do not need to understand how all of this works in order to have a Second Brain agent. Frankly, even I have trouble conceptualizing all of it. Simply ask Second Brain a question about its own code, and it'll use its available tools to dig in and find you an answer (of course, you'll need to have at least the read_file tool installed).

## Conversation Runtime

The conversation runtime is the heart of the current system. It's like the spinal cord connecting the plugins/body to the LLM brain.

`ConversationRuntime.handle_action(...)` is the adapter-facing entry point. A frontend, scheduled job, or other driver submits a labeled action such as `send_text`, `send_attachment`, `call_command`, `submit_form_text`, `answer_approval`, or `cancel`. The runtime loads the session, refreshes command and tool specs, enters the state machine, persists the marker, and drives the agent turn when the action hands priority to the agent.

The state machine models conversations the same way a turn-based game does (think Magic: The Gathering):

- participants have permissions and identities
- one participant has turn priority
- actions are legal or illegal depending on phase
- forms and approvals suspend the current flow
- phase frames are serializable, so interrupted flows can be restored on crash
- attachments are carried into the next agent turn with explicit lifecycle rules

Frontends do not own that flow. `BaseFrontend` turns transport input into runtime actions, then renders `RuntimeResult`, attachments, forms, approvals, buttons, errors, and progress events. This is why the REPL and Telegram can share command behavior, approval behavior, form behavior, cancellation, status updates, and session persistence without duplicating the core conversation logic.

## Plugin System

Everything user-extensible has its own plugin family:

| Family | Built-in path | Sandbox path | Contract |
|---|---|---|---|
| Tools | `plugins/tools/` | `sandbox_tools/` | LLM-callable actions via `BaseTool` |
| Tasks | `plugins/tasks/` | `sandbox_tasks/` | Pipeline and event work via `BaseTask` |
| Services | `plugins/services/` | `sandbox_services/` | Shared backends via `BaseService` |
| Commands | `plugins/commands/` | `sandbox_commands/` | User slash commands via `BaseCommand` |
| Frontends | `plugins/frontends/` | `sandbox_frontends/` | User transports via `BaseFrontend` |

Built-in plugins are source-controlled. Sandbox plugins live in the Second Brain data directory and can be created while the app is running. Valid plugins are discovered on startup; and adds, edits, and deletes are synced live when `plugin_watcher` is loaded (which is always).

The agent can create new plugins on-the-fly. When you ask Second Brain to make a new plugin (and test_plugin is installed), it'll follow these instructions:

1. Read the relevant template in `templates/`.
2. Read a similar built-in plugin.
3. Create or edit the plugin file with `edit_file`.
4. Let `plugin_watcher` auto-load the file when it is enabled, and call `test_plugin(plugin_path=...)` for purpose-built diagnostics.
5. If testing fails, fix the same file and call `test_plugin` again. Repeat until it is fixed and the plugin loads.
6. To remove it durably and from the live runtime, delete the sandbox file; `plugin_watcher` unloads it when enabled.

These instructions are found within the test_plugin tool, inside the def agent_prompt_for method. Yes: plugins can declare their own system prompt text. If the plugin isn't loaded, then this stuff won't take up precious context. This agent-facing text can also be written as static text: agent_prompt = "blah blah blah"; agent_prompt_for is for prompts that update based on the current information. The system prompt is recalculated with every step of the conversation, making it always completely up-to-date. To maintain prompt caching (and reduce costs $) the most volatile system prompt text is inserted at the bottom of the conversation, with stable and semi-stable at the start.

In other words, the system prompt has been fully engineered.

## Security

Rogue plugins are the primary security issue for Second Brain. If a plugin gets an infection, it's important to prevent it from taking down the whole system. Second Brain monitors memory usage through the Plugin Supervisor, which automatically quarantines plugins that exceed the threshold. Also: tools, tasks, and service loads are given a timeout; if they can't get the job done within that timeframe, it gets cancelled. This fixes freezes. There's also a system fallback for crashes. If Second Brain crashes, then it will restart itself automatically. When this happens, all your conversations will be reloaded exactly where they were, even if you were in the middle of a command. Second Brain has a literal heartbeat which monitors for system-wide freezes. If there's no heartbeat for a long time, the system initiates a restart, like just discussed.

The bad news: rogue plugins can still kill Second Brain. All plugins run in-process, which means that there are no subprocesses. If a plugin calls `os._exit()`, then it will take down the whole process. Like malware, a plugin can theoretically prompt the agent to mine for private information and share it. Although no plugins like this exist in the store (they have been vetted by me personally), you should make sure you trust plugins before adding them to your runtime. And when writing new plugins for Second Brain, you should use an intelligent model that won't mess things up. Consider using Codex or Claude Code for in-depth coding exercises.

## File Indexing And Retrieval

Indexing and retrieval are store capabilities — install the `full` bundle (or the indexing/search and parser bundles) to enable them. Once installed, point Second Brain at folders with `sync_directories` and it keeps a live SQLite knowledge base over those files. The kernel ships the pipeline basics (file watcher, task queue, orchestrator DAG); these packages add the processing stages that run on it. You can set a `sync_directory` in the settings to create a database.

The full pipeline includes:

- file watching and debounced change detection
- parser service dispatch by extension and modality
- text extraction
- OCR for images
- speech-to-text for audio and video (`service_whisper` + `parse_voice` enable Telegram speech-to-text)
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

These tools give Second Brain the ability to find a needle in a haystack.

Supported modalities:

| Modality | Examples |
|---|---|
| Text | `.txt`, `.md`, `.py`, `.js`, `.ts`, `.html`, `.css`, `.json`, `.yaml`, `.toml`, `.xml`, `.pdf`, `.docx`, `.pptx`, `.gdoc` |
| Image | `.png`, `.jpg`, `.jpeg`, `.webp`, `.tiff`, `.bmp`, `.ico`, `.heic`, `.heif` |
| Audio | `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.aac`, `.wma` |
| Video | `.mp4`, `.mkv`, `.avi`, `.mov`, `.webm`, `.wmv`, `.flv` |
| Tabular | `.csv`, `.tsv`, `.xlsx`, `.xls`, `.parquet`, `.feather`, `.sqlite`, `.db` |
| Container | `.zip`, `.tar`, `.gz`, `.7z`, `.rar` |

To parse all of these, you'll need to install the full Parser bundle from the package store:

`/packages install bundle_all_parsers`

This will also give you the ability to process attachments with these file extensions.

## Events, Cron Jobs, And Subagents

Second Brain is proactive, not just reactive — once the scheduling package is installed:

`/packages install bundle_scheduling`

Path-driven tasks process files. Event-driven tasks respond to bus events. Timekeeper is the kernel service that creates one-time and recurring event emissions using cron expressions; the `bundle_scheduling` package adds the `/schedule` command, scheduled-subagent task, and scheduling tool on top. Scheduled subagents can wake up, read their conversation history, run tools, and optionally send their final result back into chat, depending on their notification mode.

This supports workflows like:

- reminders and follow-ups
- daily or weekly briefings
- recurring research checks
- inbox checks and message triage
- "watch this folder and tell me what changed"
- scheduled maintenance or database cleanup
- background subagents that remember prior runs

It is calendar-capable. Jobs can run silently or notify the active frontend, and subagent conversations remain available through the conversation system. Google and Apple Calendar can be added to Second Brain as well with the right plugins.

## Frontends

The kernel ships one base frontend, the REPL (`frontend_repl.py`, a local terminal interface). Telegram — a private mobile chat interface (`frontend_telegram.py`) — is a store package, installed with the `bundle_starter`/`bundle_full` bundles or directly via `/packages install frontend_telegram`. Both live under `plugins/frontends/` once present. Telegram is highly recommended for the ease of use, but it takes a hot second to set up (again, use /setup for this).

Telegram is useful because the local runtime can reach you anywhere: approvals, proactive reminders, file delivery, scheduled-agent results, and mobile command menus all become part of the same conversation system.

`BaseFrontend` provides the shared runtime binding, command parsing path, form and approval submission, bus subscriptions, progress rendering hooks, session helpers, and `FrontendCapabilities` model. Each frontend implements only the transport-specific parts: receiving input, deriving a session key, rendering messages, sending attachments, showing buttons, and stopping cleanly.

Custom frontends are first-class plugins. A Discord bot, HTTP bridge, desktop shell, or narrow operational UI can be built as a sandbox frontend, tested with `test_plugin`, and live-loaded by `plugin_watcher` when enabled.

## Setup

### Requirements

- Python 3.11+
- An LLM subscription or local setup — technically optional, but you won't be able to chat without it. I recommend Atlas Cloud's token plan, it's cheap and has good models for this.
- A Telegram bot token and allowed user ID if you install the Telegram frontend, which is highly recommended.
- A computer with a GPU for embedding models, if desired.

### Install

```bash
git clone <https://github.com/henrydaum/second-brain>
cd "Second Brain"
pip install -r requirements.txt
```

`requirements.txt` is intentionally minimal — the kernel stays close to pure Python (`watchdog`, `croniter`, `psutil`, and a few others). Heavier dependencies (`openai`/`litellm`, `Pillow`, `sentence-transformers`, `faster-whisper`, `PyMuPDF`, `python-docx`, `python-pptx`, `pandas`, `python-telegram-bot`, …) belong to store packages and are installed automatically when you install the package that needs them.

### Configure

On first run, Second Brain creates its data directory (the DATA_DIR) automatically:

- Windows: `%LOCALAPPDATA%/Second Brain/`
- macOS: `~/Library/Application Support/Second Brain/`
- Linux: `${XDG_DATA_HOME:-~/.local/share}/Second Brain/`

From there, `/setup` writes the LLM profile and Telegram settings for you, and installing packages extends `enabled_frontends`/`autoload_services` as needed. A fresh kernel starts with `enabled_frontends: ["repl"]` and `autoload_services: ["llm", "timekeeper"]`.

The most important setting once indexing is installed is `sync_directories`: the folders Second Brain should watch and index. The attachment cache is included by default so files sent through frontends can enter the same pipeline. You can add multiple folders here, and they all get synced automatically. As soon as you set a sync directory, the REPL and app.log will be flooded with task status messages. Don't worry — that's the sync working as intended. It'll stop once the sync is complete. (You can use Telegram if you prefer a cleaner chat experience.)

Illustrative shape after the `starter` bundle and `/setup` (LiteLLM backend, Telegram enabled):

```json
{
  "sync_directories": [
    "C:/Users/you/Documents",
    "C:/Users/you/AppData/Local/Second Brain/attachment_cache"
  ],
  "enabled_frontends": ["repl", "telegram"],
  "autoload_services": ["llm", "timekeeper"],
  "telegram_bot_token": "",
  "telegram_allowed_user_id": 0,
  "llm_profiles": {
    "default": {
      "llm_endpoint": "https://api.atlascloud.ai/v1",
      "llm_api_key": "ATLAS_API_KEY",
      "llm_context_size": 0,
      "llm_service_class": "LiteLLMService"
    }
  },
  "default_llm_profile": "default",
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

- Run `/setup` for guided onboarding; it installs a bundle and writes the LLM/Telegram config.
- Configure LLM profiles with `/llm`, agent profiles with `/agent`, and app/plugin settings with `/config`.
- `llm_context_size: 0` lets automatic compaction manage context.
- `LiteLLMService` (from the `starter` bundle) reaches most providers; point `llm_endpoint`/`llm_api_key` at whichever you use.
- `LiteLLMService`: be careful with the model_name parameter. It may need to be prefixed (like 'openai/gpt-5.4'), but it depends on the cloud provider you are using. Look this up if not sure.
- Each `llm_profiles` entry is registered as its own service, and the `llm` router follows `default_llm_profile`.
- Installed 'extension'-type services auto-load when present; you don't need to list them in `autoload_services`.
- You can edit config.json and plugin_config.json directly.

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

The kernel ships REPL UX and introspection commands only:

| Command | Purpose |
|---|---|
| `/setup` | Guided onboarding: install a bundle, configure the LLM and Telegram |
| `/packages` | Browse, install, and uninstall store packages and bundles |
| `/agent` | Select, switch, edit, or remove agent profiles |
| `/llm` | Select, edit, set default, or remove LLM profiles |
| `/config` | Select and edit config and plugin settings |
| `/conversations` | Browse, switch, and manage conversations |
| `/clear` | Clear the current conversation |
| `/cancel` | Cancel the current interaction |
| `/frontends` | Enable or disable frontend plugins |
| `/services` | Select and load or unload services |
| `/tasks` | Pause, resume, reset, retry, or trigger tasks |
| `/tools` | Select and call tools |
| `/commands` | List available commands |
| `/locations` | Show project and plugin directories |
| `/debug` | Inspect runtime state & recent errors |
| `/update` | Pull most recent Repo state |

Other commands (for example `/schedule` for cron jobs, or MCP commands) arrive with the packages that provide them.

The kernel ships **no built-in tools** — a fresh install can converse but has no agent-callable actions. Tools come from the store; the `starter` and `full` bundles install the common ones, and you can add others individually with `/packages install <stem>`. Frequently installed tools include:

| Tool | Purpose | Bundle |
|---|---|---|
| `read_file` | Read exact text from files | starter |
| `edit_file` | Create, overwrite, replace, append to, or delete UTF-8 text files | starter |
| `run_command` | Run scoped terminal commands, with approval for broad actions | starter |
| `sql_query` | Query SQLite read-only | starter |
| `ask_user_question` | Ask the user a structured question | starter |
| `test_plugin` | Diagnose a plugin source file and summarize broad regression tests | starter |
| `hybrid_search` | Search local files with fused lexical and semantic ranking | full |
| `lexical_search` | Search local files by exact terms and keywords | full |
| `semantic_search` | Search local files by embedding similarity | full |
| `web_search` | Search the public web | web_search |

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

Second Brain is inspired by the human brain. Explorations into neurons turned into the creation of artificial neural networks, which then paved the way for attention mechanisms and transformers. From there came LLMs, and then came the agentic abilities: RAG, tool calls, and cron jobs. With each iteration, Second Brain became closer to its biological inspiration.

Second Brain is still pretty far from the real brain, in many ways. However, it can also do many things better than the human brain ever could. Building it has helped me to better understand the role of AI in my life, and in society. I found the process of building to be extremely valuable, because I realized that the value of AI is that it can be built into so many things. The role of the person is to guide it into productive and creative areas.

Second Brain ships as an unfinished product: a tiny, pure Python kernel. It's up to you to decide how you want to finish it.

## License

MIT

---

An agent by Henry Daum
