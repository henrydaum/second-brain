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

<mark>*NEW!* You can now use Second Brain to replace Claude Code by installing the `claude_code` package! It's built on the same toolset which Claude Code uses, making Second Brain behave like Claude Code.</mark>

<mark>*NEW!* Second Brain now has token streaming. Telegram now has silky smooth token streaming plus rich text.</mark>

*For an example of what Second Brain can do, visit https://second-brain.art! It's an interactive art exhibition.*

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
- Push reminders, findings, daily briefs, and alerts into Telegram.
- Author, test, and live-load new tools, tasks, services, commands, and frontends. The possibilities are endless.

This might seem like a lot. However, Second Brain can only do what you tell it to do. It will never edit your files or share information unless you explicitly enable it to. Be careful with which plugins you enable. Although all of the built-in ones are highly tested, they still carry risk — and some more than others.

## The Kernel And The Package Store

Second Brain ships as a microkernel plus a package store.

**The kernel** is what lives in this repository's main tree. It is almost pure Python and boots *fast*. It holds the runtime, runs the conversation state machine and agent turn, persists conversations, manages config, and discovers and loads plugins. It ships only the plugins it cannot run without: the LLM router, the compactor (for LLM context size management), Timekeeper (the event clock), the plugin watcher (live install and reload), the REPL frontend, and a small set of REPL admin commands. Parsing is kernel routing rather than a plugin (`parsing/`), and ships one dependency-light text parser. There are **no built-in tools or tasks** — a fresh kernel can hold a conversation, but it cannot search your files or edit code until you install packages. *Even the LLM backends are plugins! LLMs involve heavy and unstable dependencies, so to ensure kernel stability it was best to make them plugins.*

**The store** is a parallel branch (`store`) that mirrors what a fully loaded install looks like: every optional tool, task, service, command, frontend, and helper is there, plus named *bundles* that group them. You browse and install from it with `/packages`, and the kernel copies the files into your data directory and live-loads them with the Plugin Watcher.

### Getting started

A fresh install has no LLM backend and no frontend beyond the REPL. The fastest path is the onboarding wizard:

```
/setup
```

It installs a bundle with basic plugins, configures an LLM profile, and optionally sets up Telegram. If you would rather drive it by hand, install the essentials bundle directly:

```
/packages install bundle_essentials
```

The **essentials** bundle is the recommended first install: an LLM backend (LiteLLM, which reaches most providers), the Telegram frontend, file read/edit/search, shell and script running, SQL, ask-user-question, plugin validation, subagents, web search, and auto-titling. Note: you cannot chat with an LLM in Second Brain until you install an LLM backend (LiteLLM recommended). Once that works, the natural next step is the **knowledge base** — every file parser, OCR, audio/video transcription, embeddings, and the lexical/semantic/hybrid search tools, so the agent can find things in your own files:

```
/packages install bundle_knowledgebase
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

The /packages command automatically maintains a clean separation between the kernel and plugins. All installed plugins and helpers will be within the `installed` folder of the DATA_DIR (data directory, see below). You don't have to use the /packages command to install or remove plugins. You can also simply drag and drop plugins and their helpers into this folder. It'll be automatically picked up by the Plugin Watcher service. However, if you do it this way you will have to pip install their Python dependencies as well, if you haven't got them already.

### Contributing to the store

The store is just a git branch, so adding a plugin is a pull request. Author and test your plugin as a sandbox plugin (see [Plugin System](#plugin-system) and the [Extension Authoring Guide](#extension-authoring-guide)), then open a pull request against the `store` branch that adds your `tool_*.py` / `task_*.py` / `service_*.py` / `command_*.py` / `frontend_*.py` (and any `helpers/`) under the matching family directory. Declare dependencies with the `dependencies_files` and `dependencies_pip` fields so the package manager can resolve them, and to group several files under one install, add a `bundles/<name>.json` manifest listing the store-relative files.

You can also simply send me an email at henrydaum8609@gmail.com with what you want to make :-)

## Core Architecture

Second Brain is built from a few durable pieces:

- `state_machine/` contains the pure conversation primitives: participants, turns, phases, actions, forms, approvals, and serializable phase frames.
- `runtime/` owns sessions, persistence, approvals, state-machine dispatch, agent turns, and the context passed into plugins.
- `plugins/` is extension substrate: discovery, watching, registries, and the native adapters used by the sandbox bridge. Implementations live in the three plugin trees described below.
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

| Family | Folder | Prefix | Contract |
|---|---|---|---|
| Tools | `tools/` | `tool_` | LLM-callable actions via `BaseTool` |
| Tasks | `tasks/` | `task_` | Pipeline and event work via `BaseTask` |
| Services | `services/` | `service_` | Shared backends via `BaseService` |
| Commands | `commands/` | `command_` | User slash commands via `BaseCommand` |
| Frontends | `frontends/` | `frontend_` | User transports via `BaseFrontend` |

Three more folders sit beside them, holding code the kernel routes to without a
base class: `parsers/` (`parse_*.py`, reached by file extension), `llm/`
(`llm_*.py`, reached by model profile), and `scripts/` (any name — SDK code the
agent runs rather than registers, where the directory is the whole declaration).
The prefix is the rule: a folder whose files carry one is *scanned*, a folder
without one is reached by something naming the file. Code belonging to a single
plugin goes in that family's own `helpers/` subfolder.

That same set of folders appears in three **trees**, named for where the code
came from: `bundled/` in the project root ships with the app, `DATA_DIR/installed`
is delivered by the package store, and `DATA_DIR/workspace` is the agent's own —
its free-write code tree, because everything under it runs in a
subprocess. The user can separately open user-owned folders through
`fs_writable_dirs`; those are work destinations, not extension trees. Bundled plugins are source-controlled; the other two live in the
Second Brain data directory and can be created while the app is running. Valid plugins are discovered on startup; the kernel plugin watcher then syncs adds, edits, and deletes live.

The agent can create new plugins on-the-fly. When you ask Second Brain to make
one, the kernel prompt teaches this install-independent workflow:

1. Read `docs/SDK.md`.
2. Read the relevant file in `templates/` from top to bottom.
3. Follow the code pointers in those documents when a detail is still unclear.
4. Create or edit the file under the matching `DATA_DIR/workspace/` root.
5. Validate it through `sdk.plugins.validate` or any currently installed tool
   exposing that Request, then run or register it and verify a small behavior.
6. If validation or loading fails, fix the same file and repeat until it
   conforms and works.

Plugins can declare their own system prompt text. If a plugin is not loaded,
its text takes no context. One name has two shapes: write
`agent_prompt = "..."` when the text never changes, or
`def agent_prompt(self, sdk)` when it depends on live state. The system prompt
is recalculated with every step of the conversation while preserving stable
prompt blocks for caching.

In other words, the system prompt has been fully engineered.

## Security

Sandbox plugins cannot act on the outside world directly. Every mediated effect
is a typed Request that the kernel classifies, executes, and records on their
behalf. Validation blocks direct kernel access; provenance determines process
isolation; permission policy decides whether each Request is allowed, refused,
or shown to the user. Agent-authored workspace code is always subprocessed.
Installed code is subprocessed when its imports or declarations require it.

Third-party libraries are the important limit: their internal I/O cannot be
converted into Requests, so they are isolated and disclosed rather than
treated as fully mediated. Treat extension installation as a capability change,
not as a harmless file copy. See `docs/The Second Brain Security Contract.md`
for the model, `docs/PERMISSIONS_MAP.md` for the decision order, and
`docs/SECURITY_CONTRACT_APPENDIX.md` for the Request catalogue.

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

The kernel itself parses only plain text, code and CSV/TSV. To parse the rest, install the knowledge-base bundle, which carries every parser:

`/packages install bundle_knowledgebase`

This will also give you the ability to process attachments with these file extensions. Individual parsers install by name too — `/packages install parse_pdf` — if you only want one.

## Events, Cron Jobs, And Subagents

Second Brain is proactive, not just reactive. `/schedule` and the Timekeeper are kernel; the agent-facing half arrives with the essentials bundle:

`/packages install tool_schedule_subagent`

Path-driven tasks process files. Event-driven tasks respond to bus events. Timekeeper is the kernel service that creates one-time and recurring event emissions using cron expressions; `tool_schedule_subagent` and `tool_spawn_subagent` are what let the agent use it. Scheduled subagents can wake up, read their conversation history, run tools, and optionally send their final result back into chat, depending on their notification mode.

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

The kernel ships the REPL (`bundled/frontends/frontend_repl.py`, a local terminal interface). Telegram — a private mobile chat interface (`frontend_telegram.py`) — is a store package, installed with the `bundle_essentials` bundle or directly via `/packages install frontend_telegram`; once installed it lives under `DATA_DIR/installed/frontends/`. Telegram is highly recommended for the ease of use, but it takes a hot second to set up (again, use /setup for this).

Telegram is useful because the local runtime can reach you anywhere: approvals, proactive reminders, file delivery, scheduled-agent results, and mobile command menus all become part of the same conversation system.

`BaseFrontend` provides the shared runtime binding, command parsing path, form and approval submission, bus subscriptions, progress rendering hooks, session helpers, and `FrontendCapabilities` model. Each frontend implements only the transport-specific parts: receiving input, deriving a session key, rendering messages, sending attachments, showing buttons, and stopping cleanly.

Custom frontends are first-class plugins. A Discord bot, HTTP bridge, desktop shell, or narrow operational UI can be built as a sandbox frontend, checked with the validate tool, and live-loaded by the kernel plugin watcher.

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

From there, `/setup` writes the LLM profile and Telegram settings for you, and installing packages extends `enabled_frontends`/`autoload_services` as needed. A fresh kernel starts with `enabled_frontends: ["repl"]` and `autoload_services: ["timekeeper"]`; the kernel-owned LLM registry loads the default profile separately.

The most important setting once indexing is installed is `sync_directories`: the folders Second Brain should watch and index. The attachment cache is included by default so files sent through frontends can enter the same pipeline. You can add multiple folders here, and they all get synced automatically. As soon as you set a sync directory, the REPL and app.log will be flooded with task status messages. Don't worry — that's the sync working as intended. It'll stop once the sync is complete. (You can use Telegram if you prefer a cleaner chat experience.)

Illustrative shape after the `starter` bundle and `/setup` (LiteLLM backend, Telegram enabled):

```json
{
  "sync_directories": [
    "C:/Users/you/Documents",
    "C:/Users/you/AppData/Local/Second Brain/attachment_cache"
  ],
  "enabled_frontends": ["repl", "telegram"],
  "autoload_services": ["timekeeper"],
  "telegram_bot_token": "",
  "telegram_allowed_user_id": 0,
  "llm_profiles": {
    "default": {
      "llm_endpoint": "https://api.atlascloud.ai/v1",
      "secret_llm_api_key": "ATLAS_API_KEY",
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
- `LiteLLMService` (from the `starter` bundle) reaches most providers; point `llm_endpoint`/`secret_llm_api_key` at whichever you use.
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
| `validate` | Validate sandbox source and explain contract violations | starter |
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
├── plugins/                # Discovery, watcher, registries, native adapters
│   ├── native/
│   ├── plugin_discovery.py
│   └── plugin_watcher.py
├── bundled/                # Implementations shipped with the app
│   ├── commands/
│   ├── frontends/
│   ├── parsers/
│   └── services/
│
├── pipeline/
│   ├── database.py
│   ├── event_trigger.py
│   ├── orchestrator.py
│   └── watcher.py
│
├── agent/
│   ├── system_prompt.py
│   └── tool_registry.py
│
├── attachments/
├── config/
├── events/
├── docs/
│   ├── SDK.md                      # The sandbox SDK reference
│   ├── SECURITY_CONTRACT_APPENDIX.md # The Request catalogue
│   └── MIGRATING_PLUGINS.md        # Converting a native plugin
├── templates/
│   ├── command_template.py
│   ├── frontend_template.py
│   ├── hook_template.py
│   ├── llm_backend_template.py
│   ├── parser_template.py
│   ├── script_template.py
│   ├── service_template.py
│   ├── task_template.py
│   └── tool_template.py
└── DATA_DIR/
    ├── config.json
    ├── plugin_config.json
    ├── database.db
    ├── attachment_cache/
    ├── memory/
    ├── workspace/          # the agent-owned, freely writable tree
    │   ├── tools/
    │   ├── tasks/
    │   ├── services/
    │   ├── commands/
    │   ├── frontends/
    │   ├── parsers/
    │   ├── llm/
    │   └── scripts/
    └── installed/          # the package store's tree, same shape
```

## Extension Authoring Guide

New plugins are written against the sandbox SDK. Read `docs/SDK.md` for the Request
vocabulary and the return idiom, `sandbox/guest/bases.py` for what each family
may declare, and then the template for what is specific to that family:

- `templates/tool_template.py`
- `templates/task_template.py`
- `templates/service_template.py`
- `templates/command_template.py`
- `templates/script_template.py` — sandboxed code that is not a plugin
- `templates/frontend_template.py`
- `templates/hook_template.py`
- `templates/parser_template.py`
- `templates/llm_backend_template.py`

`docs/MIGRATING_PLUGINS.md` covers converting an existing native plugin.

Authoring rules:

- Tools expose LLM-callable capabilities and return whatever they like; reach
  for `sdk.ok(...)` only to attach an `llm_summary` or attachments.
- Tasks are pipeline/event workers and should be idempotent where possible.
- Services own reusable backends and are the natural persistent box; their
  state stays inside it and never crosses out.
- Commands are user-facing conversation actions and can define `FormStep` flows.
- Frontends are transports; they submit runtime actions and render runtime output.
- Plugins can declare `config_settings`, which appear in config views and are stored in `plugin_config.json`.
- Sandbox plugins must follow naming conventions: `tool_*.py`, `task_*.py`, `service_*.py`, `command_*.py`, and `frontend_*.py`.

For source-controlled additions, place stable app-shipped plugins under the matching `bundled/` family. For live experimentation, keep them under `DATA_DIR/workspace`, validate them, and then run or register them. `DATA_DIR/workspace` is agent-owned. The separate `fs_writable_dirs` setting may open user-owned project folders for no-dialog writes; those remain user data and are not extension discovery roots.

## Philosophy

Second Brain is inspired by the human brain. Explorations into neurons turned into the creation of artificial neural networks, which then paved the way for attention mechanisms and transformers. From there came LLMs, and then came the agentic abilities: RAG, tool calls, and cron jobs. With each iteration, Second Brain became closer to its biological inspiration.

Second Brain is still pretty far from the real brain, in many ways. However, it can also do many things better than the human brain ever could. Building it has helped me to better understand the role of AI in my life, and in society. I found the process of building to be extremely valuable, because I realized that the value of AI is that it can be built into so many things. The role of the person is to guide it into productive and creative areas.

Second Brain ships as an unfinished product: a tiny, pure Python kernel. It's up to you to decide how you want to finish it.

## License

MIT

---

An agent by Henry Daum
