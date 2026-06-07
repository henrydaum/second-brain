<img width="1440" height="569" alt="Second Brain logo" src="https://github.com/user-attachments/assets/598ab57f-ed6b-491a-9cd6-142b93b09244" />

# Sponsor

<div align="center">
  <img src="https://github.com/user-attachments/assets/9e7ff971-8159-4081-b8bc-9b9ff5edd4ff#gh-light-mode-only" width="500" alt="Atlas Cloud Logo">
  <img src="https://github.com/user-attachments/assets/8497513e-09a4-4151-8b8d-ed8be782a389#gh-dark-mode-only" width="500" alt="Atlas Cloud Logo">
</div>

[Atlas Cloud](https://www.atlascloud.ai/?utm_source=github&utm_medium=link&utm_campaign=second-brain) is a full-modal AI inference platform that gives developers one API for LLMs, image generation, video generation, and more. Atlas Cloud's coding plan promotion is here: [https://www.atlascloud.ai/console/coding-plan](https://www.atlascloud.ai/console/coding-plan)

# Second Brain Lite

Second Brain Lite is the kernel of Second Brain: a small local-first AI runtime that boots the conversation engine, loads plugins, and lets the rest of the product arrive as installable packages.

This branch is intentionally stripped down. The full Second Brain product can search files, schedule agents, talk through Telegram, index media, use integrations, and run many tools. Lite keeps the host that makes those things possible:

- a durable conversation state machine
- an agent turn loop that uses whatever tools are installed
- SQLite persistence
- the five plugin families
- a REPL frontend
- package-store install and uninstall
- a live plugin watcher
- a parser service with lightweight text and image helpers
- an LLM router that follows the configured default profile

Everything else belongs in the store.

## Kernel Boundary

The kernel ships only the pieces needed to run, extend, and recover the runtime.

Built-in services:

| Service | Role | Lifecycle |
|---|---|---|
| `llm` | Router for the configured default LLM profile | managed |
| `compactor` | Conversation summary helper for tight context windows | extension |
| `parser` | Parser registry and helper discovery | extension |
| `plugin_watcher` | Live plugin/package reload substrate | extension |

Built-in frontend:

| Frontend | Role |
|---|---|
| `repl` | Local terminal chat interface |

Built-in tools: none in the tracked kernel tree. Everyday tools such as
`read_file`, `ask_user_question`, file editing, shell, SQL, retrieval, and plugin
authoring arrive through packages such as `starter`.

The pipeline substrate still exists, but the lite kernel ships no indexing tasks. Install parser, extraction, chunking, embedding, search, scheduling, Gmail, Telegram, MCP, or other packages from the store when you want those capabilities.

## Architecture

Second Brain is organized around a few stable layers:

| Directory | Purpose |
|---|---|
| `state_machine/` | Pure conversation primitives: participants, turns, phases, forms, approvals, and serializable phase frames |
| `runtime/` | Sessions, persistence, approvals, command dispatch, frontend actions, and agent-turn orchestration |
| `agent/` | System prompt assembly and tool registry |
| `plugins/` | Built-in plugin contracts and the kernel plugin set |
| `pipeline/` | SQLite task queue, orchestrator, event trigger, and file watcher substrate |
| `events/` | In-process pub/sub bus |
| `config/` | Config schema, defaults, plugin config, and user-scoped settings |
| `templates/` | Authoring templates for each plugin family |

The state machine stays pure. Frontends submit typed runtime actions. Plugins receive a `SecondBrainContext` instead of reaching through the whole app.

## Plugin System

Everything user-extensible is one of five plugin families:

| Family | Base class | Built-in path | Installed path | File prefix |
|---|---|---|---|---|
| Tools | `BaseTool` | `plugins/tools/` | `installed_plugins/tools/` | `tool_*.py` |
| Tasks | `BaseTask` | `plugins/tasks/` | `installed_plugins/tasks/` | `task_*.py` |
| Services | `BaseService` | `plugins/services/` | `installed_plugins/services/` | `service_*.py` |
| Commands | `BaseCommand` | `plugins/commands/` | `installed_plugins/commands/` | `command_*.py` |
| Frontends | `BaseFrontend` | `plugins/frontends/` | `installed_plugins/frontends/` | `frontend_*.py` |

Sandbox plugins live under the Second Brain data directory. Installed store packages live beside them and are tracked with package receipts.

Service lifecycle matters:

- `managed` services are user-loadable backends. They appear as load/unload candidates in `/services`.
- `extension` services auto-load when installed and are not manually unloaded. Use this for runtime hooks, registries, and infrastructure with no useful manual off switch.

## Package Store

Lite includes the local package-store client. Store packages are read from the `origin/store` git ref without switching branches. Installing a package:

1. Resolves package manifests and dependencies.
2. Detects needed Python packages from third-party imports or manifest metadata.
3. Installs missing Python packages with the current interpreter.
4. Copies package files into `installed_plugins/`.
5. Writes receipts under `DATA_DIR/packages/receipts/`.
6. Reloads affected services when needed.

Uninstall reverses that path, refuses live dependents, and offers cleanup for package-owned config keys, tables, and safe Python dependencies.

Useful starter packages include:

- `starter`
- `service-litellm`
- `frontend-telegram`
- `scheduling`
- `web-search`
- `gmail`
- `mcp`
- `google-drive`
- `indexing-search`
- parser packages such as `parser-pdf`, `parser-office`, `parser-audio`, and `parser-video`

Use `/packages` from the REPL to search, install, inspect, and uninstall packages.

## Setup

Requirements:

- Python 3.11+
- An LLM backend package for agent turns, usually `service-litellm`
- An API key or environment variable for your chosen provider

Install:

```bash
pip install -r requirements.txt
python main.py
```

Run tests:

```bash
python -m pytest -q --basetemp .pytest_tmp_full
```

The kernel dependencies are intentionally small. Optional capabilities bring their own dependencies through the store.

## Configuration

On first run, Second Brain creates its data directory:

- Windows: `%LOCALAPPDATA%/Second Brain/`
- macOS: `~/Library/Application Support/Second Brain/`
- Linux: `${XDG_DATA_HOME:-~/.local/share}/Second Brain/`

Core config lives in `config.json`. Plugin and user-scoped values live in `plugin_config.json`.

Fresh lite defaults keep managed autoload minimal:

```json
{
  "enabled_frontends": ["repl"],
  "autoload_services": ["llm"]
}
```

Extension services such as `parser`, `compactor`, and `plugin_watcher` load automatically because of their lifecycle.

LLM profiles are stored by model/profile name:

```json
{
  "llm_profiles": {
    "openai/gpt-4o-mini": {
      "llm_endpoint": "",
      "llm_api_key": "OPENAI_API_KEY",
      "llm_context_size": 0,
      "llm_service_class": "LiteLLMService",
      "llm_capabilities": {
        "image": true,
        "audio": false,
        "video": false
      }
    }
  },
  "default_llm_profile": "openai/gpt-4o-mini"
}
```

Use `/setup` for first-run onboarding and `/llm` to add, edit, set, or remove LLM profiles.

## Commands

The kernel command set is intentionally operational:

| Command | Purpose |
|---|---|
| `/cancel` | Cancel the active form, approval, or interaction |
| `/clear` | Clear the current conversation |
| `/commands` | List available slash commands |
| `/config` | Inspect and edit config settings |
| `/conversations` | Browse, switch, and manage conversations |
| `/frontends` | Inspect and control frontend plugins |
| `/llm` | Manage LLM profiles and defaults |
| `/locations` | Show data, sandbox, installed package, and repo paths |
| `/new` | Start a new conversation |
| `/packages` | Search, install, list, inspect, or uninstall store packages |
| `/services` | Inspect managed and extension services |
| `/setup` | Install starter capabilities and configure the first LLM |
| `/tasks` | Inspect task pipeline state |
| `/tools` | List and call tools |

Store packages add more commands, tools, services, tasks, and frontends.

## Runtime Flow

Frontends submit actions to `ConversationRuntime.handle_action(...)`: `send_text`, `send_attachment`, `call_command`, `submit_form_text`, `answer_approval`, `cancel`, and related typed actions.

The runtime:

1. Opens or restores the session.
2. Refreshes command and tool specs.
3. Applies the action through the state machine.
4. Persists the phase marker and messages.
5. Runs the agent turn when priority passes to the agent.
6. Returns a `RuntimeResult` for the frontend to render.

That shape lets the REPL, an installed Telegram bot, an installed scheduled task, or a future HTTP frontend use the same conversation rules.

## Extension Authoring

Use the templates as the source of truth:

- `templates/tool_template.py`
- `templates/task_template.py`
- `templates/service_template.py`
- `templates/command_template.py`
- `templates/frontend_template.py`

Recommended loop:

1. Read the matching template.
2. Read a similar built-in or installed plugin.
3. Write the plugin into the sandbox tree with the editing capability available in your environment.
4. Let `plugin_watcher` load it.
5. Use `/tools`, `/services`, `/commands`, or `/tasks` to inspect it.
6. Move stable capabilities into the store when they belong outside the kernel.

Keep plugins cheap to discover. Put heavy imports inside `_load()` or `run()`, and declare `requires_services` for tools that need optional backends.

## Project Layout

```text
Second Brain Lite/
├── main.py
├── main.pyw
├── paths.py
├── agent/
├── attachments/
├── config/
├── events/
├── pipeline/
├── plugins/
│   ├── commands/
│   ├── frontends/
│   ├── services/
│   ├── tasks/
│   └── tools/
├── runtime/
├── state_machine/
├── templates/
└── tests/
```

Runtime data lives outside the repo under `DATA_DIR`.

## Philosophy

Lite is the small, reliable host. The store is where capabilities grow.

That split keeps startup boring, installs understandable, and failures local. The kernel should be easy to reason about: boot the runtime, preserve conversations, route agent turns, load plugins, and get out of the way.

## License

MIT

---

An agent by Henry Daum
