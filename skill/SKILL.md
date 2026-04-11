---
name: second-brain
description: File intelligence toolkit — index, search, and query local files via MCP
metadata: {"openclaw":{"emoji":"🧠","requires":{"env":["SECOND_BRAIN_MCP_URL"]}}}
---

# Second Brain

Second Brain is your file intelligence toolkit. It indexes files on disk (documents, images, spreadsheets, code, audio, video), extracts content, builds search indexes, and exposes everything over MCP (Model Context Protocol). Use it to manage and query your sync directory like a personal knowledge base.

## Connection

Second Brain runs an MCP server over streamable-HTTP.

- `SECOND_BRAIN_MCP_URL` (required): Full MCP endpoint URL (e.g. `http://192.168.1.50:5123/mcp`)

Connect using any MCP-compatible client. All tools, resources, and prompts are auto-discovered on connection.

## Tools

Tools are the primary interface. Call them directly — no chat wrapper needed.

### Search & Query

| Tool | Description |
|------|-------------|
| `hybrid_search` | Combined semantic + lexical search across all indexed files. Best general-purpose search. |
| `semantic_search` | Pure vector similarity search. Good for conceptual/meaning-based queries. |
| `lexical_search` | Full-text keyword search with BM25 ranking. Good for exact terms. |
| `sql_query` | Execute read-only SQL (SELECT/PRAGMA) against the file database. |
| `web_search` | Search the web and return summarized results. |

### File Operations

| Tool | Description |
|------|-------------|
| `read_file` | Read the extracted text content of an indexed file by path. |
| `write_note` | Create or overwrite a text file in the sync directory. |
| `run_command` | Execute a shell command (requires approval). |

### System

| Tool | Description |
|------|-------------|
| `build_plugin` | Create or update sandbox plugins (tools, tasks, or services). |
| `system_command` | Run a system administration slash command (see below). |

### System commands (via `system_command` tool)

The `system_command` tool accepts a single `command` string. These manage services, tasks, tools, config, and the processing pipeline.

```
system_command(command="services")
system_command(command="load llm")
system_command(command="stats")
```

| Command | Description |
|---------|-------------|
| `help` | List all available commands |
| `stats` | System overview (file counts, pipeline status) |
| `services` | List services and their load status |
| `load <service>` | Load a service (e.g. `llm`, `text_embedder`, `ocr`) |
| `unload <service>` | Unload a service |
| `tasks` | List tasks with status counts |
| `pipeline` | Show task dependency graph |
| `pause <task>` | Pause a task |
| `unpause <task>` | Unpause a task |
| `reset <task>` | Reset a task to pending |
| `retry <task>\|all` | Retry failed entries |
| `tools` | List registered tools |
| `enable <tool>` | Enable a tool for agent use |
| `disable <tool>` | Disable a tool |
| `locations [tools\|tasks\|services]` | List file system locations |
| `reload` | Hot-reload plugins |
| `call <tool> {json}` | Call a tool directly |
| `clear` | Clear agent conversation history |
| `config [key]` | Show all config settings, or one setting by key |
| `configure <key> <value>` | Update a config setting |

Call `system_command(command="stats")` or `system_command(command="services")` to understand the current system state before taking action.

## Resources

Resources provide read-only context without making a tool call — useful for populating context windows.

| URI | Description |
|-----|-------------|
| `secondbrain://stats` | File counts by modality and pipeline status |
| `secondbrain://services` | All registered services and their load status |
| `secondbrain://tasks` | All registered tasks and their status |
| `secondbrain://tools` | All registered tools with descriptions and parameters |
| `secondbrain://pipeline` | Task dependency graph and queue counts |
| `secondbrain://tables` | List of all database tables |
| `secondbrain://schema/{table_name}` | Column schema for a specific table |

## Prompts

| Name | Description |
|------|-------------|
| `second_brain_identity` | Load the full Second Brain system prompt with persona, tools, and rules |

## Handling media in tool results

Some tools return mixed content (text + media files). When a result includes media:

- **Images**: Display inline alongside the text response.
- **Audio**: Attach as audio when possible.
- **Files**: Attach as documents. The text summary is already in the response.

## When to use Second Brain

- Searching or querying the user's local files
- Reading or retrieving indexed documents
- Getting summaries or insights from file content
- Running SQL queries against the file database
- Building new tools/tasks via the sandbox plugin system
- Managing the system (loading models, checking pipeline status)

## When NOT to use Second Brain

- General knowledge questions unrelated to the user's files
- Tasks that don't involve local file intelligence
