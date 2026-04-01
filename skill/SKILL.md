---
name: second-brain
description: File intelligence toolkit — index, search, and query local files via REST API
metadata: {"openclaw":{"emoji":"🧠","requires":{"env":["SECOND_BRAIN_URL"]}}}
---

# Second Brain

Second Brain is your file intelligence toolkit. It indexes files on disk (documents, images, spreadsheets, code, audio, video), extracts content, builds search indexes, and exposes everything over a REST API. Use it to manage and query your sync directory like a personal knowledge base.

## Configuration

- `SECOND_BRAIN_URL` (required): Base URL of the running instance (e.g. `http://192.168.1.50:5123`)
- `SECOND_BRAIN_TOKEN` (optional): Bearer token for auth. Include as `Authorization: Bearer <token>` if set.

## Discovering tools

Call `GET {SECOND_BRAIN_URL}/tools` to list all available tools with their names, descriptions, and parameter schemas. The tool set can change dynamically (sandbox plugins can be added at runtime).

## Calling tools

```
POST {SECOND_BRAIN_URL}/tools/{tool_name}
Content-Type: application/json

{"param1": "value1", "param2": "value2"}
```

Response:

```json
{
  "success": true,
  "error": null,
  "data": {},
  "llm_summary": "Human-readable summary of the result",
  "attachments": [
    {"path": "/path/to/file.png", "modality": "image", "url": "http://host:5123/files?path=..."}
  ]
}
```

- `llm_summary`: Read this to understand the result.
- `attachments`: Files associated with the result. Each has a fetchable `url` and a `modality` (image, audio, video, tabular, text).

## REPL commands (system administration)

```
POST {SECOND_BRAIN_URL}/repl
Content-Type: application/json

{"command": "load", "arg": "llm"}
```

Response: `{"output": "Loaded llm in 2.3s"}`

Use REPL commands to manage the system. Available commands:

| Command | Arg | Description |
|---------|-----|-------------|
| `help` | | List all available commands |
| `stats` | | System overview (file counts, pipeline status) |
| `services` | | List services and their load status |
| `load` | `<service>` | Load a service (e.g. `llm`, `text_embedder`, `ocr`) |
| `unload` | `<service>` | Unload a service |
| `tasks` | | List tasks with status counts |
| `pipeline` | | Show task dependency graph |
| `pause` | `<task>` | Pause a task |
| `unpause` | `<task>` | Unpause a task |
| `reset` | `<task>` | Reset a task to pending |
| `retry` | `<task>\|all` | Retry failed entries |
| `tools` | | List registered tools |
| `enable` | `<tool>` | Enable a tool for agent use |
| `disable` | `<tool>` | Disable a tool |
| `reload` | | Hot-reload plugins |
| `call` | `<tool> {json}` | Call a tool directly |
| `config` | `[key]` | Show all config settings, or one setting by key |
| `configure` | `<key> <value>` | Update a config setting (value is JSON or plain string) |

Call `stats` or `services` to understand the current system state before taking action.

## Fetching files

```
GET {SECOND_BRAIN_URL}/files?path={url_encoded_path}
```

Returns raw file bytes with correct Content-Type. Use the `url` from attachments directly. Only files within configured sync directories are served.

## Handling attachments

When a response includes attachments, send relevant ones to the user:

- **image**: Always send as an image attachment alongside your text reply.
- **audio**: Send as audio attachment when relevant.
- **video**: Send as video attachment when relevant.
- **tabular**: Send as document attachment. Text summary is already in `llm_summary`.
- **text**: Usually no need to attach — content is in `llm_summary`. Attach only if the user asks for the file.

## Error codes

| Code | Meaning |
|------|---------|
| 200  | Success |
| 400  | Malformed JSON or missing required fields |
| 401  | Bad or missing Bearer token |
| 403  | File path outside allowed sync directories |
| 404  | Unknown tool, endpoint, or file not found |
| 500  | Internal error |

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
