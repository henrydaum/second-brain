# Which files the agent touched

Everything needed for a files drawer: what the agent **edited** and what it
**showed you**, per conversation, surviving a reload.

All of it comes from one Request.

```ts
import { sdk } from "@/lib/client";

const rows = await sdk<LedgerRow[]>("ledger.read", {
  conversation_id: 7,
  action_types: ["fs.write", "fs.write_bytes", "fs.delete", "fs.move",
                 "proc.run", "proc.start", "call_tool"],
});
```

Rows come back **newest first**. `ledger.read` is read-only, so it never raises
an approval dialog. Naming a conversation the user does not own is refused.

## Why not `conv.read`

Because it isn't there. `conversation_messages` has seven columns and no
metadata blob — attachment paths and tool narration are event-only, so a page
reload cannot tell that a turn showed you anything at all. The ledger is the
only place either fact is kept.

## A row

```ts
type LedgerRow = {
  id: number;              // monotonic; also the `since_id` cursor
  ts: number;              // epoch seconds, fractional
  origin: string;          // "sandbox" | "agent_enact" | "user_enact" | "system"
  action_type: string;     // "fs.write", "proc.run", "call_tool", …
  conversation_id: number | null;
  ok: 0 | 1;
  error_code: string | null;
  args_json: string;       // do not parse this for paths — see below
  data_json: string;       // parse this
};
```

`data_json` always has `chain`, `level` and `reason`. What matters here is what
else it has.

## Files the agent edited

`origin: "sandbox"`, `action_type` one of `fs.write` / `fs.write_bytes` /
`fs.delete` / `fs.move`:

**`origin` does not tell the agent's writes from the frontend's.** A file the
browser uploads goes through the same `fs.write_bytes` and is recorded
`"sandbox"` too. The chain's last hop is the one that answers:
`"http:web -> frontend:http"` for our own upload against
`"http:web -> edit_file"` for the agent's. Filter on that, or a files panel
lists the person's own attachments as things the agent did.

```jsonc
{ "paths": ["/srv/app/notes.md"], "bytes": 14400,
  "level": "safe", "reason": "workspace" }
```

`fs.move` carries both ends, source first. `action_type` tells you the effect.

**Read paths from `data_json.paths`, never from `args_json`.** That field is
capped at 4000 chars and past the cap the whole object is replaced by a
`{_truncated_chars, head, tail}` wrapper — and the argument that blows the cap
is the file's own contents. Parsing it would silently lose exactly the largest
edits.

## Files the agent edited by shell

`action_type: "proc.run"` or `"proc.start"`, when the command line named files
and the command exited zero:

```jsonc
{ "paths":   ["/srv/app/build"],
  "deleted": ["/srv/app/build"],
  "via":     "shell",
  "command": "rm -rf build" }
```

`via: "shell"` means the paths were **read out of a command line**, not
serviced by the kernel. It is a weaker claim — badge it, sort it lower, or
ignore the distinction; the flag is there so the choice is yours. `deleted` is
the subset that no longer exists.

Recognised: `rm`, `rmdir`, `mv`, `cp`, `mkdir`, `touch`, `ln`, `unlink`, plus
`del`, `erase`, `copy`, `move`, `md`. Paths are absolute, resolved against the
command's `cwd`.

Everything else records no paths at all and you will see no `paths` key:
unlisted programs (`npm install`, `git checkout .`), globs (`rm *.log` names
nothing until a shell expands it), redirects, `$(…)`, subshells, `cmd /c …`,
and any command that failed. Under-reporting is deliberate — a file shown as
deleted that is still there is worse than one missing from the list.

## Files the agent showed you

`origin: "agent_enact"`, `action_type: "call_tool"`:

```jsonc
{ "attachments": ["/srv/app/chart.png", "/srv/app/report.md"],
  "llm": "claude-sonnet-5" }
```

These are the same paths that arrive live as an `attachments` render frame —
this is just the copy that survives a reload.

This is `ToolResult.attachment_paths`, i.e. whatever a tool returned from
`sdk.ok(attachments=[...])`. (If you find `gui_display_paths` in any older
notes: it was an alias for the same field and was deleted in May 2026.)

**It covers tools only.** A task, service or slash command has no `ToolResult`
to hang files on and uses `runtime.push_message(attachments=[...])` instead,
which goes straight to the bus and is not recorded — so files pushed that way
render live and are gone on reload.

## Keeping it current

Renders are events; the ledger is state. Load once when the conversation opens,
then poll incrementally rather than re-reading:

```ts
// after a `tool_status` frame with status "finished", or when `typing` → false
const fresh = await sdk<LedgerRow[]>("ledger.read", {
  conversation_id: id,
  since_id: cursor,          // highest id you already hold
  action_types: [...],
});
```

`since_id` filters in SQL on the `(conversation_id, id)` index, so an idle poll
is cheap. There is no push notification for ledger rows — the existing frames
are your trigger.

## Opening a file

The paths are **host filesystem paths**, not URLs — but there is a route that
turns one into a URL:

```
GET /files?path=<encodeURIComponent(hostPath)>
Authorization: Bearer <token>
```

It reads through the same `fs.read_bytes`, so it is the same policy check and
the same ledger row; what it adds is a real HTTP body with a real
`Content-Type`, which is what an `<img>`, `<video>` or `<audio>` needs.

**Media elements cannot send an `Authorization` header** — `<img src>` and
`<video src>` issue their own requests, and there is nowhere to put one. So
`/files` accepts `?token=` as well, for exactly the reason `/events` does.
These are the only two routes that do; everything a *script* calls sends the
header.

```ts
export function fileUrl(hostPath: string): string {
  const query = [
    `thread=${encodeURIComponent(THREAD)}`,
    `path=${encodeURIComponent(hostPath)}`,
    `token=${encodeURIComponent(TOKEN)}`,   // media elements send no headers
  ].join("&");
  return new URL(`/files?${query}`, window.location.origin).toString();
}
```

**Build the query by hand.** `URLSearchParams` — and therefore
`serverUrl().searchParams.set()` — serialises as
`application/x-www-form-urlencoded`, where a space becomes `+` rather than
`%20`. The route percent-decodes, so every path under
`AppData\Local\Second Brain\` arrives as `Second+Brain` and answers `404`. The
signature at the top of this section says `encodeURIComponent` and means it.

This is a hard one to catch: `curl` and any hand-written test encode the space
correctly, so the route looks fine from everywhere except the application.

### Picking a renderer

**`/files` serves every extension.** The bytes come back whatever the file is;
the extension only decides the `Content-Type` label, falling back to
`application/octet-stream` (a download) when nothing recognises it.

So the intended flow works as you'd hope: **categorise by Second Brain's
modality map, then render by modality.**

```ts
const modality = await sdk<string>("parse.modality", { extension: ".mp4" });
```

Verified answers, on a machine with no heavy parsers installed:

| Extension | Answer |
|---|---|
| `.png` `.heic` | `"image"` |
| `.mp4` `.avi` | `"video"` |
| `.wav` | `"audio"` |
| `.txt` `.md` `.py` `.csv` | `"text"` |
| `.pdf` `.docx` `.xlsx` `.gguf` | `"unknown"` |

Two things to take from that. It answers `"unknown"`, **never `null`** — and
image/audio/video come from a static map that works with no parser installed,
so those three are free and never depend on what's on the machine. Installing
`parser-pdf` or `parser-office` moves `.pdf`/`.docx` off `"unknown"`.

**Don't gate rendering on modality alone.** `"unknown"` means *no parser is
registered*, not *not renderable* — a PDF is `"unknown"` and the browser
renders it perfectly. Use modality for the three media kinds and text, then
fall back to the extension:

| Test | How |
|---|---|
| modality `image` | `<img src={fileUrl(p)}>` |
| modality `video` | `<video controls src={fileUrl(p)}>` — seeks, via Range |
| modality `audio` | `<audio controls src={fileUrl(p)}>` |
| ext `.csv` `.tsv` | fetch `fileUrl(p)`, parse in JS, render a table |
| ext `.xlsx` `.xls` `.parquet` | fetch `fileUrl(p)`, parse in JS (SheetJS etc.) |
| modality `text` (anything else) | `sdk("fs.read", { path })` → text or code |
| ext `.pdf` `.svg` | `<embed src={fileUrl(p)}>` — browser-native, modality-blind |
| ext `.docx` `.pptx` | `sdk("parse.file", { path, modality: "text" })` |
| anything else | download link to `fileUrl(p)` |

Every extension in the modality map is guaranteed a `Content-Type` whose
top-level type matches — `.avi` is `video/x-msvideo`, `.wma` is
`audio/x-ms-wma` — so an element you picked *by modality* is never handed
something it refuses to play. A server-side test enforces that pairing; it is
not a coincidence you have to re-check.

### Tabular: do it client-side, and key off the extension

Two traps here, both verified rather than assumed.

**`.csv` answers `"text"`, and always will.** The bundled `parse_text` claims
`.csv`/`.tsv` and the first registration for an extension wins, so installing
`parser-tabular` does *not* move them to `"tabular"` — it registers the same
extensions second and loses. That is correct for the question modality
actually answers, which is *how should the model ingest this*; CSV-as-text is
right for that. The drawer asks a different question — *how should a person
look at this* — and for CSV the two answers legitimately differ. So branch on
the extension, not the modality.

**`parse.file` cannot return spreadsheet content.** `parser-tabular` registers
`.xlsx` under `"tabular"` only, and tabular is not in `CROSSABLE` — a parse
result may carry `text` or `container` across a box boundary and nothing else.
So there is no Request that hands you a parsed sheet, by design.

Which is fine, because the browser is the better place for it anyway: fetch the
raw bytes from `/files` and parse them in JS. That is full fidelity, no round
trip per cell, and no dependency on which parser packages are installed.

`.docx` and `.pptx` are the opposite case and *do* work through `parse.file` —
`parser-office` registers them under `"text"` explicitly. It returns a failure
when the package is not installed, so treat that as "fall through to download"
rather than an error.

## Keeping this in sync with the kernel

Nothing binds this client to the kernel the way the sandbox binds a plugin, so
alignment is a thing to maintain rather than a thing you get. The cheapest
protection is to **ask rather than copy** wherever a Request exists:

| Instead of hardcoding | Ask |
|---|---|
| a modality table | `parse.modality` — free for image/audio/video |
| a content type | the `Content-Type` header `/files` already sends |
| which commands exist | `command.list` |
| which tools/models exist | `tool.list`, `llm.list` |
| conversation columns | `conv.read` with `details: true` |

What genuinely cannot be queried, and is therefore copied — check these when
the kernel moves:

- **The ten render kinds** (`src/lib/events.ts`). Source of truth is
  `sandbox/frontends.py:KINDS`. The union type makes an unhandled kind a
  compile error, which catches *removal* but not *addition*.
- **Error codes** — `approval_declined` in `src/lib/client.ts`. Source of truth
  is `sandbox/guest/codes.py`.
- **The state-marker sentinel** `__second_brain_state_machine__` in
  `src/lib/history.ts`, and the assistant-row `{content, tool_calls}` shape.
  Source of truth is `state_machine/serialization.py`.
- **`conversation_messages` column names** in `src/lib/history.ts`.

None of those has a Request that answers it, so a boot-time assertion is not
possible today. If drift here ever bites, the fix is a contract test in this
repo that runs against a live server — not a second copy of the table.

### Range and large files

`Accept-Ranges: bytes` is always advertised, and `Range` is honored — that is
what makes video seek instead of downloading everything before the point you
clicked.

**A large file answers `206` even with no `Range` header.** One response body
crosses in one wire message and that message is capped, so the route serves the
first window and puts the real total in `Content-Range: bytes 0-N/<size>`. Media
elements follow up by themselves. A plain `fetch` must loop until it has the
full length, so prefer `<img src>`/`<video src>` and let the browser do it.
`HEAD /files?path=…` answers the full `Content-Length` with no body if you need
to size something first.

### Failure

| Status | Meaning |
|---|---|
| `400` | No `?path=`, or it names a directory |
| `403` | Policy refused the read |
| `404` | Gone — deleted or moved since the row was written |
| `416` | Range starts past the end; `Content-Range` carries the real size |

A path in a row is a record of what happened, not a promise the file is still
there. A `404` means it has moved or been deleted — say so rather than showing
an empty pane.

**Do not use `HEAD` to find out which failure this was.** On its error path the
route sends a `Content-Length` for a body it then does not write, and a proxy in
front of it turns that mismatch into a `502` — so the one request whose entire
job is to report the real status reports `502` for every failure there is. Ask
for one byte instead: `Range: bytes=0-0` answers `206` when the file is fine and
the true status when it is not, with no body worth mentioning. This matters
because `<img>` and `<video>` report failure as a bare `onError` with no status
on it, and that follow-up request is the only way to tell "gone" from "refused".

## Three honest gaps

- `runtime.push_message(attachments=[...])` — how a task, service or slash
  command shows you a file, since none of them has a `ToolResult`. Bus only,
  not recorded, so it survives no reload.
- `plugin.install` / `plugin.uninstall` write and delete files directly through
  the package manager, with no `fs.*` Request. Invisible here.
- `config.write` rewrites `config.json`. Also invisible here.

Neither is worth special-casing for a files drawer, but don't claim the panel
shows every change on disk.

## Poking at it live

From the Second Brain REPL, to see real shapes before writing UI against them:

```
/conversations          # find the id you want
```

Then, from a shell in the Second Brain repo:

```bash
python -c "\
from config import config_manager; from pipeline.database import Database; import json; \
c=config_manager.load(); db=Database(c['db_path']); \
[print(r['action_type'], '|', r['data_json']) \
 for r in db.get_ledger_rows(conversation_id=7, limit=20)]"
```

Or hit the bridge exactly as the client does:

```bash
curl -s -X POST 'http://127.0.0.1:8787/sdk/ledger.read?thread=main' \
  -H 'Authorization: Bearer <secret_http_token>' \
  -H 'Content-Type: application/json' \
  -d '{"conversation_id": 7, "action_types": ["fs.write", "proc.run"]}'
```

Full Request reference: `docs/HTTP_PROTOCOL.md`.
