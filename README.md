<img width="1440" height="569" alt="Second Brain" src="https://github.com/user-attachments/assets/598ab57f-ed6b-491a-9cd6-142b93b09244" />

<p align="center">
  <b>A local-first AI runtime that can safely write its own features while it's running.</b>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> ·
  <a href="#the-phone-app">Phone App</a> ·
  <a href="#self-evolution">Self-Evolution</a> ·
  <a href="#security">Security</a> ·
  <a href="https://second-brain.art">Live Demo</a>
</p>

<p align="center">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-blue">
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-blue">
  <img alt="Tests" src="https://img.shields.io/badge/tests-2431%20passing%20in%2015s-brightgreen">
  <img alt="Kernel" src="https://img.shields.io/badge/kernel-~17k%20lines-lightgrey">
</p>

---

Most AI agents can write code. Almost none of them can safely **run** what they wrote.

Second Brain can. Ask it for a tool it doesn't have, and it will write the plugin, validate it against a capability contract, load it into the running process, and use it — without a restart, and without ever getting direct access to your disk, your network, or your kernel. Every effect it has on the outside world goes through a typed Request that the kernel classifies, approves, and records.

That's the whole idea: **an agent that grows new abilities on demand, inside a boundary that doesn't grow with it.**

```
you › I want to track my electricity bill from the PDFs in ~/bills

    ⚙ writing tool_parse_bill.py …
    ⚙ validating … ✓ conforms
    ⚙ loading … ✓ registered as parse_bill
    ⚙ scheduling monthly job … ✓

Done. I made a parser and a monthly check. Your average is $84/mo,
up 12% since March. Want me to alert you if it crosses $100?
```

---

## Why this one

Three things that are individually rare and, together, the point.

### 🔒 A real sandbox, not a promise

Plugin code cannot touch the outside world. It can't call `open()`. It can't import `requests`. It *asks* — through one of 121 typed Requests — and the kernel decides.

A policy function classifies every Request by what it does, who asked, and where it's headed. Safe ones run. Unsafe ones surface a dialog with the full **chain of provenance** — written by the kernel, not by the plugin, so it can't lie about who called it. Agent-authored code always runs in a subprocess with capped CPU, RAM, and filesystem visibility.

This is ~19,000 lines of working machinery with a green test suite, not a design document. Sandboxing the agent's own shell and filesystem access is a known-open problem across most self-hosted AI tools — usually an issue on the roadmap. Here it's the foundation everything else was built on top of.

### 🧬 Self-evolution with a ceiling

The agent extends itself by writing plugins — tools, background tasks, services, slash commands, even whole frontends — and hot-loading them. But extending what it can *ask for* never extends what it's *authorized to affect*. The permission boundary is the kernel's, and self-written code is the least trusted code in the system.

Growth and containment usually trade off against each other. Here they're independent.

### 🪶 A kernel small enough to actually trust

~17,000 lines, near-pure Python, boots fast. It holds a conversation, runs the state machine, persists to SQLite, and loads plugins. That's it. No built-in tools. No bundled model provider. Nothing you didn't ask for.

Everything else — search, embeddings, OCR, transcription, web access, Telegram, shell — installs from a package store as you need it. A dependency you never install is a dependency that can never break your runtime or read your files.

> **Also:** conversations run through a turn-based state machine (think Magic: The Gathering — priority, legal actions, suspendable phases, serializable frames). Interrupted flows survive a crash. Frontends don't own conversation logic, so the REPL, Telegram, and the web app all get approvals, forms, cancellation, and history for free.

---

## Quick Start

```bash
git clone https://github.com/henrydaum/second-brain
cd second-brain
pip install -r requirements.txt
python main.py
```

Then, in the REPL:

```
/setup
```

The wizard installs a starter bundle, configures an LLM provider, and optionally wires up Telegram. Two minutes, and you have a working assistant.

Prefer to drive it by hand?

```
/packages install bundle_essentials     # LLM, file tools, shell, search, subagents
/packages install bundle_knowledgebase  # every parser, OCR, transcription, embeddings
/packages install bundle_memory         # the durable memory library
/packages install bundle_gmail          # mail access
```

Or browse everything with `/packages` and pick à la carte — individual packages install by name, e.g. `/packages install tool_web_search`.

**Requirements:** Python 3.11+, and an LLM endpoint (any provider — LiteLLM reaches most of them).

<details>
<summary><b>Where your data lives</b></summary>

Created automatically on first run:

| OS | Path |
|---|---|
| Windows | `%LOCALAPPDATA%/Second Brain/` |
| macOS | `~/Library/Application Support/Second Brain/` |
| Linux | `${XDG_DATA_HOME:-~/.local/share}/Second Brain/` |

Inside: `config.json`, `plugin_config.json`, `database.db`, `installed/` (store packages), and `workspace/` (the agent's own tree — everything under it runs subprocessed).

</details>

---

## What it can do

Point it at your folders and it builds a live, searchable index of everything you own.

| | |
|---|---|
| 📄 **Read anything** | PDFs, Office docs, code, spreadsheets, archives, images (OCR), audio and video (transcription) |
| 🔍 **Find anything** | Lexical, semantic, or hybrid-ranked search across your own files, with citations and exact reads |
| 🧠 **Remember** | Durable memory library, plus full conversation history in SQLite |
| ⏰ **Act on its own** | Cron-scheduled subagents: daily briefs, inbox triage, folder watches, recurring research |
| 📱 **Reach you anywhere** | Telegram and a web app — approvals, reminders, files, and results wherever you are |
| 🛠️ **Build itself** | Author, validate, and hot-load new tools, tasks, services, commands, and frontends |
| 🌐 **Go outside** | Web search, HTTP, scoped shell — each one an explicit, revocable capability |

Everything above is opt-in. A fresh install can hold a conversation and nothing more. **It can only do what you install.**

> See it running: **[second-brain.art](https://second-brain.art)** — an interactive art exhibition built on Second Brain.

---

## The Phone App

<!-- TODO: drop a phone screenshot here — this is the highest-value image in the README.
     <img width="220" align="right" alt="Second Brain running as a PWA" src="..."> -->

A modern React frontend that installs to your iPhone or Android home screen as a PWA and talks to the runtime on your own machine over Tailscale. Familiar ChatGPT-style UI, built on [assistant-ui](https://www.assistant-ui.com/). Streaming, approvals, forms, and attachments all work.

Repo: **[henrydaum/second-brain-ui](https://github.com/henrydaum/second-brain-ui)**

Nothing is exposed to the public internet. The kernel binds loopback only; Tailscale gives your phone a private path to it.

### 1 — Install the HTTP frontend

In the Second Brain REPL:

```
/packages install frontend_http
```

### 2 — Configure it

```
/config
```

Set these four values:

| Setting | Value |
|---|---|
| `secret_http_token` | A long random string. Generate one: `python -c "import secrets; print(secrets.token_urlsafe(32))"` |
| `http_port` | `8787` (default) |
| `http_allowed_origins` | The origin you'll serve the app from, or `*` while testing |
| `http_static_dir` | Optional — point it at the app's built `dist/` to serve everything from one port |

Then enable it:

```
/frontends
```

### 3 — Set up Tailscale

Install [Tailscale](https://tailscale.com/download) on the machine running Second Brain **and** on your phone, and sign both into the same tailnet. Grab the host's tailnet address:

```bash
tailscale ip -4          # e.g. 100.101.102.103
tailscale status         # or use the MagicDNS name, e.g. my-desktop.tailnet-name.ts.net
```

Your phone can now reach `http://100.101.102.103:8787` from anywhere in the world, and nobody else can.

### 4 — Build the app

Requires Node `^20.19.0 || >=22.12.0`.

```bash
git clone https://github.com/henrydaum/second-brain-ui
cd second-brain-ui
npm install
cp .env.example .env.local
```

Edit `.env.local`:

```ini
VITE_SB_URL=http://127.0.0.1:8787   # used by the dev server's proxy
VITE_SB_TOKEN=<your secret_http_token>
VITE_SB_THREAD=main                 # two threads = two independent conversations
```

Run it in development:

```bash
npm run dev
```

Or build for real:

```bash
npm run build      # outputs to dist/
```

Serve `dist/` however you like — the simplest path is to set `http_static_dir` to that folder in step 2, so Second Brain itself serves the app on port 8787 and there's no CORS to configure at all.

> For a permanent always-on setup (Mac Mini + Caddy, with the token held in a private runtime env instead of the bundle), see [`docs/MACOS_DEPLOYMENT.md`](https://github.com/henrydaum/second-brain-ui/blob/main/docs/MACOS_DEPLOYMENT.md) in the UI repo.

### 5 — Install to your home screen

On your phone, open the app's URL in **Safari** (iOS) or **Chrome** (Android), then:

- **iOS:** Share → *Add to Home Screen*
- **Android:** ⋮ menu → *Install app*

It launches fullscreen, with no browser chrome. It's your assistant, running on your own hardware, in your pocket.

### Building your own client

The entire HTTP surface is three endpoints — an SSE render stream, one POST that carries any of the 121 Requests, and a file route. There is no second API and no database access. [`docs/HTTP_PROTOCOL.md`](docs/HTTP_PROTOCOL.md) documents all of it, and [`docs/http_reference_client.html`](docs/http_reference_client.html) is a working client to check yours against.

---

## Self-Evolution

Ask for something it can't do, and it builds it:

```
you › every morning at 7, check my sync folders for files that changed
      overnight and send me a summary on Telegram
```

The agent reads the SDK, writes a task plugin into its own workspace tree, validates it against the contract, fixes anything the validator flags, registers it, schedules it through the Timekeeper, and confirms. The plugin persists. Tomorrow at 7am it just runs.

Five plugin families, each a single file with a naming convention:

| Family | Prefix | What it is |
|---|---|---|
| **Tools** | `tool_*.py` | LLM-callable actions |
| **Tasks** | `task_*.py` | Pipeline and event-driven background work |
| **Services** | `service_*.py` | Shared, stateful backends |
| **Commands** | `command_*.py` | User-facing slash commands |
| **Frontends** | `frontend_*.py` | Transports — Discord, HTTP, whatever you want |

Plus `parsers/`, `llm/`, and `scripts/`. Drop a file into the right folder and the plugin watcher picks it up live — no restart, ever.

Plugins can even declare their own system-prompt text, which costs zero context when the plugin isn't loaded.

**Write one yourself:** start with [`docs/SDK.md`](docs/SDK.md), then the matching file in [`templates/`](templates/). Contributions go to the [`store`](../../tree/store) branch as a pull request.

---

## Security

The threat model is honest: **an agent with good intentions and no judgement.**

- **Validation** is an AST pass that never imports the file, so checking code can't run it. It catches direct effects and contract violations, and every error names the Request you should have used instead. It's a conformance linter, and [the source says so out loud](sandbox/validator.py) — static analysis of Python is defeatable by anyone trying, which is exactly why the subprocess exists.
- **Isolation** follows provenance. Agent-written workspace code is *always* subprocessed. Installed code is subprocessed when its imports or declarations require it.
- **Policy** decides each Request: allowed, refused, or shown to you. Secrets raise the level. Attendance matters — if nobody is watching a session, an unsafe Request is refused rather than silently approved.
- **Foreign libraries are the honest limit.** A binary wheel's internal I/O can't be turned into Requests, so it's isolated and *disclosed*, not pretended away.

Treat installing an extension as a capability change, not a file copy.

📖 [The Security Contract](docs/The%20Second%20Brain%20Security%20Contract.md) · [Permissions Map](docs/PERMISSIONS_MAP.md) · [Request Catalogue](docs/SECURITY_CONTRACT_APPENDIX.md)

---

## Documentation

| Doc | What's in it |
|---|---|
| [`docs/SDK.md`](docs/SDK.md) | The Request vocabulary and return idiom — start here to write a plugin |
| [`docs/HTTP_PROTOCOL.md`](docs/HTTP_PROTOCOL.md) | The complete client-facing API surface |
| [`docs/PERMISSIONS_MAP.md`](docs/PERMISSIONS_MAP.md) | Scope, isolation, approval modes, standing permissions |
| [`docs/SECURITY_CONTRACT_APPENDIX.md`](docs/SECURITY_CONTRACT_APPENDIX.md) | Every Request, classified |
| [`docs/MIGRATING_PLUGINS.md`](docs/MIGRATING_PLUGINS.md) | Converting a native plugin to the sandbox |
| [`templates/`](templates/) | A commented starting point for each plugin family |

Or just ask it. Install `read_file` and Second Brain will explain its own source to you.

---

## Contributing

The package store is a git branch, so shipping a plugin is a pull request against [`store`](../../tree/store). Write it as a sandbox plugin, validate it, declare `dependencies_files` and `dependencies_pip`, and open the PR.

Ideas, bug reports, and "I wish it could ___" are all welcome — open an issue, or email **henrydaum8609@gmail.com**.

---

## Philosophy

Second Brain ships as an unfinished product: a tiny, pure-Python kernel with a boundary around it. The brain is the LLM, the plugins are the body, the state machine is the spinal cord, and the sandbox is the immune system.

The value of AI isn't any single feature — it's that it can be built into almost anything. The role of the person is to guide it somewhere productive. So the kernel stays small, the contract stays strict, and what it becomes is up to you.

---

## Sponsor

<div align="center">
  <img src="https://github.com/user-attachments/assets/9e7ff971-8159-4081-b8bc-9b9ff5edd4ff#gh-light-mode-only" width="380" alt="Atlas Cloud">
  <img src="https://github.com/user-attachments/assets/8497513e-09a4-4151-8b8d-ed8be782a389#gh-dark-mode-only" width="380" alt="Atlas Cloud">
</div>

[**Atlas Cloud**](https://www.atlascloud.ai/?utm_source=github&utm_medium=link&utm_campaign=second-brain) is a full-modal AI inference platform — one API for video, image, and LLM generation across 300+ curated models, instead of managing a vendor integration per modality. Their [coding plan](https://www.atlascloud.ai/console/coding-plan) is a cheap, capable default for running Second Brain.

---

## License

MIT — do what you want with it.

<p align="center"><sub>An agent by <a href="https://github.com/henrydaum">Henry Daum</a></sub></p>
