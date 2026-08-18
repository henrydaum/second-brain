<img width="1440" height="569" alt="highreslogotypecrop" src="https://github.com/user-attachments/assets/598ab57f-ed6b-491a-9cd6-142b93b09244" />

[Quick install](#install)

# Sponsor
<div align="center">
  <img src="https://github.com/user-attachments/assets/9e7ff971-8159-4081-b8bc-9b9ff5edd4ff#gh-light-mode-only" width="500" alt="Atlas Cloud Logo">
  <img src="https://github.com/user-attachments/assets/8497513e-09a4-4151-8b8d-ed8be782a389#gh-dark-mode-only" width="500" alt="Atlas Cloud Logo">
</div>

---

[Atlas Cloud](https://www.atlascloud.ai/?utm_source=github&utm_medium=link&utm_campaign=second-brain) is a full-modal AI inference platform that gives developers a single AI API to access video generation, image generation, and LLM APIs. Instead of managing multiple vendor integrations, you connect once and get unified access to 300+ curated models across all modalities.
Check out Atlas Cloud's new coding plan promotion for more budget-friendly API access: [https://www.atlascloud.ai/console/coding-plan](https://www.atlascloud.ai/console/coding-plan)

# Second Brain Example

*For an example of what Second Brain can do, visit https://second-brain.art! It's an interactive art exhibition.*

# How Second Brain works, in infographic form

## 1. Attachment Parsing
<img width="1035" height="664" alt="1  Attachment Parsing" src="https://github.com/user-attachments/assets/29d1563f-200e-4617-a2b2-c14bb3816da1" />

## 2. File Parsing
<img width="775" height="872" alt="2  File Parsing" src="https://github.com/user-attachments/assets/2703d99c-c504-479c-a51c-0b414a56225c" />

## 3. The LLM Loop
<img width="1080" height="870" alt="3  The LLM Loop" src="https://github.com/user-attachments/assets/2b10e1e4-51aa-4356-baf8-bf3497661b76" />

## 4. Path-Driven Tasks
<img width="628" height="662" alt="4  Path-Driven Tasks" src="https://github.com/user-attachments/assets/a41f0e5f-7df2-4f60-9ab6-86aa71f5fd26" />

## 5. Event-Driven Tasks
<img width="1248" height="688" alt="5  Event-Driven Tasks" src="https://github.com/user-attachments/assets/e24c0ed0-08fa-4813-beeb-e4260e720708" />

## 6. Conversation Runtime
<img width="944" height="715" alt="6  Conversation Runtime" src="https://github.com/user-attachments/assets/2840c28b-557e-47b2-9c30-5207453099e0" />

## 7. Frontends
<img width="757" height="544" alt="7  Frontends" src="https://github.com/user-attachments/assets/a72ecf8f-4579-42c2-acc7-3675f00d8325" />

## 8. Commands
<img width="1440" height="658" alt="8  Commands" src="https://github.com/user-attachments/assets/9faaa379-7da7-4973-8974-337fd0c21923" />

## 9. Plugins
<img width="533" height="874" alt="9  Plugins" src="https://github.com/user-attachments/assets/b8551628-f78a-4170-a92c-33e8be57a915" />

## 10. Sandbox & SDK
<img width="960" height="720" alt="10  Sandbox   SDK" src="https://github.com/user-attachments/assets/aed96e0d-e4ac-4d1e-9a62-625f2d5f2b6b" />

# Video demo

https://github.com/user-attachments/assets/26124782-12e3-41d1-8e8f-43256887acc3

# Install

Second Brain runs on your own machine. Two things to install: **the app** (required, ~2 minutes) and **the UI** (optional, a ChatGPT-style web app you can add to your phone's home screen).

## 1. Install the app

You need [Python 3.11+](https://www.python.org/downloads/) and [git](https://git-scm.com/downloads).

```bash
git clone https://github.com/henrydaum/second-brain
cd second-brain
python -m venv .venv
```

Activate the virtual environment:

| | |
|---|---|
| **Windows** | `.venv\Scripts\activate` |
| **macOS / Linux** | `source .venv/bin/activate` |

Then install and run:

```bash
pip install -r requirements.txt
python main.py
```

That's it — you're in the REPL. `requirements.txt` is nearly pure Python.

## 2. Run `/setup`

```
/setup
```

The wizard walks you through everything in one pass:

1. **Install the `essentials` bundle** — an LLM backend, file read/edit/search, shell and script running, SQL, web search, subagents, and the Telegram frontend.
2. **Connect a model** — paste an API key. [Atlas Cloud](https://www.atlascloud.ai/console/coding-plan) is the sponsored fast path (300+ models behind one key), but any provider works.
3. **Telegram (optional)** — chat with your Second Brain from your phone. Needs a bot token from [@BotFather](https://t.me/BotFather) and your user ID from [@userinfobot](https://t.me/userinfobot).
4. **Web UI (optional)** — installs the HTTP frontend and generates your API token, then prints the two commands that build the app. [More below.](#install-the-ui)

Say hello. You now have a working assistant. If you have questions, just ask.

## 3. Get out of the terminal

The REPL works, but it isn't where you want to live. Two much nicer options, in order of effort:

| | Effort | What it's like |
|---|---|---|
| **Telegram** | ~5 minutes — `/setup` shows the way | Push notifications, attachments, inline buttons, and available on all major platforms. |
| **Web UI** | ~10 minutes | A ChatGPT-style app you open in a browser or add to your phone's home screen. [Set it up below.](#install-the-ui) |

Skipped one during `/setup`? Just run `/setup` again.

## 4. Add more (optional)

A fresh install is deliberately small. Add capabilities whenever you want:

```
/packages install
```

On the Web UI, you can do this in Settings. That opens a picker — browse by category and choose. The bundles worth knowing:

| Bundle | What you get |
|---|---|
| `bundle_knowledgebase` | Index and search your own files — PDF, Office, images, audio, video, spreadsheets, archives. OCR, transcription, embeddings, and three search tools. **Large download.** |
| `bundle_memory` | Durable memory that maintains itself. Notes and skills are surfaced when relevant and written down in the background, as plain markdown you can edit. |
| `bundle_gmail` | Read, send, reply, label. |

Then tell it which folders to watch:

```
/config
```

Set **`sync_directories`** to the folders you want indexed. Expect a flood of task messages while the first sync runs — that's normal, and it stops when it finishes.

---

# Install the UI

A modern React frontend built on [assistant-ui](https://www.assistant-ui.com/) — familiar chat interface, installable on your phone as a PWA. It lives in its own repo: **[second-brain-ui](https://github.com/henrydaum/second-brain-ui)**.

Works on Windows, macOS and Linux.

### 1. Open the HTTP door in Second Brain

In the REPL:

```
/packages install frontend_http
```

Then run `/config` and set:

| Setting | Value |
|---|---|
| `secret_http_token` | Any long random string. Keep it handy — you paste it in step 2. |
| `http_port` | `8787` (the default is fine) |

Restart Second Brain. The frontend is enabled automatically when you install it.

### 2. Run the UI

You need [Node 20.19+ or 22.12+](https://nodejs.org/).

```bash
git clone https://github.com/henrydaum/second-brain-ui
cd second-brain-ui
npm install
cp .env.example .env.local
```

Open `.env.local` and paste your token from earlier into `VITE_SB_TOKEN`. Then:

```bash
npm run dev
```

Open **http://localhost:5173**. That's it — the UI proxies to Second Brain on its own origin, so CORS never enters the picture.

### On your phone

Run the dev server on your computer with `npm run dev -- --host`, then reach your machine over [Tailscale](https://tailscale.com/) and choose *Add to Home Screen* — on iPhone, press the three dots, then Share and scroll down to 'Add to home screen'. Click it, and you're done. It's like a real app from there.

### A note on production builds

`npm run build` produces a `dist` folder, and `frontend_http` can serve it directly via the `http_static_dir` setting — but **a production bundle deliberately contains no API token**, so a browser pointed straight at it gets `401` on every request, including the page itself. A production deployment needs a loopback reverse proxy that adds the bearer token upstream, which is what the UI repo's [macOS deployment](https://github.com/henrydaum/second-brain-ui/blob/main/docs/MACOS_DEPLOYMENT.md) sets up with Caddy. On Windows and Linux, use the dev server above.

---

# Where things live

Second Brain creates its data directory on first run. Config, database, installed packages, and the agent's workspace all live there:

| | |
|---|---|
| **Windows** | `%LOCALAPPDATA%\Second Brain\` |
| **macOS** | `~/Library/Application Support/Second Brain/` |
| **Linux** | `~/.local/share/Second Brain/` |

Run `/locations` to see the paths on your machine. Useful commands once you're up:

| Command | What it does |
|---|---|
| `/packages` | Install and remove capabilities |
| `/config` | Every setting, including plugin settings |
| `/llm` | Add, switch, or edit model profiles |
| `/conversations` | Browse and switch conversations |
| `/commands` | List everything available |

## Philosophy

Second Brain is inspired by the human brain. Explorations into neurons turned into the creation of artificial neural networks, which then paved the way for attention mechanisms and transformers. From there came LLMs, and then came the agentic abilities: RAG, tool calls, and cron jobs. With each iteration, Second Brain became closer to its biological inspiration.

Second Brain is still pretty far from the real brain, in many ways. However, it can also do many things better than the human brain ever could. Building it has helped me to better understand the role of AI in my life, and in society. I found the process of building to be extremely valuable, because I realized that the value of AI is that it can be built into so many things. The role of the person is to guide it into productive and creative areas.

## License

MIT

---

An agent by Henry Daum
