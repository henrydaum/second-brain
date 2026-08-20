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

**There's also a Dockerfile**, if you'd rather not install anything:

```bash
docker build -t second-brain .
docker run --rm -it --init -v sb-data:/data second-brain
```

Same REPL, with everything kept in the `sb-data` volume. The path above is
still the better one for day-to-day use — Second Brain is an assistant for
*your machine*, and a container starts out unable to see it. Reach for this
when the machine isn't one you want to install on: a server or a NAS, or a
reproducible Linux to test against. [docs/DOCKER.md](docs/DOCKER.md) covers it,
including Docker itself if this is your first time.

## 2. Run `/setup`

```
/setup
```

The wizard walks you through everything in one pass:

1. **Install the `essentials` bundle** — an LLM backend, file read/edit/search, shell and script running, SQL, web search, subagents, and the Telegram frontend.
2. **Connect a model** — paste an API key. [Atlas Cloud](https://www.atlascloud.ai/console/coding-plan) is the sponsored fast path (300+ models behind one key), but any provider works.
3. **Telegram (optional)** — chat with your Second Brain from your phone. Needs a bot token from [@BotFather](https://t.me/BotFather) and your user ID from [@userinfobot](https://t.me/userinfobot).
4. **Web UI (optional)** — installs the HTTP frontend and generates your API token, then prints the exact steps to set the app up. [More below.](#install-the-ui)

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

A modern React frontend built on [assistant-ui](https://www.assistant-ui.com/) — a familiar chat interface you can add to your phone's home screen. It lives in its own repo: **[second-brain-ui](https://github.com/henrydaum/second-brain-ui)**.

Works on Windows, macOS and Linux. Takes about ten minutes.

**Before you start:** Second Brain has to be *running* while you use the UI — the UI is just a face for it. Leave it going in its terminal and open a **second terminal** for everything below. You'll also need [Node 20.19+ or 22.12+](https://nodejs.org/).

### 1. Get your API token

The UI proves it's allowed to talk to Second Brain with a token. You need one before anything else.

**If you said yes to the web UI during `/setup`,** you already have it — the wizard printed it. Copy it and go to step 2.

**Otherwise,** in the Second Brain REPL:

```
/packages install frontend_http
```

Then run `/config`, find **`secret_http_token`**, and set it to any long random string — mash the keyboard, it just has to be hard to guess. Copy what you set. Then restart Second Brain so the frontend comes online:

```
/restart
```

### 2. Set up the UI

In your second terminal:

```bash
git clone https://github.com/henrydaum/second-brain-ui
cd second-brain-ui
npm install
```

Now make your own config file from the example:

| | |
|---|---|
| **Windows** | `copy .env.example .env.local` |
| **macOS / Linux** | `cp .env.example .env.local` |

Open **`.env.local`** in any editor. Find the line that reads `VITE_SB_TOKEN=` and paste your token right after the `=`, with no quotes and no spaces:

```
VITE_SB_TOKEN=the-token-you-copied
```

Leave the other two lines (`VITE_SB_URL`, `VITE_SB_THREAD`) exactly as they are — the defaults are correct. Save and close.

### 3. Start it

```bash
npm run dev
```

Open **http://localhost:5173** in your browser. You should see your Second Brain, ready to chat.

If you get a blank screen or a `401`, the token in `.env.local` doesn't match the one in `/config` — that's almost always the problem. Fix it and restart `npm run dev`.

### Put it on your phone

Start the dev server so it accepts connections from other devices on your network:

```bash
npm run dev -- --host
```

That prints a second URL (a `192.168.x.x` address). To reach it from anywhere rather than just your home Wi-Fi, install [Tailscale](https://tailscale.com/) on both your computer and your phone, and use your machine's Tailscale address instead.

Open that URL in your phone's browser, then add it to your home screen — on iPhone, press the three dots, then **Share**, and scroll down to **Add to Home Screen**. Click it, and you're done. It's like a real app from there.

### Why the dev server?

Because it's the only thing that works on every platform today. `npm run build` produces a `dist` folder that `frontend_http` can serve directly (the `http_static_dir` setting), but a production build deliberately ships **no API token** — so a browser pointed at it is refused before the page even loads. Serving a build needs a reverse proxy that adds the token, which today exists only in the UI repo's [macOS deployment](https://github.com/henrydaum/second-brain-ui/blob/main/docs/MACOS_DEPLOYMENT.md). The dev server is perfectly fine for personal use.

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
