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

# Head-to-Head Evaluation

Second Brain performs higher than OpenClaw and Hermes on [harness-bench](https://www.harness-bench.ai/), which measures "model-harness configuration effects across 106 sandboxed offline agent tasks". Here are the results:
<img width="1024" height="490" alt="Captura de pantalla 2026-08-23 213236" src="https://github.com/user-attachments/assets/0eee51f5-fccc-4d43-acc5-fd709bf50499" />

The full evaluation framework I used is [available on GitHub](https://github.com/henrydaum/second-brain-evals), and the full results are available [here](https://github.com/henrydaum/second-brain-eval-results). I did this testing because I was curious how Second Brain would stack up against other agents. This provides the real results, but with a few small caveats, which you can read in the links provided.

# Using Second Brain to play DOOM

https://github.com/user-attachments/assets/6c740d80-0830-4703-849a-8f00fab9e865

Second Brain has `widgets`, which are essentially like Anthropic Artifacts. Widgets are HTML files that the agent can write. They run in a sandboxed iframe for security. They can even run DOOM!

# How it looks

<img width="2560" height="1326" alt="Captura de pantalla 2026-09-10 202738" src="https://github.com/user-attachments/assets/cd23560e-717c-4727-aa46-33bcb3ef9f70" />

<img width="2560" height="1330" alt="Captura de pantalla 2026-09-10 202430" src="https://github.com/user-attachments/assets/1466ada6-0b2e-4563-b195-feb9caf21981" />

# Install

Second Brain runs on your own machine. Two things to install: **the app** (required, ~2 minutes) and **the UI** (optional, a ChatGPT-style web app you can add to your phone's home screen).

## 1. Install the app

You need [Python 3.11+](https://www.python.org/downloads/) and [git](https://git-scm.com/downloads). If you want the web UI as well, you need [Node 20.19+ or 22.12+](https://nodejs.org/) — it ships with `npm`, and there is nothing to add to `requirements.txt`, which is pip's and stays kernel-minimal.

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

The UI lives in this repo, in **`frame_ui/`** — a React app on [assistant-ui](https://www.assistant-ui.com/) with conversations, attachments, settings, notifications and widget panels. It used to be a second repository you cloned separately; it is not any more, so a pull of Second Brain is a pull of the UI and the two halves of the protocol cannot drift apart.

Works on Windows, macOS and Linux. Takes about two minutes.

**Before you start:** Second Brain has to be *running* while you use the UI — the UI is just a face for it. Leave it going in its terminal and open a **second terminal** for everything below. You'll also need [Node 20.19+ or 22.12+](https://nodejs.org/); `npm` comes with it.

### 1. Turn the HTTP frontend on

If you said yes to the web UI during `/setup`, this is already done — skip to step 2. Otherwise, in the Second Brain REPL:

```
/frontends enable http
/restart
```

**There is no token to set.** Second Brain mints `secret_http_token` at boot and the UI reads it out of `config.json` itself, so nothing gets copied anywhere.

### 2. Install the UI's dependencies

Once, in your second terminal:

```bash
cd frame_ui
npm install
```

(`frame_ui` is in your Second Brain folder — the same one you cloned to run the server. There is no `.env` file to make.)

From then on **Second Brain starts the UI itself** and tells you where it is:

```
UI is reachable at: http://localhost:5174
```

### 3. Open it

Open that address. You should see the thread name, `ok` beside **Request** and `open` beside **Stream** — that is the whole bridge working.

No notification at all means the UI never came up: `web_ui.log` in your data directory is the dev server's own output, and is where that explains itself — most often a missing `npm install`. A `401` on the page means the dev server started before the token existed; `/restart`, since it reads `config.json` once, at startup.

Prefer to run it yourself? Set `ui_autostart` to off in `/config` and use `npm run dev` as before. Anything already serving is left alone either way, so a dev server you keep running is adopted rather than duplicated.

Keeping it current is `/update`: it pulls the repo (the UI comes with it), runs `npm install`, and rebuilds and reactivates the deployment on platforms that have one.

### Reaching it from another machine

One setting, in the REPL:

```
/config   →  http_client_url = http://my-box.tail1234.ts.net:8787
```

`/restart` and that's the whole change. The browser never sees that value — it only ever talks to the dev server, which proxies onward and adds the credential itself — so nothing about CORS or tokens moves with it.

### Put it on your phone

Reaching it from your phone means the dev server has to accept connections from other devices. Turn `ui_autostart` off and run it yourself with the flag that does that:

```bash
cd frame_ui
npm run dev -- --host
```

That prints a second URL (a `192.168.x.x` address). To reach it from anywhere rather than just your home Wi-Fi, install [Tailscale](https://tailscale.com/) on both your computer and your phone, and use your machine's Tailscale address instead. Note this exposes the *dev server*, which holds the credential and adds it for whoever connects — fine on a tailnet, not something to put on a café Wi-Fi.

Open that URL in your phone's browser, then add it to your home screen — on iPhone, press the three dots, then **Share**, and scroll down to **Add to Home Screen**. Click it, and you're done. It's like a real app from there.

### Why the dev server?

Because it's the only thing that works on every platform today. `npm run build` produces a `dist` folder that `frontend_http` can serve directly (the `http_static_dir` setting), but a build served that way has to get its token from somewhere, and putting it in the bundle hands the credential to every browser that loads the page. Serving a build properly needs a reverse proxy that adds the token on its own hop — which today exists for macOS only, as a Caddy gateway: [frame_ui/docs/MACOS_DEPLOYMENT.md](frame_ui/docs/MACOS_DEPLOYMENT.md), with the scripts in `frame_ui/deploy/macos/`. The dev server is perfectly fine for personal use.

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
