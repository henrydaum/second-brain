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

Second Brain runs on your own machine. You need [Python 3.11+](https://www.python.org/downloads/), [git](https://git-scm.com/downloads), and [Node 20.19+ or 22.12+](https://nodejs.org/) for the web UI.

## The easy way: let your AI agent do it

Paste this into Claude Code, Codex, Cursor, or any coding agent that can run commands on your computer:

```text
Install Second Brain (https://github.com/henrydaum/second-brain) on this computer for me.

1. Check that Python 3.11+, git, and Node.js (20.19+ or 22.12+, which includes npm) are installed. Install anything missing with the normal installer for my operating system, and tell me before you do.
2. Clone the repository into my home folder (or ask me where I'd like it).
3. Inside the clone, create a virtual environment named .venv and run `pip install -r requirements.txt` with that environment's pip.
4. In the clone's frame_ui folder, run `npm install`.
5. Don't start Second Brain yourself. It's an interactive program that needs my terminal. Instead, give me the exact commands for my operating system to activate the virtual environment and run `python main.py`.
6. Tell me that once it's running, the web UI opens at http://localhost:5174, and the next step is to type /setup to connect an AI model.

Don't change any of Second Brain's settings or files beyond these steps. If something fails, show me the error and explain it plainly.
```

## Or do it yourself

```bash
git clone https://github.com/henrydaum/second-brain
cd second-brain
python -m venv .venv
```

Activate the virtual environment. On **Windows**, run `.venv\Scripts\activate`. On **macOS / Linux**, run `source .venv/bin/activate`. Then:

```bash
pip install -r requirements.txt
cd frame_ui
npm install
cd ..
python main.py
```

Second Brain starts the web UI for you. After a few seconds it tells you where to find it, usually **http://localhost:5174**. The terminal it runs in works as a chat too. Leave it open: the UI needs Second Brain running.

## Connect a model

This is the only step that really matters. In the web UI or the terminal, run:

```
/setup
```

It installs the `essentials` bundle (file tools, shell, web search, subagents and more) and asks how you want to connect a model:

| Option | What you need |
|---|---|
| **ChatGPT account (Codex)** | A ChatGPT plan, and no API key. In ChatGPT, go to **Settings → Security** and turn on **device code authorization for Codex**. Then run `/codex` and choose **Sign in**. |
| **OpenRouter** | An API key from [openrouter.ai](https://openrouter.ai/settings/keys). One key gets you hundreds of models. |
| **Another provider** | OpenAI, Anthropic, Gemini, a local model via Ollama, or any OpenAI-compatible endpoint. |

Say hello. **You're done.** Everything else is a question you can ask Second Brain itself, like *"how do I use you from my phone?"*, *"index my Documents folder"*, *"set up Telegram"* or *"what can you do?"*.

## Going further

- **More capabilities:** run `/packages install` (or open **Settings** in the web UI) to browse the store. Good ones to start with are `bundle_knowledgebase` (search your own PDFs, Office files, images and audio; a large download), `bundle_memory` (memory that maintains itself, stored as markdown you can edit), and `bundle_gmail`.
- **Your phone:** add the web UI to your home screen over [Tailscale](https://tailscale.com/). Prefer a chat app? `/packages install frontend_telegram`. Ask Second Brain how to set up either.
- **Updating:** run `/update`. It pulls the repo and the UI together.
- **Docker:** there's a Dockerfile for servers and NAS boxes. See [docs/DOCKER.md](docs/DOCKER.md).
- **No web UI?** Check `web_ui.log` in your data directory (see below). The usual cause is a skipped `npm install`.

---

# Where things live

Second Brain lives in two folders.

**The kernel** is the folder you cloned (`second-brain/`, wherever you ran `git clone`). It holds the app's code, the web UI in `frame_ui/`, and the `.venv` with every Python library Second Brain or its packages installed.

**The data directory** (`DATA_DIR`) is created on first run. Your config, database, conversations, installed packages, and the agent's workspace all live there:

| | |
|---|---|
| **Windows** | `%LOCALAPPDATA%\Second Brain\` |
| **macOS** | `~/Library/Application Support/Second Brain/` |
| **Linux** | `~/.local/share/Second Brain/` |

Run `/locations` to see the paths on your machine.

**To uninstall Second Brain completely,** quit it, then delete both folders: the kernel folder and `DATA_DIR`. Second Brain installs nothing outside them, with one exception: if you set up the optional macOS deployment, run `sh frame_ui/deploy/macos/manage.sh uninstall` first. If you used Docker, remove the `sb-data` volume instead (`docker volume rm sb-data`). Deleting `DATA_DIR` erases your conversations and settings, so copy anything you want to keep first.

## Philosophy

Second Brain is inspired by the human brain. Explorations into neurons turned into the creation of artificial neural networks, which then paved the way for attention mechanisms and transformers. From there came LLMs, and then came the agentic abilities: RAG, tool calls, and cron jobs. With each iteration, Second Brain became closer to its biological inspiration.

Second Brain is still pretty far from the real brain, in many ways. However, it can also do many things better than the human brain ever could. Building it has helped me to better understand the role of AI in my life, and in society. I found the process of building to be extremely valuable, because I realized that the value of AI is that it can be built into so many things. The role of the person is to guide it into productive and creative areas.

## License

MIT

---

An agent by Henry Daum
