"""Slash command plugin for `/setup` — onboarding ramp.

Three phases, all in one pass, aimed at the fastest route to a working
Second Brain. Everything after that is a question the agent can answer.
  1. Packages — a fresh kernel ships no LLM backend, so setup leads by
     installing the `essentials` bundle. Skipped automatically once an LLM
     backend is already installed.
  2. LLM — a ChatGPT account through Codex (no API key), an OpenRouter key
     (one key, most models), or any other provider via the LiteLLM backend.
  3. Telegram — configure the bot, but only when the Telegram frontend is
     (being) installed.

There is no web-UI phase. The HTTP frontend is on by default and the kernel
starts `frame_ui/` itself, so the only step left is an `npm install` that
belongs in the install instructions rather than in a wizard running inside
the thing it would be installing.

Codex sign-in is deliberately *not* run from here. It is a device-code wait of
up to several minutes owned by `/codex`, a command this wizard has only just
installed; nesting it would put that wait, a second approval and a hot-reload
race inside onboarding. So setup installs Codex and names the one command left.
"""


from guest.bases import BaseCommand
from guest.forms import FormStep


# The name a profile falls back to when discovery has not run or has found
# nothing. Deliberately the *retired* spelling: the live backend declares
# ``replaces = ["LiteLLMService"]``, so this still resolves through the alias
# map, whereas guessing the current class name would break the moment the
# backend is renamed again. Anything actually installed is preferred over it.
DEFAULT_BACKEND = "LiteLLMService"

ESSENTIALS_BUNDLE = "bundle_essentials"
KNOWLEDGEBASE_BUNDLE = "bundle_knowledgebase"
CODEX_BUNDLE = "bundle_codex"
#: What the Codex backend calls itself, which is how "is Codex installed" is
#: answered — a bundle leaves no receipt of its own, its files do.
CODEX_BACKEND = "CodexBackend"
TELEGRAM_PACKAGE = "frontend_telegram"
#: What each install choice actually installs, in order. A *list* rather than
#: one name because the second choice is the first plus one — the knowledge
#: base is what you add to a working instance, not an alternative to it, so
#: offering it alone would be offering a broken install.
BUNDLE_CHOICES = {
    ESSENTIALS_BUNDLE: [ESSENTIALS_BUNDLE],
    "essentials_and_knowledgebase": [ESSENTIALS_BUNDLE, KNOWLEDGEBASE_BUNDLE],
}

OPENROUTER_KEYS_URL = "https://openrouter.ai/settings/keys"
OPENROUTER_MODELS_URL = "https://openrouter.ai/models"

WELCOME_PROMPT = (
    "Welcome to Second Brain.\n\n"
    "The kernel ships almost nothing on its own — capabilities are installed from a "
    "package store. The `essentials` bundle is the recommended first install: an LLM "
    "backend, file read/edit/search, shell and script running, SQL, web search, "
    "subagents, and the Telegram frontend. Adding `knowledgebase` indexes your files "
    "and makes them searchable (a much larger download — fine to add later).\n\n"
    "You can browse and install more anytime with /packages."
)

LLM_INTRO_PROMPT = (
    "Now connect a model. This is the only step that really matters — once a "
    "model answers, you can ask Second Brain itself about everything else.\n\n"
    "  • ChatGPT account (Codex) — no API key; uses your ChatGPT plan. The "
    "fastest route if you already pay for ChatGPT.\n"
    "  • OpenRouter — one API key, hundreds of models, pay as you go.\n"
    "  • Another provider — OpenAI, Anthropic, Gemini, a local model, any "
    "OpenAI-compatible endpoint."
)

OPENROUTER_KEY_PROMPT = (
    f"Create a key at {OPENROUTER_KEYS_URL} and paste it here. "
    "(The name of an environment variable holding it works too.)"
)
OPENROUTER_MODEL_PROMPT = (
    f"Which model? Copy its id from {OPENROUTER_MODELS_URL} — for example "
    "`openai/gpt-5` or `anthropic/claude-sonnet-4.5`. You can add more later "
    "with /llm."
)

OTHER_MODEL_PROMPT = (
    "Enter the LiteLLM model name, including the provider prefix when needed. "
    "Examples: `openai/gpt-5`, `anthropic/claude-sonnet-4-5`, "
    "`gemini/gemini-2.5-pro`, `ollama/qwen3`. For an OpenAI-compatible endpoint "
    "(set the base URL below), a plain id like `deepseek-ai/deepseek-v4-pro` is "
    "auto-routed through the openai provider."
)
OTHER_SERVICE_PROMPT = (
    "How should Second Brain connect to this model?\n\n"
    "Installed LLM backends are normal service plugins."
)
OTHER_ENDPOINT_PROMPT = (
    "Optional provider base URL or LiteLLM proxy URL. Leave blank for the provider default. "
    "For local models or self-hosted gateways, paste the full base URL."
)
OTHER_KEY_PROMPT = (
    "API key. You can paste the key directly, enter the name of an environment variable that holds it, or leave it blank to let the backend read its own environment."
)
OTHER_CONTEXT_PROMPT = (
    "Context window size in tokens. Use 0 if you don't know — Second Brain will still work, it just won't proactively compact."
)

TELEGRAM_PROMPT = (
    "Optional: Telegram. Chat with Second Brain from your phone — push "
    "notifications, attachments, inline buttons.\n\n"
    "You'll need:\n"
    "  1. A bot token from @BotFather on Telegram (https://t.me/BotFather → /newbot)\n"
    "  2. Your Telegram user ID — message @userinfobot and it will reply with your numeric ID"
)
TELEGRAM_TOKEN_PROMPT = "Paste the bot token from @BotFather."
TELEGRAM_USER_PROMPT = (
    "Enter your Telegram user ID (a number from @userinfobot). Only this user will be allowed to talk to the bot."
)

CODEX_NEXT_STEPS = (
    "One step left — sign in:\n"
    "  1. In ChatGPT (chatgpt.com), open Settings → Security and turn on "
    "device code authorization for Codex.\n"
    "  2. Run `/codex` and choose **Sign in**. It shows a code to enter at "
    "auth.openai.com, then creates your model profile.\n"
    "  If you already had a default model, use /llm to switch to the Codex one."
)

PACKAGES_SECTION = (
    "Get more with /packages:\n"
    "  /packages install          — browse by category and pick a package\n"
    "  /packages install <id>     — install a package or bundle by name\n"
    f"  Good next steps: `{KNOWLEDGEBASE_BUNDLE}` (index and search your own "
    "files), `bundle_memory` (durable memory), `bundle_gmail` (email)."
)


class SetupCommand(BaseCommand):
    """Slash-command handler for `/setup`."""
    name = "setup"
    description = "Onboarding: install a starter bundle and connect a model"
    category = "System"
    # No per-action split is available — the wizard has no ``action``
    # argument, and every route through it installs packages or writes
    # settings. So it asks once, at the door, naming the whole grant. That is
    # also the honest shape for onboarding: the user is being told what the
    # wizard is about to do before it starts, not interrupted halfway.
    require_approval = True
    approval_actor_id = "user"
    requests = [
        "plugin.list", "plugin.install", "config.read", "config.write",
        "paths.get", "net.http", "llm.list",
    ]

    def form(self, sdk, args):
        """Build the dynamic onboarding form."""
        steps = []
        backends = _llm_backends(sdk)
        backend_ready = bool(backends)

        # Phase 1 — packages. Only lead with this when there's no LLM backend yet
        # (a fresh install). A returning user skips straight to reconfiguring.
        if not backend_ready:
            steps.append(FormStep(
                "install_choice", WELCOME_PROMPT, True,
                enum=[ESSENTIALS_BUNDLE, "essentials_and_knowledgebase",
                      "skip"],
                enum_labels=[
                    "Install the essentials bundle (recommended)",
                    "Essentials + knowledge base (indexes your files — much larger download)",
                    "Skip — I'll use /packages myself",
                ],
                columns=1,
            ))
            choice = args.get("install_choice")
            if not choice or choice == "skip":
                return steps
            # Both choices include the LiteLLM backend + Telegram frontend.
            will_have_telegram = True
        else:
            will_have_telegram = _package_installed(sdk, TELEGRAM_PACKAGE)

        # Phase 2 — LLM.
        steps.append(FormStep(
            "llm_choice", LLM_INTRO_PROMPT, True,
            enum=["codex", "openrouter", "other"],
            enum_labels=["Use my ChatGPT account (Codex)",
                         "Use an OpenRouter API key",
                         "Use another provider"],
            columns=1,
        ))
        llm_choice = args.get("llm_choice")
        if llm_choice == "openrouter":
            steps.extend(self._openrouter_steps())
        elif llm_choice == "other":
            steps.extend(self._other_steps(backends))

        # Phase 3 — Telegram, once the LLM branch is satisfied and the frontend
        # is (being) installed.
        if will_have_telegram and _llm_steps_complete(args, llm_choice):
            steps.extend(self._telegram_steps(args))
        return steps

    def _openrouter_steps(self):
        """OpenRouter key and model. Two questions, because that is all a
        LiteLLM profile for a provider it knows needs — no endpoint to guess."""
        return [
            FormStep("openrouter_api_key", OPENROUTER_KEY_PROMPT, True),
            FormStep("openrouter_model", OPENROUTER_MODEL_PROMPT, True),
        ]

    def _other_steps(self, backends):
        """Generic LLM profile collection (mirrors /llm add)."""
        backends = backends or [(DEFAULT_BACKEND, DEFAULT_BACKEND)]
        names = [name for name, _label in backends]
        return [
            FormStep("other_model_name", OTHER_MODEL_PROMPT, True),
            FormStep("other_service_class", OTHER_SERVICE_PROMPT, True,
                     enum=names, default=names[0], columns=1,
                     enum_labels=[label for _name, label in backends]),
            FormStep("other_endpoint", OTHER_ENDPOINT_PROMPT, False, default="", prompt_when_missing=True),
            FormStep("other_api_key", OTHER_KEY_PROMPT, False, default="", prompt_when_missing=True),
            FormStep("other_context_size", OTHER_CONTEXT_PROMPT, False, "integer", default=0, prompt_when_missing=True),
        ]

    def _telegram_steps(self, args):
        """Telegram bot credential collection."""
        steps = [FormStep(
            "telegram_choice", TELEGRAM_PROMPT, True,
            enum=["setup", "skip"],
            enum_labels=["Set up Telegram", "Skip — not now"],
            columns=1,
        )]
        if args.get("telegram_choice") == "setup":
            steps.append(FormStep("telegram_bot_token", TELEGRAM_TOKEN_PROMPT, True))
            steps.append(FormStep("telegram_allowed_user_id", TELEGRAM_USER_PROMPT, True, "integer"))
        return steps

    def run(self, sdk, args):
        """Execute `/setup` for the active session."""
        install_choice = args.get("install_choice")
        if install_choice == "skip":
            return self._skip_section()

        sections = []
        llm_choice = args.get("llm_choice")

        # Phase 1 — install the chosen bundles before configuring anything
        # that depends on them. Codex rides along here when it was picked, so
        # one connectivity check and one report cover every download. Bail
        # clearly on failure, so we don't pretend a half-set-up instance is
        # ready.
        bundles = list(BUNDLE_CHOICES.get(install_choice, ()))
        if llm_choice == "codex" and not _codex_installed(sdk):
            bundles.append(CODEX_BUNDLE)
        if bundles and not _has_internet(sdk):
            return (
                "No internet connection detected. Setup needs to download "
                "packages and their dependencies. Connect to the internet and "
                "run /setup again."
            )
        for bundle in bundles:
            try:
                result = sdk.plugins.install(bundle)
            except sdk.Failed as e:
                # Reported rather than raised past the remaining bundles: the
                # essentials install is what everything else depends on, so a
                # later failure must not lose the report of the one that worked.
                return "\n\n".join(sections + [
                    f"Couldn't install the `{bundle}` bundle: {e.error}\n\n"
                    f"Resolve the issue (or try `/packages install {bundle}`), "
                    "then re-run /setup."])
            sections.append(f"Installed the `{bundle}` bundle.\n"
                            + _indent(result))

        # Phase 2 — LLM.
        if llm_choice == "codex":
            sections.append("LLM: Codex is installed.\n" + _indent(CODEX_NEXT_STEPS))
        elif llm_choice == "openrouter":
            sections.append(self._save_openrouter(sdk, args))
        elif llm_choice == "other":
            result = self._save_other(sdk, args)
            if result is None:
                return "Model name is required."
            sections.append(result)

        # Phase 3 — Telegram.
        if args.get("telegram_choice") == "setup":
            sections.append(self._save_telegram(sdk, args))
        elif args.get("telegram_choice") == "skip":
            sections.append("Telegram: skipped. Run /setup again whenever you want it.")

        sections.append(PACKAGES_SECTION)
        sections.append(self._location_section(sdk))
        sections.append(self._hint_section(sdk, llm_choice))
        return "\n\n".join(s for s in sections if s)

    # ──────────────────────────────────────────────────────────────────
    # Persistence helpers
    # ──────────────────────────────────────────────────────────────────

    def _save_openrouter(self, sdk, args):
        """Persist an OpenRouter profile through the LiteLLM backend."""
        key = (args.get("openrouter_api_key") or "").strip()
        model = (args.get("openrouter_model") or "").strip()
        # The ``openrouter/`` prefix is how LiteLLM picks the provider, and it
        # is not part of the id the OpenRouter site shows — so it is added
        # here rather than asked for.
        if not model.startswith("openrouter/"):
            model = "openrouter/" + model
        _install_llm_profile(sdk, model, {
            "llm_endpoint": "",
            "secret_llm_api_key": key,
            "llm_context_size": 0,
            "llm_service_class": DEFAULT_BACKEND,
        })
        return (
            f"LLM: OpenRouter set up. Default profile: {model}\n"
            "  Use /llm to edit the profile or add more models."
        )

    def _save_other(self, sdk, args):
        """Persist a generic LLM profile. None when no model was named."""
        name = (args.get("other_model_name") or "").strip()
        if not name:
            return None
        profile = {
            "llm_endpoint": (args.get("other_endpoint") or "").strip(),
            "secret_llm_api_key": (args.get("other_api_key") or "").strip(),
            "llm_context_size": int(args.get("other_context_size") or 0),
            "llm_service_class": (args.get("other_service_class") or DEFAULT_BACKEND).strip() or DEFAULT_BACKEND,
        }
        _install_llm_profile(sdk, name, profile)
        endpoint = profile["llm_endpoint"] or "(provider default)"
        return (
            f"LLM: profile `{name}` added and set as default.\n"
            f"  Service class: {profile['llm_service_class']}\n"
            f"  Endpoint: {endpoint}\n"
            "  Use /llm to edit or add more models."
        )

    def _save_telegram(self, sdk, args):
        """Persist Telegram credentials into plugin_config."""
        token = (args.get("telegram_bot_token") or "").strip()
        user_id = int(args.get("telegram_allowed_user_id") or 0)
        sdk.config.write(
            "telegram_bot_token", token, scope="plugin")
        sdk.config.write(
            "telegram_allowed_user_id", user_id, scope="plugin")
        return (
            f"Telegram: configured for user {user_id}.\n"
            "  Restart Second Brain to bring the bot online, then send /start to your bot in Telegram."
        )

    def _skip_section(self):
        """Guidance when the user declines the starter install."""
        return (
            "Skipped package install.\n\n"
            "Second Brain needs at least an LLM backend before it can do anything. "
            "When you're ready:\n"
            f"  /packages install {ESSENTIALS_BUNDLE}      — the recommended baseline\n"
            f"  /packages install {KNOWLEDGEBASE_BUNDLE}   — then this, to index and search your files\n"
            "  /packages install               — browse the store by category\n\n"
            "Then run /setup again to connect a model."
        )

    def _location_section(self, sdk):
        """One-paragraph summary of where things live on disk."""
        return (
            "Files & data:\n"
            f"  DATA_DIR: {sdk.paths.get('data')}\n"
            "  Holds your config, the SQLite database, installed packages, and "
            "anything the agent writes for itself. /locations shows the rest."
        )

    def _hint_section(self, sdk, llm_choice):
        """Closing hint about how to continue."""
        first = ("Sign in with /codex, then run /new"
                 if llm_choice == "codex" else "Run /new")
        ui = str(sdk.config.read("ui_url") or "").strip()
        where = f" The web UI is at {ui}." if ui else ""
        return (
            f"You're ready. {first} and just ask — how Second Brain works, what "
            f"it can do, how to add Telegram, memory, or a schedule.{where}"
        )


def _llm_steps_complete(args, choice):
    """Return True once the LLM branch has collected enough to move on to Telegram."""
    if choice == "codex":
        return True
    if choice == "openrouter":
        return bool(args.get("openrouter_api_key") and args.get("openrouter_model"))
    if choice == "other":
        return bool(args.get("other_model_name") and args.get("other_service_class"))
    return False


def _install_llm_profile(sdk, name, profile):
    """Register a new LLM profile, set it as default, hot-load it, and persist."""
    sdk.config.write(
        "llm_profiles", {name: profile}, merge=True, scope="plugin")
    sdk.config.write(
        "default_llm_profile", name, scope="plugin")


def _package_installed(sdk, package_id):
    """Whether a package id has an install receipt."""
    try:
        return any(
            p.get("id") == package_id
            for p in sdk.plugins.list(source="installed")
        )
    except sdk.Failed:
        return False


def _codex_installed(sdk):
    """Whether the Codex backend is already present, so setup can skip the
    download on a second run."""
    return any(name == CODEX_BACKEND for name, _label in _llm_backends(sdk))


def _has_internet(sdk) -> bool:
    """Best-effort connectivity check before a package download."""
    try:
        sdk.net.http("https://github.com", method="HEAD")
        return True
    except sdk.Failed:
        return False


def _llm_backends(sdk):
    """Installed LLM backends as ``(class_name, label)`` pairs.

    The class name is what a profile stores; the label is what the file
    declares in ``display_name`` and is the only one of the two worth showing
    a person. Nothing read that declaration before, so every picker in the app
    offered raw class names.
    """
    try:
        return [(entry["name"], entry.get("display_name") or entry["name"])
                for entry in (sdk.llm.list() or {}).get("backends") or []]
    except sdk.Failed:
        return []


def _indent(text: str) -> str:
    """Indent a block two spaces for nesting under a section header."""
    return "\n".join(f"  {line}" if line else line for line in (text or "").splitlines())
