"""Slash command plugin for `/setup` — onboarding ramp.

Three phases, all in one pass:
  1. Packages — a fresh kernel ships no LLM backend or frontend, so setup leads
     by installing the `starter` bundle (or `full`), and points at /packages for
     more. Skipped automatically once an LLM backend is already installed.
  2. LLM — configure a default profile (Atlas Cloud fast-path or another provider,
     via the LiteLLM backend).
  3. Telegram — configure the bot, but only when the Telegram frontend is (being)
     installed.
"""

from guest.bases import BaseCommand
from guest.forms import FormStep


ATLAS_BASE_URL = "https://api.atlascloud.ai/v1"
ATLAS_CODING_PLAN_URL = "https://www.atlascloud.ai/console/coding-plan"
ATLAS_DEFAULT_MODEL = "minimaxai/minimax-m2.7"
DEFAULT_ENV_VAR = "ATLAS_API_KEY"
DEFAULT_CONTEXT_SIZE = 0
# The name a profile falls back to when discovery has not run or has found
# nothing. Deliberately the *retired* spelling: the live backend declares
# ``replaces = ["LiteLLMService"]``, so this still resolves through the alias
# map, whereas guessing the current class name would break the moment the
# backend is renamed again. Anything actually installed is preferred over it.
DEFAULT_BACKEND = "LiteLLMService"

ESSENTIALS_BUNDLE = "bundle_essentials"
KNOWLEDGEBASE_BUNDLE = "bundle_knowledgebase"
TELEGRAM_PACKAGE = "frontend_telegram"

#: What each install choice actually installs, in order. A *list* rather than
#: one name because the second choice is the first plus one — the knowledge
#: base is what you add to a working instance, not an alternative to it, so
#: offering it alone would be offering a broken install.
BUNDLE_CHOICES = {
    ESSENTIALS_BUNDLE: [ESSENTIALS_BUNDLE],
    "essentials_and_knowledgebase": [ESSENTIALS_BUNDLE, KNOWLEDGEBASE_BUNDLE],
}

WELCOME_PROMPT = (
    "Welcome to Second Brain.\n\n"
    "The kernel ships almost nothing on its own — capabilities are installed from a "
    "package store. The `essentials` bundle is the recommended first install: an LLM "
    "backend (LiteLLM, which reaches most providers), the Telegram frontend, file "
    "read/edit/search, shell and script running, SQL, ask-user-question, plugin "
    "validation, subagents, web search, and auto-titling. Adding `knowledgebase` "
    "indexes your files and makes them searchable — every file parser, OCR, "
    "transcription, embeddings and the three search tools (a much larger download, "
    "and the natural next step once the basics work).\n\n"
    "You can browse and install more anytime with /packages.\n\n"
    "Second Brain is sponsored by Atlas Cloud — a fast way to get an API key: "
    f"{ATLAS_CODING_PLAN_URL}"
)

LLM_INTRO_PROMPT = (
    "Let's set your default LLM profile. Atlas Cloud is the sponsored fast-path "
    "(300+ models behind one key); or point Second Brain at any other provider."
)

KEY_SOURCE_PROMPT = (
    "To use Atlas Cloud you need an API key. Sign up at "
    f"{ATLAS_CODING_PLAN_URL} and create an API key, then choose how you want to supply it:"
)

ENV_VAR_PROMPT = (
    "Enter the name of the environment variable that holds your Atlas key. "
    "You'll need to set this variable in your shell/system before Second Brain can call Atlas (for example on Windows: `setx ATLAS_API_KEY your-key`)."
)

OTHER_MODEL_PROMPT = (
    "Enter the LiteLLM model name, including the provider prefix when needed. "
    "Examples: `openai/gpt-4o-mini`, `anthropic/claude-3-5-sonnet-latest`, "
    "`minimax/MiniMax-M2.7`. For an OpenAI-compatible endpoint (set the base URL "
    "below), a plain id like `deepseek-ai/deepseek-v4-pro` is auto-routed through "
    "the openai provider."
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
    "Now let's set up Telegram. The Telegram frontend gives you a much better experience than the REPL — "
    "push notifications, attachments, inline buttons, and access from your phone.\n\n"
    "You'll need:\n"
    "  1. A bot token from @BotFather on Telegram (https://t.me/BotFather → /newbot)\n"
    "  2. Your Telegram user ID — message @userinfobot and it will reply with your numeric ID"
)
TELEGRAM_TOKEN_PROMPT = "Paste the bot token from @BotFather."
TELEGRAM_USER_PROMPT = (
    "Enter your Telegram user ID (a number from @userinfobot). Only this user will be allowed to talk to the bot."
)

PACKAGES_SECTION = (
    "Get more with /packages:\n"
    "  /packages available        — browse the store by category\n"
    "  /packages install <id>     — install a package or bundle\n"
    f"  Next step: `{KNOWLEDGEBASE_BUNDLE}` — parsers, OCR, transcription, "
    "embeddings and the three search tools, so the agent can find things in "
    "your own files. Individual packages (gmail, google_drive, mcp, "
    "plan_mode) install by name."
)


class SetupCommand(BaseCommand):
    """Slash-command handler for `/setup`."""
    name = "setup"
    description = "Onboarding: install a starter bundle, then configure an LLM and Telegram"
    category = "System"
    # No per-action split is available — the wizard has no ``action``
    # argument, and every route through it installs packages or writes
    # settings. So it asks once, at the door, naming the whole grant. That is
    # also the honest shape for onboarding: the user is being told what the
    # wizard is about to do before it starts, not interrupted halfway.
    require_approval = True
    approval_actor_id = "user"
    requests = [
        "plugin.list", "plugin.install", "config.write",
        "paths.get", "env.read", "net.http", "llm.list",
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
            # starter and full both include the LiteLLM backend + Telegram frontend.
            will_have_telegram = True
        else:
            will_have_telegram = _package_installed(sdk, TELEGRAM_PACKAGE)

        # Phase 2 — LLM profile.
        steps.append(FormStep(
            "llm_choice", LLM_INTRO_PROMPT, True,
            enum=["atlas", "other"],
            enum_labels=["Set up Atlas Cloud", "Use another provider"],
            columns=1,
        ))
        llm_choice = args.get("llm_choice")
        if llm_choice == "atlas":
            steps.extend(self._atlas_steps(args))
        elif llm_choice == "other":
            steps.extend(self._other_steps(args, backends))

        # Phase 3 — Telegram, once the LLM branch is satisfied and the frontend
        # is (being) installed.
        if will_have_telegram and _llm_steps_complete(args, llm_choice):
            steps.extend(self._telegram_steps(args))
        return steps

    def _atlas_steps(self, args):
        """Atlas Cloud key/model collection."""
        steps = [FormStep(
            "key_source", KEY_SOURCE_PROMPT, True,
            enum=["direct", "env_var"],
            enum_labels=["Paste the key directly", "Use an environment variable (you'll set it yourself)"],
            columns=1,
        )]
        if args.get("key_source") == "direct":
            steps.append(FormStep("api_key", "Paste your Atlas Cloud API key.", True))
        elif args.get("key_source") == "env_var":
            steps.append(FormStep("env_var_name", ENV_VAR_PROMPT, True, default=DEFAULT_ENV_VAR))
        if args.get("key_source"):
            steps.append(FormStep(
                "model_name",
                "Model name to use as your default profile. You can change this later with /llm.",
                False, default=ATLAS_DEFAULT_MODEL, prompt_when_missing=True,
            ))
        return steps

    def _other_steps(self, args, backends):
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
            enum_labels=["Set up Telegram", "Skip — I'll use the REPL for now"],
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
        env_warning = None

        # Phase 1 — install the chosen bundle before configuring anything that
        # depends on it. Bail clearly if there's no connectivity or the install
        # fails, so we don't pretend a half-set-up instance is ready.
        for bundle in BUNDLE_CHOICES.get(install_choice, ()):
            if not _has_internet(sdk):
                return (
                    f"No internet connection detected. Installing the `{bundle}` "
                    "bundle needs to download packages and their dependencies. Connect "
                    "to the internet and run /setup again."
                )
            try:
                result = sdk.plugins.install(bundle)
            except sdk.Failed as e:
                # Reported rather than raised past the remaining bundles: the
                # essentials install is what everything else depends on, so a
                # knowledge-base failure must not lose the report of the one
                # that worked.
                return "\n\n".join(sections + [
                    f"Couldn't install the `{bundle}` bundle: {e.error}\n\n"
                    f"Resolve the issue (or try `/packages install {bundle}`), "
                    "then re-run /setup."])
            sections.append(f"Installed the `{bundle}` bundle.\n"
                            + _indent(result))

        # Phase 2 — LLM profile.
        llm_choice = args.get("llm_choice")
        if llm_choice == "atlas":
            result = self._save_atlas(sdk, args)
            if isinstance(result, str):
                return result
            sections.append(result[0])
            env_warning = result[1]
        elif llm_choice == "other":
            result = self._save_other(sdk, args)
            if isinstance(result, str):
                return result
            sections.append(result)

        # Phase 3 — Telegram.
        if args.get("telegram_choice") == "setup":
            sections.append(self._save_telegram(sdk, args))
        elif args.get("telegram_choice") == "skip":
            sections.append("Telegram: skipped. Use /config to add `telegram_bot_token` and `telegram_allowed_user_id` later.")

        sections.append(PACKAGES_SECTION)
        sections.append(self._location_section(sdk))
        sections.append(self._hint_section())
        if env_warning:
            sections.insert(0, env_warning)
        return "\n\n".join(s for s in sections if s)

    # ──────────────────────────────────────────────────────────────────
    # Persistence helpers
    # ──────────────────────────────────────────────────────────────────

    def _save_atlas(self, sdk, args):
        """Persist an Atlas Cloud LLM profile. Returns (section, warning|None) or error string."""
        key_source = args.get("key_source")
        if key_source == "direct":
            api_key_field = (args.get("api_key") or "").strip()
            env_var_set = True
        elif key_source == "env_var":
            api_key_field = (args.get("env_var_name") or DEFAULT_ENV_VAR).strip() or DEFAULT_ENV_VAR
            env_var_set = bool(sdk.env.read(api_key_field))
        else:
            return "Setup cancelled."
        if not api_key_field:
            return "An API key (or environment variable name) is required."
        model_name = (args.get("model_name") or ATLAS_DEFAULT_MODEL).strip() or ATLAS_DEFAULT_MODEL

        profile = {
            "llm_endpoint": ATLAS_BASE_URL,
            "secret_llm_api_key": api_key_field,
            "llm_context_size": DEFAULT_CONTEXT_SIZE,
            "llm_service_class": DEFAULT_BACKEND,
        }
        _install_llm_profile(sdk, model_name, profile)

        section = (
            f"LLM: Atlas Cloud set up. Default profile: {model_name}\n"
            f"  Endpoint: {ATLAS_BASE_URL}\n"
            f"  Coding plan: {ATLAS_CODING_PLAN_URL}\n"
            "  Use /llm to edit the profile or add more models."
        )
        warning = None
        if key_source == "env_var" and not env_var_set:
            warning = (
                f"Note: ${api_key_field} is not currently set in this environment. "
                "Set it before sending your first message or Atlas calls will fail."
            )
        return section, warning

    def _save_other(self, sdk, args):
        """Persist a generic LLM profile. Returns section string or error string."""
        name = (args.get("other_model_name") or "").strip()
        if not name:
            return "Model name is required."
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
            "  /packages available             — browse the store by category\n\n"
            "Then run /setup again to configure your LLM and Telegram."
        )

    def _location_section(self, sdk):
        """One-paragraph summary of where things live on disk."""
        return (
            "Files & data:\n"
            f"  DATA_DIR: {sdk.paths.get('data')}\n"
            "  Holds your config (config.json, plugin_config.json), the SQLite database, the attachment cache, installed packages, and any sandbox plugins the agent writes for itself.\n"
            "  Run /locations to see existing plugins, and /config to view and edit your config files."
        )

    def _hint_section(self):
        """Closing hint about how to continue."""
        return (
            "You're ready. Run /new to start a conversation, then just ask the LLM anything — "
            "how Second Brain works, what tools are available, how to set up a task, and more!"
        )


def _llm_steps_complete(args, choice):
    """Return True once the LLM branch has collected enough to move on to Telegram."""
    if choice == "atlas":
        key_source = args.get("key_source")
        if key_source == "direct":
            return bool(args.get("api_key"))
        if key_source == "env_var":
            return bool(args.get("env_var_name"))
        return False
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
