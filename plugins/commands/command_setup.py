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

import os
import socket

from config import config_manager
from paths import DATA_DIR
from plugins.BaseCommand import BaseCommand
from plugins.commands.helpers import package_manager
from plugins.services.service_llm import llm_backend_names
from state_machine.conversation import FormStep


ATLAS_BASE_URL = "https://api.atlascloud.ai/v1"
ATLAS_CODING_PLAN_URL = "https://www.atlascloud.ai/console/coding-plan"
ATLAS_DEFAULT_MODEL = "minimaxai/minimax-m2.7"
DEFAULT_ENV_VAR = "ATLAS_API_KEY"
DEFAULT_CONTEXT_SIZE = 0
DEFAULT_BACKEND = "LiteLLMService"

STARTER_BUNDLE = "bundle_starter"
FULL_BUNDLE = "bundle_full"
TELEGRAM_PACKAGE = "frontend_telegram"

WELCOME_PROMPT = (
    "Welcome to Second Brain.\n\n"
    "The kernel ships almost nothing on its own — capabilities are installed from a "
    "package store. The `starter` bundle is the recommended first install: an LLM "
    "backend (LiteLLM, which reaches most providers), the Telegram frontend, file "
    "read/edit, sql & shell tools, ask-user-question, plugin authoring, and memory + "
    "auto-title tasks. `full` adds every file parser, transcription/OCR, and the "
    "indexing & search pipeline (a larger download).\n\n"
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
    "  Handy bundles: bundle_all_parsers, bundle_indexing_search, "
    "bundle_web_search, bundle_gmail, bundle_mcp, bundle_google_drive, "
    "bundle_scheduling, bundle_plan_mode, bundle_full."
)


class SetupCommand(BaseCommand):
    """Slash-command handler for `/setup`."""
    name = "setup"
    description = "Onboarding: install a starter bundle, then configure an LLM and Telegram"
    category = "System"

    def form(self, args, context):
        """Build the dynamic onboarding form."""
        steps = []
        backend_ready = bool(llm_backend_names())

        # Phase 1 — packages. Only lead with this when there's no LLM backend yet
        # (a fresh install). A returning user skips straight to reconfiguring.
        if not backend_ready:
            steps.append(FormStep(
                "install_choice", WELCOME_PROMPT, True,
                enum=[STARTER_BUNDLE, FULL_BUNDLE, "skip"],
                enum_labels=[
                    "Install the starter bundle (recommended)",
                    "Install the full bundle (everything — larger download)",
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
            will_have_telegram = _package_installed(TELEGRAM_PACKAGE)

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
            steps.extend(self._other_steps(args))

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

    def _other_steps(self, args):
        """Generic LLM profile collection (mirrors /llm add)."""
        backends = llm_backend_names() or [DEFAULT_BACKEND]
        return [
            FormStep("other_model_name", OTHER_MODEL_PROMPT, True),
            FormStep("other_service_class", OTHER_SERVICE_PROMPT, True,
                     enum=backends, default=backends[0], columns=1),
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

    def run(self, args, context):
        """Execute `/setup` for the active session."""
        install_choice = args.get("install_choice")
        if install_choice == "skip":
            return self._skip_section()

        sections = []
        env_warning = None

        # Phase 1 — install the chosen bundle before configuring anything that
        # depends on it. Bail clearly if there's no connectivity or the install
        # fails, so we don't pretend a half-set-up instance is ready.
        if install_choice in (STARTER_BUNDLE, FULL_BUNDLE):
            if not _has_internet():
                return (
                    f"No internet connection detected. Installing the `{install_choice}` "
                    "bundle needs to download packages and their dependencies. Connect "
                    "to the internet and run /setup again."
                )
            try:
                result = package_manager.install_package(context.root_dir, install_choice, context)
            except Exception as e:
                return (
                    f"Couldn't install the `{install_choice}` bundle: {e}\n\n"
                    f"Resolve the issue (or try `/packages install {install_choice}`), then re-run /setup."
                )
            sections.append(f"Installed the `{install_choice}` bundle.\n" + _indent(result.text()))

        # Phase 2 — LLM profile.
        llm_choice = args.get("llm_choice")
        if llm_choice == "atlas":
            result = self._save_atlas(args, context)
            if isinstance(result, str):
                return result
            sections.append(result[0])
            env_warning = result[1]
        elif llm_choice == "other":
            result = self._save_other(args, context)
            if isinstance(result, str):
                return result
            sections.append(result)

        # Phase 3 — Telegram.
        if args.get("telegram_choice") == "setup":
            sections.append(self._save_telegram(args))
        elif args.get("telegram_choice") == "skip":
            sections.append("Telegram: skipped. Use /config to add `telegram_bot_token` and `telegram_allowed_user_id` later.")

        sections.append(PACKAGES_SECTION)
        sections.append(self._location_section())
        sections.append(self._hint_section())
        if env_warning:
            sections.insert(0, env_warning)
        return "\n\n".join(s for s in sections if s)

    # ──────────────────────────────────────────────────────────────────
    # Persistence helpers
    # ──────────────────────────────────────────────────────────────────

    def _save_atlas(self, args, context):
        """Persist an Atlas Cloud LLM profile. Returns (section, warning|None) or error string."""
        key_source = args.get("key_source")
        if key_source == "direct":
            api_key_field = (args.get("api_key") or "").strip()
            env_var_set = True
        elif key_source == "env_var":
            api_key_field = (args.get("env_var_name") or DEFAULT_ENV_VAR).strip() or DEFAULT_ENV_VAR
            env_var_set = bool(os.environ.get(api_key_field))
        else:
            return "Setup cancelled."
        if not api_key_field:
            return "An API key (or environment variable name) is required."
        model_name = (args.get("model_name") or ATLAS_DEFAULT_MODEL).strip() or ATLAS_DEFAULT_MODEL

        profile = {
            "llm_endpoint": ATLAS_BASE_URL,
            "llm_api_key": api_key_field,
            "llm_context_size": DEFAULT_CONTEXT_SIZE,
            "llm_service_class": DEFAULT_BACKEND,
        }
        _install_llm_profile(context, model_name, profile)

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

    def _save_other(self, args, context):
        """Persist a generic LLM profile. Returns section string or error string."""
        name = (args.get("other_model_name") or "").strip()
        if not name:
            return "Model name is required."
        profile = {
            "llm_endpoint": (args.get("other_endpoint") or "").strip(),
            "llm_api_key": (args.get("other_api_key") or "").strip(),
            "llm_context_size": int(args.get("other_context_size") or 0),
            "llm_service_class": (args.get("other_service_class") or DEFAULT_BACKEND).strip() or DEFAULT_BACKEND,
        }
        _install_llm_profile(context, name, profile)
        endpoint = profile["llm_endpoint"] or "(provider default)"
        return (
            f"LLM: profile `{name}` added and set as default.\n"
            f"  Service class: {profile['llm_service_class']}\n"
            f"  Endpoint: {endpoint}\n"
            "  Use /llm to edit or add more models."
        )

    def _save_telegram(self, args):
        """Persist Telegram credentials into plugin_config."""
        token = (args.get("telegram_bot_token") or "").strip()
        user_id = int(args.get("telegram_allowed_user_id") or 0)
        saved = config_manager.load_plugin_config()
        saved["telegram_bot_token"] = token
        saved["telegram_allowed_user_id"] = user_id
        config_manager.save_plugin_config(saved)
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
            f"  /packages install {STARTER_BUNDLE}   — the recommended baseline\n"
            f"  /packages install {FULL_BUNDLE}      — everything\n"
            "  /packages available        — browse the store by category\n\n"
            "Then run /setup again to configure your LLM and Telegram."
        )

    def _location_section(self):
        """One-paragraph summary of where things live on disk."""
        return (
            "Files & data:\n"
            f"  DATA_DIR: {DATA_DIR}\n"
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


def _install_llm_profile(context, name, profile):
    """Register a new LLM profile, set it as default, hot-load it, and persist."""
    profiles = context.config.setdefault("llm_profiles", {})
    profiles[name] = profile
    context.config["default_llm_profile"] = name
    router = (context.services or {}).get("llm")
    if router and hasattr(router, "add_llm"):
        try:
            router.add_llm(name, profile)
        except Exception:
            # Config is still persisted below; the profile loads on next start
            # even if hot-loading the backend now didn't take.
            pass
    _save(context.config)


def _save(config):
    """Internal helper to save setup-affected keys to plugin config."""
    saved = config_manager.load_plugin_config()
    saved.update({k: config.get(k) for k in ("llm_profiles", "default_llm_profile")})
    config_manager.save_plugin_config(saved)


def _package_installed(package_id):
    """Whether a package id has an install receipt."""
    try:
        return any(p.get("id") == package_id for p in package_manager.installed_packages())
    except Exception:
        return False


def _has_internet(timeout: float = 3.0) -> bool:
    """Best-effort connectivity check before a package download."""
    for host, port in (("github.com", 443), ("1.1.1.1", 53)):
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            continue
    return False


def _indent(text: str) -> str:
    """Indent a block two spaces for nesting under a section header."""
    return "\n".join(f"  {line}" if line else line for line in (text or "").splitlines())
