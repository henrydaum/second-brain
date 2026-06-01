"""Slash command plugin for `/setup` — onboarding ramp for LLM and Telegram."""

import os

from config import config_manager
from paths import DATA_DIR
from plugins.BaseCommand import BaseCommand
from plugins.services.service_llm import llm_backend_names
from state_machine.conversation import FormStep


ATLAS_BASE_URL = "https://api.atlascloud.ai/v1"
ATLAS_PROMO_URL = "https://www.atlascloud.ai/?utm_source=github&utm_medium=link&utm_campaign=second-brain"
ATLAS_CODING_PLAN_URL = "https://www.atlascloud.ai/console/coding-plan"
ATLAS_DEFAULT_MODEL = "minimaxai/minimax-m2.7"
DEFAULT_ENV_VAR = "ATLAS_API_KEY"
DEFAULT_CONTEXT_SIZE = 0

DEFAULT_BACKEND = "LiteLLMService"

WELCOME_PROMPT = (
    "Welcome to Second Brain.\n\n"
    "This setup will walk you through configuring an LLM, as well as the Telegram frontend.\n\n"
    "Second Brain is sponsored by Atlas Cloud, an AI inference platform with access to 300+ models. Their coding plan is a fast and seamless way to get Second Brain up and running — sign up here: "
    f"{ATLAS_CODING_PLAN_URL}"
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
    "Enter the LiteLLM model name exactly, including provider prefix when needed (for example `openai/gpt-4o-mini` or `anthropic/claude-3-5-sonnet-latest`)."
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


class SetupCommand(BaseCommand):
    """Slash-command handler for `/setup`."""
    name = "setup"
    description = "Onboarding: configure an LLM and the Telegram frontend"
    category = "System"

    def form(self, args, context):
        """Handle form."""
        steps = [FormStep(
            "llm_choice", WELCOME_PROMPT, True,
            enum=["atlas", "other"],
            enum_labels=["Set up Atlas Cloud", "Use another provider"],
            columns=1,
        )]
        choice = args.get("llm_choice")
        if choice == "atlas":
            steps.extend(self._atlas_steps(args))
        elif choice == "other":
            steps.extend(self._other_steps(args))
        if _llm_steps_complete(args, choice):
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
        sections = []
        env_warning = None

        choice = args.get("llm_choice")
        if choice == "atlas":
            result = self._save_atlas(args, context)
            if isinstance(result, str):
                return result
            sections.append(result[0])
            env_warning = result[1]
        elif choice == "other":
            result = self._save_other(args, context)
            if isinstance(result, str):
                return result
            sections.append(result)
        else:
            return "Setup cancelled."

        if args.get("telegram_choice") == "setup":
            sections.append(self._save_telegram(args))
        elif args.get("telegram_choice") == "skip":
            sections.append("Telegram: skipped. Use /config to add `telegram_bot_token` and `telegram_allowed_user_id` later.")

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

    def _location_section(self):
        """One-paragraph summary of where things live on disk."""
        return (
            "Files & data:\n"
            f"  DATA_DIR: {DATA_DIR}\n"
            "  Holds your config (config.json, plugin_config.json), the SQLite database, the attachment cache, and any sandbox plugins the agent writes for itself.\n"
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
        router.add_llm(name, profile)
    _save(context.config)


def _save(config):
    """Internal helper to save setup-affected keys to plugin config."""
    saved = config_manager.load_plugin_config()
    saved.update({k: config.get(k) for k in ("llm_profiles", "default_llm_profile")})
    config_manager.save_plugin_config(saved)
