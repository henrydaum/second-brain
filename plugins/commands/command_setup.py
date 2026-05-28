"""Slash command plugin for `/setup` — onboarding ramp with optional Atlas Cloud setup."""

import os

from config import config_manager
from plugins.BaseCommand import BaseCommand
from state_machine.conversation import FormStep


ATLAS_BASE_URL = "https://api.atlascloud.ai/v1"
ATLAS_PROMO_URL = "https://www.atlascloud.ai/?utm_source=github&utm_medium=link&utm_campaign=second-brain"
ATLAS_CODING_PLAN_URL = "https://www.atlascloud.ai/console/coding-plan"
ATLAS_DEFAULT_MODEL = "minimaxai/minimax-m2.5"
DEFAULT_ENV_VAR = "ATLAS_API_KEY"
DEFAULT_CONTEXT_SIZE = 0

WELCOME_PROMPT = (
    "Welcome to Second Brain.\n\n"
    "---\n\n"
    "**Sponsor disclosure.** Second Brain is sponsored by Atlas Cloud, a "
    "full-modal AI inference platform with OpenAI-compatible APIs and access "
    "to 300+ models. They have an onboarding promotion for their coding plan: "
    f"{ATLAS_CODING_PLAN_URL}\n\n"
    "You can configure Atlas as your default LLM in a few steps, or skip and "
    "configure any provider later with /llm."
)

KEY_SOURCE_PROMPT = (
    "To use Atlas Cloud you need an API key. Sign up at "
    f"{ATLAS_CODING_PLAN_URL} and create an API key, then choose how you want "
    "to supply it:"
)

ENV_VAR_PROMPT = (
    "Enter the name of the environment variable that holds your Atlas key. "
    "You'll need to set this variable in your shell/system before Second Brain "
    "can call Atlas (for example on Windows: `setx ATLAS_API_KEY your-key`)."
)


class SetupCommand(BaseCommand):
    """Slash-command handler for `/setup`."""
    name = "setup"
    description = "Onboarding: configure Atlas Cloud as your default LLM, or skip"
    category = "System"

    def form(self, args, context):
        """Handle form."""
        steps = [FormStep(
            "do_atlas_setup", WELCOME_PROMPT, True,
            enum=["atlas", "skip"],
            enum_labels=["Set up Atlas Cloud", "Skip — I'll configure /llm myself"],
            columns=1,
        )]
        if args.get("do_atlas_setup") != "atlas":
            return steps
        steps.append(FormStep(
            "key_source", KEY_SOURCE_PROMPT, True,
            enum=["direct", "env_var"],
            enum_labels=["Paste the key directly", "Use an environment variable (you'll set it yourself)"],
            columns=1,
        ))
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

    def run(self, args, context):
        """Execute `/setup` for the active session."""
        if args.get("do_atlas_setup") == "skip":
            return (
                "Setup skipped. Use /llm to configure an LLM profile and /agent "
                "to configure an agent profile when you're ready."
            )
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
            "llm_service_class": "OpenAILLM",
            "prompt_cache_key": "",
            "prompt_cache_retention": "",
        }
        profiles = context.config.setdefault("llm_profiles", {})
        profiles[model_name] = profile
        context.config["default_llm_profile"] = model_name

        router = (context.services or {}).get("llm")
        if router and hasattr(router, "add_llm"):
            router.add_llm(model_name, profile)

        _save(context.config)

        lines = [
            f"Atlas Cloud is set up. Default LLM profile: {model_name}",
            f"Endpoint: {ATLAS_BASE_URL}",
            f"Coding plan: {ATLAS_CODING_PLAN_URL}",
            "Use /llm to edit the profile or add more models.",
        ]
        if key_source == "env_var" and not env_var_set:
            lines.insert(1, (
                f"Note: ${api_key_field} is not currently set in this environment. "
                "Set it before sending your first message or Atlas calls will fail."
            ))
        return "\n".join(lines)


def _save(config):
    """Internal helper to save setup-affected keys to plugin config."""
    saved = config_manager.load_plugin_config()
    saved.update({k: config.get(k) for k in ("llm_profiles", "default_llm_profile")})
    config_manager.save_plugin_config(saved)
