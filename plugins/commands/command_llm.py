import json

from config import config_manager
from plugins.BaseCommand import BaseCommand
from state_machine.conversation import FormStep


ACTIONS = ["edit", "set_default", "remove"]
ACTION_LABELS = ["Edit", "Set default", "Remove"]
FIELDS = ["llm_endpoint", "llm_api_key", "llm_context_size", "llm_service_class"]
FIELD_LABELS = ["Endpoint", "API key", "Context size", "Service class"]


class LlmCommand(BaseCommand):
    name = "llm"
    description = "Select an LLM profile, then edit, set default, or remove it"
    category = "System"

    def form(self, args, context):
        profiles = context.config.get("llm_profiles", {}) or {}
        steps = [FormStep("model_name", _default_prompt(context), True, enum=[*sorted(profiles), "add"])]
        if args.get("model_name") == "add":
            return steps + [
                FormStep("new_model_name", "Enter the provider's model name exactly.", True),
                FormStep("llm_service_class", "Choose how Second Brain should connect to this model.", True, enum=["OpenAILLM", "LMStudioLLM"], default="OpenAILLM"),
                FormStep("llm_endpoint", "Optional OpenAI-compatible endpoint URL.", False, default="", prompt_when_missing=True),
                FormStep("llm_api_key", "API key value, or the environment variable name that contains it.", False, default="OPENAI_API_KEY", prompt_when_missing=True),
                FormStep("llm_context_size", "Optional context window size in tokens. Use 0 if unknown.", False, "integer", default=0, prompt_when_missing=True),
            ]
        if args.get("model_name"):
            steps.append(FormStep("action", f"What do you want to do with this LLM profile?\n\n{_describe(context, args['model_name'])}", True, enum=ACTIONS, enum_labels=ACTION_LABELS))
        if args.get("action") == "edit":
            steps += [FormStep("field", "Choose which LLM setting to edit.", True, enum=FIELDS, enum_labels=FIELD_LABELS), FormStep("value", _value_prompt(args.get("field")), True)]
        return steps

    def run(self, args, context):
        profiles = context.config.setdefault("llm_profiles", {})
        router = (context.services or {}).get("llm")
        name = args.get("model_name")
        if name == "add":
            name = args.get("new_model_name", "").strip()
            if not name:
                return "Model name is required."
            profiles[name] = _profile(args)
            if router and hasattr(router, "add_llm"):
                router.add_llm(name, profiles[name])
            context.config.setdefault("default_llm_profile", name)
            _save(context.config)
            return f"Added LLM profile: {name}"
        if name not in profiles:
            return "Unknown LLM profile."
        if args.get("action") == "edit":
            field = args.get("field")
            profiles[name][field] = _coerce(field, args.get("value"))
            if router and hasattr(router, "add_llm"):
                router.add_llm(name, profiles[name])
            _save(context.config)
            return f"Updated LLM profile: {name}"
        if args.get("action") == "set_default":
            context.config["default_llm_profile"] = name
            _save(context.config)
            return f"Default LLM profile set to: {name}"
        if args.get("action") == "remove":
            profiles.pop(name, None)
            if router and hasattr(router, "remove_llm"):
                router.remove_llm(name)
            if context.config.get("default_llm_profile") == name:
                context.config["default_llm_profile"] = next(iter(profiles), "")
            _save(context.config)
            return f"Removed LLM profile: {name}"
        return f"Unknown action: {args.get('action')}"


def _profile(args):
    return {f: _coerce(f, args.get(f)) for f in FIELDS}


def _coerce(field, value):
    if field == "llm_context_size":
        return int(value or 0)
    return "" if value is None else str(value)


def _describe(context, name):
    p = (context.config.get("llm_profiles", {}) or {}).get(name)
    if not p:
        return "Action"
    loaded = getattr((context.services or {}).get(name), "loaded", False)
    mark = " default" if context.config.get("default_llm_profile") == name else ""
    return f"{name}{mark}\nStatus: {'Loaded' if loaded else 'Unloaded'}\nClass: {p.get('llm_service_class', 'OpenAILLM')}\nContext: {p.get('llm_context_size', 0)}"


def _default_prompt(context):
    default = (context.config.get("default_llm_profile") or "").strip()
    return f"Select an LLM profile, or add a new one.\nDefault: {default or '(none)'}"


def _value_prompt(field):
    return {
        "llm_endpoint": "Enter the endpoint URL, or leave it blank for the provider default.",
        "llm_api_key": "Enter the API key value or environment variable name.",
        "llm_context_size": "Enter the context window size in tokens. Use 0 if unknown.",
        "llm_service_class": "Enter OpenAILLM or LMStudioLLM.",
    }.get(field, "Enter the new value.")


def _save(config):
    saved = config_manager.load_plugin_config()
    saved.update({k: config.get(k) for k in ("llm_profiles", "default_llm_profile")})
    config_manager.save_plugin_config(saved)
