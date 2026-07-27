"""Slash command plugin for `/llm`."""

import json

from config import config_manager
from plugins.BaseCommand import BaseCommand
from plugins.frontends.helpers.formatters import detail_card
from llm import backend_names
from state_machine.conversation import FormStep


ACTIONS = ["edit", "set_default", "load", "unload", "remove"]
ACTION_LABELS = ["Edit", "Set default", "Load", "Unload", "Remove"]
PROFILE_FIELDS = ["llm_endpoint", "llm_api_key", "llm_context_size", "llm_service_class", "llm_capability_image", "llm_capability_audio", "llm_capability_video"]
FIELDS = ["llm_model_name", *PROFILE_FIELDS]
FIELD_LABELS = ["Model name", "Endpoint", "API key", "Context size", "Service class", "Images", "Audio", "Video"]
DEFAULT_BACKEND = "LiteLLMService"
CAPABILITY_FIELDS = {
    "llm_capability_image": "image",
    "llm_capability_audio": "audio",
    "llm_capability_video": "video",
}


class LlmCommand(BaseCommand):
    """Slash-command handler for `/llm`."""
    name = "llm"
    description = "Select an LLM profile, then edit, set default, or remove it"
    category = "System"
    agent_prompt = (
        "The default LLM can be switched mid-conversation with /llm. Earlier "
        "assistant turns in this conversation may have been produced by a "
        "different model; the [SYSTEM CONTEXT UPDATE] block names the model "
        "driving the current turn. A changed model is normal, not manipulation."
    )

    def form(self, args, context):
        """Handle form."""
        profiles = context.config.get("llm_profiles", {}) or {}
        names = [*sorted(profiles), "add"]
        steps = [FormStep("model_name", _default_prompt(context), True, enum=names, enum_labels=[_model_label(context, n) for n in names])]
        if args.get("model_name") == "add":
            backends = backend_names() or [DEFAULT_BACKEND]
            return steps + [
                FormStep("llm_service_class", "Choose how Second Brain should connect to this model.", True, enum=backends, default=backends[0]),
                FormStep("new_model_name", "Enter the model name exactly, including provider prefix when needed (for example `openai/gpt-4o-mini` or `anthropic/claude-3-5-sonnet-latest`).", True),
                FormStep("llm_endpoint", "Enter the provider base URL [optional]. Leave blank for the provider default.", False, default="", prompt_when_missing=True),
                FormStep("llm_api_key", "Enter the API key, or the environment variable name that contains it. Leave blank to use the provider default.", False, default="", prompt_when_missing=True),
                FormStep("llm_context_size", "Optional context window size in tokens. Use 0 for dynamic compaction or if unknown.", False, "integer", default=0, prompt_when_missing=True),
                FormStep("llm_capability_image", "Can this model read images natively? Choose yes/no, or /skip if unsure.", False, "boolean", default=None, prompt_when_missing=True),
                FormStep("llm_capability_audio", "Can this model read audio natively? Choose yes/no, or /skip if unsure.", False, "boolean", default=None, prompt_when_missing=True),
                FormStep("llm_capability_video", "Can this model read video natively? Choose yes/no, or /skip if unsure.", False, "boolean", default=None, prompt_when_missing=True),
            ]
        if args.get("model_name"):
            steps.append(FormStep("action", f"What do you want to do with this LLM profile?\n\n{_describe(context, args['model_name'])}", True, enum=ACTIONS, enum_labels=ACTION_LABELS))
        if args.get("action") == "edit":
            steps += [FormStep("field", "Choose which LLM setting to edit.", True, enum=FIELDS, enum_labels=FIELD_LABELS), FormStep("value", _value_prompt(args.get("field")), True, _value_type(args.get("field")))]
        return steps

    def run(self, args, context):
        """Execute `/llm` for the active session.

        Profiles are config; *loading* is a separate, explicit act. That split
        is new: models used to be registered one service per profile, so
        editing a profile silently rebuilt a live object as a side effect.
        Now a brain holds a pool of boxes — real processes under isolation —
        and opening or closing one is something the user asks for.
        """
        profiles = context.config.setdefault("llm_profiles", {})
        name = args.get("model_name")
        if name == "add":
            name = args.get("new_model_name", "").strip()
            if not name:
                return "Model name is required."
            first_profile = not profiles
            profiles[name] = _profile(args)
            if first_profile:
                context.config["default_llm_profile"] = name
            _save(context)
            return f"Added LLM profile: {name}"
        if name not in profiles:
            return "Unknown LLM profile."
        if args.get("action") == "edit":
            field = args.get("field")
            if field == "llm_model_name":
                new_name = _coerce(field, args.get("value")).strip()
                if not new_name:
                    return "Model name is required."
                if new_name != name and new_name in profiles:
                    return f"LLM profile already exists: {new_name}"
                profiles[new_name] = profiles.pop(name)
                if context.config.get("default_llm_profile") == name:
                    context.config["default_llm_profile"] = new_name
                name = new_name
            elif field in CAPABILITY_FIELDS:
                profiles[name].setdefault("llm_capabilities", {})[CAPABILITY_FIELDS[field]] = _coerce(field, args.get("value"))
            else:
                profiles[name][field] = _coerce(field, args.get("value"))
            # An edited profile is rebuilt by ``refresh``; if it was loaded,
            # reopen it so the new settings are actually in effect rather than
            # waiting behind a box that still holds the old ones.
            was_loaded = _is_loaded(name)
            _save(context)
            if was_loaded:
                _brain(name) and _brain(name).load()
            return f"Updated LLM profile: {name}"
        if args.get("action") == "set_default":
            context.config["default_llm_profile"] = name
            _save(context)
            return f"Default LLM profile set to: {name}"
        if args.get("action") == "load":
            _save(context)
            target = _brain(name)
            if target is None:
                return f"No backend is installed for {name}."
            return (f"Loaded LLM profile: {name}" if target.load()
                    else f"Could not load {name}. Check the app log.")
        if args.get("action") == "unload":
            target = _brain(name)
            if target is None or not target.loaded:
                return f"LLM profile {name} is not loaded."
            target.unload()
            return f"Unloaded LLM profile: {name}"
        if args.get("action") == "remove":
            names = sorted(profiles)
            target = _brain(name)
            if target is not None:
                target.unload()
            profiles.pop(name, None)
            if context.config.get("default_llm_profile") == name:
                remaining = [n for n in names if n != name]
                context.config["default_llm_profile"] = remaining[min(names.index(name), len(remaining) - 1)] if remaining else ""
            _save(context)
            return f"Removed LLM profile: {name}"
        return f"Unknown action: {args.get('action')}"


def _brain(name):
    """The registry's brain for a profile, or None."""
    import llm

    return llm.brain(name)


def _is_loaded(name) -> bool:
    """Whether a profile currently holds an open box."""
    target = _brain(name)
    return bool(target is not None and target.loaded)


def _profile(args):
    """Internal helper to handle profile."""
    profile = {f: _coerce(f, args.get(f)) for f in PROFILE_FIELDS if f not in CAPABILITY_FIELDS}
    caps = {cap: _coerce(field, args.get(field)) for field, cap in CAPABILITY_FIELDS.items() if field in args and args.get(field) is not None}
    if caps:
        profile["llm_capabilities"] = caps
    return profile


def _coerce(field, value):
    """Internal helper to handle coerce."""
    if field == "llm_context_size":
        return int(value or 0)
    if field in CAPABILITY_FIELDS:
        return value if isinstance(value, bool) else str(value).strip().lower() in {"true", "yes", "1", "y"}
    return "" if value is None else str(value)


def _value_type(field):
    """Internal helper to handle form type."""
    return "integer" if field == "llm_context_size" else "boolean" if field in CAPABILITY_FIELDS else "string"


def _describe(context, name):
    """Internal helper to handle describe."""
    p = (context.config.get("llm_profiles", {}) or {}).get(name)
    if not p:
        return "Action"
    target = _brain(name)
    mark = " (default)" if context.config.get("default_llm_profile") == name else ""
    ctx = int(p.get("llm_context_size", 0) or 0)
    ctx_str = "0 (reactive compaction)" if ctx == 0 else f"{ctx:,}"
    caps = ", ".join(k for k, v in (p.get("llm_capabilities") or {}).items() if v) or "none declared"
    backend = p.get("llm_service_class", DEFAULT_BACKEND)
    if target is not None and not target.available:
        backend += " (not installed)"
    return detail_card(f"{name}{mark}", [
        ("Status", "Loaded" if _is_loaded(name) else "Unloaded"),
        ("Class", backend),
        ("Context", ctx_str),
        ("Native attachments", caps),
    ])


def _model_label(context, name):
    """Internal helper to handle model label."""
    return "Add profile" if name == "add" else f"{name} (default)" if context.config.get("default_llm_profile") == name else name


def _default_prompt(context):
    """Return default prompt."""
    default = (context.config.get("default_llm_profile") or "").strip()
    return f"Select an LLM profile, or add a new one.\nDefault: {default or '(none)'}"


def _value_prompt(field):
    """Internal helper to handle value prompt."""
    return {
        "llm_endpoint": "Enter a provider base URL, or leave it blank for the provider default.",
        "llm_model_name": "Enter the model name for this profile.",
        "llm_api_key": "Enter the API key value or environment variable name. Leave blank to let the backend read its own environment.",
        "llm_context_size": "Enter the context window size in tokens. Use 0 if unknown.",
        "llm_service_class": f"Enter one of: {', '.join(backend_names() or [DEFAULT_BACKEND])}.",
        "llm_capability_image": "Can this model read images natively?",
        "llm_capability_audio": "Can this model read audio natively?",
        "llm_capability_video": "Can this model read video natively?",
    }.get(field, "Enter the new value.")


def _save(context):
    """Internal helper to save LLM."""
    config = context.config
    saved = config_manager.load_plugin_config()
    saved.update({k: config.get(k) for k in ("llm_profiles", "default_llm_profile")})
    config_manager.save_plugin_config(saved)
    runtime = getattr(context, "runtime", None)
    if runtime is not None and getattr(runtime, "config", None) is not None:
        runtime.config["llm_profiles"] = config.get("llm_profiles", {})
        runtime.config["default_llm_profile"] = config.get("default_llm_profile", "")
    # Config is the source of truth for which brains exist, so saving it is
    # exactly when the registry should be rebuilt. Brains whose settings did
    # not move keep their boxes; the rest are closed and replaced.
    import llm

    llm.refresh(runtime.config if runtime is not None
                and getattr(runtime, "config", None) is not None else config)
