"""Slash command plugin for `/llm`."""

from guest.bases import BaseCommand
from guest.forms import FormStep


ACTIONS = ["edit", "set_default", "load", "unload", "remove"]
ACTION_LABELS = ["Edit", "Set default", "Load", "Unload", "Remove"]
PROFILE_FIELDS = [
    "llm_endpoint", "secret_llm_api_key", "llm_context_size",
    "llm_service_class", "llm_capability_image",
    "llm_capability_audio", "llm_capability_video",
]
FIELDS = ["llm_model_name", *PROFILE_FIELDS]
FIELD_LABELS = [
    "Model name", "Endpoint", "API key", "Context size",
    "Service class", "Images", "Audio", "Video",
]
DEFAULT_BACKEND = "LiteLLMService"
CAPABILITY_FIELDS = {
    "llm_capability_image": "image",
    "llm_capability_audio": "audio",
    "llm_capability_video": "video",
}


class LlmCommand(BaseCommand):
    """Inspect and manage configured LLM profiles."""

    name = "llm"
    description = "Select an LLM profile, then edit, set default, or remove it"
    category = "System"
    requests = [
        "config.read", "config.write", "plugin.list",
        "service.list", "service.load", "service.unload",
    ]
    agent_prompt = (
        "The default LLM can be switched mid-conversation with /llm. Earlier "
        "assistant turns in this conversation may have been produced by a "
        "different model; the [SYSTEM CONTEXT UPDATE] block names the model "
        "driving the current turn. A changed model is normal, not manipulation."
    )

    def form(self, sdk, args):
        profiles = sdk.config.read("llm_profiles") or {}
        default = sdk.config.read("default_llm_profile") or ""
        names = [*sorted(profiles), "add"]
        steps = [FormStep(
            "model_name",
            _default_prompt(default),
            True,
            enum=names,
            enum_labels=[_model_label(default, item) for item in names],
        )]
        if args.get("model_name") == "add":
            backends = _backends(sdk)
            return steps + [
                FormStep(
                    "llm_service_class",
                    "Choose how Second Brain should connect to this model.",
                    True, enum=backends, default=backends[0]),
                FormStep(
                    "new_model_name",
                    "Enter the model name exactly, including provider prefix "
                    "when needed (for example `openai/gpt-4o-mini` or "
                    "`anthropic/claude-3-5-sonnet-latest`).",
                    True),
                FormStep(
                    "llm_endpoint",
                    "Enter the provider base URL [optional]. Leave blank for "
                    "the provider default.",
                    False, default="", prompt_when_missing=True),
                FormStep(
                    "secret_llm_api_key",
                    "Enter the API key, or the environment variable name that "
                    "contains it. Leave blank to use the provider default.",
                    False, default="", prompt_when_missing=True),
                FormStep(
                    "llm_context_size",
                    "Optional context window size in tokens. Use 0 for "
                    "dynamic compaction or if unknown.",
                    False, "integer", default=0, prompt_when_missing=True),
                *_capability_steps(),
            ]
        name = args.get("model_name")
        if name:
            steps.append(FormStep(
                "action",
                "What do you want to do with this LLM profile?\n\n"
                + _describe(sdk, profiles, default, name),
                True, enum=ACTIONS, enum_labels=ACTION_LABELS))
        if args.get("action") == "edit":
            field = args.get("field")
            steps += [
                FormStep(
                    "field", "Choose which LLM setting to edit.", True,
                    enum=FIELDS, enum_labels=FIELD_LABELS),
                FormStep(
                    "value", _value_prompt(field, _backends(sdk)), True,
                    _value_type(field)),
            ]
        return steps

    def run(self, sdk, args):
        profiles = sdk.config.read("llm_profiles") or {}
        default = sdk.config.read("default_llm_profile") or ""
        name = args.get("model_name")
        if name == "add":
            name = (args.get("new_model_name") or "").strip()
            if not name:
                return "Model name is required."
            first = not profiles
            profiles[name] = _profile(args)
            sdk.config.write("llm_profiles", profiles, scope="plugin")
            if first:
                sdk.config.write(
                    "default_llm_profile", name, scope="plugin")
            return f"Added LLM profile: {name}"
        if name not in profiles:
            return "Unknown LLM profile."
        action = args.get("action")
        if action == "edit":
            field = args.get("field")
            was_loaded = _service(sdk, name).get("loaded", False)
            if field == "llm_model_name":
                new_name = _coerce(field, args.get("value")).strip()
                if not new_name:
                    return "Model name is required."
                if new_name != name and new_name in profiles:
                    return f"LLM profile already exists: {new_name}"
                profiles[new_name] = profiles.pop(name)
                if default == name:
                    default = new_name
                    sdk.config.write(
                        "default_llm_profile", default, scope="plugin")
                name = new_name
            elif field in CAPABILITY_FIELDS:
                profiles[name].setdefault("llm_capabilities", {})[
                    CAPABILITY_FIELDS[field]
                ] = _coerce(field, args.get("value"))
            else:
                profiles[name][field] = _coerce(field, args.get("value"))
            sdk.config.write("llm_profiles", profiles, scope="plugin")
            if was_loaded:
                sdk.services.load(name)
            return f"Updated LLM profile: {name}"
        if action == "set_default":
            sdk.config.write(
                "default_llm_profile", name, scope="plugin")
            return f"Default LLM profile set to: {name}"
        if action == "load":
            try:
                loaded = sdk.services.load(name)
            except sdk.Failed as exc:
                if "not registered" in exc.error:
                    return f"No backend is installed for {name}."
                raise
            return (
                f"Loaded LLM profile: {name}"
                if loaded
                else f"Could not load {name}. Check the app log."
            )
        if action == "unload":
            service = _service(sdk, name)
            if not service or not service.get("loaded"):
                return f"LLM profile {name} is not loaded."
            sdk.services.unload(name)
            return f"Unloaded LLM profile: {name}"
        if action == "remove":
            names = sorted(profiles)
            service = _service(sdk, name)
            if service and service.get("loaded"):
                sdk.services.unload(name)
            profiles.pop(name, None)
            if default == name:
                remaining = [item for item in names if item != name]
                replacement = (
                    remaining[min(names.index(name), len(remaining) - 1)]
                    if remaining else ""
                )
                sdk.config.write(
                    "default_llm_profile", replacement, scope="plugin")
            sdk.config.write("llm_profiles", profiles, scope="plugin")
            return f"Removed LLM profile: {name}"
        return f"Unknown action: {action}"


def _capability_steps():
    return [
        FormStep(
            field,
            f"Can this model read {label} natively? Choose yes/no, or "
            "/skip if unsure.",
            False, "boolean", default=None, prompt_when_missing=True,
        )
        for field, label in (
            ("llm_capability_image", "images"),
            ("llm_capability_audio", "audio"),
            ("llm_capability_video", "video"),
        )
    ]


def _backends(sdk):
    return (
        sdk.plugins.list(category="services", role="llm_backend")
        or [DEFAULT_BACKEND]
    )


def _services(sdk):
    return {
        service["name"]: service
        for service in sdk.services.list(details=True)
    }


def _service(sdk, name):
    return _services(sdk).get(name, {})


def _profile(args):
    profile = {
        field: _coerce(field, args.get(field))
        for field in PROFILE_FIELDS
        if field not in CAPABILITY_FIELDS
    }
    capabilities = {
        capability: _coerce(field, args.get(field))
        for field, capability in CAPABILITY_FIELDS.items()
        if field in args and args.get(field) is not None
    }
    if capabilities:
        profile["llm_capabilities"] = capabilities
    return profile


def _coerce(field, value):
    if field == "llm_context_size":
        return int(value or 0)
    if field in CAPABILITY_FIELDS:
        return (
            value if isinstance(value, bool)
            else str(value).strip().lower() in {"true", "yes", "1", "y"}
        )
    return "" if value is None else str(value)


def _describe(sdk, profiles, default, name):
    profile = profiles.get(name)
    if not profile:
        return "Action"
    service = _service(sdk, name)
    mark = " (default)" if default == name else ""
    context = int(profile.get("llm_context_size", 0) or 0)
    context_text = (
        "0 (reactive compaction)" if context == 0 else f"{context:,}")
    capabilities = ", ".join(
        key for key, value in (
            profile.get("llm_capabilities") or {}).items() if value
    ) or "none declared"
    backend = profile.get("llm_service_class", DEFAULT_BACKEND)
    if not service:
        backend += " (not installed)"
    return sdk.md.card(f"{name}{mark}", [
        ("Status", "Loaded" if service.get("loaded") else "Unloaded"),
        ("Class", backend),
        ("Context", context_text),
        ("Native attachments", capabilities),
    ])


def _model_label(default, name):
    if name == "add":
        return "Add profile"
    return f"{name} (default)" if default == name else name


def _default_prompt(default):
    return (
        "Select an LLM profile, or add a new one.\n"
        f"Default: {default or '(none)'}"
    )


def _value_type(field):
    if field == "llm_context_size":
        return "integer"
    if field in CAPABILITY_FIELDS:
        return "boolean"
    return "string"


def _value_prompt(field, backends):
    return {
        "llm_endpoint": (
            "Enter a provider base URL, or leave it blank for the provider "
            "default."),
        "llm_model_name": "Enter the model name for this profile.",
        "secret_llm_api_key": (
            "Enter the API key value or environment variable name. Leave "
            "blank to let the backend read its own environment."),
        "llm_context_size": (
            "Enter the context window size in tokens. Use 0 if unknown."),
        "llm_service_class": f"Enter one of: {', '.join(backends)}.",
        "llm_capability_image": "Can this model read images natively?",
        "llm_capability_audio": "Can this model read audio natively?",
        "llm_capability_video": "Can this model read video natively?",
    }.get(field, "Enter the new value.")
