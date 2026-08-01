"""Slash command plugin for `/llm`."""

from guest.bases import BaseCommand
from guest.forms import FormStep


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
CAPABILITY_FIELDS = {
    "llm_capability_image": "image",
    "llm_capability_audio": "audio",
    "llm_capability_video": "video",
}


class LlmCommand(BaseCommand):
    """Inspect and manage configured LLM profiles."""

    name = "llm"
    description = "Select an LLM profile, then edit, set default, or remove it"
    category = "Capabilities"
    # Every one of the five actions is consequential — they open or close real
    # processes and rewrite profile settings including API keys. The read-only
    # path is `run()` with no profile named, which never reaches an action, so
    # this stays a per-action predicate rather than a blanket
    # ``require_approval`` that would gate merely looking at the list.
    # Spelled out rather than derived from ACTIONS: declarations are read by
    # AST, which sees literals and not a call.
    approval_actions = ("edit", "set_default", "load", "unload", "remove")
    approval_actor_id = "user"
    requests = [
        "config.read", "config.write", "plugin.list",
        "llm.list", "llm.load", "llm.unload",
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
        registry = _registry(sdk)
        names = [*sorted(profiles), "add"]
        steps = [FormStep(
            "model_name",
            _default_prompt(sdk, registry, profiles, default),
            True,
            enum=names,
            enum_labels=[_model_label(default, item) for item in names],
        )]
        if args.get("model_name") == "add":
            backends = _backend_names(registry)
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
            actions, labels = _actions_for(registry, default, name)
            steps.append(FormStep(
                "action",
                "What do you want to do with this LLM profile?\n\n"
                + _describe(sdk, registry, profiles, default, name),
                True, enum=actions, enum_labels=labels))
        if args.get("action") == "edit":
            field = args.get("field")
            steps += [
                FormStep(
                    "field", "Choose which LLM setting to edit.", True,
                    enum=FIELDS, enum_labels=FIELD_LABELS),
                FormStep(
                    "value", _value_prompt(field, _backend_names(registry)),
                    True, _value_type(field)),
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
            was_loaded = _profile_row(_registry(sdk), name).get("loaded", False)
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
                # Reopen on the edited profile. This never fired before: the
                # loaded flag came from the service registry, which has never
                # heard of an LLM profile, so it read False every time and an
                # edit silently left the old settings running in the open box.
                sdk.llm.load(name)
            return f"Updated LLM profile: {name}"
        if action == "set_default":
            sdk.config.write(
                "default_llm_profile", name, scope="plugin")
            return f"Default LLM profile set to: {name}"
        if action == "load":
            try:
                loaded = sdk.llm.load(name)
            except sdk.Failed as exc:
                # The registry names the backend the profile asked for, which
                # is the thing the person has to go and install.
                return exc.error
            return (
                f"Loaded LLM profile: {name}"
                if loaded
                else f"Could not load {name}. Check the app log."
            )
        if action == "unload":
            sdk.llm.unload(name)
            return f"Unloaded LLM profile: {name}"
        if action == "remove":
            names = sorted(profiles)
            if _profile_row(_registry(sdk), name).get("loaded"):
                sdk.llm.unload(name)
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


def _registry(sdk):
    """Everything the kernel knows about models, in one Request.

    This whole command used to ask ``sdk.services`` these questions, and had
    done since before ``service_llm.py`` was deleted and profiles stopped
    being services. Nothing raised — the service registry simply had no key
    for any profile, so every lookup returned ``{}`` and the command reported
    each model uninstalled and unloaded while conversations drove those very
    models without trouble.
    """
    try:
        return sdk.llm.list() or {}
    except sdk.Failed:
        return {}


def _profile_row(registry, name):
    """One profile's live registry row: is it open, what backend serves it."""
    for row in registry.get("profiles") or []:
        if row.get("model_name") == name:
            return row
    return {}


def _backend_names(registry):
    """Backend class names, which is what a profile stores."""
    return [entry["name"] for entry in registry.get("backends") or []] or [""]


def _backend_label(registry, configured):
    """What a person should read for a configured backend name.

    Two hops, and both are needed. A profile stores whatever name it was
    written with, and a migrated backend claims its predecessor's — so
    ``LiteLLMService`` has to become ``LiteLLMBackend`` before the display
    name for it can be found. Skipping that is why the card said
    "LiteLLMService" for a backend whose file has declared
    ``display_name = "LiteLLM (any provider)"` all along.
    """
    resolved = (registry.get("aliases") or {}).get(configured, configured)
    for entry in registry.get("backends") or []:
        if entry["name"] == resolved:
            return entry.get("display_name") or resolved
    return f"{configured} (not installed)"


def _actions_for(registry, default, name):
    """Which actions this profile can actually take, with live labels.

    Only one half of a toggle is ever offered — ``/services`` has always done
    this and the rest of the commands showed both, so half of every menu was a
    no-op the user had to know to avoid. The action *value* stays stable and
    only the label moves, so ``run`` still branches on one name.
    """
    if name == "add":
        return ["edit"], ["Edit"]
    row = _profile_row(registry, name)
    actions = ["edit"]
    labels = ["Edit"]
    if default != name:
        actions.append("set_default")
        labels.append("Set default")
    actions.append("unload" if row.get("loaded") else "load")
    labels.append("Unload it" if row.get("loaded") else "Load it")
    actions.append("remove")
    labels.append("Remove")
    return actions, labels


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


def _describe(sdk, registry, profiles, default, name):
    profile = profiles.get(name)
    if not profile:
        return "Action"
    row = _profile_row(registry, name)
    mark = " (default)" if default == name else ""
    context = int(profile.get("llm_context_size", 0) or 0)
    context_text = (
        "0 (reactive compaction)" if context == 0 else f"{context:,}")
    capabilities = ", ".join(
        key for key, value in (
            profile.get("llm_capabilities") or {}).items() if value
    ) or "none declared"
    return sdk.md.card(f"{name}{mark}", [
        ("Status", "Loaded" if row.get("loaded") else "Unloaded"),
        ("Backend", _backend_label(
            registry, profile.get("llm_service_class", ""))),
        ("Context", context_text),
        ("Native attachments", capabilities),
    ])


def _model_label(default, name):
    if name == "add":
        return "Add profile"
    return f"{name} (default)" if default == name else name


def _show(sdk, registry, profiles, default):
    """Every profile at a glance.

    This command had no listing at all: the picker is required, so ``run``'s
    no-name branch was unreachable and the only overview was a marker per
    option. Which profile is loaded, on what backend, with how much context is
    exactly the comparison a person opens ``/llm`` to make.
    """
    if not profiles:
        return "No LLM profiles are configured."
    rows = []
    for name in sorted(profiles):
        context = int((profiles[name] or {}).get("llm_context_size", 0) or 0)
        rows.append((
            f"{name} (default)" if name == default else name,
            "Loaded" if _profile_row(registry, name).get("loaded")
            else "Unloaded",
            _backend_label(
                registry, (profiles[name] or {}).get("llm_service_class", "")),
            "reactive" if context == 0 else f"{context:,}",
        ))
    return "LLM profiles:\n\n" + sdk.md.table(
        ["Profile", "Status", "Backend", "Context"], rows,
        leading_blank=False)


def _default_prompt(sdk, registry, profiles, default):
    return (
        _show(sdk, registry, profiles, default)
        + "\n\nSelect an LLM profile, or add a new one."
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
