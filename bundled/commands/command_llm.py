"""Slash command plugin for `/llm`."""

from guest.bases import BaseCommand
from guest.forms import FormStep


PROFILE_FIELDS = [
    "llm_endpoint", "secret_llm_api_key", "llm_context_size",
    "llm_service_class", "llm_capability_image",
    "llm_capability_audio", "llm_capability_video",
]
# What a call carries beyond the connection: how hard the model should think,
# and anything else the provider takes. Editable but never *asked for* when
# adding a profile — a wizard that interrogates you about sampling before you
# have sent one message is worse than a menu entry you find when you want it.
# Kept out of ``PROFILE_FIELDS`` rather than filtered out of the add flow,
# because that list is what ``_profile`` builds a new profile from: a name in
# it is a key written to every profile, empty or not.
#
# ``extra_param`` is not a stored key. It is the guided route into
# ``llm_extra_params``: pick a parameter the model takes, then pick a value.
# It replaced a dedicated ``llm_reasoning_effort`` entry, which was one member
# of that dict promoted to a field of its own because it was the only member
# anybody could name. The backend can name the rest now, so the special case
# stopped paying for itself — and a menu that offers reasoning and hides
# ``temperature`` teaches that reasoning is the only thing there is.
#
# The raw JSON entry stays beside it, for clearing several at once and for a
# value no picker can express.
EXTRA_PARAM = "extra_param"
TUNING_FIELDS = [EXTRA_PARAM, "llm_extra_params"]
FIELDS = ["llm_model_name", *PROFILE_FIELDS, *TUNING_FIELDS]
FIELD_LABELS = [
    "Model name", "Endpoint", "API key", "Context size",
    "Service class", "Images", "Audio", "Video",
    "Configure extra parameter", "Extra parameters (raw JSON)",
]
CAPABILITY_FIELDS = {
    "llm_capability_image": "image",
    "llm_capability_audio": "audio",
    "llm_capability_video": "video",
}
# This command's spelling of "send nothing", which is a real choice for any
# parameter now that the kernel supplies a level for a profile that says
# nothing. Stored as JSON ``null`` rather than the word, so it cannot be
# confused with ``none`` — a level several providers accept, meaning "think
# as little as possible".
OFF = "off"
# What "none of these" is called in the two menus that can be incomplete.
# Both lists come from a backend introspecting somebody else's catalogue, so
# neither can ever be assumed complete — and a menu with no way past it turns
# a missing entry into an unusable command.
CUSTOM = "custom"
# "I am finished adding parameters." Present because the add flow offers this
# step unprompted and most profiles want none of it, so there has to be a way
# past that is not Cancel — Cancel abandons the profile.
NONE = "__none__"
# Keys the profile sets through its own fields, or that are the call itself.
# The backend merges extras with ``setdefault``, so one of these here *wins*
# over the profile silently — and an ``api_key`` also lands in plaintext
# config instead of behind the ``secret_`` prefix that declares it a
# credential. Refused where somebody typed it, rather than surfacing later as
# a call going somewhere unexpected.
RESERVED_PARAMS = {
    "api_key": "the API key field",
    "api_base": "the Endpoint field",
    "model": "the profile's model name",
    "messages": "the conversation itself",
    "tools": "the agent's tool catalog",
    "stream": "the kernel, per call",
}
# The effort ladder is no longer declared here. It arrives per model from the
# backend, as the ``choices`` on the ``reasoning_effort`` row — so a provider
# offering a level this file never heard of can still be set to it, and one
# that takes no effort at all does not advertise the setting as though it did.


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
    # Deliberately no ``net.http``. The authoritative model catalogue is the
    # one the endpoint serves, and this command cannot fetch it: approval is
    # evaluated on *completed* form arguments, so everything a form does runs
    # ungranted, and a dialog raised mid-form deadlocks against the session
    # lock the form is holding. Declaring the Request would not have helped,
    # because the grant does not exist yet at that point. So the model menu
    # comes from what the backend knows offline, and an endpoint it has no
    # index for falls through to typing the name.
    requests = [
        "config.read", "config.write", "plugin.list",
        "llm.list", "llm.load", "llm.unload",
    ]
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
            return steps + _add_steps(sdk, registry, args)
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
            profile = profiles.get(args.get("model_name")) or {}
            steps.append(FormStep(
                "field", "Choose which LLM setting to edit.", True,
                enum=FIELDS, enum_labels=FIELD_LABELS))
            if field == EXTRA_PARAM:
                # The same two questions the add flow asks, against the model
                # this profile already names. Required here: reaching this
                # entry is itself the statement that you want to set one.
                steps.extend(_extra_param_steps(
                    sdk, args, args.get("model_name") or "",
                    profile.get("llm_endpoint") or "", profile,
                    required=True))
            elif field:
                steps.append(FormStep(
                    "value",
                    _value_prompt(field, _backend_names(registry), profile),
                    True, _value_type(field)))
        return steps

    def run(self, sdk, args):
        profiles = sdk.config.read("llm_profiles") or {}
        default = sdk.config.read("default_llm_profile") or ""
        name = args.get("model_name")
        if name == "add":
            name = _chosen_model(args)
            if not name:
                return "Model name is required."
            first = not profiles
            profiles[name] = _profile(args)
            chosen = _chosen_param(args)
            if chosen:
                profiles[name]["llm_extra_params"] = {
                    chosen: _extra_value(args)}
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
            elif field == EXTRA_PARAM:
                chosen = _chosen_param(args)
                if not chosen:
                    return "No parameter chosen."
                refused = _reserved({chosen: None})
                if refused:
                    return refused
                # Into the dict, beside whatever else the profile sends. A
                # ``None`` stays: it is the stored form of "send nothing", and
                # dropping it would hand back the kernel's default.
                profiles[name].setdefault("llm_extra_params", {})[
                    chosen] = _extra_value(args)
            elif field in TUNING_FIELDS:
                # An empty dict is the absence of the key, never a stored
                # ``{}``: a profile carrying one reads as configured to
                # anything scanning config by hand, and every profile written
                # before this existed carries nothing.
                value = _coerce(field, args.get("value"))
                refused = _reserved(value)
                if refused:
                    return refused
                if value:
                    profiles[name][field] = value
                else:
                    profiles[name].pop(field, None)
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


def _add_steps(sdk, registry, args):
    """Adding a profile, asked from least specific to most.

    Provider, then endpoint, then model, then the parameters that model takes
    — each answer narrowing what the next one offers. The ordering is forced
    rather than chosen: listing models means asking the endpoint, and asking
    the endpoint means already holding its URL and key.

    Every step degrades to a typed value, and that is the *common* path rather
    than a fallback. Aggregators appear in no provider list, plenty of
    endpoints serve no catalogue, and a backend need not introspect at all —
    so a step whose lookup came back empty simply asks for the value the way
    it always did. Nothing here treats an empty answer as a problem.
    """
    backends = _backend_names(registry)
    provider = args.get("llm_provider") or ""
    endpoint = (args.get("llm_endpoint") or "").strip()
    steps = [
        FormStep(
            "llm_service_class",
            "Choose how Second Brain should connect to this model.",
            True, enum=backends, default=backends[0]),
    ]

    providers = _providers(sdk)
    if providers:
        ids = [row["id"] for row in providers] + [CUSTOM]
        labels = [row.get("label") or row["id"] for row in providers]
        steps.append(FormStep(
            "llm_provider",
            "Which provider is this model served by? Choose "
            f"`{CUSTOM}` for anything reached through its own URL, which "
            "includes every multi-provider gateway.",
            True, enum=ids, enum_labels=labels + ["Something else"]))

    resolved = _endpoint_for(sdk, provider)
    steps.append(FormStep(
        "llm_endpoint",
        # The URL is written into the prompt as well as pre-filled. A default
        # only helps if it is on screen: a client that does not render one
        # shows an empty box under a sentence promising it was filled in,
        # which is worse than never claiming it.
        (f"Base URL for {provider}. Its default is `{resolved}` — that is "
         "already filled in, so continue unless you reach this provider "
         "somewhere else."
         if resolved else
         "Enter the provider base URL. Nothing here knows this provider's "
         "default, so a blank almost certainly means the model will not be "
         "found."),
        False, default=resolved, prompt_when_missing=True))
    steps.append(FormStep(
        "secret_llm_api_key",
        # Not "the provider default" — no provider has a default API key, and
        # saying so invited somebody to leave it blank and wait for a 401.
        # Blank is right for exactly two cases, and both are worth naming.
        "Enter the API key, or the name of an environment variable holding "
        "it. Leave it blank only for a provider that needs no key, such as a "
        "local server, or when the key is already in the environment under "
        "the name this provider looks for.",
        False, default="", prompt_when_missing=True))

    # The model step only becomes a menu once there is something to ask.
    # ``llm_endpoint`` is answered by the step above, so on the pass that
    # renders this one it is already in ``args``.
    catalogue = _models(sdk, endpoint, args.get("secret_llm_api_key") or "",
                        "" if provider == CUSTOM else provider)
    if catalogue:
        names = [row["name"] for row in catalogue] + [CUSTOM]
        labels = [row.get("label") or row["name"] for row in catalogue]
        steps.append(FormStep(
            "new_model_name",
            "Which model? The name is stored exactly as shown, prefix "
            f"included, so it routes correctly. Choose `{CUSTOM}` to type "
            "one that is not listed.",
            True, enum=names, enum_labels=labels + ["Type it myself"]))
        if args.get("new_model_name") == CUSTOM:
            steps.append(FormStep(
                "custom_model_name",
                "Enter the model name exactly, including the provider prefix "
                "when one is needed (for example `openai/gpt-4o-mini`).",
                True))
    else:
        steps.append(FormStep(
            "new_model_name",
            "Enter the model name exactly, including provider prefix when "
            "needed (for example `openai/gpt-4o-mini` or "
            "`anthropic/claude-3-5-sonnet-latest`).",
            True))

    steps.append(FormStep(
        "llm_context_size",
        "Optional context window size in tokens. Use 0 for dynamic "
        "compaction or if unknown.",
        False, "integer", default=0, prompt_when_missing=True))
    steps.extend(_capability_steps())
    # Last, and skippable. Every step before this one is needed to reach the
    # model at all; this one is tuning, so it defaults to "Done" and a profile
    # that answers nothing is complete. It is offered here rather than left to
    # a second trip through Edit because the reasoning level is the setting
    # people most often want from the start, and the flow already knows which
    # model it is asking about.
    steps.extend(_extra_param_steps(
        sdk, args, _chosen_model(args), endpoint, {}))
    return steps


def _extra_param_steps(sdk, args, model, endpoint, profile, required=False):
    """Pick a provider parameter, then pick its value. Used by add and by edit.

    One flow in both places because they are the same act: the add flow
    reaches it once with the model just chosen, and ``/llm`` -> Edit reaches
    it again later against the same model. Two spellings of it would drift,
    and the second one would be the one nobody tested.

    The menu is the backend's answer for *this* model, so it lists what the
    model actually takes rather than a fixed vocabulary — and it keeps
    unsupported entries, labelled, because that answer is a lookup in
    somebody else's table and those have gaps. ``custom`` is always last for
    the same reason: an endpoint the table has never heard of still has
    parameters, and refusing to let them be typed would make this menu a
    smaller version of the problem it replaces.
    """
    rows = _param_options(sdk, model, endpoint)
    current = (profile or {}).get("llm_extra_params") or {}
    names = [row["name"] for row in rows]
    choices = names + [CUSTOM] + ([] if required else [NONE])
    labels = [_param_label(row, current) for row in rows]
    labels += ["Something else — type its name"]
    labels += ([] if required else ["Done — no more parameters"])

    steps = [FormStep(
        EXTRA_PARAM,
        ("Which provider parameter do you want to set? These are forwarded on "
         "every call this profile makes."
         + ("" if rows else
            "\n\nNothing could be listed for this model, so type the name "
            "yourself.")),
        True, enum=choices, enum_labels=labels,
        default=None if required else NONE)]

    picked = args.get(EXTRA_PARAM)
    if picked == NONE or not picked:
        return steps
    if picked == CUSTOM:
        steps.append(FormStep(
            "custom_param_name",
            "Enter the parameter name exactly as the provider spells it, for "
            "example `enable_thinking`.",
            True))
    chosen = _chosen_param(args)
    spec = next((row for row in rows if row["name"] == chosen), None)
    if chosen:
        steps.append(_value_step(chosen, spec, current))
    return steps


def _value_step(name, spec, current):
    """The step that collects one parameter's value, shaped by its kind."""
    kind = (spec or {}).get("kind") or "text"
    held = current.get(name, "__unset__")
    now = ("" if held == "__unset__" else
           f"Currently `off` (nothing sent).\n\n" if held is None else
           f"Currently `{held}`.\n\n")
    note = (spec or {}).get("note") or ""
    warning = f"\n\nNote: {note}" if note else ""
    if kind == "choice" and (spec or {}).get("choices"):
        levels = list(spec["choices"]) + [OFF]
        return FormStep(
            "extra_value",
            f"{now}Choose a value for `{name}`.{warning}",
            True, enum=levels,
            enum_labels=[str(item) for item in spec["choices"]]
                        + ["Off — send nothing"])
    if kind == "bool":
        return FormStep("extra_value", f"{now}Set `{name}` to?{warning}",
                        True, "boolean")
    if kind == "number":
        return FormStep(
            "extra_value",
            f"{now}Enter a number for `{name}`, or `off` to send "
            f"nothing.{warning}",
            True)
    return FormStep(
        "extra_value",
        f"{now}Enter a value for `{name}`, or `off` to send nothing. JSON is "
        f"accepted, so `true`, `2`, or `{{\"type\": \"enabled\"}}` all "
        f"work.{warning}",
        True)


def _param_label(row, current):
    """One menu entry: the name, what it is set to, and any caveat."""
    name = row["name"]
    label = row.get("label") or name
    if name in current:
        held = current[name]
        label += " = off" if held is None else f" = {held}"
    if not row.get("supported", True):
        label += " (not listed for this model)"
    return label


def _chosen_param(args):
    """The parameter name the flow settled on, menu or typed."""
    picked = (args.get(EXTRA_PARAM) or "").strip()
    if picked == CUSTOM:
        return (args.get("custom_param_name") or "").strip()
    return "" if picked in ("", NONE) else picked


def _param_options(sdk, model, endpoint):
    """What the backend says this model accepts. ``[]`` when it cannot say."""
    if not model:
        return []
    try:
        answer = sdk.llm.list(params=model, endpoint=endpoint or "") or {}
    except sdk.Failed:
        return []
    return [row for row in (answer.get("params") or [])
            if isinstance(row, dict) and row.get("name")]


def _extra_value(args):
    """The value to store, from whatever the value step collected.

    ``off`` becomes ``None``, which is the stored form of "send nothing" and
    the one value that must survive: dropping the key instead would hand back
    the kernel's default, which is the opposite of what was asked for.

    Everything else is read as JSON when it can be, so `2`, `true` and an
    object all arrive as themselves rather than as strings. A provider that
    wants the literal text still gets it, because a bare word is not valid
    JSON and falls through unchanged.
    """
    raw = args.get("extra_value")
    if isinstance(raw, str):
        text = raw.strip()
        if text.lower() in (OFF, "null", "none — send nothing"):
            return None
        try:
            import json

            return json.loads(text)
        except (TypeError, ValueError):
            return text
    return raw


def _chosen_model(args):
    """The model name the add flow settled on, menu or typed."""
    picked = (args.get("new_model_name") or "").strip()
    if picked == CUSTOM:
        return (args.get("custom_model_name") or "").strip()
    return picked


def _providers(sdk):
    """Step one, or ``[]`` when no backend can name any."""
    try:
        return (sdk.llm.list(providers=True) or {}).get("providers") or []
    except sdk.Failed:
        return []


def _models(sdk, endpoint, api_key, provider):
    """Step two. Needs an endpoint or a provider; answers ``[]`` without both.

    Guarded rather than always asked, because this is the one question that
    can reach the network — and a form redraws on every step.
    """
    if not endpoint and not provider:
        return []
    try:
        answer = sdk.llm.list(models=endpoint, key=api_key,
                              provider=provider) or {}
    except sdk.Failed:
        return []
    return answer.get("models") or []


def _endpoint_for(sdk, provider):
    """The chosen provider's own base URL, or ``""``.

    A second, narrower call rather than a field read off the menu, because the
    menu deliberately carries no endpoints: resolving one is expensive enough
    that doing it a hundred and fifty times to draw a list is what made this
    hang. Asked here, about one provider, it is a single lookup.

    And it is the whole point of the provider step existing. Without it that
    step collects a name and the next step asks for the URL the name was
    supposed to supply - two questions, where answering the second already
    required knowing everything the first was meant to spare you.
    """
    if not provider or provider == CUSTOM:
        return ""
    try:
        rows = (sdk.llm.list(providers=provider) or {}).get("providers") or []
    except sdk.Failed:
        return ""
    for row in rows:
        if str(row.get("id") or "").lower() == provider.lower():
            return str(row.get("endpoint") or "")
    return ""


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


def _reserved(params):
    """Refuse extras that would override the profile's own settings.

    Returns a sentence naming each offender and where it is really set, or
    an empty string when there is nothing wrong. A message rather than a
    silent drop: the person meant something by typing it, and which field
    they wanted is the part worth telling them.
    """
    named = [key for key in RESERVED_PARAMS if key in (params or {})]
    if not named:
        return ""
    lines = "\n".join(
        f"- `{key}` is set by {RESERVED_PARAMS[key]}" for key in named)
    return ("These parameters cannot be set here:\n\n" + lines
            + "\n\nRemove them and try again.")


def _coerce(field, value):
    if field == "llm_context_size":
        return int(value or 0)
    if field in CAPABILITY_FIELDS:
        return (
            value if isinstance(value, bool)
            else str(value).strip().lower() in {"true", "yes", "1", "y"}
        )
    if field == "llm_extra_params":
        # The step is declared ``object``, so the kernel has already parsed
        # the JSON and re-prompted if it would not — this only has to decide
        # what a non-object means, and it means nothing to send.
        return dict(value) if isinstance(value, dict) else {}
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
    pairs = [
        ("Status", "Loaded" if row.get("loaded") else "Unloaded"),
        ("Backend", _backend_label(
            registry, profile.get("llm_service_class", ""))),
        ("Context", context_text),
        ("Native attachments", capabilities),
    ]
    # Reasoning is always shown, because there is always an answer now: a
    # profile that says nothing still thinks at whatever the kernel supplies,
    # and a card that stayed silent would be the only place you could not
    # find that out.
    pairs.append(("Reasoning", _effort_text(profile, row)))
    # Then everything else the profile sends, one row each with its own
    # caveat. Previously this was a single JSON blob, which is unreadable at a
    # glance and had nowhere to put the warning that a value is being
    # discarded — the warning only existed for reasoning because reasoning was
    # the only one with a row of its own.
    for key, value in sorted((profile.get("llm_extra_params") or {}).items()):
        if key == "reasoning_effort":
            continue
        shown = "off (nothing sent)" if value is None else str(value)
        note = _param_note(row, key)
        pairs.append((key, f"{shown} — {note}" if note else shown))
    return sdk.md.card(f"{name}{mark}", pairs)


def _effort_text(profile, row=None):
    """What this profile's reasoning effort reads as, in all four states.

    The fourth is new and is the reason this takes a registry row: a level can
    be set, displayed, and *discarded before the call*, which used to read
    exactly like one that was working. A dial that cannot be trusted is worse
    than no dial, so when the backend says the value will not survive, the
    card says so beside it.

    Worded as what happens to the *value*, never as a claim about the model.
    The case that forced this rule was a provider whose model reasons perfectly
    well and whose entry in the middleman's table simply omits the parameter —
    "not supported" would have been a lie, and one the user could not check.
    """
    extras = profile.get("llm_extra_params") or {}
    if "reasoning_effort" not in extras:
        text = "default"
    elif extras["reasoning_effort"] is None:
        return "off (nothing sent)"
    else:
        text = str(extras["reasoning_effort"])
    note = _param_note(row, "reasoning_effort")
    return f"{text} — {note}" if note else text


def _param_note(row, param):
    """The caveat for one param, or ``""`` when there is nothing to say.

    Reads the *note* rather than the boolean beside it, and that is the whole
    of the logic: the registry writes a note exactly when a param needs one,
    and there are two such cases pointing opposite ways — a value being
    discarded, and a value being forced through that the provider may refuse.
    Branching on the boolean would print the first and swallow the second,
    which is the warning somebody most needs before a call fails.

    Silent in three cases that look identical from here and should: the
    backend cannot introspect, the profile's box is closed so nobody has
    asked, and the setting is simply fine.
    """
    entry = ((row or {}).get("param_status") or {}).get(param)
    if not isinstance(entry, (list, tuple)) or len(entry) < 2:
        return ""
    return str(entry[1] or "")


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
    if field == "llm_extra_params":
        # Declared rather than parsed in ``run``, so bad JSON is rejected at
        # the step and asked again. Returning a sentence from the handler
        # would mean re-running the whole command to fix a typo.
        return "object"
    return "string"


def _value_prompt(field, backends, profile=None):
    """What to ask for one field, and what it is set to now.

    The current value matters most for the two tuning fields: a dict is
    tedious to retype from memory, and there is nowhere else in the flow that
    shows it before you overwrite it.
    """
    current = _current_value(field, profile or {})
    return current + {
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
        "llm_extra_params": (
            "Enter a JSON object of extra provider parameters, for example "
            "`{\"temperature\": 0.2}`. These are forwarded verbatim on every "
            "call this profile makes, and a `null` value means send nothing "
            "for that parameter. Enter `{}` to clear them."),
    }.get(field, "Enter the new value.")


def _current_value(field, profile):
    """A one-line "currently:" preamble, or nothing when there is nothing."""
    if field == "llm_extra_params":
        import json

        value = profile.get(field) or {}
        return f"Currently: {json.dumps(value)}\n\n" if value else ""
    return ""
