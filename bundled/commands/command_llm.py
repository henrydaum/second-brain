"""Slash command plugin for `/llm`."""

from guest.bases import BaseCommand
from guest.forms import FormStep


# ══════════════════════════════════════════════════════════════════════
# A profile has two halves, and the difference runs through this whole file.
#
# **Its settings** are how to reach the model: endpoint, key, context size,
# which backend, what it can see. Every profile has all of them, whether or
# not anybody chose a value, so they can be edited and never removed.
#
# **Its provider parameters** are how the model should behave: temperature,
# reasoning effort, anything else the provider takes. They live together in
# ``llm_extra_params``, the kernel names no member of it, and each one is
# there only because somebody put it there — so each can be removed.
#
# The second half is not asked for when adding a profile. Everything in the
# first half is needed to reach the model at all; tuning is not, and a wizard
# that interrogates you about sampling before you have sent one message is
# worse than a menu entry you find when you want it.
# ══════════════════════════════════════════════════════════════════════

PROFILE_FIELDS = [
    "llm_endpoint", "secret_llm_api_key", "llm_context_size",
    "llm_service_class", "llm_capability_image",
    "llm_capability_audio", "llm_capability_video",
]
BASE_FIELDS = ["llm_model_name", *PROFILE_FIELDS]
BASE_LABELS = [
    "Model name", "Endpoint", "API key", "Context size",
    "Backend", "Images", "Audio", "Video",
]
CAPABILITY_FIELDS = {
    "llm_capability_image": "image",
    "llm_capability_audio": "audio",
    "llm_capability_video": "video",
}

# The edit menu is a flat list of field names, so a parameter needs a prefix
# to sit in it: a provider is free to call something ``llm_endpoint``, and
# without this that parameter would impersonate a profile field.
PARAM_PREFIX = "param:"
# Adding one, as opposed to opening one already there. Not a stored key —
# the guided route into ``llm_extra_params``: pick a parameter, pick a value.
EXTRA_PARAM = "extra_param"
# What to do with a parameter already set. Its own question, because the only
# other place to put "delete" is inside the value — as a word, which then
# cannot be told apart from a provider that accepts that word as a value.
EDIT = "edit"
REMOVE = "remove"
# "None of these." Both menus that offer it are built from a backend reading
# somebody else's catalogue, so neither can be assumed complete, and a menu
# with no way past turns a missing entry into an unusable command.
CUSTOM = "custom"
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
            field_names, field_labels = _fields(profile)
            steps.append(FormStep(
                "field", "Choose which LLM setting to edit.", True,
                enum=field_names, enum_labels=field_labels))
            held = _param_field(field)
            if held:
                # Change it or delete it, asked before either happens. Two
                # questions where one nearly did, because the one had to carry
                # "delete" as a word inside a free-text value.
                steps.append(FormStep(
                    "param_action",
                    f"What do you want to do with `{held}`?", True,
                    enum=[EDIT, REMOVE],
                    enum_labels=["Change its value",
                                 "Remove it from this profile"]))
                if args.get("param_action") == EDIT:
                    spec = next(
                        (row for row in _param_options(
                            sdk, args.get("model_name") or "",
                            profile.get("llm_endpoint") or "")
                         if row["name"] == held), None)
                    steps.append(_value_step(
                        held, spec, profile.get("llm_extra_params") or {}))
            elif field == EXTRA_PARAM:
                # Adding one: which parameter, then its value.
                steps.extend(_extra_param_steps(
                    sdk, args, args.get("model_name") or "",
                    profile.get("llm_endpoint") or "", profile))
            elif field:
                backends = _backend_names(registry)
                steps.append(FormStep(
                    "value",
                    _value_prompt(field, profile, registry),
                    True, _value_type(field),
                    enum=_value_enum(field, backends),
                    enum_labels=_value_enum_labels(
                        field, backends, registry)))
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
            sdk.config.write("llm_profiles", profiles, scope="plugin")
            if first:
                sdk.config.write(
                    "default_llm_profile", name, scope="plugin")
            return (
                f"Added LLM profile: {name}\n\n"
                "That is everything needed to reach it. To tune how it "
                "behaves — reasoning effort, temperature, anything else this "
                "provider takes — run `/llm`, pick this profile, choose "
                "**Edit**, then **Add a parameter**.")
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
            elif field == EXTRA_PARAM or _param_field(field):
                chosen = _param_field(field) or _chosen_param(args)
                if not chosen:
                    return "No parameter chosen."
                refused = _reserved({chosen: None})
                if refused:
                    return refused
                extras = profiles[name].setdefault("llm_extra_params", {})
                if args.get("param_action") == REMOVE:
                    extras.pop(chosen, None)
                    # An empty dict is the absence of the key, never a stored
                    # ``{}``: a profile carrying one reads as configured to
                    # anything scanning the file, and every profile written
                    # before extras existed carries nothing.
                    if not extras:
                        profiles[name].pop("llm_extra_params", None)
                    sdk.config.write("llm_profiles", profiles, scope="plugin")
                    if was_loaded:
                        sdk.llm.load(name)
                    return f"Removed `{chosen}` from {name}."
                # Into the dict, beside whatever else this profile sends.
                extras[chosen] = _extra_value(args)
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

    Backend, then provider, then endpoint, then model — each answer
    narrowing what the next one offers. The ordering is forced rather than
    chosen: listing models means asking the endpoint, and asking the endpoint
    means already holding its URL and key.

    Three of those arrive already answered, which is the whole point of asking
    in this order: the provider supplies its own endpoint, the endpoint
    supplies its models with the prefix each one needs, and the model supplies
    its context window.

    Every step degrades to a typed value, and that is the *common* path rather
    than a fallback. Gateways appear in no provider list, plenty of endpoints
    publish no catalogue, and a backend need not introspect at all — so a
    step whose lookup came back empty simply asks for the value the way it
    always did. Nothing here treats an empty answer as a problem.
    """
    backends = _backend_names(registry)
    provider = args.get("llm_provider") or ""
    endpoint = (args.get("llm_endpoint") or "").strip()
    steps = [
        FormStep(
            "llm_service_class",
            "Choose how Second Brain connects to this model.",
            True, enum=backends, default=backends[0]),
    ]

    providers = _providers(sdk)
    if providers:
        ids = [row["id"] for row in providers] + [CUSTOM]
        labels = [row.get("label") or row["id"] for row in providers]
        steps.append(FormStep(
            "llm_provider",
            "Which provider serves this model? Pick **Something else** for "
            "anything reached through its own URL, which includes every "
            "multi-provider gateway.",
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
            "included, so it routes correctly. Pick **Type it myself** for "
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

    window = _context_size(sdk, _chosen_model(args), endpoint)
    steps.append(FormStep(
        "llm_context_size",
        (f"Context window in tokens. This model reports {window:,}, already "
         "filled in — change it only if you know better."
         if window else
         "Context window size in tokens. Nothing could look this up for this "
         "model, so enter it if you know it, or use 0 to let the kernel "
         "compact reactively instead."),
        False, "integer", default=window, prompt_when_missing=True))
    steps.extend(_capability_steps())
    # And that is the whole of setting a model up. Provider parameters are
    # deliberately not asked for here: everything above is needed to reach the
    # model at all, while temperature and reasoning are tuning, and a wizard
    # that interrogates you about sampling before you have sent one message is
    # worse than a menu entry you find when you want it. ``run`` says where
    # that entry is, since a setting nobody can find is the same as no
    # setting.
    return steps


def _extra_param_steps(sdk, args, model, endpoint, profile):
    """Adding a parameter: pick which one, then pick its value.

    The menu is the backend's answer for *this* model, so it lists what the
    model actually takes rather than a fixed vocabulary, and it lists only
    that. Offering a parameter the backend says this model does not take is
    offering a setting that does nothing, and a menu of mostly-inert entries
    teaches you to distrust all of them.

    Hiding is safe here and would not be safe in ``params`` itself, which
    reports and never gates: a *lookup* may not decide what is possible, but a
    *menu* may decide what is worth suggesting — and only because ``custom``
    at the end of it keeps everything reachable anyway.

    Parameters this profile already sets are left out too, and that is the
    restructure paying off rather than a second rule: each of them has its own
    row in the edit menu now, showing its value and opening straight on it.
    This menu once had to keep an unsupported-but-set parameter visible so it
    could still be fixed — it was the only way to reach one — and that
    exception left with the rows' arrival.

    There is no "skip" entry. Reaching this is choosing **Add a parameter**
    from the edit menu, which is already the statement that you want one;
    Back is how you leave.
    """
    current = (profile or {}).get("llm_extra_params") or {}
    rows = [row for row in _param_options(sdk, model, endpoint)
            if row.get("supported", True) and row["name"] not in current]
    names = [row["name"] for row in rows]
    choices = names + [CUSTOM]
    labels = [row.get("label") or row["name"] for row in rows]
    labels += ["Something else — type its name"]

    steps = [FormStep(
        EXTRA_PARAM,
        ("Which parameter do you want to set? These are sent on every call "
         "this profile makes."
         + ("" if rows else
            "\n\nNothing could be listed for this model, so pick "
            "**Something else** and type the name.")),
        True, enum=choices, enum_labels=labels)]

    picked = args.get(EXTRA_PARAM)
    if not picked:
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
    """The step that collects one parameter's value, shaped by its kind.

    Nothing here is a word you type to mean something other than itself.
    Declining a parameter is Remove, on the step before this one; there is no
    second spelling of it hidden in the value.
    """
    kind = (spec or {}).get("kind") or "text"
    now = ("" if name not in current else
           f"Currently `{_shown(current[name])}`.\n\n")
    note = (spec or {}).get("note") or ""
    warning = f"\n\nNote: {note}" if note else ""
    if kind == "choice" and (spec or {}).get("choices"):
        return FormStep(
            "extra_value",
            f"{now}Choose a value for `{name}`.{warning}",
            True, enum=[str(item) for item in spec["choices"]])
    if kind == "bool":
        return FormStep(
            "extra_value", f"{now}Set `{name}` to?{warning}", True,
            enum=["true", "false"], enum_labels=["True", "False"])
    if kind == "number":
        return FormStep(
            "extra_value", f"{now}Enter a number for `{name}`.{warning}", True)
    return FormStep(
        "extra_value",
        f"{now}Enter a value for `{name}`. JSON is accepted, so `true`, `2`, "
        f"or `{{\"type\": \"enabled\"}}` all work.{warning}",
        True)


def _shown(value):
    """How a stored parameter value reads on screen.

    One helper because it appears in four places — both menus, the value
    step's preamble and the profile card — and they had drifted into three
    spellings of the same null. A parameter that reads "off" in one list and
    "(sends nothing)" in the next looks like two different settings.
    """
    return "(sends nothing)" if value is None else str(value)


def _chosen_param(args):
    """The parameter name the flow settled on, menu or typed."""
    picked = (args.get(EXTRA_PARAM) or "").strip()
    if picked == CUSTOM:
        return (args.get("custom_param_name") or "").strip()
    return picked


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

    Read as JSON when it can be, so `2`, `true`, `null` and an object all
    arrive as themselves rather than as strings. A provider that wants literal
    text still gets it, because a bare word is not valid JSON and falls
    through unchanged.

    No word means anything but itself. This used to read ``off`` as an
    instruction, which made ``off`` unsettable for a provider that takes it as
    a real value.
    """
    raw = args.get("extra_value")
    if isinstance(raw, str):
        text = raw.strip()
        try:
            import json

            return json.loads(text)
        except (TypeError, ValueError):
            return text
    return raw


def _context_size(sdk, model, endpoint):
    """The model's own input context window, or ``0``.

    ``0`` is not a failure here: the kernel reads it as "compact reactively",
    which works. That is why a blank is offered rather than a guess — a wrong
    window is budgeted against every turn, so it either wastes most of the
    context or overflows it, and both look like a bad model rather than a bad
    number.
    """
    if not model:
        return 0
    try:
        answer = sdk.llm.list(info=model, endpoint=endpoint or "") or {}
    except sdk.Failed:
        return 0
    try:
        return int((answer.get("info") or {}).get("context_size") or 0)
    except (TypeError, ValueError):
        return 0


def _fields(profile):
    """The edit menu for one profile: its settings, its parameters, then add.

    Configured parameters sit in the same list as endpoint and context size,
    because from the reader's side they are the same kind of thing — something
    this profile is set to. Keeping them behind a separate "extra parameters"
    entry meant the only way to see what a profile sent was to open a JSON
    blob and read it.

    They differ in exactly one way, and it is the reason the prefix exists
    rather than the two being merged outright: a parameter can be removed, and
    a profile field cannot. There is no meaningful "no endpoint" distinct from
    an empty one, and every profile has a context size whether or not anybody
    chose it.
    """
    extras = sorted((profile or {}).get("llm_extra_params") or {})
    names = list(BASE_FIELDS)
    labels = list(BASE_LABELS)
    for key in extras:
        names.append(PARAM_PREFIX + key)
        labels.append(f"{key} = {_shown(profile['llm_extra_params'][key])}")
    names.append(EXTRA_PARAM)
    labels.append("Add a parameter")
    return names, labels


def _param_field(field):
    """The parameter an edit-menu entry names, or ``""`` for a profile field."""
    return (field or "")[len(PARAM_PREFIX):] if (
        field or "").startswith(PARAM_PREFIX) else ""


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
            f"Can this model read {label} natively? Skip if unsure — this "
            f"only controls whether {label} are sent to it as files rather "
            "than as text.",
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
    # Everything the profile sends, one row each with its own caveat. This
    # was a single JSON blob with reasoning promoted out of it into a row of
    # its own — unreadable at a glance, and the promotion was the only reason
    # one parameter could carry a warning and the rest could not.
    for key, value in sorted((profile.get("llm_extra_params") or {}).items()):
        shown = _shown(value)
        note = _param_note(row, key)
        pairs.append((key, f"{shown} — {note}" if note else shown))
    return sdk.md.card(f"{name}{mark}", pairs)


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
    return "string"


def _value_enum(field, backends):
    """The closed choices for a profile field, or ``None`` for free text.

    The add flow offers a picker for the backend and yes/no for each
    capability; editing the same settings offered neither, so changing your
    backend meant typing a class name from memory and getting
    "Unknown LLM profile"-shaped failures for a typo. The two flows write the
    same keys and should ask the same way.
    """
    if field == "llm_service_class":
        return list(backends)
    return None


def _value_enum_labels(field, backends, registry):
    """Human labels for whatever ``_value_enum`` offered.

    Backends are shown by their declared display name, the same as everywhere
    else a person picks one — the stored value is a class name and reading
    it is not the user's job.
    """
    if field == "llm_service_class":
        return [_backend_label(registry, name) for name in backends]
    return None


def _value_prompt(field, profile=None, registry=None):
    """What to ask for one field, and what it is set to now.

    Worded to match the add flow question for question. They write the same
    keys, and two descriptions of one setting is how a person ends up
    believing there are two settings.
    """
    return _current_value(field, profile or {}, registry) + {
        "llm_model_name": (
            "Enter the model name, exactly as the provider spells it and "
            "including any prefix it needs."),
        "llm_endpoint": (
            "Enter the provider base URL. Leave it blank only if this backend "
            "already knows where to reach the provider."),
        "secret_llm_api_key": (
            "Enter the API key, or the name of an environment variable "
            "holding it. Leave it blank only for a provider that needs no "
            "key, such as a local server, or when the key is already in the "
            "environment under the name this provider looks for."),
        "llm_context_size": (
            "Enter the context window in tokens. Use 0 to let the kernel "
            "compact reactively instead."),
        "llm_service_class": "Choose how Second Brain connects to this model.",
        "llm_capability_image": "Can this model read images natively?",
        "llm_capability_audio": "Can this model read audio natively?",
        "llm_capability_video": "Can this model read video natively?",
    }.get(field, "Enter the new value.")


def _current_value(field, profile, registry=None):
    """A "currently:" preamble, or nothing when there is nothing to say.

    Editing a parameter has shown its present value since the value step was
    built; editing a *setting* showed nothing, so the two halves of one menu
    behaved differently for no reason anybody chose.

    The key is the exception and is never echoed. Whether one is set is the
    useful half and the only half that is safe to print: this renders into a
    prompt, and a prompt is transcript.
    """
    if field == "secret_llm_api_key":
        return ("Currently set.\n\n" if profile.get(field)
                else "Not currently set.\n\n")
    if field in CAPABILITY_FIELDS:
        held = (profile.get("llm_capabilities") or {}).get(
            CAPABILITY_FIELDS[field])
        return "" if held is None else f"Currently `{bool(held)}`.\n\n"
    if field == "llm_context_size":
        held = int(profile.get(field, 0) or 0)
        # No gloss on the zero: the prompt below already explains what it
        # means, and saying it twice reads as two different facts.
        return f"Currently `{held:,}`.\n\n"
    held = profile.get(field)
    if field == "llm_service_class" and held:
        # The display name, matching the picker below it. The stored value is
        # a class name and often a *retired* one, since a migrated backend
        # claims its predecessor's — printing it raw names something that no
        # longer exists, directly above a menu that has renamed it.
        held = _backend_label(registry or {}, held)
    return f"Currently `{held}`.\n\n" if held else ""
