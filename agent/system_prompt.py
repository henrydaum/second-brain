"""Cache-friendly system prompt assembly.

Returns two messages:
- A combined ``system`` message at position 0 — the static prompt followed
  by the semi-stable readouts, one document with no header between them,
  cacheable across turns.
- A ``user`` message tagged ``[SYSTEM CONTEXT UPDATE]`` carrying the
  dynamic runtime context. ``ConversationLoop._messages`` places it as its
  own row immediately before the latest real user turn.

The user-role wrapper exists because some providers (MiniMax) reject
``system`` messages anywhere except position 0. It is the weakest part of
this arrangement and it is a compromise, not a preference: a model reads the
role before it reads the disclaimer, so context the kernel generated arrives
wearing the user's voice. Giving it its own row rather than welding it onto
the person's own words is as far as that can be fixed without knowing what
the backend accepts.

A caution about where the dynamic block actually ends up, because this used
to claim it rides "at the tail of the prompt" and that is only half true. It
sits ahead of whatever ``_split_current_turn`` finds, which is the *last*
row with ``role == "user"``. In a conversation that is near the end, and the
cacheable prefix is everything before it. In an agentic run there is exactly
one user message — the task — followed by dozens of assistant and tool rows,
so "last" is also "first" and the block sits at index 1, ahead of the whole
transcript. Every byte that moves in it therefore re-bills every row behind
it. Treat anything in the ``dynamic`` list as costing the entire
conversation, not a suffix of it, and see ``_current_datetime`` for what that
cost measured.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import monotonic as _monotonic
from typing import Any, Callable

import prompt_cues
from runtime.agent_scope import AgentScope
from runtime.security_modes import security_mode as normalize_security_mode

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_STATIC_PROMPT_PATH = Path(__file__).with_name("system_prompt_static.md")

SYSTEM_CONTEXT_MARKER = "[SYSTEM CONTEXT UPDATE]"

#: How long the turn-stable clock may go without re-rendering, regardless of
#: whether a turn ended. See :func:`_current_datetime` — a turn is normally
#: seconds, but one blocked on a person is unbounded, and telling an agent the
#: wrong time for an unbounded stretch is worse than one cache miss per
#: quarter hour.
_CLOCK_CEILING_SECONDS = 900

#: ``(turn_counter, rendered_at_monotonic, text)`` for the clock below.
_clock_memo: tuple[int, float, str] | None = None
_clock_lock = threading.Lock()

#: ``((mtime_ns, size), text)`` for :func:`_static_prompt`.
_static_memo: tuple[tuple[int, int], str] | None = None

#: Fallback for ``memory_index_cap`` when there is no config to read — the
#: setting is the source of truth (``config/config_data.py``), because anything
#: that *curates* ``MEMORY.md`` has to know the same budget and a plugin cannot
#: import this module to find out.
MEMORY_INDEX_CAP = 4000


@dataclass
class PromptContext:
    """Read-only bag passed to each plugin's ``agent_prompt``.

    Native plugins read whatever they need to build their contribution. For a
    sandboxed plugin, ``session_key`` lets the bridge lend its SDK the same
    session context without passing this kernel object across the boundary.

    It is also what the ``session`` rung of ``prompt_cues`` is keyed on, which
    is why the fields it names (``prompt_cues.SESSION_FACTS``) are exactly the
    ones ``sdk.session.get()`` answers with. A fact a prompt can read and this
    bag cannot see is one nothing would invalidate.
    """
    db: Any = None
    services: dict = field(default_factory=dict)
    orchestrator: Any = None
    config: dict = field(default_factory=dict)
    scope: "AgentScope | None" = None
    profile_name: str = "default"
    frontend_name: str | None = None
    session_key: str | None = None
    security_mode: str = "ask"
    conversation_id: int | None = None
    user_id: int | None = None


def _static_prompt() -> str:
    """The static block, re-read only when the file behind it has moved.

    This is ~7 KB decoded on every LLM call, and the bytes are the same every
    time — it changes no prefix and re-bills nothing, so this is a latency fix
    and not a caching one. It is memoized behind a ``stat`` rather than read
    once at import because that costs a syscall instead of a read-and-decode
    while still picking up an edit: the project's own contract is that source
    changes need ``/restart`` (a process restart, so a plain module-level cache
    would be correct too), but a guard that does not depend on that promise is
    cheaper to be sure about than an argument that it holds.
    """
    global _static_memo
    try:
        info = _STATIC_PROMPT_PATH.stat()
        key = (info.st_mtime_ns, info.st_size)
    except OSError:
        key = None
    memo = _static_memo
    if memo is not None and key is not None and memo[0] == key:
        return memo[1]
    text = _STATIC_PROMPT_PATH.read_text(encoding="utf-8").strip()
    if key is not None:
        _static_memo = (key, text)
    return text


def build_prompt_sections(
    db,
    orchestrator,
    tool_registry,
    services: dict,
    *,
    scope: AgentScope | None = None,
    profile_name: str = "default",
    extra_suffix: str = "",
    commands=None,
    config: dict | None = None,
    conversation_metadata: dict[str, Any] | None = None,
    prompt_extras: dict[str, Any] | None = None,
    notification_suffix: str = "",
    frontend_name: str | None = None,
    frontend=None,
    command_filter: Callable[[str], bool] | None = None,
    active_llm=None,
    session_key: str | None = None,
    security_mode: str = "ask",
    conversation_id: int | None = None,
    user_id: int | None = None,
    context_tokens: int | None = None,
    sections_out: list | None = None,
) -> list[dict[str, str]]:
    """Build ordered system prompt messages.

    ``sections_out``, when given, is filled with ``(block, text)`` for every
    section that made it in, in order. It exists for ``dev/dump_agent_text.py``
    and costs every other caller nothing. The alternative was for the dump to
    recover the boundaries by parsing the rendered blocks, which cannot be done
    correctly: sections are joined with a blank line, so an *unheaded* one
    following a headed one — the clock, the model line — is indistinguishable
    from a continuation of the section above it, and gets silently attributed
    to whatever came before. A map that quietly credits the wrong author is
    worse than no map, since the whole reason to read one is to find out whose
    line it is.

    Optional per-plugin guidance is collected from whatever tools/services/tasks/
    commands/frontend are currently in scope (each plugin's ``agent_prompt``),
    so installed packages bring their own guidance and uninstalling removes it —
    the kernel no longer hardcodes prompt text for plugins it may not ship.

    ``context_tokens`` is the previous model call's billed input, frozen at the
    start of the turn by ``RuntimeSession.turn_prompt_tokens``. It arrives as an
    argument rather than being read here because the freeze is per *session* and
    this module is shared: a memo of the kind ``_current_datetime`` keeps would
    thrash between concurrent sessions and report one session's size in
    another's prompt.
    """
    r = tool_registry
    pctx = PromptContext(
        db=db, services=services or {}, orchestrator=orchestrator,
        config=config or {}, scope=scope, profile_name=profile_name,
        frontend_name=frontend_name,
        session_key=session_key,
        security_mode=normalize_security_mode(security_mode),
        conversation_id=conversation_id,
        user_id=user_id,
    )
    populations = _in_scope(r, services, orchestrator, commands,
                            command_filter, frontend)
    # Four catalogs used to ride here and in the dynamic block — the tool
    # roster, the slash-command list, the service line and the pipeline graph.
    # All four are answerable on demand through the store's ``info`` tool, and
    # a catalog nobody consulted this turn is the clearest case of text every
    # call pays for and almost no call reads. The tool roster was the starkest:
    # it listed names the injected schemas already carry in full.
    #
    # What survives is what a lookup cannot replace — the environment's paths,
    # and the guidance plugins contribute about themselves.
    # The two blocks are split by *what moves*, and the two folder sections
    # were on the wrong side of that line. Both read nothing but config —
    # ``fs_writable_dirs`` and ``sync_directories`` — which is the ``config``
    # rung, at or below ``prompt_cues.STABLE_THROUGH``, so by the ladder's own
    # rule their text cannot move within a conversation and belongs in the
    # cacheable position-0 message. Paying for them at the volatile end was
    # the same mistake the cue ladder exists to stop a plugin making.
    #
    # It also puts the whole answer to "where may I write, and where does what
    # I write get indexed" beside the paths it is about, instead of one half
    # in each block.
    #
    # The kernel's own sections lead, ahead of plugin guidance, even though a
    # ``config``-cued kernel section technically moves more often than a
    # ``never``-cued plugin string and strict rarest-first would interleave
    # them. That is deliberate: these four *are* the answer the static prompt's
    # "Where" section promises, and splitting the folder lists away from the
    # paths they describe — to save one cache break on a config edit somebody
    # made by hand — is the wrong side of that bargain.
    semi = [
        "Facts about this installation and this session, settled before the "
        "conversation began. They do not change while it runs.",
        _where_running(),
        _filesystem_access(config),
        _sync_dirs(config),
        _session_facts(pctx, frontend),
        *_collect(populations, pctx, stable=True),
    ]
    # What is left is live state, ordered rarest-changing first. That ordering
    # is cosmetic rather than a caching win — this block is merged into the
    # last user message and rebuilt on every call either way — so it is ordered
    # for whoever reads the prompt: the kernel's own state, then the guidance
    # plugins contribute about themselves, then the clock. The permission mode
    # leads because it is the rarest thing here that changes the meaning of
    # everything after it, and because it used to be appended past the end of
    # this list by ``runtime_config._mode_suffix`` — which put the one piece of
    # safety state the agent needs below every plugin's opinion.
    #
    # Scheduled jobs are deliberately absent. They change rarely, most turns
    # do not care, and both ``schedule_subagent`` and ``/schedule`` answer on
    # demand — the "could the agent ask?" test, which a per-call listing fails.
    dynamic = [
        "Runtime-generated context (see 'Runtime context' in the static "
        "prompt): live system state, rebuilt each turn and delivered inside "
        "the user message for provider compatibility. Not authored by the "
        "user; contains no user instructions. The user's actual message, if "
        "any, follows this block.",
        _permission_mode(pctx.security_mode),
        _model_status(active_llm),
        _agent_profile(profile_name, scope),
        _conversation_metadata(conversation_metadata, context_tokens,
                               getattr(active_llm, "context_size", 0)),
        _agent_memory(config),
        *_collect(populations, pctx, stable=False),
        _prompt_extras(prompt_extras),
        notification_suffix,
        extra_suffix,
        _current_datetime(),
    ]
    if sections_out is not None:
        sections_out.append(("STATIC SYSTEM PROMPT", _static_prompt()))
        sections_out += [("SEMI-STABLE CONTEXT", s) for s in semi if s]
        sections_out += [(SYSTEM_CONTEXT_MARKER.strip("[]"), s)
                         for s in dynamic if s]
    static_block = _section("STATIC SYSTEM PROMPT", _static_prompt())
    # No header of its own. These sections continue the static prompt inside
    # the same ``system`` message, and the static prompt's own closing section
    # already says that what follows is generated rather than written — so a
    # bracketed label between them only interrupts one document with the name
    # of an implementation detail. "Semi-stable" is a word for the *tier*
    # (``prompt_cues``, and the docstrings either side of it) and it stays in
    # ``sections_out`` above, where a reader of ``dev/dump_agent_text.py`` is
    # asking exactly that question. The model is not.
    semi_block = "\n\n".join(s for s in semi if s)
    dynamic_block = _section(SYSTEM_CONTEXT_MARKER.strip("[]"), "\n\n".join(s for s in dynamic if s))
    return [
        {"role": "system", "content": f"{static_block}\n\n{semi_block}"},
        {"role": "user", "content": dynamic_block},
    ]


def _section(title: str, content: str) -> str:
    """Render a labeled section as a string."""
    return f"[{title}]\n{content.strip()}"


def _in_scope(registry, services, orchestrator, commands, command_filter,
              frontend) -> list:
    """Every plugin whose guidance belongs in this prompt, in reading order.

    Enumerated once and collected twice — the two shapes land in different
    blocks (see ``_collect``), and the populations are the same either way.
    """
    return [
        *_visible_tools_for_prompt(registry),
        *_loaded_services_for_prompt(services),
        *_tasks_for_prompt(orchestrator),
        *_visible_commands_for_prompt(commands, command_filter),
        *([frontend] if frontend is not None else []),
    ]


def _collect(plugins, ctx: PromptContext, *, stable: bool) -> list[str]:
    """Join the ``agent_prompt`` contributions belonging in one block.

    One name, two shapes. A plugin with nothing conditional to say declares a
    plain string (``agent_prompt = "..."``), which the loader reads by AST and
    copies onto the adapter for free; one whose text depends on live state
    declares a method taking the context. Accepting both under one name is
    load-bearing rather than merely tidy, because a string shadowing a method
    would otherwise raise ``TypeError`` into the ``except`` below and the
    guidance would vanish with no symptom at all.

    The shape decides *how to ask*, and it is still ``callable`` that answers —
    deliberately its own question, since the tolerance above is what depends on
    it. What decides *where the text lands* is the plugin's declared cue:
    ``prompt_cues.stable`` is true for the rungs that cannot move within a
    conversation, and their text rides in the semi-stable block of the
    cacheable position-0 prefix. Everything finer goes to the dynamic block
    with the kernel's own live state, which is the argument ``_mode_suffix``
    makes for itself in ``runtime/runtime_config.py``.

    The two used to be one question — a string in the prefix, a method in the
    dynamic block, and no declaration needed to tell them apart. That was right
    about the string and too coarse about the method: a prompt that only
    follows the permission mode does not move within a conversation any more
    than a fixed one does, and paying for it at the volatile end of the prompt
    on every call was the only option it had.

    Within a block, rarest first. That ordering pays in the prefix, where a
    provider re-reads from the first byte that changed; in the dynamic block it
    is cosmetic, since the whole message is rebuilt per call either way. The
    sort is **stable**, so contributions on one rung keep ``_in_scope``'s
    reading order — tools, then services, tasks, commands and the frontend.

    Answers with the contributions as a *list*, one per plugin, which the
    caller splats into its block list and joins there. Joining here instead
    produced identical bytes — ``"\n\n".join`` is associative over the same
    separator — but it meant one opaque blob where the prompt actually has
    several sections, so nothing downstream could say which plugin wrote
    which. ``dev/dump_agent_text.py`` needs exactly that, and inferring it
    from the rendered text cannot be done correctly (see ``sections_out``).
    """
    entries = []
    for plugin in plugins:
        try:
            raw = getattr(plugin, "agent_prompt", "")
            cue = prompt_cues.of(plugin)
            if prompt_cues.stable(cue) != stable:
                continue
            text = ((raw(ctx) if callable(raw) else raw) or "").strip()
        except Exception:
            text = ""
        if text:
            entries.append((prompt_cues.rank(cue), _sourced(plugin, text)))
    entries.sort(key=lambda entry: entry[0])
    return [text for _, text in entries]


def _sourced(plugin, text: str) -> str:
    """Stamp a contribution with the file it came from."""
    path = str(getattr(plugin, "_source_path", "") or "")
    stem = Path(path).stem if path else (getattr(plugin, "name", "") or "?")
    return f"{text}\n(source: {stem})"


def _visible_tools_for_prompt(registry):
    """Tools the agent can currently see (profile-scoped), sorted by name."""
    if not registry or not hasattr(registry, "_visible_tools"):
        return []
    return sorted(registry._visible_tools(), key=lambda t: getattr(t, "name", ""))


def _loaded_services_for_prompt(services: dict):
    """Loaded service instances, sorted by registry name."""
    return [svc for _, svc in sorted((services or {}).items()) if getattr(svc, "loaded", False)]


def _tasks_for_prompt(orchestrator):
    """Registered task instances, sorted by name."""
    tasks = getattr(orchestrator, "tasks", {}) or {}
    return [tasks[name] for name in sorted(tasks)]


def _visible_commands_for_prompt(commands, command_filter):
    """Commands visible under the current frontend's policy, sorted by name."""
    if not commands or not hasattr(commands, "visible_commands"):
        return []
    try:
        return commands.visible_commands(command_filter)
    except Exception:
        return []


def _where_running() -> str:
    """The three paths the static prompt's "Where" section talks about.

    Named for the question rather than for a category. The static prompt tells
    the agent to work out where it is in relation to the kernel, the DATA_DIR
    and the user's folders; this is the first part of that answer, and the two
    sections below it finish it.
    """
    import platform

    from paths import DATA_DIR

    return "\n".join([
        "## Where you are running",
        f"- Host operating system: {platform.system()} {platform.release()}",
        f"- Kernel source, the protected half (ROOT_DIR): {_PROJECT_ROOT}",
        "- Mutable data — database, config, installed plugins, workspace "
        f"(DATA_DIR): {DATA_DIR}",
    ])


def _render_datetime() -> str:
    now = datetime.now().astimezone()
    return (
        "## Right now\n"
        f"Current date and time: {now.strftime('%A, %B %d, %Y %I:%M %p')} "
        f"(local time, UTC{now.strftime('%z')[:3]}:{now.strftime('%z')[3:]})"
    )


def _current_datetime() -> str:
    """The clock, rendered once per turn rather than once per call.

    This line is the most expensive string in the prompt, for a reason that has
    nothing to do with its length. The dynamic block is merged into the latest
    user-led turn, and an agentic run has exactly one user message — so the
    block sits ahead of the entire tool-call transcript, and any byte that
    moves in it invalidates the cached prefix for every row behind it.
    ``%I:%M %p`` is *fixed width*, so a minute rollover changes the prefix and
    changes nothing about the length: measured across 636 benchmark trials,
    calls built inside one minute collapsed the prefix 1.4% of the time and
    calls that crossed a minute boundary collapsed it 33.3% of the time,
    together carrying 27–45% of all billed-uncached input.

    Rendering once per turn costs the agent nothing it was promised. The block
    announces itself as state "refreshed each turn", and
    ``system_prompt_static.md`` tells the reader to expect it to change
    *between* turns. Minute precision is kept, because relative-date reasoning
    is what the line is for.

    Keyed on the raw ``prompt_cues`` turn counter and **not** on
    ``prompt_cues.stamp``: a stamp at rank ``session`` or finer folds the
    session's own facts into the key, so one global slot keyed that way would
    thrash between concurrent sessions — session A caches, session B evicts, A
    re-renders and picks up the rollover. The wall clock is not a per-session
    fact and must not be keyed as one. Sharing one string across sessions is
    correct; the worst a race can do here is render twice.

    The ceiling exists because a turn is not time-bounded. ``turn`` fires once
    per driven turn, but a tool that asks the user a question blocks on a
    person, and approval dialogs hold for minutes at a time. Without it, a turn
    parked on a dialog would report the wrong time for as long as it waited.
    """
    global _clock_memo
    turn = prompt_cues.value(prompt_cues.TURN)
    now = _monotonic()
    with _clock_lock:
        memo = _clock_memo
        if (memo is not None and memo[0] == turn
                and now - memo[1] < _CLOCK_CEILING_SECONDS):
            return memo[2]
        text = _render_datetime()
        _clock_memo = (turn, now, text)
        return text


def _reset_turn_clock() -> None:
    """Drop the memoized clock. For tests; nothing in the kernel calls it."""
    global _clock_memo
    with _clock_lock:
        _clock_memo = None


def _model_status(llm=None) -> str:
    """Describe the session's profile-resolved brain to the model itself.

    ``native_modalities`` is the *backend's* half and ``capabilities`` the
    *model's*; a modality counts only when both agree, which is the same pair
    ``ConversationLoop._route_attachments`` splits an attachment bundle on. It
    is worth reading them from the same names routing does — this asked for
    ``native_attachment_modalities``, which no ``Brain`` has ever had, so the
    ``getattr`` default quietly made every model blind in its own prompt while
    routing worked perfectly.
    """
    if not llm:
        return "## Model\nCurrent model: unavailable."
    model = getattr(llm, "model_name", None)
    caps = getattr(llm, "capabilities", {}) or {}
    native = set(getattr(llm, "native_modalities", set()) or set())
    parts = []
    for modality, label in (("image", "images"), ("audio", "audio"), ("video", "video")):
        parts.append(f"{label}: {'yes' if caps.get(modality) and modality in native else 'no'}")
    return (
        "## Model\n"
        f"Current model: {model or 'unknown'}.\n"
        f"Native attachment processing: {'; '.join(parts)}. "
        "For unsupported modalities, rely only on parsed text or file pointers."
    )


def _agent_profile(profile_name: str, scope: AgentScope | None) -> str:
    """The profile, its tool limits and its own instructions, under one heading.

    These were three items scattered across the block: the name near the top,
    the limits note near the bottom, and the profile's ``prompt_suffix`` last
    of all as a bare paragraph with nothing above it. The suffix is the part
    that matters — it is what the static prompt calls the agent's "specific
    instructions" — and a model reading it unattributed cannot tell whose
    instructions it has been handed, which is exactly the kind of line nobody
    can explain the presence of.
    """
    lines = ["## Agent profile", f"Active profile: {profile_name or 'default'}."]
    if scope and scope.has_tool_filter:
        lines.append(
            "Tool access is limited to the tools exposed in this prompt. If a "
            "task needs a tool outside this profile, say so and name the tool "
            "rather than improvising around the restriction.")
    suffix = ((getattr(scope, "prompt_suffix", "") or "").strip()
              if scope else "")
    if suffix:
        lines.append(f"Specific instructions from this profile:\n{suffix}")
    return "\n".join(lines)


def _permission_mode(mode: str) -> str:
    """Name the standing answer this conversation gives to approval dialogs.

    The static prompt names all three modes; a reader told about the mechanism
    and never told its current state has been given half of something.
    ``security_modes.prompt_note`` keeps its own contract — it carries only the
    guidance a *non-default* mode needs — and this names the mode in every
    case, including ``ask``.

    ``MODE_BLURBS`` is deliberately not reused even though it is the single
    source for the ``/mode`` table and the ``session.set_mode`` dialog. Those
    lines are addressed to the **user** ("Ask you about anything that needs
    your approval"), and in this prompt "you" is the agent, so borrowing them
    would invert every one of the three.
    """
    from runtime.security_modes import ASK, prompt_note
    from runtime.security_modes import security_mode as _normalize

    mode = _normalize(mode)
    lines = ["## Permission mode", f"Mode: `{mode}`."]
    if mode == ASK:
        lines.append(
            "The default. Anything the policy classes as consequential is put "
            "to the user before it runs, so you are interrupted rather than "
            "refused.")
    note = prompt_note(mode)
    if note:
        lines.append(note)
    return "\n".join(lines)


def _session_facts(ctx: PromptContext, frontend=None) -> str:
    """Which surface the user is on, what it can show, and who they are.

    The static prompt tells the agent that "one of your goals should be to
    find out where you are in relation to these" — this is that answer for the
    frontend half of it. The capabilities named are the ones with an agent-side
    decision behind them: whether a file can be sent in, whether a file it
    produces can be displayed at all, and whether markdown or buttons render.
    Streaming and notification support are transport details the agent cannot
    act on, so they are left out rather than listed for completeness.
    """
    lines = ["## This session"]
    name = (ctx.frontend_name or "").strip()
    if name:
        lines.append(f"- Frontend: {name} — the surface the user is talking "
                     "to you through.")
    else:
        lines.append("- Frontend: none — this drive has no surface attached "
                     "(a background or subagent conversation).")
    caps = getattr(frontend, "capabilities", None)
    if caps is not None:
        def yes_no(*attrs) -> str:
            return "yes" if any(getattr(caps, a, False) for a in attrs) else "no"
        lines.append(
            f"- Can send you files: {yes_no('supports_attachments_in')}. "
            f"Can display files you produce: {yes_no('supports_attachments_out')}. "
            f"Renders markdown: {yes_no('supports_rich_text')}. "
            "Buttons and inline forms: "
            f"{yes_no('supports_buttons', 'supports_inline_forms')}.")
    binding = getattr(frontend, "user_binding", "")
    if binding == "per_user":
        lines.append("- This frontend gives each identity its own account.")
    elif binding:
        lines.append("- This frontend maps every session to one account.")
    account = _account_name(ctx.db, ctx.user_id)
    if account:
        lines.append(f'- You are assisting the user "{account}".')
    met = _first_met(ctx.db, ctx.user_id)
    if met:
        lines.append(f"- First met this user: {met}.")
    return "\n".join(lines)


def _account_name(db, user_id) -> str:
    """The account's login name, when the session belongs to a real account.

    Anonymous, base and guest sessions carry no username and the line would add
    nothing, so it never becomes noise on a single-operator frontend. Absorbed
    from ``runtime_config._account_suffix``, which appended the same fact to
    the tail of the dynamic block: it belongs beside the frontend that
    established the identity, and it is stable for the life of a session.
    """
    if db is None or not user_id:
        return ""
    try:
        return str((db.get_user(user_id) or {}).get("username") or "")
    except Exception:
        return ""


#: ``user_id -> rendered date``, and only ever a *non-empty* one. See
#: :func:`_first_met`.
_first_met_memo: dict = {}


def _first_met(db, user_id) -> str:
    """When this user's oldest surviving conversation was created.

    Scoped to the user rather than to the installation, because what is worth
    saying is how long the agent and this person have been talking. A figure
    read off the disk would be a different fact wearing the same words, and on
    a ``per_user`` frontend it would frequently be somebody else's.

    Answers "" for a user with no conversations and for a build with no session
    at all — the bootstrap prompt and ``dev/dump_agent_text.py`` both pass no
    ``user_id`` — rather than falling back to a global minimum. One meaning per
    line: a sentence that silently changes what it is measuring is worse than a
    missing one.

    Only a real answer is memoized. The value moves only when that user's
    oldest conversation is pruned by ``data_retention_days``, so caching it for
    the process is free; caching the *empty* answer is not, because a brand new
    user would then be told nothing for the life of the process even after the
    conversation they are in is written.
    """
    if db is None or not user_id:
        return ""
    cached = _first_met_memo.get(user_id)
    if cached:
        return cached
    from datetime import datetime as _datetime
    try:
        rows = db.query_rows(
            "SELECT MIN(created_at) AS first FROM conversations "
            "WHERE user_id = ?", (user_id,), max_rows=1)
        stamp = rows[0]["first"] if rows else None
        text = (_datetime.fromtimestamp(float(stamp)).strftime("%B %d, %Y")
                if stamp else "")
    except Exception:
        return ""
    if text:
        _first_met_memo[user_id] = text
    return text







def _sync_dirs(config: dict | None) -> str:
    dirs = (config or {}).get("sync_directories") or []
    return "## Sync directories\n" + ("\n".join(f"- {d}" for d in dirs) if dirs else "None configured.")


def _filesystem_access(config: dict | None) -> str:
    """Tell the agent which write locations are its own and which are the user's.

    The distinction is security-relevant and cannot live only in static prose:
    ``fs_writable_dirs`` is configuration, so the model needs its current value
    in the same prompt that tells it what the grant means.

    The attachment cache is named as its own line even though it is inside the
    workspace path directly above it. That it is *inside* is exactly the fact
    worth stating: an upload used to land in a folder of its own beside the
    tree, and a model that has to infer "so I may edit this one too" from a
    path prefix will ask instead.
    """
    import trees
    from paths import DATA_DIR

    raw = (config or {}).get("fs_writable_dirs") or []
    if isinstance(raw, str):
        raw = [part.strip() for part in raw.split(",")]
    writable = [str(item).strip() for item in raw if str(item).strip()]

    lines = [
        "## Filesystem access",
        f"Agent-owned workspace (free write): {DATA_DIR / 'workspace'}",
        f"Incoming attachments land in {trees.attachment_cache()} — inside "
        f"the workspace, so they carry the same free-write grant.",
        "User-owned writable folders (free write only when the user's task "
        "calls for it):",
    ]
    lines.extend(f"- {path}" for path in writable)
    if not writable:
        lines.append("- None configured.")
    # Name the complement, or the grant reads as a description of the disk. A
    # model told only where it *may* write has been told nothing about what
    # happens elsewhere, and the honest answer — it still works, it just costs
    # the user a dialog — is the difference between an agent that asks and one
    # that reads the omission as a wall and improvises around it.
    lines.append(
        "Folders outside these lists remain protected: work there is "
        "classified one action at a time, and anything consequential raises "
        "an approval dialog rather than failing.")
    return "\n".join(lines)


def _memory_index_cap(config: dict | None) -> int:
    """The budget, from config, falling back to the constant above."""
    try:
        return max(1, int((config or {}).get("memory_index_cap")
                          or MEMORY_INDEX_CAP))
    except (TypeError, ValueError):
        return MEMORY_INDEX_CAP


def _agent_memory(config: dict | None = None) -> str:
    """Inline the one memory artifact the kernel owns: ``MEMORY.md``.

    Topic layout, validation and file operations belong to the installed
    memory tool. The kernel knows only the fixed workspace index path needed
    to assemble the prompt.
    """
    from paths import DATA_DIR

    index_path = DATA_DIR / "workspace" / "memory" / "MEMORY.md"
    try:
        index = index_path.read_text(encoding="utf-8").strip() if index_path.exists() else ""
    except OSError:
        index = ""
    # What the index *says* is the agent's business — it writes this file and
    # nothing here edits it. What it may *cost* is the kernel's: this text is
    # inlined on every call, so an index nobody pruned would grow the prompt
    # without bound. Truncation is visible on purpose, so the agent can see
    # that its own index has outgrown the window and prune it.
    cap = _memory_index_cap(config)
    if len(index) > cap:
        index = (index[:cap].rstrip()
                 + f"\n... (index truncated at {cap} characters "
                   "— prune MEMORY.md)")
    lines = [
        "## Memory",
        f"Index path: {index_path}",
        "Durable notes that persist across sessions.",
        "",
        "Index (MEMORY.md):",
        index or "(empty)",
    ]
    return "\n".join(lines)


def _conversation_metadata(meta: dict[str, Any] | None, tokens=None,
                           context_size=0) -> str:
    if not meta:
        return ""
    lines = ["## Current conversation",
             f"Number: {meta.get('id')}",
             f"Category: {(meta.get('category') or '').strip() or 'Main'}",
             f"Title: {(meta.get('title') or '').strip() or 'New Conversation'}"]
    # Directly above the compaction sentence, which is what makes the number
    # actionable: it is the one fact that says how close that sentence is to
    # describing what is about to happen.
    usage = _context_usage(tokens, context_size)
    if usage:
        lines.append(usage)
    lines.append(
        "When a conversation gets too long, it will be compacted to save "
        "space. History prior to the compaction will still be available in "
        "the database, but won't be visible in the conversation context for "
        "new messages.")
    return "\n".join(lines)


def _context_usage(tokens, context_size) -> str:
    """How full the context window was at the start of this turn.

    Both halves can be unknown and neither may be guessed. ``tokens`` is the
    provider's own count for the previous model call, and ``None`` there means
    *the provider did not say*, which is not zero (see
    ``ConversationLoop._emit_llm_finished``); ``context_size`` is 0 whenever
    the profile has not declared ``llm_context_size``. Either one missing and
    the line is omitted rather than rendered around a number nobody measured.

    The figure is frozen for the turn by ``RuntimeSession.turn_prompt_tokens``,
    for the reason ``_current_datetime`` sets out at length: this block sits
    ahead of the whole transcript in an agentic run, so a count that climbed
    with every tool call would re-bill every row behind it on every call. It
    says "at the start of this turn" because that is exactly what it is — and
    an agent that reads a stale-looking number without being told its vintage
    will distrust the whole block.
    """
    try:
        used, window = int(tokens or 0), int(context_size or 0)
    except (TypeError, ValueError):
        return ""
    if used <= 0 or window <= 0:
        return ""
    return (f"Context: about {used:,} of {window:,} tokens "
            f"({round(100 * used / window)}%) at the start of this turn.")


def _prompt_extras(extras: dict[str, Any] | None) -> str:
    values = [v for v in (extras or {}).values() if isinstance(v, str) and v]
    return "\n\n".join(values)
