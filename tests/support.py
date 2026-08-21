"""Shared test doubles.

These existed five times over, copied between files as each new test needed a
fake model. The copies drifted — some recorded calls as bare message lists,
some as dicts with kwargs; some raised when the queued responses ran out and
some kept answering — so a change to ``ConversationRuntime``'s signature was a
fifteen-file edit and a change to the loop's call shape broke whichever copies
happened to care.

Anything here is deliberately a *superset* of what the copies did: declaring an
attribute no test reads costs nothing, while a fake missing one produces a
failure that looks like a kernel bug. Fixtures wrapping these live in
``conftest.py``; import the classes directly when a test needs one at module
scope.
"""

import subprocess
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]


def response(content="", tool_calls=None, **extra):
    """One model response, shaped like what a backend returns.

    ``extra`` covers the occasional test that needs ``is_error`` or a token
    count, without every caller restating the full shape.
    """
    fields = dict(
        content=content,
        tool_calls=tool_calls or [],
        has_tool_calls=bool(tool_calls),
        is_error=False,
        prompt_tokens=0,
    )
    fields.update(extra)
    return SimpleNamespace(**fields)


class FakeLLM:
    """Answers from a queued list, and records what it was asked.

    Brain-shaped: one ``chat(request, on_delta=None)`` taking an
    :class:`LLMRequest` and returning an :class:`LLMResponse`, exactly like the
    real thing. It used to expose ``chat_with_tools`` and be adapted by
    ``llm.registry.as_brain`` on the way into the loop; that adapter existed
    for unmigrated in-process backends and went with them, so a double now
    speaks the only language there is.

    Runs out gracefully: once the queue empties it keeps answering ``"done."``.
    A turn that loops one more time than the test expected should fail on the
    transcript it produced, not on an ``IndexError`` from the fake — the second
    tells you nothing about what went wrong.
    """

    # 0 disables proactive compaction, which is what nearly every test wants;
    # the compaction tests set it explicitly.
    context_size = 0
    model_name = "fake"
    name = "fake"
    loaded = True
    supports_streaming = False
    supports_tool_choice = False
    # Attachment routing is the kernel's job and it asks the model what it can
    # read. A fake declaring nothing gets the text fallback for everything —
    # correct, but it would make a vision test assert nothing.
    capabilities = {"image": True, "audio": True, "video": True}
    native_modalities = {"image", "audio", "video"}

    def __init__(self, responses=None):
        """Queue zero or more responses."""
        self._responses = list(responses or [])
        # Bare message lists — the shape most assertions want.
        self.calls = []
        # The same calls with everything else that came along, for the tests
        # that assert on tools or provider kwargs.
        self.records = []
        # One entry per call: the routed ``{path, modality, file_name}`` dicts
        # the kernel decided this model should send natively. These used to be
        # rebuilt ``Attachment`` objects, because the adapter handed the old
        # contract a live ``AttachmentBundle``; nothing rebuilds them now.
        self.attachments = []

    def load(self):
        """Already loaded."""
        return True

    def chat(self, request, on_delta=None, on_call=None):
        """Record the call and answer with the next queued response.

        ``on_call`` is accepted and ignored: a real ``Brain`` uses it to hand
        back a stopper for the box serving this call, and there is no box
        here. A double that *wants* to be interruptible arms it itself — see
        ``tests/test_cancel_immediacy.py``.
        """
        self.calls.append(list(request.messages))
        self.records.append({"messages": list(request.messages),
                             "tools": request.tools,
                             "attachments": request.attachments,
                             "kwargs": dict(request.params or {})})
        self.attachments.append(request.attachments)
        if self._responses:
            return self._responses.pop(0)
        return response(content="done.")


class ToolChoiceLLM(FakeLLM):
    """A model that admits to supporting ``tool_choice``."""

    supports_tool_choice = True


class FakeRegistry:
    """A tool registry that only knows how to hand out schemas."""

    tools = {}  # empty -> no per-tool budget enforcement

    def __init__(self, schemas=None, max_tool_calls=5):
        """Hold the schemas the agent should see."""
        self._schemas = list(schemas or [])
        self.max_tool_calls = max_tool_calls

    def get_all_schemas(self):
        """Every schema the agent may call."""
        return self._schemas


def agent_state(tools=None, cache=None):
    """A ConversationState with turn priority already on the agent."""
    from state_machine.conversation import ConversationState, Participant
    from state_machine.conversation_phases import BASE_PHASE

    base = {"session_key": "chat",
            "agent_scoped_tool_names": list((tools or {}).keys())}
    base.update(cache or {})
    return ConversationState(
        [Participant("user", "user"),
         Participant("agent", "agent", tools=tools or {})],
        "agent", BASE_PHASE, base)


def plain_runtime(db, **kwargs):
    """A ConversationRuntime on an existing database, with no model.

    The single most repeated line in the suite — twenty-two copies of
    ``ConversationRuntime(db=db, services={}, config={})`` across seven files,
    which is why adding one required argument used to be a seven-file edit.
    Tests that need a model want :func:`make_runtime` instead.
    """
    from runtime.conversation_runtime import ConversationRuntime

    kwargs.setdefault("services", {})
    kwargs.setdefault("config", {})
    return ConversationRuntime(db=db, **kwargs)


def make_runtime(tmp_path, responses=None, *, name="test.db", services=None,
                 config=None, session_key="s", title="x", **kwargs):
    """A ConversationRuntime on a fresh database with one open conversation.

    Returns ``(runtime, session, llm)``. The ``llm`` is the :class:`FakeLLM`
    that was installed unless ``services`` overrode it, so a test can queue
    responses and then read back what the loop actually sent.
    """
    from pipeline.database import Database
    from runtime.conversation_runtime import ConversationRuntime

    db = Database(str(tmp_path / name))
    cid = db.create_conversation(title)
    llm = FakeLLM(responses)
    if services is None:
        services = {"llm": llm}
    runtime = ConversationRuntime(db=db, services=services,
                                  config=config or {}, **kwargs)
    session = runtime.load_conversation(session_key, cid)
    return runtime, session, llm


# ── Reaching the store branch ─────────────────────────────────────────
#
# Some kernel invariants are about *store* files: what the validator says
# about them, and what declarations the bridge reads off them. Those files are
# not in this tree, so they are materialized from the store branch.
#
# A worktree is preferred over the committed ref when the clone has one.
# Mid-migration the interesting version of a file is the one being edited, and
# a test that silently checks the last commit instead is a test that passes
# while the work is broken.

def store_worktree():
    """A checkout of the store tree, including a feature branch on it."""
    proc = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "worktree", "list", "--porcelain"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        encoding="utf-8", check=False)
    root = None
    conventional = None
    for line in proc.stdout.splitlines():
        if line.startswith("worktree "):
            root = Path(line.split(" ", 1)[1])
            if root.name.casefold() == "secondbrain-store":
                conventional = root
        elif line.strip() in {"branch refs/heads/store", "branch store"}:
            return root
    return conventional


def store_source(relative: str):
    """The store's copy of one file, from a worktree or a ref, or None."""
    worktree = store_worktree()
    if worktree is not None and (worktree / relative).is_file():
        return (worktree / relative).read_text(encoding="utf-8")
    for ref in ("store", "origin/store"):
        proc = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "show", f"{ref}:{relative}"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            encoding="utf-8", check=False)
        if proc.returncode == 0:
            return proc.stdout
    return None


# ── Pointing the layout somewhere disposable ──────────────────────────

def retarget_trees(monkeypatch, tmp_path, **overrides):
    """Point the tree layout at ``tmp_path`` for the duration of one test.

    Replaces ``trees.TREES`` wholesale rather than patching a path constant,
    because the tuple is what every lookup walks — ``locate``, ``dirs_for``,
    ``tree`` and therefore ``isolation``, ``policy`` and discovery all read it
    at call time, so one swap moves the whole layout consistently.

    Returns ``{name: path}`` for the trees it built. Pass an override to place
    one somewhere specific::

        roots = retarget_trees(monkeypatch, tmp_path, bundled=repo / "bundled")
    """
    import trees

    built = {}
    replacements = []
    for original in trees.TREES:
        path = Path(overrides.get(original.name, tmp_path / original.name))
        built[original.name] = path
        replacements.append(
            trees.Tree(original.name, path, original.module,
                       builtin=original.builtin))
    monkeypatch.setattr(trees, "TREES", tuple(replacements))
    return built


# ── Standing at doorways, and writing down what happened ──────────────
#
# A hook test asserts about *visits*: which doorway, in what order, shown
# what. Recording that is the whole rig, and it has to record identically on
# both sides of the sandbox boundary or the two can never be compared.
#
# A native probe appends to a list. A sandboxed one cannot — it shares no
# memory with the test, and journalling to a file would mean ``sdk.fs.write``,
# which is a policy event in the middle of the thing being measured. So a
# boxed probe accumulates in ``self._seen`` and the test reads it back through
# an export, which is what ``tests/test_sandbox_hooks.py`` already does.
#
# Both sides therefore produce *the same list of dicts*, and that is the
# point: ``assert boxed == native`` is one line that checks the whole
# projection layer in ``sandbox/hooks.py`` is faithful. Dicts rather than
# tuples because a tuple crosses the wire as a list.

#: Each moment's payload, narrowed to the fields worth comparing. Kept small
#: on purpose — a record holding ``messages`` would differ between the two
#: sides for reasons that are not about hooks.
def visit(moment, session_key="", user_id=0, conversation_id=0, attended=True,
          **payload):
    """One doorway visit, in the shape both sides agree to write."""
    return {"moment": moment, "session_key": str(session_key or ""),
            "user_id": int(user_id or 0),
            "conversation_id": int(conversation_id or 0),
            "attended": bool(attended), **payload}


def native_probe(journal, moments, answers=None):
    """Native hook callables that write to ``journal``, one per moment.

    ``answers`` maps a moment to what its hook returns — a value, or a
    callable taking the payload. Anything unnamed abstains.

    Returns ``{moment: fn}``; register with ``hooks.add(moment, fn)``. The
    records are byte-identical to what :func:`probe_source` produces in a box,
    so a test can assert the two journals are equal.
    """
    answers = dict(answers or {})

    def _answer(moment, payload):
        reply = answers.get(moment)
        return reply(payload) if callable(reply) else reply

    def _ident(ctx):
        session = getattr(ctx, "session", None)
        runtime = getattr(ctx, "runtime", None)
        key = str(getattr(session, "key", "") or "")
        attended = True
        reader = getattr(runtime, "is_attended", None)
        if callable(reader) and key:
            try:
                attended = bool(reader(key))
            except Exception:
                attended = True
        return {"session_key": key,
                "user_id": getattr(session, "user_id", 0),
                "conversation_id": getattr(session, "conversation_id", 0),
                "attended": attended}

    def turn_start(ctx, payload):
        journal.append(visit("turn_start", **_ident(ctx)))
        return _answer("turn_start", payload)

    def shape_scope(ctx, registry):
        # Native sees a live registry; the guest sees names. Record names, so
        # the two agree.
        names = sorted(getattr(registry, "tools", None) or {})
        journal.append(visit("shape_scope", tools=names, **_ident(ctx)))
        return _answer("shape_scope", names)

    def vet_permission(ctx, query):
        journal.append(visit(
            "vet_permission", **_ident(ctx),
            tool_name=str(getattr(query, "tool_name", "") or ""),
            command=str(getattr(query, "command", "") or ""),
            stage=str(getattr(query, "stage", "") or ""),
            origin=str(getattr(query, "origin", "") or "")))
        return _answer("vet_permission", query)

    def llm_call(ctx, request, proceed):
        journal.append(visit(
            "llm_call", **_ident(ctx),
            llm=str(getattr(request, "llm", "") or ""),
            messages=len(getattr(request, "messages", None) or []),
            tools=len(getattr(request, "tools", None) or [])))
        reply = answers.get("llm_call")
        return reply(request, proceed) if callable(reply) else proceed(request)

    def end_turn(ctx, ending):
        journal.append(visit(
            "end_turn", **_ident(ctx),
            final_text=str(getattr(ending, "final_text", "") or ""),
            reason=str(getattr(ending, "reason", "") or ""),
            doorman_fires=int(getattr(ending, "doorman_fires", 0) or 0)))
        return _answer("end_turn", ending)

    def turn_finish(ctx, outcome):
        journal.append(visit(
            "turn_finish", **_ident(ctx),
            ok=bool(getattr(outcome, "ok", True)),
            cancelled=bool(getattr(outcome, "cancelled", False)),
            final_text=str(getattr(outcome, "final_text", "") or ""),
            reason=str(getattr(outcome, "reason", "") or "")))
        return _answer("turn_finish", outcome)

    built = {"turn_start": turn_start, "shape_scope": shape_scope,
             "vet_permission": vet_permission, "llm_call": llm_call,
             "end_turn": end_turn, "turn_finish": turn_finish}
    return {m: built[m] for m in moments}


def moments_in(journal):
    """Just the doorway names, in order — what most assertions want."""
    return [entry["moment"] for entry in journal]


def visited(journal):
    """The doorways that fired, deduplicated, in first-seen order.

    ``shape_scope`` is consulted several times per turn (``tool_specs_for``,
    ``scoped_tool_names``, ``new_state``, ``build_loop``), which is real and
    is pinned on its own in ``tests/test_hooks_turn_paths.py``. Every other
    assertion wants the *set* of doorways a turn reached, not that count.
    """
    seen = []
    for moment in moments_in(journal):
        if moment not in seen:
            seen.append(moment)
    return seen


# ── A rig at the loop, and a rig at the runtime ───────────────────────

def loop_rig(tools=None, schemas=None, llm=None, max_tool_calls=5,
             session_key="s"):
    """A real ``ConversationLoop`` over a real ``HookRegistry``, no database.

    The fastest place to test a hook that only cares about the loop's own
    doorways (``shape_scope``, ``llm_call``, ``end_turn``). Turn starters and
    finishers live one level up and need :func:`make_runtime` instead.

    Lifted from ``tests/test_hooks_moments.py``'s file-local ``_rig`` so the
    composition tests can share it rather than grow a fourth copy.
    """
    import state_machine  # noqa: F401  - settles the runtime import cycle
    from runtime.conversation_loop import ConversationLoop
    from runtime.hooks import HookRegistry
    from runtime.session import RuntimeSession
    from state_machine.conversation import ConversationState, Participant
    from state_machine.conversation_phases import BASE_PHASE

    cs = ConversationState(
        [Participant("user", "user"), Participant("agent", "agent",
                                                  tools=tools or {})],
        "agent", BASE_PHASE,
        {"session_key": session_key,
         "agent_scoped_tool_names": list((tools or {}).keys())})
    session = RuntimeSession(session_key, cs)
    hooks = HookRegistry()
    runtime = SimpleNamespace(sessions={session_key: session}, hooks=hooks,
                              services={}, push_message=lambda *a, **k: None)
    llm = llm or FakeLLM()
    loop = ConversationLoop(llm, FakeRegistry(schemas or [], max_tool_calls),
                            {}, "You are a helpful agent.", runtime=runtime,
                            session_key=session_key)
    return SimpleNamespace(loop=loop, cs=cs, session=session, hooks=hooks,
                           llm=llm, runtime=runtime)


def echo_tool(record=None, name="echo", result=None):
    """One callable tool and its schema, for a turn that actually acts."""
    from plugins.native.tool import ToolResult
    from state_machine.conversation import CallableSpec

    def handler(cs, actor, args):
        if record is not None:
            record.append(args)
        return result or ToolResult(llm_summary="echoed", data={"ok": True})

    schema = {"type": "function",
              "function": {"name": name, "parameters": {}}}
    return {name: CallableSpec(name, handler=handler)}, [schema]


def tool_call(name="echo", args="{}", call_id="c1"):
    """One tool call as a fake model answers with it.

    Flat, not the nested ``{"function": {...}}`` provider envelope — the loop
    reads ``tc["name"]`` / ``tc["arguments"]`` directly
    (``conversation_loop.py:557-563``) and re-nests it itself on the way into
    history.
    """
    return {"id": call_id, "name": name, "arguments": args}


# ── Loading a sandboxed plugin the way the bridge does ────────────────

def boxed_service(tmp_path, runtime, source, *, filename, name, load=True,
                  validate=True):
    """Write a service source, bridge it, bind it, load it.

    The ``_service`` helper that ``tests/test_sandbox_hooks.py`` and
    ``tests/test_sandbox_bridge.py`` each grew their own copy of. Validates
    first, because a source with a typo'd hook moment bridges fine and then
    stands at no doorway at all — silent, which is the failure hooks are most
    prone to.
    """
    from sandbox.bridge import adapt
    from sandbox.validator import validate_file

    path = Path(tmp_path) / filename
    path.write_text(source, encoding="utf-8")
    if validate:
        report = validate_file(path)
        assert report.ok, report.render()
    module = adapt(path)
    assert module is not None, f"{filename} did not bridge"
    service = module.build_services({})[name]
    service.bind_runtime(runtime=runtime)
    if load:
        assert service.load() is True, f"{name} did not load"
    return service


def probe_source(moments, *, name="probe", answers=None):
    """Guest source for a service that journals every doorway it stands at.

    Writes the same records :func:`native_probe` writes, so the two journals
    can be compared directly. ``answers`` is baked into the source as a
    literal — a closure cannot cross into a box.

    ``answers`` maps a moment to a literal the hook returns, or to a short
    guest expression over ``payload``. ``"llm_call"`` is special: its value is
    a statement block placed inside the escort, which has a phone as well as a
    payload.
    """
    answers = dict(answers or {})
    declared = ", ".join(f'"{m}": "{_PROBE_METHODS[m]}"' for m in moments)
    chunks = []
    for moment in moments:
        if moment == "llm_call":
            chunks.append(_PROBE_BODY[moment].format(
                escort=answers.get("llm_call") or _DEFAULT_ESCORT))
        else:
            chunks.append(_PROBE_BODY[moment].format(
                answer=answers.get(moment, "None")))
    return _PROBE_TEMPLATE.format(name=name, declared=declared,
                                  bodies="\n".join(chunks))


_PROBE_METHODS = {"turn_start": "on_start", "shape_scope": "narrow",
                  "vet_permission": "gate", "llm_call": "escort",
                  "end_turn": "check_done", "turn_finish": "learn"}

_DEFAULT_ESCORT = "        return sdk.llm.proceed(request)"

_PROBE_BODY = {
    "turn_start": '''
    def on_start(self, sdk, ctx, payload):
        """Note the turn starting."""
        self._note(ctx, "turn_start")
        return {answer}
''',
    "shape_scope": '''
    def narrow(self, sdk, ctx, scope):
        """Note the toolbox, then answer."""
        self._note(ctx, "shape_scope", tools=sorted(scope.tools))
        payload = sorted(scope.tools)
        return {answer}
''',
    "vet_permission": '''
    def gate(self, sdk, ctx, payload):
        """Note the question, then answer."""
        self._note(ctx, "vet_permission", tool_name=payload.tool_name,
                   command=payload.command, stage=payload.stage,
                   origin=payload.origin)
        return {answer}
''',
    "end_turn": '''
    def check_done(self, sdk, ctx, payload):
        """Note the exit, then answer."""
        self._note(ctx, "end_turn", final_text=payload.final_text,
                   reason=payload.reason, doorman_fires=payload.doorman_fires)
        return {answer}
''',
    "turn_finish": '''
    def learn(self, sdk, ctx, payload):
        """Note the outcome. Touch nothing."""
        self._note(ctx, "turn_finish", ok=payload.ok,
                   cancelled=payload.cancelled, final_text=payload.final_text,
                   reason=payload.reason)
        return {answer}
''',
    "llm_call": '''
    def escort(self, sdk, ctx, request):
        """Note the call, then place it."""
        self._note(ctx, "llm_call", llm=request.llm,
                   messages=len(request.messages),
                   tools=len(request.tools or []))
{escort}
''',
}

_PROBE_TEMPLATE = '''
"""A service that writes down every doorway it is called at."""

from guest.bases import BaseService
from guest.hooks import (Allow, PermissionVerdict, Redrive, RequireTool,
                         SendBack)


class Probe(BaseService):
    """Stands at whichever doorways the test asked for."""

    name = "{name}"
    exports = ["journal"]
    hooks = {{{declared}}}

    def start(self, sdk):
        """Begin with an empty journal."""
        self._journal = []
        return True

    def journal(self, sdk):
        """Every visit, in order."""
        return self._journal

    def _note(self, ctx, moment, **payload):
        """One record, shaped exactly as the native probe writes it."""
        entry = {{"moment": moment, "session_key": ctx.session_key,
                 "user_id": ctx.user_id,
                 "conversation_id": ctx.conversation_id,
                 "attended": ctx.attended}}
        entry.update(payload)
        self._journal.append(entry)
{bodies}
'''


# ── Calling a handler the way production does ─────────────────────────

def call_handler(request_type: str, ctx, args: dict):
    """Run one handler under the same net ``Interpreter._execute`` puts it in.

    Tests reach for ``HANDLERS[TYPE](ctx, args)`` because it needs no
    interpreter, but that is a contract production does not have: a handler
    that raises reaches the guest as a failed Result, never as an exception.
    Calling the raw dict entry asserts a stricter promise than the kernel
    makes, so a handler dropping its own redundant ``except Exception`` would
    break such a test for a reason unrelated to behaviour.

    Mirrors ``sandbox/interpreter.py``'s handler branch, message included.
    """
    from sandbox.guest.codes import ERROR_HANDLER_ERROR, ERROR_NO_HANDLER
    from sandbox.guest.requests import Result
    from sandbox.interpreter import HANDLERS

    handler = HANDLERS.get(request_type)
    if handler is None:
        return Result.failure(f"no handler for {request_type}",
                              code=ERROR_NO_HANDLER)
    try:
        return handler(ctx, args)
    except Exception as exc:                      # noqa: BLE001 - the net
        return Result.failure(f"handler error: {exc}",
                              code=ERROR_HANDLER_ERROR)
