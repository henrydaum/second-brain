"""Sandboxed code standing at the kernel's doorways.

A hook is the one inbound thing in the SDK: everywhere else the plugin asks
and the kernel answers, here the kernel calls and the plugin answers. These
tests drive the *real* ``HookRegistry`` rather than a stand-in, because the
claim being made is that a sandboxed hook is indistinguishable from a native
one to the turn that meets it.

The other half of the claim is that it can never make things worse. A hook
that is unloaded, broken, slow, or answering nonsense has exactly one effect:
the turn proceeds as though nobody were standing there.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import sandbox  # noqa: F401  - installs the ``guest`` package alias
from guest.loader import unload_box
from runtime.hooks import (HookRegistry, ModelRequest, PermissionVerdict,
                           Redrive, RequireTool, SendBack, TurnEnding,
                           TurnOutcome)
from sandbox import Sandbox
from sandbox.bridge import adapt, configure

SERVICE = '''
"""A migrated service standing at every doorway."""

from guest.bases import BaseService
from guest.hooks import PermissionVerdict, Redrive, RequireTool, SendBack


class Doorman(BaseService):
    """Watches everything."""

    name = "doorman"
    exports = ["seen"]
    hooks = {
        "turn_start": "on_start",
        "shape_scope": "narrow",
        "vet_permission": "gate",
        "llm_call": "escort",
        "end_turn": "check_done",
        "turn_finish": "learn",
    }

    def start(self, sdk):
        """Begin."""
        self._seen = []
        return True

    def seen(self, sdk):
        """Everything observed, in order."""
        return self._seen

    def on_start(self, sdk, ctx, payload):
        """Note who is starting a turn."""
        self._seen.append(("turn_start", ctx.session_key, ctx.user_id,
                           ctx.conversation_id, ctx.attended))
        return None

    def narrow(self, sdk, ctx, scope):
        """Hide anything shell-shaped, and try to invent a tool."""
        self._seen.append(("shape_scope", sorted(scope.tools)))
        keep = [t for t in scope.tools if "shell" not in t]
        return keep + ["invented_tool"]

    def gate(self, sdk, ctx, query):
        """Refuse production, abstain otherwise."""
        self._seen.append(("vet_permission", query.tool_name, query.stage,
                           query.origin, query.command))
        if "prod" in (query.command or ""):
            return PermissionVerdict(allow=False, reason="not in production")
        return None

    def escort(self, sdk, ctx, request):
        """Retry once when the model says nothing."""
        self._seen.append(("llm_call", request.llm, len(request.messages)))
        response = sdk.llm.proceed(request)
        if not response.content.strip():
            request.messages = request.messages + [
                {"role": "user", "content": "Please answer."}]
            response = sdk.llm.proceed(request)
        return response

    def check_done(self, sdk, ctx, ending):
        """Demand text once, then relent."""
        self._seen.append(("end_turn", ending.reason, ending.doorman_fires))
        if not ending.final_text and ending.doorman_fires == 0:
            return SendBack("Say something.", ephemeral=True)
        return None

    def learn(self, sdk, ctx, outcome):
        """Observe the finished turn."""
        self._seen.append(("turn_finish", outcome.ok, outcome.cancelled,
                           outcome.final_text))
        return None
'''


@pytest.fixture
def box():
    """A sandbox the bridge routes migrated plugins through."""
    made = Sandbox()
    configure(made)
    yield made
    configure(None)
    made.shutdown()


@pytest.fixture(autouse=True)
def clean_boxes():
    """Boxes are module caches; a leak hides staleness."""
    yield
    for name in ("service_doorman", "service_quiet", "service_broken"):
        unload_box(name)


@pytest.fixture
def registry():
    """The real hook registry, not a stand-in."""
    return HookRegistry()


@pytest.fixture
def runtime(registry):
    """Enough runtime for the doorways to work."""
    return SimpleNamespace(hooks=registry, services={},
                           is_attended=lambda key: True)


@pytest.fixture
def session():
    """A session with an identity to project."""
    return SimpleNamespace(key="repl", user_id=7, conversation_id=42)


def _service(tmp_path, runtime, source=SERVICE, filename="service_doorman.py",
             load=True):
    """Build, bind and load a migrated service the way discovery would."""
    path = tmp_path / filename
    path.write_text(source, encoding="utf-8")
    module = adapt(path)
    assert module is not None, "the service did not bridge"
    service = module.build_services({})[
        "doorman" if filename == "service_doorman.py" else "quiet"]
    service.bind_runtime(runtime=runtime)
    if load:
        assert service.load() is True
    return service


# ──────────────────────────────────────────────────────────────────────
# The four data-only moments.
# ──────────────────────────────────────────────────────────────────────

def test_turn_start_receives_the_session_identity(tmp_path, box, runtime,
                                                  registry, session):
    """A hook cannot hold the session, so it is told who the session is."""
    service = _service(tmp_path, runtime)
    registry.start_turn(session, runtime)

    moment, key, user, conversation, attended = service.seen()[0]
    assert (moment, key, user, conversation) == ("turn_start", "repl", 7, 42)
    assert attended is True
    service.unload()


def test_a_gate_can_refuse_and_can_abstain(tmp_path, box, runtime, registry,
                                           session):
    """A verdict must arrive as the kernel's own type, not a lookalike."""
    service = _service(tmp_path, runtime)

    verdict = registry.vet_permission(session, "shell", "deploy to prod",
                                      runtime=runtime)
    assert isinstance(verdict, PermissionVerdict)
    assert verdict.allow is False
    assert verdict.reason == "not in production"

    # Abstention has to be distinguishable from "allow", or a silent hook
    # would start overriding the kernel's own default.
    assert registry.vet_permission(session, "shell", "ls",
                                   runtime=runtime) is None
    service.unload()


def test_a_doorman_can_send_the_agent_back(tmp_path, box, runtime, registry,
                                           session):
    """The verdict crosses as data and comes back a real SendBack."""
    service = _service(tmp_path, runtime)

    verdict = registry.vet_end_turn(session, runtime,
                                    TurnEnding(final_text="",
                                               reason="model_finished"))
    assert isinstance(verdict, SendBack)
    assert verdict.note == "Say something."
    assert verdict.ephemeral is True
    assert verdict.allow_tools is True        # defaulted, not dropped

    # The fire budget is visible to the hook, so it can relent.
    assert registry.vet_end_turn(
        session, runtime,
        TurnEnding(final_text="done", doorman_fires=1)) is None
    service.unload()


def test_turn_finish_observes_the_outcome(tmp_path, box, runtime, registry,
                                          session):
    """An observer sees the whole outcome and changes nothing."""
    service = _service(tmp_path, runtime)
    registry.finish_turn(session, TurnOutcome(ok=False, cancelled=True,
                                              final_text="partial"), runtime)
    # Compared element-wise, not as a tuple: what the guest appended crosses
    # as plain data, and a tuple comes back a list. The sibling tests unpack
    # for the same reason — an assertion that only holds in-process would
    # quietly stop testing the isolated path.
    assert [list(row) for row in service.seen()] == [
        ["turn_finish", False, True, "partial"]]
    service.unload()


# ──────────────────────────────────────────────────────────────────────
# Scope shaping: names only, and narrowing only.
# ──────────────────────────────────────────────────────────────────────

def test_a_shaper_hides_tools_but_cannot_invent_them(tmp_path, box, runtime,
                                                     registry, session):
    """Narrowing is safe and widening is not, so the answer is intersected.

    The fixture deliberately returns a tool that does not exist. Trusting it
    would put a name in scope that resolves to nothing.
    """
    service = _service(tmp_path, runtime)
    scope = SimpleNamespace(tools={"read_file": 1, "shell": 2, "grep": 3},
                            visible_tool_names=None)

    shaped = registry.shape_scope(session, scope, runtime=runtime)
    assert shaped.visible_tool_names == {"read_file", "grep"}
    assert "invented_tool" not in shaped.visible_tool_names
    service.unload()


def test_a_shaper_only_narrows_an_existing_restriction(tmp_path, box, runtime,
                                                       registry, session):
    """A shaper joining an already-restricted scope cannot widen it back."""
    service = _service(tmp_path, runtime)
    scope = SimpleNamespace(tools={"read_file": 1, "shell": 2, "grep": 3},
                            visible_tool_names={"grep"})

    shaped = registry.shape_scope(session, scope, runtime=runtime)
    assert shaped.visible_tool_names == {"grep"}
    service.unload()


# ──────────────────────────────────────────────────────────────────────
# The escort: the one doorway that calls back into the kernel.
# ──────────────────────────────────────────────────────────────────────

def _backend(*replies):
    """A fake model that answers the given things in order."""
    calls = []

    def base(request):
        """One round trip."""
        calls.append(list(request.messages))
        text = replies[min(len(calls) - 1, len(replies) - 1)]
        return SimpleNamespace(content=text, tool_calls=[], error=None,
                               is_error=False)
    return base, calls


def test_an_escort_can_place_the_call_twice(tmp_path, box, runtime, registry,
                                            session):
    """The escort holds the phone: it decides when to dial, and how often."""
    service = _service(tmp_path, runtime)
    base, calls = _backend("", "A real answer.")

    handler = registry.wrap_llm_call(session, runtime, base)
    brain = SimpleNamespace(model_name="gpt-4o", loaded=True)
    response = handler(ModelRequest(llm=brain,
                                    messages=[{"role": "user", "content": "hi"}]))

    assert response.content == "A real answer."
    assert len(calls) == 2, "the escort should have retried the empty answer"
    # The retry carried the escort's added message, so the rewrite reached the
    # live request rather than a copy that was thrown away.
    assert calls[1][-1] == {"role": "user", "content": "Please answer."}
    service.unload()


@pytest.fixture
def configured_brains(monkeypatch):
    """Declare which profile names exist, without building real brains.

    ``apply_model_request`` asks the llm registry whether a name is a
    configured profile, which is the only thing it needs to know to decide
    whether an escort's swap is honoured.
    """
    import llm

    known = {"gpt-4o", "claude", "other"}
    monkeypatch.setattr(llm, "brain",
                        lambda name: object() if name in known else None)
    return known


def test_an_escort_sees_the_brain_by_name(tmp_path, box, runtime, registry,
                                          session):
    """A live backend cannot cross; its name can, and that is the contract.

    ``ModelRequest.llm`` *is* the name — the loop resolves it when it places
    the call. This test used to hand in an object with a ``model_name``
    attribute, which is what the kernel carried before the LLM became kernel
    routing, and it was the reason a real escort was always shown "".
    """
    service = _service(tmp_path, runtime)
    base, _ = _backend("hello")
    handler = registry.wrap_llm_call(session, runtime, base)
    handler(ModelRequest(llm="claude",
                         messages=[{"role": "user", "content": "hi"}]))

    moment, llm, count = next(s for s in service.seen() if s[0] == "llm_call")
    assert (llm, count) == ("claude", 1)
    service.unload()


def test_an_escort_can_swap_the_brain(tmp_path, box, runtime, registry,
                                      session, configured_brains):
    """Naming another configured profile switches which brain takes the call."""
    source = SERVICE.replace(
        "        response = sdk.llm.proceed(request)\n"
        "        if not response.content.strip():",
        "        request.llm = 'other'\n"
        "        response = sdk.llm.proceed(request)\n"
        "        if not response.content.strip():")
    service = _service(tmp_path, runtime, source=source)

    seen = {}

    def base(request):
        """Record which brain actually took the call."""
        seen["llm"] = request.llm
        return SimpleNamespace(content="ok", tool_calls=[], error=None,
                               is_error=False)

    handler = registry.wrap_llm_call(session, runtime, base)
    handler(ModelRequest(llm="gpt-4o",
                         messages=[{"role": "user", "content": "hi"}]))
    # The *name*, not a brain: putting an object here would work only by
    # accident, since the loop calls _brain() on whatever it finds.
    assert seen["llm"] == "other"
    service.unload()


def test_naming_an_unknown_brain_leaves_the_call_alone(tmp_path, box, runtime,
                                                       registry, session,
                                                       configured_brains):
    """An escort naming a profile that is not configured must not retarget it.

    Silently falling back to the default would be the worst outcome: the turn
    succeeds and quietly uses the wrong model.
    """
    source = SERVICE.replace(
        "        response = sdk.llm.proceed(request)\n"
        "        if not response.content.strip():",
        "        request.llm = 'nonexistent'\n"
        "        response = sdk.llm.proceed(request)\n"
        "        if not response.content.strip():")
    service = _service(tmp_path, runtime, source=source)

    seen = {}

    def base(request):
        """Record the brain."""
        seen["llm"] = request.llm
        return SimpleNamespace(content="ok", tool_calls=[], error=None,
                               is_error=False)

    handler = registry.wrap_llm_call(session, runtime, base)
    handler(ModelRequest(llm="gpt-4o",
                         messages=[{"role": "user", "content": "hi"}]))
    assert seen["llm"] == "gpt-4o"
    service.unload()


def test_proceed_is_refused_outside_a_llm_call_hook(tmp_path, box, runtime,
                                                      registry, session):
    """The token is the whole gate: no token, no call.

    Without this, any sandboxed code could reach whatever escort happened to
    be in flight — which is the sort of ambient authority the Request model
    exists to remove.
    """
    from sandbox.hooks import phone

    assert phone("") is None
    assert phone("not-a-real-token") is None


# ──────────────────────────────────────────────────────────────────────
# A hook can never make the turn worse.
# ──────────────────────────────────────────────────────────────────────

def test_an_unloaded_service_abstains(tmp_path, box, runtime, registry,
                                      session):
    """Hooks stay standing across an unload/reload; they just fall silent."""
    service = _service(tmp_path, runtime)
    service.unload()

    assert registry.vet_permission(session, "shell", "deploy to prod",
                                   runtime=runtime) is None
    assert registry.vet_end_turn(session, runtime, TurnEnding()) is None

    # And come back when it does.
    assert service.load() is True
    assert registry.vet_permission(session, "shell", "deploy to prod",
                                   runtime=runtime) is not None
    service.unload()


def test_a_raising_hook_abstains(tmp_path, box, runtime, registry, session):
    """A hook that breaks is a hook that said nothing."""
    source = SERVICE.replace(
        'if "prod" in (query.command or ""):',
        'raise ValueError("boom")\n        if False:')
    service = _service(tmp_path, runtime, source=source)

    assert registry.vet_permission(session, "shell", "deploy to prod",
                                   runtime=runtime) is None
    service.unload()


def test_a_raising_escort_still_places_the_call(tmp_path, box, runtime,
                                                registry, session):
    """Escort failure is transparent, not fatal: the model is still called."""
    source = SERVICE.replace(
        "        response = sdk.llm.proceed(request)\n"
        "        if not response.content.strip():",
        "        raise ValueError('boom')\n"
        "        if False:")
    service = _service(tmp_path, runtime, source=source)
    base, calls = _backend("still answered")

    handler = registry.wrap_llm_call(session, runtime, base)
    response = handler(ModelRequest(
        llm=SimpleNamespace(model_name="gpt-4o", loaded=True),
        messages=[{"role": "user", "content": "hi"}]))

    assert response.content == "still answered"
    assert len(calls) == 1
    service.unload()


def test_unloading_removes_every_hook(tmp_path, box, runtime, registry):
    """A hook outliving its plugin is a leak with no symptom."""
    service = _service(tmp_path, runtime)
    assert any(registry._hooks[m] for m in registry._hooks)

    service.unload()
    assert not any(registry._hooks[m] for m in registry._hooks)


def test_hooks_register_when_load_precedes_bind(tmp_path, box, runtime,
                                                registry, session):
    """Boot binds then loads; a live reload can do the opposite. Both work."""
    path = tmp_path / "service_doorman.py"
    path.write_text(SERVICE, encoding="utf-8")
    service = adapt(path).build_services({})["doorman"]

    assert service.load() is True            # loaded with no runtime yet
    assert not any(registry._hooks[m] for m in registry._hooks)

    service.bind_runtime(runtime=runtime)    # runtime arrives afterwards
    assert registry.vet_permission(session, "shell", "deploy to prod",
                                   runtime=runtime) is not None
    service.unload()


# ──────────────────────────────────────────────────────────────────────
# Declarations are checked before anything runs.
# ──────────────────────────────────────────────────────────────────────

QUIET = '''
"""A service with a bad hook declaration."""

from guest.bases import BaseService


class Quiet(BaseService):
    """Does little."""

    name = "quiet"
    hooks = {MOMENT: "handler"}

    def start(self, sdk):
        """Begin."""
        return True

    def handler(self, sdk, ctx, payload):
        """Do nothing."""
        return None
'''


def test_an_unknown_moment_is_rejected(tmp_path):
    """A typo'd moment is a doorway nobody stands at — silent without this."""
    from sandbox.validator import validate_file

    path = tmp_path / "service_quiet.py"
    path.write_text(QUIET.replace("MOMENT", '"turn_finished"'),
                    encoding="utf-8")
    report = validate_file(path)
    assert not report.ok
    assert "not a hook moment" in report.render()


def test_a_missing_handler_is_rejected(tmp_path):
    """Naming a method that does not exist fails at the doorway otherwise."""
    from sandbox.validator import validate_file

    path = tmp_path / "service_quiet.py"
    path.write_text(QUIET.replace("MOMENT", '"end_turn"').replace(
        '"handler"', '"absent"'), encoding="utf-8")
    report = validate_file(path)
    assert not report.ok
    assert "no such method" in report.render()


import time
from sandbox import Sandbox, provenance
from sandbox.console import Console
from sandbox.guest.requests import Request, Result
from sandbox.interpreter import Execution, Interpreter
from sandbox.policy import SAFE, UNSAFE, Chain, Decision, classify

# ──────────────────────────────────────────────────────────────────────
# The escort's view of the model.
# ──────────────────────────────────────────────────────────────────────

def test_an_escort_is_shown_the_profile_name(monkeypatch):
    """``ModelRequest.llm`` is a name. Reaching for ``.model_name`` on a string
    yields nothing, so every escort ever built was shown ""."""
    from runtime.hooks import ModelRequest
    from sandbox.hooks import project_model_request

    request = ModelRequest(llm="fast-profile", messages=[])
    assert project_model_request(request)["llm"] == "fast-profile"


def test_an_escort_swaps_by_name_and_unknown_names_are_ignored(monkeypatch):
    """Backends stopped being services when the LLM became kernel routing, so
    looking a profile up in ``runtime.services`` found nothing and no
    sandboxed escort could swap a model at all."""
    import llm as llm_registry
    from runtime.hooks import ModelRequest
    from sandbox.hooks import apply_model_request

    monkeypatch.setattr(llm_registry, "brain",
                        lambda name: object() if name == "big" else None)
    runtime = SimpleNamespace(services={}, config={})
    request = ModelRequest(llm="small", messages=[])

    apply_model_request(request, {"llm": "big"}, runtime)
    assert request.llm == "big"          # the name, never a brain object

    apply_model_request(request, {"llm": "does-not-exist"}, runtime)
    assert request.llm == "big"          # silently retargeting is the worst case


# ──────────────────────────────────────────────────────────────────────
# Whose work is this? A hook stands in the turn that called it.
# ──────────────────────────────────────────────────────────────────────

class _SpyBox:
    """A box that records who was asking when it was called."""

    alive = True

    def __init__(self):
        self.caller = None
        self.calls = 0

    def call(self, method, *args, target="", **kwargs):
        from sandbox import provenance
        from sandbox.guest.requests import Result

        self.caller = provenance.current()
        self.calls += 1
        return Result(data=None)


def _spied(moment="turn_start", method="on_start"):
    """A shim over a spy box, plus the spy."""
    from sandbox.hooks import build_shim

    spy = _SpyBox()
    service = SimpleNamespace(name="memory", _sandbox_box=spy)
    return build_shim(service, moment, method), spy


def _ctx(session_key, runtime=None):
    """A host-side hook context: a *live* session, not the guest projection."""
    session = SimpleNamespace(key=session_key, user_id=1, conversation_id=3)
    return SimpleNamespace(session=session, runtime=runtime, moment="turn_start")


def test_a_hook_stands_in_the_turn_that_called_it():
    """The kernel calls a doorway on the drive thread, during one session's
    turn, so the turn is what caused the hook's work — which is exactly what a
    chain records. Before this the box was called with nobody on the thread and
    fell back to its own chain, rooted at the service."""
    shim, spy = _spied()
    shim(_ctx("repl"), SimpleNamespace())

    assert spy.caller is not None, "the box was called with no caller"
    assert spy.caller.chain.root == "repl"


def test_the_key_is_read_where_project_context_reads_it():
    """The host context spells it ``session.key``; the guest projection spells
    it ``session_key``. Reading the guest name here finds nothing, falls
    through to the null context, and leaves the fix looking applied while
    changing nothing — silent in the one direction that matters."""
    shim, spy = _spied()
    # Shaped like the *guest* projection. Nothing to stand in.
    shim(SimpleNamespace(session_key="repl", runtime=None), SimpleNamespace())

    assert spy.calls == 1
    assert spy.caller is None


def test_only_the_chain_travels_and_never_the_hook_context():
    """The ``ctx`` a doorway gets is a ``HookContext`` — session, runtime,
    moment. It is *not* the ``SecondBrainContext`` handlers read ``config``,
    ``db`` and ``user_id`` off. Handing it over as the caller's context
    replaced a working context with one missing every field a handler needs,
    and the write this mechanism exists to permit came back "config is not
    available in this kernel" — approved by a person, then failed.

    ``None`` leaves the box on the context the interpreter built for it.
    """
    shim, spy = _spied()
    ctx = _ctx("repl")
    shim(ctx, SimpleNamespace())

    assert spy.caller.chain.root == "repl", "the chain still travels"
    assert spy.caller.context is None
    assert spy.caller.context is not ctx


def test_a_hook_in_a_subagents_turn_stays_unattended():
    """The safety property, and it falls out of the same line rather than
    needing its own rule: a child's session key is real but is never the active
    one, so ``attended_now`` refuses it exactly as it does everywhere else."""
    from runtime.subagents import SESSION_PREFIX
    from sandbox.policy import attended_now

    runtime = SimpleNamespace(is_attended=lambda key: key == "repl")

    child_shim, child_spy = _spied()
    child_shim(_ctx(f"{SESSION_PREFIX}42", runtime), SimpleNamespace())
    foreground_shim, foreground_spy = _spied()
    foreground_shim(_ctx("repl", runtime), SimpleNamespace())

    assert child_spy.caller.chain.root == f"{SESSION_PREFIX}42"
    assert attended_now(child_spy.caller.chain, runtime) is False
    assert attended_now(foreground_spy.caller.chain, runtime) is True


def test_the_hook_can_now_be_asked_about_an_unsafe_request():
    """The whole point, stated as the symptom it fixes.

    ``config.write`` is UNSAFE either way — that has not changed and must not.
    What changed is whether anybody is asked: on the service's own chain the
    root is not a session key at all, so ``attended_now`` said no and the
    Request was refused outright while a person sat watching the very turn
    that triggered it.
    """
    from sandbox import Chain, Request
    from sandbox.policy import attended_now, classify

    runtime = SimpleNamespace(is_attended=lambda key: key == "repl")
    request = Request("config.write", {"key": "sync_directories", "value": []})

    service_chain = Chain(root="service:memory")
    turn_chain = Chain(root="repl").push("memory")

    assert not classify(request, service_chain).safe
    assert not classify(request, turn_chain).safe, "still unsafe, still asked"

    assert attended_now(service_chain, runtime) is False, "refused, never asked"
    assert attended_now(turn_chain, runtime) is True, "a dialog is reachable"


SEEDER = '''
"""A service that needs a kernel setting to do its job."""

from guest.bases import BaseService


class Seeder(BaseService):
    """Seeds a kernel setting on the first attended turn."""

    name = "seeder"
    exports = ["outcome"]
    hooks = {"turn_start": "on_start"}
    requests = ["config.read", "config.write"]

    def start(self, sdk):
        """Begin."""
        self._outcome = "never ran"
        return True

    def outcome(self, sdk):
        """What happened when it tried."""
        return self._outcome

    def on_start(self, sdk, ctx, payload):
        """Try to add a folder to the indexer."""
        if not ctx.attended:
            self._outcome = "not attended"
            return None
        try:
            sdk.config.write("sync_directories", ["/memory"])
            self._outcome = "written"
        except sdk.Denied as denied:
            self._outcome = f"denied: {denied}"
        except sdk.Failed as failed:
            self._outcome = f"failed: {failed}"
        return None
'''


def test_an_unsafe_request_from_a_hook_reaches_the_approver(tmp_path, runtime,
                                                            registry, session):
    """The end-to-end claim, driven through the real registry and a real box.

    This is the shape of the bug that started it: a service gating on
    ``ctx.attended``, passing, and then being refused because the chain said
    nobody was there. The approver is a spy rather than a dialog, so what is
    pinned is that the question is *asked* — which is the whole difference.
    """
    from sandbox import Sandbox
    from sandbox.bridge import adapt, configure

    asked = []

    def approve(chain, request, decision):
        asked.append((chain.root, request.type, request.args.get("key")))
        return True

    # A context shaped like the real one: handlers read ``config`` off it, and
    # a bare namespace is what made this test report the *product* bug it was
    # written to catch as though it were a harness gap.
    settings = {}
    made = Sandbox(approve=approve, runtime=runtime,
                   context=SimpleNamespace(config=settings, db=None,
                                           services={}, user_id=1,
                                           session_key="repl"))
    configure(made)
    try:
        path = tmp_path / "service_seeder.py"
        path.write_text(SEEDER, encoding="utf-8")
        service = adapt(path).build_services({})["seeder"]
        service.bind_runtime(runtime=runtime)
        assert service.load() is True

        registry.start_turn(session, runtime)
        outcome = service.outcome()
        service.unload()
    finally:
        configure(None)
        made.shutdown()
        unload_box("service_seeder")

    assert asked, ("the hook's unsafe Request never reached the approver — it "
                   "was refused on the service's own unattended chain")
    root, kind, key = asked[0]
    assert (root, kind, key) == ("repl", "config.write", "sync_directories")

    # Being asked is only half of it. The write has to *land* — the first
    # version of this fix handed the hook's own context to the handler, so an
    # approved write failed with "config is not available in this kernel" and
    # sync_directories stayed empty.
    assert outcome == "written", outcome
    assert settings.get("sync_directories") == ["/memory"]
