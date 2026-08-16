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
# A doorway is opened on a session's behalf, and the Requests must know it.
# ──────────────────────────────────────────────────────────────────────

INJECTOR = '''
"""A service that writes prompt text from the doorway it stands at."""

from guest.bases import BaseService


class Injector(BaseService):
    """Injects a pointer block at the start of every turn."""

    name = "injector"
    exports = ["outcome"]
    hooks = {"turn_start": "on_start"}
    requests = ["session.add_prompt_extra"]

    def start(self, sdk):
        self._outcome = "never ran"
        return True

    def outcome(self, sdk):
        return self._outcome

    def on_start(self, sdk, ctx, payload):
        try:
            sdk.session.add_prompt("## Things you have done before",
                                   slot="memory")
            self._outcome = "injected"
        except sdk.Failed as error:
            self._outcome = f"refused: {error}"
        return None
'''


def _injector(tmp_path, runtime):
    """Build, bind and load the injecting service the way discovery would."""
    path = tmp_path / "service_injector.py"
    path.write_text(INJECTOR, encoding="utf-8")
    module = adapt(path)
    assert module is not None, "the service did not bridge"
    service = module.build_services({})["injector"]
    service.bind_runtime(runtime=runtime)
    assert service.load() is True
    return service


def test_a_hook_writes_prompt_text_into_the_session_it_stands_in(
        tmp_path, box, runtime, registry, session):
    """The bug this exists for, end to end rather than by classification.

    A ``turn_start`` hook is opened *for* a session, but nobody is on the
    thread, so the box kept its own context — the kernel's, whose session key
    is ``None``. ``add_system_prompt_extra`` then did ``sessions.get(None)``,
    returned False, and the pointer block reached nothing. Every turn, in
    silence, with the guest's own ``ctx.session_key`` naming the session
    correctly the whole time.

    Driven through the real registry and a real box, because every layer in
    between was individually convinced it was working: the projection carried
    the right key, the handler had a sane fallback, and the policy branch had
    a passing test — which classified a call shape with no ``key`` that no
    caller makes.
    """
    written = {}
    runtime.sessions = {"repl": session}
    runtime.add_system_prompt_extra = (
        lambda key, slot, value: written.setdefault(key, {}).update({slot: value}))
    box.bind_context(lambda session_key=None: SimpleNamespace(
        config={}, db=None, services={}, runtime=runtime, user_id=7,
        session_key=session_key))

    service = _injector(tmp_path, runtime)
    try:
        registry.start_turn(session, runtime)
        assert service.outcome() == "injected"
        assert written == {"repl": {"memory": "## Things you have done before"}}
    finally:
        service.unload()
        unload_box("service_injector")


def test_lending_a_session_moves_the_world_and_never_the_grant(
        tmp_path, box, runtime, registry, session):
    """Context, not chain — which is the whole of why this is safe to do.

    Rooting a hook's chain at the session would have worked too, and would
    have made the hook *attended*: unsafe Requests from a service acting on
    its own initiative would start raising approval dialogs at the top of
    every turn, which is a design that was built, shipped and deliberately
    reverted. The chain answers who is asking and stays put; only the world
    the answer is drawn from moves.
    """
    from sandbox.policy import Chain, chain_session

    seen = {}
    runtime.sessions = {"repl": session}
    runtime.add_system_prompt_extra = lambda key, slot, value: True
    box.bind_context(lambda session_key=None: SimpleNamespace(
        config={}, db=None, services={}, runtime=runtime, user_id=7,
        session_key=session_key))

    service = _injector(tmp_path, runtime)
    try:
        held = service._sandbox_box._box
        seen["chain"] = None

        original = held._call

        def watch(method, args, kwargs, target):
            """Read the chain as it stands *during* the doorway visit."""
            seen["chain"] = held.execution.chain
            seen["context_session"] = getattr(
                held.execution.context, "session_key", "<none>")
            return original(method, args, kwargs, target)

        held._call = watch
        registry.start_turn(session, runtime)
    finally:
        service.unload()
        unload_box("service_injector")

    chain: Chain = seen["chain"]
    assert seen["context_session"] == "repl", "the world moved"
    assert chain.root == "service:injector", "the grant did not"
    assert chain_session(chain) != "repl"
    assert not chain.attended, "a hook must not become askable"


# ──────────────────────────────────────────────────────────────────────
# What the escort doorway carries that the other five do not.
#
# ``build_shim`` and ``_build_escort`` are separate code paths, and the
# differences between them are not all deliberate.
# ──────────────────────────────────────────────────────────────────────

ESCORT_WRITER = '''
"""An escort that writes prompt text from inside the call it escorts."""

from guest.bases import BaseService


class Writer(BaseService):
    """Escorts the model call, and tries to touch its own session."""

    name = "writer"
    exports = ["outcome"]
    hooks = {"llm_call": "escort"}
    requests = ["session.add_prompt_extra"]

    def start(self, sdk):
        """Begin."""
        self._outcome = "never ran"
        return True

    def outcome(self, sdk):
        """What happened when it tried."""
        return self._outcome

    def escort(self, sdk, ctx, request):
        """Place the call, and write a note against this session."""
        try:
            sdk.session.add_prompt("a note from the escort", slot="escort")
            self._outcome = "injected"
        except sdk.Failed as error:
            self._outcome = f"refused: {error}"
        return sdk.llm.proceed(request)
'''


def test_an_escort_is_lent_the_session_it_is_escorting(
        tmp_path, box, runtime, registry, session):
    """An escort's own Requests resolve against the session it stands in.

    ``build_shim`` hands the box ``for_session=<session key>`` so a hook's own
    Requests reach the session whose doorway it stands at — the fix for
    ``sessions.get(None)`` silently swallowing every ``turn_start`` injection.
    ``_build_escort`` is a separate function and did not get that fix when the
    other five did, so a boxed escort touching ``sdk.session.*`` reached the
    kernel's default session instead.

    Silent in exactly the way the original was: nothing raises, the write
    returns, and the text lands nowhere.
    """
    written = {}
    runtime.sessions = {"repl": session}
    runtime.add_system_prompt_extra = (
        lambda key, slot, value: written.setdefault(str(key), {}).update(
            {slot: value}))
    box.bind_context(lambda session_key=None: SimpleNamespace(
        config={}, db=None, services={}, runtime=runtime, user_id=7,
        session_key=session_key))

    path = tmp_path / "service_writer.py"
    path.write_text(ESCORT_WRITER, encoding="utf-8")
    module = adapt(path)
    assert module is not None, "the service did not bridge"
    service = module.build_services({})["writer"]
    service.bind_runtime(runtime=runtime)
    assert service.load() is True

    base, _calls = _backend("answered")
    try:
        handler = registry.wrap_llm_call(session, runtime, base)
        handler(ModelRequest(llm="gpt-4o",
                             messages=[{"role": "user", "content": "hi"}]))
        outcome = service.outcome()
    finally:
        service.unload()
        unload_box("service_writer")

    assert outcome != "never ran", "the escort never ran at all"
    assert outcome == "injected", outcome
    assert written == {"repl": {"escort": "a note from the escort"}}


ESCORT_ANSWERS = '''
"""An escort that answers for itself after placing the call."""

from guest.bases import BaseService
from guest.hooks import ModelResponse


class Answerer(BaseService):
    """Dials, then hands back a response of its own construction."""

    name = "answerer"
    hooks = {"llm_call": "escort"}

    def start(self, sdk):
        """Begin."""
        return True

    def escort(self, sdk, ctx, request):
        """Place the call, then answer with tool calls of our own."""
        sdk.llm.proceed(request)
        return ModelResponse(content="rewritten",
                             tool_calls=[{"id": "x1", "name": "echo",
                                          "arguments": "{}"}])
'''


def test_a_boxed_escort_can_hand_back_tool_calls(tmp_path, box, runtime,
                                                 registry, session):
    """An escort shapes what the model wants to *do*, not just what it said.

    This used to depend on whether the escort had placed the call.
    ``bridge._make_response`` — the path taken when an escort answers *without*
    dialing — has always carried ``content``, ``tool_calls`` and ``error``, so
    the capability existed through one door. The dialed branch copied only
    ``content``, so the same escort keeping the same object lost the other two.

    That was an inconsistency between two code paths rather than a policy
    about what a plugin may do, which is why closing it is a fix and not a
    grant: nothing new became possible, it just stopped depending on a detail
    the author has no reason to think about.
    """
    path = tmp_path / "service_answerer.py"
    path.write_text(ESCORT_ANSWERS, encoding="utf-8")
    module = adapt(path)
    assert module is not None, "the service did not bridge"
    service = module.build_services({})["answerer"]
    service.bind_runtime(runtime=runtime)
    assert service.load() is True

    base, calls = _backend("from the model")
    try:
        handler = registry.wrap_llm_call(session, runtime, base)
        response = handler(ModelRequest(
            llm="gpt-4o", messages=[{"role": "user", "content": "hi"}]))
    finally:
        service.unload()
        unload_box("service_answerer")

    assert len(calls) == 1
    assert response.content == "rewritten", "the escort's content was ignored"
    assert response.tool_calls == [
        {"id": "x1", "name": "echo", "arguments": "{}"}], (
        "the escort's tool_calls were dropped on the dialed path")


ESCORT_QUIET = '''
"""An escort that rewrites the text and says nothing about tool calls."""

from guest.bases import BaseService


class Quietly(BaseService):
    """Edits what was said, leaves what was wanted alone."""

    name = "quietly"
    hooks = {"llm_call": "escort"}

    def start(self, sdk):
        """Begin."""
        return True

    def escort(self, sdk, ctx, request):
        """Place the call, then hand back its own response, edited."""
        response = sdk.llm.proceed(request)
        response.content = response.content.upper()
        return response
'''


def test_an_escort_that_only_edits_text_leaves_tool_calls_alone(
        tmp_path, box, runtime, registry, session):
    """The other half of the rule: a round trip must be a no-op.

    ``tool_calls`` is applied only when the answer carries the key, so an
    escort that took the model's response and changed one field writes back
    what was already there. Without that condition, closing the asymmetry
    above would have let an escort silently clear a model's tool calls just by
    returning a response it had not thought about.
    """
    path = tmp_path / "service_quietly.py"
    path.write_text(ESCORT_QUIET, encoding="utf-8")
    module = adapt(path)
    assert module is not None, "the service did not bridge"
    service = module.build_services({})["quietly"]
    service.bind_runtime(runtime=runtime)
    assert service.load() is True

    wanted = [{"id": "t1", "name": "echo", "arguments": "{}"}]

    def base(request):
        """A model that wants to act."""
        return SimpleNamespace(content="hello", tool_calls=list(wanted),
                               error=None, is_error=False)

    try:
        handler = registry.wrap_llm_call(session, runtime, base)
        response = handler(ModelRequest(
            llm="gpt-4o", messages=[{"role": "user", "content": "hi"}]))
    finally:
        service.unload()
        unload_box("service_quietly")

    assert response.content == "HELLO"
    assert response.tool_calls == wanted, (
        "an escort that never mentioned tool_calls still changed them")


def test_a_boxed_gate_is_not_shown_the_decision():
    """``PermissionQuery.decision`` reaches a native gate and not a boxed one.

    ``HookRegistry.vet_permission`` takes a ``decision`` and puts it on the
    query; ``project_payload`` carries ``tool_name``, ``command``, ``stage``,
    ``origin``, ``request`` and ``chain`` — and stops. So the kernel's own
    reason for asking is the one thing a sandboxed gate cannot read, which is
    awkward for the gate that would most like it.

    Recorded rather than fixed: projecting a ``Decision`` means deciding what
    of it may cross, which is a policy question rather than an oversight.
    """
    from sandbox.hooks import project_payload

    query = SimpleNamespace(tool_name="shell", command="rm -rf /",
                            stage="approval", origin="request",
                            request=None, chain=None,
                            decision=SimpleNamespace(reason="unsafe"))

    projected = project_payload("vet_permission", query)

    assert set(projected) == {"tool_name", "command", "stage", "origin",
                              "request", "chain"}


def test_a_shaper_cannot_reorder_the_toolbox_whatever_the_docs_say():
    """Narrowing keeps names, not order — so "reorder" is not a capability.

    ``guest/hooks.Scope`` and ``templates/hook_template.py`` both tell an
    author a shaper may "hide and reorder". It cannot: the payload arrives
    ``sorted()`` and ``narrow_scope`` stores a ``set``, so the order a shaper
    returns is discarded at both ends. Harmless in itself, and a docs fix
    rather than a code one — but an author who believed it would write a
    prioritizer that silently did nothing.
    """
    from sandbox.hooks import narrow_scope

    registry = SimpleNamespace(tools={"a": 1, "b": 2, "c": 3},
                               visible_tool_names=None)

    narrow_scope(registry, ["c", "a"])

    assert registry.visible_tool_names == {"a", "c"}
    assert isinstance(registry.visible_tool_names, set), (
        "order now survives; the docs' 'reorder' claim could be made true")
