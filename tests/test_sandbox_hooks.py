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
        "model_call": "escort",
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
        self._seen.append(("model_call", request.llm, len(request.messages)))
        response = sdk.model.proceed(request)
        if not response.content.strip():
            request.messages = request.messages + [
                {"role": "user", "content": "Please answer."}]
            response = sdk.model.proceed(request)
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
    assert ("turn_finish", False, True, "partial") in service.seen()
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

    handler = registry.wrap_model_call(session, runtime, base)
    brain = SimpleNamespace(model_name="gpt-4o", loaded=True)
    response = handler(ModelRequest(llm=brain,
                                    messages=[{"role": "user", "content": "hi"}]))

    assert response.content == "A real answer."
    assert len(calls) == 2, "the escort should have retried the empty answer"
    # The retry carried the escort's added message, so the rewrite reached the
    # live request rather than a copy that was thrown away.
    assert calls[1][-1] == {"role": "user", "content": "Please answer."}
    service.unload()


def test_an_escort_sees_the_brain_by_name(tmp_path, box, runtime, registry,
                                          session):
    """A live backend cannot cross; its name can, and that is the contract."""
    service = _service(tmp_path, runtime)
    base, _ = _backend("hello")
    handler = registry.wrap_model_call(session, runtime, base)
    handler(ModelRequest(llm=SimpleNamespace(model_name="claude", loaded=True),
                         messages=[{"role": "user", "content": "hi"}]))

    moment, llm, count = next(s for s in service.seen() if s[0] == "model_call")
    assert (llm, count) == ("claude", 1)
    service.unload()


def test_an_escort_can_swap_the_brain(tmp_path, box, runtime, registry,
                                      session):
    """Setting request.llm to another loaded backend's name switches it."""
    source = SERVICE.replace(
        "        response = sdk.model.proceed(request)\n"
        "        if not response.content.strip():",
        "        request.llm = 'other'\n"
        "        response = sdk.model.proceed(request)\n"
        "        if not response.content.strip():")
    other = SimpleNamespace(model_name="other", loaded=True)
    runtime.services["other"] = other
    service = _service(tmp_path, runtime, source=source)

    seen = {}

    def base(request):
        """Record which brain actually took the call."""
        seen["llm"] = request.llm
        return SimpleNamespace(content="ok", tool_calls=[], error=None,
                               is_error=False)

    handler = registry.wrap_model_call(session, runtime, base)
    handler(ModelRequest(llm=SimpleNamespace(model_name="gpt-4o", loaded=True),
                         messages=[{"role": "user", "content": "hi"}]))
    assert seen["llm"] is other
    service.unload()


def test_naming_an_unloaded_brain_leaves_the_call_alone(tmp_path, box, runtime,
                                                        registry, session):
    """An escort naming a brain that is not there must not break the turn."""
    source = SERVICE.replace(
        "        response = sdk.model.proceed(request)\n"
        "        if not response.content.strip():",
        "        request.llm = 'nonexistent'\n"
        "        response = sdk.model.proceed(request)\n"
        "        if not response.content.strip():")
    service = _service(tmp_path, runtime, source=source)
    original = SimpleNamespace(model_name="gpt-4o", loaded=True)

    seen = {}

    def base(request):
        """Record the brain."""
        seen["llm"] = request.llm
        return SimpleNamespace(content="ok", tool_calls=[], error=None,
                               is_error=False)

    handler = registry.wrap_model_call(session, runtime, base)
    handler(ModelRequest(llm=original,
                         messages=[{"role": "user", "content": "hi"}]))
    assert seen["llm"] is original
    service.unload()


def test_proceed_is_refused_outside_a_model_call_hook(tmp_path, box, runtime,
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
        "        response = sdk.model.proceed(request)\n"
        "        if not response.content.strip():",
        "        raise ValueError('boom')\n"
        "        if False:")
    service = _service(tmp_path, runtime, source=source)
    base, calls = _backend("still answered")

    handler = registry.wrap_model_call(session, runtime, base)
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
