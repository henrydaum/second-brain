"""The security mode: a standing answer to the dialog, and its exact limits.

Three modes, but almost every test here is about a *boundary* rather than the
happy path — because the happy path ("yolo allows, lockdown refuses") is one
line and the boundaries are where a permission feature goes wrong:

- yolo must not reach unattended work,
- lockdown must not be a trap,
- neither may survive the conversation it was set in,
- and neither may reach what the kernel refuses structurally.
"""

import pytest

import sandbox  # noqa: F401  - installs the ``guest`` package alias
from sandbox import Chain, Request
from sandbox.approval import build_approver
from sandbox.guest import requests as R
from sandbox.policy import CONSEQUENTIAL, classify

from runtime.security_modes import (
    ASK,
    DEFAULT_SECURITY_MODE,
    LOCKDOWN,
    SECURITY_MODES,
    TURN_SCOPE,
    YOLO,
    prompt_note,
    security_mode,
    standing_answer,
    tightens,
)

SHELL = Request(R.PROC_RUN, {"argv": ["git", "push"]})


# ──────────────────────────────────────────────────────────────────────
# Doubles.
# ──────────────────────────────────────────────────────────────────────

class FakeSession:
    """Only the fields the mode reader touches."""

    def __init__(self, conversation_id=7):
        self.conversation_id = conversation_id
        self.security_mode = None
        self.security_mode_conversation = None
        self.turn_security_mode = None


class FakeRequestInput:
    """Stands in for a pending dialog. Answering "allow" if ever reached."""

    def __init__(self):
        self.id = 1
        self.value = "allow"
        self.metadata = {}

    def wait(self, timeout=None):
        return True


class FakeRuntime:
    """A runtime carrying the real reader/writer semantics, not stubs.

    The mode logic under test lives in ``ConversationRuntime.security_mode``,
    so this reimplements it rather than importing — deliberately, because the
    property being pinned ("a mode does not apply to another conversation") is
    a *rule*, and a double that shared the implementation could not fail when
    the rule changed.
    """

    def __init__(self, *, attended=True):
        self.active_session_key = "repl"
        self.sessions = {"repl": FakeSession(), "spawn_subagent:9": FakeSession(99)}
        self.hooks = None
        self.asked = []
        self._attended = attended

    def is_attended(self, key):
        return self._attended and key == self.active_session_key

    def security_mode(self, key):
        session = self.sessions.get(key)
        if session is None:
            return DEFAULT_SECURITY_MODE
        if session.turn_security_mode:
            return security_mode(session.turn_security_mode)
        if session.security_mode is None:
            return DEFAULT_SECURITY_MODE
        if session.security_mode_conversation != session.conversation_id:
            return DEFAULT_SECURITY_MODE
        return security_mode(session.security_mode)

    def set_security_mode(self, key, mode, *, scope="conversation"):
        session = self.sessions.get(key)
        if session is None:
            return None
        resolved = security_mode(mode)
        if scope == TURN_SCOPE:
            session.turn_security_mode = resolved
        else:
            session.security_mode = resolved
            session.security_mode_conversation = session.conversation_id
        return resolved

    def request_input(self, key, title, prompt, **kwargs):
        self.asked.append({"key": key, "title": title})
        return FakeRequestInput()

    def answer_request(self, key, request_id, value):
        pass


def agent_chain(key="repl"):
    """What an agent's own tool call looks like: rooted at its session."""
    return Chain(root=key, links=("some_tool",))


# ──────────────────────────────────────────────────────────────────────
# The vocabulary.
# ──────────────────────────────────────────────────────────────────────

def test_ask_is_the_default_and_answers_nothing():
    """``ask`` is the absence of a standing answer, not a third verdict."""
    assert DEFAULT_SECURITY_MODE == ASK
    assert standing_answer(ASK) is None
    assert standing_answer(YOLO) is True
    assert standing_answer(LOCKDOWN) is False


@pytest.mark.parametrize("junk", ["", None, "nonsense", "  ", 0, [], "YOLO!!"])
def test_an_unreadable_mode_degrades_to_ask_never_to_yolo(junk):
    """The normalizer's failure direction is the whole of its safety."""
    assert security_mode(junk) == ASK
    assert standing_answer(junk) is None


def test_only_lockdown_tightens():
    """The polarity rule ``classify`` rests on."""
    assert tightens(LOCKDOWN) is True
    assert tightens(ASK) is False
    assert tightens(YOLO) is False


def test_the_default_mode_contributes_no_prompt_text():
    """A prompt that restates the default on every turn is tokens for nothing."""
    assert prompt_note(ASK) == ""
    assert "lockdown" in prompt_note(LOCKDOWN).lower()
    assert prompt_note(YOLO)


def test_lockdown_prompt_tells_the_agent_not_to_retry():
    """A refusal an agent cannot explain is a refusal it retries.

    The specific failure: an agent reads a denial as transient, tries again,
    is denied again, and burns the turn. It only stops if it is told the wall
    is permanent — so the text has to say so, not merely imply it.
    """
    note = prompt_note(LOCKDOWN).lower()
    assert "retry" in note or "again" in note
    assert "/mode" in note


# ──────────────────────────────────────────────────────────────────────
# The approver: where the mode is actually spent.
# ──────────────────────────────────────────────────────────────────────

def test_ask_still_raises_a_dialog():
    """The default must be indistinguishable from before this feature."""
    runtime = FakeRuntime()
    approve = build_approver(runtime)
    assert approve(agent_chain(), SHELL, classify(SHELL, agent_chain())) is True
    assert len(runtime.asked) == 1


def test_yolo_allows_without_asking_anybody():
    """Not merely allowed — the dialog is never drawn at all."""
    runtime = FakeRuntime()
    runtime.set_security_mode("repl", YOLO)
    approve = build_approver(runtime)
    assert approve(agent_chain(), SHELL, classify(SHELL, agent_chain())) is True
    assert runtime.asked == []


def test_lockdown_refuses_without_asking_anybody():
    runtime = FakeRuntime()
    runtime.set_security_mode("repl", LOCKDOWN)
    approve = build_approver(runtime)
    assert approve(agent_chain(), SHELL, classify(SHELL, agent_chain())) is False
    assert runtime.asked == []


def test_yolo_never_reaches_unattended_work():
    """The single most important property here.

    A subagent, a cron job and a service poll tick are all "a chain whose
    session nobody is watching". The mode is consulted *after* attendance, so
    a grant given for a foreground task cannot be spent by work the person
    cannot see — and this holds even when the unattended session has somehow
    been put into yolo itself, which is what makes it a property of the
    ordering rather than of where the mode happens to be set.
    """
    runtime = FakeRuntime()
    runtime.set_security_mode("repl", YOLO)
    runtime.set_security_mode("spawn_subagent:9", YOLO)
    approve = build_approver(runtime)
    chain = agent_chain("spawn_subagent:9")
    assert approve(chain, SHELL, classify(SHELL, chain)) is False
    assert runtime.asked == []


def test_a_subagent_starts_in_ask_whatever_the_parent_was():
    """The second belt: per-conversation means a child inherits nothing."""
    runtime = FakeRuntime()
    runtime.set_security_mode("repl", YOLO)
    assert runtime.security_mode("spawn_subagent:9") == ASK


def test_the_mode_does_not_survive_a_conversation_switch():
    """Per conversation, and structurally so.

    Nothing resets the field — the reader simply stops honouring it once the
    session is showing a different conversation. That is what removes the list
    of reset call sites (``/new``, ``/clear``, ``load_conversation``, the
    three paths that null the id) that would otherwise have to be kept in step.
    """
    runtime = FakeRuntime()
    runtime.set_security_mode("repl", YOLO)
    assert runtime.security_mode("repl") == YOLO

    runtime.sessions["repl"].conversation_id = 8      # /new, or a load
    assert runtime.security_mode("repl") == ASK

    approve = build_approver(runtime)
    assert approve(agent_chain(), SHELL, classify(SHELL, agent_chain())) is True
    assert len(runtime.asked) == 1, "a fresh conversation must ask again"


def test_a_missing_session_asks_rather_than_assuming():
    runtime = FakeRuntime()
    assert runtime.security_mode("nobody") == ASK
    assert runtime.set_security_mode("nobody", YOLO) is None


def test_a_runtime_with_no_reader_asks():
    """The feature is invisible against a runtime that predates it."""

    class Older(FakeRuntime):
        security_mode = None

    runtime = Older()
    approve = build_approver(runtime)
    assert approve(agent_chain(), SHELL, classify(SHELL, agent_chain())) is True
    assert len(runtime.asked) == 1


def test_lockdown_does_not_countermand_a_plugin_that_allowed():
    """Lockdown answers the dialog; it does not overrule an earlier verdict.

    A ``vet_permission`` gate runs before the mode, so a plugin with a positive
    opinion still wins. That is the difference between "stop asking me, the
    answer is no" and "break the plugins I already configured", and only the
    first is a promise this can keep.
    """

    class Verdict:
        allow = True
        reason = ""

    class Gate:
        def vet_permission(self, *a, **k):
            return Verdict()

    runtime = FakeRuntime()
    runtime.hooks = Gate()
    runtime.set_security_mode("repl", LOCKDOWN)
    approve = build_approver(runtime)
    assert approve(agent_chain(), SHELL, classify(SHELL, agent_chain())) is True


# ──────────────────────────────────────────────────────────────────────
# Turn scope — the substrate a plan hands the turn that follows it.
# ──────────────────────────────────────────────────────────────────────

def test_a_turn_scoped_mode_outranks_the_conversation():
    runtime = FakeRuntime()
    runtime.set_security_mode("repl", LOCKDOWN)
    runtime.set_security_mode("repl", YOLO, scope=TURN_SCOPE)
    assert runtime.security_mode("repl") == YOLO


def test_the_turn_scoped_mode_is_dropped_when_the_turn_ends():
    """Cleared by the kernel at ``finish_turn``, not by a registered hook.

    A grant that expires only when some plugin happens to be installed is not
    a grant that expires — the same argument the compaction layer and the
    subagent barrier make for being stacked rather than registered.
    """
    from runtime.hooks import HookRegistry, TurnOutcome

    runtime = FakeRuntime()
    session = runtime.sessions["repl"]
    runtime.set_security_mode("repl", YOLO, scope=TURN_SCOPE)

    HookRegistry().finish_turn(session, TurnOutcome(ok=True))

    assert session.turn_security_mode is None
    assert runtime.security_mode("repl") == ASK


# ──────────────────────────────────────────────────────────────────────
# Who may change it.
# ──────────────────────────────────────────────────────────────────────

def test_lockdown_is_not_a_trap():
    """The one that would have shipped broken.

    The mode is enforced at the approver, so the act that *leaves* it must
    never reach the approver — otherwise ``/mode ask`` is auto-refused by the
    very thing it exists to lift, and restarting the app is the only way out.
    ``chain.typed_command`` is what prevents it.
    """
    typed = Chain(root="user:command", links=("mode",))
    for mode in SECURITY_MODES:
        request = Request(R.SESSION_SET_MODE, {"mode": mode})
        assert classify(request, typed).safe, mode


def test_an_agent_may_tighten_but_never_loosen():
    """Polarity: arriving at lockdown widens nothing, whatever we were in."""
    chain = agent_chain()
    assert classify(Request(R.SESSION_SET_MODE, {"mode": LOCKDOWN}), chain).safe
    for mode in (ASK, YOLO):
        assert not classify(Request(R.SESSION_SET_MODE, {"mode": mode}), chain).safe


def test_a_command_cannot_lend_its_standing_to_what_it_calls():
    """Scoped to the command's own code, like ``config.write`` beside it."""
    delegated = Chain(root="user:command", links=("mode", "some_helper"))
    request = Request(R.SESSION_SET_MODE, {"mode": YOLO})
    assert not classify(request, delegated).safe


def test_an_unreadable_mode_is_asked_about_rather_than_waved_through():
    """The branch's own failure direction, matching the normalizer's."""
    chain = agent_chain()
    assert not classify(Request(R.SESSION_SET_MODE, {"mode": "nonsense"}), chain).safe
    assert not classify(Request(R.SESSION_SET_MODE, {}), chain).safe


def test_setting_the_mode_is_consequential_so_a_command_must_gate_it():
    """What makes ``/mode``'s own declaration enforced rather than remembered."""
    assert R.SESSION_SET_MODE in CONSEQUENTIAL


def test_the_dialog_names_the_mode_and_what_it_means():
    """The word alone is the one thing a person cannot check.

    "yolo" tells you it is permissive and nothing about what stops being
    asked, so the blurb travels with it.
    """
    from sandbox.approval import describe

    chain = agent_chain()
    request = Request(R.SESSION_SET_MODE, {"mode": YOLO})
    title, body = describe(chain, request, classify(request, chain))
    assert "yolo" in body.lower()
    assert "without asking" in body.lower()


# ──────────────────────────────────────────────────────────────────────
# The layer-6 half: a command grant the person answered in advance.
# ──────────────────────────────────────────────────────────────────────

def test_the_state_machine_asks_by_default():
    """``auto_approve`` defaults to asking, so a bare state machine is unchanged."""
    from state_machine.conversation import ConversationState, Participant

    cs = ConversationState([Participant("user", "user"),
                            Participant("agent", "agent")])
    assert cs.auto_approve() is False


def test_a_raising_auto_approve_asks():
    """The safe direction at layer 6 is the dialog."""
    from state_machine.action import _CallableAction

    class Boom:
        def auto_approve(self):
            raise RuntimeError("nope")

    action = _CallableAction.__new__(_CallableAction)
    action.cs = Boom()
    assert action._pre_approved() is False


# ──────────────────────────────────────────────────────────────────────
# The real runtime. Everything above drives a double that reimplements the
# reader — deliberately, so a rule change can fail the rule tests — which
# leaves exactly one thing unpinned: that the real implementation agrees.
# ──────────────────────────────────────────────────────────────────────

def test_the_real_runtime_reader_agrees_with_the_rules(tmp_path):
    from tests.support import make_runtime

    runtime, session, _ = make_runtime(tmp_path)
    key = session.key

    assert runtime.security_mode(key) == ASK

    assert runtime.set_security_mode(key, YOLO) == YOLO
    assert runtime.security_mode(key) == YOLO

    # Turn scope outranks it, and clearing falls back rather than to ask.
    runtime.set_security_mode(key, LOCKDOWN, scope=TURN_SCOPE)
    assert runtime.security_mode(key) == LOCKDOWN
    runtime.clear_turn_security_mode(key)
    assert runtime.security_mode(key) == YOLO

    # And the conversation rule, against a real conversation_id.
    session.conversation_id = (session.conversation_id or 0) + 1
    assert runtime.security_mode(key) == ASK

    assert runtime.security_mode("no-such-session") == ASK
    assert runtime.set_security_mode("no-such-session", YOLO) is None


def test_a_mode_set_before_the_conversation_exists_survives_its_creation(tmp_path):
    """A session exists from the frontend's first breath; its conversation is
    created by the first message. So ``/mode lockdown`` at a fresh prompt
    stamped ``None``, and plain equality then dropped the mode the instant the
    user said anything — silently, and in the permissive direction.

    The rule the check was always trying to express is "is this still the same
    piece of work", and the conversation a session opens right after the mode
    was set is that same work.
    """
    from tests.support import plain_runtime
    from pipeline.database import Database
    from runtime.persistence import get_or_create_session

    db = Database(str(tmp_path / "late.db"))
    runtime = plain_runtime(db)
    session = get_or_create_session(runtime, "repl")
    assert session.conversation_id is None

    runtime.set_security_mode("repl", LOCKDOWN)
    assert runtime.security_mode("repl") == LOCKDOWN

    session.conversation_id = db.create_conversation("first message")
    assert runtime.security_mode("repl") == LOCKDOWN, (
        "the conversation the session just opened is the work the mode was "
        "set for")


def test_binding_late_does_not_reopen_the_leak_it_was_checking_for(tmp_path):
    """The whole point of the check: switching *away* still resets."""
    from tests.support import plain_runtime
    from pipeline.database import Database
    from runtime.persistence import get_or_create_session

    db = Database(str(tmp_path / "late2.db"))
    runtime = plain_runtime(db)
    session = get_or_create_session(runtime, "repl")

    runtime.set_security_mode("repl", YOLO)          # unbound
    session.conversation_id = db.create_conversation("one")
    assert runtime.security_mode("repl") == YOLO     # adopts it

    session.conversation_id = db.create_conversation("two")
    assert runtime.security_mode("repl") == ASK, (
        "an adopted stamp is an ordinary stamp - a second conversation is a "
        "different piece of work")


def test_the_real_runtime_normalizes_junk_to_ask(tmp_path):
    from tests.support import make_runtime

    runtime, session, _ = make_runtime(tmp_path)
    assert runtime.set_security_mode(session.key, "nonsense") == ASK
    assert runtime.security_mode(session.key) == ASK


def test_the_mode_is_not_persisted_in_the_marker(tmp_path):
    """Ephemeral means a restart returns to ask.

    A forgotten ``yolo`` surviving a restart is the one failure this feature
    must not have, and ``to_marker`` is the exact line that would cause it.
    """
    from tests.support import make_runtime

    runtime, session, _ = make_runtime(tmp_path)
    runtime.set_security_mode(session.key, YOLO)
    runtime.set_security_mode(session.key, LOCKDOWN, scope=TURN_SCOPE)

    marker = session.to_marker()
    for field in ("security_mode", "security_mode_conversation",
                  "turn_security_mode"):
        assert field not in marker, f"{field} must not survive a restart"


def test_a_live_session_binds_auto_approve_to_its_mode(tmp_path):
    """The layer-6 half, end to end through the real ``new_state``."""
    from tests.support import make_runtime

    runtime, session, _ = make_runtime(tmp_path)
    assert session.cs.auto_approve() is False

    runtime.set_security_mode(session.key, YOLO)
    assert session.cs.auto_approve() is True, (
        "the closure must read the live session, not a value captured when "
        "the state machine was built")

    runtime.set_security_mode(session.key, LOCKDOWN)
    assert session.cs.auto_approve() is False, (
        "lockdown must not reach layer 6 - the dialog it would refuse is "
        "about a command the person just typed")


def test_the_agent_is_told_the_mode_through_the_dynamic_prompt(tmp_path):
    """Not via ``agent_prompt`` (a plugin's, and the mode is the kernel's) and
    not via ``system_prompt_extras`` (persisted) — the mode changes within a
    conversation and resets structurally, and a store package is the wrong
    owner for a safety surface that must not stop working when it is
    uninstalled."""
    from runtime.runtime_config import session_system_prompt
    from tests.support import make_runtime

    runtime, session, _ = make_runtime(tmp_path)

    def rendered():
        prompt = session_system_prompt(runtime, session)
        sections = prompt() if callable(prompt) else prompt
        if isinstance(sections, list):
            return "\n".join(m.get("content") or "" for m in sections)
        return sections or ""

    assert "lockdown" not in rendered().lower()
    runtime.set_security_mode(session.key, LOCKDOWN)
    assert "lockdown" in rendered().lower()


# ──────────────────────────────────────────────────────────────────────
# The command, driven through its own helpers and a stub SDK.
# ──────────────────────────────────────────────────────────────────────

class _Markdown:
    @staticmethod
    def table(headers, rows, **kwargs):
        return "\n" + "\n".join(" | ".join(str(c) for c in row) for row in rows)


class _SessionNS:
    def __init__(self, mode):
        self.mode = mode
        self.writes = []

    def get(self, key="", details=False):
        return {"key": "repl", "mode": self.mode}

    def set_mode(self, mode, key="", scope="conversation"):
        self.writes.append((mode, scope))
        self.mode = mode
        return mode


class _SDK:
    Failed = RuntimeError

    def __init__(self, mode=ASK):
        self.session = _SessionNS(mode)
        self.md = _Markdown()


@pytest.fixture
def command():
    import importlib.util
    from pathlib import Path

    path = (Path(__file__).resolve().parents[1] / "bundled" / "commands"
            / "command_mode.py")
    spec = importlib.util.spec_from_file_location("_mode_command", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_command_gates_only_the_loosening_direction(command):
    """Tightening and lifting must not be gated; handing away a decision must.

    Gating ``lockdown`` would be friction with no decision in it. Gating
    ``ask`` would gate the exit from lockdown, which is the trap again one
    layer up.
    """
    cls = command.ModeCommand
    assert cls.approval_actions == ("yolo",)
    assert isinstance(cls.approval_actions, tuple)
    assert all(isinstance(a, str) for a in cls.approval_actions), (
        "must be literal strings - declarations are read by AST, and a module "
        "constant here reads as no gate at all")


def test_the_landing_view_states_what_no_mode_changes(command):
    """A view that shows only the grant leaves 'what can it reach' wrong."""
    sdk = _SDK(YOLO)
    out = command.ModeCommand().run(sdk, {})
    assert "yolo" in out.lower()
    assert "not root" in out.lower()
    assert "unattended" in out.lower()


def test_switching_writes_the_mode_and_says_what_changed(command):
    sdk = _SDK(ASK)
    out = command.ModeCommand().run(sdk, {"action": YOLO})
    assert sdk.session.writes == [(YOLO, "conversation")]
    assert "ask" in out.lower() and "yolo" in out.lower()


def test_switching_to_yolo_restates_the_limits(command):
    """The permissive answer is the one that has to say what it does not cover."""
    out = command.ModeCommand().run(_SDK(ASK), {"action": YOLO})
    assert "not root" in out.lower()


def test_switching_to_lockdown_says_how_to_leave(command):
    """Otherwise the person has to guess, about the mode that refuses things."""
    out = command.ModeCommand().run(_SDK(ASK), {"action": LOCKDOWN})
    assert "/mode ask" in out


def test_an_unknown_mode_is_named_rather_than_written(command):
    sdk = _SDK(ASK)
    out = command.ModeCommand().run(sdk, {"action": "paranoid"})
    assert sdk.session.writes == []
    assert "paranoid" in out


def test_switching_to_the_mode_already_in_force_writes_nothing(command):
    sdk = _SDK(YOLO)
    out = command.ModeCommand().run(sdk, {"action": YOLO})
    assert sdk.session.writes == []
    assert "already" in out.lower()


def test_the_form_is_skipped_when_the_mode_was_named_on_the_line(command):
    """``/mode yolo`` must not then ask which mode."""
    assert command.ModeCommand().form(_SDK(), {"action": YOLO}) == []
    assert command.ModeCommand().form(_SDK(), {}) != []


def test_the_form_step_is_called_action(command):
    """``approval_actions`` is matched against ``args["action"]`` and nothing
    else, so a differently-named step silently disables the gate."""
    step = command.ModeCommand().form(_SDK(), {})[0]
    assert step["name"] == "action"
    assert list(step["enum"]) == list(SECURITY_MODES)


def test_the_step_is_required_so_the_buttons_actually_render(command):
    """The regression that shipped: ``required=False`` means no UI at all.

    ``_missing`` keeps a step only when ``required or prompt_when_missing``, so
    an optional one is never missing, the form never suspends, and ``run`` is
    reached with no action — which prints the text fallback and no buttons.
    Nothing raises and the output looks plausible, which is why this needs a
    test rather than a reviewer.
    """
    from sandbox.bridge import _form_step
    from state_machine.action import _missing
    from state_machine.conversation import CallableSpec

    step = command.ModeCommand().form(_SDK(), {})[0]
    assert step["required"] or step["prompt_when_missing"]

    # Through the real predicate, and through the real guest->kernel step
    # conversion the bridge does on the way — a bare /mode must suspend.
    sdk = _SDK()

    def steps(args, cs=None):
        return [_form_step(s) for s in command.ModeCommand().form(sdk, args)]

    spec = CallableSpec("mode", lambda *a, **k: None, form_factory=steps)
    assert [s.name for s in _missing(spec, {})] == ["action"]
    assert _missing(spec, {"action": YOLO}) == []


def test_every_mode_is_offered_and_the_current_one_is_marked(command):
    """Three fixed buttons in a fixed order, so a click lands in one place."""
    for current in SECURITY_MODES:
        step = command.ModeCommand().form(_SDK(current), {})[0]
        assert list(step["enum"]) == list(SECURITY_MODES)
        marked = [label for label in step["enum_labels"] if "(current)" in label]
        assert len(marked) == 1
        assert marked[0].lower().startswith(current[:4])


def test_the_picker_does_not_repeat_what_its_buttons_say(command):
    """The dialog's old 'Run shell commands / Run shell commands' bug.

    Each label carries its mode's blurb, so a prompt printing them again is
    one screen saying everything twice.
    """
    step = command.ModeCommand().form(_SDK(), {})[0]
    prompt = step["prompt"].lower()
    for blurb in command.BLURBS.values():
        assert blurb.lower() not in prompt
    assert "ask" in prompt, "the prompt must still say where we are"


def test_the_command_offers_exactly_the_kernel_modes(command):
    """Two lists of three that must not drift apart."""
    assert tuple(command.MODES) == SECURITY_MODES


# ──────────────────────────────────────────────────────────────────────
# ``/new`` says the mode it just reset.
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def new_command():
    import importlib.util
    from pathlib import Path

    path = (Path(__file__).resolve().parents[1] / "bundled" / "commands"
            / "command_new.py")
    spec = importlib.util.spec_from_file_location("_new_command", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_new_says_the_mode_the_new_conversation_is_in(new_command):
    """The silent reset that cost an afternoon.

    A mode belongs to the conversation it was set in, so ``/new`` returns you
    to ``ask`` — and the person who typed ``/mode lockdown`` has no reason to
    re-check. The next thing they see is an approval dialog for something they
    believed would be refused outright. Stating the mode is what breaks that.

    It used to also name what it *reset from* (``_mode_line``, dropped in
    393e3be); the line states the mode plainly now, the same way the restore
    message does. Pinned as the mode being *read from the session* rather than
    assumed, which is the half that has to keep working.
    """
    from types import SimpleNamespace

    class _Sdk:
        Failed = RuntimeError
        session = SimpleNamespace(get=lambda: {"mode": LOCKDOWN})

    assert new_command._mode(_Sdk()) == LOCKDOWN


def test_new_falls_back_to_the_default_when_the_mode_cannot_be_read(new_command):
    """A command that cannot ask must not claim a mode nobody set."""
    from types import SimpleNamespace

    class _Sdk:
        Failed = RuntimeError
        session = SimpleNamespace(get=lambda: None)

    assert new_command._mode(_Sdk()) == ASK


def test_restoring_a_conversation_says_the_permission_mode(tmp_path):
    """A restart is the other place the mode silently goes back to the default.

    The conversation is restored, the agent profile is restored, and the mode
    is not — it is ephemeral on purpose, so a forgotten ``yolo`` cannot outlive
    the process. Which makes the restore message the one moment that can say
    so to somebody who set lockdown before quitting.
    """
    from pipeline.database import Database
    from tests.support import plain_runtime

    db = Database(str(tmp_path / "restore.db"))
    cid = db.create_conversation("New conversation (Main)")

    before = plain_runtime(db)
    before.load_conversation("repl", cid)
    before.active_session_key = "repl"
    before.set_security_mode("repl", LOCKDOWN)
    assert before.security_mode("repl") == LOCKDOWN
    before._remember_last_active(before.session_user_id("repl"), cid)

    # A brand new runtime over the same database: the restart.
    after = plain_runtime(db)
    message = after.restore_last_active("repl")

    assert f"Permission mode: {ASK}" in message
    assert after.security_mode("repl") == ASK, "a mode must not survive a restart"
    # Reported, not assumed: the line has to follow the load.
    assert message.index("Permission mode") > message.index("Agent:")
