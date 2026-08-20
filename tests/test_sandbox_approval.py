"""Approval: the moment the sandbox becomes visible to a person.

The order under test is the whole design — policy hooks, then a dialog whose
options can keep the answer, then a refusal when nobody is home. Each step
reuses machinery the kernel already has rather than duplicating it.
"""

import pytest

import sandbox  # noqa: F401  - installs the ``guest`` package alias
from sandbox import Chain, Interpreter, Request, Sandbox
from sandbox.approval import build_approver, describe
from sandbox.guest import requests as R
from sandbox.policy import classify
from sandbox.guest.codes import ERROR_NOT_PERMITTED
from sandbox.guest.requests import SECRET_REVEAL


class FakeRequest:
    """Stands in for the kernel's pending-input object.

    ``answer`` is a bool for readability at the call sites, but what the
    approver actually reads is ``value`` — the *option value* a person chose.
    The real object's ``approved`` is ``bool(self.value)``, so it is True for
    the string ``"deny"``; carrying both here is what lets a test say that out
    loud rather than passing by accident.
    """

    def __init__(self, answer=True, cancelled=False, answers=True, value=None):
        self.id = 1
        self.value = value if value is not None else (
            "allow" if answer else "deny")
        self.metadata = {"cancelled": cancelled}
        self._answers = answers

    @property
    def approved(self):
        """Mirrors the real request: truthiness of the answer, nothing more."""
        return bool(self.value)

    def wait(self, timeout=None):
        """Whether the user got back to us."""
        return self._answers


class FakeRuntime:
    """Just enough runtime to render a dialog."""

    def __init__(self, *, answer=True, attended=True,
                 hooks=None, answers=True, cancelled=False):
        self.active_session_key = "repl"
        self.sessions = {"repl": object()}
        self.hooks = hooks
        self.asked = []
        self._answer = answer
        self._attended = attended
        self._answers = answers
        self._cancelled = cancelled
        self.answered = []
        #: Set to answer with a specific option value rather than allow/deny.
        self._answer_value = None

    def is_attended(self, key):
        """Whether a human is present at ``key``, keyed like the real one.

        ``ConversationRuntime.is_attended`` is the single-active-session rule,
        so it is a question *about a session* and not a global flag. A fake
        that answered the same for every key would call a subagent's session
        attended because the foreground one is — which is precisely the
        confusion these tests exist to pin.
        """
        return self._attended and key == self.active_session_key

    def request_input(self, key, title, prompt, **kwargs):
        """Render the dialog."""
        self.asked.append({"key": key, "title": title, "prompt": prompt,
                           "kwargs": kwargs})
        return FakeRequest(self._answer, self._cancelled, self._answers,
                           value=self._answer_value)

    def answer_request(self, key, request_id, value):
        """Record a forced answer."""
        self.answered.append(value)


class Gate:
    """A stand-in for the hook *registry* with one fixed opinion.

    Note this mimics ``HookRegistry.vet_permission``, not a hook. Hooks
    themselves still receive ``(ctx, query)`` — that contract did not change,
    which is why gates in the wild are unaffected.
    """

    def __init__(self, verdict):
        self.verdict = verdict
        self.queries = []

    def vet_permission(self, session, tool_name, command, runtime=None,
                       stage="approval", **extra):
        """Answer, recording what was asked."""
        self.queries.append({"tool": tool_name, "command": command,
                             "stage": stage, **extra})
        return self.verdict


class Verdict:
    """A PermissionVerdict stand-in."""

    def __init__(self, allow, reason=""):
        self.allow = allow
        self.reason = reason


def _egress(url="https://example.invalid/collect"):
    """An unsafe Request and its decision."""
    request = Request(R.NET_HTTP, {"url": url, "method": "POST"})
    return request, classify(request, Chain())


# ──────────────────────────────────────────────────────────────────────
# The dialog is built from the chain.
# ──────────────────────────────────────────────────────────────────────

def test_the_dialog_names_the_effect_in_plain_terms():
    """'net.http' is jargon; 'send a POST to x' is a question."""
    request, decision = _egress()
    title, body = describe(Chain(root="user").push("summarize"), request,
                           decision)
    assert "POST" in body
    assert "example.invalid" in body
    # The title names the effect. It used to be one constant string shared by
    # every askable type, which both frontends printed above a body that then
    # said the same thing better.
    assert "network" in title.lower()


def test_the_dialog_names_the_plugin_that_asked():
    """The reason provenance exists, shown in the only place it matters."""
    request, decision = _egress()
    chain = Chain(root="user").push("task_index").push("fetch")
    _, body = describe(chain, request, decision)
    assert "Asked by fetch" in body


def test_only_attended_roots_reach_a_dialog_so_no_root_is_worth_naming():
    """Why ``describe_asker`` is the leaf and stops.

    The origin clauses this once carried - "running on the nightly_index
    schedule", "running in a background agent" - described work that is
    *refused* rather than asked, so a dialog could never render them. What
    can reach one is an attended chain, and every attended root either means
    "you did this" or names the session the dialog is being delivered to.

    Pinned by driving the approver, because the fact is about the order of
    its steps rather than about the renderer: change step 3 and this line is
    what says the renderer has to change with it.
    """
    request, decision = _egress()
    rendered = []

    class Runtime:
        active_session_key = "telegram:1:1:0"
        sessions = {"telegram:1:1:0": object()}
        hooks = None

        def is_attended(self, key):
            return key == self.active_session_key

        def request_input(self, key, title, prompt, **kwargs):
            rendered.append((key, prompt))
            return FakeRequest(answer=False)

    for root in ("cron:nightly_index", "spawn_subagent:47", "service:drive",
                 "agent", "kernel"):
        approve = build_approver(Runtime())
        assert approve(Chain(root=root).push("leaf"), request,
                       decision) is False
    assert rendered == [], "an unattended chain reached a dialog"

    # And the attended ones do render, with the leaf and no root.
    for root in ("user", "user:command", "telegram:1:1:0"):
        build_approver(Runtime())(
            Chain(root=root).push("leaf"), request, decision)
    assert len(rendered) == 3
    for _, prompt in rendered:
        assert "Asked by leaf" in prompt
        assert "telegram" not in prompt and "user" not in prompt


def test_a_person_acting_for_themselves_is_not_told_so():
    """A bare ``user`` root with no links has nothing to attribute, and a
    line saying so is one more line between them and the decision."""
    request, decision = _egress()
    _, body = describe(Chain(root="user"), request, decision)
    assert "Asked by" not in body


def test_the_title_is_not_repeated_by_the_body():
    """Both shipped frontends print the title above the body.

    While the title was one constant string that cost nothing; the moment it
    became the effect's own phrase, a body that opened with the action line
    said it twice - "Run shell commands" over "Run shell commands: `echo x`".
    So the title carries the phrase and the body carries only the arguments.
    """
    request = Request(R.PROC_RUN, {"argv": ["echo", "hi"]})
    title, body = describe(Chain(), request, classify(request, Chain()))

    assert title == "Run shell commands"
    assert title not in body
    assert "echo hi" in body


def test_a_request_with_no_arguments_to_show_has_a_body_or_no_dialog():
    """Every askable type still renders something a person can answer.

    The body is allowed to be empty of *detail* - some Requests have no
    argument worth printing - but then the title has to be carrying the whole
    question on its own, so it must never be the bare dotted type.
    """
    from sandbox.approval import phrase_for

    for kind in sorted(R.ALL_TYPES):
        title, _ = describe(Chain(), Request(kind, {}),
                            classify(Request(kind, {}), Chain()))
        assert title
        assert phrase_for(kind) != kind, f"{kind} has no phrase"


def test_the_dialog_does_not_say_the_same_thing_twice():
    """``decision.reason`` is written for the ledger and the model; printing
    it here restated the action line the dialog had just rendered."""
    request = Request(R.PROC_RUN, {"argv": ["git", "pull"], "cwd": "/tmp"})
    decision = classify(request, Chain())
    _, body = describe(Chain(), request, decision)
    assert body.count("git pull") == 1
    assert decision.reason not in body


def test_shell_commands_are_shown_as_code():
    """A command the user has to judge must be legible, not prose-wrapped."""
    request = Request(R.PROC_RUN, {"argv": ["rm", "-rf", "build"]})
    _, body = describe(Chain(), request, classify(request, Chain()))
    assert "```" in body
    assert "rm -rf build" in body


def test_scheduling_shows_the_schedule_and_the_instructions():
    """The two things being authorised, neither of which was ever shown.

    ``agent.schedule`` fell through to the generic field scan, which reaches
    for ``name`` — empty by default — so the dialog read "Schedule unattended
    work" and stopped. A person was being asked to approve work they could
    neither time nor read.
    """
    request = Request(R.AGENT_SCHEDULE, {
        "prompt": "Summarise my unread email.",
        "cron": "0 9 * * 1-5", "title": "Weekday digest"})
    _, body = describe(Chain(root="user"), request, classify(request, Chain()))

    assert "At 09:00, Monday through Friday" in body
    assert "0 9 * * 1-5" in body
    assert "Summarise my unread email." in body
    assert "Weekday digest" in body
    # cp1252: the REPL console cannot print what the library's own locale
    # would have produced on a non-English machine.
    body.encode("ascii")


def test_a_scheduled_job_shows_its_schedule_too():
    """``cron.create`` is the same question one layer down, and showed only
    the job name — never when it runs or what it will do."""
    request = Request(R.CRON_CREATE, {
        "name": "nightly", "job": {"cron": "0 3 * * *", "channel": "x",
                                   "payload": {"prompt": "Reindex."}}})
    _, body = describe(Chain(root="user"), request, classify(request, Chain()))

    assert "nightly" in body and "At 03:00" in body and "Reindex." in body


def test_an_unreadable_cron_is_shown_once_not_twice():
    """The description falls back to the expression, which must not then be
    printed beside itself as though it were a translation."""
    request = Request(R.AGENT_SCHEDULE, {"prompt": "x", "cron": "nonsense"})
    _, body = describe(Chain(root="user"), request, classify(request, Chain()))

    assert body.count("nonsense") == 1


# ──────────────────────────────────────────────────────────────────────
# The order of consultation.
# ──────────────────────────────────────────────────────────────────────

def test_a_policy_hook_wins_over_everything():
    """Plan mode refuses at this doorway; it must not be overridden."""
    gate = Gate(Verdict(False, "plan mode"))
    runtime = FakeRuntime(answer=True, hooks=gate)
    approve = build_approver(runtime)
    request, decision = _egress()

    assert approve(Chain().push("t"), request, decision) is False
    assert runtime.asked == []          # never bothered the user


def test_a_policy_hook_can_allow_without_asking():
    """A trust plugin standing at the same doorway."""
    runtime = FakeRuntime(answer=False, hooks=Gate(Verdict(True)))
    request, decision = _egress()
    assert build_approver(runtime)(Chain().push("t"), request, decision) is True
    assert runtime.asked == []


def test_an_abstaining_hook_falls_through_to_the_dialog():
    """Abstain means 'no opinion', not 'no'."""
    runtime = FakeRuntime(answer=True, hooks=Gate(None))
    request, decision = _egress()
    assert build_approver(runtime)(Chain().push("t"), request, decision) is True
    assert len(runtime.asked) == 1


def test_the_hook_is_told_whether_anyone_is_present():
    """The kernel's two stages, chosen by attendance."""
    for attended, stage in ((True, "approval"), (False, "unattended_call")):
        gate = Gate(None)
        runtime = FakeRuntime(attended=attended, hooks=gate)
        request, decision = _egress()
        build_approver(runtime)(Chain().push("t"), request, decision)
        assert gate.queries[0]["stage"] == stage


# ──────────────────────────────────────────────────────────────────────
# The dialog itself.
# ──────────────────────────────────────────────────────────────────────

def test_approval_lets_the_request_through():
    """Yes means yes."""
    runtime = FakeRuntime(answer=True)
    request, decision = _egress()
    assert build_approver(runtime)(Chain().push("t"), request, decision) is True


def test_refusal_stops_it():
    """No means no."""
    runtime = FakeRuntime(answer=False)
    request, decision = _egress()
    assert build_approver(runtime)(Chain().push("t"), request,
                                   decision) is False


def test_a_cancelled_dialog_is_a_refusal():
    """Walking away is not consent."""
    runtime = FakeRuntime(answer=True, cancelled=True)
    request, decision = _egress()
    assert build_approver(runtime)(Chain().push("t"), request,
                                   decision) is False


def test_a_timed_out_dialog_is_a_refusal_and_is_answered_by_name():
    """The pending request must not be left hanging in the session.

    And it must be denied by the *option value*, not by ``False``. The dialog
    is a string request, so ``False`` fails ``match_enum``; a failed coercion
    never reaches ``pop_phase``, leaving the session parked in
    ``approving_request`` where only answering or cancelling is legal — every
    ordinary keystroke coming back ``invalid_action``, forever, about a dialog
    that expired.
    """
    runtime = FakeRuntime(answer=True, answers=False)
    request, decision = _egress()
    assert build_approver(runtime)(Chain().push("t"), request,
                                   decision) is False
    assert runtime.answered == ["deny"]


def test_nobody_present_means_refused_not_blocked():
    """An unattended session must never wait on a dialog nobody sees."""
    runtime = FakeRuntime(attended=False)
    request, decision = _egress()
    assert build_approver(runtime)(Chain(root="cron:x").push("t"), request,
                                   decision) is False
    assert runtime.asked == []


def test_a_background_chain_is_unattended_even_where_somebody_is_sitting():
    """The chain is a floor, and this is the case that made it one.

    ``build_approver`` is wired with no session key, so the key fell back to
    ``active_session_key`` — which ``is_attended`` calls attended by
    definition. An unsafe Request from a scheduled subagent therefore raised a
    dialog on the *foreground* session, parking it in ``approving_request``
    where ordinary input comes back ``invalid_action``, once per firing,
    about work the person could not see and never started.

    ``sandbox/policy.py`` rests the safety of ``agent.spawn`` on exactly this:
    a subagent is safe because it can approve nothing.
    """
    # Attended in the runtime's sense — somebody really is at the REPL.
    runtime = FakeRuntime(attended=True)
    request, decision = _egress()

    allowed = build_approver(runtime)(
        Chain(root="spawn_subagent:41").push("tool"), request, decision)

    assert allowed is False
    assert runtime.asked == [], "interrupted a session that asked for nothing"


def test_a_user_chain_still_defers_to_the_frontend_on_attendance():
    """The floor tightens; it must not take over.

    A frontend owning its own attendance policy (a socket that disconnected,
    say) still gets to say nobody is there for work the *user* started.
    """
    runtime = FakeRuntime(attended=False)
    request, decision = _egress()

    assert build_approver(runtime)(Chain(root="user").push("tool"), request,
                                   decision) is False
    assert runtime.asked == []


def test_the_agents_own_tool_call_is_attended_and_a_subagents_is_not():
    """The pair is the invariant; either alone is a bug that has shipped.

    ``bridge._root_for`` roots an agent-caused call at the *session key*, so
    both of these chains read False for ``Chain.attended`` — the property is
    True only for a root of ``user``. Judging attendance on that alone refused
    every unsafe Request any tool ever made, with the person watching the turn
    it happened in. Judging it on the runtime alone put a subagent's dialog on
    the foreground session. The root is what tells them apart, and it does so
    without any rule about plugin families.
    """
    runtime = FakeRuntime(attended=True)          # somebody is at the REPL
    request, decision = _egress()
    approve = build_approver(runtime)

    foreground = Chain(root="repl").push("tool_web_search").push("service_web")
    background = Chain(root="spawn_subagent:41").push("tool_web_search")
    assert not foreground.attended and not background.attended

    assert approve(foreground, request, decision) is True
    assert approve(background, request, decision) is False
    assert len(runtime.asked) == 1, "asked about the wrong one, or twice"
    assert runtime.asked[0]["key"] == "repl"


def test_an_unattended_session_refuses_its_own_agents_tool_call():
    """The session the chain names is the one whose attendance decides.

    A background driver holding a session nobody is looking at roots its tool
    calls there, and gets the same answer a subagent does. Nothing here keys
    off the ``spawn_subagent:`` prefix — a session with no one at it is enough.
    """
    runtime = FakeRuntime(attended=True)
    request, decision = _egress()

    allowed = build_approver(runtime)(
        Chain(root="nightly_sweep").push("tool_web_search"), request, decision)

    assert allowed is False
    assert runtime.asked == []


def test_asking_a_question_is_safe_from_the_agents_own_tool_call():
    """``ui.ask`` read the bare property too, and lost interactive tools.

    The classifier answered "nobody is present to answer this question" for a
    tool the agent called mid-turn — so the one Request whose entire purpose
    is reaching the user was refused for the user's absence.
    """
    import runtime as runtime_pkg
    from sandbox import policy
    from sandbox.guest.requests import Request

    ask = Request(type="ui.ask", args={"prompt": "which one?"})
    fake = FakeRuntime(attended=True)
    # ``attended_now`` reaches the composition root the same way the egress
    # allowlist does, so this patches what it reads rather than what it is.
    original = runtime_pkg.context.kernel_runtime
    try:
        runtime_pkg.context.kernel_runtime = lambda: fake
        foreground = policy.classify(ask, Chain(root="repl").push("tool_ask"))
        background = policy.classify(
            ask, Chain(root="spawn_subagent:41").push("tool_ask"))
    finally:
        runtime_pkg.context.kernel_runtime = original

    assert foreground.level == policy.SAFE
    assert background.level == policy.UNSAFE


def test_answering_deny_refuses_even_though_approved_reads_true():
    """The trap this dialog's rewrite walks straight into.

    ``StateMachineApprovalRequest.approved`` is ``bool(self.value)``, and the
    deny *option value* is the non-empty string "deny" — so the old
    ``return bool(pending.approved)`` would have turned every refusal into an
    approval, silently, for every unsafe Request in the system. The approver
    reads ``.value``; this asserts both halves so the reason is visible.
    """
    runtime = FakeRuntime(answer=False)
    request, decision = _egress()

    pending_shape = FakeRequest(answer=False)
    assert pending_shape.value == "deny"
    assert pending_shape.approved is True, "the trap is still live"

    assert build_approver(runtime)(Chain().push("t"), request,
                                   decision) is False


def test_the_dialog_is_asked_as_a_string_with_matched_options():
    """Boolean would short-circuit ``_coerce`` before the enum is consulted."""
    runtime = FakeRuntime()
    request, decision = _egress()
    build_approver(runtime)(Chain().push("t"), request, decision)

    asked = runtime.asked[0]["kwargs"]
    assert asked["type"] == "string"
    assert len(asked["enum"]) == len(asked["enum_labels"])
    assert asked["enum"][0] == "allow" and asked["enum"][-1] == "deny"


def test_the_dialog_carries_its_machine_readable_detail():
    """A client that answers by rule matches on data, not on the prose.

    ``title`` and ``body`` are renderings; a policy client (a benchmark
    driver, a GUI with per-host rules) needs the same facts as data, or it is
    back to parsing English — the mistake the pointer-line rewrite killed for
    attachments. The detail rides ``request_input`` so it lands in the request
    metadata and the phase frame together.
    """
    runtime = FakeRuntime()
    request, decision = _egress()
    build_approver(runtime)(Chain(root="user").push("summarize"), request,
                            decision)

    detail = runtime.asked[0]["kwargs"]["detail"]
    assert detail["type"] == "net.http"
    assert detail["method"] == "POST"
    assert detail["url"] == "https://example.invalid/collect"
    assert detail["asker"] == "summarize"


def test_shell_detail_speaks_the_grant_vocabulary():
    """``prefixes`` uses ``shell.command_prefix`` units, or is absent.

    A rule stored as ``git status`` must match a question asked as
    ``git status`` — one vocabulary, same as ``shell_allowed_prefixes``. A
    line the lexer refuses offers no prefixes at all rather than wrong ones,
    which is the recognizers' own all-or-nothing answer.
    """
    from sandbox.approval import detail_for

    chain = Chain(root="user").push("run_command")
    plain = detail_for(chain, Request(R.PROC_RUN,
                                      {"argv": ["git", "status"]}))
    assert plain["prefixes"] == ["git status"]
    assert plain["command"]

    opaque = detail_for(chain, Request(R.PROC_RUN,
                                       {"argv": "echo $HOME > ~/.bashrc",
                                        "shell": "default"}))
    assert "prefixes" not in opaque
    assert opaque["command"]


def test_choosing_a_remembering_option_grants_and_allows(monkeypatch):
    """And a grant that cannot be written down is still an approval."""
    from sandbox import options

    calls = []
    option = options.Option("always:x", "Always allow x",
                            remember=lambda: calls.append("wrote") or True)
    monkeypatch.setattr(options, "OPTION_BUILDERS",
                        [lambda chain, request, decision: [option]])

    runtime = FakeRuntime()
    runtime._answer_value = "always:x"
    request, decision = _egress()
    assert build_approver(runtime)(Chain().push("t"), request,
                                   decision) is True
    assert calls == ["wrote"]

    def boom():
        raise OSError("disk full")

    monkeypatch.setattr(options, "OPTION_BUILDERS", [
        lambda chain, request, decision: [
            options.Option("always:x", "Always allow x", remember=boom)]])
    assert build_approver(runtime)(Chain().push("t"), request,
                                   decision) is True


def test_an_unrecognised_answer_is_a_refusal():
    """A restored session can answer a dialog an older build wrote."""
    runtime = FakeRuntime()
    runtime._answer_value = "always:something-this-build-never-offered"
    request, decision = _egress()
    assert build_approver(runtime)(Chain().push("t"), request,
                                   decision) is False


def test_no_runtime_means_refuse():
    """The same default the kernel uses when every gate abstains."""
    request, decision = _egress()
    assert build_approver(None)(Chain().push("t"), request, decision) is False


def test_a_broken_runtime_refuses_rather_than_raising():
    """A failure to ask is a refusal, never an exception into the gate."""
    class Broken(FakeRuntime):
        """Cannot render dialogs."""
        def request_input(self, *a, **kw):
            """Fail."""
            raise RuntimeError("no frontend")

    request, decision = _egress()
    assert build_approver(Broken())(Chain().push("t"), request,
                                    decision) is False


# ──────────────────────────────────────────────────────────────────────
# End to end.
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def script(tmp_path):
    """A script that tries to reach the network."""
    path = tmp_path / "reach.py"
    path.write_text(
        "def go(sdk):\n"
        "    try:\n"
        "        sdk.net.http('https://example.invalid/x')\n"
        "        return {'denied': False}\n"
        "    except sdk.Denied:\n"
        "        return {'denied': True}\n",
        encoding="utf-8")
    return path


def test_a_refused_request_reaches_the_plugin_as_a_denial(script):
    """The whole path: policy, dialog, refusal, ordinary failure."""
    from guest.loader import unload_box

    runtime = FakeRuntime(answer=False)
    box = Sandbox(runtime=runtime)
    try:
        result = box.run(script, "go", chain=Chain(root="user"))
    finally:
        box.shutdown()
        unload_box("reach")

    assert result.ok
    assert result.data == {"denied": True}
    assert len(runtime.asked) == 1
    assert "example.invalid" in runtime.asked[0]["prompt"]
    assert "Asked by reach" in runtime.asked[0]["prompt"]


def test_safe_requests_never_reach_the_dialog(tmp_path):
    """Approval fatigue is the failure mode; safe work must be silent."""
    from guest.loader import unload_box

    target = tmp_path / "note.txt"
    target.write_text("hello", encoding="utf-8")
    path = tmp_path / "reader.py"
    path.write_text(
        "def go(sdk, p):\n"
        "    return sdk.fs.read(p)\n", encoding="utf-8")

    runtime = FakeRuntime(answer=True)
    box = Sandbox(runtime=runtime)
    try:
        result = box.run(path, "go", kwargs={"p": str(target)})
    finally:
        box.shutdown()
        unload_box("reader")

    assert result.data == "hello"
    assert runtime.asked == []


# ──────────────────────────────────────────────────────────────────────
# The doorway was built for tools; a Request is a different question.
# ──────────────────────────────────────────────────────────────────────

def test_a_gate_receives_the_request_whole_not_flattened():
    """The old contract offered a command string. A Request has a type,
    arguments, a chain, and a decision — all of which a gate wants."""
    from runtime.hooks import HookRegistry, PermissionVerdict

    seen = {}

    def gate(ctx, query):
        """Inspect the query and abstain."""
        seen["origin"] = query.origin
        seen["type"] = query.request.type
        seen["url"] = query.request.args.get("url")
        seen["chain"] = query.chain.render()
        seen["reason"] = query.decision.reason
        seen["command"] = query.command
        return None

    hooks = HookRegistry()
    hooks.add("vet_permission", gate)
    runtime = FakeRuntime(answer=True, hooks=hooks)
    request, decision = _egress()
    chain = Chain(root="cron:nightly").push("task_index")

    build_approver(runtime)(chain, request, decision)

    assert seen["origin"] == "request"
    assert seen["type"] == "net.http"
    assert seen["url"] == "https://example.invalid/collect"
    assert seen["chain"] == "cron:nightly -> task_index"
    assert "example.invalid" in seen["reason"]
    # The readable rendering is still there, so gates written against the
    # original shape keep working.
    assert "net.http" in seen["command"]


def test_a_gate_written_for_tools_still_works():
    """Backward compatibility is the point of carrying both shapes."""
    from runtime.hooks import HookRegistry, PermissionVerdict

    def old_style_gate(ctx, query):
        """Only knows about tool_name and command."""
        if "example.invalid" in query.command:
            return PermissionVerdict(False, "blocked host")
        return None

    hooks = HookRegistry()
    hooks.add("vet_permission", old_style_gate)
    runtime = FakeRuntime(answer=True, hooks=hooks)
    request, decision = _egress()

    assert build_approver(runtime)(Chain().push("t"), request,
                                   decision) is False
    assert runtime.asked == []


def test_tool_approvals_still_report_themselves_as_tools():
    """origin distinguishes the two askers; the default must not change."""
    from runtime.hooks import PermissionQuery

    query = PermissionQuery(tool_name="run_command", command="ls")
    assert query.origin == "tool"
    assert query.request is None and query.chain is None


# ──────────────────────────────────────────────────────────────────────
# A secret belongs to the plugin that declared it.
# ──────────────────────────────────────────────────────────────────────

def _reveal(name="litellm_api_key"):
    """A reveal Request and its decision."""
    request = Request(R.SECRET_REVEAL, {"name": name})
    return request, classify(request, Chain())


def test_a_plugin_revealing_its_own_credential_is_not_asked(monkeypatch):
    """Configuring a key *for* a service is the consent.

    Asking again on every service load is exactly the approval fatigue that
    kills permission systems - and the user already answered by setting it up.
    """
    monkeypatch.setattr(
        "plugins.plugin_discovery.get_setting_plugin_names",
        lambda key: ["litellm"] if key == "litellm_api_key" else [])

    runtime = FakeRuntime(answer=False)
    request, decision = _reveal()
    chain = Chain(root="user").push("litellm")

    assert build_approver(runtime)(chain, request, decision) is True
    assert runtime.asked == []


def test_a_different_plugin_asking_for_it_is_asked(monkeypatch):
    """That one is a genuinely different question."""
    monkeypatch.setattr(
        "plugins.plugin_discovery.get_setting_plugin_names",
        lambda key: ["litellm"] if key == "litellm_api_key" else [])

    runtime = FakeRuntime(answer=False)
    request, decision = _reveal()
    chain = Chain(root="user").push("some_other_tool")

    assert build_approver(runtime)(chain, request, decision) is False
    assert len(runtime.asked) == 1
    assert "litellm_api_key" in runtime.asked[0]["prompt"]


def test_an_unowned_secret_is_always_asked(monkeypatch):
    """No declared owner means nobody has agreed to anything yet."""
    monkeypatch.setattr(
        "plugins.plugin_discovery.get_setting_plugin_names", lambda key: [])

    runtime = FakeRuntime(answer=False)
    request, decision = _reveal("some_loose_token")
    assert build_approver(runtime)(Chain().push("x"), request,
                                   decision) is False
    assert len(runtime.asked) == 1


def test_ownership_never_short_circuits_anything_else(monkeypatch):
    """The rule is about secrets only - it must not soften egress."""
    monkeypatch.setattr(
        "plugins.plugin_discovery.get_setting_plugin_names",
        lambda key: ["litellm"])

    runtime = FakeRuntime(answer=False)
    request, decision = _egress()
    assert build_approver(runtime)(Chain().push("litellm"), request,
                                   decision) is False
    assert len(runtime.asked) == 1


def test_resident_service_may_persist_its_own_declared_setting(monkeypatch):
    monkeypatch.setattr(
        "plugins.plugin_discovery.get_setting_plugin_names",
        lambda key: ["timekeeper"] if key == "scheduled_jobs" else [],
    )
    request = Request(
        R.CONFIG_WRITE,
        {"key": "scheduled_jobs", "value": {}, "scope": "plugin"},
    )

    decision = classify(request, Chain(root="service:timekeeper"))

    assert decision.safe
    assert "persists its own" in decision.reason


def test_resident_service_cannot_persist_another_plugins_setting(monkeypatch):
    monkeypatch.setattr(
        "plugins.plugin_discovery.get_setting_plugin_names",
        lambda key: ["other_plugin"],
    )
    request = Request(
        R.CONFIG_WRITE,
        {"key": "scheduled_jobs", "value": {}, "scope": "plugin"},
    )

    decision = classify(request, Chain(root="service:timekeeper"))

    assert not decision.safe


# ────────────────────────────────────────────────────────────────────
# The sandbox must have an approver (was test_sandbox_approval_wiring.py)
# ────────────────────────────────────────────────────────────────────

from types import SimpleNamespace
from sandbox.facade import Sandbox
from sandbox.guest.requests import PLUGIN_UPDATE, Request
from sandbox.interpreter import Execution
from sandbox.policy import Chain


class _Pending:
    """A dialog the user has already answered."""

    def __init__(self, approved: bool):
        self.id = "p1"
        self.value = "allow" if approved else "deny"
        self.metadata: dict = {}

    @property
    def approved(self) -> bool:
        return bool(self.value)

    def wait(self, timeout=None) -> bool:
        return True


def _runtime(approved: bool, asked: list):
    """A runtime that renders an approval dialog and gets an answer."""
    def request_input(key, title, body, **kwargs):
        asked.append((title, body))
        return _Pending(approved)

    return SimpleNamespace(
        active_session_key="repl",
        sessions={"repl": SimpleNamespace(attended=True)},
        hooks=None,
        user_setting=lambda key, name: [],
        is_attended=lambda key: True,
        request_input=request_input,
        answer_request=lambda *a, **k: None,
    )


def _gate(sandbox, request):
    """Push one Request through the real gate and return its Result.

    The read blocks because the answer is now *asynchronous*: the gate hands
    an unsafe Request to an approval worker and returns immediately, so the
    settle lands on another thread.
    """
    execution = Execution(name="packages",
                          chain=Chain(root="user").push("packages"))
    sandbox.interpreter._gate_one(execution, request)
    return execution.inbox.get(timeout=5)


def test_asking_the_user_does_not_block_the_gate():
    """The deadlock, stated.

    The gate is the single ordering point for every Request in the process,
    including the ones the frontend makes to *draw* the dialog and to read the
    answer. Asking on the gate thread therefore made the question unaskable:
    ``/packages install`` showed no dialog, ignored ``y``, and froze the whole
    app until the wait expired.

    So while one dialog is open, an unrelated Request must still be classified
    and served.
    """
    import threading

    from sandbox.guest.requests import PATH_GET

    opened = threading.Event()
    release = threading.Event()

    def request_input(key, title, body, **kwargs):
        opened.set()
        release.wait(5)
        return _Pending(True)

    runtime = _runtime(True, [])
    runtime.request_input = request_input

    sandbox = Sandbox()
    sandbox.bind_runtime(runtime)
    interpreter = sandbox.interpreter

    def ask():
        """Occupy an approval with a dialog nobody has answered."""
        interpreter.submit(
            Execution(name="packages",
                      chain=Chain(root="user").push("packages")),
            Request(PLUGIN_UPDATE, {"name": "tool_edit_file"}))

    served: list = []

    def unrelated():
        """A perfectly safe Request from somewhere else entirely."""
        served.append(interpreter.submit(
            Execution(name="repl", chain=Chain(root="frontend:repl")),
            Request(PATH_GET, {"name": "data"})))

    try:
        # Both go through the real gate queue and the real gate thread —
        # calling ``_gate_one`` directly would run the approver on the test's
        # own thread and prove nothing about the gate.
        threading.Thread(target=ask, daemon=True).start()
        assert opened.wait(5), "the approver was never reached"

        answered = threading.Thread(target=unrelated, daemon=True)
        answered.start()
        answered.join(timeout=5)
        assert served, "the gate was blocked behind an open dialog"
        assert served[0].ok
    finally:
        release.set()
        sandbox.shutdown()


@pytest.fixture()
def request_update():
    return Request(PLUGIN_UPDATE, {"name": "tool_edit_file"})


def test_an_unwired_sandbox_refuses_without_asking(request_update):
    """The bug, stated: this is what /packages update hit."""
    sandbox = Sandbox()

    assert sandbox.interpreter.can_ask is False
    result = _gate(sandbox, request_update)
    assert not result.ok
    assert "changes what the system can do" in result.error


def test_binding_a_runtime_puts_the_question_to_the_user(request_update):
    """The dialog is rendered and the Request is *permitted*.

    Deliberately not ``result.ok``. What happens after the yes is the
    package manager's business, and it shells out to ``git`` against a real
    store — so in a checkout without one (a container built from the source
    tree, where ``.dockerignore`` drops ``.git``) this failed on the work
    rather than on the permission, which is the one thing it is not about.
    A refusal is what the sibling test above pins; the absence of one is what
    this pins.
    """
    asked: list = []
    sandbox = Sandbox()
    sandbox.bind_runtime(_runtime(True, asked))

    assert sandbox.interpreter.can_ask is True
    result = _gate(sandbox, request_update)

    assert asked, "no dialog was rendered"
    assert result.code != ERROR_NOT_PERMITTED
    assert "changes what the system can do" not in (result.error or "")


def test_a_user_saying_no_is_still_a_refusal(request_update):
    """Wiring the dialog must not turn approval into a formality."""
    asked: list = []
    sandbox = Sandbox()
    sandbox.bind_runtime(_runtime(False, asked))

    result = _gate(sandbox, request_update)
    assert asked and not result.ok


def test_binding_does_not_clobber_an_explicit_approver(request_update):
    """A caller that supplied its own decision keeps it.

    Tests and the stress harness wire an approver directly; bootstrap calling
    ``bind_runtime`` afterwards must not silently replace it.
    """
    calls: list = []
    sandbox = Sandbox(approve=lambda chain, req, dec: calls.append(req) or True)
    sandbox.bind_runtime(_runtime(False, []))

    result = _gate(sandbox, request_update)
    # The supplied approver ran and its yes stood — not that the update
    # itself succeeded, which needs a store. See the note two tests up.
    assert calls
    assert result.code != ERROR_NOT_PERMITTED


def test_binding_nothing_is_a_no_op():
    """Absent a runtime there is still nobody to ask, and that must hold."""
    sandbox = Sandbox()
    sandbox.bind_runtime(None)
    assert sandbox.interpreter.can_ask is False


def test_the_lazy_sandbox_knows_where_plugin_trees_are():
    """``get_sandbox`` used to build a bare Sandbox directly.

    That skipped ``configure``, which is what sets ``plugin_roots`` — so
    ``dependencies_files`` resolved only inside a plugin's own tree, and an
    installed tool declaring a kernel helper would not have found it.
    """
    import sandbox.bridge as bridge

    saved = bridge._SANDBOX
    bridge._SANDBOX = None
    try:
        assert bridge.get_sandbox().plugin_roots
    finally:
        bridge._SANDBOX = saved
