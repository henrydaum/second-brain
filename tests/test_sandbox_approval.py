"""Approval: the moment the sandbox becomes visible to a person.

The order under test is the whole design — policy hooks, then the user's
trusted list, then a dialog, then a refusal when nobody is home. Each step
reuses machinery the kernel already has rather than duplicating it.
"""

import pytest

import sandbox  # noqa: F401  - installs the ``guest`` package alias
from sandbox import Chain, Interpreter, Request, Sandbox
from sandbox.approval import build_approver, describe
from sandbox.guest import requests as R
from sandbox.policy import classify


class FakeRequest:
    """Stands in for the kernel's pending-input object."""

    def __init__(self, answer=True, cancelled=False, answers=True):
        self.id = 1
        self.approved = answer
        self.metadata = {"cancelled": cancelled}
        self._answers = answers

    def wait(self, timeout=None):
        """Whether the user got back to us."""
        return self._answers


class FakeRuntime:
    """Just enough runtime to render a dialog."""

    def __init__(self, *, answer=True, attended=True, trusted=(),
                 hooks=None, answers=True, cancelled=False):
        self.active_session_key = "repl"
        self.sessions = {"repl": object()}
        self.hooks = hooks
        self.asked = []
        self._answer = answer
        self._attended = attended
        self._trusted = list(trusted)
        self._answers = answers
        self._cancelled = cancelled
        self.answered = []

    def is_attended(self, key):
        """Whether a human is present."""
        return self._attended

    def user_setting(self, key, name):
        """The user's trusted list."""
        return self._trusted if name == "skip_permissions" else None

    def request_input(self, key, title, prompt, **kwargs):
        """Render the dialog."""
        self.asked.append({"key": key, "title": title, "prompt": prompt,
                           "kwargs": kwargs})
        return FakeRequest(self._answer, self._cancelled, self._answers)

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
    assert "approval" in title.lower()


def test_the_dialog_shows_the_whole_chain():
    """The reason provenance exists, shown in the only place it matters."""
    request, decision = _egress()
    chain = Chain(root="cron:nightly_index").push("task_index").push("fetch")
    _, body = describe(chain, request, decision)
    assert "cron:nightly_index -> task_index -> fetch" in body


def test_shell_commands_are_shown_as_code():
    """A command the user has to judge must be legible, not prose-wrapped."""
    request = Request(R.PROC_RUN, {"argv": ["rm", "-rf", "build"]})
    _, body = describe(Chain(), request, classify(request, Chain()))
    assert "```" in body
    assert "rm -rf build" in body


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


def test_the_trusted_list_short_circuits_the_dialog():
    """Things the user already decided about are not re-asked."""
    runtime = FakeRuntime(answer=False, trusted=["summarize"])
    request, decision = _egress()
    chain = Chain(root="user").push("summarize")
    assert build_approver(runtime)(chain, request, decision) is True
    assert runtime.asked == []


def test_trust_is_checked_against_the_whole_chain():
    """Trusting a tool trusts what it does, including through a service."""
    runtime = FakeRuntime(answer=False, trusted=["summarize"])
    request, decision = _egress()
    chain = Chain(root="user").push("summarize").push("service_web")
    assert build_approver(runtime)(chain, request, decision) is True


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


def test_a_timed_out_dialog_is_a_refusal_and_is_answered():
    """The pending request must not be left hanging in the session."""
    runtime = FakeRuntime(answer=True, answers=False)
    request, decision = _egress()
    assert build_approver(runtime)(Chain().push("t"), request,
                                   decision) is False
    assert runtime.answered == [False]


def test_nobody_present_means_refused_not_blocked():
    """An unattended session must never wait on a dialog nobody sees."""
    runtime = FakeRuntime(attended=False)
    request, decision = _egress()
    assert build_approver(runtime)(Chain(root="cron:x").push("t"), request,
                                   decision) is False
    assert runtime.asked == []


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
    assert "user -> reach" in runtime.asked[0]["prompt"]


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
