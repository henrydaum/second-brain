"""Two things the kernel already knew and had no way to say.

Both of these spend a fact the kernel was computing anyway.

``Result.retryable`` is set at eighteen handler sites — a locked file, an HTTP
timeout, a box that died — and until :meth:`SDK.retry` existed it was read by
nobody at all. It is the better half of the bargain than a guest-side rule
about exception types would be: whether a failure is transient is known where
it happened, not where it is caught.

``sdk.budget`` is the same shape one layer over. The watchdog compares two
numbers every half second to decide whether to kill a box, and the box could
see neither of them — so the only way to discover a deadline was to be killed
by it, discarding whatever had been computed on the way.
"""

import time

import pytest

import sandbox  # noqa: F401  - installs the ``guest`` package alias
from sandbox import Chain, Sandbox
from sandbox.guest.requests import RequestFailed, Result
from sandbox.guest.sdk import SDK
from tests.support import retarget_trees


class Wire:
    """A channel that answers from a queued list, and counts the asks."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.sent = []
        self.logs = []

    def send(self, request):
        """Hand back the next queued Result."""
        self.sent.append(request.type)
        return self.answers.pop(0) if self.answers else Result(data=None)

    def log(self, level, message):
        """Record, so a test can see the retry announce itself."""
        self.logs.append(message)


def sdk_over(*answers):
    """An SDK whose next Requests answer with ``answers``."""
    wire = Wire(answers)
    return SDK(wire), wire


# ──────────────────────────────────────────────────────────────────────
# Retrying.
# ──────────────────────────────────────────────────────────────────────

def test_a_transient_failure_is_retried_and_then_succeeds():
    """The whole point: the kernel said it was worth trying again."""
    sdk, wire = sdk_over(
        Result.failure("locked", retryable=True),
        Result.failure("locked", retryable=True),
        Result(data="the text"),
    )

    got = sdk.retry(lambda: sdk.fs.read("notes.md"), backoff=0)

    assert got == "the text"
    assert len(wire.sent) == 3


def test_a_permanent_failure_is_not_retried_at_all():
    """A malformed query does not get better by being asked twice.

    This is the half that makes the helper worth having: a blind ``for`` loop
    would spend three attempts on something that can never work, and take three
    times as long to report the same error.
    """
    sdk, wire = sdk_over(Result.failure("no such column: nope"))

    with pytest.raises(RequestFailed) as raised:
        sdk.retry(lambda: sdk.db.query("SELECT nope"), backoff=0)

    assert "no such column" in raised.value.error
    assert len(wire.sent) == 1


def test_a_refusal_is_never_retried_whatever_the_predicate_says():
    """Policy is not a transient condition.

    The ordering is load-bearing rather than tidy: ``Denied`` subclasses
    ``Failed``, so a helper that caught the general case first would sweep
    every refusal into the retry loop — and each attempt is a fresh dialog in
    front of a person who already answered. Pinned with an ``on`` predicate
    that says yes to everything, because that is the one arrangement in which a
    wrong implementation looks right.
    """
    sdk, wire = sdk_over(Result.refusal("the user declined"))

    with pytest.raises(sdk.Denied):
        sdk.retry(lambda: sdk.net.http("https://example.com"),
                  on=lambda exc: True, backoff=0)

    assert len(wire.sent) == 1


def test_the_last_attempt_raises_rather_than_returning_nothing():
    """Giving up is a failure, not a quiet ``None``.

    Worth pinning because the loop's final iteration is the one path that does
    not go through the ``on`` check, and a fall-through there would hand the
    caller ``None`` for a read that never happened.
    """
    sdk, _ = sdk_over(*[Result.failure("busy", retryable=True)] * 3)

    with pytest.raises(RequestFailed) as raised:
        sdk.retry(lambda: sdk.fs.read("x"), attempts=3, backoff=0)

    assert "busy" in raised.value.error


def test_a_predicate_overrides_the_kernels_verdict_in_both_directions():
    """``on`` is how a caller who knows better says so.

    Both directions, because a predicate that could only ever widen would be a
    footgun and one that could only narrow would not cover the retry-a-404 case
    it exists for.
    """
    sdk, wire = sdk_over(Result.failure("gone", retryable=False),
                         Result(data="found it"))
    assert sdk.retry(lambda: sdk.fs.read("x"),
                     on=lambda exc: True, backoff=0) == "found it"
    assert len(wire.sent) == 2

    sdk, wire = sdk_over(Result.failure("busy", retryable=True))
    with pytest.raises(RequestFailed):
        sdk.retry(lambda: sdk.fs.read("x"), on=lambda exc: False, backoff=0)
    assert len(wire.sent) == 1


def test_one_attempt_is_a_plain_call():
    """``attempts=1`` is how a caller turns the helper off without unwrapping."""
    sdk, wire = sdk_over(Result.failure("busy", retryable=True))

    with pytest.raises(RequestFailed):
        sdk.retry(lambda: sdk.fs.read("x"), attempts=1, backoff=0)

    assert len(wire.sent) == 1


def test_zero_attempts_is_a_mistake_and_says_so():
    """Silently doing nothing and returning ``None`` is the worse answer."""
    sdk, _ = sdk_over()
    with pytest.raises(ValueError):
        sdk.retry(lambda: sdk.fs.read("x"), attempts=0)


def test_teardown_is_not_a_retryable_failure():
    """``Terminated`` is a ``BaseException`` precisely so this cannot happen.

    A retry loop is the worst possible place to swallow the kernel tearing a
    box down: it would keep the box alive, keep making Requests nobody will
    answer, and do it in a loop. Nothing in the helper catches it, and this
    says so out loud rather than leaving it to the class hierarchy.
    """
    from sandbox.guest.channel import Terminated

    sdk, _ = sdk_over()
    attempts = []

    def torn_down():
        """Fail the way a cancelled guest fails."""
        attempts.append(1)
        raise Terminated(None)

    with pytest.raises(Terminated):
        sdk.retry(torn_down, backoff=0)

    assert len(attempts) == 1


def test_backoff_waits_between_attempts():
    """Retrying instantly three times is not retrying, it is hammering."""
    sdk, _ = sdk_over(Result.failure("busy", retryable=True),
                      Result(data="ok"))

    started = time.monotonic()
    sdk.retry(lambda: sdk.fs.read("x"), backoff=0.05)
    elapsed = time.monotonic() - started

    assert elapsed >= 0.05


# ──────────────────────────────────────────────────────────────────────
# The budget.
# ──────────────────────────────────────────────────────────────────────

BUDGET = '''\
timeout = {timeout}


def main(sdk):
    """Answer with what the kernel says is left."""
    return sdk.budget()


def twice(sdk):
    """Two readings with real work between them."""
    import time

    first = sdk.budget()
    end = time.monotonic() + 0.4
    while time.monotonic() < end:
        pass
    return [first, sdk.budget()]


def wrap_up(sdk, items=20):
    """Stop early and hand back partial work, which is the whole point."""
    import time

    done = []
    for n in range(items):
        if sdk.budget()["running"] < 0.6:
            break
        end = time.monotonic() + 0.1
        while time.monotonic() < end:
            pass
        done.append(n)
    return {{"done": done, "resume_at": len(done)}}
'''


@pytest.fixture
def sb():
    """A sandbox that refuses everything unsafe."""
    made = Sandbox()
    yield made
    made.shutdown()


@pytest.fixture
def budget_script(tmp_path, monkeypatch):
    """A script that reports its own budget, by declared timeout."""
    root = retarget_trees(monkeypatch, tmp_path)["workspace"]
    (root / "scripts").mkdir(parents=True)

    def write(timeout=2):
        """Put it on disk and answer with the path."""
        target = root / "scripts" / "budget.py"
        target.write_text(BUDGET.format(timeout=timeout), encoding="utf-8")
        return str(target)

    return write


def test_a_script_can_read_the_deadline_it_was_actually_given(sb,
                                                              budget_script):
    """The declared number, clamped — not the one in the file.

    A plugin may ask for a longer leash and does not get to grant itself one,
    so the honest thing to report is what is being enforced. Reporting the
    declaration would be worse than reporting nothing: it would be believed.
    """
    from sandbox.watchdog import HARD_CEILING

    result = sb.run(budget_script(2), "main", chain=Chain(root="user"))

    assert result.ok, result.error
    assert result.data["deadline"] == 2.0
    assert result.data["ceiling"] == HARD_CEILING
    assert 0 < result.data["running"] <= 2.0
    assert 0 < result.data["wall"] <= HARD_CEILING


def test_the_declared_ceiling_is_what_a_long_declaration_reports(sb,
                                                                budget_script):
    """Asking for 5000 seconds reports the clamp, not the ask."""
    from sandbox.interpreter import MAX_TIMEOUT_SECONDS

    result = sb.run(budget_script(5000), "main", chain=Chain(root="user"))

    assert result.ok, result.error
    assert result.data["deadline"] == MAX_TIMEOUT_SECONDS


def test_the_budget_falls_as_the_guest_burns_it(sb, budget_script):
    """Running time, which is what the deadline measures.

    Burned in a spin loop rather than a sleep because those are the same thing
    to this number and only one of them is what a deadline exists to catch.
    """
    result = sb.run(budget_script(5), "twice", chain=Chain(root="user"))

    assert result.ok, result.error
    first, second = result.data
    assert first["running"] - second["running"] >= 0.3


def test_long_work_can_stop_itself_and_return_what_it_has(sb, budget_script):
    """The reason the Request exists.

    Without it the watchdog is the only thing that ends an over-long run, and
    it ends it by killing the box — so a loop most of the way through a corpus
    returns *nothing*. Here the same loop hands back what it finished and where
    to pick up, which is a result a caller can act on.
    """
    result = sb.run(budget_script(2), "wrap_up", chain=Chain(root="user"))

    assert result.ok, result.error
    assert 0 < len(result.data["done"]) < 20
    assert result.data["resume_at"] == len(result.data["done"])


def test_asking_costs_no_dialog_and_no_ledger_row(sb, budget_script):
    """It is read-only, and that is load-bearing rather than a nicety.

    A loop asks every iteration. Left out of ``READ_ONLY`` that would be a
    ledger row per tick — and, worse, a ``prompt_cues`` bump per tick, which
    invalidates every cached ``agent_prompt`` in the process. The same trap
    ``llm.delta`` is kept out of the ledger sink for.
    """
    from sandbox.guest import requests as R
    from sandbox.policy import SAFE, classify

    decision = classify(R.Request(R.SELF_BUDGET, {}),
                        Chain(root="cron:nightly").push("task_index"))

    assert decision.level == SAFE
    assert R.SELF_BUDGET in R.READ_ONLY


def test_nothing_in_force_answers_null_rather_than_a_number():
    """An invented deadline would be believed and acted on.

    A resident box between calls is under no deadline at all, and the honest
    answer is that there is none — a caller checking ``if budget < 20`` reads a
    fabricated ceiling as "plenty of time" and a null as "not applicable",
    which are different decisions.
    """
    from sandbox.interpreter import Execution

    execution = Execution(name="idle", chain=Chain())
    left = execution.remaining()

    assert left["running"] is None
    assert left["wall"] is None
    assert left["deadline"] is None
