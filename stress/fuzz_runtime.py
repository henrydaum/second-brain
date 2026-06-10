"""Stateful fuzzer for the conversation runtime — the syzkaller analog.

``ConversationRuntime`` is the kernel's single dispatcher and (per the project
notes) its accepted "ugly duckling": ~940 lines where ordering and the
user/ownership dimension intersect. That is exactly where a stateful fuzzer
earns its keep: bugs there hide in *sequences* (open → switch user → load →
delete → inject → turn), not in single calls.

This uses ``hypothesis``'s :class:`RuleBasedStateMachine`. Each rule is one
operation against the runtime's stable lifecycle API; hypothesis searches for
operation *orderings* that break an invariant. After every rule we run the full
:func:`stress.invariants.check_invariants` oracle, and we additionally keep a
tiny in-Python model of conversation ownership to assert the access guard's
return value matches reality (cross-user access must be refused; the owner must
succeed).

Run it::

    pytest stress/fuzz_runtime.py -q
    # deeper search:
    pytest stress/fuzz_runtime.py -q --hypothesis-seed=random \
        -o "addopts=" --hypothesis-verbosity=normal

A failing example is shrunk by hypothesis to a minimal reproducing sequence and
printed — turn that sequence into a regression test in ``tests/`` (the syzbot
"every bug becomes a reproducer" discipline).
"""

from __future__ import annotations

from hypothesis import HealthCheck, settings
from hypothesis.stateful import (
    Bundle,
    RuleBasedStateMachine,
    initialize,
    invariant,
    precondition,
    rule,
)
import hypothesis.strategies as st

from stress.boot import boot_kernel
from stress.fake_llm import MonkeyLLM
from stress.invariants import check_invariants, thread_names
from state_machine.conversation_phases import BASE_PHASE
from state_machine.serialization import save_state_marker

# Bounded populations keep the search space tractable and the shrinker fast.
SESSION_KEYS = ["s0", "s1", "s2"]
USER_LOGINS = ["alice", "bob", "carol"]
CATEGORIES = [None, "work", "personal", ""]
NOTIFY_MODES = ["on", "off", "mentions"]


class RuntimeStateMachine(RuleBasedStateMachine):
    """Throws random valid lifecycle sequences at one headless kernel."""

    conversations = Bundle("conversations")

    def __init__(self):
        super().__init__()
        self._baseline_threads = thread_names()
        self.kernel = boot_kernel(
            llm=MonkeyLLM(seed=1234),
            # Real agent profiles so profile-switching exercises scope/spec
            # rebinding rather than no-opping. Both route to the fake LLM.
            config_overrides={"agent_profiles": {
                "default": {"llm": "default"},
                "terse": {"llm": "default"},
                "research": {"llm": "default"},
            }},
        )
        self.rt = self.kernel.runtime
        # External-id -> user_id, lazily minted. The base user (1) is implicit.
        self.users: dict[str, int] = {}
        # Our own model of "who owns conversation X" to cross-check the guard.
        self.owner: dict[int, int] = {}

    # ── identity helpers ────────────────────────────────────────────

    def _user_id(self, login: str) -> int:
        if login not in self.users:
            self.users[login] = self.kernel.db.upsert_user("fuzz", login)
        return self.users[login]

    def _bind(self, session_key: str, login: str) -> int:
        """Bind a session to a user's identity.

        We call ``set_session_user`` directly — the kernel itself now treats an
        identity change on a live session as an account switch (detach the
        departing user's conversation, load the new user's last-active), so the
        fuzzer exercises that real guarantee rather than papering over it.
        """
        uid = self._user_id(login)
        self.rt.set_session_user(session_key, uid)
        return uid

    # ── rules: identity / conversations ─────────────────────────────

    @rule(target=conversations, login=st.sampled_from(USER_LOGINS),
          title=st.text(min_size=0, max_size=20))
    def create_conversation(self, login, title):
        uid = self._user_id(login)
        cid = self.rt.create_conversation(title=title or "untitled", user_id=uid)
        if cid is not None:
            self.owner[cid] = uid
        return cid

    @rule(session_key=st.sampled_from(SESSION_KEYS), login=st.sampled_from(USER_LOGINS))
    def open_fresh_session(self, session_key, login):
        self._bind(session_key, login)
        self.rt.new_conversation(session_key)
        # Record ownership of whatever conversation the session now holds.
        sess = self.rt.sessions.get(session_key)
        if sess and sess.conversation_id is not None:
            self.owner[sess.conversation_id] = self.rt.session_user_id(session_key)

    @rule(session_key=st.sampled_from(SESSION_KEYS), login=st.sampled_from(USER_LOGINS),
          cid=conversations)
    def load_conversation(self, session_key, login, cid):
        if cid is None:
            return
        uid = self._bind(session_key, login)
        # Close any prior binding so we exercise load on a clean session
        # (rebinding a live session to a different conversation is a separate,
        # intentionally-refused path covered by the SessionConflict tests).
        self.rt.close_session(session_key)
        self._bind(session_key, login)
        expected_ok = self.owner.get(cid) == uid
        try:
            self.rt.load_conversation(session_key, cid)
            got_ok = True
        except PermissionError:
            got_ok = False
        # Owner must succeed; non-owner must be refused. This is the core guard.
        assert got_ok == expected_ok, (
            f"load_conversation guard mismatch: user {uid} on conversation {cid} "
            f"owned by {self.owner.get(cid)} -> got_ok={got_ok}, expected={expected_ok}"
        )

    @rule(session_key=st.sampled_from(SESSION_KEYS), login=st.sampled_from(USER_LOGINS),
          cid=conversations)
    def load_history_guard(self, session_key, login, cid):
        if cid is None:
            return
        uid = self._bind(session_key, login)
        # Same clean-session discipline as load_conversation: rebinding a live
        # session is the separately-covered SessionConflict path.
        self.rt.close_session(session_key)
        self._bind(session_key, login)
        result = self.rt.load_history(session_key, cid)
        expected = self.owner.get(cid) == uid
        assert result.ok == expected, (
            f"load_history guard mismatch: user {uid} on conversation {cid} "
            f"owned by {self.owner.get(cid)} -> ok={result.ok}, expected={expected}"
        )
        if not expected:
            # Refusal must not leak existence.
            assert result.messages == ["No such conversation."]

    @rule(session_key=st.sampled_from(SESSION_KEYS), login=st.sampled_from(USER_LOGINS),
          cid=conversations, text=st.text(min_size=1, max_size=30))
    def inject_user_message_guard(self, session_key, login, cid, text):
        if cid is None:
            return
        uid = self._bind(session_key, login)
        self.rt.close_session(session_key)
        self._bind(session_key, login)
        # Count only user-authored rows: loading a conversation may legitimately
        # persist system marker rows alongside the injected message.
        def _user_rows():
            return sum(1 for m in self.kernel.db.get_conversation_messages(cid)
                       if m["role"] == "user")
        before = _user_rows()
        result = self.rt.inject_user_message(session_key, text, conversation_id=cid)
        after = _user_rows()
        expected = self.owner.get(cid) == uid
        assert result.ok == expected, (
            f"inject guard mismatch: user {uid} into conversation {cid} "
            f"owned by {self.owner.get(cid)} -> ok={result.ok}, expected={expected}"
        )
        # Owner's message lands; a refused non-owner writes nothing.
        assert after == before + (1 if expected else 0)

    @rule(session_key=st.sampled_from(SESSION_KEYS), login=st.sampled_from(USER_LOGINS),
          text=st.text(min_size=1, max_size=40))
    def turn(self, session_key, login, text):
        self._bind(session_key, login)
        result = self.rt.iterate_agent_turn(session_key, text)
        assert result is not None
        sess = self.rt.sessions.get(session_key)
        if sess and sess.conversation_id is not None:
            self.owner.setdefault(sess.conversation_id, self.rt.session_user_id(session_key))

    @rule(session_key=st.sampled_from(SESSION_KEYS), login=st.sampled_from(USER_LOGINS),
          cid=conversations)
    def delete_conversation(self, session_key, login, cid):
        if cid is None:
            return
        uid = self._bind(session_key, login)
        expected = self.owner.get(cid) == uid
        got = self.rt.delete_conversation(session_key, cid)
        assert got == expected, (
            f"delete guard mismatch: user {uid} deleting conversation {cid} "
            f"owned by {self.owner.get(cid)} -> {got}, expected {expected}"
        )
        if got:
            self.owner.pop(cid, None)

    @rule(session_key=st.sampled_from(SESSION_KEYS), login=st.sampled_from(USER_LOGINS),
          cid=conversations, category=st.sampled_from(CATEGORIES))
    def set_category(self, session_key, login, cid, category):
        if cid is None:
            return
        uid = self._bind(session_key, login)
        expected = self.owner.get(cid) == uid
        got = self.rt.set_conversation_category(session_key, cid, category)
        assert got == expected

    @rule(session_key=st.sampled_from(SESSION_KEYS), login=st.sampled_from(USER_LOGINS),
          cid=conversations, mode=st.sampled_from(NOTIFY_MODES))
    def set_notify(self, session_key, login, cid, mode):
        if cid is None:
            return
        self._bind(session_key, login)
        # Return contract here is looser (None on refusal/no-op); we just assert
        # it never raises and the invariants hold afterward.
        self.rt.set_conversation_notification_mode(session_key, cid, mode)

    @rule(session_key=st.sampled_from(SESSION_KEYS))
    def close_session(self, session_key):
        self.rt.close_session(session_key)

    @rule(session_key=st.sampled_from(SESSION_KEYS), login=st.sampled_from(USER_LOGINS),
          profile=st.sampled_from(["default", "terse", "research"]))
    def switch_agent_profile(self, session_key, login, profile):
        """Switch the agent profile on a (possibly fresh) session and keep using it."""
        self._bind(session_key, login)
        # Ensure the session exists for set_agent_profile to act on.
        self.rt.iterate_agent_turn(session_key, "warm up")
        ok = self.rt.set_agent_profile(session_key, profile)
        assert ok is True
        sess = self.rt.sessions.get(session_key)
        assert sess is not None and sess.profile_override == profile
        # A turn under the new profile must still complete.
        self.rt.iterate_agent_turn(session_key, "go")

    @rule(session_key=st.sampled_from(SESSION_KEYS), login=st.sampled_from(USER_LOGINS),
          plugin=st.sampled_from(["notes", "search", "x"]),
          key=st.sampled_from(["a", "b", "c"]),
          value=st.one_of(st.integers(-3, 9), st.text(max_size=12), st.booleans(), st.none()))
    def session_plugin_state(self, session_key, login, plugin, key, value):
        """Round-trip session plugin state through update/get/clear (+ persist)."""
        self._bind(session_key, login)
        self.rt.iterate_agent_turn(session_key, "warm up")
        assert self.rt.update_session_plugin_state(session_key, plugin, {key: value}) is True
        assert self.rt.get_session_plugin_state(session_key, plugin, key) == value
        assert self.rt.clear_session_plugin_state(session_key, plugin, key) is True
        assert self.rt.get_session_plugin_state(session_key, plugin, key, "_gone") == "_gone"

    @rule(session_key=st.sampled_from(SESSION_KEYS), login=st.sampled_from(USER_LOGINS),
          key=st.sampled_from(["persona", "rules"]),
          value=st.one_of(st.text(max_size=20), st.none()))
    def system_prompt_extra(self, session_key, login, key, value):
        """Attach/clear a system-prompt overlay; must never desync the session."""
        self._bind(session_key, login)
        self.rt.iterate_agent_turn(session_key, "warm up")
        self.rt.add_system_prompt_extra(session_key, key, value)

    @rule(session_key=st.sampled_from(SESSION_KEYS), login=st.sampled_from(USER_LOGINS))
    def persist_and_reload(self, session_key, login):
        """Crash-recovery path: persist a live conversation, drop the in-memory
        session, reload it from the DB as its owner, and confirm the restored
        session is healthy and keeps working — the provider history must survive
        the round-trip intact.
        """
        self._bind(session_key, login)
        self.rt.iterate_agent_turn(session_key, "remember this")
        sess = self.rt.sessions.get(session_key)
        if sess is None or sess.conversation_id is None:
            return
        cid = sess.conversation_id
        owner = self.owner.get(cid, self.rt.session_user_id(session_key))
        before = list(sess.history)

        self.rt.close_session(session_key)
        self.rt.set_session_user(session_key, owner)
        self.rt.load_conversation(session_key, cid)

        restored = self.rt.sessions.get(session_key)
        assert restored is not None and restored.conversation_id == cid
        # History round-trips through the DB unchanged.
        assert restored.history == before, (
            f"history drift on reload of {cid}: {len(before)} -> {len(restored.history)}"
        )
        # And the restored session still drives a turn.
        self.rt.iterate_agent_turn(session_key, "still there?")

    @rule(session_key=st.sampled_from(SESSION_KEYS), login=st.sampled_from(USER_LOGINS))
    def stale_busy_reload(self, session_key, login):
        self._bind(session_key, login)
        self.rt.iterate_agent_turn(session_key, "warm up")
        sess = self.rt.sessions.get(session_key)
        if sess is None or sess.conversation_id is None:
            return
        cid = sess.conversation_id
        marker = {**sess.to_marker(), "busy": True, "turn_priority": "agent", "phase": BASE_PHASE}
        marker["cache"] = {**(marker.get("cache") or {}), "phases": []}
        save_state_marker(self.kernel.db, cid, marker)
        self.rt.close_session(session_key)
        self._bind(session_key, login)
        self.rt.load_conversation(session_key, cid)
        restored = self.rt.sessions[session_key]
        assert restored.cs.turn_priority == "user" and restored.cs.phase == BASE_PHASE

    @rule(cid=conversations)
    def raw_delete_then_drive(self, cid):
        """Delete a conversation via the *raw* db path (bypassing the runtime's
        own detach), then drive a benign action on every session so the
        write-path backstop must self-heal any session that was holding it —
        rather than dangle (invariant) or crash on the trailing persist_marker.
        """
        if cid is None:
            return
        self.kernel.db.delete_conversation(cid)
        self.owner.pop(cid, None)
        # Drive a no-op through handle_action on each session: the backstop runs
        # at entry and detaches stale holders before persist_marker writes.
        for key in SESSION_KEYS:
            if key in self.rt.sessions:
                self.rt.handle_action(key, "cancel")

    # ── oracle ──────────────────────────────────────────────────────

    @invariant()
    def kernel_is_healthy(self):
        violations = check_invariants(self.kernel, baseline_threads=self._baseline_threads)
        assert not violations, "Kernel invariant(s) broken:\n" + "\n".join(map(str, violations))

    def teardown(self):
        self.kernel.close()


# Background daemon threads (compactor, etc.) and per-turn DB work make the
# default deadline/health checks too strict for an integration-shaped machine.
RuntimeStateMachine.TestCase.settings = settings(
    max_examples=50,
    stateful_step_count=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
    # CI failures print a reproduction blob (`@reproduce_failure(...)`) so a
    # seed-dependent failure on the workflow runner can be replayed locally.
    print_blob=True,
)

# pytest entrypoint
TestRuntimeFuzz = RuntimeStateMachine.TestCase
