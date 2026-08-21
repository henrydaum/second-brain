"""Notifications: the kind of event a frontend may put somewhere of its own.

Three things are worth pinning here, and each one fails silently if it breaks.

**The split.** A push made while the agent's turn owns the session is
conversation; everything else is a notification. That is decided at each emit
site rather than inferred, so nothing but a test states it.

**The fallback.** A frontend that never heard of notifications must keep
showing them in the chat exactly as before. If it stops, the announcement does
not error — it simply never appears, which looks identical to a system with
nothing to say.

**The attribution.** ``source`` comes off the provenance chain, so a plugin
cannot claim to be the plugin watcher. A forged source is not a crash either;
it is a lie a reader believes.
"""

# Import the state_machine package before runtime modules to settle the
# package-init circular import (state_machine/__init__ pulls in the runtime).
import state_machine  # noqa: F401

import os
import tempfile
import time

import pytest

from events.event_bus import bus
from events.event_channels import CHAT_MESSAGE_PUSHED, NOTIFICATION_PUSHED
from pipeline.database import Database
from plugins.native.frontend import BaseFrontend, FrontendCapabilities
from runtime import notifications as N
from sandbox import frontends as sandbox_frontends
from sandbox import policy
from sandbox.guest import requests as R


# ── fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def db(tmp_path):
    """A real database, pointed at by the notification layer."""
    database = Database(str(tmp_path / "notify.db"))
    previous = N._DB
    N.bind_db(database)
    yield database
    N.bind_db(previous)


@pytest.fixture
def captured():
    """Every NOTIFICATION_PUSHED emitted while the test runs."""
    seen: list[dict] = []
    unsub = bus.subscribe(NOTIFICATION_PUSHED, seen.append)
    yield seen
    unsub()


@pytest.fixture
def pushed():
    """Every CHAT_MESSAGE_PUSHED emitted while the test runs."""
    seen: list[dict] = []
    unsub = bus.subscribe(CHAT_MESSAGE_PUSHED, seen.append)
    yield seen
    unsub()


class _Frontend(BaseFrontend):
    """A frontend with one session and a record of what it was shown."""

    name = "cap"
    capabilities = FrontendCapabilities()

    def __init__(self, **caps):
        super().__init__()
        if caps:
            self.capabilities = FrontendCapabilities(**caps)
        self.messages: list[str] = []
        self.notifications: list[dict] = []

    def render_messages(self, _key, messages):
        self.messages.extend(messages)

    def render_notification(self, _key, payload):
        self.notifications.append(payload)

    def _live_session_keys(self):
        return ["s"]

    def _broadcast_session_keys(self):
        return ["s"]


# ── the row ────────────────────────────────────────────────────────────

def test_a_notification_is_persisted_and_delivered(db, captured):
    """Both halves, from one call. The row is for a panel filling itself on a
    fresh load; the emit is for whoever is already watching."""
    nid = N.notify(title="Plugin registered", body="tool_x",
                   source="plugin_watcher", level="success", user_id=1)

    assert nid is not None
    row = db.get_notifications()[0]
    assert (row["title"], row["body"], row["source"], row["level"]) == (
        "Plugin registered", "tool_x", "plugin_watcher", "success")
    assert row["read_at"] is None

    assert captured[0]["notification_id"] == nid
    assert captured[0]["source"] == "plugin_watcher"
    assert captured[0]["level"] == "success"


def test_an_unpersisted_notification_still_arrives(db, captured):
    """``persist=False`` is for progress — "Compacting conversation…" is worth
    interrupting for and worth nothing an hour later. It must still be *shown*,
    or the flag would silently mean "do not notify"."""
    N.notify(title="Conversation", body="Compacting...", source="runtime",
             persist=False)

    assert db.get_notifications() == []
    assert captured[0]["body"] == "Compacting..."
    assert "notification_id" not in captured[0]


def test_an_empty_notification_is_not_raised(db, captured):
    """Nothing to say is not something to show. Guarded because the emitters
    pass through interpolated strings that can come out empty."""
    assert N.notify(title="", body="   ", source="runtime") is None
    assert captured == []
    assert db.get_notifications() == []


def test_an_unknown_level_falls_back_rather_than_raising(db, captured):
    """A level only styles the result, so an unrecognised one must not cost
    the notification — a frontend keying off it would get a KeyError instead."""
    N.notify(title="x", body="y", source="s", level="catastrophe")
    assert captured[0]["level"] == "info"


def test_a_failed_write_still_delivers(captured, monkeypatch):
    """The write is best-effort and the emit is not conditional on it. A
    notification must never be lost because the database was busy."""
    class _Broken:
        def record_notification(self, **_):
            raise RuntimeError("disk on fire")

    N.notify(title="Still shown", body="b", source="s", db=_Broken())
    assert captured[0]["title"] == "Still shown"
    assert "notification_id" not in captured[0]


def test_notify_never_raises_into_its_caller(captured):
    """The whole point of the layer. Producers call this from inside a config
    write, a plugin load, a turn — none of which may fail because telling the
    user about it did."""
    class _Exploding:
        def record_notification(self, **_):
            raise RuntimeError("boom")

        def get_conversation(self, _):
            raise RuntimeError("boom")

    assert N.notify(title="t", body="b", source="s", conversation_id=3,
                    db=_Exploding()) is None


# ── retention ──────────────────────────────────────────────────────────

def test_retention_covers_notifications(db):
    """Folded into the one ``data_retention_days`` knob rather than getting a
    setting of its own — an unbounded table with no sweep is the failure this
    check exists for."""
    N.notify(title="old", body="b", source="s", user_id=1)
    db.conn.execute("UPDATE notifications SET ts = ?",
                    (time.time() - 99 * 86400,))
    db.conn.commit()

    db.prune_expired(30)
    assert db.get_notifications() == []


def test_retention_leaves_recent_rows(db):
    N.notify(title="new", body="b", source="s", user_id=1)
    db.prune_expired(30)
    assert len(db.get_notifications()) == 1


def test_the_ledger_only_sweep_leaves_notifications_alone(db):
    """``record_action`` runs the cheap sweep every 256 inserts. Catching
    notifications in it would delete them on a schedule nothing announced."""
    N.notify(title="keep", body="b", source="s", user_id=1)
    db.conn.execute("UPDATE notifications SET ts = ?",
                    (time.time() - 99 * 86400,))
    db.conn.commit()

    db.prune_expired(30, ledger_only=True)
    assert len(db.get_notifications()) == 1


# ── reading back ───────────────────────────────────────────────────────

def test_every_persisted_producer_reaches_the_panel(db):
    """The end-to-end read, driven by the *real* emitters.

    The test that was missing, and its absence hid a bug that made three of
    the four persisted sources unreachable. Everything below writes
    ``user_id = NULL`` — none of these belong to a person — and a panel
    filtering on ``user_id = ?`` returned exactly none of them. Nothing failed:
    the rows were written, the read came back empty, and an empty panel looks
    just like a system with nothing to say.

    The earlier scoping test passed throughout, because it wrote its own rows
    with an explicit ``user_id``. It proved the mechanism and never touched the
    seam.
    """
    N.notify(title="Plugin registered", body="tool_x", source="plugin_watcher",
             level="success")
    N.announce_config_change({"keys": ["theme"], "scope": "core"},
                             session_key="s")
    N.emit_fallback_push(session_key="spawn_subagent:7", conversation_id=None,
                         title="Nightly", final_text="Indexed 12 files.",
                         db=db, user_id=1)

    seen = {r["source"] for r in db.get_notifications(user_id=1)}
    assert seen == {"plugin_watcher", "config", "session"}


def test_reads_are_scoped_to_one_user(db):
    """A row belonging to somebody stays theirs; a system row is everyone's."""
    N.notify(title="mine", body="b", source="s", user_id=1)
    N.notify(title="theirs", body="b", source="s", user_id=2)
    N.notify(title="everyones", body="b", source="plugin_watcher")

    assert {r["title"] for r in db.get_notifications(user_id=1)} == {
        "mine", "everyones"}
    assert {r["title"] for r in db.get_notifications(user_id=2)} == {
        "theirs", "everyones"}


def test_a_system_notification_can_be_dismissed(db):
    """Shown to everyone, so settleable by anyone who was shown it.

    Excluding NULL rows from the update left every plugin registration
    permanently unread — drawn, clicked, and still there on the next load,
    with the count honestly reporting that nothing changed.
    """
    nid = N.notify(title="Plugin registered", body="tool_x",
                   source="plugin_watcher")

    assert db.mark_notifications_read([nid], user_id=1) == 1
    assert db.get_notifications(user_id=1, unread_only=True) == []


def test_one_user_still_cannot_settle_anothers(db):
    """The NULL widening must not have widened this too."""
    nid = N.notify(title="mine", body="b", source="s", user_id=1)

    assert db.mark_notifications_read([nid], user_id=2) == 0
    assert len(db.get_notifications(user_id=1, unread_only=True)) == 1


def test_since_id_is_the_incremental_read(db):
    first = N.notify(title="a", body="b", source="s", user_id=1)
    N.notify(title="b", body="b", source="s", user_id=1)

    assert [r["title"] for r in
            db.get_notifications(user_id=1, since_id=first)] == ["b"]


def test_marking_read_is_idempotent_and_owned(db):
    """The count is what actually changed, so a client calling twice does not
    double-report — and one user settling another's rows changes nothing."""
    nid = N.notify(title="a", body="b", source="s", user_id=1)

    assert db.mark_notifications_read([nid], user_id=1) == 1
    assert db.mark_notifications_read([nid], user_id=1) == 0
    assert db.get_notifications(user_id=1, unread_only=True) == []

    other = N.notify(title="c", body="d", source="s", user_id=1)
    assert db.mark_notifications_read([other], user_id=2) == 0


def test_settling_everything_requires_saying_so(db):
    """Omitting both arguments must not mean "all of them" — a client sending
    an empty selection would silently clear the panel."""
    N.notify(title="a", body="b", source="s", user_id=1)
    assert db.mark_notifications_read() == 0
    assert len(db.get_notifications(unread_only=True)) == 1


def test_before_id_settles_everything_up_to_a_row(db):
    first = N.notify(title="a", body="b", source="s", user_id=1)
    N.notify(title="b", body="b", source="s", user_id=1)
    N.notify(title="c", body="b", source="s", user_id=1)

    assert db.mark_notifications_read(before_id=first + 1, user_id=1) == 2
    assert len(db.get_notifications(user_id=1, unread_only=True)) == 1


# ── the frontend contract ──────────────────────────────────────────────

def test_a_plain_frontend_still_shows_notifications_in_chat(db):
    """The compatibility guarantee. Every frontend written before this kind
    existed keeps working unchanged, because the base flattens the payload and
    sends it down the path it always used."""
    frontend = _Frontend()
    frontend.on_bus_notification_pushed({
        "title": "Plugin registered", "body": "tool_x",
        "source": "plugin_watcher", "level": "success"})

    assert frontend.notifications == []
    assert frontend.messages == ["Plugin registered\n\ntool_x"]


def test_a_declaring_frontend_gets_the_payload_whole(db):
    """Opting in is what buys the structure — the same arrangement
    ``supports_streaming`` makes for deltas."""
    frontend = _Frontend(supports_notifications=True)
    frontend.on_bus_notification_pushed({
        "title": "Plugin registered", "body": "tool_x",
        "source": "plugin_watcher", "level": "success"})

    assert frontend.messages == []
    assert frontend.notifications[0]["source"] == "plugin_watcher"
    assert frontend.notifications[0]["level"] == "success"


def test_the_load_hint_reaches_a_text_frontend_and_not_a_rich_one(db):
    """``load_hint`` is a slash command, which is an affordance for a surface
    that has no better one. A client holding ``conversation_id`` can open the
    conversation itself, so it gets the id and decides for itself."""
    payload = {"title": "Done", "body": "result", "source": "session",
               "conversation_id": 7,
               "load_hint": "/conversations Main 7 'Load conversation'"}

    plain = _Frontend()
    plain.on_bus_notification_pushed(dict(payload))
    assert "/conversations Main 7" in plain.messages[0]

    rich = _Frontend(supports_notifications=True)
    rich.on_bus_notification_pushed(dict(payload))
    assert rich.notifications[0]["conversation_id"] == 7


def test_an_empty_notification_renders_nothing(db):
    frontend = _Frontend()
    frontend.on_bus_notification_pushed({"source": "s"})
    assert frontend.messages == []


def test_a_notification_reaches_only_live_sessions(db):
    """Same ownership filter a push uses: a session no frontend owns has
    nowhere to render, and a background agent's own session is exactly that."""
    frontend = _Frontend()
    frontend.on_bus_notification_pushed(
        {"title": "t", "body": "b", "source": "s", "session_key": "nobody"})
    assert frontend.messages == []


def test_notification_is_a_render_kind_both_halves_agree_on():
    """``KINDS`` is the guest's half and ``RENDER_METHODS`` is the host's. A typo
    in either shows nothing rather than failing, which is why this is a test.

    It used to check only the guest half, because the host's map was a local
    inside ``_adapt_frontend`` and nothing outside could see it — so the exact
    drift the docstring described was the drift it could not catch.
    """
    from sandbox.residency import RENDER_METHODS

    assert "notification" in sandbox_frontends.KINDS
    assert RENDER_METHODS["notification"] == "render_notification"


# ── the split ──────────────────────────────────────────────────────────

def test_the_agents_own_narration_is_not_a_notification(db, captured, pushed):
    """The case a broad definition gets wrong. Mid-turn narration is the model
    speaking inside its own turn — it belongs in the chat of every frontend,
    and a notification panel filling with it would be unreadable."""
    from runtime.conversation_runtime import ConversationRuntime

    runtime = ConversationRuntime.__new__(ConversationRuntime)
    ConversationRuntime.push_message(
        runtime, "s", "Let me check that file.")

    assert [p["message"] for p in pushed] == ["Let me check that file."]
    assert captured == []


def test_a_config_change_is_a_notification(db, captured, pushed):
    """And the counterpart: a setting changing is the system telling you
    something, not the conversation saying it."""
    N.announce_config_change({"keys": ["theme"], "scope": "core"},
                             session_key="s")

    assert pushed == []
    assert captured[0]["source"] == "config"
    assert captured[0]["title"] == "Settings changed"
    assert "theme" in captured[0]["body"]


def test_a_background_turns_answer_is_a_notification(db, captured):
    """A scheduled agent's reply is the agent speaking, but into a chat nobody
    is reading. Delivery is deliberately unset so it reaches whatever surface
    the user is actually at; the child's session travels as origin."""
    N.emit_fallback_push(session_key="spawn_subagent:7", conversation_id=None,
                         title="Nightly index", final_text="Indexed 12 files.",
                         db=db)

    assert captured[0]["title"] == "Nightly index"
    assert captured[0]["body"] == "Indexed 12 files."
    assert captured[0]["source_session_key"] == "spawn_subagent:7"
    assert "session_key" not in captured[0]


# ── attribution ────────────────────────────────────────────────────────

def test_source_is_read_off_the_chain_not_taken_from_the_guest():
    """The property the whole design leans on. A plugin naming its own source
    could claim to be the plugin watcher, and a reader deciding whether to care
    would believe it."""
    from sandbox.handlers.kernel import _notification_source
    from sandbox.policy import Chain
    from sandbox import provenance

    chain = Chain(root="user").push("service_x").push("tool_y")
    with provenance.serving(chain, None, None):
        assert _notification_source() == "tool_y"


def test_source_falls_back_to_the_root_when_there_are_no_links():
    from sandbox.handlers.kernel import _notification_source
    from sandbox.policy import Chain
    from sandbox import provenance

    with provenance.serving(Chain(root="cron:nightly"), None, None):
        assert _notification_source() == "cron:nightly"


def test_source_outside_any_chain_is_generic():
    """No chain means no claim. Naming something specific here would be a
    fabrication, which is worse than being vague."""
    from sandbox.handlers.kernel import _notification_source

    assert _notification_source() == "plugin"


def test_a_guest_cannot_forge_its_source_through_the_real_handler(db, captured):
    """The end-to-end version, driven through ``_session_push`` itself.

    The helper being right in isolation is not the claim worth pinning — the
    claim is that a plugin passing ``source`` in its arguments cannot get it
    onto the payload. This test fails if the handler ever starts reading one.
    """
    from types import SimpleNamespace

    from sandbox.handlers.kernel import _session_push
    from sandbox.policy import Chain
    from sandbox import provenance
    from runtime import notifications as N

    runtime = SimpleNamespace(
        notify=lambda **kw: N.notify(**kw, db=db),
        push_message=lambda *a, **k: None)
    ctx = SimpleNamespace(runtime=runtime, session_key="s",
                          conversation_id=None, user_id=1)

    chain = Chain(root="user").push("tool_memory")
    with provenance.serving(chain, None, None):
        result = _session_push(ctx, {
            "message": "Updated a memory.", "title": "Memory",
            "notify": True, "level": "success",
            "source": "plugin_watcher",   # the forgery attempt
        })

    assert result.ok
    assert captured[0]["source"] == "tool_memory"


def test_a_plain_push_through_the_handler_is_not_a_notification(db, captured,
                                                                pushed):
    """``notify`` is the whole switch, and defaults off — every existing caller
    of ``sdk.session.push`` keeps landing in the chat."""
    from types import SimpleNamespace

    from sandbox.handlers.kernel import _session_push

    sent = []
    runtime = SimpleNamespace(
        push_message=lambda key, text, **kw: sent.append((key, text)),
        notify=lambda **kw: pytest.fail("a plain push must not notify"))
    ctx = SimpleNamespace(runtime=runtime, session_key="s",
                          conversation_id=None, user_id=1)

    assert _session_push(ctx, {"message": "hello"}).ok
    assert sent == [("s", "hello")]
    assert captured == []


# ── the Request surface ────────────────────────────────────────────────

def test_raising_a_notification_did_not_grow_the_vocabulary():
    """``session.push`` grew an argument instead. Pushing text and raising a
    notification are the same act aimed at a different surface."""
    assert "notification.raise" not in R.ALL_TYPES
    assert R.SESSION_PUSH in policy.ALWAYS_SAFE


def test_reading_notifications_is_safe_and_read_only():
    assert R.NOTIFICATION_LIST in policy.ALWAYS_SAFE
    assert R.NOTIFICATION_MARK_READ in policy.ALWAYS_SAFE
    # LIST is in READ_ONLY so a polling panel neither writes a ledger row per
    # tick nor bumps the prompt_cues write counter, which would invalidate every cached
    # agent_prompt in the process.
    assert R.NOTIFICATION_LIST in R.READ_ONLY
    assert R.NOTIFICATION_MARK_READ not in R.READ_ONLY


def test_every_new_type_has_a_handler():
    from sandbox.handlers.kernel import HANDLERS

    assert R.NOTIFICATION_LIST in HANDLERS
    assert R.NOTIFICATION_MARK_READ in HANDLERS
