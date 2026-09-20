"""Widgets: one HTML document, bound to a conversation.

A widget is the one thing the kernel routes but never loads — it runs in a
browser, in a frame with no credential, and everything here is about *which*
file rather than what it does. Four things are worth pinning, and every one of
them fails silently if it breaks.

**The binding is on the conversation row**, so it survives a restart and
follows the conversation between sessions. The failure it prevents is a panel
that empties itself every morning.

**Own conversation is SAFE, a named one is not.** The decision rests on the
argument being *absent* rather than on comparing it to a context ``classify``
does not hold, so the handler's default and the policy's branch agree by
construction. A regression here is a dialog per widget swap (which teaches
somebody to stop reading dialogs) or no dialog for writing into somebody
else's conversation.

**The capability gate is on the bus handler**, not on ``render_widget``, which
``residency.RENDER_METHODS`` replaces wholesale for every sandboxed frontend —
so a check written in the default implementation would never run.

**The state is capped.** The column has no paging, and a widget with more to
keep has a filesystem.
"""

# Import the state_machine package before runtime modules to settle the
# package-init circular import.
import state_machine  # noqa: F401

import pytest

from pipeline.database import Database
from plugins.native.frontend import BaseFrontend, FrontendCapabilities
from sandbox import policy
from sandbox.guest import requests as R
from sandbox.guest.requests import Request
from sandbox.handlers import kernel as H
from sandbox.policy import Chain


@pytest.fixture
def db(tmp_path):
    return Database(str(tmp_path / "widgets.db"))


class _Ctx:
    """Just enough context for the widget handlers."""

    def __init__(self, runtime=None, session_key="s"):
        self.runtime = runtime
        self.session_key = session_key
        self.db = getattr(runtime, "db", None)
        self.user_id = 1


class _Session:
    def __init__(self, key, conversation_id):
        self.key = key
        self.conversation_id = conversation_id
        self.pending_widget = None
        self.pending_widget_states = {}


class _Runtime:
    """The narrow slice of ConversationRuntime the handlers reach for."""

    def __init__(self, db, conversation_id=1, allow=True):
        self.db = db
        self.allow = allow
        self.sessions = {"s": _Session("s", conversation_id)}
        self.announced: list[int] = []

    def assert_conversation_access(self, _key, _cid, override=False):
        return self.allow

    def conversation_widget(self, cid):
        from runtime.conversation_runtime import ConversationRuntime
        return ConversationRuntime.conversation_widget(self, cid)

    def announce_widget(self, cid):
        self.announced.append(cid)

    def set_conversation_widget(self, key, cid, name, override=False):
        from runtime.conversation_runtime import ConversationRuntime
        return ConversationRuntime.set_conversation_widget(
            self, key, cid, name, override=override)

    def set_conversation_widget_state(self, key, cid, state, override=False, *, widget_name=None):
        from runtime.conversation_runtime import ConversationRuntime
        return ConversationRuntime.set_conversation_widget_state(
            self, key, cid, state, override=override, widget_name=widget_name)

    def session_user_id(self, _key):
        return 1


# ── the row ────────────────────────────────────────────────────────────

def test_a_conversation_starts_with_no_widget(db):
    """NULL is legal and ordinary. Most conversations never hold one, so a
    reader must not have to treat absence as a failure."""
    cid = db.create_conversation("Main")

    assert db.get_conversation(cid)["widget"] is None


def test_unbinding_preserves_saved_state(db):
    cid = db.create_conversation("Main")
    db.set_conversation_widget(cid, "clock")
    db.set_conversation_widget_state(cid, '{"tz": "UTC"}')

    db.set_conversation_widget(cid, None)

    row = db.get_conversation(cid)
    assert row["widget"] is None
    assert db.get_conversation_widget_state(cid, "clock") == '{"tz": "UTC"}'


def test_a_listing_never_carries_the_state_blob(db):
    """The sidebar reads pages of these and looks inside none of them. This is
    the ``conv.read`` lesson applied before it costs anything."""
    cid = db.create_conversation("Main")
    db.set_conversation_widget(cid, "clock")
    db.set_conversation_widget_state(cid, '{"big": "x"}')

    listed = db.list_conversations()[0]
    paged, _ = db.list_conversations_page()

    assert "widget_state" not in listed and listed["widget"] == "clock"
    assert "widget_state" not in paged[0]
    # State is read separately, using both parts of its key.
    assert db.get_conversation_widget_state(cid, "clock") == '{"big": "x"}'


# ── policy ─────────────────────────────────────────────────────────────

def test_arranging_your_own_conversation_needs_no_approval():
    """The capability is already free: any loaded plugin puts arbitrary text in
    front of the person every turn with nobody asked, and a widget runs in an
    opaque origin with less reach than that."""
    for kind in (R.WIDGET_SET, R.WIDGET_STATE_SET):
        decision = policy.classify(Request(kind, {"name": "clock"}), Chain())
        assert decision.safe, kind


def test_naming_another_conversation_is_asked_about():
    """It may belong to another user — the same question
    ``session.add_prompt_extra`` asks about a session key."""
    for kind in (R.WIDGET_SET, R.WIDGET_STATE_SET):
        decision = policy.classify(
            Request(kind, {"name": "clock", "conversation_id": 9}), Chain())
        assert not decision.safe, kind
        assert "9" in decision.reason


def test_reading_widgets_is_always_safe_and_counts_as_a_read():
    """Listing what exists and asking what is showing change nothing. Both are
    in READ_ONLY so the ledger's sink drops them — a panel polls."""
    assert R.WIDGET_LIST in policy.ALWAYS_SAFE
    assert R.WIDGET_GET in policy.ALWAYS_SAFE
    assert {R.WIDGET_LIST, R.WIDGET_GET} <= R.READ_ONLY
    # And the writes are not, or a swap would never be recorded.
    assert R.WIDGET_SET not in R.READ_ONLY


# ── the handlers ───────────────────────────────────────────────────────

def test_an_absent_conversation_id_means_the_one_you_are_in(db, monkeypatch):
    """The handler's default and the policy branch have to agree, and this is
    the half that makes "no argument" mean anything at all. Reading it off the
    *context* would answer None for every caller, because a context carries no
    conversation — it is the session that binds the two."""
    cid = db.create_conversation("Main")
    db.set_conversation_widget(cid, "clock")
    runtime = _Runtime(db, conversation_id=cid)
    monkeypatch.setattr(H, "widget_named",
                        lambda name: {"name": name, "path": "/w.html",
                                      "tree": "bundled"})

    result = H._widget_get(_Ctx(runtime), {})

    assert result.data["name"] == "clock"


def test_a_caller_with_no_session_at_all_says_so(db):
    """Rather than acting on nothing and reporting success. This is a service
    polling on its own initiative or a background driver: it has no
    conversation to mean and none coming, which is what separates it from the
    unbound session below."""
    runtime = _Runtime(db, conversation_id=None)
    runtime.sessions = {}

    result = H._widget_get(_Ctx(runtime), {})

    assert not result.ok and result.code == "not_found"


# ── before the first message ───────────────────────────────────────────

def test_a_session_with_no_conversation_yet_holds_the_pick_itself(db, monkeypatch):
    """Pressing "new chat" leaves a session with no row to write a binding
    onto, because a conversation is created by the first *message*. Failing
    there made the panel a dead end for the whole gap; the session holds the
    pick instead, and answers reads from it in the row's own shape so nothing
    downstream can tell which side of the first message it is on."""
    runtime = _Runtime(db, conversation_id=None)
    monkeypatch.setattr(H, "widget_named",
                        lambda name: {"name": name, "path": "/w.html",
                                      "tree": "bundled"})

    assert H._widget_set(_Ctx(runtime), {"name": "clock"}).ok

    binding = H._widget_get(_Ctx(runtime), {}).data
    assert binding == {"name": "clock", "state": None, "path": "/w.html",
                       "tree": "bundled", "installed": True,
                       "conversation_id": None}
    # And nothing was created to hold it. A blank conversation per opened panel
    # is the litter this arrangement exists to avoid, and is the whole reason
    # the slot is on the session rather than a row made on demand.
    assert db.list_conversations() == []


def test_state_saved_before_there_was_a_conversation_is_kept(db, monkeypatch):
    """The widget frame has no same-origin credential, so ``widget.state_set``
    is not that document's preferred storage — it is its only storage. A
    pre-conversation save that failed would lose a game played before the
    person typed anything, with nowhere else it could have gone."""
    runtime = _Runtime(db, conversation_id=None)
    monkeypatch.setattr(H, "widget_named",
                        lambda name: {"name": name, "path": "/w.html",
                                      "tree": "bundled"})

    H._widget_set(_Ctx(runtime), {"name": "2048"})
    assert H._widget_state_set(_Ctx(runtime), {"value": {"score": 12}}).ok

    assert H._widget_get(_Ctx(runtime), {}).data["state"] == '{"score": 12}'


def test_the_pending_state_is_capped_like_the_column(db, monkeypatch):
    """The cap is about what a mount reads back, which is the same question
    whether the value is on the row or on its way there. Checking it only on
    the row would let a session accumulate megabytes that the first message
    then refuses to store."""
    runtime = _Runtime(db, conversation_id=None)

    result = H._widget_state_set(
        _Ctx(runtime), {"value": "x" * (H.WIDGET_STATE_MAX + 1)})

    assert not result.ok and result.code == "too_large"
    assert runtime.sessions["s"].pending_widget_states == {}


def test_unbinding_before_a_conversation_preserves_state(db, monkeypatch):
    runtime = _Runtime(db, conversation_id=None)
    monkeypatch.setattr(H, "widget_named",
                        lambda name: {"name": name, "path": "/w.html",
                                      "tree": "bundled"})

    H._widget_set(_Ctx(runtime), {"name": "2048"})
    H._widget_state_set(_Ctx(runtime), {"value": {"score": 12}})
    H._widget_set(_Ctx(runtime), {"name": None})

    session = runtime.sessions["s"]
    assert session.pending_widget is None
    assert session.pending_widget_states == {"2048": '{"score": 12}'}


def test_an_unknown_widget_is_refused_before_there_is_a_conversation_too(db, monkeypatch):
    """The typo check must not be something only the row-backed path does, or
    the pending slot becomes the way to store a name nobody has."""
    runtime = _Runtime(db, conversation_id=None)
    monkeypatch.setattr(H, "widget_named", lambda name: None)

    result = H._widget_set(_Ctx(runtime), {"name": "nope"})

    assert not result.ok and result.code == "not_found"
    assert runtime.sessions["s"].pending_widget is None


def test_binding_an_unknown_widget_fails_rather_than_being_stored(db, monkeypatch):
    """A typo would otherwise present as an empty panel with nothing anywhere
    explaining it."""
    cid = db.create_conversation("Main")
    runtime = _Runtime(db, conversation_id=cid)
    monkeypatch.setattr(H, "widget_named", lambda name: None)

    result = H._widget_set(_Ctx(runtime), {"name": "nope"})

    assert not result.ok and result.code == "not_found"
    assert db.get_conversation(cid)["widget"] is None


def test_a_binding_survives_its_file_going_missing(db, monkeypatch):
    """Uninstalling a package or renaming a workspace file must not silently
    clear the conversation — a reinstall should find it still pointing there.
    ``installed`` is what lets a client say so."""
    cid = db.create_conversation("Main")
    db.set_conversation_widget(cid, "clock")
    runtime = _Runtime(db, conversation_id=cid)
    monkeypatch.setattr(H, "widget_named", lambda name: None)

    result = H._widget_get(_Ctx(runtime), {})

    assert result.data["name"] == "clock"
    assert result.data["installed"] is False and result.data["path"] == ""


def test_oversized_state_is_refused_and_says_where_to_put_it(db, monkeypatch):
    """The column has no paging. A widget with more to keep has a filesystem,
    and the refusal has to say so or it is only a wall."""
    cid = db.create_conversation("Main")
    runtime = _Runtime(db, conversation_id=cid)

    result = H._widget_state_set(
        _Ctx(runtime), {"value": {"blob": "x" * (H.WIDGET_STATE_MAX + 1)}})

    assert not result.ok and result.code == "too_large"
    assert "file" in result.error
    assert db.get_conversation(cid)["widget_state"] is None


def test_state_that_will_not_serialize_is_the_guest_s_mistake(db):
    """Reported as an invalid argument rather than as a kernel fault, because
    the guest supplied the value."""
    cid = db.create_conversation("Main")
    runtime = _Runtime(db, conversation_id=cid)

    result = H._widget_state_set(_Ctx(runtime), {"value": {"o": object()}})

    assert not result.ok and result.code == "invalid_argument"


def test_setting_a_widget_announces_and_storing_state_does_not(db, monkeypatch):
    """Binding changes what the person is looking at, so every live session on
    that conversation is told. Storing state must *not* re-render: the only
    thing that writes state is the widget itself, which already holds it, and
    handing a live document its own state back reloads it — the one thing a
    persistence mechanism must never do."""
    cid = db.create_conversation("Main")
    runtime = _Runtime(db, conversation_id=cid)
    monkeypatch.setattr(H, "widget_named",
                        lambda name: {"name": name, "path": "/w.html",
                                      "tree": "bundled"})

    H._widget_set(_Ctx(runtime), {"name": "clock"})
    assert runtime.announced == [cid]

    H._widget_state_set(_Ctx(runtime), {"value": {"tz": "UTC"}})
    assert runtime.announced == [cid]


def test_a_conversation_someone_else_owns_is_refused(db, monkeypatch):
    """The dialog decides whether to *ask*; this is what happens after a yes
    from somebody who still does not own the row."""
    cid = db.create_conversation("Main")
    runtime = _Runtime(db, conversation_id=cid, allow=False)
    monkeypatch.setattr(H, "widget_named",
                        lambda name: {"name": name, "path": "/w.html",
                                      "tree": "bundled"})

    result = H._widget_set(_Ctx(runtime), {"name": "clock",
                                           "conversation_id": cid})

    assert not result.ok


# ── the render kind ────────────────────────────────────────────────────

class _Frontend(BaseFrontend):
    name = "cap"
    capabilities = FrontendCapabilities()

    def __init__(self, **caps):
        super().__init__()
        if caps:
            self.capabilities = FrontendCapabilities(**caps)
        self.widgets: list[dict] = []
        self.catalog: list[tuple] = []
        self.messages: list[str] = []

    def render_widget(self, _key, info):
        self.widgets.append(info)

    def render_widget_catalog(self, key, change):
        self.catalog.append((key, change))

    def render_messages(self, _key, messages):
        self.messages.extend(messages)

    def _live_session_keys(self):
        return ["s"]


def test_a_frontend_that_cannot_draw_a_widget_is_told_nothing():
    """The opposite of the notification fallback, and deliberately. There is no
    markdown a widget flattens into, so a transport that cannot draw one gets
    silence rather than a description of a thing it cannot show."""
    frontend = _Frontend()

    frontend.on_bus_session_widget_changed(
        {"session_key": "s", "conversation_id": 1, "name": "clock"})

    assert frontend.widgets == [] and frontend.messages == []


def test_declaring_the_capability_is_what_delivers_it():
    """And the gate lives on the bus handler, because ``RENDER_METHODS``
    replaces ``render_widget`` wholesale on every sandboxed frontend — a check
    inside the default implementation would never run for any of them."""
    frontend = _Frontend(supports_widgets=True)

    frontend.on_bus_session_widget_changed(
        {"session_key": "s", "conversation_id": 1, "name": "clock",
         "state": '{"tz": "UTC"}'})

    assert frontend.widgets[0]["name"] == "clock"
    assert frontend.widgets[0]["state"] == '{"tz": "UTC"}'


# ── the catalog ────────────────────────────────────────────────────────

def test_an_edited_widget_is_announced_under_the_name_the_listing_uses(tmp_path,
                                                                      monkeypatch):
    """The watcher sees ``widget_clock.html``; every client holds a binding
    under ``clock``. Emitting the stem would compare equal to nothing on the
    other side, and it would do it silently."""
    from events.event_bus import bus
    from events.event_channels import WIDGET_CATALOG_CHANGED
    from plugins.plugin_watcher import PluginWatcher

    directory = tmp_path / "workspace" / "widgets"
    directory.mkdir(parents=True)
    path = directory / "widget_clock.html"
    path.write_text("<main>tick</main>", encoding="utf-8")

    seen: list[dict] = []
    unsubscribe = bus.subscribe(WIDGET_CATALOG_CHANGED, seen.append)
    try:
        watcher = PluginWatcher({})
        monkeypatch.setattr(watcher, "_notify", lambda *a, **k: None)
        # Which root a path sits in is the layout's question and has its own
        # tests; this one is about what the announcement says once it is
        # answered, and a temporary directory is inside no tree.
        monkeypatch.setattr(PluginWatcher, "_root_of",
                            staticmethod(lambda _p: "widgets"))

        assert watcher.register(path, edited=True)["ok"]
        watcher.unregister(path)
    finally:
        unsubscribe()

    assert [(e["action"], e["name"]) for e in seen] == [
        ("reloaded", "clock"), ("removed", "clock")]
    assert seen[0]["path"] == str(path.resolve())
    # Outside every tree, which a temporary directory is: said as unknown
    # rather than guessed at.
    assert seen[0]["tree"] == ""


def test_a_frontend_that_cannot_draw_a_widget_is_not_told_about_the_catalog():
    """Same gate as the binding, and on the bus handler for the same reason —
    a check inside the default method never runs for a sandboxed frontend."""
    change = {"action": "registered", "name": "clock",
              "path": "/w/widget_clock.html", "tree": "workspace"}

    blind = _Frontend()
    blind.on_bus_widget_catalog_changed(dict(change))
    assert blind.catalog == []

    drawing = _Frontend(supports_widgets=True)
    drawing.on_bus_widget_catalog_changed(dict(change))
    assert drawing.catalog == [("s", change)]


def test_the_catalog_announcement_carries_no_listing():
    """``widget.list`` stays the one answer to what exists. A copy of it
    travelling on a bus is a copy that can be wrong — and a client that trusted
    it would be showing a listing nothing rebuilt."""
    frontend = _Frontend(supports_widgets=True)

    frontend.on_bus_widget_catalog_changed(
        {"action": "reloaded", "name": "clock", "path": "/w/widget_clock.html",
         "tree": "bundled"})

    assert set(frontend.catalog[0][1]) == {"action", "name", "path", "tree"}


def test_the_kind_and_its_method_name_each_other():
    """A kind in one half and not the other shows a person nothing and raises
    nothing."""
    from sandbox.frontends import KINDS
    from sandbox.residency import RENDER_METHODS

    assert "widget" in KINDS
    assert RENDER_METHODS["widget"] == "render_widget"
    assert hasattr(BaseFrontend, "render_widget")
    assert "widget_catalog" in KINDS
    assert RENDER_METHODS["widget_catalog"] == "render_widget_catalog"
    assert hasattr(BaseFrontend, "render_widget_catalog")


# ── the prompt ─────────────────────────────────────────────────────────

def test_the_guidance_follows_the_declaration_not_the_transport_name():
    """It used to ask whether this machine was serving its own UI, which tested
    the *server* rather than the client. A frontend states what it draws."""
    from agent.system_prompt import _widgets

    class _Caps:
        supports_widgets = True

    class _Yes:
        capabilities = _Caps()

    assert "Widgets" in _widgets(_Yes())
    assert _widgets(None) == ""


def test_the_active_widget_is_named_only_while_one_is_bound(db):
    """Dynamic rather than semi-stable: one session walks through many
    conversations, so a name in the cacheable prefix goes stale the moment
    somebody loads another one. And silence when there is none, because a line
    saying "no widget" would be noise on nearly every turn."""
    from agent.system_prompt import PromptContext, _active_widget

    cid = db.create_conversation("Main")
    ctx = PromptContext(db=db, conversation_id=cid)
    assert _active_widget(ctx) == ""

    db.set_conversation_widget(cid, "clock")
    assert "clock" in _active_widget(ctx)


# ── isolation: the panel follows the conversation ──────────────────────
#
# Every test here is about what the *client* is told, because the panel has no
# other way to find out. A widget that stays on screen after its conversation
# has gone looks exactly like one that was deliberately kept, and the person is
# then playing with a document bound to nothing.

@pytest.fixture
def live(db):
    """A real runtime, a real bus, and whatever the session was told."""
    from events.event_bus import bus
    from events.event_channels import SESSION_WIDGET_CHANGED
    from tests.support import plain_runtime

    runtime = plain_runtime(db, config={"llm_profiles": {"m": {"backend": "x"}}})
    runtime.get_session("repl")
    runtime.active_session_key = "repl"
    told: list[dict] = []
    unsubscribe = bus.subscribe(SESSION_WIDGET_CHANGED, told.append)
    try:
        yield runtime, told
    finally:
        unsubscribe() if callable(unsubscribe) else bus.unsubscribe(
            SESSION_WIDGET_CHANGED, told.append)


def test_starting_a_new_chat_says_the_panel_is_empty(live, db, monkeypatch):
    """The failure that started this: nothing announced the *absence*.

    ``announce_widget`` is keyed by conversation and answered None for "no
    conversation", so leaving one told the client nothing at all and the panel
    went on drawing the widget it had. Worse than a cosmetic stale frame: the
    person then plays with a document the kernel has no binding for, and their
    next message adopts an empty slot and wipes it.
    """
    monkeypatch.setattr(H, "widget_named",
                        lambda name: {"name": name, "path": "/w.html",
                                      "tree": "bundled"})
    runtime, told = live
    cid = db.create_conversation("Main")
    runtime.load_conversation("repl", cid)
    runtime.set_conversation_widget("repl", cid, "2048")
    told.clear()

    runtime.new_conversation("repl")

    assert told, "leaving a conversation has to announce the empty panel"
    assert told[-1]["name"] is None
    assert told[-1]["session_key"] == "repl"


def test_switching_conversations_replaces_the_widget_and_its_state(live, db, monkeypatch):
    """Two conversations, the same widget, different saved games. The name
    alone is not the identity — a client keying on it would keep the first
    board while claiming to show the second."""
    monkeypatch.setattr(H, "widget_named",
                        lambda name: {"name": name, "path": "/w.html",
                                      "tree": "bundled"})
    runtime, told = live
    first = db.create_conversation("First")
    second = db.create_conversation("Second")
    for cid, score in ((first, 12), (second, 99)):
        db.set_conversation_widget(cid, "2048")
        db.set_conversation_widget_state(cid, '{"score": %d}' % score)

    runtime.load_conversation("repl", first)
    runtime.load_history("repl", second)

    assert told[-1]["name"] == "2048"
    assert told[-1]["state"] == '{"score": 99}'
    assert told[-1]["conversation_id"] == second


def test_a_pick_made_before_the_first_message_is_announced(live, db, monkeypatch):
    """The pending slot is only useful if the client hears about it. Without
    this the panel's own optimistic update is the only record, so a second
    window — or a re-read — disagrees with what the person is looking at."""
    monkeypatch.setattr(H, "widget_named",
                        lambda name: {"name": name, "path": "/w.html",
                                      "tree": "bundled"})
    runtime, told = live
    told.clear()

    H._widget_set(_Ctx(runtime, "repl"), {"name": "2048"})

    assert told[-1]["name"] == "2048"
    assert told[-1]["conversation_id"] is None


def test_deleting_the_open_conversation_empties_the_panel(live, db, monkeypatch):
    """The other way a session loses its conversation. It detaches to None in
    place rather than being rebuilt, so it is the one path where a stale
    binding could survive on the session as well as on the screen."""
    monkeypatch.setattr(H, "widget_named",
                        lambda name: {"name": name, "path": "/w.html",
                                      "tree": "bundled"})
    runtime, told = live
    cid = db.create_conversation("Main")
    runtime.load_conversation("repl", cid)
    runtime.set_conversation_widget("repl", cid, "2048")
    told.clear()

    runtime.delete_conversation("repl", cid)

    assert told and told[-1]["name"] is None


def test_widget_state_migration_preserves_existing_rows_and_does_not_resurrect(tmp_path):
    path = str(tmp_path / "migration.db")
    original = Database(path)
    cid = original.create_conversation("Legacy")
    original.conn.execute("DROP TABLE conversation_widget_states")
    original.conn.execute(
        "UPDATE conversations SET widget = 'clock', widget_state = ? WHERE id = ?",
        ('{"tz": "UTC"}', cid))
    original.conn.commit()
    original.conn.close()

    migrated = Database(path)
    assert migrated.get_conversation_widget_state(cid, "clock") == '{"tz": "UTC"}'
    assert migrated.get_conversation(cid)["widget_state"] is None
    # A pre-existing new-table row wins, including an explicitly cleared value.
    migrated.set_conversation_widget_state(cid, None, widget_name="clock")
    migrated.conn.execute("UPDATE conversations SET widget_state = ? WHERE id = ?",
                          ('{"stale": true}', cid))
    migrated.conn.commit()
    migrated.conn.close()

    for _ in range(2):
        reopened = Database(path)
        assert reopened.get_conversation_widget_state(cid, "clock") is None
        assert reopened.get_conversation(cid)["widget_state"] is None
        reopened.conn.close()


@pytest.mark.parametrize("pending", [False, True])
def test_switching_widgets_and_late_saves_keep_separate_state(db, monkeypatch, pending):
    cid = None if pending else db.create_conversation("Main")
    runtime = _Runtime(db, conversation_id=cid)
    ctx = _Ctx(runtime)
    monkeypatch.setattr(H, "widget_named", lambda name: {"name": name, "path": f"/{name}.html"})
    for name, value in [("a", 1), ("b", 2)]:
        assert H._widget_set(ctx, {"name": name}).ok
        assert H._widget_get(ctx, {}).data["state"] is None
        assert H._widget_state_set(ctx, {"value": value}).ok
    # A delayed request from A must not write into the currently selected B.
    assert H._widget_state_set(ctx, {"name": "a", "value": 3}).ok
    assert H._widget_get(ctx, {}).data["state"] == "2"
    assert H._widget_set(ctx, {"name": None}).ok
    for name, expected in [("a", "3"), ("b", "2")]:
        assert H._widget_set(ctx, {"name": name}).ok
        assert H._widget_get(ctx, {}).data["state"] == expected


def test_conversation_state_is_independent_and_deleted_with_its_conversation(db):
    first, second = [db.create_conversation(name) for name in ("First", "Second")]
    for cid, state in [(first, "1"), (second, "2")]:
        db.set_conversation_widget_state(cid, state, widget_name="clock")
    assert db.get_conversation_widget_state(first, "clock") == "1"
    assert db.get_conversation_widget_state(second, "clock") == "2"
    db.delete_conversation(first)
    assert db.get_conversation_widget_state(first, "clock") is None
    assert db.get_conversation_widget_state(second, "clock") == "2"


def test_state_write_respects_database_owner_scope(db):
    cid = db.create_conversation("Owned")
    owner = db.get_conversation(cid)["user_id"]
    db.set_conversation_widget_state(cid, "1", user_id=owner + 1, widget_name="clock")
    assert db.get_conversation_widget_state(cid, "clock") is None
