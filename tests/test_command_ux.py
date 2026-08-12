"""Tests for command-UX polish: parse-error propagation, /services toggles,
config quicklinks, the Used-by map, and the session-conversation banner event.
"""

from types import SimpleNamespace

import state_machine  # noqa: F401  (import-order: break the runtime import cycle)

from events.event_bus import bus
from events.event_channels import (SESSION_CONVERSATION_CHANGED,
                                   SESSION_CONVERSATION_ENDED)
from pipeline.database import Database
from plugins.native.frontend import BaseFrontend, FrontendCapabilities
from runtime.conversation_runtime import ConversationRuntime


# ── Invalid one-shot command args are rendered, not swallowed ────────

class _CaptureFrontend(BaseFrontend):
    name = "capture"
    capabilities = FrontendCapabilities()

    def __init__(self):
        super().__init__()
        self.rendered = []
        self.errors = []

    def render_messages(self, _key, messages):
        self.rendered.extend(messages)

    def render_error(self, _key, error):
        self.errors.append(error)

    def _current_approval_request(self, _key):
        return None


def test_invalid_command_args_render_an_error():
    """Bad arguments are a refusal, and refusals leave the chat.

    The text is unchanged and still names the enum it rejected — what moved is
    the *kind*. A client drawing a conversation could not previously tell this
    apart from the agent speaking, because both arrived as ``messages``.
    """
    fe = _CaptureFrontend()
    fe.commands = SimpleNamespace(parse_args=lambda *a, **k: (_ for _ in ()).throw(
        ValueError("job_name must be one of: fifa_world_cup_daily_update, add.")))

    args, handled = fe._parse_command_args("s", "schedule", "fifa world cup daily update")

    assert args is None
    assert handled is not None and not handled.ok
    assert fe.rendered == []
    assert len(fe.errors) == 1
    assert fe.errors[0]["code"] == "bad_command_args"
    assert "Invalid arguments for `/schedule`" in fe.errors[0]["message"]
    assert "fifa_world_cup_daily_update" in fe.errors[0]["message"]


def test_a_queued_message_says_so_without_the_agent_speaking():
    """The receipt is a notification now, and carries no text on the result.

    It used to ride on ``RuntimeResult.messages`` as "Got it — I'll read that
    as soon as I finish this step" — the agent's voice, first person, on a
    receipt the agent had no part in, mid-turn. ``data["queued"]`` stays,
    because that is what callers branch on; only the sentence is gone.

    ``render_queued_ack`` went with it. Its whole job was suppressing that
    sentence for a frontend that would rather react with an emoji, and there
    is no longer a sentence to suppress — nor was it ever reachable, since it
    is not one of the render kinds a sandboxed frontend receives.
    """
    from runtime.session import RuntimeResult

    fe = _CaptureFrontend()
    fe._render_result("s", RuntimeResult(data={"queued": True}))

    assert fe.rendered == []
    assert not hasattr(fe, "render_queued_ack")


# ── Quicklinks ───────────────────────────────────────────────────────

# ── Used-by map ──────────────────────────────────────────────────────

def test_setting_plugin_names_accumulate_across_declarers():
    from plugins import plugin_discovery as pd

    class A:
        name = "tool_a"
        config_settings = [("Shared", "shared_key_ux_test", "d", 1, {"type": "text"})]

    class B:
        name = "svc_b"
        config_settings = [("Shared", "shared_key_ux_test", "d", 1, {"type": "text"})]

    pd._collect_config_settings(A(), plugin_type="tool")
    pd._collect_config_settings(B(), service_names=["svc_b"], plugin_type="service")
    try:
        assert pd.get_setting_plugin_names("shared_key_ux_test") == ["svc_b", "tool_a"]
    finally:
        pd._setting_to_plugins.pop("shared_key_ux_test", None)
        pd._setting_to_services.pop("shared_key_ux_test", None)


# ── Session conversation banner event ────────────────────────────────

def test_load_conversation_emits_session_conversation_changed(tmp_path):
    db = Database(str(tmp_path / "banner.db"))
    cid = db.create_conversation("FIFA Briefings")
    rt = ConversationRuntime(db=db, services={}, config={})
    seen = []
    unsub = bus.subscribe(SESSION_CONVERSATION_CHANGED, seen.append)
    try:
        rt.load_conversation("s", cid)
    finally:
        unsub()

    assert any(p["session_key"] == "s" and p["conversation_id"] == cid
               and p["title"] == "FIFA Briefings" for p in seen)


# ── The other half: the conversation being left ──────────────────────
#
# CHANGED names the conversation being switched *to*, which is what a banner
# needs and the opposite of what anything treating a conversation as a unit of
# work needs. These pin the three ways a session lets one go, because a
# consumer keyed on this channel has no other way to learn that the work
# finished — and silence is indistinguishable from "still going".

def _ended(tmp_path, name, act):
    db = Database(str(tmp_path / f"{name}.db"))
    cid = db.create_conversation("Ended Conversation")
    rt = ConversationRuntime(db=db, services={}, config={})
    rt.load_conversation("s", cid)
    seen = []
    unsub = bus.subscribe(SESSION_CONVERSATION_ENDED, seen.append)
    try:
        act(rt, cid)
    finally:
        unsub()
    return cid, seen


def test_starting_a_new_conversation_ends_the_old_one(tmp_path):
    cid, seen = _ended(tmp_path, "switch",
                       lambda rt, _cid: rt.reset_conversation("s"))

    assert [p["conversation_id"] for p in seen] == [cid]
    assert seen[0]["reason"] == "switched"
    assert seen[0]["session_key"] == "s"


def test_closing_a_session_ends_the_conversation_it_held(tmp_path):
    cid, seen = _ended(tmp_path, "close",
                       lambda rt, _cid: rt.close_session("s"))

    assert [p["conversation_id"] for p in seen] == [cid]
    assert seen[0]["reason"] == "closed"


def test_deleting_a_conversation_ends_it_for_its_holder(tmp_path):
    """The most final ending there is, and the one with no switch to ride on.

    A consumer waiting for a switch would wait forever: the session is detached
    in place and never moves anywhere.
    """
    cid, seen = _ended(tmp_path, "delete",
                       lambda rt, c: rt.delete_conversation("s", c))

    assert [p["conversation_id"] for p in seen] == [cid]
    assert seen[0]["reason"] == "deleted"


def test_a_session_holding_nothing_ends_nothing(tmp_path):
    """No conversation, no event — an empty session closing is not an episode."""
    db = Database(str(tmp_path / "empty.db"))
    rt = ConversationRuntime(db=db, services={}, config={})
    seen = []
    unsub = bus.subscribe(SESSION_CONVERSATION_ENDED, seen.append)
    try:
        rt.close_session("s")
        rt.reset_conversation("s")
    finally:
        unsub()

    assert seen == []
