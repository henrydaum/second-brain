import threading
import unittest
from types import SimpleNamespace

from event_bus import bus
from frontend.runtime import FrontendRuntime
from frontend.types import FrontendEvent, FrontendSession, PlatformCapabilities


class DummyCtrl:
    class DB:
        def list_user_conversations(self, limit=10):
            return [{"id": 7, "title": "Saved chat", "updated_at": 0}]

    db = DB()
    orchestrator = None
    tool_registry = None
    services = {}

    def maybe_generate_conversation_title_async(self, conversation_id):
        return None


class DummyApproval:
    def __init__(self, req_id="req-1"):
        self.id = req_id
        self.metadata = {}
        self.is_resolved = False
        self.value = None

    def resolve(self, approved):
        self.is_resolved = True
        self.value = approved


class DummyAdapter:
    def __init__(self, name="demo", default_session=None):
        self.name = name
        self.capabilities = PlatformCapabilities(supports_proactive_push=True, supports_buttons=True)
        self._default_session = default_session
        self.sent = []
        self.runtime = None

    def bind_runtime(self, runtime):
        self.runtime = runtime

    def default_session(self):
        return self._default_session

    def send_action(self, session, action):
        self.sent.append((session, action))


class FrontendRuntimeTests(unittest.TestCase):
    def setUp(self):
        self._saved_subs = bus._subs
        bus._subs = {}
        self.runtime = FrontendRuntime(DummyCtrl(), {}, {}, None, None)
        self.registry = self.runtime.create_registry(FrontendSession("demo", "u", "c"))

    def tearDown(self):
        bus._subs = self._saved_subs

    def test_slash_command_event_dispatch(self):
        self.registry.register(SimpleNamespace(name="ping", handler=lambda _arg: "pong"))
        result = self.runtime.handle_frontend_event(
            FrontendEvent(type="slash_command", session=FrontendSession("demo", "u", "c"), text="/ping"),
            self.registry,
        )
        self.assertEqual(result.text, "pong")

    def test_callback_history_routes_to_command(self):
        self.registry.register(SimpleNamespace(name="history", handler=lambda arg: f"loaded:{arg}"))
        result = self.runtime.handle_frontend_event(
            FrontendEvent(
                type="callback_response",
                session=FrontendSession("demo", "u", "c"),
                payload={"kind": "history", "conversation_id": "42"},
            ),
            self.registry,
        )
        self.assertEqual(result.text, "loaded:42")

    def test_approval_response_resolves_pending_request(self):
        req = DummyApproval()
        self.runtime._pending_approvals["demo"] = {req.id: req}
        self.runtime._pending_approval_order["demo"] = [req.id]
        result = self.runtime.handle_frontend_event(
            FrontendEvent(
                type="approval_response",
                session=FrontendSession("demo", "u", "c"),
                payload={"approved": True},
            ),
            self.registry,
        )
        self.assertTrue(req.is_resolved)
        self.assertTrue(req.value)
        self.assertEqual(result.text, "Approval granted.")

    def test_chat_message_pushed_prefers_last_active_session(self):
        adapter = DummyAdapter(default_session=FrontendSession("demo", "default", "default"))
        self.runtime.register_adapter(adapter)
        active_session = FrontendSession("demo", "active", "active")
        self.runtime.get_state(active_session)
        self.runtime._on_chat_message_pushed({"title": "Reminder", "message": "Ping"})
        self.assertEqual(adapter.sent[-1][0].chat_id, "active")
        self.assertIn("Reminder", adapter.sent[-1][1].text)


if __name__ == "__main__":
    unittest.main()
