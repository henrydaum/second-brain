import threading

from runtime import bootstrap


def test_telegram_unregister_does_not_signal_host_shutdown(monkeypatch):
    host_shutdown = threading.Event()
    seen = {}

    class Telegram:
        name = "telegram"

        def __init__(self, shutdown_event, services):
            self.shutdown_event = shutdown_event
            seen["event"] = shutdown_event

        def bind(self, *_):
            pass

        def start(self):
            pass

        def stop(self):
            self.shutdown_event.set()

    runtime = type("Runtime", (), {"command_registry": object()})()
    monkeypatch.setattr(bootstrap, "_conversation_runtime", lambda *a: runtime)
    monkeypatch.setattr(bootstrap, "discover_frontends", lambda *_: {"telegram": Telegram})
    monkeypatch.setattr(bootstrap.config_manager, "reconcile_plugin_config", lambda *_: None)
    monkeypatch.setattr(bootstrap, "get_plugin_settings", lambda: [])

    runtime, adapters, _threads = bootstrap.start_frontends({"telegram"}, object(), lambda: None, host_shutdown, None, {}, {}, ".")
    runtime.frontend_manager.unregister("telegram")

    assert seen["event"].is_set()
    assert not host_shutdown.is_set()
