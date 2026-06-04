import threading

from runtime import bootstrap


def test_frontend_unregister_does_not_signal_host_shutdown(monkeypatch):
    host_shutdown = threading.Event()
    seen = {}

    class DemoFrontend:
        name = "demo"

        def __init__(self):
            self.local_shutdown = threading.Event()
            seen["event"] = self.local_shutdown

        def bind(self, *_):
            pass

        def start(self):
            pass

        def stop(self):
            self.local_shutdown.set()

    runtime = type("Runtime", (), {"command_registry": object()})()
    monkeypatch.setattr(bootstrap, "_conversation_runtime", lambda *a: runtime)
    monkeypatch.setattr(bootstrap, "discover_frontends", lambda *_: {"demo": DemoFrontend})
    monkeypatch.setattr(bootstrap.config_manager, "reconcile_plugin_config", lambda *_: None)
    monkeypatch.setattr(bootstrap, "get_plugin_settings", lambda: [])

    runtime, _adapters, _threads = bootstrap.start_frontends(
        {"demo"}, object(), lambda: None, host_shutdown, None, {}, {}, "."
    )
    runtime.frontend_manager.unregister("demo")

    assert seen["event"].is_set()
    assert not host_shutdown.is_set()
