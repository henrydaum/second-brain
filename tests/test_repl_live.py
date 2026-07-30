"""Live subprocess round trip through the migrated REPL frontend."""

import io
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import state_machine  # noqa: F401  (import-order: break runtime cycle)
from pipeline.database import Database
from runtime.context import build_context
from runtime.conversation_runtime import ConversationRuntime
from sandbox import Sandbox
from sandbox.bridge import adapt, configure
from sandbox.console import CONSOLE
from state_machine.conversation import CallableSpec


def test_repl_subprocess_drives_a_live_command(tmp_path, monkeypatch):
    """Console -> guest poll -> SDK -> runtime -> render -> console."""
    db = Database(str(tmp_path / "repl-live.db"))
    spec = CallableSpec("ping", lambda *_: "pong")
    runtime = ConversationRuntime(
        db=db, services={}, config={}, commands={"ping": spec}
    )

    class Registry:
        def all_commands(self):
            return [SimpleNamespace(name="ping")]

        def parse_args(self, *_args, **_kwargs):
            return {}

        def context(self, session_key=None):
            return build_context(
                db, {}, {}, runtime=runtime, session_key=session_key
            )

    sandbox = Sandbox()
    configure(sandbox)
    written = []
    original_claim = CONSOLE.claim

    def claim(token, source=None, writer=None):
        return original_claim(
            token, source=io.StringIO("/ping\n"), writer=written.append
        )

    monkeypatch.setattr(CONSOLE, "claim", claim)
    module = adapt(Path("plugins/frontends/frontend_repl.py").resolve())
    frontend_cls = next(
        value for value in vars(module).values()
        if isinstance(value, type) and getattr(value, "_sandboxed", False)
    )
    frontend = frontend_cls(shutdown_event=threading.Event())
    frontend.bind(runtime, Registry(), {})
    thread = threading.Thread(target=frontend.start, daemon=True)

    try:
        thread.start()
        deadline = time.time() + 5
        while time.time() < deadline and not any(
            "pong" in text for text in written
        ):
            time.sleep(0.01)
        assert any("Second Brain REPL ready" in text for text in written)
        assert any("pong" in text for text in written)
    finally:
        frontend.unbind()
        frontend.stop()
        thread.join(timeout=2)
        sandbox.shutdown()
        configure(None)


# ────────────────────────────────────────────────────────────────────
# Stopping the REPL (was test_repl_stop.py)
# ────────────────────────────────────────────────────────────────────

import state_machine  # noqa: F401  (import-order: break the runtime import cycle)
from sandbox.bridge import adapt


def _frontend(app_shutdown):
    module = adapt(Path("plugins/frontends/frontend_repl.py"))
    cls = next(
        value for value in vars(module).values()
        if isinstance(value, type) and getattr(value, "_sandboxed", False)
    )
    return cls(shutdown_event=app_shutdown)


def test_stop_does_not_set_the_shared_shutdown_event():
    app_shutdown = threading.Event()
    fe = _frontend(app_shutdown)

    fe.stop()

    assert not app_shutdown.is_set()  # the app keeps running
    assert fe._stopping.is_set()      # the adapter loop ends this instance


def test_app_shutdown_still_ends_the_loop_condition():
    app_shutdown = threading.Event()
    fe = _frontend(app_shutdown)

    app_shutdown.set()

    # Mirrors the start() loop condition.
    assert fe._done()
