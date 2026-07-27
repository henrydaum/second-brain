"""Regression: stopping the REPL frontend must not shut down the app.

The REPL shares the app-wide shutdown event so it exits with the app, but
stop() — called by the plugin watcher when /update's git pull rewrites
frontend_repl.py on disk — must only stop the frontend instance.
"""

import threading
from pathlib import Path

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
