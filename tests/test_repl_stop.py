"""Regression: stopping the REPL frontend must not shut down the app.

The REPL shares the app-wide shutdown event so it exits with the app, but
stop() — called by the plugin watcher when /update's git pull rewrites
frontend_repl.py on disk — must only stop the frontend instance.
"""

import threading

import state_machine  # noqa: F401  (import-order: break the runtime import cycle)
from plugins.frontends.frontend_repl import ReplFrontend


def test_stop_does_not_set_the_shared_shutdown_event():
    app_shutdown = threading.Event()
    fe = ReplFrontend(shutdown_fn=None, shutdown_event=app_shutdown)

    fe.stop()

    assert not app_shutdown.is_set()  # the app keeps running
    assert fe._stop_event.is_set()    # the loop condition ends this instance


def test_app_shutdown_still_ends_the_loop_condition():
    app_shutdown = threading.Event()
    fe = ReplFrontend(shutdown_fn=None, shutdown_event=app_shutdown)

    app_shutdown.set()

    # Mirrors the start() loop condition.
    assert fe.shutdown_event.is_set() or fe._stop_event.is_set()
