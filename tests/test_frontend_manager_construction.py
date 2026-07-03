"""Tests for FrontendManager's plugin-agnostic construction.

The kernel must not carry frontend-specific wiring (a store frontend like
Telegram is a plugin): instead, a frontend requests host resources by naming
them as constructor parameters, and ``register`` supplies whatever the
signature asks for from ``host_kwargs``.
"""

import threading

# Import the state_machine package before runtime modules to settle the
# package-init circular import (state_machine/__init__ pulls in the runtime).
import state_machine  # noqa: F401

from runtime.bootstrap import FrontendManager


class _Base:
    """Minimal frontend stand-in: bind/start so register() completes."""

    def bind(self, runtime, commands, config=None):
        self.bound = True

    def start(self):
        return


def _manager():
    m = FrontendManager(runtime=None, command_registry=None, config={})
    m.host_kwargs = {
        "shutdown_fn": lambda: "the-shutdown-fn",
        "shutdown_event": lambda: threading.Event(),
        "services": lambda: {"llm": "router"},
    }
    return m


def test_ctor_params_are_matched_by_name():
    class TelegramShaped(_Base):
        name = "tgish"

        def __init__(self, shutdown_event: threading.Event | None = None, services: dict | None = None):
            self.shutdown_event = shutdown_event
            self.services = services

    m = _manager()
    assert m.register(TelegramShaped) is None
    adapter = m.adapters["tgish"]
    assert isinstance(adapter.shutdown_event, threading.Event)
    assert adapter.services == {"llm": "router"}


def test_zero_arg_frontend_still_constructs():
    class Plain(_Base):
        name = "plain"

        def __init__(self):
            self.ok = True

    m = _manager()
    assert m.register(Plain) is None
    assert m.adapters["plain"].ok


def test_unknown_params_are_left_to_their_defaults():
    class Picky(_Base):
        name = "picky"

        def __init__(self, services=None, favorite_color="blue"):
            self.services = services
            self.favorite_color = favorite_color

    m = _manager()
    assert m.register(Picky) is None
    adapter = m.adapters["picky"]
    assert adapter.services == {"llm": "router"}
    assert adapter.favorite_color == "blue"


def test_each_instance_gets_a_fresh_shutdown_event():
    class A(_Base):
        name = "a"

        def __init__(self, shutdown_event=None):
            self.shutdown_event = shutdown_event

    class B(A):
        name = "b"

    m = _manager()
    m.register(A)
    m.register(B)
    ev_a, ev_b = m.adapters["a"].shutdown_event, m.adapters["b"].shutdown_event
    assert ev_a is not ev_b
    ev_a.set()
    assert not ev_b.is_set()


def test_explicit_factory_still_wins():
    class Special(_Base):
        name = "special"

        def __init__(self, marker=None):
            self.marker = marker

    m = _manager()
    m.set_factory("special", lambda cls: cls(marker="from-factory"))
    assert m.register(Special) is None
    assert m.adapters["special"].marker == "from-factory"
