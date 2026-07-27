"""The bus, inbound: a sandboxed plugin hearing what the kernel emits.

Publishing was never the hard half — ``sdk.events.emit`` is an ordinary
Request. Receiving is, because a subscription outlives the call that made it,
so the interesting claims are about *lifetime* (a listener dies with its
plugin) and about *what crosses* (a live ``threading.Event`` must not).
"""

import threading
from pathlib import Path

import pytest

import sandbox  # noqa: F401  - installs the ``guest`` package alias
from events.event_bus import bus
from guest.loader import unload_box
from sandbox import Sandbox
from sandbox.bridge import adapt
from sandbox.events import project
from sandbox.validator import validate

LISTENING_SERVICE = '''
"""A migrated service that listens."""

from guest.bases import BaseService


class Ears(BaseService):
    """Remembers everything it was told."""

    name = "ears"
    exports = ["heard"]
    subscribed_channels = ["task_completed", "config_changed"]
    ISOLATION

    def start(self, sdk):
        """Begin with an empty log."""
        self._log = []
        return True

    def on_event(self, sdk, channel, payload):
        """Write it down."""
        self._log.append({"channel": channel, "payload": payload})

    def heard(self, sdk):
        """Everything so far."""
        return list(self._log)
'''
# Dicts rather than tuples on purpose: a subprocess answers over JSON, which
# has no tuple, so a tuple comes back a list and the two runners would need
# different assertions for the same behaviour. The wire shapes the contract.


@pytest.fixture
def box():
    """A sandbox that is torn down even if a test fails."""
    made = Sandbox()
    yield made
    made.shutdown()


@pytest.fixture
def service(tmp_path, box, request):
    """A loaded listening service, unloaded afterwards.

    Unloading matters more here than in most fixtures: a leaked subscription
    would deliver into a dead box for the rest of the session and quietly
    corrupt whatever test ran next.
    """
    isolation = getattr(request, "param", "")
    source = LISTENING_SERVICE.replace(
        "ISOLATION", f'isolation = "{isolation}"' if isolation else "")
    path = tmp_path / "service_ears.py"
    path.write_text(source, encoding="utf-8")

    made = adapt(path).build_services({})["ears"]
    assert made.load() is True
    yield made
    made.unload()
    unload_box("service_ears")


# ──────────────────────────────────────────────────────────────────────
# Delivery.
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("service", ["", "subprocess"], indirect=True)
def test_a_declared_channel_is_delivered(service):
    """The whole point: what the kernel emits reaches the box."""
    bus.emit("task_completed", {"task_name": "embed", "rows_written": 3})

    assert service.heard() == [
        {"channel": "task_completed",
         "payload": {"task_name": "embed", "rows_written": 3}}]


def test_every_declared_channel_is_subscribed(service):
    """Two declarations, two live subscriptions — not just the first."""
    bus.emit("config_changed", {"keys": ["db_path"]})
    bus.emit("task_completed", {"task_name": "index"})

    assert [entry["channel"] for entry in service.heard()] == [
        "config_changed", "task_completed"]


def test_an_undeclared_channel_is_not_delivered(service):
    """A plugin hears what it declared and nothing else.

    The declaration is the security story, so this is the test that would
    catch someone 'helpfully' subscribing to everything.
    """
    bus.emit("task_failed", {"task_name": "embed", "error": "boom"})

    assert service.heard() == []


def test_unloading_stops_delivery(tmp_path, box):
    """The leak with no symptom: a listener outliving its plugin.

    Nothing would raise if this regressed — deliveries would land on a dead
    box and be swallowed forever — so it has to be asserted directly rather
    than noticed.
    """
    path = tmp_path / "service_ears.py"
    path.write_text(LISTENING_SERVICE.replace("ISOLATION", ""),
                    encoding="utf-8")
    made = adapt(path).build_services({})["ears"]
    assert made.load() is True

    bus.emit("task_completed", {"task_name": "before"})
    assert len(made.heard()) == 1

    made.unload()
    unload_box("service_ears")

    # The bus must no longer be holding anything that points at this plugin.
    assert not bus.has_subscribers("task_completed")


def test_a_raising_handler_does_not_reach_the_publisher(tmp_path, box):
    """A subscriber's failure is its own. ``bus.emit`` promises this."""
    source = LISTENING_SERVICE.replace("ISOLATION", "").replace(
        'self._log.append({"channel": channel, "payload": payload})',
        'raise RuntimeError("nope")')
    path = tmp_path / "service_ears.py"
    path.write_text(source, encoding="utf-8")
    made = adapt(path).build_services({})["ears"]
    assert made.load() is True
    try:
        bus.emit("task_completed", {"task_name": "embed"})   # must not raise
    finally:
        made.unload()
        unload_box("service_ears")


# ──────────────────────────────────────────────────────────────────────
# Projection: what is allowed to cross.
# ──────────────────────────────────────────────────────────────────────

def test_a_round_trip_payload_loses_its_machinery(service):
    """``bus.request`` attaches a live Event and a result list.

    Neither can cross a process boundary, and a sandboxed subscriber must not
    be able to answer a round trip it cannot be trusted to finish — a box that
    hung would hang the publisher. So it arrives looking like a plain event.
    """
    reply = threading.Event()
    bus.emit("task_completed", {"task_name": "embed", "reply": reply,
                                "result": [None]})

    entry, = service.heard()
    assert entry["payload"] == {"task_name": "embed"}


def test_projection_drops_what_cannot_serialize():
    """Values are dropped rather than stringified.

    A plugin receiving ``"<Thread(...)>"`` where it expected an object is
    worse off than one receiving nothing, because absence is checkable.
    """
    assert project({"ok": 1, "lock": threading.Lock()}) == {"ok": 1}
    assert project({"nested": {"fine": "yes", "bad": threading.Event()}}) == {
        "nested": {"fine": "yes"}}
    assert project({"items": [1, "two", threading.Lock()]}) == {
        "items": [1, "two"]}
    # An explicit None is a value, not a failure to represent one.
    assert project({"error": None}) == {"error": None}


def test_projection_survives_a_self_referential_payload():
    """Depth is bounded, so a cyclic payload cannot spin."""
    payload = {"name": "loop"}
    payload["self"] = payload

    assert project(payload) is not None      # returns rather than recursing


# ──────────────────────────────────────────────────────────────────────
# The declaration, checked before anything loads.
# ──────────────────────────────────────────────────────────────────────

def _report(body: str, filename: str = "service_x.py"):
    """Validate a plugin class with the given body."""
    return validate(
        '"""A service."""\n\n'
        "from guest.bases import BaseService\n\n\n"
        "class X(BaseService):\n"
        '    """A service."""\n'
        '    name = "x"\n'
        + body, filename=filename)


def test_a_good_declaration_conforms():
    """The shape the template teaches has to pass."""
    report = _report(
        '    subscribed_channels = ["task_completed"]\n\n'
        "    def on_event(self, sdk, channel, payload):\n"
        '        """Handle it."""\n'
        "        return None\n")
    assert report.ok, report.render()


def test_subscribing_without_a_handler_is_refused():
    """Otherwise every delivery reaches the base class and is discarded."""
    report = _report('    subscribed_channels = ["task_completed"]\n')
    assert not report.ok
    assert "on_event" in report.render()


def test_a_tool_cannot_subscribe():
    """A tool is a call that ends; there would be nothing to deliver to."""
    report = validate(
        '"""A tool."""\n\n'
        "from guest.bases import BaseTool\n\n\n"
        "class X(BaseTool):\n"
        '    """A tool."""\n'
        '    name = "x"\n'
        '    subscribed_channels = ["task_completed"]\n\n'
        "    def on_event(self, sdk, channel, payload):\n"
        '        """Handle it."""\n'
        "        return None\n\n"
        "    def run(self, sdk):\n"
        '        """Run."""\n'
        "        return 1\n", filename="tool_x.py")
    assert not report.ok
    assert "does not stay loaded" in report.render()


def test_the_declaration_must_be_literal():
    """It is read by AST, so a computed list is unreadable."""
    report = _report(
        "    subscribed_channels = [c for c in ('a', 'b')]\n\n"
        "    def on_event(self, sdk, channel, payload):\n"
        '        """Handle it."""\n'
        "        return None\n")
    assert not report.ok
    assert "literal list" in report.render()


def test_unknown_channel_names_are_allowed():
    """Plugins own their own channels — see events/event_channels.py.

    An allowlist of kernel channels would refuse exactly the case this
    feature exists for: one plugin listening to another.
    """
    report = _report(
        '    subscribed_channels = ["my_plugin.something_happened"]\n\n'
        "    def on_event(self, sdk, channel, payload):\n"
        '        """Handle it."""\n'
        "        return None\n")
    assert report.ok, report.render()


def test_the_guest_refuses_an_undeclared_delivery():
    """Belt and braces: the host only subscribes to what was declared, but a
    story told in one place is one edit away from not being true."""
    from guest.bases import BaseService

    class Ears(BaseService):
        """Listens."""

        name = "ears"
        subscribed_channels = ["declared"]

        def on_event(self, sdk, channel, payload):
            """Handle it."""
            return "heard"

    assert Ears().__event__(None, "declared", {}) == "heard"
    with pytest.raises(ValueError):
        Ears().__event__(None, "undeclared", {})
