"""A sandboxed hook inside a real driven turn.

This is the join nothing previously made. ``tests/test_hooks_moments.py``
drives a real loop with native callables; ``tests/test_sandbox_hooks.py`` loads
a real boxed service and then calls ``HookRegistry`` methods by hand. Between
them a sandboxed hook had never once been visited by a turn.

The central claim is one line: **a boxed probe writes the same journal a
native one does.** If that holds, every other hook test in the suite may keep
using native callables and still be about sandboxed plugins, because the
projection layer in ``sandbox/hooks.py`` has been shown faithful end to end.
That is also why this file is small — boxes are the expensive thing in the
suite, so it buys the equivalence once and the fast files spend it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

# Import the state_machine package before runtime modules to settle the
# package-init circular import.
import state_machine  # noqa: F401

import sandbox  # noqa: F401  - installs the bare ``guest`` package alias
from guest.loader import unload_box
from sandbox import Sandbox
from sandbox.bridge import configure
from tests.support import (FakeRegistry, boxed_service, make_runtime,
                           native_probe, probe_source, response, visited)

ALL = ["turn_start", "shape_scope", "vet_permission", "llm_call", "end_turn",
       "turn_finish"]

#: Every doorway a plain turn actually reaches, in first-seen order.
#: ``vet_permission`` is absent because only the sandbox approver knocks
#: there, and an ordinary turn never asks it anything.
REACHED = ["shape_scope", "turn_start", "llm_call", "end_turn",
           "turn_finish"]


@pytest.fixture
def box():
    """A sandbox the bridge routes migrated plugins through."""
    made = Sandbox()
    configure(made)
    yield made
    configure(None)
    made.shutdown()


@pytest.fixture(autouse=True)
def clean_boxes():
    """Boxes are module caches; a leak hides staleness between tests."""
    yield
    unload_box("service_probe")


def _drive(tmp_path, responses=None, name="test.db"):
    """A real runtime with a real conversation, ready to be driven."""
    return make_runtime(tmp_path, responses or [response(content="hi")],
                        name=name, tool_registry=FakeRegistry([]))


# ──────────────────────────────────────────────────────────────────────
# The equivalence the rest of the suite rests on
# ──────────────────────────────────────────────────────────────────────

def test_a_boxed_probe_writes_the_same_journal_as_a_native_one(tmp_path, box):
    """One turn, two probes, identical records.

    Whatever ``sandbox/hooks.py`` narrows on the way out and rebuilds on the
    way back must be indistinguishable from the live objects a native hook is
    handed — for every moment a turn reaches, in one comparison.
    """
    native_journal = []
    rt_native, _, _ = _drive(tmp_path, name="native.db")
    for moment, fn in native_probe(native_journal, ALL).items():
        rt_native.hooks.add(moment, fn)
    rt_native.handle_action("s", "send_text", "hello")

    rt_boxed, _, _ = _drive(tmp_path, name="boxed.db")
    service = boxed_service(tmp_path, rt_boxed, probe_source(ALL),
                            filename="service_probe.py", name="probe")

    # Guard the premise. A bare ``tmp_path`` is an unknown tree and fails
    # closed to a subprocess — but if that ever stopped being true this file
    # would go on passing while testing nothing but in-process calls, which is
    # the one way it could become worthless without failing.
    assert type(service._sandbox_box._box).__name__ == "SubprocessBox", (
        "the probe is not actually isolated; this file proves nothing")

    rt_boxed.handle_action("s", "send_text", "hello")
    boxed_journal = service.journal()

    assert visited(native_journal) == REACHED
    assert boxed_journal == native_journal


def test_the_boxed_escort_places_the_call_and_the_reply_arrives(tmp_path, box):
    """The one moment with a callback going the other way.

    ``sdk.llm.proceed`` is a parked closure reached by a one-shot token, so an
    escort inside a box is the only hook whose payload travels *both* ways
    during one visit.
    """
    rt, _, llm = _drive(tmp_path, [response(content="escorted")])
    boxed_service(tmp_path, rt, probe_source(["llm_call"]),
                  filename="service_probe.py", name="probe")

    out = rt.handle_action("s", "send_text", "hello")

    assert len(llm.calls) == 1, "the boxed escort dialed the wrong number"
    assert any("escorted" in str(m) for m in out.messages)


def test_a_boxed_doorman_sends_the_agent_back(tmp_path, box):
    """A verdict built in a box is obeyed by the loop like a native one."""
    rt, _, llm = _drive(tmp_path, [response(content="first"),
                                   response(content="second")])
    boxed_service(
        tmp_path, rt,
        probe_source(["end_turn"], answers={
            "end_turn": 'SendBack("go back", ephemeral=True) '
                        'if payload.doorman_fires == 0 else None'}),
        filename="service_probe.py", name="probe")

    rt.handle_action("s", "send_text", "hello")

    assert len(llm.calls) == 2, "the boxed SendBack was not obeyed"


def test_unloading_mid_conversation_leaves_the_doorways_empty(tmp_path, box):
    """A hook cannot outlive its plugin, and the next turn is clean."""
    rt, _, llm = _drive(tmp_path, [response(content="one"),
                                   response(content="two")])
    service = boxed_service(tmp_path, rt, probe_source(ALL),
                            filename="service_probe.py", name="probe")

    rt.handle_action("s", "send_text", "hello")
    first = len(service.journal())
    assert first, "the probe never stood anywhere"

    service.unload()
    rt.handle_action("s", "send_text", "again")

    assert all(not bucket for bucket in rt.hooks._hooks.values()), (
        "a shim outlived the service that declared it")


def test_a_dead_box_abstains_instead_of_breaking_the_turn(tmp_path, box):
    """The worst a sandboxed hook can do is fall silent.

    Its box is closed out from under it mid-conversation — the shim is still
    standing at every doorway, and every visit has to come back ``None``.
    """
    rt, _, llm = _drive(tmp_path, [response(content="survived")])
    service = boxed_service(tmp_path, rt, probe_source(ALL),
                            filename="service_probe.py", name="probe")
    # Kill the box under the standing shims, without unloading the service —
    # so every doorway still walks to a hook whose box cannot answer.
    service._sandbox_box._box.stop()

    out = rt.handle_action("s", "send_text", "hello")

    assert out.ok, "a dead hook box broke the turn"
    assert any("survived" in str(m) for m in out.messages)


# ──────────────────────────────────────────────────────────────────────
# The path boot actually takes
# ──────────────────────────────────────────────────────────────────────

def test_discovery_stands_a_declared_hook_at_its_doorway(tmp_path, box,
                                                         monkeypatch):
    """Loaded the way the app loads it, not by calling ``adapt`` directly.

    Every other sandboxed hook test reaches for the bridge itself. Boot does
    not: it walks a tree and hands paths to ``plugin_discovery``, which is
    where the declaration is read and the shim stood up. A hook that works
    under ``adapt`` and never registers under discovery would look, from the
    app, exactly like a plugin with nothing to say.
    """
    import plugins.plugin_paths as plugin_paths
    from plugins import plugin_discovery

    rt, _, llm = _drive(tmp_path, [response(content="discovered")])

    # The file has to sit in a directory the layout recognises as the services
    # family — discovery does not take the caller's word for what a file is.
    services_dir = tmp_path / "tree" / "services"
    services_dir.mkdir(parents=True)
    root = plugin_paths.PluginRoot("test", services_dir.parent, "test_plugins")
    config = dict(plugin_paths.PLUGIN_CONFIG)
    config["service"] = (plugin_paths.PluginDir(root, "service", "services",
                                                "service_"),)
    monkeypatch.setattr(plugin_paths, "PLUGIN_CONFIG", config)

    path = services_dir / "service_probe.py"
    path.write_text(probe_source(ALL), encoding="utf-8")

    services = {}
    _, err = plugin_discovery._load_single_service(
        path, services, {}, {"runtime": rt})

    assert err is None, err
    assert "probe" in services, "discovery did not build the service"
    # The shims stand up at *registration*, before the box is ever opened —
    # which is what lets a service be unloaded and reloaded under a hook that
    # never moves. Opening it is the separate autoload step boot does next.
    assert rt.hooks._hooks["end_turn"], "discovery registered no end_turn shim"
    assert services["probe"].load() is True

    rt.handle_action("s", "send_text", "hello")

    assert visited(services["probe"].journal()) == REACHED
