"""Tests for the pipeline orchestrator.

The orchestrator subscribes to service-load events to wire up tasks; this
guards its bus lifecycle so a stopped orchestrator leaves no dangling
subscribers (which would leak across the kernel's hot-reload cycles).
"""

from events.event_bus import bus
from events.event_channels import SERVICE_LOADED
from pipeline.orchestrator import Orchestrator


def test_orchestrator_stop_unsubscribes_service_loaded_handler():
    """Verify orchestrator stop unsubscribes service loaded handler."""
    before = len(bus._subs.get(SERVICE_LOADED, []))
    orch = Orchestrator(None, {})
    assert len(bus._subs.get(SERVICE_LOADED, [])) == before + 1
    orch.stop()
    assert len(bus._subs.get(SERVICE_LOADED, [])) == before
