"""Tests for the store location service (``services/service_location.py``).

The package lives on the ``store`` branch, so the module is materialized
from the local store ref via ``git show`` and loaded with importlib —
mirroring what ``/packages install`` would copy. Skips cleanly when no
store ref is available. All network access is stubbed: the contract under
test is that the prompt path is cache-only and degrades quietly.
"""

import importlib.util
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO = Path(__file__).resolve().parents[1]
_STORE_REL = "services/service_location.py"


def _store_module_source() -> str | None:
    for ref in ("store", "origin/store"):
        proc = subprocess.run(
            ["git", "-C", str(_REPO), "show", f"{ref}:{_STORE_REL}"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", check=False)
        if proc.returncode == 0:
            return proc.stdout
    return None


@pytest.fixture(scope="module")
def location_module(tmp_path_factory):
    source = _store_module_source()
    if source is None:
        pytest.skip(f"{_STORE_REL} not present on a local store ref")
    path = tmp_path_factory.mktemp("location_service") / "service_location.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("service_location_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _service(location_module, config=None, payloads=None, loaded=True):
    """Build a service with network stubbed to canned per-URL payloads."""
    svc = location_module.LocationService(config or {})
    svc._fetch_json = lambda url: dict((payloads or {}).get(url, {}))
    svc.loaded = loaded
    return svc


def test_declares_store_contract(location_module):
    svc = location_module.build_services({})["location"]
    assert svc.model_name == "location"
    assert svc.lifecycle == "managed"
    assert svc.prompt_when_unloaded is True  # speaks even while off (consent gate)
    assert svc.dependencies_pip == []
    source = _store_module_source()
    assert "dependencies_pip = []" in source  # module-level literal for AST parsing


def test_manual_location_wins_and_skips_network(location_module):
    svc = _service(location_module, {"location_manual": "Seattle, WA"})
    svc._fetch_json = lambda url: pytest.fail("manual location must not hit the network")
    assert svc.load()

    prompt = svc.agent_prompt_for(SimpleNamespace(config={"location_manual": "Seattle, WA"}))

    assert "Seattle, WA" in prompt
    assert prompt.startswith("## Location")


def test_live_config_overrides_boot_config(location_module):
    svc = _service(location_module, {})
    prompt = svc.agent_prompt_for(SimpleNamespace(config={"location_manual": "Tokyo"}))
    assert "Tokyo" in prompt


def test_auto_location_serves_cache_without_io(location_module):
    payload = {"city": "Portland", "region": "Oregon", "country": "US",
               "timezone": "America/Los_Angeles"}
    svc = _service(location_module, {}, {"https://ipinfo.io/json": payload})
    svc._refresh()  # synchronous stand-in for the background warm-up

    svc._fetch_json = lambda url: pytest.fail("warm cache must not refetch")
    prompt = svc.agent_prompt_for(SimpleNamespace(config={}))

    assert "Portland, Oregon, US" in prompt
    assert "America/Los_Angeles" in prompt
    assert "IP-based" in prompt


def test_provider_fallback_and_quiet_failure(location_module):
    # First provider empty → ip-api fallback shape is used.
    svc = _service(location_module, {}, {
        "http://ip-api.com/json": {"status": "success", "city": "Boise",
                                   "regionName": "Idaho", "country": "US",
                                   "timezone": "America/Boise"}})
    svc._refresh()
    assert "Boise, Idaho, US" in svc.agent_prompt_for(SimpleNamespace(config={}))

    # All providers down → empty contribution, and the failed sweep still
    # stamps the clock so prompt builds don't hammer the network.
    svc = _service(location_module, {})
    svc._refresh()
    assert svc.agent_prompt_for(SimpleNamespace(config={})) == ""
    assert not svc._stale()


def test_unloaded_service_tells_the_agent_it_is_off(location_module):
    svc = _service(location_module, {"location_manual": "Seattle, WA"}, loaded=False)
    svc._fetch_json = lambda url: pytest.fail("unloaded service must not hit the network")

    prompt = svc.agent_prompt_for(SimpleNamespace(config={"location_manual": "Seattle, WA"}))

    assert "unloaded" in prompt
    assert "location service" in prompt
    assert "Seattle" not in prompt  # nothing is shared without an explicit load


def test_stale_cache_refreshes_in_background_not_inline(location_module):
    svc = _service(location_module, {"location_refresh_minutes": 5},
                   {"https://ipinfo.io/json": {"city": "Denver"}})
    svc._refresh()
    svc._fetched_at = time.time() - 3600  # force staleness

    prompt = svc.agent_prompt_for(SimpleNamespace(config={}))
    assert "Denver" in prompt  # stale value served immediately

    deadline = time.time() + 5
    while svc._stale() and time.time() < deadline:
        time.sleep(0.02)
    assert not svc._stale()  # background refresh landed


def test_unload_clears_cache(location_module):
    svc = _service(location_module, {}, {"https://ipinfo.io/json": {"city": "Austin"}})
    svc._refresh()
    assert "Austin" in svc.agent_prompt_for(SimpleNamespace(config={}))

    svc.unload()

    assert svc._cache == {}
    assert not svc.loaded
