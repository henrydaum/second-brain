"""The LLM registry: profiles in, brains out.

The claims worth pinning are the ones that used to be the router's job and
are now nobody's: that a profile resolves without a service registry, that
"loaded" means a live box rather than a flag, that the pool grows under
concurrency and closes completely on unload, and that an unmigrated backend
still answers — because until every backend migrates, dual mode *is* the
contract.
"""

import threading
import time
from pathlib import Path

import pytest

import sandbox  # noqa: F401  - installs the ``guest`` package alias
import llm
from tests.support import retarget_trees
from llm.registry import Brain, _pool_ceiling

BACKEND = '''
"""A backend that answers without a provider."""

ISOLATION
supports_streaming = True
supports_tool_choice = True
display_name = "Echo"

from guest.llm import BaseLLMBackend, LLMResponse


class EchoBackend(BaseLLMBackend):
    """Says back what it was told, so the plumbing is the only variable."""

    def chat(self, sdk, request):
        """Echo the last message, reporting what crossed with it."""
        last = request.messages[-1]["content"] if request.messages else ""
        if request.stream:
            for piece in str(last).split():
                sdk.llm.delta(piece + " ")
        return LLMResponse(
            content=f"echo:{last}",
            prompt_tokens=len(request.messages),
            tool_calls=[{"id": "c1", "name": "t", "arguments": "{}"}]
            if request.tools else [],
        )
'''

FAILING_BACKEND = '''
"""A backend whose provider always says the prompt is too long."""

supports_streaming = False
display_name = "Overflowing"

from guest.llm import BaseLLMBackend


class OverflowBackend(BaseLLMBackend):
    """Raise the way a real provider does, from inside the box."""

    def chat(self, sdk, request):
        """Fail with something the classifier should recognise."""
        raise RuntimeError("This model's maximum context length is 8192 tokens")
'''


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A plugin tree whose ``helpers/`` holds our backends.

    Discovery reads declarations off disk, so a test only has to write files —
    there is no registration call to fake.
    """
    helpers = retarget_trees(monkeypatch, tmp_path)["workspace"] / "llm"
    helpers.mkdir(parents=True)
    # Native backends are a separate world; keep them out unless a test asks.
    yield helpers
    llm.registry._BRAINS.clear()
    llm.registry._BACKENDS.clear()


def _write(helpers, source, isolation="", stem="llm_echo"):
    """Drop a backend into the tree and rescan."""
    (helpers / f"{stem}.py").write_text(
        source.replace("ISOLATION",
                       f'isolation = "{isolation}"' if isolation else ""),
        encoding="utf-8")
    llm.discover()


def _config(helpers, backend="EchoBackend", **profile):
    """A config naming one profile served by one backend."""
    settings = {"llm_service_class": backend, **profile}
    return {"llm_profiles": {"gpt-test": settings},
            "default_llm_profile": "gpt-test", "max_workers": 2}


# ──────────────────────────────────────────────────────────────────────
# Discovery reads, it does not import.
# ──────────────────────────────────────────────────────────────────────

def test_declarations_are_read_without_importing(tree):
    """The point of AST discovery: knowing what a backend can do is free."""
    _write(tree, BACKEND)

    names = llm.backend_names()

    assert names == ["EchoBackend"]
    assert llm.backend_display_names()["EchoBackend"] == "Echo"
    # Nothing was imported, so the module is absent from sys.modules.
    import sys
    assert not [m for m in sys.modules if m.endswith("llm_echo")]


def test_a_broken_backend_does_not_take_the_others_with_it(tree):
    """One of the survivors may be the only way to reach a model at all."""
    _write(tree, BACKEND)
    (tree / "llm_broken.py").write_text("this is not python(", encoding="utf-8")

    llm.discover()

    assert "EchoBackend" in llm.backend_names()


# ──────────────────────────────────────────────────────────────────────
# Resolution: no service registry involved.
# ──────────────────────────────────────────────────────────────────────

def test_a_profile_resolves_to_a_brain(tree):
    """What the router used to do, without registering anything."""
    _write(tree, BACKEND)
    config = _config(tree, llm_context_size=8192)

    llm.refresh(config)
    target = llm.brain("gpt-test")

    assert isinstance(target, Brain)
    assert target.model_name == "gpt-test"
    assert target.context_size == 8192
    assert target.supports_streaming is True
    assert target.supports_tool_choice is True


def test_a_misspelled_default_falls_back_rather_than_refusing(tree):
    """A typo should not make the application unusable."""
    _write(tree, BACKEND)
    config = _config(tree)
    config["default_llm_profile"] = "does-not-exist"

    llm.refresh(config)

    assert llm.default_name(config) == "gpt-test"


def test_capabilities_default_to_unknown_and_route_as_false(tree):
    """Unknown must not be optimistic: the text fallback is the safe wrong."""
    _write(tree, BACKEND)
    llm.refresh(_config(tree))

    target = llm.brain("gpt-test")

    assert target.capabilities == {"image": None, "audio": None, "video": None}
    assert target.has_capability("image") is False


def test_refresh_keeps_a_brain_whose_settings_did_not_move(tree):
    """Rebuilding an unchanged brain would close a working box for nothing."""
    _write(tree, BACKEND)
    config = _config(tree)
    llm.refresh(config)
    first = llm.brain("gpt-test")

    llm.refresh(config)

    assert llm.brain("gpt-test") is first


def test_a_removed_profile_is_unloaded_and_forgotten(tree):
    """Config is the source of truth; a dropped profile takes its boxes."""
    _write(tree, BACKEND)
    config = _config(tree)
    llm.refresh(config)
    llm.brain("gpt-test").load()

    config["llm_profiles"] = {}
    llm.refresh(config)

    assert llm.brain("gpt-test") is None


# ──────────────────────────────────────────────────────────────────────
# Loading means a process, not a flag.
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("isolation", ["", "subprocess"])
def test_a_call_crosses_and_comes_back(tree, isolation):
    """The whole point, in both runners."""
    _write(tree, BACKEND, isolation)
    llm.refresh(_config(tree))
    target = llm.brain("gpt-test")

    response = target.chat(llm.LLMRequest(
        messages=[{"role": "user", "content": "hello"}]))

    assert response.content == "echo:hello"
    assert response.prompt_tokens == 1
    assert target.loaded is True
    target.unload()
    assert target.loaded is False


def test_tools_cross_as_data(tree):
    """Tool schemas are plain JSON, so they need no special handling."""
    _write(tree, BACKEND)
    llm.refresh(_config(tree))

    response = llm.brain("gpt-test").chat(llm.LLMRequest(
        messages=[{"role": "user", "content": "x"}],
        tools=[{"name": "t", "parameters": {}}]))

    assert response.has_tool_calls
    assert response.tool_calls[0]["name"] == "t"


def test_unload_closes_every_box_in_the_pool(tree):
    """A leaked box is an orphaned process that outlives the config."""
    _write(tree, BACKEND)
    llm.refresh(_config(tree))
    target = llm.brain("gpt-test")
    target.load()
    target._grow()
    assert len(target._boxes) == 2

    target.unload()

    assert target._boxes == []
    assert target.loaded is False


# ──────────────────────────────────────────────────────────────────────
# The pool.
# ──────────────────────────────────────────────────────────────────────

def test_the_ceiling_follows_the_subagent_count(tree):
    """Derived, not chosen: concurrent subagents plus the foreground turn.

    It followed ``max_workers`` while a subagent ran on an orchestrator
    worker. It runs on its own pool now, and the two numbers have to be the
    same setting rather than two that happen to agree — otherwise a fan-out
    wider than the pool serializes behind one box lock and reads as slow.
    """
    assert _pool_ceiling({"max_concurrent_subagents": 4}) == 5
    assert _pool_ceiling({"max_concurrent_subagents": 1}) == 2
    # A nonsense setting must not make the ceiling zero.
    assert _pool_ceiling({"max_concurrent_subagents": "banana"}) == 5
    assert _pool_ceiling({}) == 5
    # And the setting it used to read no longer moves it.
    assert _pool_ceiling({"max_workers": 12}) == 5


def test_concurrent_calls_grow_the_pool_instead_of_queueing(tree):
    """The regression this exists to prevent: a box serializes its calls."""
    _write(tree, BACKEND)
    llm.refresh(_config(tree))
    target = llm.brain("gpt-test")
    target.load()

    barrier = threading.Barrier(2, timeout=10)
    seen = []

    def call():
        """Two calls that must be in flight at the same moment."""
        box = target._lease()
        seen.append(box)
        barrier.wait()
        target._release(box)

    threads = [threading.Thread(target=call) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert len(seen) == 2
    assert seen[0] is not seen[1], "both calls took the same box"


def test_loading_then_calling_uses_one_box_not_two(tree):
    """``load`` opens a box it does not use, so it must free it.

    A box that is never released is never leased: the first call would open a
    second box and the first would idle forever. Under isolation that is a
    wasted process per profile, and nothing would ever surface it — both boxes
    are alive and answering.
    """
    _write(tree, BACKEND)
    llm.refresh(_config(tree))
    target = llm.brain("gpt-test")

    target.load()
    target.chat(llm.LLMRequest(messages=[{"role": "user", "content": "x"}]))

    assert len(target._boxes) == 1
    target.unload()


def test_the_pool_stops_growing_at_the_ceiling(tree):
    """Unbounded growth under a runaway loop would be a fork bomb."""
    _write(tree, BACKEND)
    config = _config(tree)
    config["max_concurrent_subagents"] = 1     # ceiling of 2
    llm.refresh(config)
    target = llm.brain("gpt-test")

    held = [target._lease() for _ in range(4)]

    assert len(target._boxes) == 2
    assert all(box is not None for box in held)


# ──────────────────────────────────────────────────────────────────────
# Failure shapes.
# ──────────────────────────────────────────────────────────────────────

def test_a_context_overflow_raises_so_compaction_can_catch_it(tree):
    """Returned as a response it would look ordinary and the turn would fail."""
    _write(tree, FAILING_BACKEND, stem="llm_overflow")
    llm.refresh(_config(tree, backend="OverflowBackend"))

    with pytest.raises(llm.LLMProviderError) as raised:
        llm.brain("gpt-test").chat(llm.LLMRequest(
            messages=[{"role": "user", "content": "x"}]))

    assert raised.value.code == "context_limit"


def test_an_uninstalled_backend_fails_honestly(tree):
    """A profile naming a backend nobody installed is a message, not a crash."""
    _write(tree, BACKEND)
    llm.refresh(_config(tree, backend="NotInstalled"))

    target = llm.brain("gpt-test")
    response = target.chat(llm.LLMRequest(
        messages=[{"role": "user", "content": "x"}]))

    assert response.is_error
    assert response.error_code == "not_loaded"


def test_forced_refresh_replaces_loaded_backend_boxes(tree):
    """A hot-edited backend cannot keep serving from its old process."""
    _write(tree, BACKEND)
    config = _config(tree)
    llm.refresh(config)
    original = llm.brain("gpt-test")
    assert original.load()

    llm.refresh(config, force=True)

    replacement = llm.brain("gpt-test")
    assert replacement is not original
    assert replacement.loaded
    assert not original.loaded


# ──────────────────────────────────────────────────────────────────────
# Dual mode.
# ──────────────────────────────────────────────────────────────────────

def test_a_sandboxed_backend_is_the_profile_brain(tree):
    """Discovered backends always run through the isolated brain."""
    _write(tree, BACKEND)

    llm.refresh(_config(tree))

    assert type(llm.brain("gpt-test")) is Brain
