"""The LLM registry: profiles in, brains out.

The claims worth pinning are the ones that used to be the router's job and
are now nobody's: that a profile resolves without a service registry, that
"loaded" means a live box rather than a flag, and that the pool grows under
concurrency and closes completely on unload.
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


# ──────────────────────────────────────────────────────────────────────
# The doorway a command reaches the registry through.
#
# ``/llm`` asked the *service* registry about profiles, and had done since
# before ``service_llm.py`` was deleted and profiles stopped being services.
# Nothing raised: ``ctx.services`` simply had no key for any profile, so every
# lookup answered ``{}`` and the command reported each model "not installed"
# and "Unloaded" while conversations drove those same models perfectly well.
# Two registries, one question, and the UI was reading the wrong one.
# ──────────────────────────────────────────────────────────────────────

def test_llm_list_answers_from_the_registry_not_the_service_table(tree):
    """The whole bug in one assertion: no services anywhere, full answer."""
    from sandbox.handlers.kernel import _llm_list

    _write(tree, BACKEND)
    config = _config(tree)
    llm.refresh(config)

    class Ctx:
        """A kernel context with an empty service registry, as in real life."""
        services = {}
        config = None

    ctx = Ctx()
    ctx.config = config
    answer = _llm_list(ctx, {}).data

    assert [row["model_name"] for row in answer["profiles"]] == ["gpt-test"]
    assert answer["default"] == "gpt-test"
    assert answer["profiles"][0]["loaded"] is False


def test_llm_list_carries_the_declared_display_name(tree):
    """``display_name`` was declared, read into the registry, and shown to
    nobody — so every picker in the app offered raw class names and the
    profile card said "LiteLLMService" for a backend calling itself
    "LiteLLM (any provider)"."""
    from sandbox.handlers.kernel import _llm_list

    _write(tree, BACKEND)
    config = _config(tree)
    llm.refresh(config)

    class Ctx:
        """Minimal context."""
        services = {}
        config = None

    ctx = Ctx()
    ctx.config = config
    answer = _llm_list(ctx, {}).data

    assert answer["backends"] == [
        {"name": "EchoBackend", "display_name": "Echo"}]


def test_a_retired_backend_name_still_resolves_for_display(tree):
    """A profile stores the name it was written with, and a migrated backend
    claims its predecessor's. Displaying one therefore needs the alias hop
    first, which is exactly the hop that was missing."""
    _write(tree, BACKEND.replace(
        'display_name = "Echo"', 'display_name = "Echo"\nreplaces = ["OldEcho"]'))

    assert llm.backend_aliases() == {"OldEcho": "EchoBackend"}
    assert llm.backend_display_names()["EchoBackend"] == "Echo"


def test_llm_load_and_unload_open_and_close_a_real_pool(tree):
    """"Loaded" means live processes, so the handlers must move that and not
    a flag."""
    from sandbox.handlers.kernel import _llm_load, _llm_unload

    _write(tree, BACKEND)
    config = _config(tree)
    llm.refresh(config)

    class Ctx:
        """Minimal context."""
        services = {}
        config = None

    ctx = Ctx()
    ctx.config = config

    assert _llm_load(ctx, {"name": "gpt-test"}).data is True
    assert llm.brain("gpt-test").loaded

    assert _llm_unload(ctx, {"name": "gpt-test"}).data is True
    assert not llm.brain("gpt-test").loaded


def test_loading_an_uninstallable_profile_names_the_missing_backend(tree):
    """The message ``/llm`` used to invent was "No backend is installed for
    <profile>", which reported service-registry absence as backend absence and
    named the wrong thing entirely. The person has to go install a *backend*,
    so the failure has to say which one."""
    from sandbox.handlers.kernel import _llm_load

    _write(tree, BACKEND)
    config = _config(tree, backend="NoSuchBackend")
    llm.refresh(config)

    class Ctx:
        """Minimal context."""
        services = {}
        config = None

    ctx = Ctx()
    ctx.config = config
    result = _llm_load(ctx, {"name": "gpt-test"})

    assert not result.ok
    assert "NoSuchBackend" in result.error


# ──────────────────────────────────────────────────────────────────────
# Per-profile provider params.
#
# One key holds them (``llm_extra_params``) and the kernel supplies a
# reasoning effort for a profile that says nothing. What is worth pinning is
# therefore not "reasoning works" — the provider decides that — but the four
# states a profile can be in, and that a malformed blob cannot take a model
# off the air.
# ──────────────────────────────────────────────────────────────────────

def test_a_profile_that_says_nothing_sends_nothing():
    """The kernel names no provider parameter, and that is the whole rule.

    It used to supply ``reasoning_effort``, on the argument that "whatever the
    provider felt like" is not a decision anybody made and left one profile
    silently thinking hard and its neighbour not at all. That rested on the
    comparison being invisible — and ``/llm`` now lists every parameter a
    profile sends, on its own row, with the backend's verdict beside it. So
    the default bought a guess where there is a table, while costing real
    breakage: a Claude profile with thinking on cannot hand its signed
    ``thinking_blocks`` back, and the API refuses the next tool call."""
    assert Brain("gpt-test", {}).params == {}

def test_a_declared_effort_wins_over_the_kernels():
    """It lives in the extras dict like any other provider param — ``/llm``
    gives it a picker, not a key of its own."""
    target = Brain("gpt-test", {"llm_extra_params": {"reasoning_effort": "high"}})

    assert target.params == {"reasoning_effort": "high"}


def test_a_null_means_send_nothing():
    """The one convention this adds, and the reason it is needed: without it
    ``reasoning_effort`` would be the single param a profile could not decline,
    since the kernel now supplies one for anybody who stays quiet."""
    target = Brain("gpt-test", {"llm_extra_params": {"reasoning_effort": None}})

    assert target.params == {}


def test_a_null_declines_a_param():
    """A rule about every parameter, with no member of the dict named
    anywhere in the kernel."""
    target = Brain("gpt-test", {"llm_extra_params": {"temperature": None,
                                                     "seed": 7}})

    assert target.params == {"seed": 7}

def test_extra_params_are_forwarded_verbatim():
    """The escape hatch: anything a provider takes, without a kernel edit."""
    target = Brain("gpt-test", {"llm_extra_params": {"temperature": 0.2,
                                                     "seed": 7}})

    assert target.params == {"temperature": 0.2, "seed": 7}

@pytest.mark.parametrize("junk", ["high", ["a"], 3, None])
def test_malformed_extras_cannot_make_a_model_unreachable(junk):
    """This is hand-edited config, and talking to a model is the one
    capability the kernel cannot degrade without. A blob somebody typed
    wrongly must cost them the params, never the profile."""
    target = Brain("gpt-test", {"llm_extra_params": junk})

    assert target.params == {}

def test_params_are_detached_so_a_caller_may_mutate_them():
    """``_invoke_inner`` layers ``tool_choice`` on top of this dict."""
    profile = {"llm_extra_params": {"temperature": 0.2}}
    target = Brain("gpt-test", profile)

    target.params["tool_choice"] = "auto"

    assert "tool_choice" not in target.params
    assert profile["llm_extra_params"] == {"temperature": 0.2}


def test_llm_list_carries_what_a_profile_will_send(tree):
    """``/llm`` renders the resolved dict, so what it shows is what goes on
    the wire rather than what config happens to spell."""
    from sandbox.handlers.kernel import _llm_list

    _write(tree, BACKEND)
    config = _config(tree, llm_extra_params={"reasoning_effort": "high"})
    llm.refresh(config)

    class Ctx:
        """Minimal context."""
        services = {}
        config = None

    ctx = Ctx()
    ctx.config = config
    row = _llm_list(ctx, {}).data["profiles"][0]

    assert row["params"] == {"reasoning_effort": "high"}


def test_no_word_is_read_as_an_instruction():
    """``off`` was aliased to a null, back when the kernel supplied a level
    and declining it needed a spelling. It supplies nothing now, so removing
    the key *is* declining — and the alias was a hazard for any provider that
    takes ``off`` as a real value, which the kernel cannot know."""
    assert Brain("gpt-test",
                 {"llm_extra_params": {"service_tier": "off"}}).params == {
        "service_tier": "off"}
    assert Brain("gpt-test",
                 {"llm_extra_params": {"reasoning_effort": "none"}}).params == {
        "reasoning_effort": "none"}

def test_unusable_extras_say_so_once(caplog):
    """Ignored in silence is how somebody spends an afternoon wondering why a
    setting they can see in a file does nothing. Once, because ``params`` is
    read on every model call and a per-call warning is a log nobody reads."""
    target = Brain("gpt-test", {"llm_extra_params": "reasoning_effort=high"})

    with caplog.at_level("WARNING", logger="LLMClass"):
        assert target.params == {}
        assert target.params == {}

    warnings = [r for r in caplog.records
                if "llm_extra_params" in r.getMessage()]
    assert len(warnings) == 1
    assert "gpt-test" in warnings[0].getMessage()

def test_a_well_formed_profile_says_nothing(caplog):
    """The complaint must not fire for the ordinary cases, or it is noise."""
    with caplog.at_level("WARNING", logger="LLMClass"):
        Brain("a", {}).params
        Brain("b", {"llm_extra_params": {}}).params
        Brain("c", {"llm_extra_params": {"temperature": 0.2}}).params

    assert not [r for r in caplog.records if "llm_extra_params" in r.getMessage()]


# ──────────────────────────────────────────────────────────────────────
# What a failed call says.
#
# The kernel puts a param on calls whose profile never asked for one, and a
# backend does not forward that param — it translates it, into a dialect it
# picks from the model *name*. So a refusal can be about ``reasoning_effort``
# without anything on the wire, in the log, or on the screen containing the
# word. Measured against one aggregator the entire refusal was
# ``{'code': 400, 'msg': 'bad request'}``.
#
# What is pinned here is therefore not a diagnosis — nothing at this layer
# can name the param a provider objected to — but that the inputs ride along
# on the failure, and that the kernel's own contribution is marked as such.
# ──────────────────────────────────────────────────────────────────────

def test_a_failure_names_the_params_the_call_carried():
    """The missing half of the report: what was asked of the provider."""
    target = Brain("gpt-test", {"llm_extra_params": {"temperature": 0.2}})

    said = target._explained("bad request", {"temperature": 0.2})

    assert "bad request" in said
    assert "temperature=0.2" in said


def test_a_failure_names_where_the_params_can_be_changed():
    """A provider's refusal is often not legible on its own — one aggregator
    answered a parameter it would not take with ``{'code': 400, 'msg': 'bad
    request'}``, naming nothing. Listing what the call carried is the smallest
    fix that does not require the kernel to learn provider names."""
    target = Brain("gpt-test", {"llm_extra_params": {"temperature": 0.2}})

    said = target._explained("bad request", target.params)

    assert "temperature=0.2" in said
    assert "/llm" in said

def test_a_declared_param_is_listed_but_not_called_a_default():
    """Naming it — at any value, ``null`` included — is a decision, and a
    decision needs no explaining back to whoever made it."""
    target = Brain("gpt-test",
                   {"llm_extra_params": {"reasoning_effort": "high"}})

    said = target._explained("bad request", target.params)

    assert "reasoning_effort='high'" in said
    assert "kernel default" not in said


def test_a_call_that_carried_nothing_is_left_alone():
    """A profile that declined every param has nothing to be told, and the
    provider's own sentence should not grow a paragraph for no reason."""
    target = Brain("gpt-test",
                   {"llm_extra_params": {"reasoning_effort": None}})

    assert target._explained("bad request", target.params) == "bad request"


def test_a_credential_in_the_extras_is_not_printed_at_the_failure():
    """``/llm`` refuses ``api_key`` as an extra, so one only arrives through a
    hand-edited config — which is exactly the path that would otherwise put it
    on somebody's screen."""
    target = Brain("gpt-test", {"llm_extra_params": {"api_key": "sk-live-42"}})

    said = target._explained("bad request", target.params)

    assert "sk-live-42" not in said
    assert "redacted" in said


def test_the_kernel_adds_no_parameter_of_its_own():
    """The property the whole normalization rests on: what a profile sends is
    what somebody configured, with nothing appended and no member of the dict
    named anywhere in ``llm/``."""
    assert Brain("a", {}).params == {}
    assert Brain("b", {"llm_extra_params": {"seed": 7}}).params == {"seed": 7}
    assert not hasattr(Brain("c", {}), "default_params")

REFUSING_BACKEND = '''
"""A backend whose provider refuses without saying what it objected to."""

supports_streaming = False
display_name = "Refusing"

from guest.llm import BaseLLMBackend


class RefusingBackend(BaseLLMBackend):
    """Answer the way one aggregator really did."""

    def chat(self, sdk, request):
        """Refuse in the provider's own words, which name nothing."""
        raise RuntimeError("Error code: 400 - {'code': 400, 'msg': 'bad request'}")
'''


def test_a_bare_refusal_comes_back_naming_what_was_sent(tree):
    """The whole point, driven through a real box.

    The guest catches the provider exception and hands back an error-shaped
    response, so this is the path a person actually meets: a 400 whose text
    mentions no parameter at all, on a call that carried one.
    """
    from llm import LLMRequest

    _write(tree, REFUSING_BACKEND, stem="llm_refusing")
    llm.refresh(_config(tree, backend="RefusingBackend",
                        llm_extra_params={"reasoning_effort": "high"}))
    target = llm.brain("gpt-test")
    assert target.load()

    try:
        answer = target.chat(LLMRequest(
            messages=[{"role": "user", "content": "hi"}],
            params=target.params))
    finally:
        target.unload()

    assert answer.is_error
    assert "bad request" in answer.error                 # the provider's half
    assert "reasoning_effort='high'" in answer.error     # ours
    assert "/llm" in answer.error
    # ``content`` is what a caller reading only text will render, so it must
    # not be left holding the un-annotated sentence.
    assert "reasoning_effort" in answer.content

# ──────────────────────────────────────────────────────────────────────
# Describing what can be configured.
#
# Three questions asked while somebody is *setting a model up*, so none of
# them may require a working one. What matters most is the degradation: a
# backend that cannot answer has to be indistinguishable from one that was
# never asked, because the flow that asked falls back to a typed value and a
# raised exception there would strand somebody mid-form.
# ──────────────────────────────────────────────────────────────────────

DESCRIBING_BACKEND = '''
"""A backend that answers the three setup questions."""

ISOLATION
display_name = "Describer"

from guest.llm import BaseLLMBackend, LLMResponse


class DescribingBackend(BaseLLMBackend):
    """Answers from a fixed table, so the plumbing is the only variable."""

    def chat(self, sdk, request):
        return LLMResponse(content="ok")

    def providers(self, sdk, provider=""):
        # Asked bare it is the menu, with no endpoint; asked about one it is
        # that provider with its URL resolved.
        if provider:
            return [{"id": provider, "label": provider.title(),
                     "endpoint": f"https://{provider}.test/v1"}]
        return [{"id": "acme", "label": "Acme", "endpoint": ""}]

    def models(self, sdk, endpoint, api_key, provider="", live=False):
        if not endpoint and not provider:
            return []
        return [{"name": "acme/big", "label": "big"}]

    def params(self, sdk, model_name, endpoint):
        return [
            {"name": "reasoning_effort", "label": "Reasoning effort",
             "kind": "choice", "choices": ["low", "high"],
             "supported": False, "note": "discarded for acme; try 'thinking'"},
            {"name": "temperature", "label": "Temperature", "kind": "number",
             "choices": [], "supported": True, "note": ""},
        ]
'''

SILENT_BACKEND = '''
"""A backend that cannot introspect, which is the default and must be fine."""

ISOLATION
display_name = "Silent"

from guest.llm import BaseLLMBackend, LLMResponse


class SilentBackend(BaseLLMBackend):
    """Implements nothing beyond chat, like every backend written before."""

    def chat(self, sdk, request):
        return LLMResponse(content="ok")
'''

BROKEN_BACKEND = '''
"""A backend whose introspection raises."""

ISOLATION
display_name = "Broken"

from guest.llm import BaseLLMBackend, LLMResponse


class BrokenBackend(BaseLLMBackend):
    """Answers chat fine and blows up on every question about itself."""

    def chat(self, sdk, request):
        return LLMResponse(content="ok")

    def providers(self, sdk, provider=""):
        raise RuntimeError("provider table unavailable")

    def params(self, sdk, model_name, endpoint):
        raise RuntimeError("no such model")
'''


def test_a_backend_answers_the_three_setup_questions(tree):
    """The pyramid, driven through a real box: provider, model, params."""
    _write(tree, DESCRIBING_BACKEND, stem="llm_describing")
    llm.refresh(_config(tree, backend="DescribingBackend"))
    target = llm.brain("gpt-test")
    try:
        assert target.providers() == [
            {"id": "acme", "label": "Acme", "endpoint": ""}]
        # Naming one is what resolves its endpoint — the whole reason the
        # provider step earns its place ahead of the endpoint step.
        assert llm.registry.endpoint_for("acme") == "https://acme.test/v1"
        assert target.models("https://acme.test/v1", "k") == [
            {"name": "acme/big", "label": "big"}]
        names = [row["name"] for row in target.param_options("acme/big")]
        assert names == ["reasoning_effort", "temperature"]
    finally:
        target.unload()


def test_a_backend_that_cannot_introspect_is_not_an_error(tree):
    """Silence is an ordinary answer, and the whole reason it is optional.

    Every backend written before these methods existed is this one. If an
    absent answer were a failure rather than ``[]``, adding the contract would
    have broken all of them at once.
    """
    _write(tree, SILENT_BACKEND, stem="llm_silent")
    llm.refresh(_config(tree, backend="SilentBackend"))
    target = llm.brain("gpt-test")
    try:
        assert target.providers() == []
        assert target.models("https://x.test/v1", "k") == []
        assert target.param_options("anything") == []
        assert target.param_status == {}
    finally:
        target.unload()


def test_introspection_that_raises_is_indistinguishable_from_silence(tree):
    """A question asked mid-form must never be able to strand the form.

    The guest catches it, so this pins the whole path rather than the base
    class in isolation — the case that matters is a real backend whose
    provider library moved on underneath it.
    """
    _write(tree, BROKEN_BACKEND, stem="llm_broken")
    llm.refresh(_config(tree, backend="BrokenBackend"))
    target = llm.brain("gpt-test")
    try:
        assert target.providers() == []
        assert target.param_options("acme/big") == []
    finally:
        target.unload()


def test_param_status_reports_only_the_params_the_profile_sends(tree):
    """The card's question is about *this profile*, not the whole menu.

    ``param_options`` lists everything the model takes; ``param_status``
    narrows that to what is actually being sent, which is the only part a
    profile card can honestly warn about.
    """
    _write(tree, DESCRIBING_BACKEND, stem="llm_describing")
    llm.refresh(_config(tree, backend="DescribingBackend",
                        llm_extra_params={"reasoning_effort": "high"}))
    target = llm.brain("gpt-test")
    try:
        assert target.load()
        status = target.param_status
        # ``temperature`` is supported and unset, so it is not in the answer.
        assert set(status) == {"reasoning_effort"}
        arrives, note = status["reasoning_effort"]
        # Set by hand, so it is insisted on rather than dropped — and the note
        # warns that the provider may refuse it.
        assert arrives is True
        assert "you set it" in note
    finally:
        target.unload()

def test_a_closed_profile_reports_no_status_rather_than_opening_a_box(tree):
    """Rendering a list of profiles must not start subprocesses.

    ``{}`` here means "nobody has asked", which reads the same as "nothing to
    say" on a card — and that is the right trade, because the alternative is
    a menu that costs one process per row to draw.
    """
    _write(tree, DESCRIBING_BACKEND, stem="llm_describing")
    llm.refresh(_config(tree, backend="DescribingBackend"))
    target = llm.brain("gpt-test")

    assert not target.loaded
    assert target.param_status == {}
    assert not target.loaded


def test_describe_carries_param_status_for_a_ui_to_render(tree):
    """``llm.list`` is what a web client reads, so the verdict rides there."""
    _write(tree, DESCRIBING_BACKEND, stem="llm_describing")
    llm.refresh(_config(tree, backend="DescribingBackend",
                        llm_extra_params={"reasoning_effort": "high"}))
    target = llm.brain("gpt-test")
    try:
        assert target.load()
        row = [r for r in llm.describe() if r["model_name"] == "gpt-test"][0]
        assert row["param_status"]["reasoning_effort"][0] is True
        assert "you set it" in row["param_status"]["reasoning_effort"][1]
        # A list, not a tuple: this crosses the wire as JSON.
        assert isinstance(row["param_status"]["reasoning_effort"], list)
    finally:
        target.unload()

def test_an_answer_is_cached_per_question_and_arguments(tree):
    """Forms redraw on every step, so asking must not cost a box call each time.

    Keyed on the arguments rather than on the question alone, or step two
    would answer for whichever endpoint happened to be asked about first.
    """
    _write(tree, DESCRIBING_BACKEND, stem="llm_describing")
    llm.refresh(_config(tree, backend="DescribingBackend"))
    target = llm.brain("gpt-test")
    try:
        first = target.models("https://acme.test/v1", "k")
        assert target.models("https://acme.test/v1", "k") is first
        # A different endpoint is a different question.
        assert target.models("", "", "") is not first
    finally:
        target.unload()


# ──────────────────────────────────────────────────────────────────────
# Insisting on a value somebody chose.
#
# A backend may drop a parameter its provider table does not list. That is
# right for a value the kernel supplied and wrong for one a person picked, so
# the request carries which is which and the two outcomes must stay visibly
# different — a silently inert setting and a call the provider might refuse
# are opposite failures and want opposite warnings.
# ──────────────────────────────────────────────────────────────────────

def test_a_call_carries_exactly_what_the_profile_configured(tree):
    """Nothing is appended on the way to the backend.

    This once carried a second list saying which params the *profile* chose,
    because the kernel added one itself and a backend had to be told which
    ones to insist on. With nothing appended, every param is chosen and the
    list said the same thing as the dict beside it.
    """
    from llm import LLMRequest

    _write(tree, BACKEND)
    llm.refresh(_config(tree, llm_extra_params={"temperature": 0.2}))
    target = llm.brain("gpt-test")
    try:
        assert target.load()
        request = LLMRequest(messages=[{"role": "user", "content": "hi"}],
                             params=target.params)
        target.chat(request)
        assert request.params == {"temperature": 0.2}
        assert not hasattr(request, "chosen_params")
    finally:
        target.unload()

def test_a_set_param_reads_as_arriving_even_when_unlisted(tree):
    """A backend reports what its provider table says; the kernel knows the
    value was set by hand and is insisted on rather than dropped. So the card
    warns that the provider may refuse it — a different warning, aimed at a
    different outcome, from one that silently does nothing."""
    _write(tree, DESCRIBING_BACKEND, stem="llm_describing")
    settings = _config(tree, backend="DescribingBackend")

    chosen = Brain("gpt-test",
                   {"llm_service_class": "DescribingBackend",
                    "llm_extra_params": {"reasoning_effort": "high"}},
                   settings)
    try:
        assert chosen.load()
        arrives, note = chosen.param_status["reasoning_effort"]
        assert arrives is True
        assert "you set it" in note
    finally:
        chosen.unload()

    # A supported one says nothing at all, which is what a quiet card means.
    plain = Brain("gpt-test",
                  {"llm_service_class": "DescribingBackend",
                   "llm_extra_params": {"temperature": 0.2}}, settings)
    try:
        assert plain.load()
        assert plain.param_status["temperature"] == (True, "")
    finally:
        plain.unload()

def test_editing_a_profile_does_not_restart_the_backend_to_re_answer(tree):
    """The freeze this caused, pinned.

    ``refresh`` rebuilds a brain whose profile dict moved and ``unload``s the
    old one, closing the backend's process. The discovery cache lived on the
    brain, so it died with it — and the one thing guaranteed to edit a profile
    repeatedly is the settings form doing the asking. Every step rebuilt the
    brain, killed the box, and started the provider library again, which on a
    modest machine is indistinguishable from the command hanging.

    So the cache is keyed by *backend*, and this asserts the property that
    matters rather than the mechanism: the same question across a rebuild
    comes back without needing a live box.
    """
    _write(tree, DESCRIBING_BACKEND, stem="llm_describing")
    llm.refresh(_config(tree, backend="DescribingBackend"))

    before = llm.brain("gpt-test")
    answer = before.models("https://acme.test/v1", "k")
    assert answer, "the backend should have answered at least once"
    before.unload()

    # An edit, exactly as saving one field of the form produces.
    llm.refresh(_config(tree, backend="DescribingBackend",
                        llm_context_size=4096))
    after = llm.brain("gpt-test")
    assert after is not before, "this test is pointless if nothing rebuilt"
    assert not after.loaded

    assert after.models("https://acme.test/v1", "k") == answer
    # The whole point: answering did not have to open anything.
    assert not after.loaded


def test_a_backend_rescan_is_what_drops_the_answers(tree):
    """The one event that can change what a backend says about itself."""
    _write(tree, DESCRIBING_BACKEND, stem="llm_describing")
    llm.refresh(_config(tree, backend="DescribingBackend"))
    target = llm.brain("gpt-test")
    try:
        assert target.models("https://acme.test/v1", "k")
    finally:
        target.unload()

    llm.registry.forget_descriptions()
    assert not target.loaded
    # Nothing cached, so answering now needs a box again.
    assert target.models("https://acme.test/v1", "k")
    assert target.loaded
    target.unload()


def test_naming_a_provider_is_what_fills_in_its_endpoint(tree):
    """The provider step has to answer the endpoint step, or it is a wasted question.

    Asked bare, ``providers`` is a menu and carries no URLs — resolving one is
    expensive enough that doing it across the whole list is what made the
    command hang. Asked about one, it resolves. Both halves matter: without
    the first the menu is unusable, and without the second the user is asked
    for a URL immediately after telling us which provider they wanted.
    """
    _write(tree, DESCRIBING_BACKEND, stem="llm_describing")
    llm.refresh(_config(tree, backend="DescribingBackend"))
    target = llm.brain("gpt-test")
    try:
        menu = target.providers()
        assert menu and all(not row["endpoint"] for row in menu)
        assert llm.registry.endpoint_for("acme") == "https://acme.test/v1"
    finally:
        target.unload()


def test_a_provider_with_no_endpoint_to_offer_says_so(tree):
    """Blank is honest. A guessed URL fails later and blames the model."""
    _write(tree, SILENT_BACKEND, stem="llm_silent")
    llm.refresh(_config(tree, backend="SilentBackend"))
    target = llm.brain("gpt-test")
    try:
        assert llm.registry.endpoint_for("whoever") == ""
    finally:
        target.unload()


def test_asking_about_one_provider_sends_its_name_not_a_flag():
    """A boolean where a name belonged, pinned at the layer that lost it.

    ``sdk.llm.list`` took ``providers`` as a flag first and grew the name
    later, and the body kept coercing it to ``True``. The failure is silent in
    the worst way: the Request succeeds, the menu comes back, and the caller
    simply finds no endpoint on it — so the setup form showed an empty box
    under a sentence promising the URL was filled in.
    """
    import sandbox  # noqa: F401  - installs the ``guest`` alias
    from guest.sdk import _LLM

    sent = {}

    class Recording(_LLM):
        def __init__(self):
            pass

        def _ask(self, request_type, **args):
            sent.clear()
            sent.update(args)
            return {}

    Recording().list(providers=True)
    assert sent["providers"] is True                 # the menu

    Recording().list(providers="minimax")
    assert sent["providers"] == "minimax"            # one row, with its URL

    Recording().list()
    assert "providers" not in sent
