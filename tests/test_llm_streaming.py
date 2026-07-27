"""Streaming, inverted.

The old contract handed the backend a live ``on_delta`` callable whose *return
value* was the abort signal. Neither half can cross a boundary: a callable
cannot be serialized, and answering a boolean per token would cost a round
trip per token.

Both are replaced by one idea. Text goes out through ``sdk.model.delta`` —
one-way, token-scoped, a frame per chunk and no reply. Stopping is not
something the backend is told; it is cancellation, which the kernel already
owns, and a cancelled backend's next Request raises ``Terminated``.

So the claims here are: fragments arrive, in order, in both runners; the sink
is unreachable without the token; and nothing a sink does can break the call.
"""

import threading

import pytest

import sandbox  # noqa: F401  - installs the ``guest`` package alias
import llm
from sandbox import Sandbox
from sandbox.guest import requests as R
from sandbox.policy import Chain, classify
from sandbox.streams import park, sink, unpark

STREAMER = '''
"""A backend that streams a known sequence."""

ISOLATION
supports_streaming = True
display_name = "Streamer"

from guest.llm import BaseLLMBackend, LLMResponse


class StreamBackend(BaseLLMBackend):
    """Emit each word as its own delta, then return the whole thing."""

    def chat(self, sdk, request):
        """Stream when asked; otherwise answer in one piece."""
        words = ["Hello", " ", "there", " ", "world"]
        if not request.stream:
            return LLMResponse(content="".join(words))
        for word in words:
            sdk.model.delta(word)
        return LLMResponse(content="".join(words), prompt_tokens=7)
'''

LEAKY = '''
"""A backend that tries to stream outside a call it was given."""

supports_streaming = True

from guest.llm import BaseLLMBackend, LLMResponse


class LeakyBackend(BaseLLMBackend):
    """Push a delta under a token nobody parked."""

    def chat(self, sdk, request):
        """Forge a token and try to reach somebody else's stream."""
        sdk._delta_token = "not-a-real-token"
        sdk.model.delta("smuggled")
        return LLMResponse(content="tried")
'''


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A plugin tree holding streaming backends."""
    helpers = tmp_path / "helpers"
    helpers.mkdir()
    monkeypatch.setattr("plugins.helpers.plugin_paths.helper_dirs",
                        lambda: ((None, helpers),))
    monkeypatch.setattr(llm.registry, "_native_backends", dict)
    yield helpers
    llm.registry._BRAINS.clear()
    llm.registry._BACKENDS.clear()


def _brain(helpers, source, isolation="", stem="llm_stream",
           backend="StreamBackend"):
    """Write a backend, discover it, and return its brain."""
    (helpers / f"{stem}.py").write_text(
        source.replace("ISOLATION",
                       f'isolation = "{isolation}"' if isolation else ""),
        encoding="utf-8")
    llm.discover()
    llm.refresh({"llm_profiles": {"m": {"llm_service_class": backend}},
                 "default_llm_profile": "m", "max_workers": 1})
    return llm.brain("m")


# ──────────────────────────────────────────────────────────────────────
# Delivery.
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("isolation", ["", "subprocess"])
def test_fragments_arrive_in_order(tree, isolation):
    """A pipe preserves order; this pins that nothing else reorders them."""
    target = _brain(tree, STREAMER, isolation)
    seen = []

    response = target.chat(
        llm.LLMRequest(messages=[{"role": "user", "content": "hi"}],
                       stream=True),
        on_delta=seen.append)

    assert "".join(seen) == "Hello there world"
    # The accumulated response is what gets recorded; the deltas were only
    # ever for the user's eyes.
    assert response.content == "Hello there world"
    assert response.prompt_tokens == 7
    target.unload()


def test_no_deltas_are_sent_when_nobody_is_listening(tree):
    """Streaming is decided per call, so an unwatched call must not stream."""
    target = _brain(tree, STREAMER)
    seen = []

    response = target.chat(
        llm.LLMRequest(messages=[{"role": "user", "content": "hi"}],
                       stream=True))

    assert seen == []
    assert response.content == "Hello there world"
    target.unload()


def test_a_backend_that_cannot_stream_still_answers(tree):
    """``supports_streaming`` is read from the file, so it gates the request."""
    source = STREAMER.replace("supports_streaming = True",
                              "supports_streaming = False")
    target = _brain(tree, source)
    seen = []

    response = target.chat(
        llm.LLMRequest(messages=[{"role": "user", "content": "hi"}],
                       stream=True),
        on_delta=seen.append)

    assert seen == []
    assert response.content == "Hello there world"
    target.unload()


# ──────────────────────────────────────────────────────────────────────
# The desk: reachability, not a verdict.
# ──────────────────────────────────────────────────────────────────────

def test_a_forged_token_reaches_nothing(tree):
    """The scoping claim, made from inside a box.

    Asserted on the *host* side deliberately. A delta is one-way, so the guest
    is never told whether it landed — which is the honest consequence of not
    waiting for an answer, and is why the token has to be unguessable rather
    than merely checked.
    """
    target = _brain(tree, LEAKY, stem="llm_leaky", backend="LeakyBackend")
    eavesdropper = []
    honest_token = park(eavesdropper.append)

    try:
        response = target.chat(
            llm.LLMRequest(messages=[{"role": "user", "content": "x"}]))
    finally:
        unpark(honest_token)

    assert response.content == "tried"
    assert eavesdropper == [], "a forged token reached a parked sink"
    target.unload()


def test_the_token_is_cleared_when_the_call_ends(tree):
    """A token that outlived its call would be a way back in later."""
    target = _brain(tree, STREAMER)
    tokens = []
    original = park

    import sandbox.streams as streams
    def remember(sink_fn):
        """Record every token handed out."""
        token = original(sink_fn)
        tokens.append(token)
        return token

    streams.park = remember
    try:
        target.chat(llm.LLMRequest(messages=[{"role": "user", "content": "x"}],
                                   stream=True), on_delta=lambda _: None)
    finally:
        streams.park = original

    assert tokens, "no token was parked for a streaming call"
    assert all(sink(token) is None for token in tokens)
    target.unload()


def test_a_raising_sink_cannot_fail_the_model_call(tree):
    """A frontend that cannot draw a character must not lose the response."""
    target = _brain(tree, STREAMER)

    def explode(_fragment):
        """The worst a rendering path can do."""
        raise RuntimeError("terminal on fire")

    response = target.chat(
        llm.LLMRequest(messages=[{"role": "user", "content": "x"}],
                       stream=True), on_delta=explode)

    assert response.content == "Hello there world"
    target.unload()


# ──────────────────────────────────────────────────────────────────────
# The desk in isolation.
# ──────────────────────────────────────────────────────────────────────

def test_parking_and_unparking_are_symmetric():
    """The mechanism, without a box in the way."""
    seen = []
    token = park(seen.append)

    assert sink(token) is not None
    from sandbox.streams import deliver
    assert deliver(token, "x") is True
    assert seen == ["x"]

    unpark(token)
    assert sink(token) is None
    assert deliver(token, "y") is False
    assert seen == ["x"]


def test_delta_is_safe_because_it_is_scoped_not_because_it_is_harmless():
    """Classified safe on reachability, exactly like model.proceed."""
    assert classify(R.Request(R.MODEL_DELTA, {"token": "x", "text": "y"}),
                    Chain()).safe
    # But it is not read-only: it pushes text at a person.
    assert not R.Request(R.MODEL_DELTA, {}).read_only


def test_a_delta_does_not_wait_for_an_answer():
    """The property the whole design rests on.

    If ``delta`` went through ``send`` it would block for a Result per token,
    which is a full round trip per token across a pipe — enough to make
    streaming from a subprocess slower than not streaming at all.
    """
    from sandbox.guest.sdk import SDK

    sent, notified = [], []

    class Channel:
        """Records which half of the wire was used."""

        def send(self, request):
            """The blocking path."""
            sent.append(request.type)
            return R.Result(data=None)

        def notify(self, request):
            """The one-way path."""
            notified.append(request.type)

        def log(self, level, message):
            """Ignore."""

    sdk = SDK(Channel())
    sdk._delta_token = "t"
    sdk.model.delta("hello")

    assert notified == [R.MODEL_DELTA]
    assert sent == []


def test_a_notice_still_passes_the_gate():
    """One-way is not a way around policy — only around waiting."""
    from sandbox.interpreter import Interpreter

    seen = []
    interpreter = Interpreter(approve=lambda *a, **k: False)
    original = interpreter.submit
    interpreter.submit = lambda execution, request: (
        seen.append(request.type), original(execution, request))[1]

    from sandbox.interpreter import Execution
    channel = interpreter.channel(Execution(name="t", chain=Chain()))
    channel.notify(R.Request(R.MODEL_DELTA, {"token": "", "text": "x"}))

    assert seen == [R.MODEL_DELTA]
    interpreter.shutdown()


def test_an_empty_fragment_is_dropped_before_it_costs_a_frame():
    """Providers emit empty deltas routinely; each one should be free."""
    from sandbox.handlers.kernel import _model_delta

    assert _model_delta(None, {"token": "nope", "text": ""}).data is False
