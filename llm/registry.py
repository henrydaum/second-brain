"""The LLM registry — the kernel's answer to "which model, and how do I reach it?"

The LLM used to be a service, and like parsing that was a mistake worth
naming. A service is a *runtime object other code calls*, so every backend had
to be a live ``BaseLLM`` instance the kernel held by reference, and the router
existed to keep a dict of those instances consistent with a dict of config.
Roughly 250 of ``service_llm.py``'s 576 lines were that bookkeeping:
registering one service per profile, resyncing them when a backend file
changed, mirroring five attributes off the resolved default so the router
could impersonate it.

None of it survives the move, because none of it was about talking to a model.

- **The kernel routes.** Which profiles exist, which is the default, which
  backend serves each — standing knowledge, answered in strings, owned here.
- **A backend is a residency, not an endpoint.** It lives in a box, and
  ``load``/``unload`` mean opening and closing that box. That is what makes
  "loaded" honest: it is a live process, not a flag.
- **What crosses is data.** An :class:`LLMRequest` in, an :class:`LLMResponse`
  out, text deltas in between.

A :class:`Brain` is the handle: one per configured profile, holding the
profile's connection settings and a pool of boxes to run them in.

**Why a pool.** ``PersistentBox.call`` serializes under one lock, so a single
box per profile would queue a scheduled subagent behind a foreground turn —
a regression, since provider SDKs are thread-safe today. Backends are
stateless with respect to the profile (every model name and key arrives on the
request), so any box in a pool can serve any call, and the pool simply grows
to meet demand. Its ceiling is the real one: the orchestrator's worker threads
plus the foreground turn is the most concurrent LLM calls that can exist.
"""

from __future__ import annotations

import ast
import logging
import os
import threading
import uuid
from pathlib import Path

from sandbox.guest.llm import (LLMProviderError, LLMRequest, LLMResponse,
                               extract_llm_error_text)

logger = logging.getLogger("LLMClass")

# What a backend file is called, and what it subclasses.
BACKEND_PREFIX = "llm_"
BACKEND_BASE = "BaseLLMBackend"

# Fallback when a profile names no backend. Kept as the historical class name
# so existing configs keep resolving while the store still ships the native
# LiteLLM service.
DEFAULT_BACKEND = "LiteLLMService"


# ──────────────────────────────────────────────────────────────────────
# Discovery — by declaration, never by import.
#
# Importing a backend to ask whether it can stream would mean importing
# litellm to answer a question the file already states. So the declarations
# are read out of the source with AST, exactly as the bridge reads ``exports``
# and ``hooks``.
# ──────────────────────────────────────────────────────────────────────

_BACKENDS: dict[str, dict] = {}
# Old backend name -> the migrated one that claims it, built from each
# backend's ``replaces`` declaration. Existing configs name the class that
# used to serve them; without this, installing the migrated package would
# silently orphan every profile a user already had.
_ALIASES: dict[str, str] = {}
_BRAINS: dict[str, "Brain"] = {}
_LOCK = threading.RLock()


def _entry_from(source: str) -> str:
    """The backend class's name, read out of the source."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id == BACKEND_BASE:
                    return node.name
    return ""


def discover() -> int:
    """Rebuild the backend catalogue by scanning every plugin tree.

    Scans ``llm/llm_*.py`` at each tree's root in precedence order (bundled
    first). A file that will not validate is skipped with a logged reason
    rather than failing the scan — one broken backend must not take the others
    with it, because one of the others may be the only way to reach a model at
    all.

    Installing a backend package and calling this makes it selectable;
    uninstalling and calling this drops it.
    """
    import trees
    from sandbox.validator import validate_file

    with _LOCK:
        _BACKENDS.clear()
        _ALIASES.clear()
        seen: set[str] = set()
        for _root, backends in trees.dirs_for("llm"):
            if not backends.exists():
                continue
            for py_file in sorted(backends.glob(f"{BACKEND_PREFIX}*.py")):
                if py_file.stem in seen:
                    continue          # a higher-precedence tree won
                seen.add(py_file.stem)
                report = validate_file(py_file)
                if not report.ok:
                    logger.warning("LLM backend %s will not load:\n%s",
                                   py_file.name, report.render())
                    continue
                entry = _entry_from(report.source)
                if not entry:
                    logger.warning("LLM backend %s declares no %s subclass",
                                   py_file.name, BACKEND_BASE)
                    continue
                declared = report.declarations
                # A migrated backend claims the name its predecessor had, so
                # profiles written against the old contract keep resolving.
                # Declared by the file rather than kept as a table here: the
                # backend knows what it replaced, and the kernel should not
                # accumulate a list of every rename anyone has ever done.
                for old in (declared.get("replaces") or []):
                    if isinstance(old, str) and old and old not in _ALIASES:
                        _ALIASES[old] = entry
                _BACKENDS[entry] = {
                    "name": entry,
                    "path": py_file,
                    "stem": py_file.stem,
                    "display_name": declared.get("display_name") or entry,
                    "supports_streaming": bool(declared.get("supports_streaming")),
                    "supports_tool_choice": bool(declared.get("supports_tool_choice")),
                    "native_modalities": declared.get("native_modalities"),
                    "sandboxed": True,
                }
        logger.info("LLM backend discovery: %d sandboxed backend(s).",
                    len(_BACKENDS))
        return len(_BACKENDS)


def backend_names() -> list[str]:
    """Every sandboxed backend a profile may name."""
    with _LOCK:
        if not _BACKENDS:
            discover()
        return sorted(_BACKENDS)


def backend_display_names() -> dict[str, str]:
    """Backend name to the label a person should see."""
    with _LOCK:
        labels = {name: spec["display_name"]
                  for name, spec in _BACKENDS.items()}
    return labels


# ──────────────────────────────────────────────────────────────────────
# A brain: one profile, one backend, a pool of boxes.
# ──────────────────────────────────────────────────────────────────────

def _pool_ceiling(config: dict) -> int:
    """How many boxes one profile may ever open.

    Derived rather than chosen: a subagent is the only thing that places a
    model call concurrently with the foreground turn, so the subagent ceiling
    plus one is the largest number of calls that can be in flight at once. A
    constant would be either wasteful or wrong the moment that setting changed.

    This used to read ``max_workers``, from when a subagent ran on an
    orchestrator worker. It runs on its own pool now (runtime/subagents.py),
    and the two numbers must be the *same* number rather than two that happen
    to agree — otherwise a fan-out wider than the pool serializes its model
    calls behind one box lock and looks merely slow.
    """
    try:
        return max(1, int((config or {}).get("max_concurrent_subagents", 4))) + 1
    except (TypeError, ValueError):
        return 5


class Brain:
    """One configured model, and the way to reach it.

    Replaces the per-profile ``BaseLLM`` *service*. The difference that
    matters: this object holds no provider library and no connection. It holds
    settings and a pool of boxes, so it is cheap to construct, safe to keep,
    and honest about what "loaded" means.
    """

    def __init__(self, name: str, profile: dict, config: dict | None = None):
        self.name = name
        self.profile = dict(profile or {})
        self.config = config if config is not None else {}
        self._boxes: list = []
        self._idle: list = []
        self._lock = threading.Lock()
        # Growth is serialized separately from the pool bookkeeping: opening a
        # box is slow, and holding ``_lock`` across it would make ``loaded``
        # block on a subprocess start.
        self._growing = threading.Lock()
        self._sandbox = None
        # The sandbox keys resident boxes by name and hands back an existing
        # one under the same name, so the name has to identify *this* brain,
        # not just its profile. Two brains for one profile exist routinely —
        # every time a profile is edited, ``refresh`` builds a replacement —
        # and without this the new brain would inherit the old one's box, and
        # with it the settings the edit was meant to change.
        self._id = uuid.uuid4().hex[:8]

    # --- what the profile says -------------------------------------

    @property
    def model_name(self) -> str:
        """The name the provider knows this model by. The profile key is it."""
        return self.name

    @property
    def backend_name(self) -> str:
        """Which backend serves this profile."""
        return self.profile.get("llm_service_class") or DEFAULT_BACKEND

    @property
    def capabilities(self) -> dict:
        """Which modalities this model ingests natively.

        ``None`` means unknown and routes as False — a model that might read
        images gets the text fallback, which is wrong in a way the user can
        see and fix, rather than a provider error they cannot.
        """
        declared = self.profile.get("llm_capabilities") or {}
        caps = {"image": None, "audio": None, "video": None}
        caps.update({k: v for k, v in declared.items() if k in caps})
        return caps

    def has_capability(self, modality: str) -> bool:
        """Whether this model reads a modality natively."""
        return bool(self.capabilities.get(modality))

    @property
    def context_size(self) -> int:
        """Context window in tokens; 0 means unknown (reactive compaction)."""
        try:
            return int(self.profile.get("llm_context_size", 0) or 0)
        except (TypeError, ValueError):
            return 0

    @property
    def api_key(self) -> str:
        """The key, resolved through the environment if it names a variable."""
        raw = (
            self.profile.get("secret_llm_api_key")
            or self.profile.get("llm_api_key", "")
            or ""
        )
        return os.environ.get(raw, raw) if raw else ""

    @property
    def base_url(self) -> str:
        """The provider endpoint, or empty for the provider default."""
        return self.profile.get("llm_endpoint", "") or ""

    @property
    def spec(self) -> dict | None:
        """The backend's declarations, or None when it is not installed.

        Resolved through the alias table, so a profile naming a backend that
        has since been migrated and renamed still finds its successor.
        """
        with _LOCK:
            if not _BACKENDS:
                discover()
            name = self.backend_name
            return _BACKENDS.get(name) or _BACKENDS.get(_ALIASES.get(name, ""))

    @property
    def supports_streaming(self) -> bool:
        """Whether this backend can push deltas."""
        spec = self.spec
        return bool(spec and spec["supports_streaming"])

    @property
    def supports_tool_choice(self) -> bool:
        """Whether this backend honours a forced tool choice."""
        spec = self.spec
        return bool(spec and spec["supports_tool_choice"])

    @property
    def native_modalities(self) -> set:
        """Which modalities this *backend* can put on the wire.

        Distinct from ``capabilities``, which is about the *model*. Sending a
        photo natively needs both: a model that can see it and a backend that
        knows how to encode it. Defaults to all three, since a backend that
        does not say is far more likely to be a general provider client than a
        text-only one.
        """
        spec = self.spec
        declared = (spec or {}).get("native_modalities")
        return set(declared) if declared else {"image", "audio", "video"}

    @property
    def available(self) -> bool:
        """Whether this brain could actually talk, if asked.

        A profile naming a backend nobody installed still gets a Brain — it
        answers questions about context size and capabilities — but it must
        not *win* resolution against a working model registered elsewhere.
        Existing and being usable are different questions.
        """
        return self.spec is not None

    @property
    def loaded(self) -> bool:
        """Whether at least one box is live. Not a flag — a process."""
        with self._lock:
            return any(box.alive for box in self._boxes)

    # --- lifecycle -------------------------------------------------

    def load(self) -> bool:
        """Open the first box. Idempotent.

        The box is released into the free list immediately: ``load`` opens one
        but does not use it, and a box that is never freed is never leased —
        the next call would open a second one and the first would idle forever
        as a wasted process.
        """
        if self.loaded:
            return True
        try:
            box = self._grow()
            if box is not None:
                self._release(box)
            return box is not None
        except Exception as exc:
            logger.error("LLM profile %r failed to load: %s", self.name, exc)
            return False

    def unload(self) -> None:
        """Close every box. Tolerates never having loaded."""
        with self._lock:
            boxes, self._boxes, self._idle = self._boxes, [], []
        for box in boxes:
            try:
                box.stop()
            except Exception:
                logger.exception("closing a box for %r failed", self.name)

    # --- the pool --------------------------------------------------

    def _open_sandbox(self):
        """The Sandbox this brain's boxes live in."""
        if self._sandbox is None:
            from sandbox.bridge import get_sandbox
            self._sandbox = get_sandbox()
        return self._sandbox

    def _grow(self):
        """Open one more box and return it, or None at the ceiling.

        Serialized: two callers racing here would compute the same index,
        derive the same box name, and the second would be handed the first's
        half-built box.
        """
        spec = self.spec
        if spec is None:
            raise RuntimeError(
                f"No LLM backend named {self.backend_name!r} is installed.")
        with self._growing:
            with self._lock:
                index = len(self._boxes)
                if index >= _pool_ceiling(self.config):
                    return None
            box = self._open_sandbox().open(
                spec["path"], spec["name"],
                name=f"llm_{self._id}_{index}")
            with self._lock:
                self._boxes.append(box)
            return box

    def _lease(self):
        """Take an idle box, or open one, or wait for a busy one.

        Returning None means the pool is at its ceiling and every box is busy;
        the caller blocks on the box lock instead, which is the correct
        behaviour — that is exactly the queueing the ceiling exists to permit.
        """
        with self._lock:
            if self._idle:
                return self._idle.pop()
            at_ceiling = len(self._boxes) >= _pool_ceiling(self.config)
            if at_ceiling:
                # Every box is busy. Serialize onto the first one; its own
                # lock does the waiting.
                return self._boxes[0] if self._boxes else None
        return self._grow() or self._lease()

    def _release(self, box):
        """Return a box to the pool."""
        with self._lock:
            if box in self._boxes and box not in self._idle and box.alive:
                self._idle.append(box)

    # --- the call --------------------------------------------------

    def chat(self, request: LLMRequest, on_delta=None) -> LLMResponse:
        """Place one call and return what the model said.

        ``on_delta`` is a host-side callable receiving text fragments. It is
        parked under a one-shot token for the duration of this call and never
        crosses the boundary — the box gets the token, not the callable.

        Raises :class:`LLMProviderError` when the failure is one the kernel
        reacts to (a context overflow triggers compaction and a retry). Other
        failures come back as an error-shaped response, because the caller
        already knows how to render one.
        """
        from sandbox.streams import park, unpark

        if not self.loaded and not self.load():
            return LLMResponse.failure("no LLM loaded", "not_loaded")

        request.model_name = request.model_name or self.model_name
        request.api_key = request.api_key or self.api_key
        request.base_url = request.base_url or self.base_url
        streaming = bool(request.stream and on_delta is not None
                         and self.supports_streaming)
        request.stream = streaming

        token = park(on_delta) if streaming else ""
        box = self._lease()
        if box is None:
            unpark(token)
            return LLMResponse.failure("no box available", "not_loaded")
        try:
            result = box.call("__chat__", request=request.to_dict(),
                              token=token)
        finally:
            unpark(token)
            self._release(box)

        if not result.ok:
            message = result.error or "LLM backend call failed"
            if "context" in message.lower():
                raise LLMProviderError(message, code="context_limit")
            return LLMResponse.failure(message)

        response = LLMResponse.from_dict(result.data)
        # A context overflow has to *raise*: the conversation loop's compaction
        # layer catches it, rebuilds the prompt from compacted history and
        # retries. Returned as a response it would look like an ordinary
        # provider error and the turn would simply fail.
        if response.error_code == "context_limit":
            raise LLMProviderError(response.error or "context limit exceeded",
                                   code="context_limit")
        return response


class NativeBrain(Brain):
    """A :class:`Brain` over an unmigrated ``BaseLLM`` service instance.

    Same interface, none of the isolation: the backend runs in the kernel's
    own process, holding its own provider library and its own connection. This
    is the contract being migrated away from, kept working so that migrating
    is a choice per backend rather than a flag day.
    """

    def __init__(self, name: str, profile: dict, config: dict | None = None,
                 service=None):
        super().__init__(name, profile, config)
        self._service = service

    @property
    def supports_streaming(self) -> bool:
        """Asked of the instance, since a native backend declares nothing."""
        return bool(getattr(self._service, "supports_streaming", False))

    @property
    def supports_tool_choice(self) -> bool:
        """Asked of the instance."""
        return bool(getattr(self._service, "supports_tool_choice", False))

    @property
    def loaded(self) -> bool:
        """The instance's own flag; there is no process to point at.

        Absent means *ready*, matching the kernel's long-standing
        ``getattr(llm, "loaded", True)`` convention: an object that does not
        track loadedness has nothing to load, and treating it as unloaded
        would make it permanently unusable.
        """
        if self._service is None:
            return False
        return bool(getattr(self._service, "loaded", True))

    def load(self) -> bool:
        """Load the service, if it is the sort of thing that loads."""
        if self._service is None:
            return False
        if self.loaded:
            return True
        loader = getattr(self._service, "load", None)
        if not callable(loader):
            return True
        try:
            return bool(loader())
        except Exception:
            logger.exception("native LLM %r failed to load", self.name)
            return False

    @property
    def available(self) -> bool:
        """Whether there is an instance behind this at all."""
        return self._service is not None

    @property
    def native_modalities(self) -> set:
        """Asked of the instance, which declares it as an attribute."""
        declared = getattr(self._service, "native_attachment_modalities", None)
        return set(declared) if declared else {"image", "audio", "video"}

    def unload(self) -> None:
        """Unload the service, if it is the sort of thing that unloads."""
        unloader = getattr(self._service, "unload", None)
        if self._service is None or not self.loaded or not callable(unloader):
            return
        try:
            unloader()
        except Exception:
            logger.exception("native LLM %r failed to unload", self.name)

    def chat(self, request: LLMRequest, on_delta=None) -> LLMResponse:
        """Call the native instance, translating both directions.

        The native contract still wants a live ``AttachmentBundle`` and an
        ``on_delta`` returning a bool, so this is where the old shapes are
        rebuilt — in one place, on its way out, rather than spread through the
        kernel.
        """
        if not self.loaded and not self.load():
            return LLMResponse.failure("no LLM loaded", "not_loaded")

        kwargs = dict(request.params or {})
        bundle = _native_bundle(request.attachments)
        streaming = bool(request.stream and on_delta is not None
                         and self.supports_streaming)
        try:
            if streaming:
                def sink(fragment):
                    """The old contract, preserved where it still works.

                    A native backend runs in this process, so it *can* be told
                    to stop consuming — the falsy return the kernel's
                    ``_emit_delta`` produces on cancellation still reaches it.
                    A sandboxed backend cannot be told anything mid-call; it
                    is cancelled instead, which is the stronger mechanism.
                    """
                    return bool(on_delta(fragment))
                native = self._service.chat_with_tools_streaming(
                    request.messages, request.tools, on_delta=sink,
                    attachments=bundle, **kwargs)
            else:
                native = self._service.chat_with_tools(
                    request.messages, request.tools, attachments=bundle,
                    **kwargs)
        except Exception as exc:
            from sandbox.guest.llm import is_context_limit_error
            message = extract_llm_error_text(exc)
            if is_context_limit_error(exc) or getattr(exc, "code", "") == "context_limit":
                raise LLMProviderError(message, code="context_limit") from exc
            return LLMResponse.failure(message)

        response = LLMResponse(
            content=getattr(native, "content", "") or "",
            tool_calls=list(getattr(native, "tool_calls", []) or []),
            prompt_tokens=getattr(native, "prompt_tokens", None),
            cached_prompt_tokens=getattr(native, "cached_prompt_tokens", None),
            error=getattr(native, "error", None),
            error_code=getattr(native, "error_code", None),
        )
        if response.error_code == "context_limit":
            raise LLMProviderError(response.error or "context limit exceeded",
                                   code="context_limit")
        return response


def as_brain(target, name: str = "", config: dict | None = None):
    """Whatever a caller is holding, as something with ``.chat``.

    A :class:`Brain` passes through. Anything else exposing the old
    ``chat_with_tools`` interface — an unmigrated backend, a stress harness's
    fake, a test double — is wrapped in a :class:`NativeBrain`, which is what
    that class was for.

    This is the seam that lets the kernel speak exactly one language while the
    old contract is still out there. Without it, every caller that injects its
    own model object would have to be migrated in the same commit as the
    kernel, which is the flag day dual mode exists to avoid.
    """
    if target is None or isinstance(target, Brain):
        return target
    if not hasattr(target, "chat_with_tools"):
        return target
    wrapped = NativeBrain(
        name or getattr(target, "model_name", "") or "llm",
        {"llm_context_size": getattr(target, "context_size", 0) or 0,
         "llm_capabilities": dict(getattr(target, "capabilities", {}) or {})},
        config or {}, service=target)
    return wrapped


def _native_bundle(attachments):
    """Rebuild an ``AttachmentBundle`` for a backend still expecting one."""
    if not attachments:
        return None
    try:
        from attachments.attachment import Attachment, AttachmentBundle
    except Exception:
        return None
    rebuilt = []
    for item in attachments:
        if not isinstance(item, dict):
            rebuilt.append(item)
            continue
        try:
            rebuilt.append(Attachment.from_dict(item))
        except Exception:
            logger.warning("could not rebuild an attachment for a native "
                           "backend: %r", item)
    return AttachmentBundle.from_iterable(rebuilt) if rebuilt else None


# ──────────────────────────────────────────────────────────────────────
# The registry proper: profiles in, brains out.
# ──────────────────────────────────────────────────────────────────────

def _build(name: str, profile: dict, config: dict) -> Brain:
    """Build a profile brain; missing backends fail honestly when called."""
    return Brain(name, profile, config)


def refresh(config: dict, *, force: bool = False) -> dict[str, Brain]:
    """Rebuild every brain from config. Call after profiles change.

    Brains that survive the change keep their pools: rebuilding a brain whose
    settings did not move would close a working box for nothing.
    ``force`` is for backend source changes and preserves which profiles were
    loaded while replacing their boxes.
    """
    profiles = (config or {}).get("llm_profiles", {}) or {}
    with _LOCK:
        for name in [n for n in _BRAINS if n not in profiles]:
            _BRAINS.pop(name).unload()
        for name, profile in profiles.items():
            current = _BRAINS.get(name)
            if (
                not force
                and current is not None
                and current.profile == profile
            ):
                current.config = config
                continue
            was_loaded = bool(current and current.loaded)
            if current is not None:
                current.unload()
            _BRAINS[name] = _build(name, profile, config)
            if was_loaded:
                _BRAINS[name].load()
        if not (config or {}).get("default_llm_profile") and profiles:
            config["default_llm_profile"] = next(iter(profiles))
        return dict(_BRAINS)


def brains() -> dict[str, Brain]:
    """Every configured brain."""
    with _LOCK:
        return dict(_BRAINS)


def brain(name: str) -> Brain | None:
    """One brain by profile name."""
    with _LOCK:
        return _BRAINS.get(name or "")


def usable_brain(name: str) -> Brain | None:
    """One brain by name, but only if its backend is actually installed."""
    found = brain(name)
    return found if found is not None and found.available else None


def default_name(config: dict) -> str:
    """The profile that should drive when nothing else is named.

    Falls back to the first configured profile rather than to nothing: a
    misspelled default is a typo, and refusing to run at all would be a
    strange punishment for it.
    """
    with _LOCK:
        configured = (config or {}).get("default_llm_profile") or ""
        if configured and configured in _BRAINS:
            return configured
        remaining = sorted(name for name, target in _BRAINS.items()
                           if target.available)
        if configured and remaining:
            logger.warning("default_llm_profile %r not configured — "
                           "falling back to %r", configured, remaining[0])
        return remaining[0] if remaining else ""


def default_brain(config: dict) -> Brain | None:
    """The brain the default profile names, if it can actually run.

    Returning an unusable brain here would shadow a working model registered
    elsewhere — during the migration that is the unmigrated router, and a turn
    that fails because a *configured but uninstalled* profile won resolution
    is a confusing way to lose.
    """
    return usable_brain(default_name(config))


def resolve(ref, config: dict) -> Brain | None:
    """Turn whatever a caller is holding into a brain.

    ``ref`` may be a profile name, ``"default"``, nothing, or a model object.

    The contract is *names*, and that is what the kernel produces everywhere.
    But an escort written against the old contract assigns a live model, and
    refusing it would silently retarget the call onto the default — the worst
    possible failure, since the turn still succeeds and simply uses the wrong
    brain. So anything that is not a string is taken at face value and passed
    through to be adapted.
    """
    if ref is not None and not isinstance(ref, str):
        return ref
    name = (ref or "").strip()
    if not name or name == "default":
        return default_brain(config)
    return brain(name) or default_brain(config)


def load_default(config: dict) -> bool:
    """Open the default profile's first box at boot.

    Only the default: opening every configured profile would start a process
    per model the user has ever written down, most of which this session will
    never touch. The rest load on first use.
    """
    target = default_brain(config)
    return bool(target and target.load())


def unload_all() -> None:
    """Close every box. Called at shutdown."""
    for target in brains().values():
        target.unload()


def describe() -> list[dict]:
    """One row per profile, for ``/llm`` and ``/services``."""
    return [{
        "model_name": name,
        "class": target.backend_name,
        "endpoint": target.base_url,
        "context_size": target.context_size,
        "loaded": target.loaded,
        "sandboxed": not isinstance(target, NativeBrain),
    } for name, target in sorted(brains().items())]
