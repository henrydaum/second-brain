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

from sandbox.guest.llm import LLMProviderError, LLMRequest, LLMResponse

logger = logging.getLogger("LLMClass")

# What a backend file is called, and what it subclasses.
BACKEND_PREFIX = "llm_"
BACKEND_BASE = "BaseLLMBackend"

# Fallback when a profile names no backend. Deliberately the *historical*
# class name: the store's backend calls itself ``LiteLLMBackend`` and claims
# this one with ``replaces = ["LiteLLMService"]``, so every config a user
# already has keeps resolving through ``backend_aliases``. Not a migration
# shim — a stored value nobody rewrites, and the alias is how it stays valid.
DEFAULT_BACKEND = "LiteLLMService"

# No provider parameter is named here, and that is the point. The kernel once
# supplied ``reasoning_effort`` for a profile that said nothing, which made it
# the one parameter with a default, an alias, a picker of its own and a branch
# in four separate places. See ``Brain.params`` for why it no longer does. A
# parameter is whatever the user configured, and they are all the same kind of
# thing.


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
        # A rescan is the one event that can change what a backend would
        # answer about itself, so it is the one thing that drops the cache.
        _DESCRIBED.clear()
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


def forget_descriptions() -> None:
    """Drop cached discovery answers. Called when backend source may have moved.

    Nothing else invalidates them, on purpose: what a backend knows about a
    provider does not change because a profile was edited, and tying the two
    together is what made a settings form restart a subprocess per step.
    """
    with _LOCK:
        _DESCRIBED.clear()


def backend_names() -> list[str]:
    """Every sandboxed backend a profile may name."""
    with _LOCK:
        if not _BACKENDS:
            discover()
        return sorted(_BACKENDS)


def backend_display_names() -> dict[str, str]:
    """Backend name to the label a person should see."""
    with _LOCK:
        if not _BACKENDS:
            discover()
        labels = {name: spec["display_name"]
                  for name, spec in _BACKENDS.items()}
    return labels


def backend_aliases() -> dict[str, str]:
    """Retired backend name to the one that replaced it.

    A profile stores whatever backend name it was written with, and a migrated
    backend claims its predecessor's (``replaces``), so a config still saying
    ``LiteLLMService`` has to be resolved through here before it can be looked
    up or displayed. ``Brain.spec`` already does this internally; anything
    *showing* a configured backend to a person needs the same map.
    """
    with _LOCK:
        if not _BACKENDS:
            discover()
        return dict(_ALIASES)


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


# Substrings that make a parameter name look like a credential. Extra params
# are free-form and hand-editable, so one can land here despite ``/llm``
# refusing ``api_key`` outright — and the place it would land is an error
# message on somebody's screen.
_SECRETISH = ("key", "token", "secret", "password", "auth")


def _shown_value(name: str, value) -> str:
    """One parameter's value, as it should appear in a failure message."""
    if any(hint in name.lower() for hint in _SECRETISH):
        return "<redacted>"
    return repr(value)


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
        # Last malformed ``llm_extra_params`` complained about, so the warning
        # is one line per profile version rather than one per model call.
        self._extras_complaint = None
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
    def params(self) -> dict:
        """Extra provider kwargs this profile sends on every call.

        One key holds all of them — ``llm_extra_params`` — forwarded verbatim,
        with no member of it named anywhere in the kernel. That is the whole of
        what keeps this backend-agnostic: no provider matrix, no declaration
        for a backend to get wrong, and nothing in ``llm/`` that knows which
        library is on the other side. A backend that cannot carry a param
        degrades it (the store's LiteLLM sets ``drop_params``) or reports the
        provider's own refusal — which is not reliably a sentence anybody can
        act on, and is the cost of staying ignorant here. It is paid in
        :meth:`_explained` rather than by learning provider names.

        **The kernel used to name one of them.** ``reasoning_effort`` was
        supplied at a default level for any profile that said nothing, on the
        argument that "whatever the provider felt like" differs per model and
        left one profile silently thinking hard and its neighbour not at all.
        That argument rested on the comparison being invisible, and it is not
        any more: ``/llm`` lists every parameter a profile sends, on its own
        row, with the backend's verdict beside it. So the default bought a
        guess where there is now a table, and cost real breakage — a Claude
        profile with thinking on cannot hand its signed ``thinking_blocks``
        back and the API refuses the next tool call. A parameter nobody set is
        not sent.

        **A null still means omit**, and that is a rule about every parameter
        rather than a concession to one. It is what ``/llm`` writes when you
        say to send nothing, and what a hand-edited config can say too.

        Tolerant like ``context_size``, and for the same reason: this is
        user-entered config, and a malformed extras blob must not be able to
        stop the model being reachable. It says so out loud, though — being
        ignored in silence is how a person spends an afternoon wondering why
        a setting they can see in a file does nothing.
        """
        extra = self.profile.get("llm_extra_params")
        if extra is not None and not isinstance(extra, dict):
            self._complain_about_extras(extra)
        declared = dict(extra) if isinstance(extra, dict) else {}
        return {key: value for key, value in declared.items()
                if value is not None}

    def _explained(self, message: str, params: dict) -> str:
        """Name the provider params a failed call carried.

        ``params`` here is what the kernel put on the wire, which is not what
        the provider was asked — the backend translates first, and only it
        knows into what. So this is deliberately a list of inputs rather than
        a diagnosis: nothing at this layer can say which param an endpoint
        objected to, and a guess dressed as an answer would be worse than the
        silence it replaces.

        It exists because a provider's refusal is often not legible on its
        own. One aggregator's entire answer to a parameter it would not take
        was ``{'code': 400, 'msg': 'bad request'}`` — between the picker that
        set the value and the screen that reported the failure, the parameter
        was named nowhere. Listing what the call carried is the smallest thing
        that fixes that without the kernel learning any provider's name.

        Values are shown because a level is the whole content of the setting,
        and redacted for anything named like a credential. ``/llm`` already
        refuses ``api_key`` as an extra, so that only reaches here through a
        hand-edited config — which is exactly the case that would put one on
        someone's screen.
        """
        if not params:
            return message
        shown = ", ".join(f"{key}={_shown_value(key, value)}"
                          for key, value in sorted(params.items()))
        note = (f"The call carried these provider parameters: {shown}. "
                "`/llm` can change or clear them.")
        return f"{message} | {note}"

    def _complain_about_extras(self, extra) -> None:
        """Say once that a profile's extra params are unusable.

        Once, because ``params`` is read on every model call and a per-call
        warning is a log nobody reads. A ``Brain`` is rebuilt whenever its
        profile changes (``refresh``), so "once per brain" is really "once per
        version of this profile" — it speaks up again after an edit that
        failed to fix it, which is exactly when it is wanted.
        """
        if getattr(self, "_extras_complaint", None) == repr(extra):
            return
        self._extras_complaint = repr(extra)
        logger.warning(
            "LLM profile %r: llm_extra_params must be a JSON object like "
            "{\"temperature\": 0.2}, got %s. Ignoring it and sending only "
            "the defaults.", self.name, type(extra).__name__)

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

    def _interrupt(self, box):
        """End the call this box is serving, by ending the box.

        Nothing lighter works. A streaming backend's only outbound Request is
        ``sdk.llm.delta``, a one-way notice, so refusing to answer it reaches
        nothing and the caller stays blocked on the pipe until the provider
        finishes. This is what ``/cancel`` spends to stop a model mid-token.

        **Evicted from ``_boxes``, not merely from ``_idle``.** ``_grow``
        counts ``len(self._boxes)`` against the ceiling and ``_lease`` hands
        back ``self._boxes[0]`` once there, so a dead box left in that list
        means the pool never reopens and starts leasing a corpse — every later
        call failing with "box is not running". ``_release`` already declines
        to re-idle a dead box, which is precisely what hides the other half.
        """
        with self._lock:
            if box in self._idle:
                self._idle.remove(box)
            if box in self._boxes:
                self._boxes.remove(box)
        box.interrupt()

    # --- describing what can be configured -------------------------

    def _describe(self, question: str, **args) -> list:
        """Ask this brain's backend one discovery question.

        Cached in ``_DESCRIBED``, which is the whole of what makes this
        affordable: a form redraws on every step, and the first answer costs a
        backend process start.

        That start is the reason the cache cannot live on the brain. It did,
        and a brain does not survive its profile being edited — which is
        precisely what the form doing the asking is *for*. See ``_DESCRIBED``.

        Still needs a live box, and opens one if there is none. Acceptable
        because asking is something a person deliberately did, and it happens
        once per backend rather than once per keystroke. Contrast
        ``param_status``, which refuses to open anything at all: rendering a
        list of profiles is nobody's deliberate act.
        """
        key = (self.backend_name, question, tuple(sorted(args.items())))
        with _LOCK:
            if key in _DESCRIBED:
                return _DESCRIBED[key]
        if not self.loaded and not self.load():
            return []
        box = self._lease()
        if box is None:
            return []
        try:
            result = box.call("__describe__", question=question, args=args)
        finally:
            self._release(box)
        answer = result.data if result.ok and isinstance(result.data, list) else []
        with _LOCK:
            _DESCRIBED[key] = answer
        return answer

    def providers(self, provider: str = "") -> list:
        """Providers this brain's backend can reach. ``[]`` when it cannot say.

        Naming one asks for *its endpoint*, which is the expensive half — see
        ``BaseLLMBackend.providers``.
        """
        return self._describe("providers", provider=provider)

    def models(self, endpoint: str = "", api_key: str = "",
               provider: str = "", live: bool = False) -> list:
        """Models at *endpoint*, named the way this backend wants them back.

        Defaults to this profile's own endpoint and key, since the common
        caller is "show me what else this provider has". ``live`` lets the
        backend ask the endpoint itself and is off by default — see
        ``BaseLLMBackend.models`` for why a form may never turn it on.
        """
        return self._describe(
            "models",
            endpoint=endpoint or self.base_url,
            api_key=api_key or self.api_key,
            provider=provider, live=bool(live))

    def info(self, model_name: str = "", endpoint: str | None = None) -> dict:
        """What the backend knows about a model: ``{"context_size": int}``.

        ``{}`` when it cannot say, which is every backend that does not
        implement it and every model missing from the one that does.
        """
        rows = self._describe(
            "info",
            model_name=model_name or self.model_name,
            endpoint=self.base_url if endpoint is None else endpoint)
        return dict(rows[0]) if rows and isinstance(rows[0], dict) else {}

    def param_options(self, model_name: str = "",
                      endpoint: str | None = None) -> list:
        """Extra parameters *model_name* accepts, as reports rather than gates.

        Named apart from :attr:`params`, which is the dict this profile
        *sends*. One is the menu, the other is the order.

        ``endpoint`` defaults to this profile's own, and is passed explicitly
        only while setting a *new* profile up, where the endpoint being asked
        about is not yet anybody's.
        """
        return self._describe(
            "params",
            model_name=model_name or self.model_name,
            endpoint=self.base_url if endpoint is None else endpoint)

    @property
    def param_status(self) -> dict:
        """``{param: (arrives, note)}`` for what this profile actually sends.

        The configured-profile reading of :meth:`param_options`: it answers
        only about the params in :attr:`params`, because those are the ones a
        profile card can honestly say anything about.

        ``arrives`` is **not** the backend's ``supported`` flag, and the gap
        between them is the whole of what this property adds. A backend
        reports what its provider table says; the kernel knows that every one
        of these was set by hand, and a set parameter is insisted on rather
        than dropped. So a parameter the table rejects still arrives, and the
        note warns that the provider may refuse it — a different warning,
        aimed at a different outcome, from one that silently does nothing.

        Returns ``{}`` when the backend cannot say **or when no box is open**.
        Silence is the honest answer for a profile nobody has loaded; the
        alternative is starting a subprocess to draw a menu row.
        """
        if not self.loaded:
            return {}
        sending = self.params
        if not sending:
            return {}
        known = {row.get("name"): row for row in self.param_options()
                 if isinstance(row, dict)}
        status = {}
        for name in sending:
            row = known.get(name)
            if row is None:
                continue
            if row.get("supported", True):
                status[name] = (True, "")
            else:
                status[name] = (True, (
                    "sent because you set it, though this backend does not "
                    "list it for this model — the provider may reject the "
                    "call"))
        return status

    # --- the call --------------------------------------------------

    def chat(self, request: LLMRequest, on_delta=None, on_call=None) -> LLMResponse:
        """Place one call and return what the model said.

        ``on_delta`` is a host-side callable receiving text fragments. It is
        parked under a one-shot token for the duration of this call and never
        crosses the boundary — the box gets the token, not the callable.

        ``on_call`` is handed a zero-argument callable that ends this call,
        the moment a box has been leased to serve it. It exists because a
        caller cannot know *what* to stop until then: which box serves a
        profile is the pool's business, and a call queued behind a busy one
        has nothing to interrupt yet. See ``ConversationLoop._call_backend``,
        which arms it into the session's interrupt registry so ``/cancel``
        can fire it from another thread.

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
        if on_call is not None:
            on_call(lambda: self._interrupt(box))
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
            return LLMResponse.failure(self._explained(message, request.params))

        response = LLMResponse.from_dict(result.data)
        # A context overflow has to *raise*: the conversation loop's compaction
        # layer catches it, rebuilds the prompt from compacted history and
        # retries. Returned as a response it would look like an ordinary
        # provider error and the turn would simply fail.
        if response.error_code == "context_limit":
            raise LLMProviderError(response.error or "context limit exceeded",
                                   code="context_limit")
        # Everything else is a refusal somebody has to read, so it says what
        # was asked. A context overflow is exempt on purpose: the layer above
        # handles it without a person ever seeing it, and the params had
        # nothing to do with it.
        if response.is_error:
            explained = self._explained(response.error, request.params)
            if response.content.startswith("Error: "):
                response.content = f"Error: {explained}"
            response.error = explained
        return response



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


# ──────────────────────────────────────────────────────────────────────
# Setup questions.
#
# The three of these are asked while somebody is *configuring* a model, so
# none of them can require a configured model. They hang off the module
# rather than off a ``Brain`` for that reason, and each borrows whichever
# brain is available to do the asking — legitimate because a backend's
# answers are about the backend, not about the profile that reached it.
#
# All three answer ``[]`` when nothing can say. That is the first-run state
# and also the permanent state for a backend that does not introspect, so it
# is an ordinary answer and never an error.
# ──────────────────────────────────────────────────────────────────────

#: Discovery answers, keyed ``(backend, question, args)``. Module-level and
#: deliberately not on the ``Brain``, which was the first place it went and
#: was wrong twice over.
#:
#: ``refresh`` rebuilds a brain whenever its profile dict changes and
#: ``unload``\s the old one — closing the backend's process. A settings form
#: writes config as it goes, so every step of it rebuilt the brain, killed the
#: box, and made the next question start the provider library again. On a
#: modest machine that reads as the command freezing, and it is the same
#: process being started and torn down three or four times in a row.
#:
#: A brain is also the wrong owner on the merits: these answer what a
#: *backend* knows — which providers exist, what an endpoint serves, what a
#: model takes — and none of it varies by which profile did the asking. Keyed
#: by backend for that reason, and invalidated by ``discover()``, which is
#: exactly when a backend's source may have changed.
_DESCRIBED: dict = {}


def _asking_brains() -> list[Brain]:
    """Brains whose backend is installed, already-loaded ones first.

    Order matters because asking needs a live box. Preferring one that is
    already open means a discovery question reuses the running process
    instead of starting a second copy of a provider library beside it.
    """
    ready = [target for _name, target in sorted(brains().items())
             if target.available]
    return sorted(ready, key=lambda target: not target.loaded)


def providers(provider: str = "") -> list[dict]:
    """Providers the installed backends can reach, for step one of setup.

    Deduplicated by ``id`` across backends, first answer winning, so two
    backends offering the same provider do not show it twice.

    Naming one narrows to that row and asks for its endpoint, which is the
    part worth waiting for and the reason step one exists at all: a provider
    picked from a menu should arrive at step two with its URL already filled
    in, or the menu did nothing but ask a question it then repeats.
    """
    seen, out = set(), []
    for target in _asking_brains():
        for row in target.providers(provider):
            key = str(row.get("id") or row.get("label") or "").lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(row)
    return out


def endpoint_for(provider: str) -> str:
    """One provider's default endpoint, or ``""`` when it has none to give."""
    for row in providers(provider):
        if str(row.get("id") or "").lower() == provider.lower():
            return str(row.get("endpoint") or "")
    return ""


def models_at(endpoint: str, api_key: str = "", provider: str = "",
              live: bool = False) -> list[dict]:
    """Models reachable at *endpoint*, for step two.

    First backend with an answer wins rather than merging: a model name is
    only meaningful to the backend that produced it — the prefix it carries is
    that backend's convention — so pooling two backends' answers would build a
    list where picking the wrong row silently misroutes the call.
    """
    for target in _asking_brains():
        found = target.models(endpoint, api_key, provider, live)
        if found:
            return found
    return []


def info_for(model_name: str, endpoint: str = "") -> dict:
    """One model's facts, from the first backend that has any."""
    for target in _asking_brains():
        found = target.info(model_name, endpoint)
        if found:
            return found
    return {}


def param_options_for(model_name: str, endpoint: str = "") -> list[dict]:
    """Parameters *model_name* accepts, for step three.

    First answer wins, for the reason ``models_at`` gives: whether a param
    survives is a fact about a particular backend's translation of it.
    """
    for target in _asking_brains():
        found = target.param_options(model_name, endpoint)
        if found:
            return found
    return []


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

    Returning an unusable brain here would shadow a working model injected
    elsewhere, and a turn that fails because a *configured but uninstalled*
    profile won resolution is a confusing way to lose.
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
    """One row per profile, for ``/llm`` and ``/services``.

    ``sandboxed`` is a constant. It distinguished a ``Brain`` from the
    ``NativeBrain`` that wrapped an unmigrated in-process backend; there is no
    such thing now, every backend runs in a box. The key stays because it is
    part of what the ``llm.list`` Request answers with, and a plugin reading a
    field that quietly disappears is worse than one reading a true constant.

    ``params`` is the *resolved* dict rather than either profile key, so a
    caller showing what a profile will send never has to know that reasoning
    effort is spelled one way in config and another on the wire.

    ``param_status`` says which of those params will actually reach the
    provider — ``{name: [supported, note]}``, and ``{}`` for a profile whose
    box is closed, since nothing here opens one to answer. It is the field a
    UI needs to stop presenting a setting that is being discarded as though
    it were in force.
    """
    return [{
        "model_name": name,
        "class": target.backend_name,
        "endpoint": target.base_url,
        "context_size": target.context_size,
        "params": target.params,
        "param_status": {key: list(value)
                         for key, value in target.param_status.items()},
        "loaded": target.loaded,
        "sandboxed": True,
    } for name, target in sorted(brains().items())]
