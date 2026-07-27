"""The old LLM service — now a compatibility shim over the kernel's ``llm``.

This file used to be the one plugin the kernel could not boot without. It no
longer is: the routing moved into the kernel as :mod:`llm`, and backends
became installable ``helpers/llm_*.py`` files that run in boxes. The kernel
boundary is zero plugin imports because of it.

What is left here exists for one reason — **unmigrated backends still import
it**. The store's ``service_litellm.py`` opens with

    from plugins.services.service_llm import BaseLLM, LLMResponse, ...

and until that file moves to ``helpers/llm_litellm.py`` those names have to
resolve. So the contract types are re-exported from :mod:`llm` (one
definition, no drift) and the parts that genuinely have no new home — native
backend discovery, the router, per-profile service registration — stay as
they were.

Delete this file once every backend has migrated. Nothing in the kernel
imports it.
"""

from dataclasses import dataclass, field
import importlib
import inspect
import os
import logging
import json
import sys
import types

from plugins.BaseService import BaseService
from plugins.helpers.plugin_paths import PLUGIN_ROOTS, plugin_dirs

# One definition of each, living in the guest where a sandboxed backend can
# reach it, re-exported here so an unmigrated one still can too.
from llm import (LLMProviderError, LLMResponse, extract_llm_error_text,
                 is_context_limit_error)

logger = logging.getLogger("LLMClass")


class BaseLLM(BaseService):
    """
    Abstract base class for Large Language Models.

    All methods use the standard OpenAI messages format:
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]

    Subclasses convert to their native format internally.
    """

    def __init__(self):
        """Initialize the base LLM."""
        super().__init__()
        self.shared = True  # LLM clients are typically thread-safe
        self.native_attachment_modalities: set[str] = set()
        # Generalized capability dict so new modalities (image, audio,
        # video, ...) can be added without touching call sites. None
        # means "unknown" — treated as False for routing.
        self.capabilities: dict[str, bool | None] = {
            "image": None,
            "audio": None,
            "video": None,
        }
        self.context_size = None  # Max context window in tokens (auto-detected or from config)
        self.last_prompt_tokens = None
        self.last_cached_prompt_tokens = None

    def has_capability(self, modality: str) -> bool:
        """Return whether capability."""
        return bool(self.capabilities.get(modality))

    def _load(self):
        """Internal helper to load base LLM."""
        raise NotImplementedError

    def unload(self):
        """Handle unload."""
        raise NotImplementedError

    def invoke(self, messages: list[dict], attachments=None, **kwargs) -> LLMResponse:
        """Send messages and return a complete response."""
        raise NotImplementedError

    def stream(self, messages: list[dict], attachments=None, **kwargs):
        """Send messages and yield response chunks."""
        raise NotImplementedError

    def chat_with_tools(self, messages: list[dict], tools: list[dict] = None, **kwargs) -> LLMResponse:
        """Send messages with tool schemas. Returns tool calls or text."""
        raise NotImplementedError

    # Backends that can stream text deltas set this True and override
    # chat_with_tools_streaming. Callers must treat False as "use the
    # blocking call" — the default below is only a structural fallback.
    supports_streaming: bool = False

    # Backends that honor a ``tool_choice`` kwarg on chat_with_tools (forcing
    # the model to call a tool / a specific tool) set this True. When False,
    # the conversation loop never forwards tool_choice and doorway policies
    # that force tools degrade to prompt-level instructions instead.
    supports_tool_choice: bool = False

    def chat_with_tools_streaming(self, messages: list[dict], tools: list[dict] = None,
                                  on_delta=None, **kwargs) -> LLMResponse:
        """Streaming variant of ``chat_with_tools``.

        Implementations call ``on_delta(text_fragment) -> bool`` as assistant
        text arrives; a falsy return aborts the stream (the backend stops
        consuming and returns the partial accumulation). Tool-call deltas are
        accumulated internally — the returned ``LLMResponse`` must be shaped
        exactly like ``chat_with_tools``'s (content + tool_calls + usage), so
        the conversation loop's logic is unchanged. The default falls back to
        the blocking call and never invokes ``on_delta``.
        """
        return self.chat_with_tools(messages, tools, **kwargs)

    # =================================================================
    # ATTACHMENT ROUTING
    # =================================================================

    def _prepare_attachments(self, messages: list[dict], attachments):
        """Apply the 3-tier attachment routing for this LLM's capabilities.

        Returns ``(messages, native_bundle)``:
        - ``messages``: a copy of the input with the suffix appended to
          the last user message (only if a suffix was produced).
        - ``native_bundle``: attachments the model and backend can ingest
          natively. Subclasses serialize these into provider-specific payloads.
        """
        if attachments is None:
            from attachments.attachment import AttachmentBundle
            return messages, AttachmentBundle()
        from attachments.attachment import AttachmentBundle
        bundle = attachments if isinstance(attachments, AttachmentBundle) else AttachmentBundle.from_iterable(attachments)
        if not bundle:
            return messages, AttachmentBundle()
        native_bundle, suffix = bundle.split_for_llm(self.capabilities, self.native_attachment_modalities)
        if not suffix:
            return messages, native_bundle
        out = [m.copy() for m in messages]
        for i in range(len(out) - 1, -1, -1):
            if out[i].get("role") == "user":
                content = out[i].get("content")
                # content can be a str (most cases) or a list of OpenAI
                # content blocks (when the caller pre-built blocks). Both
                # are handled.
                if isinstance(content, list):
                    out[i]["content"] = content + [{"type": "text", "text": "\n\n" + suffix}]
                else:
                    out[i]["content"] = (str(content or "") + "\n\n" + suffix).strip()
                break
        return out, native_bundle


def _cached_prompt_tokens(usage) -> int | None:
    details = getattr(usage, "prompt_tokens_details", None) if usage else None
    return (details.get("cached_tokens") if isinstance(details, dict) else getattr(details, "cached_tokens", None)) if details else None


def _llm_backend_classes() -> dict[str, type[BaseLLM]]:
    backends = {}
    for plugin_dir in plugin_dirs("service"):
        directory = plugin_dir.path
        if not directory.exists():
            continue
        for py_file in sorted(directory.glob(f"{plugin_dir.prefix}*.py")):
            if py_file.stem in {"service_llm"} or py_file.stem.startswith("_"):
                continue
            module_name = plugin_dir.module_name(py_file.stem)
            try:
                module = importlib.import_module(module_name) if plugin_dir.root.built_in else _load_sandbox_backend(py_file, module_name)
            except Exception as e:
                logger.warning(f"Could not inspect LLM backend {py_file.name}: {e}")
                continue
            for _, cls in inspect.getmembers(module, inspect.isclass):
                if cls.__module__ == module.__name__ and issubclass(cls, BaseLLM) and getattr(cls, "is_llm_backend", False):
                    backends[cls.__name__] = cls
    return backends


def llm_backend_names() -> list[str]:
    return sorted(_llm_backend_classes())


def refresh_llm_profile_services(services: dict | None, config: dict | None) -> bool:
    """Resync live profile services after LLM backend files change."""
    if not services or config is None:
        return False
    profiles = config.get("llm_profiles", {}) or {}
    backends = _llm_backend_classes()
    router = services.get("llm")
    default_name = config.get("default_llm_profile") or ""
    changed = False
    for model_name, profile in profiles.items():
        cls_name = profile.get("llm_service_class") or "LiteLLMService"
        cls = backends.get(cls_name)
        current = services.get(model_name)
        if cls is None:
            if isinstance(current, BaseLLM):
                if getattr(current, "loaded", False):
                    current.unload()
                services.pop(model_name, None)
                changed = True
            continue
        if current is None or current.__class__ is not cls:
            was_loaded = getattr(current, "loaded", False)
            if was_loaded:
                current.unload()
            services[model_name] = _build_llm_from_profile(model_name, profile)
            services[model_name].set_peer_services(services)
            if was_loaded or model_name == default_name:
                services[model_name].load()
            changed = True
    if isinstance(router, LLMRouter):
        router.services = services
        router._mirror_active()
    return changed


def _load_sandbox_backend(path, module_name):
    _ensure_external_namespaces(module_name)
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _ensure_external_namespaces(module_name: str):
    root_paths = {root.module: root.path for root in PLUGIN_ROOTS if not root.built_in}
    parts = module_name.split(".")
    root_path = root_paths.get(parts[0])
    if root_path is None:
        return
    for i in range(1, len(parts)):
        name = ".".join(parts[:i])
        path = root_path.joinpath(*parts[1:i])
        module = sys.modules.get(name)
        if module is None:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            module.__package__ = name
            sys.modules[name] = module
        elif hasattr(module, "__path__") and str(path) not in module.__path__:
            module.__path__.append(str(path))


def _build_llm_from_profile(model_name: str, profile: dict) -> BaseLLM:
    """Instantiate an LLM from a profile config dict (does NOT load it).

    The profile dict carries connection metadata only; the model name is
    the dict key in ``llm_profiles`` and is passed in separately.
    """
    cls_name = profile.get("llm_service_class") or "LiteLLMService"

    api_key = profile.get("llm_api_key", "")
    resolved_key = os.environ.get(api_key, api_key) if api_key else None
    base_url = profile.get("llm_endpoint", "") or None

    backends = _llm_backend_classes()
    cls = backends.get(cls_name) or backends.get("LiteLLMService")
    if cls is None:
        raise RuntimeError(f"No LLM backend named {cls_name!r} is installed.")
    llm = cls(model_name, api_key=resolved_key, base_url=base_url)
    llm.capabilities.update({k: v for k, v in (profile.get("llm_capabilities") or {}).items() if k in llm.capabilities})

    ctx = int(profile.get("llm_context_size", 0))
    if ctx > 0:
        llm.context_size = ctx
    return llm


# =====================================================================
# LLM ROUTER (Virtual Proxy)
#
# Registered as the "llm" service. Resolves to whichever LLM the user
# has marked as default (config["default_llm_profile"]) and delegates
# all calls to it. Each LLM is registered as its own service keyed by
# model name; this router is just the convenience handle for "the
# default LLM" — the thing tasks and non-agent code talk to.
# =====================================================================


class LLMRouter(BaseLLM):
    """Default-LLM proxy. Resolves and forwards to the LLM marked as
    ``default_llm_profile`` in config; falls back to the first registered
    LLM if the default is missing or unset.
    """

    config_settings = [
        ("LLM Profiles", "llm_profiles",
         "LLM connection configs keyed by model name.",
         {},
         {"type": "json_dict", "hidden": True}),

        ("Default LLM Profile", "default_llm_profile",
         "Model name of the LLM used when an agent profile says 'default'.",
         "",
         {"type": "text", "hidden": True}),
    ]

    def __init__(self, config: dict, services: dict | None = None):
        """Initialize the llmrouter."""
        super().__init__()
        self.config = config
        # ``services`` is the live service registry. Mutations made by
        # ``add_llm``/``remove_llm`` flow straight into the service dict.
        self.services: dict = services if services is not None else {}
        self.model_name = "LLM Router"

    # --- Resolution ---

    def _llm_keys(self) -> list[str]:
        """Service keys that correspond to LLMs in llm_profiles."""
        profiles = self.config.get("llm_profiles", {}) or {}
        return [name for name in profiles if name in self.services]

    def _resolve_default_name(self) -> str | None:
        """Internal helper to resolve default name."""
        configured = self.config.get("default_llm_profile") or ""
        if configured and configured in self.services:
            return configured
        keys = self._llm_keys()
        if configured and keys:
            logger.warning(
                f"default_llm_profile {configured!r} not registered — "
                f"falling back to {keys[0]!r}"
            )
        return keys[0] if keys else None

    @property
    def active(self) -> BaseLLM | None:
        """Return active."""
        name = self._resolve_default_name()
        return self.services.get(name) if name else None

    # --- LLM management ---

    def add_llm(self, model_name: str, profile_config: dict):
        """Register an LLM in the live service registry."""
        self.services[model_name] = _build_llm_from_profile(model_name, profile_config)

    def remove_llm(self, model_name: str) -> str:
        """Remove LLM."""
        llm = self.services.pop(model_name, None)
        if llm and getattr(llm, "loaded", False):
            llm.unload()
        return f"LLM '{model_name}' removed."

    def list_llms(self) -> list[dict]:
        """List llms."""
        profiles = self.config.get("llm_profiles", {}) or {}
        default_name = self.config.get("default_llm_profile") or ""
        result = []
        for model_name, pconf in profiles.items():
            llm = self.services.get(model_name)
            result.append({
                "model_name": model_name,
                "class": pconf.get("llm_service_class", "LiteLLMService"),
                "endpoint": pconf.get("llm_endpoint", ""),
                "context_size": pconf.get("llm_context_size", 0),
                "default": model_name == default_name,
                "loaded": llm.loaded if llm else False,
            })
        return result

    def _mirror_active(self):
        """Copy key attributes from the resolved default LLM."""
        a = self.active
        if a:
            self.capabilities = dict(a.capabilities)
            self.context_size = a.context_size
            self.loaded = a.loaded
            name = self._resolve_default_name() or "?"
            self.model_name = f"{name} ({a.model_name})"
        else:
            self.loaded = False
            self.model_name = "LLM Router (no LLM configured)"

    # --- BaseLLM interface (delegate to default) ---

    def _load(self):
        """Internal helper to load llmrouter."""
        a = self.active
        if a is None:
            logger.warning("No LLMs configured.")
            return False
        result = a.load()
        self._mirror_active()
        return result

    def unload(self):
        """Handle unload."""
        for model_name in self._llm_keys():
            svc = self.services.get(model_name)
            if getattr(svc, "loaded", False):
                svc.unload()
        self.loaded = False
        self.model_name = "LLM Router"
        logger.info("All LLMs unloaded.")

    def invoke(self, messages, attachments=None, **kwargs):
        """Handle invoke."""
        a = self.active
        if not a or not a.loaded:
            return LLMResponse(
                content="Error: no LLM loaded",
                error="no LLM loaded",
                error_code="not_loaded",
            )
        return a.invoke(messages, attachments, **kwargs)

    def stream(self, messages, attachments=None, **kwargs):
        """Handle stream."""
        a = self.active
        if not a or not a.loaded:
            return
        yield from a.stream(messages, attachments, **kwargs)

    def chat_with_tools(self, messages, tools=None, **kwargs):
        """Handle chat with tools."""
        a = self.active
        if not a or not a.loaded:
            return LLMResponse(
                content="Error: no LLM loaded",
                error="no LLM loaded",
                error_code="not_loaded",
            )
        return a.chat_with_tools(messages, tools, **kwargs)

    @property
    def supports_streaming(self) -> bool:
        """Mirror the active backend's streaming capability."""
        a = self.active
        return bool(a and a.loaded and getattr(a, "supports_streaming", False))

    def chat_with_tools_streaming(self, messages, tools=None, on_delta=None, **kwargs):
        """Handle chat with tools, streaming deltas via ``on_delta``."""
        a = self.active
        if not a or not a.loaded:
            return LLMResponse(
                content="Error: no LLM loaded",
                error="no LLM loaded",
                error_code="not_loaded",
            )
        return a.chat_with_tools_streaming(messages, tools, on_delta=on_delta, **kwargs)


def build_services(config: dict) -> dict:
    """Register one service per LLM (keyed by model name) plus the ``llm``
    router that resolves to the default LLM.
    """

    services: dict = {}
    profiles = config.get("llm_profiles", {}) or {}

    for model_name, pconf in profiles.items():
        # A profile whose backend class is not installed yet (e.g. the LiteLLM
        # backend before `/packages install`) must not abort the whole service
        # build — the ``llm`` router still has to register so dependent tasks
        # see it and so it lights up live once the backend lands.
        try:
            services[model_name] = _build_llm_from_profile(model_name, pconf)
        except Exception as e:
            logger.warning(f"LLM profile '{model_name}' unavailable: {e}")

    # Pick a default LLM if none is set.
    if not config.get("default_llm_profile") and profiles:
        config["default_llm_profile"] = next(iter(profiles))

    services["llm"] = LLMRouter(config, services)
    return services
