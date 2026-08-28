"""LLM — the kernel's model authority.

Import from here rather than from the submodules::

    from llm import LLMRequest, LLMResponse, brain, is_context_limit_error

Talking to a model is the one capability Second Brain cannot degrade without:
every other plugin can be absent and the kernel still boots. So the routing
lives in the kernel — which profiles exist, which is the default, which
backend serves each — while the *backends* are installable packages running
in boxes, because a provider SDK is exactly the volatile foreign code the
sandbox exists for.

The contract itself lives in the guest (:mod:`sandbox.guest.llm`), since a
backend is guest code and the child process cannot see the kernel. Kernel
callers reach it from here so they need not care where it physically lives —
the same arrangement :mod:`parsing` has with :mod:`sandbox.guest.parsing`.
"""

from .registry import (DEFAULT_BACKEND, Brain, backend_aliases,
                       backend_display_names, backend_names, brain, brains,
                       default_brain, default_name, describe, discover,
                       info_for, load_default, models_at, param_options_for,
                       providers, refresh, resolve, unload_all, usable_brain)
from sandbox.guest.llm import (BaseLLMBackend, LLMProviderError, LLMRequest,
                               LLMResponse, extract_llm_error_text,
                               is_context_limit_error)

__all__ = [
    "BaseLLMBackend", "Brain", "DEFAULT_BACKEND", "LLMProviderError",
    "LLMRequest", "LLMResponse", "backend_aliases",
    "backend_display_names", "backend_names", "brain", "brains",
    "default_brain", "default_name", "describe", "discover",
    "extract_llm_error_text", "is_context_limit_error", "load_default",
    "info_for", "models_at", "param_options_for", "providers", "refresh",
    "resolve",
    "unload_all", "usable_brain",
]
