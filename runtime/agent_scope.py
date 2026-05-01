"""
Agent scope - per-profile lens over the LLM prompt and tool registry.

An agent profile stores an LLM reference plus optional prompt/tool scope.
Data-source narrowing intentionally lives in custom tools now;
the shared runtime only decides which tools the agent can see and call.
"""

import inspect
import re
from dataclasses import dataclass

from agent.tool_registry import ToolRegistry

_CALL_TOOL_RE = re.compile(r'context\.call_tool\(\s*["\']([^"\']+)["\']')


@dataclass
class AgentScope:
    profile_name: str
    prompt_suffix: str = ""
    tools_allow: set[str] | None = None
    tools_deny: set[str] | None = None

    @property
    def has_tool_filter(self) -> bool:
        return self.tools_allow is not None or bool(self.tools_deny)


def load_scope(profile_name: str, config: dict) -> AgentScope:
    """Parse a profile's supported scope fields into an ``AgentScope``."""
    profile = config.get("agent_profiles", {}).get(profile_name, {}) or {}
    tools_mode = _scope_mode(profile_name, profile)
    tools_list = _scope_list(profile_name, profile)

    return AgentScope(
        profile_name=profile_name,
        prompt_suffix=str(profile.get("prompt_suffix") or ""),
        tools_allow=set(tools_list) if tools_mode == "whitelist" else None,
        tools_deny=set(tools_list) if tools_mode == "blacklist" else None,
    )


def _scope_mode(profile_name: str, profile: dict) -> str:
    key = "whitelist_or_blacklist_tools"
    mode = profile.get(key, "blacklist")
    if mode not in ("whitelist", "blacklist"):
        raise ValueError(
            f"Profile '{profile_name}' has invalid {key}: {mode!r}. "
            "Use 'whitelist' or 'blacklist'."
        )
    return mode


def _scope_list(profile_name: str, profile: dict) -> list:
    key = "tools_list"
    value = profile.get(key, [])
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise ValueError(f"Profile '{profile_name}' field {key} must be an array.")
    return value


def scoped_registry(base_registry: ToolRegistry, scope: AgentScope, db=None) -> ToolRegistry:
    """Return a ``ToolRegistry`` that only exposes tools the scope allows."""
    target_db = base_registry.db if db is None else db
    if not scope.has_tool_filter and target_db is base_registry.db:
        return base_registry

    new_registry = ToolRegistry(target_db, base_registry.config, base_registry.services)
    new_registry.orchestrator = base_registry.orchestrator
    new_registry.is_subagent = base_registry.is_subagent

    if not scope.has_tool_filter:
        allowed_names = set(base_registry.tools.keys())
    elif scope.tools_allow is not None:
        allowed_names = set(scope.tools_allow)
    else:
        allowed_names = {n for n in base_registry.tools if n not in (scope.tools_deny or set())}
    allowed_names = _expand_tool_dependencies(base_registry.tools, allowed_names)

    for name, tool in base_registry.tools.items():
        if name in allowed_names:
            new_registry.tools[name] = tool
    return new_registry


def resolve_agent_llm(profile_name: str, config: dict, services: dict):
    """Resolve the LLM service an agent profile should use."""
    profile = (config.get("agent_profiles", {}) or {}).get(profile_name, {}) or {}
    llm_ref = profile.get("llm") or "default"
    if llm_ref == "default":
        llm_ref = config.get("default_llm_profile") or ""
    if not llm_ref:
        return services.get("llm")
    return services.get(llm_ref) or services.get("llm")


def _expand_tool_dependencies(tools: dict, names: set[str]) -> set[str]:
    expanded, pending = set(names), list(names)
    while pending:
        tool = tools.get(pending.pop())
        if tool is None:
            continue
        try:
            source = inspect.getsource(tool.__class__)
        except (OSError, TypeError):
            continue
        for dep in _CALL_TOOL_RE.findall(source):
            if dep in tools and dep not in expanded:
                expanded.add(dep)
                pending.append(dep)
    return expanded
