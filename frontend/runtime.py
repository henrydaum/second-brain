from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field

from agent.agent import Agent
from agent.history_utils import heal_orphan_tool_calls
from agent.system_prompt import build_system_prompt
from runtime.agent_scope import load_scope, resolve_agent_llm, scoped_db, scoped_registry
from events.event_bus import bus
from events.event_channels import APPROVAL_REQUESTED, APPROVAL_RESOLVED, CHAT_MESSAGE_PUSHED
from frontend.commands import CommandRegistry, active_agent_line, register_core_commands
from frontend.dispatch import InputResult, route_input
from frontend.presenter import FrontendPresenter
from frontend.types import FrontendEvent, FrontendSession

logger = logging.getLogger("FrontendRuntime")


@dataclass
class FrontendRuntimeState:
    conversation_id: int | None = None
    agent: Agent | None = None
    busy: bool = False
    busy_gen: int = 0
    prompt_suffix: str = ""
    status_ids: list[str] = field(default_factory=list)
    last_session: FrontendSession | None = None
    active_agent_announced: bool = False


class FrontendRuntime:
    def __init__(self, ctrl, services, config, tool_registry, root_dir):
        self.ctrl = ctrl
        self.services = services
        self.config = config
        self.tool_registry = tool_registry
        self.root_dir = root_dir
        self.presenter = FrontendPresenter()
        self.adapters = {}
        self._states = {}
        self._approval_lock = threading.RLock()
        self._pending_approvals = {}
        self._pending_approval_order = {}
        bus.subscribe(APPROVAL_REQUESTED, self._on_approval_requested)
        bus.subscribe(APPROVAL_RESOLVED, self._on_approval_resolved)
        bus.subscribe(CHAT_MESSAGE_PUSHED, self._on_chat_message_pushed)

    def register_adapter(self, adapter):
        self.adapters[adapter.name] = adapter
        adapter.bind_runtime(self)
        self._pending_approvals.setdefault(adapter.name, {})
        self._pending_approval_order.setdefault(adapter.name, [])

    def session_key(self, session: FrontendSession) -> tuple[str, str, str, str | None]:
        return (session.platform, str(session.user_id), str(session.chat_id), session.thread_id)

    def get_state(self, session: FrontendSession) -> FrontendRuntimeState:
        key = self.session_key(session)
        state = self._states.get(key)
        if state is None:
            state = FrontendRuntimeState(
                conversation_id=session.conversation_id,
                last_session=session,
            )
            self._states[key] = state
        else:
            state.last_session = session
            if session.conversation_id is not None:
                state.conversation_id = session.conversation_id
        return state

    def get_last_session(self, platform: str) -> FrontendSession | None:
        for (name, *_), state in reversed(list(self._states.items())):
            if name == platform and state.last_session is not None:
                return state.last_session
        adapter = self.adapters.get(platform)
        return adapter.default_session() if adapter else None

    def create_registry(self, session: FrontendSession, overrides=None, refresh_agent=None) -> CommandRegistry:
        state = self.get_state(session)
        registry = CommandRegistry()

        def _set_conversation_id(conv_id):
            state.conversation_id = conv_id
            if conv_id is None:
                state.active_agent_announced = True

        register_core_commands(
            registry, self.ctrl, self.services, self.tool_registry, self.root_dir,
            get_agent=lambda: state.agent,
            set_conversation_id=_set_conversation_id,
            refresh_agent=refresh_agent or (lambda: self.refresh_agent(session)),
            rescope_agents=self.rescope_all_agents,
        )
        for entry in overrides or []:
            registry.register(entry)
        return registry

    def set_prompt_suffix(self, session: FrontendSession, prompt_suffix: str = ""):
        self.get_state(session).prompt_suffix = prompt_suffix or ""

    def _active_scope(self):
        """Resolve the scope for the currently active agent profile, if any.

        Returns ``None`` when no profile is active or the profile declares
        no restrictions — in that case the main agent uses the unrestricted
        tool registry and database directly.
        """
        profile_name = self.config.get("active_agent_profile")
        if not profile_name:
            return None
        try:
            scope = load_scope(profile_name, self.config)
        except ValueError as e:
            logger.warning(f"Invalid scope for profile '{profile_name}': {e}")
            return None
        if not (scope.has_tool_filter or scope.has_table_filter or scope.prompt_suffix):
            return None
        return scope

    def build_agent(self, session: FrontendSession) -> Agent | None:
        profile_name = self.config.get("active_agent_profile") or "default"
        llm = resolve_agent_llm(profile_name, self.config, self.services)
        if llm is None or not getattr(llm, "loaded", False):
            return None
        caps = self.adapters[session.platform].capabilities
        scope = self._active_scope()
        # Scope the DB and tool registry up-front; both fall back to the
        # unrestricted originals when the active profile has no filters.
        agent_db = scoped_db(self.ctrl.db, scope) if scope else self.ctrl.db
        agent_registry = scoped_registry(self.tool_registry, scope) if scope else self.tool_registry

        def _system_prompt():
            base = build_system_prompt(
                agent_db, self.ctrl.orchestrator, agent_registry, self.ctrl.services
            )
            suffix = self.get_state(session).prompt_suffix
            scope_suffix = ("\n\n" + scope.prompt_suffix) if (scope and scope.prompt_suffix) else ""
            return base + suffix + scope_suffix

        return Agent(
            llm, agent_registry, self.config,
            system_prompt=_system_prompt,
            on_message=lambda msg: self._persist_message(session, msg),
            on_tool_start=lambda tool_name: self._on_tool_start(session, tool_name, caps),
            on_tool_result=lambda tool_name, result: self._on_tool_result(session, tool_name, result, caps),
            on_notice=lambda text: self.send_action(session, self.presenter.notice(text)),
        )

    def ensure_agent(self, session: FrontendSession) -> Agent | None:
        state = self.get_state(session)
        if state.agent is None:
            state.agent = self.build_agent(session)
        return state.agent

    def refresh_agent(self, session: FrontendSession) -> Agent | None:
        state = self.get_state(session)
        state.agent = self.build_agent(session)
        state.status_ids.clear()
        return state.agent

    def rescope_agent(self, session: FrontendSession) -> Agent | None:
        """Rebuild the agent for a session while preserving its chat history.

        Used after ``/agent switch`` so the same conversation carries over to
        the new profile but now runs against the new scope's db and toolset.
        """
        state = self.get_state(session)
        preserved_history = list(state.agent.history) if state.agent else []
        state.agent = self.build_agent(session)
        state.status_ids.clear()
        if state.agent is not None and preserved_history:
            state.agent.history = preserved_history
            heal_orphan_tool_calls(state.agent.history)
        return state.agent

    def rescope_all_agents(self) -> None:
        """Rescope every session's agent. Called after the global active
        profile changes so subsequent messages on any frontend pick up the
        new scope immediately."""
        for key, state in self._states.items():
            session = state.last_session
            if session is None or state.agent is None:
                continue
            self.rescope_agent(session)

    def begin_turn(self, session: FrontendSession) -> int | None:
        state = self.get_state(session)
        if state.busy:
            return None
        gen = state.busy_gen
        state.busy = True
        return gen

    def end_turn(self, session: FrontendSession, gen: int | None = None):
        state = self.get_state(session)
        if gen is None or state.busy_gen == gen:
            state.busy = False

    def force_unbusy(self, session: FrontendSession):
        state = self.get_state(session)
        state.busy_gen += 1
        state.busy = False

    def is_busy(self, session: FrontendSession) -> bool:
        return bool(self.get_state(session).busy)

    def route_event(self, session: FrontendSession, registry: CommandRegistry, text: str,
                    image_paths: list[str] | None = None, prompt_suffix: str = ""):
        self.set_prompt_suffix(session, prompt_suffix)
        state = self.get_state(session)
        should_announce = state.conversation_id is None and not state.active_agent_announced
        agent = self.ensure_agent(session)
        result = route_input(text, registry, agent, image_paths=image_paths)
        if should_announce and result.type == "chat" and result.text:
            result.text = f"{active_agent_line(self.config)}\n\n{result.text}"
            state.active_agent_announced = True
        return result

    def handle_frontend_event(self, event: FrontendEvent, registry: CommandRegistry,
                              prompt_suffix: str = "") -> InputResult:
        if event.type == "chat_message":
            return self.route_event(event.session, registry, event.text, prompt_suffix=prompt_suffix)
        if event.type == "attachment_message":
            return self.route_event(
                event.session,
                registry,
                event.text,
                image_paths=list(event.payload.get("image_paths") or []),
                prompt_suffix=prompt_suffix,
            )
        if event.type == "slash_command":
            command_name, command_arg = self._event_command_parts(event)
            return InputResult("command", registry.dispatch(command_name, command_arg) or "")
        if event.type == "approval_response":
            approved = bool(event.payload.get("approved"))
            request_id = event.callback_id or event.payload.get("request_id")
            if request_id:
                ok = self.resolve_approval(
                    request_id, approved,
                    resolved_by=event.payload.get("resolved_by") or event.session.platform,
                )
                if not ok:
                    return InputResult("command", "Expired or already handled.")
            elif not self.resolve_next_approval(event.session, approved):
                return InputResult("command", "No pending approvals.")
            return InputResult("command", "Approval granted." if approved else "Approval denied.")
        if event.type == "callback_response":
            callback_kind = str(event.payload.get("kind") or "").strip().lower()
            if callback_kind == "command":
                return self.handle_frontend_event(FrontendEvent(
                    type="slash_command",
                    session=event.session,
                    command_name=event.payload.get("command_name"),
                    command_arg=event.payload.get("command_arg"),
                ), registry, prompt_suffix=prompt_suffix)
            if callback_kind == "history":
                return self.handle_frontend_event(FrontendEvent(
                    type="slash_command",
                    session=event.session,
                    command_name="history",
                    command_arg=str(event.payload.get("conversation_id") or event.callback_value or ""),
                ), registry, prompt_suffix=prompt_suffix)
            if callback_kind == "approval":
                return self.handle_frontend_event(FrontendEvent(
                    type="approval_response",
                    session=event.session,
                    callback_id=event.payload.get("request_id") or event.callback_id,
                    payload={
                        "approved": event.payload.get("approved"),
                        "resolved_by": event.payload.get("resolved_by") or event.session.platform,
                    },
                ), registry, prompt_suffix=prompt_suffix)
        return InputResult("error", f"Unsupported frontend event: {event.type}")

    def reset_session(self, session: FrontendSession):
        state = self.get_state(session)
        state.conversation_id = None
        state.active_agent_announced = True
        if state.agent:
            state.agent.reset()

    def send_action(self, session: FrontendSession, action):
        adapter = self.adapters.get(session.platform)
        if adapter:
            adapter.send_action(session, action)

    def list_history_action(self, session: FrontendSession, limit: int = 10):
        conversations = self.ctrl.db.list_user_conversations(limit=limit)
        if not conversations:
            return None
        return self.presenter.history_menu(conversations, self.adapters[session.platform].capabilities)

    def render_tool_result(self, result):
        from frontend.formatters import format_tool_result
        return format_tool_result(result)

    @staticmethod
    def _event_command_parts(event: FrontendEvent) -> tuple[str, str]:
        if event.command_name:
            return event.command_name.strip().lower(), (event.command_arg or "").strip()
        text = (event.text or "").strip()
        if text.startswith("/"):
            text = text[1:]
        parts = text.split(maxsplit=1)
        return (parts[0].lower() if parts else "", parts[1].strip() if len(parts) > 1 else "")

    def resolve_approval(self, request_id: str, approved: bool, resolved_by: str | None = None) -> bool:
        with self._approval_lock:
            req = None
            for adapter_name, pending in self._pending_approvals.items():
                req = pending.get(request_id)
                if req is not None:
                    break
            if req is None or req.is_resolved:
                return False
            if resolved_by:
                req.metadata["resolved_by"] = resolved_by
            req.resolve(approved)
            return True

    def resolve_next_approval(self, session: FrontendSession, approved: bool) -> bool:
        adapter_name = session.platform
        with self._approval_lock:
            order = self._pending_approval_order.get(adapter_name, [])
            pending = self._pending_approvals.get(adapter_name, {})
            while order and (order[0] not in pending or pending[order[0]].is_resolved):
                order.pop(0)
            if not order:
                return False
            req_id = order.pop(0)
            req = pending.get(req_id)
            if req is None or req.is_resolved:
                return False
            req.metadata["resolved_by"] = adapter_name
            req.resolve(approved)
            return True

    def _persist_message(self, session: FrontendSession, msg: dict):
        state = self.get_state(session)
        role = msg.get("role", "")
        content = msg.get("content") or ""
        if state.conversation_id is None:
            title = (content[:80].replace("\n", " ").strip()
                     if role == "user" else "New conversation")
            state.conversation_id = self.ctrl.db.create_conversation(title)
        save_content = content
        if msg.get("tool_calls"):
            save_content = json.dumps({
                "content": content,
                "tool_calls": msg["tool_calls"],
            })
        self.ctrl.db.save_message(
            state.conversation_id, role, save_content,
            tool_call_id=msg.get("tool_call_id"),
            tool_name=msg.get("name"),
        )
        if role == "assistant" and not msg.get("tool_calls"):
            self.ctrl.maybe_generate_conversation_title_async(state.conversation_id)

    def _on_tool_start(self, session: FrontendSession, tool_name: str, caps):
        action = self.presenter.tool_started(tool_name, caps)
        if action.status_id:
            self.get_state(session).status_ids.append(action.status_id)
        self.send_action(session, action)

    def _on_tool_result(self, session: FrontendSession, tool_name: str, result, caps):
        state = self.get_state(session)
        status_id = state.status_ids.pop(0) if state.status_ids else None
        for action in self.presenter.tool_finished(tool_name, result, caps, status_id):
            self.send_action(session, action)

    def _on_approval_requested(self, req):
        with self._approval_lock:
            for name, adapter in self.adapters.items():
                session = adapter.default_session() or self.get_last_session(name)
                if session is None:
                    continue
                self._pending_approvals[name][req.id] = req
                self._pending_approval_order[name].append(req.id)
                self.send_action(session, self.presenter.approval_request(req, adapter.capabilities))

    def _on_approval_resolved(self, req):
        resolved_by = req.metadata.get("resolved_by")
        approved = getattr(req, "approved", False)
        with self._approval_lock:
            notify = []
            for adapter_name, pending in self._pending_approvals.items():
                had_pending = req.id in pending
                pending.pop(req.id, None)
                order = self._pending_approval_order.get(adapter_name, [])
                self._pending_approval_order[adapter_name] = [item for item in order if item != req.id]
                if had_pending:
                    notify.append(adapter_name)
        for adapter_name in notify:
            adapter = self.adapters.get(adapter_name)
            if adapter is None:
                continue
            session = adapter.default_session() or self.get_last_session(adapter_name)
            if session is None:
                continue
            self.send_action(session, self.presenter.approval_resolved(req, approved, resolved_by, adapter_name))

    def _on_chat_message_pushed(self, payload: dict):
        for name, adapter in self.adapters.items():
            if not adapter.capabilities.supports_proactive_push:
                continue
            session = self.get_last_session(name) or adapter.default_session()
            if session is None:
                continue
            self.send_action(session, self.presenter.pushed_message(payload or {}, adapter.capabilities))

    def restore_agent_history(self, session: FrontendSession, messages: list[dict]):
        agent = self.ensure_agent(session)
        if agent is None:
            return
        agent_history = []
        for msg in messages:
            role = msg["role"]
            if role == "system":
                continue
            content = msg["content"] or ""
            if role == "assistant":
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict) and "tool_calls" in parsed:
                        agent_history.append({
                            "role": "assistant",
                            "content": parsed.get("content"),
                            "tool_calls": parsed["tool_calls"],
                        })
                        continue
                except (json.JSONDecodeError, TypeError):
                    pass
                agent_history.append({"role": "assistant", "content": content})
            elif role == "tool":
                agent_history.append({
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id"),
                    "content": content,
                })
            else:
                agent_history.append({"role": role, "content": content})
        heal_orphan_tool_calls(agent_history)
        agent.history = agent_history
