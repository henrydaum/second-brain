"""Command registry and shared slash-command parsing."""

from __future__ import annotations

import json
import logging
import shlex
import uuid
from typing import Callable

from plugins.BaseCommand import BaseCommand
from state_machine.conversation import CallableSpec, FormStep

logger = logging.getLogger("Commands")

_HELP_SECTIONS = ["Conversation", "System", "Services & Tools", "Tasks", "Config & System", "Other"]


class CommandRegistry:
    """Command registry."""
    def __init__(self, context_provider: Callable[[str | None], object] | None = None):
        """Initialize the command registry."""
        self._commands: dict[str, BaseCommand] = {}
        self._context_provider = context_provider

    def register(self, entry: BaseCommand):
        """Register command registry."""
        self._commands[entry.name] = entry

    def unregister(self, name: str):
        """Unregister command registry."""
        self._commands.pop(name, None)

    def context(self, session_key: str | None = None):
        """Handle context."""
        ctx = self._context_provider(session_key) if self._context_provider else None
        if ctx is not None:
            try:
                ctx.command_registry = self
            except Exception:
                pass
        return ctx

    def get_completions(self, prefix: str) -> list[BaseCommand]:
        """Get completions."""
        prefix = prefix.lower()
        return sorted([c for c in self._commands.values() if c.name.startswith(prefix)], key=lambda c: c.name)

    def dispatch_dict(self, name: str, args: dict | None = None, *, session_key: str | None = None, _emit: bool = True) -> str | None:
        """Handle dispatch dict."""
        entry = self._commands.get(name)
        if entry is None:
            return f"Unknown command: '/{name}'."
        call_id = None
        if _emit:
            call_id = _emit_started(name, args or {}, session_key)
        try:
            out = entry.run(dict(args or {}), self.context(session_key))
        except Exception as e:
            logger.exception(f"Command '/{name}' handler raised")
            if _emit:
                _emit_finished(name, call_id, session_key, ok=False, error=str(e))
            return f"Command '/{name}' failed: {e}"
        if _emit:
            _emit_finished(name, call_id, session_key, ok=True, error=None)
        return out

    def parse_args(self, name: str, raw: str, *, session_key: str | None = None) -> dict:
        """Parse args."""
        entry = self._commands.get(name)
        if not entry:
            return {}
        ctx = self.context(session_key)
        return parse_command_line(raw, lambda a, c: entry.form(a, c), ctx)

    def all_commands(self) -> list[BaseCommand]:
        """Handle all commands."""
        return sorted(self._commands.values(), key=lambda cmd: cmd.name)

    def visible_commands(self) -> list[BaseCommand]:
        """Handle visible commands."""
        return [cmd for cmd in self.all_commands() if not getattr(cmd, "hide_from_help", False)]

    def to_callable_specs(self) -> dict[str, CallableSpec]:
        """Handle to callable specs."""
        specs = {}
        for entry in self.all_commands():
            specs[entry.name] = CallableSpec(
                entry.name,
                lambda cs, _actor, args, e=entry: self.dispatch_dict(e.name, args, session_key=(cs.cache or {}).get("session_key"), _emit=False),
                form_factory=lambda args, cs, e=entry: e.form(args, self.context((cs.cache or {}).get("session_key") if cs else None)),
                require_approval=getattr(entry, "require_approval", False),
                approval_actor_id=getattr(entry, "approval_actor_id", None),
            )
        return specs

    def help_text(self) -> str:
        """Handle help text."""
        by_cat: dict[str, list[BaseCommand]] = {}
        for cmd in self.visible_commands():
            by_cat.setdefault(cmd.category or "Other", []).append(cmd)
        ordered = [c for c in _HELP_SECTIONS if c in by_cat] + [c for c in by_cat if c not in _HELP_SECTIONS]
        lines = ["Commands:"]
        ctx = self.context(None)
        for cat in ordered:
            lines += ["", f"{cat}:"]
            for cmd in by_cat[cat]:
                hint = _arg_hint_from_form(cmd.form({}, ctx))
                lines.append(f"  {'/' + cmd.name + ((' ' + hint) if hint else ''):<26} {cmd.description}")
        return "\n".join(lines)


def parse_command_line(raw: str, form_factory: Callable[[dict, object], list[FormStep]] | list[FormStep], context=None) -> dict:
    """Parse command line."""
    args, rest = {}, (raw or "").strip()
    while rest:
        steps = form_factory(args, context) if callable(form_factory) else form_factory
        missing = [s for s in steps if s.name not in args]
        if not missing:
            break
        step = missing[0]
        last = len(missing) == 1 and not step.enum
        if not step.required and step.enum:
            token, _ = _peel(rest, last=False, field_type=step.type)
            if token not in step.enum:
                if any(s.required for s in missing[1:]):
                    args[step.name] = step.default
                    continue
                args[step.name], rest = _peel(rest, last=False, field_type=step.type)
                continue
        if not step.required and step.type in {"boolean", "bool"}:
            token, _ = _peel(rest, last=False, field_type="string")
            if str(token).strip().lower() not in {"true", "yes", "1", "y", "false", "no", "0", "n"} and any(s.required for s in missing[1:]):
                args[step.name] = step.default
                continue
        value, rest = _peel(rest, last=last and step.type == "string", field_type=step.type)
        args[step.name] = step.coerce(value)
    for step in (form_factory(args, context) if callable(form_factory) else form_factory):
        if step.name not in args and not step.required:
            args[step.name] = step.default
    return args


def format_command_call(name: str, args: dict | None = None) -> str:
    """Format command call."""
    parts = ["/" + str(name or "").strip().lstrip("/")]
    for value in (args or {}).values():
        if value is None:
            continue
        text = json.dumps(value, separators=(",", ":")) if isinstance(value, (dict, list)) else str(value)
        parts.append(shlex.quote(text))
    return " ".join(parts)


def _peel(rest: str, *, last: bool, field_type: str) -> tuple[object, str]:
    """Internal helper to handle peel."""
    rest = rest.strip()
    if not rest:
        return "", ""
    if field_type in {"object", "array"} or rest[0] in "{[":
        value, end = json.JSONDecoder().raw_decode(rest)
        return value, rest[end:].strip()
    if last:
        return rest, ""
    lex = shlex.shlex(rest, posix=True)
    lex.whitespace_split = True
    token = next(lex)
    return token, rest[lex.instream.tell():].strip()


def _arg_hint_from_form(form: list[FormStep]) -> str:
    """Internal helper to handle arg hint from form."""
    out = []
    for step in form or []:
        name = step.name
        out.append(f"<{name}>" if step.required else f"[{name}]")
    return " ".join(out)


def _emit_started(name: str, args: dict, session_key: str | None):
    """Internal helper to emit started."""
    from events.event_bus import bus
    from events.event_channels import COMMAND_CALL_STARTED
    call_id = f"cmd:{name}:{uuid.uuid4().hex[:8]}"
    bus.emit(COMMAND_CALL_STARTED, {"session_key": session_key, "call_id": call_id, "command_name": name, "args": dict(args or {})})
    return call_id


def _emit_finished(name: str, call_id: str | None, session_key: str | None, *, ok: bool, error: str | None):
    """Internal helper to emit finished."""
    if not call_id:
        return
    from events.event_bus import bus
    from events.event_channels import COMMAND_CALL_FINISHED
    bus.emit(COMMAND_CALL_FINISHED, {"session_key": session_key, "call_id": call_id, "command_name": name, "ok": ok, "error": error})
