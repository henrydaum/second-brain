"""
Telegram bot frontend for Second Brain.

Provides a chat-first mobile interface with auto-ready agent behavior and
slash command autocomplete. The render_files tool sends files as Telegram
media groups; other tools do not auto-render attachment paths.
Runs on a daemon thread with its own asyncio event loop.
"""

import asyncio
import html
import json
import logging
import re
import threading
from pathlib import Path

from pipeline.attachment_cache import save as save_attachment
from plugins.services.helpers.parser_registry import get_modality
from frontend.commands import CommandEntry, new_conversation_message
from frontend.formatters import (
    format_services, format_tasks, format_tools,
    format_tool_result,
)
from frontend.platforms.platform_telegram import TelegramPlatformAdapter
from frontend.runtime import FrontendRuntime
from agent.subagent_runtime import (
    SUBAGENT_RUN_CHANNEL,
    SUBAGENT_NOTIFICATION_MODES,
    SUBAGENT_DEFAULT_NOTIFICATION_MODE,
)
from frontend.telegram.forms import (
    LLM_ADD_PARAMS,
    SCHEDULE_CREATE_STEPS,
    PendingParamForm,
    PendingScheduleCreate,
    agent_add_params,
    agent_edit_field_param,
    coerce_param_value,
    llm_edit_field_param,
    schema_to_params,
)
from frontend.telegram.renderers import prepare_media_actions
from frontend.telegram.transport import TelegramTransport
from frontend.types import FrontendEvent, FrontendSession

logger = logging.getLogger("Telegram")

_MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB Telegram bot API limit
_MAX_ATTACHMENT_TEXT = 4000         # Max chars of parsed text to append from attachments

_TELEGRAM_SUFFIX = (
    "\n\n## Telegram frontend\n"
    "You are connected via the Telegram mobile app. Keep responses concise.\n"
    "Telegram supports: **bold**, *italic*, `inline code`, and ```code blocks```.\n"
    "Do NOT use markdown tables, headers (#), horizontal rules (---), or bullet "
    "lists with -. Use plain numbered lists or line breaks for structure.\n"
    "The user can send you images and documents. Images are passed to you directly. "
    "Text and tabular files are parsed and their content is appended to the message. "
    "All attachments are saved to an attachment cache folder (indexed by the Second Brain "
    "pipeline), so you can re-read or search them later via read_file and the search tools."
)


# ── Markdown → Telegram HTML converter ──────────────────────────────

def _md_to_tg_html(text: str) -> str:
    """Convert common markdown to Telegram-compatible HTML.

    Handles fenced code blocks, inline code, bold, and italic.
    Escapes HTML entities in non-markup text.
    """
    parts = []
    # Split on fenced code blocks first
    code_block_re = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
    last_end = 0

    for m in code_block_re.finditer(text):
        # Process text before this code block
        parts.append(_convert_inline(text[last_end:m.start()]))
        lang = m.group(1)
        code = html.escape(m.group(2).rstrip())
        if lang:
            parts.append(f'<pre><code class="language-{html.escape(lang)}">{code}</code></pre>')
        else:
            parts.append(f"<pre>{code}</pre>")
        last_end = m.end()

    # Process remaining text after last code block
    parts.append(_convert_inline(text[last_end:]))
    return "".join(parts)


def _convert_inline(text: str) -> str:
    """Convert inline markdown (code, bold, italic) to HTML, escaping the rest."""
    # Split on inline code first to avoid processing markdown inside code
    result = []
    code_re = re.compile(r"`([^`]+)`")
    last_end = 0

    for m in code_re.finditer(text):
        result.append(_convert_bold_italic(text[last_end:m.start()]))
        result.append(f"<code>{html.escape(m.group(1))}</code>")
        last_end = m.end()

    result.append(_convert_bold_italic(text[last_end:]))
    return "".join(result)


def _convert_bold_italic(text: str) -> str:
    """Convert **bold** and *italic* to HTML, escape HTML entities."""
    escaped = html.escape(text)
    # Bold: **text** (non-greedy)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    # Italic: *text* (non-greedy, not preceded/followed by *)
    escaped = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", escaped)
    return escaped


def run_telegram_bot(ctrl, shutdown_fn, shutdown_event: threading.Event,
                     tool_registry, services, config, root_dir: Path,
                     runtime: FrontendRuntime | None = None,
                     adapter: TelegramPlatformAdapter | None = None):
    """Launch the Telegram bot. Blocks until shutdown_event is set."""

    adapter = adapter or TelegramPlatformAdapter(
        ctrl, shutdown_fn, shutdown_event, tool_registry, services, config, root_dir
    )
    runtime = runtime or FrontendRuntime(ctrl, services, config, tool_registry, root_dir)
    if adapter.runtime is None:
        runtime.register_adapter(adapter)

    token = config.get("telegram_bot_token", "").strip()
    if not token:
        logger.info("telegram_bot_token not configured — Telegram frontend disabled.")
        return

    # Late imports so the dependency is only required when the frontend is enabled
    from telegram import (
        BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Update,
    )
    from telegram.constants import ChatAction
    from telegram.ext import (
        Application, CallbackQueryHandler, MessageHandler, filters,
    )

    # ── State ────────────────────────────────────────────────────────
    base_session = adapter.default_session() or FrontendSession("telegram", "0", "0")
    _pending_calls: dict[int, PendingParamForm] = {}
    _pending_configures: dict = {}      # chat_id -> setting_key (waiting for value)
    _pending_llm_adds: dict[int, PendingParamForm] = {}
    _pending_agent_adds: dict[int, PendingParamForm] = {}
    # Single-field edit forms — subject is the profile/model name and the
    # single FormParam in `params` carries the field name.
    _pending_llm_edits: dict[int, PendingParamForm] = {}
    _pending_agent_edits: dict[int, PendingParamForm] = {}
    _pending_triggers: dict[int, PendingParamForm] = {}
    _loop: asyncio.AbstractEventLoop | None = None
    _app: Application | None = None
    transport = TelegramTransport(adapter, lambda: _app, lambda: _loop, _md_to_tg_html)

    def _session(chat_id: int | None = None) -> FrontendSession:
        if chat_id is None:
            return runtime.get_last_session("telegram") or base_session
        return FrontendSession(platform="telegram", user_id=str(chat_id), chat_id=str(chat_id))

    async def _dispatch_frontend_event(event: FrontendEvent, prompt_suffix: str = ""):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: runtime.handle_frontend_event(event, registry, prompt_suffix=prompt_suffix)
        )

    def _create_agent():
        runtime.set_prompt_suffix(base_session, _TELEGRAM_SUFFIX)
        if runtime.ensure_agent(base_session) is not None:
            logger.info("Agent ready (Telegram).")
            return True
        return False

    def _refresh_agent():
        """Called by /refresh — rebuild every session's agent and release the
        busy flag so the next message routes immediately, even if the previous
        handler is still spinning in a stuck tool. Rebuilding picks up the
        latest scope, tools, and tables for every chat."""
        runtime.set_prompt_suffix(base_session, _TELEGRAM_SUFFIX)
        runtime.rescope_all_agents()
        runtime.force_unbusy(base_session)
        transport.clear_statuses()
        logger.info("Agent refreshed (Telegram).")

    # ── Security ─────────────────────────────────────────────────────

    def _check_user(update: Update) -> bool:
        allowed = int(config.get("telegram_allowed_user_id", 0))
        if allowed and update.effective_user and update.effective_user.id != allowed:
            return False
        return True

    # ── Command registry ─────────────────────────────────────────────

    registry = runtime.create_registry(base_session, refresh_agent=_refresh_agent)

    # Telegram-specific overrides
    def _load_handler(arg):
        if not arg:
            return None  # Dynamic menu will handle this
        result = ctrl.load_service(arg)
        if arg == "llm" and runtime.ensure_agent(base_session) is None:
            _create_agent()
        return result

    def _unload_handler(arg):
        if not arg:
            return None  # Dynamic menu will handle this
        result = ctrl.unload_service(arg)
        if arg == "llm":
            runtime.get_state(base_session).agent = None
        return result

    def _new_handler(_arg):
        runtime.reset_session(base_session)
        return new_conversation_message(config)

    for entry in [
        CommandEntry("load", "Load a service", "<service_name>",
                     handler=_load_handler,
                     arg_completions=lambda: sorted(services.keys()),
                     category="Services & Tools"),
        CommandEntry("unload", "Unload a service", "<service_name>",
                     handler=_unload_handler,
                     arg_completions=lambda: sorted(services.keys()),
                     category="Services & Tools"),
        CommandEntry("new", "Start a new conversation", handler=_new_handler,
                     category="Conversation"),
        CommandEntry("start", "Welcome message",
                     handler=lambda _: (
                         f"Second Brain is online."
                         "Send a message to chat, or /help for commands."
                     ),
                     hide_from_help=True),
        # Compact formatter overrides (used when called with an arg or via pickers).
        CommandEntry("services", "List services and status",
                     handler=lambda _: format_services(ctrl.list_services(), compact=True),
                     category="Services & Tools"),
        CommandEntry("tasks", "List path-driven and event-driven tasks",
                     handler=lambda _: format_tasks(ctrl.list_tasks(), compact=True),
                     category="Tasks"),
        CommandEntry("tools", "List registered tools",
                     handler=lambda _: format_tools(ctrl.list_tools(), compact=True),
                     category="Services & Tools"),
    ]:
        registry.register(entry)

    # ── Helpers ──────────────────────────────────────────────────────

    adapter.set_sender(transport.dispatch_runtime_action)

    # ── Interactive /call form ───────────────────────────────────────

    def _get_tool_params(tool_name: str):
        """Extract parameter info from a tool's JSON schema."""
        tool = tool_registry.tools.get(tool_name)
        if not tool:
            return None
        return schema_to_params(tool.parameters or {})

    def _get_trigger_params(task_name: str):
        """Extract interactive trigger params from an event task schema."""
        task = ctrl.orchestrator.tasks.get(task_name)
        if not task:
            return None
        if getattr(task, "trigger", "path") != "event":
            return None
        return schema_to_params(getattr(task, "event_payload_schema", {}) or {})

    async def _ask_next_form_param(chat_id: int, forms: dict[int, PendingParamForm], done, prefix: str):
        state = forms.get(chat_id)
        if not state:
            return
        param = state.current_param
        if param is None:
            await done(chat_id)
            return
        adapter.send_action(
            _session(chat_id),
            runtime.presenter.form_field(
                param.name, param.type, param.description,
                param.required, param.enum, prefix, param.name,
            ),
        )

    async def _ask_next_param(chat_id: int):
        await _ask_next_form_param(chat_id, _pending_calls, _execute_pending_call, "call")

    async def _execute_pending_call(chat_id: int):
        """Execute a completed /call form."""
        state = _pending_calls.pop(chat_id, None)
        if not state:
            return
        tool_name = state.subject
        kwargs = state.collected
        await _app.bot.send_message(chat_id, f"Calling {tool_name}...", disable_notification=True)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: ctrl.call_tool(tool_name, kwargs))
        logger.info(f"tool: {tool_name} [{'ok' if result.success else 'fail'}]")
        if result.attachment_paths:
            await transport.execute_send_actions(chat_id, prepare_media_actions(result.attachment_paths))
            if result.llm_summary:
                await transport.send_long_message(chat_id, result.llm_summary)
        else:
            output = format_tool_result(result)
            await transport.send_long_message(chat_id, output)

    async def _ask_next_trigger_param(chat_id: int):
        await _ask_next_form_param(chat_id, _pending_triggers, _execute_pending_trigger, "trigger")

    async def _execute_pending_trigger(chat_id: int):
        """Execute a completed /trigger form."""
        state = _pending_triggers.pop(chat_id, None)
        if not state:
            return
        task_name = state.subject
        payload = state.collected
        await _app.bot.send_message(chat_id, f"Triggering {task_name}...", disable_notification=True)
        loop = asyncio.get_running_loop()
        output = await loop.run_in_executor(
            None, lambda: ctrl.trigger_event_task(task_name, payload))
        if output:
            await transport.send_long_message(chat_id, output)

    async def _start_trigger_form(chat_id: int, task_name: str) -> bool:
        """Begin interactive /trigger payload collection for an event task."""
        params = _get_trigger_params(task_name)
        if params is None:
            await _app.bot.send_message(chat_id, f"Unknown or non-event task: {task_name}")
            return True
        if not params:
            loop = asyncio.get_running_loop()
            output = await loop.run_in_executor(
                None, lambda: ctrl.trigger_event_task(task_name, {}))
            if output:
                await transport.send_long_message(chat_id, output)
            return True
        _pending_triggers[chat_id] = PendingParamForm(subject=task_name, params=params)
        await _app.bot.send_message(
            chat_id,
            f"<b>{html.escape(task_name)}</b> — fill in the trigger payload:\n"
            f"Send /skip for optional params, /cancel to abort.",
            parse_mode="HTML")
        await _ask_next_trigger_param(chat_id)
        return True

    async def _start_call_form(chat_id: int, tool_name: str):
        """Begin interactive /call parameter collection for a tool."""
        params = _get_tool_params(tool_name)
        if params is None:
            await _app.bot.send_message(chat_id, f"Unknown tool: {tool_name}")
            return
        if not params:
            # No parameters — execute immediately
            await _app.bot.send_message(chat_id, f"Calling {tool_name}...", disable_notification=True)
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, lambda: ctrl.call_tool(tool_name, {}))
            if result.attachment_paths:
                await transport.execute_send_actions(chat_id, prepare_media_actions(result.attachment_paths))
                if result.llm_summary:
                    await transport.send_long_message(chat_id, result.llm_summary)
            else:
                output = format_tool_result(result)
                await transport.send_long_message(chat_id, output)
            return
        _pending_calls[chat_id] = PendingParamForm(subject=tool_name, params=params)
        await _app.bot.send_message(
            chat_id,
            f"<b>{html.escape(tool_name)}</b> — fill in parameters:\n"
            f"Send /skip for optional params, /cancel to abort.",
            parse_mode="HTML")
        await _ask_next_param(chat_id)

    async def _show_configure_menu(chat_id: int):
        """Show inline keyboard with all config settings."""
        from config.config_data import SETTINGS_DATA
        from plugins.plugin_discovery import get_plugin_settings

        all_settings = list(SETTINGS_DATA) + list(get_plugin_settings())
        buttons = []
        for title, key, _desc, _default, _type_info in all_settings:
            # Skip settings with dedicated commands (/agent, /schedule).
            if isinstance(_type_info, dict) and _type_info.get("hidden") is True:
                continue
            # Telegram callback_data max 64 bytes — use key directly
            if len(f"cfg:{key}") <= 64:
                buttons.append([InlineKeyboardButton(title, callback_data=f"cfg:{key}")])
        if not buttons:
            await _app.bot.send_message(chat_id, "No settings available.")
            return
        await _app.bot.send_message(
            chat_id, "Choose a setting to configure:",
            reply_markup=InlineKeyboardMarkup(buttons))

    async def _ask_configure_value(chat_id: int, key: str):
        """Show current value and ask for new value."""
        from config.config_data import SETTINGS_DATA
        from plugins.plugin_discovery import get_plugin_settings

        all_settings = {k: (t, d, ti) for t, k, d, _, ti in
                        list(SETTINGS_DATA) + list(get_plugin_settings())}
        info = all_settings.get(key)
        if not info:
            await _app.bot.send_message(chat_id, f"Unknown setting: {key}")
            return

        title, desc, type_info = info
        current = config.get(key)
        widget_type = type_info.get("type", "text")

        # Bool → inline keyboard
        if widget_type == "bool":
            buttons = [[
                InlineKeyboardButton("True", callback_data=f"cfgval:{key}:true"),
                InlineKeyboardButton("False", callback_data=f"cfgval:{key}:false"),
            ]]
            await _app.bot.send_message(
                chat_id,
                f"<b>{html.escape(title)}</b>\n"
                f"{html.escape(desc)}\n\n"
                f"Current: <code>{html.escape(str(current))}</code>",
                reply_markup=InlineKeyboardMarkup(buttons),
                parse_mode="HTML")
            return

        # Everything else → ask for text input
        _pending_configures[chat_id] = (key, widget_type)
        current_str = json.dumps(current, default=str) if isinstance(current, (list, dict)) else str(current)

        if widget_type == "json_list":
            hint = "Send each item on its own line, e.g.:\n<code>first item\nsecond item</code>"
        elif widget_type == "slider":
            hint = "Type a number."
        else:
            hint = "Type your value as plain text (no quotes needed)."

        await _app.bot.send_message(
            chat_id,
            f"<b>{html.escape(title)}</b>\n"
            f"{html.escape(desc)}\n\n"
            f"Current: <code>{html.escape(current_str)}</code>\n\n"
            f"{hint}\n\nSend the new value (or /cancel):",
            parse_mode="HTML")

    # ── /llm add interactive form ───────────────────────────────────

    async def _start_llm_add_form(chat_id: int, model_name: str):
        """Begin interactive /llm add parameter collection."""
        _pending_llm_adds[chat_id] = PendingParamForm(subject=model_name, params=LLM_ADD_PARAMS)
        await _app.bot.send_message(
            chat_id,
            f"<b>New LLM: {html.escape(model_name)}</b>\n"
            f"Fill in the connection parameters below.\n"
            f"Send /skip for optional params, /cancel to abort.",
            parse_mode="HTML")
        await _ask_next_llm_param(chat_id)

    async def _ask_next_llm_param(chat_id: int):
        await _ask_next_form_param(chat_id, _pending_llm_adds, _execute_llm_add, "llmadd")

    async def _execute_llm_add(chat_id: int):
        """Finish the /llm add form and register the LLM."""
        state = _pending_llm_adds.pop(chat_id, None)
        if not state:
            return
        model_name = state.subject
        collected = state.collected

        for param in LLM_ADD_PARAMS:
            if param.name not in collected:
                collected[param.name] = param.default

        try:
            collected["llm_context_size"] = int(collected.get("llm_context_size", 0))
        except (ValueError, TypeError):
            collected["llm_context_size"] = 0

        result = await _dispatch_frontend_event(FrontendEvent(
            type="slash_command",
            session=_session(chat_id),
            command_name="llm",
            command_arg=f"add {model_name} {json.dumps(collected)}",
        ))
        if result.text:
            await transport.send_long_message(chat_id, result.text)

    # ── /agent add interactive form ─────────────────────────────────

    async def _start_agent_add_form(chat_id: int, profile_name: str):
        """Begin interactive /agent add parameter collection."""
        llm_choices = ["default"] + sorted((config.get("llm_profiles", {}) or {}).keys())
        tool_names = sorted(tool_registry.tools.keys())
        table_names = [r["name"] for r in ctrl.db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        _pending_agent_adds[chat_id] = PendingParamForm(
            subject=profile_name, params=agent_add_params(llm_choices, tool_names, table_names))
        await _app.bot.send_message(
            chat_id,
            f"<b>New agent profile: {html.escape(profile_name)}</b>\n"
            f"Pick an LLM and configure scope below.\n"
            f"Send /skip for optional params, /cancel to abort.",
            parse_mode="HTML")
        await _ask_next_agent_param(chat_id)

    async def _ask_next_agent_param(chat_id: int):
        await _ask_next_form_param(chat_id, _pending_agent_adds, _execute_agent_add, "agtadd")

    async def _execute_agent_add(chat_id: int):
        """Finish the /agent add form and register the agent profile."""
        state = _pending_agent_adds.pop(chat_id, None)
        if not state:
            return
        profile_name = state.subject
        collected = state.collected

        cleaned = {
            "llm": collected.get("llm") or "default",
            "prompt_suffix": collected.get("prompt_suffix") or "",
            "whitelist_or_blacklist_tools": collected.get("whitelist_or_blacklist_tools") or "blacklist",
            "tools_list": collected.get("tools_list") or [],
            "whitelist_or_blacklist_tables": collected.get("whitelist_or_blacklist_tables") or "blacklist",
            "tables_list": collected.get("tables_list") or [],
            "whitelist_or_blacklist_folders": collected.get("whitelist_or_blacklist_folders") or "blacklist",
            "folders_list": collected.get("folders_list") or [],
        }

        result = await _dispatch_frontend_event(FrontendEvent(
            type="slash_command",
            session=_session(chat_id),
            command_name="agent",
            command_arg=f"add {profile_name} {json.dumps(cleaned)}",
        ))
        if result.text:
            await transport.send_long_message(chat_id, result.text)

    # ── /llm edit + /agent edit single-field forms ──────────────────

    _AGENT_FIELDS = ("llm", "prompt_suffix",
                     "whitelist_or_blacklist_tools", "tools_list",
                     "whitelist_or_blacklist_tables", "tables_list",
                     "whitelist_or_blacklist_folders", "folders_list")
    _LLM_FIELDS = ("llm_endpoint", "llm_api_key",
                   "llm_context_size", "llm_service_class")

    def _format_value_short(value) -> str:
        if value is None or value == "":
            return "(none)"
        if isinstance(value, list):
            if not value:
                return "(none)"
            return ", ".join(str(v) for v in value)
        if isinstance(value, str) and len(value) > 60:
            return value[:57] + "…"
        return str(value)

    def _agent_attribute_lines(profile: dict) -> list[str]:
        lines = []
        llm_ref = profile.get("llm") or "default"
        if llm_ref == "default":
            resolved = config.get("default_llm_profile") or "?"
            lines.append(f"  llm: default (→ {html.escape(resolved)})")
        else:
            lines.append(f"  llm: {html.escape(llm_ref)}")
        lines.append(f"  prompt_suffix: {html.escape(_format_value_short(profile.get('prompt_suffix')))}")
        for field in ("whitelist_or_blacklist_tools", "tools_list",
                      "whitelist_or_blacklist_tables", "tables_list",
                      "whitelist_or_blacklist_folders", "folders_list"):
            lines.append(f"  {field}: {html.escape(_format_value_short(profile.get(field)))}")
        return lines

    def _llm_attribute_lines(profile: dict) -> list[str]:
        masked = dict(profile)
        api_key = masked.get("llm_api_key") or ""
        if api_key and not (api_key.startswith("$") or api_key.isupper()):
            masked["llm_api_key"] = "****" if len(api_key) <= 8 else f"{api_key[:4]}...{api_key[-4:]}"
        lines = []
        for field in _LLM_FIELDS:
            lines.append(f"  {field}: {html.escape(_format_value_short(masked.get(field)))}")
        return lines

    async def _show_agent_profile_actions(chat_id: int, name: str):
        profiles = config.get("agent_profiles", {}) or {}
        profile = profiles.get(name)
        if profile is None:
            await _app.bot.send_message(chat_id, f"Unknown agent profile: '{name}'.")
            return
        active = (config.get("active_agent_profile") or "default") == name
        header = f"<b>{html.escape(name)}</b>"
        if active:
            header += "  (active)"
        text = "\n".join([header] + _agent_attribute_lines(profile))
        rows = []
        if not active:
            rows.append([InlineKeyboardButton("Set active", callback_data=f"agnt:switch:{name}")])
        rows.append([InlineKeyboardButton("Edit", callback_data=f"agnt:edit:{name}")])
        if name != "default":
            rows.append([InlineKeyboardButton("Remove", callback_data=f"agnt:remove:{name}")])
        rows.append([InlineKeyboardButton("◀ Back", callback_data="agnt:back")])
        await _app.bot.send_message(
            chat_id, text, reply_markup=InlineKeyboardMarkup(rows), parse_mode="HTML")

    async def _show_llm_profile_actions(chat_id: int, name: str):
        profiles = config.get("llm_profiles", {}) or {}
        profile = profiles.get(name)
        if profile is None:
            await _app.bot.send_message(chat_id, f"Unknown LLM: '{name}'.")
            return
        is_default = (config.get("default_llm_profile") or "") == name
        header = f"<b>{html.escape(name)}</b>"
        if is_default:
            header += "  (default)"
        text = "\n".join([header] + _llm_attribute_lines(profile))
        rows = []
        if not is_default:
            rows.append([InlineKeyboardButton("Set as default", callback_data=f"llm:default:{name}")])
        rows.append([InlineKeyboardButton("Edit", callback_data=f"llm:edit:{name}")])
        rows.append([InlineKeyboardButton("Remove", callback_data=f"llm:remove:{name}")])
        rows.append([InlineKeyboardButton("◀ Back", callback_data="llm:back")])
        await _app.bot.send_message(
            chat_id, text, reply_markup=InlineKeyboardMarkup(rows), parse_mode="HTML")

    async def _show_agent_profiles_list(chat_id: int):
        profiles = config.get("agent_profiles", {}) or {}
        active = config.get("active_agent_profile") or "default"
        rows = []
        for name in sorted(profiles.keys()):
            label = f"{'* ' if name == active else '  '}{name}"
            rows.append([InlineKeyboardButton(label, callback_data=f"agnt:pick:{name}")])
        rows.append([InlineKeyboardButton("+ Add new profile", callback_data="agnt:add")])
        await _app.bot.send_message(
            chat_id, "Agent profiles:",
            reply_markup=InlineKeyboardMarkup(rows))

    async def _show_llm_list(chat_id: int):
        profiles = config.get("llm_profiles", {}) or {}
        default_llm = config.get("default_llm_profile") or ""
        rows = []
        for name in sorted(profiles.keys()):
            label = f"{'* ' if name == default_llm else '  '}{name}"
            rows.append([InlineKeyboardButton(label, callback_data=f"llm:pick:{name}")])
        rows.append([InlineKeyboardButton("+ Add new LLM", callback_data="llm:add")])
        await _app.bot.send_message(
            chat_id, "LLMs:",
            reply_markup=InlineKeyboardMarkup(rows))

    async def _show_agent_edit_fields(chat_id: int, name: str):
        profiles = config.get("agent_profiles", {}) or {}
        profile = profiles.get(name)
        if profile is None:
            await _app.bot.send_message(chat_id, f"Unknown agent profile: '{name}'.")
            return
        text = "\n".join([f"<b>Edit {html.escape(name)}</b>"] + _agent_attribute_lines(profile))
        rows = [
            [InlineKeyboardButton("llm", callback_data=f"agnt:editfield:{name}:llm"),
             InlineKeyboardButton("prompt_suffix", callback_data=f"agnt:editfield:{name}:prompt_suffix")],
            [InlineKeyboardButton("tool mode", callback_data=f"agnt:editfield:{name}:whitelist_or_blacklist_tools"),
             InlineKeyboardButton("tools_list", callback_data=f"agnt:editfield:{name}:tools_list")],
            [InlineKeyboardButton("table mode", callback_data=f"agnt:editfield:{name}:whitelist_or_blacklist_tables"),
             InlineKeyboardButton("tables_list", callback_data=f"agnt:editfield:{name}:tables_list")],
            [InlineKeyboardButton("folder mode", callback_data=f"agnt:editfield:{name}:whitelist_or_blacklist_folders"),
             InlineKeyboardButton("folders_list", callback_data=f"agnt:editfield:{name}:folders_list")],
            [InlineKeyboardButton("◀ Back", callback_data=f"agnt:pick:{name}")],
        ]
        await _app.bot.send_message(
            chat_id, text, reply_markup=InlineKeyboardMarkup(rows), parse_mode="HTML")

    async def _show_llm_edit_fields(chat_id: int, name: str):
        profiles = config.get("llm_profiles", {}) or {}
        profile = profiles.get(name)
        if profile is None:
            await _app.bot.send_message(chat_id, f"Unknown LLM: '{name}'.")
            return
        text = "\n".join([f"<b>Edit {html.escape(name)}</b>"] + _llm_attribute_lines(profile))
        rows = [
            [InlineKeyboardButton("llm_endpoint", callback_data=f"llm:editfield:{name}:llm_endpoint"),
             InlineKeyboardButton("llm_api_key", callback_data=f"llm:editfield:{name}:llm_api_key")],
            [InlineKeyboardButton("llm_context_size", callback_data=f"llm:editfield:{name}:llm_context_size"),
             InlineKeyboardButton("llm_service_class", callback_data=f"llm:editfield:{name}:llm_service_class")],
            [InlineKeyboardButton("◀ Back", callback_data=f"llm:pick:{name}")],
        ]
        await _app.bot.send_message(
            chat_id, text, reply_markup=InlineKeyboardMarkup(rows), parse_mode="HTML")

    async def _start_agent_field_edit(chat_id: int, name: str, field: str):
        if field not in _AGENT_FIELDS:
            await _app.bot.send_message(chat_id, f"Unknown field: {field}")
            return
        llm_choices = ["default"] + sorted((config.get("llm_profiles", {}) or {}).keys())
        tool_names = sorted(tool_registry.tools.keys())
        try:
            table_names = [r["name"] for r in ctrl.db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        except Exception:
            table_names = []
        param = agent_edit_field_param(field, llm_choices, tool_names, table_names)
        _pending_agent_edits[chat_id] = PendingParamForm(subject=name, params=[param])
        adapter.send_action(
            _session(chat_id),
            runtime.presenter.form_field(
                param.name, param.type, param.description,
                param.required, param.enum, "agtedit", param.name,
            ),
        )

    async def _start_llm_field_edit(chat_id: int, name: str, field: str):
        if field not in _LLM_FIELDS:
            await _app.bot.send_message(chat_id, f"Unknown field: {field}")
            return
        param = llm_edit_field_param(field)
        _pending_llm_edits[chat_id] = PendingParamForm(subject=name, params=[param])
        adapter.send_action(
            _session(chat_id),
            runtime.presenter.form_field(
                param.name, param.type, param.description,
                param.required, param.enum, "llmedit", param.name,
            ),
        )

    async def _execute_agent_field_edit(chat_id: int):
        state = _pending_agent_edits.pop(chat_id, None)
        if not state or not state.params:
            return
        name = state.subject
        field = state.params[0].name
        value = state.collected.get(field)
        # JSON-encode so the command parser receives a single token.
        raw = json.dumps(value) if value is not None else "null"
        result = await _dispatch_frontend_event(FrontendEvent(
            type="slash_command",
            session=_session(chat_id),
            command_name="agent",
            command_arg=f"edit {name} {field} {raw}",
        ))
        if result.text:
            await transport.send_long_message(chat_id, result.text)
        await _show_agent_edit_fields(chat_id, name)

    async def _execute_llm_field_edit(chat_id: int):
        state = _pending_llm_edits.pop(chat_id, None)
        if not state or not state.params:
            return
        name = state.subject
        field = state.params[0].name
        value = state.collected.get(field)
        raw = json.dumps(value) if value is not None else "null"
        result = await _dispatch_frontend_event(FrontendEvent(
            type="slash_command",
            session=_session(chat_id),
            command_name="llm",
            command_arg=f"edit {name} {field} {raw}",
        ))
        if result.text:
            await transport.send_long_message(chat_id, result.text)
        await _show_llm_edit_fields(chat_id, name)

    # ── Picker menus for list commands ───────────────────────────────
    #
    # Pattern: /<cmd> with no arg → compact text status + picker buttons.
    # Tapping a picker row opens an action menu; tapping an action dispatches
    # the underlying registry command and refreshes the picker.
    #
    # Callback prefix scheme:
    #   svc:pick:<name>      — open service action menu
    #   svc:load:<name>      — load, then refresh picker
    #   svc:unload:<name>    — unload, then refresh picker
    #   svc:back             — return to picker
    #   tsk:pick:<name>      — open task action menu
    #   tsk:{pause|unpause|reset|retry|trigger}:<name>
    #   tsk:back
    #   tool:pick:<name>     — open tool action menu
    #   tool:call:<name>
    #   tool:back
    #   loc:<filter>         — /locations tools|tasks|services|all
    #   sch:pick:<name>      — open job action menu
    #   sch:{run|enable|disable|show|delete|confirmdel}:<name>
    #   sch:new              — start job create form
    #   sch:back             — return to picker

    def _services_picker_keyboard():
        """Build an inline picker for /services: one row per service."""
        buttons = []
        for s in ctrl.list_services():
            marker = "✓" if s["loaded"] else "·"
            buttons.append([InlineKeyboardButton(
                f"{marker} {s['name']}", callback_data=f"svc:pick:{s['name']}")])
        return InlineKeyboardMarkup(buttons) if buttons else None

    async def _show_services_picker(chat_id: int, header: str | None = None):
        text = header or format_services(ctrl.list_services(), compact=True)
        kb = _services_picker_keyboard()
        if kb is None:
            await _app.bot.send_message(chat_id, "No services registered.")
            return
        await _app.bot.send_message(chat_id, text, reply_markup=kb)

    async def _show_service_actions(chat_id: int, name: str):
        services_list = {s["name"]: s for s in ctrl.list_services()}
        svc = services_list.get(name)
        if not svc:
            await _app.bot.send_message(
                chat_id, f"Unknown service: '{name}'. Run /services to see all services.")
            return
        status = "Loaded" if svc["loaded"] else "Unloaded"
        model = f"  ({svc['model_name']})" if svc["model_name"] else ""
        text = f"<b>{html.escape(name)}</b>{html.escape(model)}\nStatus: {status}"
        action_button = (InlineKeyboardButton("Unload", callback_data=f"svc:unload:{name}")
                         if svc["loaded"]
                         else InlineKeyboardButton("Load", callback_data=f"svc:load:{name}"))
        kb = InlineKeyboardMarkup([
            [action_button],
            [InlineKeyboardButton("◀ Back", callback_data="svc:back")],
        ])
        await _app.bot.send_message(chat_id, text, reply_markup=kb, parse_mode="HTML")

    def _tasks_picker_keyboard():
        """Build an inline picker for /tasks: one row per task with state abbr."""
        buttons = []
        for task in ctrl.list_tasks():
            counts = task.get("counts", {})
            running = counts.get("PROCESSING", 0)
            failed = counts.get("FAILED", 0)
            if failed:
                marker = "✗"
            elif running:
                marker = "▶"
            elif task.get("paused"):
                marker = "⏸"
            else:
                marker = "·"
            buttons.append([InlineKeyboardButton(
                f"{marker} {task['name']}", callback_data=f"tsk:pick:{task['name']}")])
        return InlineKeyboardMarkup(buttons) if buttons else None

    async def _show_tasks_picker(chat_id: int):
        tasks = ctrl.list_tasks()
        if not tasks:
            await _app.bot.send_message(chat_id, "No tasks registered.")
            return
        running = sum(t["counts"].get("PROCESSING", 0) for t in tasks)
        failed = sum(t["counts"].get("FAILED", 0) for t in tasks)
        summary = f"Tasks — {running} running, {failed} failed"
        kb = _tasks_picker_keyboard()
        await _app.bot.send_message(chat_id, summary, reply_markup=kb)

    async def _show_task_actions(chat_id: int, name: str):
        task = next((t for t in ctrl.list_tasks() if t["name"] == name), None)
        if task is None:
            await _app.bot.send_message(
                chat_id, f"Unknown task: '{name}'. Run /tasks to see all tasks.")
            return
        counts = task["counts"]
        trigger = task.get("trigger", "path")
        paused = task.get("paused", False)
        lines = [
            f"<b>{html.escape(name)}</b>",
            f"Trigger: {trigger}" + ("  (paused)" if paused else ""),
            (f"Pending: {counts['PENDING']}  "
             f"Running: {counts['PROCESSING']}  "
             f"Done: {counts['DONE']}  "
             f"Failed: {counts['FAILED']}"),
        ]
        text = "\n".join(lines)

        pause_btn = (InlineKeyboardButton("Unpause", callback_data=f"tsk:unpause:{name}")
                     if paused
                     else InlineKeyboardButton("Pause", callback_data=f"tsk:pause:{name}"))
        rows = [[pause_btn]]
        if trigger == "path":
            rows.append([
                InlineKeyboardButton("Reset", callback_data=f"tsk:reset:{name}"),
                InlineKeyboardButton("Retry failed", callback_data=f"tsk:retry:{name}"),
            ])
        elif trigger == "event":
            rows.append([InlineKeyboardButton(
                "Trigger…", callback_data=f"tsk:trigger:{name}")])
        rows.append([InlineKeyboardButton("◀ Back", callback_data="tsk:back")])

        await _app.bot.send_message(
            chat_id, text, reply_markup=InlineKeyboardMarkup(rows), parse_mode="HTML")

    def _tools_picker_keyboard():
        """Build an inline picker for /tools: one row per tool."""
        buttons = []
        for t in ctrl.list_tools():
            buttons.append([InlineKeyboardButton(
                t["name"], callback_data=f"tool:pick:{t['name']}")])
        return InlineKeyboardMarkup(buttons) if buttons else None

    async def _show_tools_picker(chat_id: int):
        tools = ctrl.list_tools()
        if not tools:
            await _app.bot.send_message(chat_id, "No tools registered.")
            return
        summary = f"Tools — {len(tools)} registered"
        kb = _tools_picker_keyboard()
        await _app.bot.send_message(chat_id, summary, reply_markup=kb)

    async def _show_tool_actions(chat_id: int, name: str):
        tool = next((t for t in ctrl.list_tools() if t["name"] == name), None)
        if tool is None:
            await _app.bot.send_message(
                chat_id, f"Unknown tool: '{name}'. Run /tools to see all tools.")
            return
        desc = (tool.get("description") or "").split("\n")[0]
        if len(desc) > 300:
            desc = desc[:297] + "..."
        text = f"<b>{html.escape(name)}</b>\n{html.escape(desc)}"
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("Call…", callback_data=f"tool:call:{name}")],
            [InlineKeyboardButton("◀ Back", callback_data="tool:back")],
        ])
        await _app.bot.send_message(chat_id, text, reply_markup=kb, parse_mode="HTML")

    async def _show_locations_picker(chat_id: int):
        kb = InlineKeyboardMarkup([[
            InlineKeyboardButton("All", callback_data="loc:all"),
            InlineKeyboardButton("Tools", callback_data="loc:tools"),
        ], [
            InlineKeyboardButton("Tasks", callback_data="loc:tasks"),
            InlineKeyboardButton("Services", callback_data="loc:services"),
        ]])
        await _app.bot.send_message(chat_id, "Choose a location view:", reply_markup=kb)

    # ── /schedule picker + create form ───────────────────────────────

    _pending_schedule_creates: dict[int, PendingScheduleCreate] = {}

    def _schedule_jobs_picker_keyboard():
        """Build the /schedule picker: one row per job + [+ New job]."""
        timekeeper = services.get("timekeeper")
        if timekeeper is None or not getattr(timekeeper, "loaded", False):
            return None
        jobs = timekeeper.list_jobs()
        buttons = []
        for name, job in sorted(jobs.items()):
            marker = "✓" if job.get("enabled", True) else "·"
            buttons.append([InlineKeyboardButton(
                f"{marker} {name}", callback_data=f"sch:pick:{name}")])
        buttons.append([InlineKeyboardButton("+ New job", callback_data="sch:new")])
        return InlineKeyboardMarkup(buttons)

    async def _show_schedule_picker(chat_id: int):
        from frontend.formatters import format_scheduled_jobs
        timekeeper = services.get("timekeeper")
        if timekeeper is None or not getattr(timekeeper, "loaded", False):
            await _app.bot.send_message(
                chat_id,
                "Timekeeper service is not loaded. Run /load timekeeper to start it.")
            return
        text = format_scheduled_jobs(timekeeper.list_jobs(), timekeeper)
        kb = _schedule_jobs_picker_keyboard()
        await _app.bot.send_message(chat_id, text, reply_markup=kb)

    async def _show_schedule_job_actions(chat_id: int, name: str):
        timekeeper = services.get("timekeeper")
        if timekeeper is None:
            await _app.bot.send_message(chat_id, "Timekeeper service is not loaded.")
            return
        job = timekeeper.get_job(name)
        if job is None:
            await _app.bot.send_message(
                chat_id, f"Unknown job: '{name}'. Run /schedule list to see all jobs.")
            return
        enabled = job.get("enabled", True)
        try:
            schedule_desc = timekeeper.describe_job(name)
        except Exception:
            schedule_desc = job.get("cron") or job.get("run_at") or "?"
        title = (job.get("payload", {}).get("title") or "").strip()
        lines = [
            f"<b>{html.escape(name)}</b>",
            f"Status: {'Enabled' if enabled else 'Disabled'}",
            f"Schedule: {html.escape(schedule_desc)}",
        ]
        if title:
            lines.append(f"Title: {html.escape(title)}")
        text = "\n".join(lines)

        toggle_btn = (InlineKeyboardButton("Disable", callback_data=f"sch:disable:{name}")
                      if enabled
                      else InlineKeyboardButton("Enable", callback_data=f"sch:enable:{name}"))
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("▶ Run now", callback_data=f"sch:run:{name}"),
             toggle_btn],
            [InlineKeyboardButton("Show", callback_data=f"sch:show:{name}"),
             InlineKeyboardButton("Delete", callback_data=f"sch:delete:{name}")],
            [InlineKeyboardButton("◀ Back", callback_data="sch:back")],
        ])
        await _app.bot.send_message(chat_id, text, reply_markup=kb, parse_mode="HTML")

    async def _start_schedule_create(chat_id: int):
        _pending_schedule_creates[chat_id] = PendingScheduleCreate()
        await _app.bot.send_message(
            chat_id,
            "<b>New scheduled job</b>\n"
            "Answer each prompt. Send /cancel to abort.",
            parse_mode="HTML")
        await _ask_schedule_step(chat_id)

    async def _ask_schedule_step(chat_id: int):
        state = _pending_schedule_creates.get(chat_id)
        if not state:
            return
        if state.step >= len(SCHEDULE_CREATE_STEPS):
            await _finalize_schedule_create(chat_id)
            return
        field = state.current_field(SCHEDULE_CREATE_STEPS)
        collected = state.collected

        if field == "job_name":
            await _app.bot.send_message(
                chat_id,
                "<b>job_name</b> (required)\n"
                "A unique identifier, e.g. <code>daily_digest</code>.",
                parse_mode="HTML")
            return

        if field == "schedule_type":
            kb = InlineKeyboardMarkup([[
                InlineKeyboardButton("Recurring (cron)", callback_data="sch:type:recurring"),
                InlineKeyboardButton("One-time", callback_data="sch:type:one_time"),
            ]])
            await _app.bot.send_message(
                chat_id, "<b>schedule_type</b> (required)",
                reply_markup=kb, parse_mode="HTML")
            return

        if field == "schedule_value":
            if collected.get("schedule_type") == "one_time":
                await _app.bot.send_message(
                    chat_id,
                    "<b>run_at</b> (required)\n"
                    "ISO datetime, e.g. <code>2026-04-20T09:00:00</code>.",
                    parse_mode="HTML")
            else:
                await _app.bot.send_message(
                    chat_id,
                    "<b>cron</b> (required)\n"
                    "Cron expression, e.g. <code>0 8 * * *</code> (daily at 08:00).",
                    parse_mode="HTML")
            return

        if field == "channel":
            seen = {SUBAGENT_RUN_CHANNEL}
            timekeeper = services.get("timekeeper")
            if timekeeper and getattr(timekeeper, "loaded", False):
                for job in timekeeper.list_jobs().values():
                    ch = (job.get("channel") or "").strip()
                    if ch:
                        seen.add(ch)
            rows = [[InlineKeyboardButton(ch, callback_data=f"sch:ch:{ch}")]
                    for ch in sorted(seen)]
            rows.append([InlineKeyboardButton("Other…", callback_data="sch:ch:__other__")])
            await _app.bot.send_message(
                chat_id,
                "<b>channel</b> (required)\n"
                "The event bus channel to emit when this job fires.",
                reply_markup=InlineKeyboardMarkup(rows), parse_mode="HTML")
            return

        if field == "prompt":
            if collected.get("channel") != SUBAGENT_RUN_CHANNEL:
                state.step += 1
                await _ask_schedule_step(chat_id)
                return
            await _app.bot.send_message(
                chat_id,
                "<b>prompt</b> (required for subagent jobs)\n"
                "The instruction sent to the subagent, e.g. "
                "<code>Summarize yesterday's inbox into 5 bullet points.</code>",
                parse_mode="HTML")
            return

        if field == "agent":
            if collected.get("channel") != SUBAGENT_RUN_CHANNEL:
                state.step += 1
                await _ask_schedule_step(chat_id)
                return
            active = config.get("active_agent_profile") or "default"
            profiles = sorted((config.get("agent_profiles", {}) or {}).keys())
            rows = [[InlineKeyboardButton(
                f"{'* ' if name == active else ''}{name}",
                callback_data=f"sch:agent:{name}")]
                for name in profiles]
            rows.append([InlineKeyboardButton("Use active profile at run time", callback_data="sch:agent:__active__")])
            await _app.bot.send_message(
                chat_id,
                "<b>agent</b> (optional)\n"
                "Choose which agent profile this scheduled subagent should run under.",
                reply_markup=InlineKeyboardMarkup(rows), parse_mode="HTML")
            return

        if field == "notifications":
            if collected.get("channel") != SUBAGENT_RUN_CHANNEL:
                state.step += 1
                await _ask_schedule_step(chat_id)
                return
            rows = [[InlineKeyboardButton(
                f"{'* ' if mode == SUBAGENT_DEFAULT_NOTIFICATION_MODE else ''}{mode}",
                callback_data=f"sch:notif:{mode}")]
                for mode in SUBAGENT_NOTIFICATION_MODES]
            await _app.bot.send_message(
                chat_id,
                "<b>notifications</b> (required for subagent jobs)\n"
                "How chatty the subagent should be when it fires.\n"
                "<code>all</code> — pushes regularly; final answer is auto-pushed if forgotten.\n"
                "<code>important</code> — push tool available, told to use it only for noteworthy findings.\n"
                "<code>off</code> — runs silently, no chat output (final answer still stored).",
                reply_markup=InlineKeyboardMarkup(rows), parse_mode="HTML")
            return

        if field == "title":
            await _app.bot.send_message(
                chat_id,
                "<b>title</b> (optional — send /skip to omit)\n"
                "Short label shown when the job fires.",
                parse_mode="HTML")
            return

    async def _finalize_schedule_create(chat_id: int):
        state = _pending_schedule_creates.pop(chat_id, None)
        if not state:
            return
        c = state.collected

        definition: dict = {
            "channel": c.get("channel", "").strip(),
            "one_time": c.get("schedule_type") == "one_time",
            "payload": {},
            "enabled": True,
        }
        if definition["one_time"]:
            definition["run_at"] = c.get("schedule_value")
        else:
            definition["cron"] = c.get("schedule_value")
        payload = definition["payload"]
        if c.get("channel") == SUBAGENT_RUN_CHANNEL:
            payload["prompt"] = c.get("prompt", "")
            if c.get("agent"):
                payload["agent"] = c["agent"]
            payload["notifications"] = (
                c.get("notifications") or SUBAGENT_DEFAULT_NOTIFICATION_MODE
            )
        if c.get("title"):
            payload["title"] = c["title"]

        timekeeper = services.get("timekeeper")
        if timekeeper is None:
            await _app.bot.send_message(chat_id, "Timekeeper service is not loaded.")
            return

        try:
            timekeeper.create_job(c["job_name"], definition)
        except ValueError as e:
            # Roll back to the relevant step so the user can retry.
            await _app.bot.send_message(chat_id, f"Failed to create job: {e}")
            # Re-open the create form at the schedule_value step so the user can fix
            # the most common failure (bad cron / duplicate name).
            _pending_schedule_creates[chat_id] = state
            state.step = 0
            await _ask_schedule_step(chat_id)
            return

        await _app.bot.send_message(
            chat_id, f"Job '{c['job_name']}' created.")
        await _show_schedule_picker(chat_id)

    # ── Handlers ─────────────────────────────────────────────────────

    async def handle_command(update: Update, _ctx):
        """Handle /slash commands via the shared registry."""
        if not _check_user(update):
            return

        text = update.message.text or ""
        parts = text.split(maxsplit=1)
        cmd_name = parts[0][1:].split("@")[0].lower() if parts else ""
        arg = parts[1].strip() if len(parts) > 1 else ""
        chat_id = update.message.chat_id
        logger.info(f"← /{cmd_name}{' ' + arg if arg else ''}")

        # /cancel — clear any pending form
        if cmd_name == "cancel":
            cancelled = False
            if chat_id in _pending_calls:
                _pending_calls.pop(chat_id)
                cancelled = True
            if chat_id in _pending_configures:
                _pending_configures.pop(chat_id)
                cancelled = True
            if chat_id in _pending_llm_adds:
                _pending_llm_adds.pop(chat_id)
                cancelled = True
            if chat_id in _pending_agent_adds:
                _pending_agent_adds.pop(chat_id)
                cancelled = True
            if chat_id in _pending_llm_edits:
                _pending_llm_edits.pop(chat_id)
                cancelled = True
            if chat_id in _pending_agent_edits:
                _pending_agent_edits.pop(chat_id)
                cancelled = True
            if chat_id in _pending_triggers:
                _pending_triggers.pop(chat_id)
                cancelled = True
            if chat_id in _pending_schedule_creates:
                _pending_schedule_creates.pop(chat_id)
                cancelled = True
            if cancelled:
                await update.message.reply_text("Cancelled.")
                return

        # /history — show inline keyboard of recent conversations
        if cmd_name == "history" and not arg:
            action = runtime.list_history_action(_session(chat_id), limit=10)
            if action is None:
                await update.message.reply_text("No conversations yet.")
                return
            adapter.send_action(_session(chat_id), action)
            return

        # /skip — skip optional parameter in /call form
        if cmd_name == "skip":
            state = _pending_calls.get(chat_id)
            if state:
                if state.skip_current():
                    await _ask_next_param(chat_id)
                    return
                else:
                    await update.message.reply_text("This parameter is required.")
                    return
            state = _pending_triggers.get(chat_id)
            if state:
                if state.skip_current():
                    await _ask_next_trigger_param(chat_id)
                    return
                else:
                    await update.message.reply_text("This parameter is required.")
                    return
            state = _pending_llm_adds.get(chat_id)
            if state:
                if state.skip_current():
                    await _ask_next_llm_param(chat_id)
                    return
                else:
                    await update.message.reply_text("This parameter is required.")
                    return
            state = _pending_agent_adds.get(chat_id)
            if state:
                if state.skip_current():
                    await _ask_next_agent_param(chat_id)
                    return
                else:
                    await update.message.reply_text("This parameter is required.")
                    return
            state = _pending_agent_edits.get(chat_id)
            if state:
                if state.params and not state.params[0].required:
                    # Skipping clears the field — store None and submit.
                    state.store(state.params[0].name, None)
                    await _execute_agent_field_edit(chat_id)
                    return
                else:
                    await update.message.reply_text("This field is required.")
                    return
            state = _pending_llm_edits.get(chat_id)
            if state:
                if state.params and not state.params[0].required:
                    state.store(state.params[0].name, None)
                    await _execute_llm_field_edit(chat_id)
                    return
                else:
                    await update.message.reply_text("This field is required.")
                    return
            state = _pending_schedule_creates.get(chat_id)
            if state:
                field = state.current_field(SCHEDULE_CREATE_STEPS) or ""
                if field in ("agent", "title"):
                    state.step += 1
                    await _ask_schedule_step(chat_id)
                    return
                else:
                    await update.message.reply_text("This field is required.")
                    return

        # /services, /tasks, /tools, /locations — picker menus with no arg
        if cmd_name == "services" and not arg:
            await _show_services_picker(chat_id)
            return
        if cmd_name == "tasks" and not arg:
            await _show_tasks_picker(chat_id)
            return
        if cmd_name == "tools" and not arg:
            await _show_tools_picker(chat_id)
            return
        if cmd_name == "locations" and not arg:
            await _show_locations_picker(chat_id)
            return

        # /schedule — picker menu with no arg, or subcommand dispatch with arg
        if cmd_name == "schedule" and not arg:
            await _show_schedule_picker(chat_id)
            return

        # /call — show tool menu or start interactive form
        if cmd_name == "call":
            if not arg:
                # No arg → show menu of ALL tools
                all_tools = list(tool_registry.tools.keys())
                if not all_tools:
                    await update.message.reply_text("No tools registered.")
                    return
                buttons = [[InlineKeyboardButton(t, callback_data=f"cmd:call:{t}")]
                           for t in all_tools[:30]]
                await update.message.reply_text(
                    "Choose a tool to call:",
                    reply_markup=InlineKeyboardMarkup(buttons))
                return
            elif not arg.lstrip().startswith("{"):
                # Tool name given but no JSON → interactive form
                tool_name = arg.split()[0]
                await _start_call_form(chat_id, tool_name)
                return

        if cmd_name == "trigger":
            trigger_parts = arg.split(maxsplit=1) if arg else []
            has_inline_json = len(trigger_parts) > 1 and trigger_parts[1].lstrip().startswith("{")
            if trigger_parts and not has_inline_json:
                task_name = trigger_parts[0]
                started = await _start_trigger_form(chat_id, task_name)
                if started:
                    return

        # /agent — show profile-list keyboard
        if cmd_name == "agent" and not arg:
            await _show_agent_profiles_list(chat_id)
            return

        # /llm — show LLM-list keyboard
        if cmd_name == "llm" and not arg:
            await _show_llm_list(chat_id)
            return

        # /configure — show settings menu or accept key+value
        if cmd_name == "configure":
            if not arg:
                await _show_configure_menu(chat_id)
                return

        # Dynamic menu: if command has arg_completions and no arg given
        entry = registry._commands.get(cmd_name)
        if entry and not arg and entry.arg_completions:
            completions = entry.arg_completions()
            if completions:
                buttons = [[InlineKeyboardButton(c, callback_data=f"cmd:{cmd_name}:{c}")]
                           for c in completions[:20]]
                await update.message.reply_text(
                    f"Choose for /{cmd_name}:",
                    reply_markup=InlineKeyboardMarkup(buttons))
                return

        # Standard dispatch
        result = await _dispatch_frontend_event(FrontendEvent(
            type="slash_command",
            session=_session(chat_id),
            command_name=cmd_name,
            command_arg=arg,
        ))
        if result.text:
            await transport.send_long_message(chat_id, result.text)

    async def handle_message(update: Update, _ctx):
        """Handle plain text — route to agent via route_input()."""
        if not _check_user(update):
            return
        text = (update.message.text or "").strip()
        if not text:
            return
        chat_id = update.message.chat_id
        preview = text[:60] + ("..." if len(text) > 60 else "")
        logger.info(f'<- "{preview}"')

        # Check for pending /call form input
        if chat_id in _pending_calls:
            state = _pending_calls[chat_id]
            param = state.current_param
            if param is not None:
                try:
                    state.store(param.name, coerce_param_value(text, param.type))
                    await _ask_next_param(chat_id)
                except (ValueError, json.JSONDecodeError) as e:
                    await update.message.reply_text(
                        f"Invalid value for {param.name} ({param.type}): {e}\nTry again.")
                return

        # Check for pending /trigger form input
        if chat_id in _pending_triggers:
            state = _pending_triggers[chat_id]
            param = state.current_param
            if param is not None:
                try:
                    state.store(param.name, coerce_param_value(text, param.type))
                    await _ask_next_trigger_param(chat_id)
                except (ValueError, json.JSONDecodeError) as e:
                    await update.message.reply_text(
                        f"Invalid value for {param.name} ({param.type}): {e}\nTry again.")
                return

        # Check for pending /llm add form input
        if chat_id in _pending_llm_adds:
            state = _pending_llm_adds[chat_id]
            if state.awaiting_name:
                model_name = text.strip()
                if not model_name or " " in model_name:
                    await update.message.reply_text("Model name must be a single word with no spaces. Try again.")
                    return
                if model_name == "default":
                    await update.message.reply_text(
                        "'default' is reserved — it's the sentinel agent profiles use to follow "
                        "whatever LLM is current. Pick a different name (e.g. the model identifier).")
                    return
                existing = config.get("llm_profiles", {}) or {}
                if model_name in existing:
                    await update.message.reply_text(f"LLM '{model_name}' already exists. Choose a different name.")
                    return
                _pending_llm_adds.pop(chat_id, None)
                await _start_llm_add_form(chat_id, model_name)
                return
            param = state.current_param
            if param is not None:
                try:
                    state.store(param.name, coerce_param_value(text, param.type))
                    await _ask_next_llm_param(chat_id)
                except (ValueError, json.JSONDecodeError) as e:
                    await update.message.reply_text(
                        f"Invalid value for {param.name} ({param.type}): {e}\nTry again.")
            return

        # Check for pending /agent add form input
        if chat_id in _pending_agent_adds:
            state = _pending_agent_adds[chat_id]
            if state.awaiting_name:
                profile_name = text.strip()
                if not profile_name or " " in profile_name:
                    await update.message.reply_text("Profile name must be a single word with no spaces. Try again.")
                    return
                existing = config.get("agent_profiles", {}) or {}
                if profile_name in existing:
                    await update.message.reply_text(f"Profile '{profile_name}' already exists. Choose a different name.")
                    return
                _pending_agent_adds.pop(chat_id, None)
                await _start_agent_add_form(chat_id, profile_name)
                return
            param = state.current_param
            if param is not None:
                try:
                    state.store(param.name, coerce_param_value(text, param.type))
                    await _ask_next_agent_param(chat_id)
                except (ValueError, json.JSONDecodeError) as e:
                    await update.message.reply_text(
                        f"Invalid value for {param.name} ({param.type}): {e}\nTry again.")
            return

        # Check for pending /agent edit single-field input
        if chat_id in _pending_agent_edits:
            state = _pending_agent_edits[chat_id]
            param = state.current_param
            if param is not None:
                try:
                    state.store(param.name, coerce_param_value(text, param.type))
                    await _execute_agent_field_edit(chat_id)
                except (ValueError, json.JSONDecodeError) as e:
                    await update.message.reply_text(
                        f"Invalid value for {param.name} ({param.type}): {e}\nTry again.")
            return

        # Check for pending /llm edit single-field input
        if chat_id in _pending_llm_edits:
            state = _pending_llm_edits[chat_id]
            param = state.current_param
            if param is not None:
                try:
                    state.store(param.name, coerce_param_value(text, param.type))
                    await _execute_llm_field_edit(chat_id)
                except (ValueError, json.JSONDecodeError) as e:
                    await update.message.reply_text(
                        f"Invalid value for {param.name} ({param.type}): {e}\nTry again.")
            return

        # Check for pending /schedule create form input
        if chat_id in _pending_schedule_creates:
            state = _pending_schedule_creates[chat_id]
            if state.step >= len(SCHEDULE_CREATE_STEPS):
                await _finalize_schedule_create(chat_id)
                return
            field = state.current_field(SCHEDULE_CREATE_STEPS)
            value = text.strip()

            if field == "job_name":
                if not value or " " in value:
                    await update.message.reply_text(
                        "Job name must be a single word with no spaces. Try again.")
                    return
                timekeeper = services.get("timekeeper")
                if timekeeper and timekeeper.get_job(value) is not None:
                    await update.message.reply_text(
                        f"Job '{value}' already exists. Choose a different name.")
                    return
            if field == "agent" and value:
                profiles = config.get("agent_profiles", {}) or {}
                if value not in profiles:
                    await update.message.reply_text(
                        f"Unknown agent profile: '{value}'. Choose one of the buttons or type an existing profile name.")
                    return
            if field == "notifications":
                value = value.lower()
                if value not in SUBAGENT_NOTIFICATION_MODES:
                    await update.message.reply_text(
                        f"notifications must be one of: {', '.join(SUBAGENT_NOTIFICATION_MODES)}.")
                    return
            state.collected[field] = value

            state.step += 1
            await _ask_schedule_step(chat_id)
            return

        # Check for pending /configure value input
        if chat_id in _pending_configures:
            key, widget_type = _pending_configures.pop(chat_id)
            # Convert newline-separated input to JSON array for list settings
            if widget_type == "json_list":
                try:
                    json.loads(text)  # already valid JSON? use as-is
                except json.JSONDecodeError:
                    items = [line.strip() for line in text.splitlines() if line.strip()]
                    text = json.dumps(items)
            result = await _dispatch_frontend_event(FrontendEvent(
                type="slash_command",
                session=_session(chat_id),
                command_name="configure",
                command_arg=f"{key} {text}",
            ))
            if result.text:
                await transport.send_long_message(chat_id, result.text)
            return

        chat_session = _session(chat_id)
        if runtime.is_busy(chat_session):
            await update.message.reply_text(
                "Still working on the previous message — send /refresh to abort.")
            return

        gen = runtime.begin_turn(chat_session)
        if gen is None:
            await update.message.reply_text(
                "Still working on the previous message — send /refresh to abort.")
            return
        stop_typing = asyncio.Event()

        async def _typing_loop():
            while not stop_typing.is_set():
                try:
                    await update.message.chat.send_action(ChatAction.TYPING)
                except Exception:
                    pass
                try:
                    await asyncio.wait_for(stop_typing.wait(), timeout=4.0)
                except asyncio.TimeoutError:
                    pass

        typing_task = asyncio.create_task(_typing_loop())

        try:
            # Outer 15-min safety net: if the whole pipeline (including LLM
            # calls themselves, not just tools) wedges, release the handler
            # so the bot stays responsive. Tool-level timeouts (Layer 1) are
            # usually tighter, but this catches everything else.
            result = await asyncio.wait_for(
                _dispatch_frontend_event(FrontendEvent(
                    type="chat_message",
                    session=chat_session,
                    text=text,
                ), prompt_suffix=_TELEGRAM_SUFFIX),
                timeout=900)

            if result.text:
                converted = _md_to_tg_html(result.text)
                await transport.send_long_message(chat_id, converted, use_html=True)
                logger.info(f"-> {len(result.text)} chars")
        except asyncio.TimeoutError:
            logger.error("Message handler exceeded 15 min — refreshing agent")
            await update.message.reply_text(
                "Previous tool call abandoned after 15 min. Agent refreshed.")
            runtime.refresh_agent(chat_session)
            runtime.force_unbusy(chat_session)
        except Exception as e:
            logger.error(f"Message handler error: {e}")
            await update.message.reply_text(f"Error: {e}")
        finally:
            stop_typing.set()
            await typing_task
            runtime.end_turn(chat_session, gen)

    async def handle_attachment(update: Update, _ctx):
        """Save incoming photos/documents to the attachment cache, then hand off to the agent.

        The cache folder is a sync_directory, so Stage_2 indexes files asynchronously.
        Text/image modalities still get inlined for fast first-turn context; the
        agent is always told the cache path so it can re-access the file later.
        """
        if not _check_user(update):
            return
        msg = update.message
        chat_id = msg.chat_id
        caption = (msg.caption or "").strip()

        # Determine which file to download
        tg_file = None
        file_name = "attachment"
        if msg.photo:
            # Photo array: last element is the largest resolution
            tg_file = await msg.photo[-1].get_file()
            file_name = "photo.jpg"
        elif msg.document:
            if msg.document.file_size and msg.document.file_size > _MAX_FILE_SIZE:
                await msg.reply_text("File too large (50 MB limit).")
                return
            tg_file = await msg.document.get_file()
            file_name = msg.document.file_name or "document"
        elif msg.voice:
            if msg.voice.file_size and msg.voice.file_size > _MAX_FILE_SIZE:
                await msg.reply_text("File too large (50 MB limit).")
                return
            tg_file = await msg.voice.get_file()
            file_name = "voice.ogg"
        elif msg.audio:
            if msg.audio.file_size and msg.audio.file_size > _MAX_FILE_SIZE:
                await msg.reply_text("File too large (50 MB limit).")
                return
            tg_file = await msg.audio.get_file()
            ext = Path(msg.audio.file_name or "").suffix or ".mp3"
            file_name = msg.audio.file_name or f"audio{ext}"

        if tg_file is None:
            return

        # Download bytes, then persist to the attachment cache. The cache folder is
        # a sync_directory, so Stage_2 will index the file in the background.
        suffix = Path(file_name).suffix or ""
        data = bytes(await tg_file.download_as_bytearray())
        cache_path = save_attachment(
            file_name, data,
            size_cap_gb=float(config.get("attachment_cache_size_gb", 2.0)))

        modality = get_modality(suffix) if suffix else "unknown"
        is_image = modality == "image" or (msg.photo and modality == "unknown")

        # ── Build user_text and image_paths based on modality ──
        user_text = caption or ""
        send_image_paths = None

        if is_image:
            llm = services.get("llm")
            has_vision = llm and llm.vision is not False
            if has_vision:
                send_image_paths = [str(cache_path)]
                user_text += f"\n\n[The user attached an image: {file_name} (cached at {cache_path})]"
            else:
                user_text += (
                    f"\n\n[The user attached an image: {file_name} (cached at {cache_path}). "
                    "The current model does not support vision, "
                    "so the image contents are not visible to you.]")
            if not caption:
                user_text = user_text.lstrip()

        elif modality in ("text", "tabular"):
            # Parse and inline for fast first-turn context. Indexing happens async.
            content = ""
            truncated = False
            try:
                parser_svc = services.get("parser")
                pr = parser_svc.parse(str(cache_path), config={"max_chars": _MAX_ATTACHMENT_TEXT})
                if pr.output:
                    if isinstance(pr.output, str):
                        raw = pr.output
                    elif isinstance(pr.output, dict):
                        df = pr.output.get("default")
                        raw = df.to_string(max_rows=50) if df is not None else str(pr.output)
                    else:
                        raw = str(pr.output)
                    if len(raw) > _MAX_ATTACHMENT_TEXT:
                        content = raw[:_MAX_ATTACHMENT_TEXT]
                        truncated = True
                    else:
                        content = raw
            except Exception as e:
                logger.warning(f"Failed to parse attachment {file_name}: {e}")

            if content:
                user_text += f"\n\n[The user attached a file: {file_name} (cached at {cache_path})]\n{content}"
                if truncated:
                    user_text += f"\n(Content truncated — only the first ~{_MAX_ATTACHMENT_TEXT} characters are shown. Use read_file on the cached path for more.)"
            else:
                user_text += f"\n\n[The user attached a file: {file_name} (cached at {cache_path}). Inline extraction failed — use read_file on the cached path.]"
            if not caption:
                user_text = user_text.lstrip()

        elif modality == "audio":
            from frontend.attachment_inliner import transcribe_audio_inline
            user_text += await transcribe_audio_inline(services, cache_path, file_name)
            if not caption:
                user_text = user_text.lstrip()

        else:
            # Video, unknown: file is saved and the Stage_2 pipeline will
            # index whatever it can. The agent can use read_file / search tools
            # to access it, or write a new parser plugin for novel extensions.
            user_text += (
                f"\n\n[The user attached a file: {file_name} (type: {modality}, "
                f"cached at {cache_path}). Indexing in background — use read_file "
                "or search tools to access it.]")
            if not caption:
                user_text = user_text.lstrip()

        # ── Send to agent ──────────────────────────────────────
        chat_session = _session(chat_id)
        if runtime.is_busy(chat_session):
            await msg.reply_text(
                "Still working on the previous message — send /refresh to abort.")
            return

        gen = runtime.begin_turn(chat_session)
        if gen is None:
            await msg.reply_text(
                "Still working on the previous message — send /refresh to abort.")
            return
        stop_typing = asyncio.Event()

        async def _typing_loop():
            while not stop_typing.is_set():
                try:
                    await msg.chat.send_action(ChatAction.TYPING)
                except Exception:
                    pass
                try:
                    await asyncio.wait_for(stop_typing.wait(), timeout=4.0)
                except asyncio.TimeoutError:
                    pass

        typing_task = asyncio.create_task(_typing_loop())
        try:
            result = await asyncio.wait_for(
                _dispatch_frontend_event(FrontendEvent(
                    type="attachment_message",
                    session=chat_session,
                    text=user_text,
                    attachments=[str(cache_path)],
                    payload={"image_paths": send_image_paths or []},
                ), prompt_suffix=_TELEGRAM_SUFFIX),
                timeout=900)
            if result.text:
                converted = _md_to_tg_html(result.text)
                await transport.send_long_message(chat_id, converted, use_html=True)
        except asyncio.TimeoutError:
            logger.error("Attachment handler exceeded 15 min — refreshing agent")
            await msg.reply_text(
                "Previous tool call abandoned after 15 min. Agent refreshed.")
            runtime.refresh_agent(chat_session)
            runtime.force_unbusy(chat_session)
        except Exception as e:
            logger.error(f"Attachment handler error: {e}")
            await msg.reply_text(f"Error: {e}")
        finally:
            stop_typing.set()
            await typing_task
            runtime.end_turn(chat_session, gen)

    async def handle_callback_query(update: Update, _ctx):
        """Handle inline keyboard responses (approval, commands, /call form)."""
        data = update.callback_query.data or ""

        try:
            # ── Command menu callbacks (cmd:<name>:<arg>) ──
            if data.startswith("cmd:"):
                parts = data.split(":", 2)
                if len(parts) == 3:
                    _, cmd_name, arg = parts
                    await update.callback_query.answer()
                    await update.callback_query.edit_message_reply_markup(reply_markup=None)
                    chat_id = update.callback_query.message.chat_id

                    # /call menu → start interactive form instead of raw dispatch
                    if cmd_name == "call":
                        await _start_call_form(chat_id, arg)
                        return
                    if cmd_name == "trigger":
                        await _start_trigger_form(chat_id, arg)
                        return

                    result = await _dispatch_frontend_event(FrontendEvent(
                        type="callback_response",
                        session=_session(chat_id),
                        payload={"kind": "command", "command_name": cmd_name, "command_arg": arg},
                    ))
                    if result.text:
                        await transport.send_long_message(chat_id, result.text)
                return

            # ── Agent profile callbacks (agnt:<action>[:<name>[:<field>]]) ──
            if data.startswith("agnt:"):
                parts = data.split(":", 3)
                action = parts[1] if len(parts) > 1 else ""
                name = parts[2] if len(parts) > 2 else ""
                field = parts[3] if len(parts) > 3 else ""
                await update.callback_query.answer()
                await update.callback_query.edit_message_reply_markup(reply_markup=None)
                chat_id = update.callback_query.message.chat_id

                if action == "back":
                    await _show_agent_profiles_list(chat_id)
                elif action == "add":
                    _pending_agent_adds[chat_id] = PendingParamForm(awaiting_name=True)
                    await _app.bot.send_message(
                        chat_id,
                        "Enter a name for the new agent profile (e.g. <code>researcher</code>):\n\n"
                        "Send /cancel to abort.",
                        parse_mode="HTML")
                elif action == "pick":
                    await _show_agent_profile_actions(chat_id, name)
                elif action == "edit":
                    await _show_agent_edit_fields(chat_id, name)
                elif action == "editfield":
                    await _start_agent_field_edit(chat_id, name, field)
                elif action in ("switch", "remove", "show"):
                    result = await _dispatch_frontend_event(FrontendEvent(
                        type="callback_response",
                        session=_session(chat_id),
                        payload={"kind": "command", "command_name": "agent", "command_arg": f"{action} {name}"},
                    ))
                    if result.text:
                        await transport.send_long_message(chat_id, result.text)
                    if action != "remove":
                        # Refresh the per-profile menu so the user lands somewhere useful.
                        await _show_agent_profile_actions(chat_id, name)
                    else:
                        await _show_agent_profiles_list(chat_id)
                return

            # ── LLM callbacks (llm:<action>[:<name>[:<field>]]) ──
            if data.startswith("llm:"):
                parts = data.split(":", 3)
                action = parts[1] if len(parts) > 1 else ""
                name = parts[2] if len(parts) > 2 else ""
                field = parts[3] if len(parts) > 3 else ""
                await update.callback_query.answer()
                await update.callback_query.edit_message_reply_markup(reply_markup=None)
                chat_id = update.callback_query.message.chat_id

                if action == "back":
                    await _show_llm_list(chat_id)
                elif action == "add":
                    _pending_llm_adds[chat_id] = PendingParamForm(awaiting_name=True)
                    await _app.bot.send_message(
                        chat_id,
                        "Enter the model name for the new LLM (e.g. <code>gpt-4o</code>, "
                        "<code>claude-opus-4-7</code>):\n\nSend /cancel to abort.",
                        parse_mode="HTML")
                elif action == "pick":
                    await _show_llm_profile_actions(chat_id, name)
                elif action == "edit":
                    await _show_llm_edit_fields(chat_id, name)
                elif action == "editfield":
                    await _start_llm_field_edit(chat_id, name, field)
                elif action in ("default", "remove", "show"):
                    result = await _dispatch_frontend_event(FrontendEvent(
                        type="callback_response",
                        session=_session(chat_id),
                        payload={"kind": "command", "command_name": "llm", "command_arg": f"{action} {name}"},
                    ))
                    if result.text:
                        await transport.send_long_message(chat_id, result.text)
                    if action != "remove":
                        await _show_llm_profile_actions(chat_id, name)
                    else:
                        await _show_llm_list(chat_id)
                return

            # ── LLM add form enum callbacks (llmadd:<chat_id>:<param>:<value>) ──
            if data.startswith("llmadd:"):
                parts = data.split(":", 3)
                if len(parts) == 4:
                    _, cid_str, param_name, value = parts
                    cid = int(cid_str)
                    await update.callback_query.answer()
                    await update.callback_query.edit_message_reply_markup(reply_markup=None)
                    state = _pending_llm_adds.get(cid)
                    if state and not state.awaiting_name:
                        state.store(param_name, value)
                        await _ask_next_llm_param(cid)
                return

            # ── LLM edit single-field enum callbacks (llmedit:<chat_id>:<param>:<value>) ──
            if data.startswith("llmedit:"):
                parts = data.split(":", 3)
                if len(parts) == 4:
                    _, cid_str, param_name, value = parts
                    cid = int(cid_str)
                    await update.callback_query.answer()
                    await update.callback_query.edit_message_reply_markup(reply_markup=None)
                    state = _pending_llm_edits.get(cid)
                    if state and state.params and state.params[0].name == param_name:
                        state.store(param_name, value)
                        await _execute_llm_field_edit(cid)
                return

            # ── Agent edit single-field enum callbacks (agtedit:<chat_id>:<param>:<value>) ──
            if data.startswith("agtedit:"):
                parts = data.split(":", 3)
                if len(parts) == 4:
                    _, cid_str, param_name, value = parts
                    cid = int(cid_str)
                    await update.callback_query.answer()
                    await update.callback_query.edit_message_reply_markup(reply_markup=None)
                    state = _pending_agent_edits.get(cid)
                    if state and state.params and state.params[0].name == param_name:
                        state.store(param_name, value)
                        await _execute_agent_field_edit(cid)
                return

            # ── Agent add form enum callbacks (agtadd:<chat_id>:<param>:<value>) ──
            if data.startswith("agtadd:"):
                parts = data.split(":", 3)
                if len(parts) == 4:
                    _, cid_str, param_name, value = parts
                    cid = int(cid_str)
                    await update.callback_query.answer()
                    await update.callback_query.edit_message_reply_markup(reply_markup=None)
                    state = _pending_agent_adds.get(cid)
                    if state and not state.awaiting_name:
                        state.store(param_name, value)
                        await _ask_next_agent_param(cid)
                return

            # ── History conversation selection (hist:<id>) ──
            if data.startswith("hist:"):
                conv_id_str = data[5:]
                await update.callback_query.answer()
                await update.callback_query.edit_message_reply_markup(reply_markup=None)
                chat_id = update.callback_query.message.chat_id
                result = await _dispatch_frontend_event(FrontendEvent(
                    type="callback_response",
                    session=_session(chat_id),
                    payload={"kind": "history", "conversation_id": conv_id_str},
                ))
                if result.text:
                    await transport.send_long_message(chat_id, result.text)
                return

            # ── Configure setting selection (cfg:<key>) ──
            if data.startswith("cfg:"):
                key = data[4:]
                await update.callback_query.answer()
                await update.callback_query.edit_message_reply_markup(reply_markup=None)
                chat_id = update.callback_query.message.chat_id
                await _ask_configure_value(chat_id, key)
                return

            # ── Configure bool value (cfgval:<key>:<value>) ──
            if data.startswith("cfgval:"):
                parts = data.split(":", 2)
                if len(parts) == 3:
                    _, key, raw_val = parts
                    await update.callback_query.answer()
                    await update.callback_query.edit_message_reply_markup(reply_markup=None)
                    chat_id = update.callback_query.message.chat_id
                    result = await _dispatch_frontend_event(FrontendEvent(
                        type="callback_response",
                        session=_session(chat_id),
                        payload={"kind": "command", "command_name": "configure", "command_arg": f"{key} {raw_val}"},
                    ))
                    if result.text:
                        await transport.send_long_message(chat_id, result.text)
                return

            # ── /call form callbacks (call:<chat_id>:<param>:<value>) ──
            if data.startswith("call:"):
                parts = data.split(":", 3)
                if len(parts) == 4:
                    _, cid_str, param_name, value = parts
                    chat_id = int(cid_str)
                    await update.callback_query.answer()
                    await update.callback_query.edit_message_reply_markup(reply_markup=None)
                    state = _pending_calls.get(chat_id)
                    param = state.current_param if state else None
                    if param is not None:
                        state.store(param_name, coerce_param_value(value, param.type))
                        await _ask_next_param(chat_id)
                return

            # ── /trigger form callbacks (trigger:<chat_id>:<param>:<value>) ──
            if data.startswith("trigger:"):
                parts = data.split(":", 3)
                if len(parts) == 4:
                    _, cid_str, param_name, value = parts
                    chat_id = int(cid_str)
                    await update.callback_query.answer()
                    await update.callback_query.edit_message_reply_markup(reply_markup=None)
                    state = _pending_triggers.get(chat_id)
                    param = state.current_param if state else None
                    if param is not None:
                        state.store(param_name, coerce_param_value(value, param.type))
                        await _ask_next_trigger_param(chat_id)
                return

            # ── /services picker (svc:pick|load|unload|back[:name]) ──
            if data.startswith("svc:"):
                parts = data.split(":", 2)
                action = parts[1] if len(parts) > 1 else ""
                name = parts[2] if len(parts) > 2 else ""
                await update.callback_query.answer()
                await update.callback_query.edit_message_reply_markup(reply_markup=None)
                chat_id = update.callback_query.message.chat_id

                if action == "pick":
                    await _show_service_actions(chat_id, name)
                elif action in ("load", "unload"):
                    result = await _dispatch_frontend_event(FrontendEvent(
                        type="callback_response",
                        session=_session(chat_id),
                        payload={"kind": "command", "command_name": action, "command_arg": name},
                    ))
                    if result.text:
                        await transport.send_long_message(chat_id, result.text)
                    await _show_services_picker(chat_id)
                elif action == "back":
                    await _show_services_picker(chat_id)
                return

            # ── /tasks picker (tsk:pick|pause|unpause|reset|retry|trigger|back[:name]) ──
            if data.startswith("tsk:"):
                parts = data.split(":", 2)
                action = parts[1] if len(parts) > 1 else ""
                name = parts[2] if len(parts) > 2 else ""
                await update.callback_query.answer()
                await update.callback_query.edit_message_reply_markup(reply_markup=None)
                chat_id = update.callback_query.message.chat_id

                if action == "pick":
                    await _show_task_actions(chat_id, name)
                elif action == "trigger":
                    await _start_trigger_form(chat_id, name)
                elif action in ("pause", "unpause", "reset", "retry"):
                    result = await _dispatch_frontend_event(FrontendEvent(
                        type="callback_response",
                        session=_session(chat_id),
                        payload={"kind": "command", "command_name": action, "command_arg": name},
                    ))
                    if result.text:
                        await transport.send_long_message(chat_id, result.text)
                    await _show_tasks_picker(chat_id)
                elif action == "back":
                    await _show_tasks_picker(chat_id)
                return

            # ── /tools picker (tool:pick|call|back[:name]) ──
            if data.startswith("tool:"):
                parts = data.split(":", 2)
                action = parts[1] if len(parts) > 1 else ""
                name = parts[2] if len(parts) > 2 else ""
                await update.callback_query.answer()
                await update.callback_query.edit_message_reply_markup(reply_markup=None)
                chat_id = update.callback_query.message.chat_id

                if action == "pick":
                    await _show_tool_actions(chat_id, name)
                elif action == "call":
                    await _start_call_form(chat_id, name)
                elif action == "back":
                    await _show_tools_picker(chat_id)
                return

            # ── /locations filter buttons (loc:<filter>) ──
            if data.startswith("loc:"):
                filt = data[4:]
                await update.callback_query.answer()
                await update.callback_query.edit_message_reply_markup(reply_markup=None)
                chat_id = update.callback_query.message.chat_id
                arg = "" if filt == "all" else filt
                result = await _dispatch_frontend_event(FrontendEvent(
                    type="callback_response",
                    session=_session(chat_id),
                    payload={"kind": "command", "command_name": "locations", "command_arg": arg},
                ))
                if result.text:
                    await transport.send_long_message(chat_id, result.text)
                return

            # ── /schedule picker + create form (sch:...) ──
            if data.startswith("sch:"):
                parts = data.split(":", 2)
                action = parts[1] if len(parts) > 1 else ""
                tail = parts[2] if len(parts) > 2 else ""
                await update.callback_query.answer()
                await update.callback_query.edit_message_reply_markup(reply_markup=None)
                chat_id = update.callback_query.message.chat_id

                if action == "pick":
                    await _show_schedule_job_actions(chat_id, tail)
                elif action == "new":
                    await _start_schedule_create(chat_id)
                elif action == "back":
                    await _show_schedule_picker(chat_id)
                elif action in ("run", "enable", "disable", "show"):
                    result = await _dispatch_frontend_event(FrontendEvent(
                        type="callback_response",
                        session=_session(chat_id),
                        payload={"kind": "command", "command_name": "schedule", "command_arg": f"{action} {tail}"},
                    ))
                    if result.text:
                        await transport.send_long_message(chat_id, result.text)
                    if action != "show":
                        await _show_schedule_picker(chat_id)
                elif action == "delete":
                    kb = InlineKeyboardMarkup([[
                        InlineKeyboardButton("Confirm delete", callback_data=f"sch:confirmdel:{tail}"),
                        InlineKeyboardButton("Cancel", callback_data="sch:back"),
                    ]])
                    await _app.bot.send_message(
                        chat_id,
                        f"Delete scheduled job '{tail}'? This cannot be undone.",
                        reply_markup=kb)
                elif action == "confirmdel":
                    result = await _dispatch_frontend_event(FrontendEvent(
                        type="callback_response",
                        session=_session(chat_id),
                        payload={"kind": "command", "command_name": "schedule", "command_arg": f"delete {tail}"},
                    ))
                    if result.text:
                        await transport.send_long_message(chat_id, result.text)
                    await _show_schedule_picker(chat_id)
                elif action == "type":
                    state = _pending_schedule_creates.get(chat_id)
                    if state:
                        state.collected["schedule_type"] = tail
                        state.step += 1
                        await _ask_schedule_step(chat_id)
                elif action == "ch":
                    state = _pending_schedule_creates.get(chat_id)
                    if state:
                        if tail == "__other__":
                            await _app.bot.send_message(chat_id, "Enter the channel name as text.")
                            return
                        state.collected["channel"] = tail
                        state.step += 1
                        await _ask_schedule_step(chat_id)
                elif action == "agent":
                    state = _pending_schedule_creates.get(chat_id)
                    if state:
                        state.collected["agent"] = "" if tail == "__active__" else tail
                        state.step += 1
                        await _ask_schedule_step(chat_id)
                elif action == "notif":
                    state = _pending_schedule_creates.get(chat_id)
                    if state and tail in SUBAGENT_NOTIFICATION_MODES:
                        state.collected["notifications"] = tail
                        state.step += 1
                        await _ask_schedule_step(chat_id)
                return

            # ── Approval callbacks (approve_xxx:allow/deny) ──
            if ":" not in data:
                await update.callback_query.answer()
                return

            callback_id, action = data.rsplit(":", 1)
            result = await _dispatch_frontend_event(FrontendEvent(
                type="approval_response",
                session=_session(update.callback_query.message.chat_id),
                callback_id=callback_id,
                payload={"approved": action == "allow", "resolved_by": "telegram"},
            ))
            if result.text == "Expired or already handled.":
                await update.callback_query.answer("Timed out")
                try:
                    await update.callback_query.edit_message_reply_markup(reply_markup=None)
                except Exception:
                    pass
                return
            verdict = "Allowed" if action == "allow" else "Denied"
            await update.callback_query.answer(verdict)

        except Exception as e:
            logger.error(f"Callback query error: {e}")
            try:
                await update.callback_query.answer("Error")
            except Exception:
                pass

    # ── Application setup ────────────────────────────────────────────

    async def _register_bot_commands():
        """Push slash commands to Telegram for the / autocomplete menu."""
        commands = []
        for cmd in registry.visible_commands():
            # Telegram: name max 32 chars (lowercase, no spaces), desc max 256 chars
            name = cmd.name[:32]
            desc = cmd.description[:256]
            commands.append(BotCommand(name, desc))
        await _app.bot.set_my_commands(commands)

    async def _run():
        nonlocal _loop, _app

        _loop = asyncio.get_running_loop()
        _app = Application.builder().token(token).concurrent_updates(True).build()

        _app.add_handler(MessageHandler(filters.COMMAND, handle_command))
        _app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        _app.add_handler(MessageHandler(
            filters.PHOTO | filters.Document.ALL | filters.VOICE | filters.AUDIO,
            handle_attachment))
        _app.add_handler(CallbackQueryHandler(handle_callback_query))

        await _app.initialize()
        await _app.start()
        await _app.updater.start_polling()

        # Register commands for autocomplete
        try:
            await _register_bot_commands()
        except Exception as e:
            logger.warning(f"Failed to register bot commands: {e}")

        # Auto-create agent if LLM is already loaded
        _create_agent()

        # Send startup message
        user_id = int(config.get("telegram_allowed_user_id", 0))
        if user_id:
            status = (
                f"Agent: {config.get("active_agent_profile", "default")}"
                if runtime.ensure_agent(base_session) is not None
                else "LLM not loaded \u2014 use /load llm."
            )
            try:
                await _app.bot.send_message(user_id, f"Second Brain online. {status}")
            except Exception as e:
                logger.warning(f"Could not send startup message: {e}")

        logger.info("Telegram bot started.")

        # Wait for shutdown
        while not shutdown_event.is_set():
            await asyncio.sleep(1.0)

        logger.info("Telegram bot shutting down...")
        await _app.updater.stop()
        await _app.stop()
        await _app.shutdown()

    # ── Run ──────────────────────────────────────────────────────────

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_run())
    except Exception as e:
        logger.error(f"Telegram bot error: {e}")
    finally:
        loop.close()
