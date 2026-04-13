"""
Telegram bot frontend for Second Brain.

Mirrors the Flet GUI experience: agent auto-ready, slash commands with
autocomplete. The render_files tool sends files as Telegram media groups;
other tools do not auto-render gui_display_paths.
Runs on a daemon thread with its own asyncio event loop.
"""

import asyncio
import html
import json
import logging
import re
import tempfile
import threading
import uuid
from pathlib import Path

from Stage_1.registry import get_modality, parse
from Stage_3.agent import Agent
from Stage_3.system_prompt import build_system_prompt
from frontend.shared.commands import CommandEntry, CommandRegistry, register_core_commands
from frontend.shared.dispatch import route_input
from frontend.shared.formatters import (
    format_services, format_tasks, format_stats, format_tools,
    format_tool_result,
)

logger = logging.getLogger("Telegram")

_TG_MAX_LEN = 4096
_MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB Telegram bot API limit
_MAX_ATTACHMENT_TEXT = 4000         # Max chars of parsed text to append from attachments

_TELEGRAM_SUFFIX = (
    "\n\n## Telegram frontend\n"
    "You are connected via the Telegram mobile app. Keep responses concise.\n"
    "Telegram supports: **bold**, *italic*, `inline code`, and ```code blocks```.\n"
    "Do NOT use markdown tables, headers (#), horizontal rules (---), or bullet "
    "lists with -. Use plain numbered lists or line breaks for structure.\n"
    "The user can send you images and documents. Images are passed to you directly. "
    "Text and tabular files are parsed and their content is appended to the message."
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
                     tool_registry, services, config, root_dir: Path):
    """Launch the Telegram bot. Blocks until shutdown_event is set."""

    token = config.get("telegram_bot_token", "").strip()
    if not token:
        logger.warning("telegram_bot_token not configured — skipping Telegram frontend.")
        return

    # Late imports so the dependency is only required when the frontend is enabled
    from telegram import (
        BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Update,
        InputMediaPhoto, InputMediaVideo, InputMediaAudio, InputMediaDocument,
    )
    from telegram.constants import ChatAction
    from telegram.ext import (
        Application, CallbackQueryHandler, MessageHandler, filters,
    )
    from frontend.telegram.renderers import (
        prepare_media_actions, prepare_photo_bytes, SendAction, VIDEO_EXTENSIONS,
    )

    # ── State ────────────────────────────────────────────────────────
    agent_ref: dict = {"agent": None}
    conversation_ref: dict = {"id": None}
    _pending_approvals: dict = {}       # callback_id -> (Event, result_dict)
    _pending_calls: dict = {}           # chat_id -> {tool, params, collected, current_idx}
    _pending_configures: dict = {}      # chat_id -> setting_key (waiting for value)
    _chat_lock = asyncio.Lock()
    _loop: asyncio.AbstractEventLoop     # set once the loop is running
    _app: Application                    # set once built

    # ── Agent lifecycle ──────────────────────────────────────────────

    def _on_agent_message(msg: dict):
        """Persist conversation messages to DB (same pattern as GUI)."""
        role = msg.get("role", "")
        content = msg.get("content") or ""

        if conversation_ref["id"] is None:
            title = (content[:80].replace("\n", " ").strip()
                     if role == "user" else "New conversation")
            conversation_ref["id"] = ctrl.db.create_conversation(title)

        save_content = content
        if msg.get("tool_calls"):
            save_content = json.dumps({
                "content": content,
                "tool_calls": msg["tool_calls"],
            })

        ctrl.db.save_message(
            conversation_ref["id"], role, save_content,
            tool_call_id=msg.get("tool_call_id"),
            tool_name=msg.get("name"),
        )

    def _on_tool_result(tool_name: str, result):
        """Handle tool results — render_files sends media inline, others get a brief notification."""
        logger.info(f"tool: {tool_name} [{'ok' if result.success else 'fail'}]")
        chat_id = int(config.get("telegram_allowed_user_id", 0))
        if not chat_id:
            return

        if tool_name == "render_files" and result.gui_display_paths:
            # Render files inline via Telegram media groups
            async def _send_rendered():
                actions = prepare_media_actions(result.gui_display_paths)
                await _execute_send_actions(chat_id, actions)
            try:
                asyncio.run_coroutine_threadsafe(_send_rendered(), _loop).result(timeout=30)
            except Exception as e:
                logger.error(f"Failed to render files: {e}")
        else:
            icon = "\u2705" if result.success else "\u274c"
            text = f"{icon} {tool_name}"
            async def _send():
                await _app.bot.send_message(chat_id, text)
            try:
                asyncio.run_coroutine_threadsafe(_send(), _loop).result(timeout=5)
            except Exception:
                pass

    def _approve_command(command: str, justification: str) -> bool:
        """Sync callback called from agent thread — bridges to async Telegram."""
        callback_id = f"approve_{uuid.uuid4().hex[:8]}"
        result_event = threading.Event()
        approved = {"value": False}
        _pending_approvals[callback_id] = (result_event, approved)

        async def _send():
            keyboard = InlineKeyboardMarkup([[
                InlineKeyboardButton("\u274c Deny", callback_data=f"{callback_id}:deny"),
                InlineKeyboardButton("\u2705 Allow", callback_data=f"{callback_id}:allow"),
            ]])
            escaped_cmd = html.escape(command)
            escaped_reason = html.escape(justification)
            text = (
                f"<b>Agent requests approval:</b>\n"
                f"<code>{escaped_cmd}</code>\n\n"
                f"<pre>{escaped_reason}</pre>"
            )
            chat_id = int(config.get("telegram_allowed_user_id", 0))
            if chat_id:
                await _app.bot.send_message(chat_id, text,
                                            reply_markup=keyboard,
                                            parse_mode="HTML")

        try:
            asyncio.run_coroutine_threadsafe(_send(), _loop).result(timeout=10)
        except Exception as e:
            logger.error(f"Failed to send approval request: {e}")
            return False

        result_event.wait(timeout=120)
        _pending_approvals.pop(callback_id, None)
        return approved["value"] if result_event.is_set() else False

    def _create_agent():
        llm = services.get("llm")
        if llm and llm.loaded:
            agent_ref["agent"] = Agent(
                llm, tool_registry, config,
                system_prompt=lambda: build_system_prompt(
                    ctrl.db, ctrl.orchestrator, ctrl.tool_registry, ctrl.services
                ) + _TELEGRAM_SUFFIX,
                on_tool_result=_on_tool_result,
                on_message=_on_agent_message,
                approve_command=_approve_command,
            )
            logger.info("Agent ready (Telegram).")
            return True
        return False

    # ── Security ─────────────────────────────────────────────────────

    def _check_user(update: Update) -> bool:
        allowed = int(config.get("telegram_allowed_user_id", 0))
        if allowed and update.effective_user and update.effective_user.id != allowed:
            return False
        return True

    # ── Command registry ─────────────────────────────────────────────

    def _set_conversation_id(conv_id):
        conversation_ref["id"] = conv_id

    registry = CommandRegistry()
    register_core_commands(registry, ctrl, services, tool_registry, root_dir,
                           get_agent=lambda: agent_ref["agent"],
                           set_conversation_id=_set_conversation_id)

    # Telegram-specific overrides
    def _load_handler(arg):
        if not arg:
            return None  # Dynamic menu will handle this
        result = ctrl.load_service(arg)
        if arg == "llm" and not agent_ref["agent"]:
            _create_agent()
        return result

    def _unload_handler(arg):
        if not arg:
            return None  # Dynamic menu will handle this
        result = ctrl.unload_service(arg)
        if arg == "llm":
            agent_ref["agent"] = None
        return result

    def _new_handler(_arg):
        conversation_ref["id"] = None
        if agent_ref["agent"]:
            agent_ref["agent"].reset()
        return "New conversation started."

    for entry in [
        CommandEntry("load", "Load a service", "<service>",
                     handler=_load_handler,
                     arg_completions=lambda: list(services.keys())),
        CommandEntry("unload", "Unload a service", "<service>",
                     handler=_unload_handler,
                     arg_completions=lambda: list(services.keys())),
        CommandEntry("new", "Start a new conversation", handler=_new_handler),
        CommandEntry("start", "Welcome message",
                     handler=lambda _: "Second Brain is online. Send a message to chat, or /help for commands."),
        # Compact formatter overrides
        CommandEntry("services", "List services and status",
                     handler=lambda _: format_services(ctrl.list_services(), compact=True)),
        CommandEntry("tasks", "List tasks with status counts",
                     handler=lambda _: format_tasks(ctrl.list_tasks(), compact=True)),
        CommandEntry("stats", "System overview",
                     handler=lambda _: format_stats(ctrl.stats(), compact=True)),
        CommandEntry("tools", "List registered tools",
                     handler=lambda _: format_tools(ctrl.list_tools(), compact=True)),
    ]:
        registry.register(entry)

    # ── Helpers ──────────────────────────────────────────────────────

    async def _send_long_message(chat_id: int, text: str, use_html: bool = False):
        """Send text, splitting into multiple messages if over 4096 chars.

        If *use_html* is True, sends with parse_mode=HTML. On failure
        (e.g. malformed HTML), retries as plain text.
        """
        if not text:
            return

        parse_mode = "HTML" if use_html else None

        if len(text) <= _TG_MAX_LEN:
            try:
                await _app.bot.send_message(chat_id, text, parse_mode=parse_mode)
            except Exception:
                if parse_mode:
                    await _app.bot.send_message(chat_id, text)
            return

        chunks = []
        current = ""
        for line in text.split("\n"):
            if len(current) + len(line) + 1 > _TG_MAX_LEN:
                if current:
                    chunks.append(current)
                current = line[:_TG_MAX_LEN]
            else:
                current = f"{current}\n{line}" if current else line
        if current:
            chunks.append(current)

        for chunk in chunks:
            try:
                await _app.bot.send_message(chat_id, chunk, parse_mode=parse_mode)
            except Exception:
                if parse_mode:
                    await _app.bot.send_message(chat_id, chunk)

    async def _execute_send_actions(chat_id: int, actions: list[SendAction]):
        """Execute a list of SendActions via the Telegram Bot API."""
        for action in actions:
            try:
                if action.method == "media_group":
                    media = []
                    for f in action.files:
                        ext = f.suffix.lower()
                        if action.group_type == "photo_video":
                            if ext in VIDEO_EXTENSIONS:
                                media.append(InputMediaVideo(open(f, "rb")))
                            else:
                                media.append(InputMediaPhoto(prepare_photo_bytes(f)))
                        elif action.group_type == "audio":
                            media.append(InputMediaAudio(open(f, "rb"), title=f.stem))
                        else:
                            media.append(InputMediaDocument(open(f, "rb"), filename=f.name))
                    await _app.bot.send_media_group(chat_id, media)

                elif action.method == "photo":
                    await _app.bot.send_photo(chat_id, photo=prepare_photo_bytes(action.files[0]))

                elif action.method == "video":
                    await _app.bot.send_video(chat_id, video=open(action.files[0], "rb"))

                elif action.method == "audio":
                    await _app.bot.send_audio(chat_id, audio=open(action.files[0], "rb"),
                                              title=action.files[0].stem)

                elif action.method == "document":
                    await _app.bot.send_document(chat_id, document=open(action.files[0], "rb"),
                                                 filename=action.files[0].name)

                elif action.method == "text":
                    await _app.bot.send_message(chat_id, action.text_content,
                                                parse_mode="HTML")

            except Exception as e:
                names = ", ".join(f.name for f in action.files) if action.files else "(text)"
                logger.error(f"Failed to send {names}: {e}")
                await _app.bot.send_message(
                    chat_id, f"(Failed to send: {names})")

    # ── Interactive /call form ───────────────────────────────────────

    def _get_tool_params(tool_name: str) -> list[dict] | None:
        """Extract parameter info from a tool's JSON schema."""
        tool = tool_registry.tools.get(tool_name)
        if not tool:
            return None
        schema = tool.parameters or {}
        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        if not props:
            return []
        params = []
        for name, info in props.items():
            params.append({
                "name": name,
                "type": info.get("type", "string"),
                "description": info.get("description", ""),
                "required": name in required,
                "enum": info.get("enum"),
            })
        return params

    async def _ask_next_param(chat_id: int):
        """Send a prompt for the next parameter in a pending /call form."""
        state = _pending_calls.get(chat_id)
        if not state:
            return
        idx = state["current_idx"]
        if idx >= len(state["params"]):
            # All params collected — execute
            await _execute_pending_call(chat_id)
            return

        param = state["params"][idx]
        req = " (required)" if param["required"] else " (optional, send /skip)"
        desc = f"\n{html.escape(param['description'])}" if param["description"] else ""

        # Type-specific input hint
        ptype = param["type"]
        if ptype == "string":
            hint = "\nType your value as plain text (no quotes needed)."
        elif ptype == "integer":
            hint = "\nType a whole number, e.g. <code>42</code>"
        elif ptype == "number":
            hint = "\nType a number, e.g. <code>3.14</code>"
        elif ptype == "array":
            hint = "\nSend each item on its own line, e.g.:\n<code>first item\nsecond item</code>"
        elif ptype == "object":
            hint = "\nSend as JSON, e.g. <code>{\"key\": \"value\"}</code>"
        else:
            hint = ""

        text = f"<b>{html.escape(param['name'])}</b> ({ptype}){req}{desc}{hint}"

        # Enum → inline keyboard
        if param.get("enum"):
            buttons = [[InlineKeyboardButton(str(v), callback_data=f"call:{chat_id}:{param['name']}:{v}")]
                       for v in param["enum"]]
            await _app.bot.send_message(chat_id, text,
                                        reply_markup=InlineKeyboardMarkup(buttons),
                                        parse_mode="HTML")
        elif param["type"] == "boolean":
            buttons = [[
                InlineKeyboardButton("True", callback_data=f"call:{chat_id}:{param['name']}:true"),
                InlineKeyboardButton("False", callback_data=f"call:{chat_id}:{param['name']}:false"),
            ]]
            await _app.bot.send_message(chat_id, text,
                                        reply_markup=InlineKeyboardMarkup(buttons),
                                        parse_mode="HTML")
        else:
            await _app.bot.send_message(chat_id, text, parse_mode="HTML")

    async def _execute_pending_call(chat_id: int):
        """Execute a completed /call form."""
        state = _pending_calls.pop(chat_id, None)
        if not state:
            return
        tool_name = state["tool"]
        kwargs = state["collected"]
        await _app.bot.send_message(chat_id, f"Calling {tool_name}...")
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: ctrl.call_tool(tool_name, kwargs))
        logger.info(f"tool: {tool_name} [{'ok' if result.success else 'fail'}]")
        if result.gui_display_paths:
            actions = prepare_media_actions(result.gui_display_paths)
            await _execute_send_actions(chat_id, actions)
            if result.llm_summary:
                await _send_long_message(chat_id, result.llm_summary)
        else:
            output = format_tool_result(result)
            await _send_long_message(chat_id, output)

    def _coerce_param_value(raw: str, param_type: str):
        """Convert a raw string to the appropriate Python type."""
        if param_type == "integer":
            return int(raw)
        elif param_type == "number":
            return float(raw)
        elif param_type == "boolean":
            return raw.lower() in ("true", "yes", "1")
        elif param_type == "array":
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                # Newline-separated (primary) or comma-separated (fallback)
                if "\n" in raw:
                    return [line.strip() for line in raw.splitlines() if line.strip()]
                return [item.strip() for item in raw.split(",") if item.strip()]
        elif param_type == "object":
            return json.loads(raw)
        return raw

    async def _start_call_form(chat_id: int, tool_name: str):
        """Begin interactive /call parameter collection for a tool."""
        params = _get_tool_params(tool_name)
        if params is None:
            await _app.bot.send_message(chat_id, f"Unknown tool: {tool_name}")
            return
        if not params:
            # No parameters — execute immediately
            await _app.bot.send_message(chat_id, f"Calling {tool_name}...")
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, lambda: ctrl.call_tool(tool_name, {}))
            if result.gui_display_paths:
                actions = prepare_media_actions(result.gui_display_paths)
                await _execute_send_actions(chat_id, actions)
                if result.llm_summary:
                    await _send_long_message(chat_id, result.llm_summary)
            else:
                output = format_tool_result(result)
                await _send_long_message(chat_id, output)
            return
        _pending_calls[chat_id] = {
            "tool": tool_name,
            "params": params,
            "collected": {},
            "current_idx": 0,
        }
        await _app.bot.send_message(
            chat_id,
            f"<b>{html.escape(tool_name)}</b> — fill in parameters:\n"
            f"Send /skip for optional params, /cancel to abort.",
            parse_mode="HTML")
        await _ask_next_param(chat_id)

    async def _show_configure_menu(chat_id: int):
        """Show inline keyboard with all config settings."""
        from config_data import SETTINGS_DATA
        from plugin_discovery import get_plugin_settings

        all_settings = list(SETTINGS_DATA) + list(get_plugin_settings())
        buttons = []
        for title, key, _desc, _default, _type_info in all_settings:
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
        from config_data import SETTINGS_DATA
        from plugin_discovery import get_plugin_settings

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
            if cancelled:
                await update.message.reply_text("Cancelled.")
                return

        # /history — show inline keyboard of recent conversations
        if cmd_name == "history" and not arg:
            from datetime import datetime
            conversations = ctrl.db.list_conversations(limit=10)
            if not conversations:
                await update.message.reply_text("No conversations yet.")
                return
            buttons = []
            for conv in conversations:
                title = (conv["title"] or "New conversation").replace("\n", " ")[:40]
                ts = conv.get("updated_at")
                time_str = datetime.fromtimestamp(ts).strftime("%b %d") if ts else ""
                label = f"{title}  ({time_str})" if time_str else title
                buttons.append([InlineKeyboardButton(
                    label, callback_data=f"hist:{conv['id']}")])
            await update.message.reply_text(
                "Recent conversations:",
                reply_markup=InlineKeyboardMarkup(buttons))
            return

        # /skip — skip optional parameter in /call form
        if cmd_name == "skip":
            state = _pending_calls.get(chat_id)
            if state:
                idx = state["current_idx"]
                if idx < len(state["params"]) and not state["params"][idx]["required"]:
                    state["current_idx"] += 1
                    await _ask_next_param(chat_id)
                    return
                else:
                    await update.message.reply_text("This parameter is required.")
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

        # /model — show subcommand menu or profile picker
        if cmd_name == "model":
            if not arg:
                # Show subcommand menu
                buttons = [
                    [InlineKeyboardButton("List profiles", callback_data="mdl:list")],
                    [InlineKeyboardButton("Switch active", callback_data="mdl:pick:switch")],
                    [InlineKeyboardButton("Show profile", callback_data="mdl:pick:show")],
                    [InlineKeyboardButton("Remove profile", callback_data="mdl:pick:remove")],
                ]
                await update.message.reply_text(
                    "LLM Profile Manager:",
                    reply_markup=InlineKeyboardMarkup(buttons))
                return
            # If arg is a subcommand needing a profile, show picker
            sub = arg.strip().split(None, 1)[0].lower()
            if sub in ("switch", "remove", "show") and len(arg.strip().split()) == 1:
                profiles = config.get("llm_profiles", {})
                if not profiles:
                    await update.message.reply_text("No LLM profiles configured.")
                    return
                buttons = [[InlineKeyboardButton(
                    f"{'* ' if config.get('active_llm_profile') == n else ''}{n}",
                    callback_data=f"mdl:{sub}:{n}")]
                    for n in profiles]
                await update.message.reply_text(
                    f"Choose a profile to {sub}:",
                    reply_markup=InlineKeyboardMarkup(buttons))
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
        loop = asyncio.get_running_loop()
        output = await loop.run_in_executor(
            None, lambda: registry.dispatch(cmd_name, arg))

        if output:
            await _send_long_message(chat_id, output)

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
            idx = state["current_idx"]
            if idx < len(state["params"]):
                param = state["params"][idx]
                try:
                    value = _coerce_param_value(text, param["type"])
                    state["collected"][param["name"]] = value
                    state["current_idx"] += 1
                    await _ask_next_param(chat_id)
                except (ValueError, json.JSONDecodeError) as e:
                    await update.message.reply_text(
                        f"Invalid value for {param['name']} ({param['type']}): {e}\nTry again.")
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
            loop = asyncio.get_running_loop()
            output = await loop.run_in_executor(
                None, lambda: registry.dispatch("configure", f"{key} {text}"))
            if output:
                await _send_long_message(chat_id, output)
            return

        async with _chat_lock:
            # Typing indicator loop
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
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None, lambda: route_input(text, registry, agent_ref["agent"]))

                if result.text:
                    converted = _md_to_tg_html(result.text)
                    await _send_long_message(chat_id, converted, use_html=True)
                    logger.info(f"-> {len(result.text)} chars")
            except Exception as e:
                logger.error(f"Message handler error: {e}")
                await update.message.reply_text(f"Error: {e}")
            finally:
                stop_typing.set()
                await typing_task

    async def handle_attachment(update: Update, _ctx):
        """Handle incoming photos and documents — parse ephemerally for the agent."""
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

        if tg_file is None:
            return

        # Download to a temp file, preserving the extension
        suffix = Path(file_name).suffix or ""
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, prefix="tg_attach_")
        tmp_path = Path(tmp.name)
        tmp.close()

        try:
            await tg_file.download_to_drive(str(tmp_path))
            modality = get_modality(suffix) if suffix else "unknown"
            is_image = modality == "image" or (msg.photo and modality == "unknown")

            # ── Build user_text and image_paths based on modality ──
            user_text = caption or ""
            send_image_paths = None

            if is_image:
                llm = services.get("llm")
                has_vision = llm and llm.vision is not False
                if has_vision:
                    send_image_paths = [str(tmp_path)]
                    user_text += f"\n\n[The user attached an image: {file_name}]"
                else:
                    user_text += (
                        f"\n\n[The user attached an image: {file_name}. "
                        "The current model does not support vision, "
                        "so the image contents are not visible to you.]")
                if not caption:
                    user_text = user_text.lstrip()

            elif modality in ("text", "tabular"):
                # Parse and inline the content
                content = ""
                truncated = False
                try:
                    pr = parse(str(tmp_path), config={"max_chars": _MAX_ATTACHMENT_TEXT},
                              services=services)
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
                    user_text += f"\n\n[The user attached a file: {file_name}]\n{content}"
                    if truncated:
                        user_text += "\n(Content truncated — only the first ~4000 characters are shown.)"
                else:
                    user_text += f"\n\n[The user attached a file: {file_name}, but its contents could not be extracted.]"
                if not caption:
                    user_text = user_text.lstrip()

            else:
                # Audio, video, unknown — can't process, but still tell the LLM
                user_text += f"\n\n[The user attached a file: {file_name} (type: {modality}). This file type cannot be processed.]"
                if not caption:
                    user_text = user_text.lstrip()

            # ── Send to agent ──────────────────────────────────────
            async with _chat_lock:
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
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(
                        None, lambda: route_input(
                            user_text, registry, agent_ref["agent"],
                            image_paths=send_image_paths))
                    if result.text:
                        converted = _md_to_tg_html(result.text)
                        await _send_long_message(chat_id, converted, use_html=True)
                except Exception as e:
                    logger.error(f"Attachment handler error: {e}")
                    await msg.reply_text(f"Error: {e}")
                finally:
                    stop_typing.set()
                    await typing_task

        finally:
            # Clean up temp file
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

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

                    loop = asyncio.get_running_loop()
                    output = await loop.run_in_executor(
                        None, lambda: registry.dispatch(cmd_name, arg))
                    if output:
                        await _send_long_message(chat_id, output)
                return

            # ── Model profile callbacks (mdl:<action>[:<name>]) ──
            if data.startswith("mdl:"):
                parts = data.split(":", 2)
                action = parts[1] if len(parts) > 1 else ""
                name = parts[2] if len(parts) > 2 else ""
                await update.callback_query.answer()
                await update.callback_query.edit_message_reply_markup(reply_markup=None)
                chat_id = update.callback_query.message.chat_id

                if action == "list":
                    loop = asyncio.get_running_loop()
                    output = await loop.run_in_executor(
                        None, lambda: registry.dispatch("model", "list"))
                    if output:
                        await _send_long_message(chat_id, output)
                elif action == "pick":
                    # Show profile picker for the subcommand in `name`
                    sub = name
                    profiles = config.get("llm_profiles", {})
                    if not profiles:
                        await _app.bot.send_message(chat_id, "No LLM profiles configured.")
                    else:
                        buttons = [[InlineKeyboardButton(
                            f"{'* ' if config.get('active_llm_profile') == n else ''}{n}",
                            callback_data=f"mdl:{sub}:{n}")]
                            for n in profiles]
                        await _app.bot.send_message(
                            chat_id, f"Choose a profile to {sub}:",
                            reply_markup=InlineKeyboardMarkup(buttons))
                elif action in ("switch", "remove", "show"):
                    loop = asyncio.get_running_loop()
                    output = await loop.run_in_executor(
                        None, lambda: registry.dispatch("model", f"{action} {name}"))
                    if output:
                        await _send_long_message(chat_id, output)
                return

            # ── History conversation selection (hist:<id>) ──
            if data.startswith("hist:"):
                conv_id_str = data[5:]
                await update.callback_query.answer()
                await update.callback_query.edit_message_reply_markup(reply_markup=None)
                chat_id = update.callback_query.message.chat_id
                loop = asyncio.get_running_loop()
                output = await loop.run_in_executor(
                    None, lambda: registry.dispatch("history", conv_id_str))
                if output:
                    await _send_long_message(chat_id, output)
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
                    loop = asyncio.get_running_loop()
                    output = await loop.run_in_executor(
                        None, lambda: registry.dispatch("configure", f"{key} {raw_val}"))
                    if output:
                        await _send_long_message(chat_id, output)
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
                    if state:
                        idx = state["current_idx"]
                        if idx < len(state["params"]):
                            param = state["params"][idx]
                            state["collected"][param_name] = _coerce_param_value(value, param["type"])
                            state["current_idx"] += 1
                            await _ask_next_param(chat_id)
                return

            # ── Approval callbacks (approve_xxx:allow/deny) ──
            if ":" not in data:
                await update.callback_query.answer()
                return

            callback_id, action = data.rsplit(":", 1)
            pending = _pending_approvals.get(callback_id)
            if pending is None:
                await update.callback_query.answer("Expired")
                return

            result_event, approved = pending
            approved["value"] = (action == "allow")
            result_event.set()

            verdict = "Allowed" if approved["value"] else "Denied"
            await update.callback_query.answer(verdict)
            try:
                await update.callback_query.edit_message_reply_markup(reply_markup=None)
            except Exception:
                pass
            try:
                await update.callback_query.message.reply_text(f"Command {verdict.lower()}.")
            except Exception:
                pass

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
        for cmd in registry.all_commands():
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
            filters.PHOTO | filters.Document.ALL, handle_attachment))
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
            status = "Agent ready." if agent_ref["agent"] else "LLM not loaded \u2014 use /load llm."
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
