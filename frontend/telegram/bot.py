"""
Telegram bot frontend for Second Brain.

Mirrors the Flet GUI experience: agent auto-ready, slash commands with
autocomplete, and gui_display_paths rendered as native Telegram media.
Runs on a daemon thread with its own asyncio event loop.
"""

import asyncio
import json
import logging
import threading
import uuid
from pathlib import Path

from Stage_1.registry import get_modality
from Stage_3.agent import Agent
from Stage_3.system_prompt import build_system_prompt
from frontend.shared.commands import CommandEntry, CommandRegistry, register_core_commands
from frontend.shared.dispatch import route_input

logger = logging.getLogger("Telegram")

_TG_MAX_LEN = 4096
_MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB Telegram bot API limit


def run_telegram_bot(ctrl, shutdown_fn, shutdown_event: threading.Event,
                     tool_registry, services, config, root_dir: Path):
    """Launch the Telegram bot. Blocks until shutdown_event is set."""

    token = config.get("telegram_bot_token", "").strip()
    if not token:
        logger.warning("telegram_bot_token not configured — skipping Telegram frontend.")
        return

    # Late imports so the dependency is only required when the frontend is enabled
    from telegram import (
        BotCommand, ChatAction, InlineKeyboardButton, InlineKeyboardMarkup, Update,
    )
    from telegram.ext import (
        Application, CallbackQueryHandler, MessageHandler, filters,
    )

    # ── State ────────────────────────────────────────────────────────
    agent_ref: dict = {"agent": None}
    conversation_ref: dict = {"id": None}
    _pending_approvals: dict = {}       # callback_id -> (Event, result_dict)
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

    def _approve_command(command: str, justification: str) -> bool:
        """Sync callback called from agent thread — bridges to async Telegram."""
        callback_id = f"approve_{uuid.uuid4().hex[:8]}"
        result_event = threading.Event()
        approved = {"value": False}
        _pending_approvals[callback_id] = (result_event, approved)

        async def _send():
            keyboard = InlineKeyboardMarkup([[
                InlineKeyboardButton("Deny", callback_data=f"{callback_id}:deny"),
                InlineKeyboardButton("Allow", callback_data=f"{callback_id}:allow"),
            ]])
            text = (
                f"*Agent wants to run a command:*\n"
                f"`{command}`\n\n"
                f"Reason: {justification}"
            )
            chat_id = int(config.get("telegram_allowed_user_id", 0))
            if chat_id:
                await _app.bot.send_message(chat_id, text,
                                            reply_markup=keyboard,
                                            parse_mode="Markdown")

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
                ),
                on_message=_on_agent_message,
                approve_command=_approve_command,
            )
            logger.info("Agent ready.")
            return True
        return False

    # ── Security ─────────────────────────────────────────────────────

    def _check_user(update: Update) -> bool:
        allowed = int(config.get("telegram_allowed_user_id", 0))
        if allowed and update.effective_user and update.effective_user.id != allowed:
            return False
        return True

    # ── Command registry ─────────────────────────────────────────────

    registry = CommandRegistry()
    register_core_commands(registry, ctrl, services, tool_registry, root_dir,
                           get_agent=lambda: agent_ref["agent"])

    # Telegram-specific overrides
    def _load_handler(arg):
        if not arg:
            return "Usage: /load <service>"
        result = ctrl.load_service(arg)
        if arg == "llm" and not agent_ref["agent"]:
            _create_agent()
        return result

    def _unload_handler(arg):
        if not arg:
            return "Usage: /unload <service>"
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
    ]:
        registry.register(entry)

    # ── Helpers ──────────────────────────────────────────────────────

    async def _send_long_message(chat_id: int, text: str):
        """Send text, splitting into multiple messages if over 4096 chars."""
        if not text:
            return
        if len(text) <= _TG_MAX_LEN:
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
            await _app.bot.send_message(chat_id, chunk)

    async def _send_attachments(chat_id: int, paths: list[str]):
        """Send gui_display_paths as native Telegram media."""
        for path_str in paths:
            p = Path(path_str)
            if not p.exists():
                continue
            if p.stat().st_size > _MAX_FILE_SIZE:
                await _app.bot.send_message(
                    chat_id, f"(File too large for Telegram: {p.name})")
                continue

            modality = get_modality(p.suffix)
            try:
                if modality == "image":
                    await _app.bot.send_photo(chat_id, photo=open(p, "rb"))
                elif modality == "audio":
                    await _app.bot.send_audio(chat_id, audio=open(p, "rb"),
                                              title=p.stem)
                elif modality == "video":
                    await _app.bot.send_video(chat_id, video=open(p, "rb"))
                elif modality == "text":
                    size = p.stat().st_size
                    if size <= 3000:
                        content = p.read_text(encoding="utf-8", errors="replace")
                        header = f"{p.name}\n```\n"
                        footer = "\n```"
                        available = _TG_MAX_LEN - len(header) - len(footer)
                        if len(content) > available:
                            content = content[:available - 20] + "\n... (truncated)"
                        await _app.bot.send_message(
                            chat_id, header + content + footer)
                    else:
                        await _app.bot.send_document(
                            chat_id, document=open(p, "rb"), filename=p.name)
                else:
                    # tabular, container, unknown — send as document
                    if p.is_file():
                        await _app.bot.send_document(
                            chat_id, document=open(p, "rb"), filename=p.name)
                    else:
                        await _app.bot.send_message(chat_id, f"(folder: {p})")
            except Exception as e:
                logger.error(f"Failed to send {p.name}: {e}")
                await _app.bot.send_message(
                    chat_id, f"(Failed to send file: {p.name})")

    # ── Handlers ─────────────────────────────────────────────────────

    async def handle_command(update: Update, _ctx):
        """Handle /slash commands via the shared registry."""
        if not _check_user(update):
            return

        text = update.message.text or ""
        parts = text.split(maxsplit=1)
        cmd_name = parts[0][1:].split("@")[0].lower() if parts else ""
        arg = parts[1].strip() if len(parts) > 1 else ""

        # Run dispatch in executor (some commands touch DB / services)
        loop = asyncio.get_running_loop()
        output = await loop.run_in_executor(
            None, lambda: registry.dispatch(cmd_name, arg))

        if output:
            await _send_long_message(update.message.chat_id, output)

    async def handle_message(update: Update, _ctx):
        """Handle plain text — route to agent via route_input()."""
        if not _check_user(update):
            return
        text = (update.message.text or "").strip()
        if not text:
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
                    await _send_long_message(update.message.chat_id, result.text)
                if result.attachments:
                    await _send_attachments(update.message.chat_id, result.attachments)
            except Exception as e:
                logger.error(f"Message handler error: {e}")
                await update.message.reply_text(f"Error: {e}")
            finally:
                stop_typing.set()
                await typing_task

    async def handle_callback_query(update: Update, _ctx):
        """Handle inline keyboard responses (command approval)."""
        data = update.callback_query.data or ""
        if ":" not in data:
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
        await update.callback_query.edit_message_reply_markup(reply_markup=None)
        await update.callback_query.message.reply_text(f"Command {verdict.lower()}.")

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
        _app = Application.builder().token(token).build()

        _app.add_handler(MessageHandler(filters.COMMAND, handle_command))
        _app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        _app.add_handler(CallbackQueryHandler(handle_callback_query))

        async with _app:
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
                status = "Agent ready." if agent_ref["agent"] else "LLM not loaded — use /load llm."
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

    # ── Run ──────────────────────────────────────────────────────────

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_run())
    except Exception as e:
        logger.error(f"Telegram bot error: {e}")
    finally:
        loop.close()
