from __future__ import annotations

import asyncio
import html
import logging
import re
import threading
import uuid
from pathlib import Path

from attachments.cache import save as save_attachment
from plugins.BaseFrontend import BaseFrontend, FrontendCapabilities
from plugins.frontends.helpers.command_registry import format_command_call
from plugins.frontends.helpers.telegram_renderers import (
    VIDEO_EXTENSIONS,
    file_bytes,
    prepare_media_actions,
    prepare_photo_bytes,
)
from state_machine.action_map import ACTION_SEND_ATTACHMENT
from state_machine.conversation_phases import PHASE_APPROVING_REQUEST

logger = logging.getLogger("TelegramFrontend")

_MAX_FILE_SIZE = 50 * 1024 * 1024


def _md_to_tg_html(text: str) -> str:
    parts, last = [], 0
    for m in re.finditer(r"```(\w*)\n(.*?)```", text or "", re.DOTALL):
        parts.append(_inline(text[last:m.start()]))
        code = html.escape(m.group(2).rstrip())
        parts.append(f'<pre><code class="language-{html.escape(m.group(1))}">{code}</code></pre>' if m.group(1) else f"<pre>{code}</pre>")
        last = m.end()
    return "".join(parts + [_inline((text or "")[last:])])


def _inline(text: str) -> str:
    out, last = [], 0
    for m in re.finditer(r"`([^`]+)`", text):
        out.append(_bold_italic(text[last:m.start()]))
        out.append(f"<code>{html.escape(m.group(1))}</code>")
        last = m.end()
    return "".join(out + [_bold_italic(text[last:])])


def _bold_italic(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    return re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", escaped)


def _chunks(text: str, max_chars: int = 4096) -> list[str]:
    if len(text or "") <= max_chars:
        return [text] if text else []
    chunks, remaining = [], text
    while len(remaining) > max_chars:
        split_at = remaining.rfind("\n", 0, max_chars)
        split_at = split_at if split_at > 0 else max_chars
        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:].lstrip("\n")
    return chunks + ([remaining] if remaining else [])


class TelegramFrontend(BaseFrontend):
    name = "telegram"
    description = "Telegram chat frontend backed by the conversation state machine."
    capabilities = FrontendCapabilities(True, True, True, True, True, True, True, True, 4096, _MAX_FILE_SIZE)

    def __init__(self, shutdown_event: threading.Event | None = None, services: dict | None = None):
        super().__init__()
        self.shutdown_event = shutdown_event or threading.Event()
        self.services = services or {}
        self.loop = None
        self.app = None
        self._chat_by_session: dict[str, int] = {}
        self._callbacks: dict[str, tuple[str, str, str | None]] = {}
        self._tool_messages: dict[str, tuple[int, int, str, str]] = {}
        self._last_keyboard: dict[str, tuple[int, int]] = {}

    def session_key(self, ctx) -> str:
        user = getattr(getattr(ctx, "effective_user", None), "id", None) or getattr(ctx, "user_id", 0)
        chat = getattr(getattr(ctx, "effective_chat", None), "id", None) or getattr(ctx, "chat_id", user)
        thread = getattr(getattr(ctx, "effective_message", None), "message_thread_id", None) or 0
        key = f"telegram:{user}:{chat}:{thread}"
        if chat:
            self._chat_by_session[key] = int(chat)
        return key

    def start(self) -> None:
        token = str(self.config.get("telegram_bot_token", "")).strip()
        if not token:
            logger.info("telegram_bot_token not configured; Telegram frontend disabled.")
            return
        try:
            from telegram import BotCommand, Update
            from telegram.constants import ChatAction
            from telegram.ext import Application, CallbackQueryHandler, MessageHandler, filters
        except ImportError:
            logger.warning("Telegram frontend not available; install python-telegram-bot.")
            return

        async def handle_text(update: Update, _ctx):
            if not self._check_user(update) or not update.message:
                return
            key = self.session_key(update)
            text = re.sub(r"^/([A-Za-z0-9_]+)@[^\s]+", r"/\1", (update.message.text or "").strip())
            if text:
                await self._with_typing(update.message.chat, lambda: self.submit_text(key, text), ChatAction)

        async def handle_attachment(update: Update, _ctx):
            if not self._check_user(update) or not update.message:
                return
            await self._handle_attachment(update, ChatAction)

        async def handle_callback(update: Update, _ctx):
            query = update.callback_query
            if not query:
                return
            await query.answer()
            token = query.data or ""
            if token in self._callbacks:
                key, value, echo = self._callbacks.pop(token)
                try:
                    await query.edit_message_reply_markup(reply_markup=None)
                except Exception:
                    pass
                self._last_keyboard.pop(key, None)
                # Disabled: echoing the command/value is redundant since the banner shows the current command
                # try:
                #     await query.message.reply_text(echo or (value.split(":", 2)[-1] if value.startswith("approval:") else value))
                # except Exception:
                #     pass
                if value.startswith("approval:"):
                    request_id, answer = value.split(":", 2)[1:]
                    resolved = True if answer == "allow" else False if answer == "deny" else answer
                    ok = self.resolve_approval(key, request_id, resolved, self.name)
                    if not ok and self._current_phase(key) == PHASE_APPROVING_REQUEST:
                        await self._run(lambda: self.submit_text(key, "yes" if resolved is True else "no" if resolved is False else str(resolved)))
                else:
                    await self._run(lambda: self.submit_text(key, value))

        async def run():
            self.loop = asyncio.get_running_loop()
            self.app = Application.builder().token(token).concurrent_updates(True).build()
            self.app.add_handler(MessageHandler(filters.COMMAND | (filters.TEXT & ~filters.COMMAND), handle_text))
            self.app.add_handler(MessageHandler(filters.PHOTO | filters.Document.ALL | filters.VOICE | filters.AUDIO, handle_attachment))
            self.app.add_handler(CallbackQueryHandler(handle_callback))
            await self.app.initialize()
            await self.app.start()
            await self.app.updater.start_polling()
            try:
                await self.app.bot.set_my_commands([BotCommand(c.name[:32], c.description[:256]) for c in self.commands.visible_commands()])
            except Exception as e:
                logger.warning(f"Failed to register Telegram commands: {e}")
            user_id = int(self.config.get("telegram_allowed_user_id", 0) or 0)
            if user_id:
                key = self.session_key(type("Ctx", (), {"user_id": user_id, "chat_id": user_id})())
                self._chat_by_session[key] = user_id
                await self.app.bot.send_message(user_id, "Second Brain online.")
                try:
                    notice = self.runtime.restore_last_active(key)
                    if notice:
                        await self.app.bot.send_message(user_id, notice)
                except Exception:
                    logger.exception("Telegram restore_last_active failed")
            while not self.shutdown_event.is_set():
                await asyncio.sleep(1)
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run())
        except Exception as e:
            logger.error(f"Telegram frontend crashed: {e}")
        finally:
            loop.close()

    def stop(self) -> None:
        self.shutdown_event.set()
        self.unbind()

    def submit_text(self, session_key: str, text: str):
        if self._current_phase(session_key) != PHASE_APPROVING_REQUEST and self.has_pending_approval(session_key):
            req = self._next_approval(session_key)
            value = self._parse_approval(text) if getattr(req, "type", "boolean") == "boolean" else text
            if value is None:
                return self.render_error(session_key, {"message": "Approval needs yes or no."})
            if self.resolve_approval(session_key, req.id, value, self.name):
                self.render_messages(session_key, ["Approval recorded."])
        else:
            return super().submit_text(session_key, text)

    def render_messages(self, session_key: str, messages: list[str]) -> None:
        self._clear_last_keyboard(session_key)
        for msg in messages:
            self._send_text(session_key, _md_to_tg_html(msg), use_html=True)

    def render_attachments(self, session_key: str, paths: list[str]) -> None:
        self._clear_last_keyboard(session_key)
        self._send(self._send_media(self._chat_id(session_key), paths))

    def render_form_field(self, session_key: str, form: dict) -> None:
        self._send_text(session_key, self._prompt(form), markup=self._enum_markup(session_key, form))

    def render_approval_request(self, session_key: str, req) -> None:
        body = html.escape(f"{getattr(req, 'title', 'Approval requested')}\n\n{getattr(req, 'body', '')}".strip())
        self._send_text(session_key, body, markup=self._approval_markup(session_key, req))

    def render_buttons(self, session_key: str, buttons: list[dict]) -> None:
        self._send_text(session_key, "Choose:", markup=self._buttons_markup(session_key, buttons))

    def render_error(self, session_key: str, error: dict) -> None:
        self._clear_last_keyboard(session_key)
        self._send_text(session_key, html.escape(f"Error: {(error or {}).get('message') or error}"))

    def render_tool_status(self, session_key: str, payload: dict) -> None:
        chat_id = self._chat_id(session_key)
        if not chat_id:
            return
        key = f"{session_key}:{payload.get('call_id')}"
        name = payload.get("tool_name") or payload.get("command_name") or "call"
        text = format_command_call(name, payload.get("args")) if payload.get("kind") == "command" else name
        status = payload.get("status")
        if status == "started":
            self._send(self._send_tool_started(chat_id, key, name, text))
        elif status == "progressed":
            self._send(self._progress_tool_message(chat_id, key, name, text))
        else:
            self._send(self._finish_tool_message(key, chat_id, name, text, bool(payload.get("ok")), payload.get("error")))

    def _live_session_keys(self) -> list[str]:
        if self._chat_by_session:
            return list(self._chat_by_session)
        user_id = int(self.config.get("telegram_allowed_user_id", 0) or 0)
        return [self.session_key(type("Ctx", (), {"user_id": user_id, "chat_id": user_id})())] if user_id else []

    def _check_user(self, update) -> bool:
        allowed = int(self.config.get("telegram_allowed_user_id", 0) or 0)
        return not allowed or bool(update.effective_user and update.effective_user.id == allowed)

    async def _with_typing(self, chat, fn, ChatAction):
        stop = asyncio.Event()
        async def pulse():
            while not stop.is_set():
                try:
                    await chat.send_action(ChatAction.TYPING)
                    await asyncio.wait_for(stop.wait(), 4)
                except asyncio.TimeoutError:
                    pass
                except Exception:
                    return
        task = asyncio.create_task(pulse())
        try:
            await self._run(fn)
        finally:
            stop.set()
            await task

    async def _run(self, fn):
        return await asyncio.get_running_loop().run_in_executor(None, fn)

    async def _handle_attachment(self, update, ChatAction):
        msg, key = update.message, self.session_key(update)
        tg_file, file_name, size = None, "attachment", 0
        if msg.photo:
            tg_file, file_name = await msg.photo[-1].get_file(), "photo.jpg"
        elif msg.document:
            tg_file, file_name, size = await msg.document.get_file(), msg.document.file_name or "document", msg.document.file_size or 0
        elif msg.voice:
            tg_file, file_name, size = await msg.voice.get_file(), "voice.ogg", msg.voice.file_size or 0
        elif msg.audio:
            tg_file, file_name, size = await msg.audio.get_file(), msg.audio.file_name or "audio.mp3", msg.audio.file_size or 0
        if not tg_file:
            return
        if size > _MAX_FILE_SIZE:
            await msg.reply_text("File too large (50 MB limit).")
            return
        cache_path = save_attachment(file_name, bytes(await tg_file.download_as_bytearray()), float(self.config.get("attachment_cache_size_gb", 2.0)))
        await self._with_typing(msg.chat, lambda: self.submit(key, ACTION_SEND_ATTACHMENT, {"path": str(cache_path), "extension": cache_path.suffix.lstrip("."), "caption": msg.caption or "", "file_name": file_name, "is_photo": bool(msg.photo)}), ChatAction)

    def _send_text(self, session_key: str, text: str, use_html: bool = True, markup=None) -> None:
        chat_id = self._chat_id(session_key)
        if chat_id:
            self._send(self._send_text_async(chat_id, text, use_html, markup))

    async def _send_text_async(self, chat_id: int, text: str, use_html: bool, markup=None):
        session_key = next((k for k, v in self._chat_by_session.items() if v == chat_id), None)
        if session_key and markup:
            await self._clear_last_keyboard_async(session_key)
        elif session_key:
            await self._clear_last_keyboard_async(session_key)
        for chunk in _chunks(text, self.capabilities.max_message_chars or 4096):
            try:
                sent = await self.app.bot.send_message(chat_id, chunk, parse_mode="HTML" if use_html else None, reply_markup=markup)
            except Exception:
                sent = await self.app.bot.send_message(chat_id, html.unescape(chunk), reply_markup=markup)
            if session_key and markup:
                self._last_keyboard[session_key] = (chat_id, sent.message_id)
            markup = None

    async def _send_media(self, chat_id: int | None, paths: list[str]):
        if not chat_id:
            return
        from telegram import InputMediaAudio, InputMediaDocument, InputMediaPhoto, InputMediaVideo
        async def one(p: Path, method: str):
            if method == "photo":
                await self.app.bot.send_photo(chat_id, photo=prepare_photo_bytes(p))
            elif method == "video":
                await self.app.bot.send_video(chat_id, video=file_bytes(p))
            elif method == "audio":
                await self.app.bot.send_audio(chat_id, audio=file_bytes(p), title=p.stem)
            else:
                await self.app.bot.send_document(chat_id, document=file_bytes(p), filename=p.name)
        for action in prepare_media_actions(paths, self.capabilities.max_upload_size or _MAX_FILE_SIZE):
            try:
                if action.method == "media_group":
                    media = []
                    for p in action.files:
                        method = "video" if action.group_type == "photo_video" and p.suffix.lower() in VIDEO_EXTENSIONS else "photo" if action.group_type == "photo_video" else "audio" if action.group_type == "audio" else "document"
                        media.append(InputMediaPhoto(prepare_photo_bytes(p)) if method == "photo" else InputMediaVideo(file_bytes(p)) if method == "video" else InputMediaAudio(file_bytes(p), title=p.stem) if method == "audio" else InputMediaDocument(file_bytes(p), filename=p.name))
                    await self.app.bot.send_media_group(chat_id, media) if len(media) > 1 else await one(action.files[0], method)
                elif action.method == "text":
                    await self.app.bot.send_message(chat_id, action.text_content, parse_mode="HTML")
                else:
                    await one(action.files[0], action.method)
            except Exception as e:
                logger.error(f"Failed to send Telegram attachment: {e}")
                await self.app.bot.send_message(chat_id, f"Failed to send attachment: {e}")

    async def _send_tool_started(self, chat_id: int, key: str, name: str, text: str):
        sent = await self.app.bot.send_message(chat_id, f"⟳ <code>{html.escape(text)}</code>", parse_mode="HTML", disable_notification=True)
        self._tool_messages[key] = (chat_id, sent.message_id, name, text)

    async def _progress_tool_message(self, chat_id: int, key: str, name: str, text: str):
        entry = self._tool_messages.get(key)
        if not entry:
            return await self._send_tool_started(chat_id, key, name, text)
        self._tool_messages[key] = (entry[0], entry[1], name, text)
        try:
            await self.app.bot.edit_message_text(f"⟳ <code>{html.escape(text)}</code>", chat_id=entry[0], message_id=entry[1], parse_mode="HTML")
        except Exception:
            pass

    async def _finish_tool_message(self, key: str, chat_id: int, name: str, text: str, ok: bool, error: str | None):
        entry = self._tool_messages.pop(key, None)
        display = entry[3] if entry else text
        text = f"{'✓' if ok else '✗'} <code>{html.escape(display)}</code>"
        if error and not ok:
            text += f" ({html.escape(str(error))})"
        if entry:
            try:
                await self.app.bot.edit_message_text(text, chat_id=entry[0], message_id=entry[1], parse_mode="HTML")
                return
            except Exception:
                pass
        await self.app.bot.send_message(chat_id, text, parse_mode="HTML", disable_notification=True)

    def _send(self, coro) -> None:
        if self.loop is None or self.app is None:
            coro.close()
            return
        try:
            if asyncio.get_running_loop() is self.loop:
                self.loop.create_task(coro)
                return
        except RuntimeError:
            pass
        asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout=30)

    def _chat_id(self, session_key: str) -> int | None:
        if session_key in self._chat_by_session:
            return self._chat_by_session[session_key]
        try:
            chat = int(session_key.split(":")[2])
            self._chat_by_session[session_key] = chat
            return chat
        except Exception:
            return None

    def _prompt(self, form: dict) -> str:
        field = form.get("field") or {}
        display = form.get("display") or {}
        prompt = display.get("prompt") or field.get("prompt") or field.get("name") or "Input required"
        bits = [html.escape(str(prompt))]
        assist = display.get("assist")
        if assist:
            bits.append(f"<i>{html.escape(str(assist))}</i>")
        return "\n".join(bits)

    def _enum_markup(self, key: str, form: dict):
        field = form.get("field") or {}
        display = form.get("display") or {}
        choices = display.get("choices") or [{"value": v, "label": str(v)} for v in (field.get("enum") or [])]
        cols, buttons = max(1, int(field.get("columns") or 1)), [self._button(str(c.get("label") or c.get("value")), key, str(c.get("value")), self._form_echo(form, c.get("value"))) for c in choices]
        rows = [buttons[i:i + cols] for i in range(0, len(buttons), cols)]
        if display.get("allow_skip", field.get("required") is False):
            rows.append([self._button("Skip", key, "/skip")])
        if display.get("allow_cancel", True):
            rows.append([self._button("Cancel", key, "/cancel")])
        return self._markup(rows)

    def _approval_markup(self, key: str, req):
        request_id = getattr(req, "id", "pending")
        if getattr(req, "enum", None):
            return self._markup([[self._button(str(v), key, f"approval:{request_id}:{v}")] for v in req.enum])
        if getattr(req, "type", "boolean") == "boolean":
            return self._markup([[self._button("Approve", key, f"approval:{request_id}:allow"), self._button("Deny", key, f"approval:{request_id}:deny")]])
        return None

    def _buttons_markup(self, key: str, buttons: list[dict]):
        return self._markup([[self._button(str(b.get("label") or b.get("text") or b.get("value") or "Option"), key, str(b.get("value") or b.get("text") or b.get("label") or ""))] for b in buttons])

    def _button(self, label: str, key: str, value: str, echo: str | None = None):
        from telegram import InlineKeyboardButton
        token = "bf:" + uuid.uuid4().hex[:16]
        self._callbacks[token] = (key, value, echo)
        return InlineKeyboardButton(label[:64], callback_data=token)

    def _form_echo(self, form: dict, value) -> str | None:
        if form.get("action_type") != "call_command" or not form.get("name"):
            return None
        parts = ["/" + str(form["name"])]
        parts += [_quote(v) for v in (form.get("collected") or {}).values()]
        parts.append(_quote(value))
        return " ".join(parts)

    def _next_approval(self, key: str):
        with self._approval_lock:
            return next(req for req in self._pending_approvals.get(key, {}).values() if not getattr(req, "is_resolved", False))

    @staticmethod
    def _parse_approval(text: str) -> bool | None:
        value = (text or "").strip().lower()
        if value in {"n", "no", "deny", "denied", "false", "0"}:
            return False
        if value in {"y", "yes", "approve", "approved", "true", "1"}:
            return True
        return None

    @staticmethod
    def _markup(rows):
        from telegram import InlineKeyboardMarkup
        return InlineKeyboardMarkup(rows) if rows else None

    def _clear_last_keyboard(self, key: str) -> None:
        self._send(self._clear_last_keyboard_async(key))

    async def _clear_last_keyboard_async(self, key: str):
        entry = self._last_keyboard.pop(key, None)
        if not entry or self.app is None:
            return
        try:
            await self.app.bot.edit_message_reply_markup(chat_id=entry[0], message_id=entry[1], reply_markup=None)
        except Exception:
            pass


def _quote(value) -> str:
    import json
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    text = str(value)
    return json.dumps(text) if any(ch.isspace() for ch in text) else text
