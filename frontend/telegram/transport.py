from __future__ import annotations

import asyncio
import logging

from frontend.telegram.renderers import VIDEO_EXTENSIONS, SendAction, prepare_media_actions, prepare_photo_bytes
from frontend.types import FrontendAction, FrontendSession

logger = logging.getLogger("TelegramTransport")


class TelegramTransport:
    def __init__(self, adapter, get_app, get_loop, render_text):
        self.adapter = adapter
        self._get_app = get_app
        self._get_loop = get_loop
        self._render_text = render_text
        self._status_message_ids: dict[str, int] = {}

    def clear_statuses(self):
        self._status_message_ids.clear()

    async def send_long_message(self, chat_id: int, text: str, use_html: bool = False):
        if not text:
            return
        app = self._get_app()
        parse_mode = "HTML" if use_html else None
        if len(text) <= self.adapter.capabilities.max_message_chars:
            try:
                await app.bot.send_message(chat_id, text, parse_mode=parse_mode)
            except Exception:
                if parse_mode:
                    await app.bot.send_message(chat_id, text)
            return
        current = ""
        for line in text.split("\n"):
            if len(current) + len(line) + 1 > self.adapter.capabilities.max_message_chars:
                await self.send_long_message(chat_id, current, use_html=use_html)
                current = line[:self.adapter.capabilities.max_message_chars]
            else:
                current = f"{current}\n{line}" if current else line
        if current:
            await self.send_long_message(chat_id, current, use_html=use_html)

    async def execute_send_actions(self, chat_id: int, actions: list[SendAction]):
        from telegram import InputMediaAudio, InputMediaDocument, InputMediaPhoto, InputMediaVideo

        app = self._get_app()
        for action in actions:
            try:
                if action.method == "media_group":
                    media = []
                    for file_path in action.files:
                        ext = file_path.suffix.lower()
                        if action.group_type == "photo_video":
                            media.append(InputMediaVideo(open(file_path, "rb")) if ext in VIDEO_EXTENSIONS else InputMediaPhoto(prepare_photo_bytes(file_path)))
                        elif action.group_type == "audio":
                            media.append(InputMediaAudio(open(file_path, "rb"), title=file_path.stem))
                        else:
                            media.append(InputMediaDocument(open(file_path, "rb"), filename=file_path.name))
                    await app.bot.send_media_group(chat_id, media)
                elif action.method == "photo":
                    await app.bot.send_photo(chat_id, photo=prepare_photo_bytes(action.files[0]))
                elif action.method == "video":
                    await app.bot.send_video(chat_id, video=open(action.files[0], "rb"))
                elif action.method == "audio":
                    await app.bot.send_audio(chat_id, audio=open(action.files[0], "rb"), title=action.files[0].stem)
                elif action.method == "document":
                    await app.bot.send_document(chat_id, document=open(action.files[0], "rb"), filename=action.files[0].name)
                elif action.method == "text":
                    await app.bot.send_message(chat_id, action.text_content, parse_mode="HTML")
            except Exception as e:
                names = ", ".join(file_path.name for file_path in action.files) if action.files else "(text)"
                logger.error(f"Failed to send {names}: {e}")
                await app.bot.send_message(chat_id, f"(Failed to send: {names})")

    def button_rows(self, chat_id: int, action: FrontendAction):
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup

        rows = []
        prefix = action.metadata.get("choice_prefix") or ""
        context = action.metadata.get("choice_context")
        request_id = action.metadata.get("request_id")
        for button in action.buttons:
            value = button["value"]
            if prefix == "approval" and request_id:
                callback = f"{request_id}:{value}"
            elif prefix in {"call", "trigger", "mdladd"} and context is not None:
                callback = f"{prefix}:{chat_id}:{context}:{value}"
            elif prefix == "cfgval" and context is not None:
                callback = f"cfgval:{context}:{value}"
            elif prefix and context is not None:
                callback = f"{prefix}:{context}:{value}"
            elif prefix:
                callback = f"{prefix}:{value}"
            else:
                callback = value
            rows.append([InlineKeyboardButton(button["label"], callback_data=callback)])
        return InlineKeyboardMarkup(rows) if rows else None

    async def send_frontend_action(self, session: FrontendSession, action: FrontendAction):
        app = self._get_app()
        chat_id = int(session.chat_id)
        use_html = action.parse_mode == "HTML" or (
            self.adapter.capabilities.supports_rich_text
            and action.type in {"send_message", "show_choices", "request_form_input"}
        )
        rendered = self._render_text(action.text) if use_html and action.text else action.text
        reply_markup = self.button_rows(chat_id, action)

        if action.type == "send_attachments":
            if action.attachments:
                await self.execute_send_actions(chat_id, prepare_media_actions(action.attachments))
            if action.text:
                await self.send_long_message(chat_id, rendered, use_html=use_html)
            return

        if action.type == "update_status" and action.status_id:
            message_id = self._status_message_ids.get(action.status_id)
            if message_id is not None:
                try:
                    await app.bot.edit_message_text(rendered, chat_id=chat_id, message_id=message_id)
                    return
                except Exception:
                    pass

        if action.type == "show_status":
            sent = await app.bot.send_message(chat_id, rendered, disable_notification=True)
            if action.status_id:
                self._status_message_ids[action.status_id] = sent.message_id
            return

        if reply_markup is not None:
            await app.bot.send_message(chat_id, rendered, reply_markup=reply_markup, parse_mode="HTML" if use_html else None)
            return

        await self.send_long_message(chat_id, rendered, use_html=use_html)

    def dispatch_runtime_action(self, session: FrontendSession, action: FrontendAction):
        loop = self._get_loop()
        app = self._get_app()
        if loop is None or app is None:
            logger.info("Telegram not ready yet; dropping action.")
            return
        try:
            if asyncio.get_running_loop() is loop:
                loop.create_task(self.send_frontend_action(session, action))
                return
        except RuntimeError:
            pass
        try:
            asyncio.run_coroutine_threadsafe(self.send_frontend_action(session, action), loop).result(timeout=30)
        except Exception as e:
            logger.error(f"Failed to dispatch Telegram action: {e}")
