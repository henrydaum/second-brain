"""Telegram frontend plugin backed by the conversation runtime."""

from __future__ import annotations

dependencies_files = ['frontends/helpers/telegram_renderers.py']
dependencies_pip = ['python-telegram-bot']

import asyncio
import html
import json
import logging
import re
import threading
import time
import uuid
from pathlib import Path

from attachments.cache import save as save_attachment
from plugins.BaseFrontend import BaseFrontend, FrontendCapabilities
from plugins.frontends.helpers.command_registry import format_command_call
from .helpers.telegram_renderers import (
    VIDEO_EXTENSIONS,
    StreamTracker,
    file_bytes,
    prepare_media_actions,
    prepare_photo_bytes,
)
from state_machine.action_map import ACTION_SEND_ATTACHMENT
from state_machine.conversation_phases import PHASE_APPROVING_REQUEST

logger = logging.getLogger("TelegramFrontend")

_MAX_FILE_SIZE = 50 * 1024 * 1024

# Titles that still look kernel-generated: no topic pin is made for these
# (mirrors the update_titles task's _needs_title). Covers "New Conversation",
# "New conversation (Main)", empty, and "/clear"-stamped "... (cleared)".
_DEFAULT_TITLE_RE = re.compile(r"^new conversation\b", re.IGNORECASE)


def _is_default_title(title: str) -> bool:
    """True while a conversation has no real, user-meaningful title yet."""
    text = (title or "").strip()
    return not text or bool(_DEFAULT_TITLE_RE.match(text)) or text.endswith("(cleared)")


class _NetworkErrorThrottle(logging.Filter):
    """Collapse repeated Telegram polling network errors into one log per window.

    When DNS/connectivity drops, python-telegram-bot's updater retries forever and
    logs a full traceback every cycle — overnight that floods the log with the same
    ``getaddrinfo failed`` stack. This filter lets the first failure through (as a
    terse one-liner, no traceback), suppresses matching records for a cooldown, and
    notes how many it swallowed when the next one is allowed through.
    """

    _MARKERS = ("polling", "networkerror", "getaddrinfo", "connecterror",
                "readerror", "httpx.connect", "pool timeout", "timed out",
                "connection reset", "remoteprotocolerror")

    def __init__(self, cooldown: float = 300.0):
        super().__init__()
        self.cooldown = cooldown
        self._suppress_until = 0.0
        self._suppressed = 0

    def filter(self, record: logging.LogRecord) -> bool:
        # The updater's polling error logs a generic message ("Exception happened
        # while polling for updates.") and stashes the real cause (getaddrinfo /
        # NetworkError / ReadError) in exc_info — so match against both.
        text = record.getMessage().lower()
        if record.exc_info and record.exc_info[1] is not None:
            text += " " + repr(record.exc_info[1]).lower()
        if record.levelno < logging.ERROR or not any(m in text for m in self._MARKERS):
            return True
        now = time.time()
        if now < self._suppress_until:
            self._suppressed += 1
            return False
        # Allowed through: drop the giant traceback and summarize any backlog.
        record.exc_info = None
        record.exc_text = None
        if self._suppressed:
            record.msg = f"{record.getMessage()} (+{self._suppressed} more suppressed in the last {int(self.cooldown)}s)"
            record.args = ()
        self._suppress_until = now + self.cooldown
        self._suppressed = 0
        return True


_NETWORK_THROTTLE_INSTALLED = False


def _install_network_log_throttle() -> None:
    """Attach the network-error throttle to the noisy PTB polling loggers once."""
    global _NETWORK_THROTTLE_INSTALLED
    if _NETWORK_THROTTLE_INSTALLED:
        return
    throttle = _NetworkErrorThrottle()
    for name in ("telegram.ext.Updater", "telegram.ext.ExtBot", "telegram.request"):
        logging.getLogger(name).addFilter(throttle)
    _NETWORK_THROTTLE_INSTALLED = True


def _md_to_tg_html(text: str) -> str:
    """Convert lightweight markdown-ish output into Telegram-safe HTML."""
    parts, last = [], 0
    for m in re.finditer(r"```(\w*)\n(.*?)```", text or "", re.DOTALL):
        parts.append(_blocks(text[last:m.start()]))
        code = html.escape(m.group(2).rstrip())
        parts.append(f'<pre><code class="language-{html.escape(m.group(1))}">{code}</code></pre>' if m.group(1) else f"<pre>{code}</pre>")
        last = m.end()
    return "".join(parts + [_blocks((text or "")[last:])])


_TABLE_BLOCK = re.compile(r"^[ \t]*\|.*\|[ \t]*\n[ \t]*\|(?:\s*:?-{3,}:?\s*\|)+[ \t]*\n(?:[ \t]*\|.*\|[ \t]*(?:\n|$))*", re.MULTILINE)
_QUOTE_BLOCK = re.compile(r"^(?:>[ \t]?.*(?:\n|$))+", re.MULTILINE)


def _compact_detail_cards(text: str) -> str:
    """Turn detail cards (two-column tables with an empty second header cell,
    the kernel's ``detail_card`` shape) into fenced code blocks.

    Full-width rendered tables are overkill for a title + a few key/value
    rows; the compact monospace card reads better on a phone. Real data
    tables (non-empty headers) stay markdown and render natively.
    """
    from plugins.frontends.helpers.formatters import align_md_tables

    def replace(m: "re.Match") -> str:
        header = [c.strip() for c in m.group(0).split("\n", 1)[0].strip().strip("|").split("|")]
        if len(header) == 2 and header[0] and not header[1]:
            return f"```\n{align_md_tables(m.group(0).strip())}\n```\n"
        return m.group(0)

    return _TABLE_BLOCK.sub(replace, text or "")


def _blocks(text: str) -> str:
    """Render markdown tables (aligned <pre>) and > blockquotes; inline the rest.

    Only the non-rich HTML fallback comes through here — the Rich Messages
    path sends raw markdown and Telegram renders it server-side.
    """
    from plugins.frontends.helpers.formatters import align_md_tables
    out, last = [], 0
    pattern = re.compile(f"(?:{_TABLE_BLOCK.pattern})|(?:{_QUOTE_BLOCK.pattern})", re.MULTILINE)
    for m in pattern.finditer(text or ""):
        out.append(_inline(text[last:m.start()]))
        block = m.group(0)
        if block.lstrip().startswith("|"):
            out.append(f"<pre>{html.escape(align_md_tables(block.strip()))}</pre>")
        else:
            quoted = "\n".join(re.sub(r"^>[ \t]?", "", line) for line in block.strip().split("\n"))
            out.append(f"<blockquote>{_inline(quoted)}</blockquote>")
        last = m.end()
    return "".join(out + [_inline((text or "")[last:])])


def _inline(text: str) -> str:
    """Render inline code spans while preserving the surrounding rich text."""
    out, last = [], 0
    for m in re.finditer(r"`([^`]+)`", text):
        out.append(_bold_italic(text[last:m.start()]))
        out.append(f"<code>{html.escape(m.group(1))}</code>")
        last = m.end()
    return "".join(out + [_bold_italic(text[last:])])


def _bold_italic(text: str) -> str:
    """Translate simple bold and italic markers into Telegram HTML tags."""
    escaped = html.escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    return re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", escaped)


def _chunks(text: str, max_chars: int = 4096) -> list[str]:
    """Split long output into Telegram-sized message chunks."""
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
    """Frontend adapter for Telegram."""
    name = "telegram"
    description = "Telegram chat frontend backed by the conversation state machine."
    capabilities = FrontendCapabilities(
        supports_typing=True,
        supports_buttons=True,
        supports_message_edit=True,
        supports_attachments_in=True,
        supports_attachments_out=True,
        supports_inline_forms=True,
        supports_proactive_push=True,
        supports_rich_text=True,
        max_message_chars=4096,
        max_upload_size=_MAX_FILE_SIZE,
        supports_streaming=True,
    )
    config_settings = [
        ("Telegram Bot Token", "telegram_bot_token", "Bot token from @BotFather. Required for Telegram frontend.", "", {"type": "text"}),
        ("Telegram Allowed User ID", "telegram_allowed_user_id", "Your Telegram user ID (integer). Only this user can interact with the bot. Send /start to @userinfobot to find yours.", 0, {"type": "text"}),
        ("Telegram Conversation Banner", "telegram_pin_banner", "Keep a pinned message at the top of the chat showing the current conversation's title, updated on switch/retitle.", True, {"type": "bool"}),
        ("Telegram Banner Messages", "telegram_banner_messages", "Pinned banner message ids per chat (internal bookkeeping).", {}, {"type": "text", "hidden": True}),
    ]
    # Fallback guidance for installs where Rich Messages are unavailable
    # (old python-telegram-bot or pre-10.1 Bot API server).
    agent_prompt = (
        "## Talking over Telegram\n"
        "This conversation is on Telegram, a mobile chat app. Keep replies concise and skimmable. Only a "
        "subset of formatting renders: **bold**, *italic*, `inline code`, and ```fenced code blocks```. "
        "Markdown headings, tables, and link syntax do NOT render — avoid them and write plainly instead. "
        "Long messages are split across multiple sends, and file uploads are capped at 50 MB."
    )

    _AGENT_PROMPT_RICH = (
        "## Talking over Telegram\n"
        "This conversation is on Telegram (a mobile chat app), rendered as native Rich Messages: standard "
        "Markdown displays with full fidelity — headings, **bold**, *italic*, ~~strikethrough~~, "
        "`inline code`, fenced code blocks with language tags, [links](https://example.com), bulleted and "
        "numbered lists, tables, > blockquotes, and --- dividers. Use whatever structure serves the reply, "
        "but it is still a phone screen: keep replies concise and skimmable. Long messages are split across "
        "multiple sends, and file uploads are capped at 50 MB."
    )

    def agent_prompt_for(self, ctx) -> str:
        """Advertise rich or basic formatting based on live capability.

        Recomputed every prompt build, so a runtime downgrade (rich send
        rejected by an older server) flips the guidance on the next turn."""
        return self._AGENT_PROMPT_RICH if self._rich_capable() else self.agent_prompt

    def __init__(self, shutdown_event: threading.Event | None = None, services: dict | None = None):
        """Initialize the Telegram frontend."""
        super().__init__()
        self.shutdown_event = shutdown_event or threading.Event()
        self.services = services or {}
        self.loop = None
        self.app = None
        self._chat_by_session: dict[str, int] = {}
        self._callbacks: dict[str, tuple[str, str, str | None]] = {}
        self._tool_messages: dict[str, tuple[int, int, str, str]] = {}
        self._last_keyboard: dict[str, tuple[int, int]] = {}
        # One StreamTracker (+ pump task) per in-flight streamed reply,
        # keyed by (session_key, stream_id).
        self._streams: dict[tuple[str, str], StreamTracker] = {}
        # Rich Messages (Bot API 10.1) capability: None = undetermined,
        # False = confirmed unavailable (old PTB or pre-10.1 server).
        self._rich: bool | None = None
        # Pinned topic banners: chat_id -> {conversation_id: pinned_title}.
        # Persisted (hidden config) so a conversation pins exactly once —
        # restarts and switches never re-pin. None until first use because
        # config is only bound after __init__.
        self._banners: dict[int, dict[int, str]] | None = None
        # Most recent inbound message per session, for reaction acks.
        self._inbound: dict[str, tuple[int, int]] = {}

    def session_key(self, ctx) -> str:
        """Build the per-user, per-chat, per-thread session key for Telegram traffic."""
        user = getattr(getattr(ctx, "effective_user", None), "id", None) or getattr(ctx, "user_id", 0)
        chat = getattr(getattr(ctx, "effective_chat", None), "id", None) or getattr(ctx, "chat_id", user)
        thread = getattr(getattr(ctx, "effective_message", None), "message_thread_id", None) or 0
        key = f"telegram:{user}:{chat}:{thread}"
        if chat:
            self._chat_by_session[key] = int(chat)
        return key

    def start(self) -> None:
        """Start the Telegram bot loop and bind it to the runtime."""
        token = str(self.config.get("telegram_bot_token", "")).strip()
        if not token:
            logger.info("telegram_bot_token not configured; Telegram frontend disabled.")
            return
        try:
            from telegram import BotCommand, Update
            from telegram.constants import ChatAction
            from telegram.error import NetworkError, TimedOut
            from telegram.ext import Application, CallbackQueryHandler, MessageHandler, filters
        except ImportError:
            logger.warning("Telegram frontend not available; install python-telegram-bot.")
            return

        # Keep transient connectivity loss from flooding the log overnight.
        _install_network_log_throttle()

        async def handle_text(update: Update, _ctx):
            """Handle one incoming Telegram text message."""
            if not self._check_user(update) or not update.message:
                return
            key = self.session_key(update)
            text = re.sub(r"^/([A-Za-z0-9_]+)@[^\s]+", r"/\1", (update.message.text or "").strip())
            if text:
                self._inbound[key] = (update.message.chat_id, update.message.message_id)
                await self._with_typing(update.message.chat, lambda: self.submit_text(key, text), ChatAction)

        async def handle_attachment(update: Update, _ctx):
            """Handle one incoming Telegram attachment message."""
            if not self._check_user(update) or not update.message:
                return
            await self._handle_attachment(update, ChatAction)

        async def handle_callback(update: Update, _ctx):
            """Handle an inline-button callback from Telegram."""
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
            """Own the Telegram app lifecycle inside the frontend thread."""
            self.loop = asyncio.get_running_loop()
            self.app = Application.builder().token(token).concurrent_updates(True).build()
            self.app.add_handler(MessageHandler(filters.COMMAND | (filters.TEXT & ~filters.COMMAND), handle_text))
            self.app.add_handler(MessageHandler(filters.PHOTO | filters.Document.ALL | filters.VOICE | filters.AUDIO, handle_attachment))
            self.app.add_handler(CallbackQueryHandler(handle_callback))

            async def on_error(_update, ctx):
                """Swallow transient network errors quietly; log the rest once."""
                err = getattr(ctx, "error", None)
                if isinstance(err, (NetworkError, TimedOut)):
                    return  # the throttle already covers the updater's own retries
                logger.error(f"Telegram handler error: {err}")
            self.app.add_error_handler(on_error)

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
        """Stop Telegram frontend."""
        self.shutdown_event.set()
        self.unbind()

    def submit_text(self, session_key: str, text: str):
        """Submit chat text or resolve a pending approval from Telegram."""
        if self._current_phase(session_key) != PHASE_APPROVING_REQUEST and self.has_pending_approval(session_key):
            req = self._next_approval(session_key)
            value = self._parse_approval(text) if getattr(req, "type", "boolean") == "boolean" else text
            if value is None:
                return self.render_error(session_key, {"message": "Approval needs yes or no."})
            if self.resolve_approval(session_key, req.id, value, self.name):
                self.render_messages(session_key, ["Received."])
        else:
            return super().submit_text(session_key, text)

    def render_messages(self, session_key: str, messages: list[str]) -> None:
        """Send chat messages — native Rich Messages (Markdown) when available."""
        self._clear_last_keyboard(session_key)
        chat_id = self._chat_id(session_key)
        if not chat_id:
            return
        for msg in messages:
            self._send(self._deliver_message_async(chat_id, msg))

    # ──────────────────────────────────────────────────────────────────
    # Rich Messages (Bot API 10.1): InputRichMessage accepts raw Markdown,
    # parsed server-side into headings/tables/lists/code — no local
    # conversion needed. python-telegram-bot has no typed support yet
    # (python-telegram-bot#5261), so calls go through the typed method when
    # present and PTB's raw request layer otherwise.
    # ──────────────────────────────────────────────────────────────────

    def _rich_capable(self) -> bool:
        """Whether Rich Message endpoints look reachable. Optimistic before
        the transport is up (so the agent prompt starts rich); downgraded to
        False the first time the API refuses."""
        if self._rich is None:
            bot = getattr(self.app, "bot", None)
            if bot is None:
                return True
            self._rich = bool(hasattr(bot, "send_rich_message") or hasattr(bot, "do_api_request"))
        return self._rich

    async def _rich_request(self, endpoint: str, payload: dict) -> None:
        """Call a Rich Message endpoint (typed PTB method or raw layer)."""
        bot = self.app.bot
        snake = re.sub(r"(?<!^)(?=[A-Z])", "_", endpoint).lower()
        method = getattr(bot, snake, None)
        if method is not None:
            import telegram
            InputRichMessage = getattr(telegram, "InputRichMessage", None)
            if InputRichMessage is not None:
                payload = {**payload, "rich_message": InputRichMessage(**payload["rich_message"])}
            await method(**payload)
            return
        await bot.do_api_request(endpoint, api_kwargs=payload)

    def _rich_refused(self, e: Exception) -> bool:
        """True when the error means Rich Messages don't exist here at all."""
        text = str(e).lower()
        return "not found" in text or "unknown method" in text

    async def _deliver_message_async(self, chat_id: int, text: str) -> None:
        """Deliver one message: rich Markdown first, HTML pipeline fallback."""
        chunks = _chunks(_compact_detail_cards(text), self.capabilities.max_message_chars or 4096)
        sent = 0
        if self._rich_capable():
            try:
                for chunk in chunks:
                    await self._rich_request("sendRichMessage", {
                        "chat_id": chat_id,
                        "rich_message": {"markdown": chunk},
                    })
                    sent += 1
                return
            except Exception as e:
                if self._rich_refused(e):
                    self._rich = False
                    logger.info("Rich Messages unavailable; using HTML rendering from now on.")
                else:
                    logger.warning(f"sendRichMessage failed ({e}); HTML fallback for this message.")
        for chunk in chunks[sent:]:
            await self._send_text_async(chat_id, _md_to_tg_html(chunk), True)

    def render_queued_ack(self, session_key: str) -> bool:
        """Acknowledge a queued mid-turn message with a 👍 reaction.

        Quieter than a whole "Got it" message. If the reaction can't be
        delivered (old PTB / API refusal) the coroutine falls back to the
        text ack, so the user always gets *some* acknowledgement.
        """
        entry = self._inbound.get(session_key)
        if not entry or self.app is None or self.loop is None:
            return False
        self._send_nowait(self._react_ack(session_key, *entry))
        return True

    async def _react_ack(self, session_key: str, chat_id: int, message_id: int) -> None:
        try:
            set_reaction = getattr(self.app.bot, "set_message_reaction", None)
            if set_reaction is not None:
                await set_reaction(chat_id, message_id, reaction="👍")
                return
            await self.app.bot.do_api_request("setMessageReaction", api_kwargs={
                "chat_id": chat_id, "message_id": message_id,
                "reaction": [{"type": "emoji", "emoji": "👍"}],
            })
        except Exception as e:
            logger.debug(f"Reaction ack failed ({e}); sending the text ack instead.")
            await self._deliver_message_async(chat_id, "Got it — I'll read that as soon as I finish this step.")

    def render_conversation_banner(self, session_key: str, info: dict) -> None:
        """Pin a fresh banner the first time a conversation earns a real title.

        Pins are topic-based and immutable: one is created when a
        conversation is (re)titled, never for a still-default "New
        Conversation", and existing pins are never edited or re-pinned.
        Dedup is per conversation id (persisted), so switches, retitle
        echoes, and restarts don't create duplicate pins.
        """
        if not (self.config or {}).get("telegram_pin_banner", True):
            return
        chat_id = self._chat_id(session_key)
        title = (info.get("title") or "").strip()
        conversation_id = info.get("conversation_id")
        if not chat_id or not title or conversation_id is None or self.app is None:
            return
        if _is_default_title(title):
            return  # not a real topic yet — wait for the retitle
        if self._pinned_map().get(chat_id, {}).get(int(conversation_id)) == title:
            return  # already pinned this topic
        self._send_nowait(self._pin_topic(chat_id, int(conversation_id), title))

    async def _pin_topic(self, chat_id: int, conversation_id: int, title: str) -> None:
        """Send and pin a new topic banner, then record it so it pins once."""
        try:
            sent = await self.app.bot.send_message(chat_id, f"💬 {title}", disable_notification=True)
            await self.app.bot.pin_chat_message(chat_id, sent.message_id, disable_notification=True)
            self._pinned_map().setdefault(chat_id, {})[conversation_id] = title
            self._persist_pinned()
        except Exception as e:
            # Pinning needs no special rights in private chats, but fail soft:
            # a missing banner must never break message flow.
            logger.info(f"Conversation banner unavailable: {e}")

    def _pinned_map(self) -> dict[int, dict[int, str]]:
        """Per-chat {conversation_id: pinned_title}, loaded once from config."""
        if self._banners is None:
            raw = (self.config or {}).get("telegram_banner_messages", {})
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw) if raw.strip() else {}
                except json.JSONDecodeError:
                    raw = {}
            pinned: dict[int, dict[int, str]] = {}
            if isinstance(raw, dict):
                for chat_id, topics in raw.items():
                    if not isinstance(topics, dict):
                        continue  # pre-topic shape ([msg_id, title]) — discard, start fresh
                    for cid, title in topics.items():
                        try:
                            pinned.setdefault(int(chat_id), {})[int(cid)] = str(title)
                        except (TypeError, ValueError):
                            continue
            self._banners = pinned
        return self._banners

    def _persist_pinned(self) -> None:
        """Write the pinned-topic map through to plugin config; failures are soft."""
        try:
            import config.config_manager as config_manager
            serialized = {
                str(chat_id): {str(cid): title for cid, title in topics.items()}
                for chat_id, topics in (self._banners or {}).items()
            }
            values = config_manager.load_plugin_config()
            values["telegram_banner_messages"] = serialized
            config_manager.save_plugin_config(values)
            if isinstance(self.config, dict):
                self.config["telegram_banner_messages"] = serialized
        except Exception as e:
            logger.debug(f"Banner persistence failed: {e}")

    def render_attachments(self, session_key: str, paths: list[str]) -> None:
        """Send rendered attachments back to Telegram."""
        self._clear_last_keyboard(session_key)
        self._send(self._send_media(self._chat_id(session_key), paths))

    def render_form_field(self, session_key: str, form: dict) -> None:
        """Prompt for one form field with Telegram-native buttons when available."""
        self._send_text(session_key, self._prompt(form), markup=self._enum_markup(session_key, form))

    def render_approval_request(self, session_key: str, req) -> None:
        """Render a pending approval request with Telegram controls."""
        body = _md_to_tg_html(f"{getattr(req, 'title', 'Approval requested')}\n\n{getattr(req, 'body', '')}".strip())
        self._send_text(session_key, body, markup=self._approval_markup(session_key, req))

    def render_buttons(self, session_key: str, buttons: list[dict]) -> None:
        """Render an ad hoc button list as an inline keyboard."""
        self._send_text(session_key, "Choose:", markup=self._buttons_markup(session_key, buttons))

    def render_error(self, session_key: str, error: dict) -> None:
        """Send an error message to Telegram."""
        self._clear_last_keyboard(session_key)
        self._send_text(session_key, html.escape(f"Error: {(error or {}).get('message') or error}"))

    _STREAM_CURSOR = " ▍"

    def render_stream_delta(self, session_key: str, payload: dict) -> None:
        """Feed streamed agent text into the per-stream tracker.

        Runs on the agent thread, so it never touches Telegram I/O — the
        first delta schedules one ``_stream_pump`` task on the event loop,
        which owns all sends/edits for that stream."""
        stream_id = payload.get("stream_id") or ""
        stream_key = (session_key, stream_id)
        if payload.get("done"):
            tracker = self._streams.get(stream_key)
            if tracker:
                tracker.finish(payload.get("final_text"), bool(payload.get("aborted")))
            return
        delta = payload.get("delta") or ""
        if not delta:
            return
        tracker = self._streams.get(stream_key)
        if tracker is None:
            chat_id = self._chat_id(session_key)
            if not chat_id or self.loop is None or self.app is None:
                return
            tracker = StreamTracker(max_chars=(self.capabilities.max_message_chars or 4096) - 96)
            self._streams[stream_key] = tracker
            tracker.feed(delta)
            self._send_nowait(self._stream_pump(stream_key, chat_id, tracker))
        else:
            tracker.feed(delta)

    def _send_nowait(self, coro) -> None:
        """Schedule a coroutine onto the Telegram loop without blocking.

        Unlike ``_send`` this never waits for the result — required on the
        agent thread, where blocking on the event loop would stall the turn."""
        if self.loop is None or self.app is None:
            coro.close()
            return
        try:
            if asyncio.get_running_loop() is self.loop:
                self.loop.create_task(coro)
                return
        except RuntimeError:
            pass
        asyncio.run_coroutine_threadsafe(coro, self.loop)

    @staticmethod
    def _draft_id_for(stream_id: str) -> int:
        """Derive a stable non-zero draft id from the kernel's stream id."""
        try:
            return (int(stream_id.rpartition("_")[2], 16) & 0x7FFFFFFF) or 1
        except ValueError:
            return (abs(hash(stream_id)) & 0x7FFFFFFF) or 1

    async def _stream_pump(self, stream_key: tuple[str, str], chat_id: int, tracker: StreamTracker):
        """Own one streamed reply, preferring native draft streaming.

        Mode ladder, downgrading in place when a call is refused:
        1. ``rich``  — ``sendRichMessageDraft`` (Bot API 10.1): partial
           Markdown streams with live rich formatting.
        2. ``draft`` — ``sendMessageDraft`` (9.3+): plain-text native typing
           animation.
        3. ``edit``  — legacy placeholder message edited on a throttle.

        Drafts are ephemeral 30s previews in private chats, so the reply is
        still delivered by ``_finalize_stream`` as a real message (the base
        dedup suppressed the whole-message copy, so this pump IS the delivery
        path). Flood-control: ``RetryAfter`` backs off and keeps buffering;
        a hard failure stops rendering but the final is still delivered."""
        from telegram.error import BadRequest, RetryAfter
        has_plain_draft = getattr(self.app.bot, "send_message_draft", None) is not None
        mode = "rich" if self._rich_capable() else ("draft" if has_plain_draft else "edit")
        draft_id = self._draft_id_for(stream_key[1])
        if mode != "edit":
            # Drafts are a dedicated streaming channel — much tighter cadence
            # than message edits without flirting with flood limits.
            tracker.edit_interval, tracker.burst_chars = 0.35, 64
        message_id = None
        next_allowed = 0.0
        broken = False

        def _downgrade(reason: Exception):
            nonlocal mode
            if mode == "rich":
                if self._rich_refused(reason):
                    self._rich = False
                mode = "draft" if has_plain_draft else "edit"
            else:
                mode = "edit"
            if mode == "edit":
                tracker.edit_interval, tracker.burst_chars = 1.75, 300
            logger.info(f"Telegram stream downgraded to '{mode}' ({reason}).")

        try:
            while True:
                done, aborted, final_text = tracker.state()
                if done:
                    break
                now = time.time()
                if not broken and now >= next_allowed and tracker.should_edit(now):
                    finals, current = tracker.take_render()
                    try:
                        if mode in {"rich", "draft"}:
                            for head in finals:
                                # Size-cap rollover: persist the head as a real
                                # message, keep drafting the tail.
                                await self._deliver_message_async(chat_id, head)
                            if current is not None:
                                if mode == "rich":
                                    await self._rich_request("sendRichMessageDraft", {
                                        "chat_id": chat_id,
                                        "draft_id": draft_id,
                                        "rich_message": {"markdown": current},
                                    })
                                else:
                                    await self.app.bot.send_message_draft(chat_id=chat_id, draft_id=draft_id, text=current)
                                tracker.mark_rendered(current, now)
                        else:
                            for head in finals:
                                if message_id is None:
                                    await self.app.bot.send_message(chat_id, head, disable_notification=True)
                                else:
                                    await self.app.bot.edit_message_text(head, chat_id=chat_id, message_id=message_id)
                                    message_id = None  # head finalized; the tail gets a fresh message
                            if current is not None:
                                if message_id is None:
                                    sent = await self.app.bot.send_message(chat_id, current + self._STREAM_CURSOR, disable_notification=True)
                                    message_id = sent.message_id
                                else:
                                    await self.app.bot.edit_message_text(current + self._STREAM_CURSOR, chat_id=chat_id, message_id=message_id)
                                tracker.mark_rendered(current, now)
                    except RetryAfter as e:
                        next_allowed = time.time() + float(getattr(e, "retry_after", 3) or 3) + 0.5
                    except BadRequest as e:
                        if mode in {"rich", "draft"}:
                            _downgrade(e)
                            continue
                        if current is not None:
                            tracker.mark_rendered(current, now)  # "message is not modified"
                    except Exception as e:
                        if mode == "rich" and self._rich_refused(e):
                            _downgrade(e)
                            continue
                        logger.warning(f"Telegram stream render failed; deferring to final delivery: {e}")
                        broken = True
                await asyncio.sleep(0.12 if mode != "edit" else 0.3)
            await self._finalize_stream(chat_id, message_id, tracker, aborted, final_text)
        except Exception:
            logger.exception("Telegram stream pump crashed")
        finally:
            self._streams.pop(stream_key, None)

    async def _finalize_stream(self, chat_id: int, message_id: int | None, tracker: StreamTracker, aborted: bool, final_text: str | None):
        """Bring the streamed message(s) to their final state."""
        remainder = tracker.remainder()
        if aborted:
            # Whatever follows (compaction retry answer, "Cancelled.") arrives
            # as a normal whole message; just drop the cursor or the empty
            # placeholder. (Draft modes: the ephemeral draft expires on its
            # own — nothing to clean up.)
            if message_id is not None:
                try:
                    if remainder:
                        await self.app.bot.edit_message_text(remainder, chat_id=chat_id, message_id=message_id)
                    else:
                        await self.app.bot.delete_message(chat_id, message_id)
                except Exception:
                    pass
            return
        if message_id is None:
            # Draft modes left no message behind — deliver the reply for real
            # (rich Markdown with HTML fallback; rolled heads already sent).
            text = remainder if tracker.rolled else (final_text or remainder)
            if text:
                await self._deliver_message_async(chat_id, text)
            return
        if tracker.rolled:
            # Rolled-over replies stay plain text; finalize the tail in place.
            text = remainder or final_text or ""
            try:
                if message_id is not None:
                    await self.app.bot.edit_message_text(text, chat_id=chat_id, message_id=message_id)
                elif text:
                    await self.app.bot.send_message(chat_id, text)
            except Exception:
                logger.exception("Telegram stream tail finalize failed")
            return
        # Common case: re-render the whole reply as HTML into the streamed
        # message, spilling extra chunks into fresh messages.
        chunks = _chunks(_md_to_tg_html(final_text or remainder), self.capabilities.max_message_chars or 4096) or [""]
        first, rest = chunks[0], chunks[1:]
        delivered = False
        if message_id is not None and first:
            for text, mode in ((first, "HTML"), (html.unescape(first), None)):
                try:
                    await self.app.bot.edit_message_text(text, chat_id=chat_id, message_id=message_id, parse_mode=mode)
                    delivered = True
                    break
                except Exception as e:
                    if "not modified" in str(e).lower():
                        delivered = True
                        break
        if not delivered and first:
            rest = [first, *rest]  # streamed message unusable — send everything fresh
        for chunk in rest:
            try:
                await self.app.bot.send_message(chat_id, chunk, parse_mode="HTML")
            except Exception:
                try:
                    await self.app.bot.send_message(chat_id, html.unescape(chunk))
                except Exception:
                    logger.exception("Telegram stream final chunk send failed")

    def render_tool_status(self, session_key: str, payload: dict) -> None:
        """Keep Telegram's single progress banner in sync with tool events."""
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
        """Return Telegram sessions that can receive proactive pushes right now."""
        if self._chat_by_session:
            return list(self._chat_by_session)
        user_id = int(self.config.get("telegram_allowed_user_id", 0) or 0)
        return [self.session_key(type("Ctx", (), {"user_id": user_id, "chat_id": user_id})())] if user_id else []

    def _check_user(self, update) -> bool:
        """Return whether the incoming update is from the allowed Telegram user."""
        allowed = int(self.config.get("telegram_allowed_user_id", 0) or 0)
        return not allowed or bool(update.effective_user and update.effective_user.id == allowed)

    async def _with_typing(self, chat, fn, ChatAction):
        """Run blocking work while periodically showing Telegram typing state."""
        stop = asyncio.Event()
        async def pulse():
            """Refresh the typing indicator until the wrapped work finishes."""
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
        """Run blocking work off the Telegram event loop."""
        return await asyncio.get_running_loop().run_in_executor(None, fn)

    async def _handle_attachment(self, update, ChatAction):
        """Download one Telegram attachment, cache it locally, and submit it to the runtime."""
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
        """Queue a text send for the chat behind a session key."""
        chat_id = self._chat_id(session_key)
        if chat_id:
            self._send(self._send_text_async(chat_id, text, use_html, markup))

    async def _send_text_async(self, chat_id: int, text: str, use_html: bool, markup=None):
        """Send one text payload to Telegram, chunking and clearing old keyboards as needed."""
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
        """Send a batch of files back to Telegram using the best available media method."""
        if not chat_id:
            return
        from telegram import InputMediaAudio, InputMediaDocument, InputMediaPhoto, InputMediaVideo
        async def one(p: Path, method: str):
            """Send one file with the Telegram API method chosen for it."""
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
        """Create the hourglass status message for a new tool or command call."""
        sent = await self.app.bot.send_message(chat_id, f"⋯ <code>{html.escape(text)}</code>", parse_mode="HTML", disable_notification=True)
        self._tool_messages[key] = (chat_id, sent.message_id, name, text)

    async def _progress_tool_message(self, chat_id: int, key: str, name: str, text: str):
        """Update the existing tool-status banner without sending a new message."""
        entry = self._tool_messages.get(key)
        if not entry:
            return await self._send_tool_started(chat_id, key, name, text)
        self._tool_messages[key] = (entry[0], entry[1], name, text)
        try:
            await self.app.bot.edit_message_text(f"⋯ <code>{html.escape(text)}</code>", chat_id=entry[0], message_id=entry[1], parse_mode="HTML")
        except Exception:
            pass

    async def _finish_tool_message(self, key: str, chat_id: int, name: str, text: str, ok: bool, error: str | None):
        """Finalize the tool-status banner with success or failure text."""
        entry = self._tool_messages.pop(key, None)
        display = entry[3] if entry else text
        text = f"{'✓' if ok else '✕'} <code>{html.escape(display)}</code>"
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
        """Schedule a coroutine onto the Telegram loop from sync code."""
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
        """Recover or memoize the Telegram chat ID for a session key."""
        if session_key in self._chat_by_session:
            return self._chat_by_session[session_key]
        try:
            chat = int(session_key.split(":")[2])
            self._chat_by_session[session_key] = chat
            return chat
        except Exception:
            return None

    def _prompt(self, form: dict) -> str:
        """Build the visible Telegram prompt for a form field."""
        field = form.get("field") or {}
        display = form.get("display") or {}
        prompt = display.get("prompt") or field.get("prompt") or field.get("name") or "Input required"
        # Form prompts can carry markdown tables (e.g. the /packages overview);
        # inline keyboards can't ride on Rich Messages, so render via the HTML
        # converter, which aligns tables into <pre> blocks.
        bits = [_md_to_tg_html(str(prompt))]
        assist = display.get("assist")
        if assist:
            bits.append(f"<i>{html.escape(str(assist))}</i>")
        return "\n".join(bits)

    def _enum_markup(self, key: str, form: dict):
        """Build inline-keyboard choices for an enum-backed form field."""
        field = form.get("field") or {}
        display = form.get("display") or {}
        choices = display.get("choices") or [{"value": v, "label": str(v)} for v in (field.get("enum") or [])]
        cols, buttons = max(1, int(field.get("columns") or 1)), [self._button(str(c.get("label") or c.get("value")), key, str(c.get("value")), self._form_echo(form, c.get("value"))) for c in choices]
        rows = [buttons[i:i + cols] for i in range(0, len(buttons), cols)]
        if display.get("allow_back"):
            rows.append([self._button("⟵ Back", key, "/back")])
        if display.get("allow_skip", field.get("required") is False):
            rows.append([self._button("⟶ Skip", key, "/skip")])
        if display.get("allow_cancel", True):
            rows.append([self._button("✕ Cancel", key, "/cancel")])
        return self._markup(rows)

    def _approval_markup(self, key: str, req):
        """Build inline-keyboard controls for an approval request."""
        request_id = getattr(req, "id", "pending")
        is_boolean = getattr(req, "type", "boolean") == "boolean"
        if getattr(req, "enum", None):
            rows = [[self._button(str(v), key, f"approval:{request_id}:{v}")] for v in req.enum]
            if not is_boolean:
                rows.append([self._button("✕ Cancel", key, "/cancel")])
            return self._markup(rows)
        if is_boolean:
            return self._markup([[self._button("Approve", key, f"approval:{request_id}:allow"), self._button("Deny", key, f"approval:{request_id}:deny")]])
        return self._markup([[self._button("✕ Cancel", key, "/cancel")]])

    def _buttons_markup(self, key: str, buttons: list[dict]):
        """Build inline-keyboard markup for a generic button list."""
        return self._markup([[self._button(str(b.get("label") or b.get("text") or b.get("value") or "Option"), key, str(b.get("value") or b.get("text") or b.get("label") or ""))] for b in buttons])

    def _button(self, label: str, key: str, value: str, echo: str | None = None):
        """Create one callback-backed Telegram button and remember its payload."""
        from telegram import InlineKeyboardButton
        token = "bf:" + uuid.uuid4().hex[:16]
        self._callbacks[token] = (key, value, echo)
        return InlineKeyboardButton(label[:64], callback_data=token)

    def _form_echo(self, form: dict, value) -> str | None:
        """Reconstruct a slash-command preview line for a form choice, when useful."""
        if form.get("action_type") != "call_command" or not form.get("name"):
            return None
        parts = ["/" + str(form["name"])]
        parts += [_quote(v) for v in (form.get("collected") or {}).values()]
        parts.append(_quote(value))
        return " ".join(parts)

    def _next_approval(self, key: str):
        """Return the next unresolved approval request for a session."""
        with self._approval_lock:
            return next(req for req in self._pending_approvals.get(key, {}).values() if not getattr(req, "is_resolved", False))

    @staticmethod
    def _parse_approval(text: str) -> bool | None:
        """Parse a Telegram text reply into an approval decision."""
        value = (text or "").strip().lower()
        if value in {"n", "no", "deny", "denied", "false", "0"}:
            return False
        if value in {"y", "yes", "approve", "approved", "true", "1"}:
            return True
        return None

    @staticmethod
    def _markup(rows):
        """Build an inline keyboard when there are rows to show."""
        from telegram import InlineKeyboardMarkup
        return InlineKeyboardMarkup(rows) if rows else None

    def _clear_last_keyboard(self, key: str) -> None:
        """Clear the last inline keyboard shown for a session."""
        self._send(self._clear_last_keyboard_async(key))

    async def _clear_last_keyboard_async(self, key: str):
        """Remove the last inline keyboard from Telegram, if it still exists."""
        entry = self._last_keyboard.pop(key, None)
        if not entry or self.app is None:
            return
        try:
            await self.app.bot.edit_message_reply_markup(chat_id=entry[0], message_id=entry[1], reply_markup=None)
        except Exception:
            pass


def _quote(value) -> str:
    """Quote a form value the same way a slash-command preview would."""
    import json
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    text = str(value)
    return json.dumps(text) if any(ch.isspace() for ch in text) else text
