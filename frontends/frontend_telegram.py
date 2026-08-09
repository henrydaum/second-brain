"""Telegram frontend, on the sandbox contract.

**The loop inverts, and that is nearly the whole migration.** A native frontend
blocks in ``start()`` forever; python-telegram-bot is asyncio-only and wants to
own the process. A box serializes one call at a time, so code that never
returns from ``start`` holds the box and no ``render`` gets in. The guest may
not spawn a thread either — ``threading`` is refused, because the kernel
schedules.

What makes the resolution possible is that a subprocess box serves *every* call
on one thread: ``poll``, ``render`` and ``stop`` all land on the child's single
serving thread. So one event loop is created in ``start`` and driven in slices
from ``poll``, and everything that used to be cross-thread is now ordinary
sequential code:

- ``poll`` runs the loop for ~80 ms, which is python-telegram-bot's updater,
  the stream pump and the typing pulses getting their turn, then drains a plain
  list the update handlers appended to.
- Handlers no longer touch the runtime. They queue; ``poll`` submits. The
  ``run_in_executor`` bounce, the ``run_coroutine_threadsafe`` bridges and the
  typing bracket around a blocking submit are all gone with the threads that
  needed them.
- ``render`` arrives between polls, when the loop is idle, so it can simply
  ``run_until_complete`` a send.

This shape *requires* subprocess isolation — an in-process resident box runs
each call on a fresh worker thread, and a loop bound in ``start`` could not be
driven from ``poll`` there. That is guaranteed rather than requested: this file
lives in the installed tree and imports a foreign library, which is exactly the
condition ``sandbox/isolation.py`` subprocesses on.

``background_submit`` is the other load-bearing declaration. Without it
``sdk.frontend.submit_text`` runs the whole agent turn inline and holds the box,
so nothing could render while the agent was thinking.

**Two things were dropped deliberately.** The 👍 reaction acknowledging a
queued mid-turn message, and the pinned conversation banner. Both hung off
``BaseFrontend`` hooks (``render_queued_ack``, ``render_conversation_banner``)
that the nine-kind render wire does not carry, and one of them needs a render
call whose *return value* matters. A queued message now gets the ordinary
textual ack. The ``telegram_pin_banner`` and ``telegram_banner_messages``
settings went with the banner.

**And one was simplified.** ``agent_prompt_for`` used to pick between rich and
basic formatting guidance from the live API's capability. The bridge carries a
static ``agent_prompt`` declaration, not a per-turn call into the box, and
putting a box round trip on the prompt-building path to recover a paragraph of
wording is a bad trade. The prompt below describes rich Markdown and says it may
degrade.
"""

dependencies_files = ['frontends/helpers/telegram_renderers.py']
dependencies_pip = ['python-telegram-bot']
requests = [
    "frontend.submit", "frontend.pending", "frontend.resolve",
    "session.get", "config.read", "secret.reveal", "command.list",
    "fs.temp", "fs.read", "fs.read_bytes", "fs.list", "parse.modality",
]

import asyncio
import html
import json
import re
import time
import uuid

from guest.bases import BaseFrontend

# Flat: the box is one namespace and the declared dependency's directory is on
# its import path, so the helper is a sibling despite shipping in a subfolder.
from .telegram_renderers import (
    StreamTracker,
    file_bytes,
    method_for,
    prepare_media_actions,
    prepare_photo_bytes,
)

_MAX_FILE_SIZE = 50 * 1024 * 1024
_MAX_MESSAGE_CHARS = 4096

_TABLE_BLOCK = re.compile(
    r"^[ \t]*\|.*\|[ \t]*\n[ \t]*\|(?:\s*:?-{3,}:?\s*\|)+[ \t]*\n"
    r"(?:[ \t]*\|.*\|[ \t]*(?:\n|$))*", re.MULTILINE)
_QUOTE_BLOCK = re.compile(r"^(?:>[ \t]?.*(?:\n|$))+", re.MULTILINE)
_BLOCKS = re.compile(f"(?:{_TABLE_BLOCK.pattern})|(?:{_QUOTE_BLOCK.pattern})",
                     re.MULTILINE)


# ──────────────────────────────────────────────────────────────────────
# Markdown → Telegram HTML. Pure string work; ``sdk`` is threaded through
# only for ``md.align_tables``, which is the guest's copy of the kernel's
# table padding.
# ──────────────────────────────────────────────────────────────────────

def _md_to_tg_html(sdk, text: str) -> str:
    """Convert lightweight markdown-ish output into Telegram-safe HTML."""
    parts, last = [], 0
    for match in re.finditer(r"```(\w*)\n(.*?)```", text or "", re.DOTALL):
        parts.append(_blocks(sdk, text[last:match.start()]))
        code = html.escape(match.group(2).rstrip())
        parts.append(
            f'<pre><code class="language-{html.escape(match.group(1))}">'
            f'{code}</code></pre>' if match.group(1) else f"<pre>{code}</pre>")
        last = match.end()
    return "".join(parts + [_blocks(sdk, (text or "")[last:])])


def _compact_detail_cards(sdk, text: str) -> str:
    """Turn detail cards into fenced code blocks.

    A detail card is a two-column table with an empty second header cell — the
    kernel's ``md.card`` shape. Full-width rendered tables are overkill for a
    title and a few key/value rows; the compact monospace card reads better on
    a phone. Real data tables (non-empty headers) stay markdown and render
    natively.
    """
    def replace(match):
        """One table block, compacted if it is a card."""
        header = [c.strip() for c in
                  match.group(0).split("\n", 1)[0].strip().strip("|").split("|")]
        if len(header) == 2 and header[0] and not header[1]:
            return f"```\n{sdk.md.align_tables(match.group(0).strip())}\n```\n"
        return match.group(0)

    return _TABLE_BLOCK.sub(replace, text or "")


def _blocks(sdk, text: str) -> str:
    """Render markdown tables (aligned <pre>) and > blockquotes; inline the rest.

    Only the non-rich HTML fallback comes through here — the Rich Messages path
    sends raw markdown and Telegram renders it server-side.
    """
    out, last = [], 0
    for match in _BLOCKS.finditer(text or ""):
        out.append(_inline(text[last:match.start()]))
        block = match.group(0)
        if block.lstrip().startswith("|"):
            out.append(
                f"<pre>{html.escape(sdk.md.align_tables(block.strip()))}</pre>")
        else:
            quoted = "\n".join(re.sub(r"^>[ \t]?", "", line)
                               for line in block.strip().split("\n"))
            out.append(f"<blockquote>{_inline(quoted)}</blockquote>")
        last = match.end()
    return "".join(out + [_inline((text or "")[last:])])


def _inline(text: str) -> str:
    """Render inline code spans while preserving the surrounding rich text."""
    out, last = [], 0
    for match in re.finditer(r"`([^`]+)`", text):
        out.append(_bold_italic(text[last:match.start()]))
        out.append(f"<code>{html.escape(match.group(1))}</code>")
        last = match.end()
    return "".join(out + [_bold_italic(text[last:])])


def _bold_italic(text: str) -> str:
    """Translate simple bold and italic markers into Telegram HTML tags."""
    escaped = html.escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    return re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", escaped)


def _chunks(text: str, max_chars: int = _MAX_MESSAGE_CHARS) -> list:
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


def _quote(value) -> str:
    """Quote one argument the way a slash-command preview would.

    The kernel's ``format_command_call`` uses ``shlex.quote``; ``shlex`` is not
    on the SDK's pure list and this is a status banner, so JSON quoting stands
    in. The difference is ``"a b"`` where the kernel writes ``'a b'``.
    """
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"))
    text = str(value)
    return json.dumps(text) if any(ch.isspace() for ch in text) else text


def _command_call(name: str, args) -> str:
    """Render a command invocation for the progress banner."""
    parts = ["/" + str(name or "").strip().lstrip("/")]
    parts += [_quote(value) for value in (args or {}).values()
              if value is not None]
    return " ".join(parts)


def _banner(mark: str, text: str, blurb: str) -> str:
    """The progress banner's inner HTML: the call, then why it was made.

    The blurb sits outside the ``<code>`` span deliberately — it is the model's
    prose about its intent, not part of the invocation, and monospacing it
    would read as though it were something the tool was passed.
    """
    body = f"{mark} <code>{html.escape(text)}</code>"
    return f"{body} <i>{html.escape(blurb)}</i>" if blurb else body


def _parse_approval(text: str):
    """Parse a Telegram text reply into an approval decision, or None."""
    value = (text or "").strip().lower()
    if value in {"/cancel", "n", "no", "deny", "denied", "false", "0"}:
        return False
    if value in {"y", "yes", "approve", "approved", "true", "1"}:
        return True
    return None


class TelegramFrontend(BaseFrontend):
    """Telegram chat frontend backed by the conversation state machine."""

    name = "telegram"
    description = "Telegram chat frontend backed by the conversation state machine."

    # A plain dict: a box cannot hold a dataclass, and the bridge rebuilds a
    # FrontendCapabilities from this. Sizes are written out because
    # declarations are read by ``ast.literal_eval``, which does not do
    # arithmetic — ``50 * 1024 * 1024`` here would read as nothing at all.
    capabilities = {
        "supports_typing": True,
        "supports_buttons": True,
        "supports_message_edit": True,
        "supports_attachments_in": True,
        "supports_attachments_out": True,
        "supports_inline_forms": True,
        "supports_proactive_push": True,
        "supports_rich_text": True,
        "supports_streaming": True,
        "max_message_chars": 4096,
        "max_upload_size": 52_428_800,      # 50 MB
    }

    user_binding = "single"
    default_user_id = 1

    # Submitting must not hold the box: the turn it starts renders back into
    # this same serialized box, and inline it would deadlock against itself.
    background_submit = True
    # Reopening the last conversation may render, so it happens after start()
    # has released the box.
    restore_on_start = True
    # No idle pause: ``poll`` already spends its time inside the event loop,
    # and a sleep on top of that is only latency.
    poll_interval = 0.0
    max_poll_failures = 5
    # Uploading 50 MB over a phone connection is guest execution, not blocked
    # time, so it is charged in full against the call deadline.
    timeout = 300.0
    memory_mb = 768

    config_settings = [
        ("Telegram Bot Token", "secret_telegram_bot_token",
         "Bot token from @BotFather. Required for Telegram frontend.",
         "", {"type": "text"}),
        ("Telegram Allowed User ID", "telegram_allowed_user_id",
         "Your Telegram user ID (integer). Only this user can interact with "
         "the bot. Send /start to @userinfobot to find yours.",
         0, {"type": "text"}),
    ]

    agent_prompt = (
        "## Talking over Telegram\n"
        "This conversation is on Telegram, a mobile chat app. Replies are sent as native "
        "Rich Messages, so standard Markdown displays with full fidelity: headings, **bold**, "
        "*italic*, ~~strikethrough~~, `inline code`, fenced code blocks with language tags, "
        "[links](https://example.com), bulleted and numbered lists, tables, > blockquotes, and "
        "--- dividers. Use whatever structure serves the reply, but it is still a phone screen: "
        "keep replies concise and skimmable. On an older Telegram server the same message falls "
        "back to a simpler renderer where only bold, italic, inline code and code blocks survive, "
        "so do not let meaning depend on a heading or a table alone. Long messages are split "
        "across multiple sends, and file uploads are capped at 50 MB."
    )

    # How long one poll spends inside the event loop. It is the frontend's
    # whole latency budget in both directions: python-telegram-bot only makes
    # progress during a slice, and a render can only land between them.
    _POLL_SLICE = 0.08
    _STREAM_CURSOR = " ▍"

    def __init__(self):
        """Set up the state the transport hangs off. No effects here."""
        self._sdk = None
        self._loop = None
        self._app = None
        self._allowed_user = 0
        # Update handlers append here; ``poll`` drains it. The whole of what
        # replaces the old cross-thread submit.
        self._queue = []
        self._chat_by_session = {}
        self._callbacks = {}
        self._tool_messages = {}
        self._last_keyboard = {}
        # The last approval each session was shown, kept because answering one
        # by typing needs to know whether it is a yes/no or free text. Whether
        # it is still *pending* is asked, never remembered.
        self._approvals = {}
        # One StreamTracker per in-flight streamed reply, keyed by
        # (session_key, stream_id).
        self._streams = {}
        # Rich Messages (Bot API 10.1): None = undetermined, False = confirmed
        # unavailable (old python-telegram-bot, or a pre-10.1 server).
        self._rich = None
        # Turn-lifecycle typing pulses: session_key -> the Event that stops one.
        self._typing_stops = {}

    # ──────────────────────────────────────────────────────────────────
    # Lifecycle.
    # ──────────────────────────────────────────────────────────────────

    def start(self, sdk):
        """Open the transport and return. The kernel drives from here on."""
        # ``secret_*`` reads back as a handle, never plaintext, so this asks
        # for the real thing. Not gated: a plugin reading its own declared
        # setting is not asked, because configuring it *was* the consent. The
        # token still has to reach python-telegram-bot in the clear — a
        # credential inside a foreign library is past the kernel's reach, and
        # the prefix buys keeping it out of /config, config dumps and the
        # ledger rather than out of the library.
        token = str(sdk.secrets.reveal("secret_telegram_bot_token")
                    or "").strip()
        if not token:
            sdk.log("secret_telegram_bot_token not configured; "
                    "Telegram frontend disabled.")
            return False
        try:
            self._allowed_user = int(
                sdk.config.read("telegram_allowed_user_id") or 0)
        except (TypeError, ValueError):
            self._allowed_user = 0

        # Held rather than passed down. A box has exactly one SDK for its whole
        # life — the child builds it once and hands the same object to every
        # call — so a coroutine scheduled by ``render`` and run during a later
        # ``poll`` is using the same handle it would have been given anyway.
        self._sdk = sdk
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._open(token))
        except Exception as exc:
            sdk.log(f"Telegram frontend did not start: {exc}", "error")
            return False
        return True

    async def _open(self, token: str):
        """Build the application, attach handlers, and start long-polling."""
        from telegram.ext import (Application, CallbackQueryHandler,
                                  MessageHandler, filters)

        self._app = (Application.builder().token(token)
                     .concurrent_updates(True).build())
        self._app.add_handler(MessageHandler(
            filters.COMMAND | (filters.TEXT & ~filters.COMMAND),
            self._on_text))
        self._app.add_handler(MessageHandler(
            filters.PHOTO | filters.Document.ALL | filters.VOICE
            | filters.AUDIO, self._on_attachment))
        self._app.add_handler(CallbackQueryHandler(self._on_callback))
        self._app.add_error_handler(self._on_error)

        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling()
        await self._announce()

    async def _announce(self):
        """Register the slash-command menu and greet the allowed user."""
        from telegram import BotCommand

        try:
            listed = self._sdk.commands.list(details=True, visible=True) or []
            await self._app.bot.set_my_commands([
                BotCommand(str(item.get("name") or "")[:32],
                           str(item.get("description") or "")[:256])
                for item in listed if item.get("name")])
        except Exception as exc:
            self._sdk.log(f"Failed to register Telegram commands: {exc}",
                          "warning")
        if not self._allowed_user:
            return
        key = self._default_key()
        self._chat_by_session[key] = self._allowed_user
        try:
            await self._app.bot.send_message(self._allowed_user,
                                             "Second Brain online.")
        except Exception as exc:
            self._sdk.log(f"Telegram greeting failed: {exc}", "warning")

    def stop(self, sdk):
        """Close the transport. Tolerates never having started."""
        for stop in list(self._typing_stops.values()):
            stop.set()
        self._typing_stops.clear()
        if self._loop is None:
            return
        try:
            if self._app is not None:
                self._loop.run_until_complete(self._close())
        except Exception as exc:
            sdk.log(f"Telegram shutdown was untidy: {exc}", "warning")
        finally:
            self._app = None
            try:
                self._loop.close()
            except Exception:
                pass
            self._loop = None

    async def _close(self):
        """Unwind python-telegram-bot in the order it expects, then the loop.

        The pumps and pulses scheduled with ``create_task`` are cancelled
        explicitly. Closing a loop with tasks still pending is legal and prints
        "Task was destroyed but it is pending" for each of them, which on a
        busy shutdown buries whatever the real reason for stopping was.
        """
        if self._app.updater is not None:
            await self._app.updater.stop()
        await self._app.stop()
        await self._app.shutdown()
        pending = [task for task in asyncio.all_tasks(self._loop)
                   if task is not asyncio.current_task()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    # ──────────────────────────────────────────────────────────────────
    # The poll loop: give the event loop its turn, then drain what arrived.
    # ──────────────────────────────────────────────────────────────────

    def poll(self, sdk):
        """Run the transport for a slice and submit whatever it produced."""
        if self._loop is None:
            return False
        self._loop.run_until_complete(asyncio.sleep(self._POLL_SLICE))
        pending, self._queue = self._queue, []
        for item in pending:
            try:
                self._deliver(sdk, item)
            except sdk.Failed as exc:
                sdk.log(f"Telegram could not submit "
                        f"{item.get('kind')}: {exc}", "warning")
        return bool(pending)

    def _deliver(self, sdk, item: dict):
        """Carry one queued update into the state machine."""
        key = item["key"]
        if item["kind"] == "file":
            return self._deliver_file(sdk, item)
        text = item.get("text") or ""
        if item["kind"] == "callback" and item.get("approval"):
            request_id, answer = item["approval"]
            shown = dict(self._approvals.get(key) or {})
            value = self._approval_value(key, answer)
            if sdk.frontend.resolve(key, value, request_id):
                self._approvals.pop(key, None)
                self._acknowledge_approval(sdk, key, shown, value)
                return None
            # It was already answered or timed out. If the session is *still*
            # blocked on an approval it is a different one, and the state
            # machine's own text path is what answers that.
            if (sdk.session.get(key) or {}).get("phase") != "approving_request":
                return None
            text = ("yes" if value is True
                    else "no" if value is False else str(value))
        if self._absorb_approval(sdk, key, text):
            return None
        # In ``approving_request`` the state machine collects the answer itself:
        # text is coerced into ``answer_approval`` rather than reaching the
        # agent, so this submit *is* the answer and saying so is ours to do.
        # ``_absorb_approval`` above declines exactly that phase, which is why
        # the acknowledgement has to be made here as well as there.
        shown = (dict(self._approvals.get(key) or {})
                 if (sdk.session.get(key) or {}).get("phase") == "approving_request"
                 else None)
        outcome = sdk.frontend.submit_text(key, text)
        if shown is not None:
            self._acknowledge_approval(sdk, key, shown, text, submitted=True)
        return outcome

    def _acknowledge_approval(self, sdk, key: str, shown: dict, value,
                              submitted: bool = False) -> None:
        """Say what an answer did to the question it answered.

        **The kernel no longer narrates this.** An approval's outcome crosses as
        the phase leaving ``approving_request`` and as ``ActionResult.data``, not
        as prose on the ``messages`` kind — which is also what the agent's own
        words ride, and which a frontend that draws its own buttons cannot tell
        apart from them. Wording it is each frontend's own business now; this is
        Telegram's, and it deliberately reuses the vocabulary ``_absorb_approval``
        already sends so a tap and a typed reply read the same.

        ``submitted`` means the answer went through ``submit_text`` and may have
        been refused as invalid input, in which case the kernel has already sent
        the error and adding to it would only disagree with it.
        """
        if submitted and (sdk.session.get(key) or {}).get("phase") == "approving_request":
            return
        if shown.get("enum") or (shown.get("type") or "boolean") != "boolean":
            self._deliver_message(self._chat_id(key), f"Answered: {value}.")
            return
        decided = value if isinstance(value, bool) else _parse_approval(str(value))
        if decided is not None:
            self._deliver_message(self._chat_id(key),
                                  "Approved." if decided else "Denied.")

    def _approval_value(self, key: str, answer: str):
        """What a tapped button answers with, in the shape that frame accepts.

        Decided the same way ``_approval_markup`` chose the buttons, because
        the two have to agree: an ``enum`` request's buttons carry its own
        values and must go back **verbatim**, and only the boolean fallback
        spells them "allow"/"deny" — a frame whose lenient parser wants a bool.

        Coercing unconditionally is what made every sandbox Request dialog
        unanswerable by button. Those are typed ``string`` with an enum, so the
        state machine validated ``True`` against ``["allow", "deny"]``, refused
        it, and left the frame up; the plugin waiting on the other side sat
        there until its dialog timed out and was denied. Nothing in the chat
        said so — the person saw a dialog they had already answered, and a
        session that took input and did nothing with it.
        """
        if (self._approvals.get(key) or {}).get("enum"):
            return answer
        return (True if answer == "allow"
                else False if answer == "deny" else answer)

    def _absorb_approval(self, sdk, key: str, text: str) -> bool:
        """Answer a pending approval by typed reply. True if it was consumed.

        Only when the session is *not* in ``approving_request``: that phase is
        the state machine collecting the answer itself, and racing it would
        answer twice. Whether an approval is still pending is asked every time
        rather than remembered, because another frontend can answer it and it
        can time out — and acting on a stale record means swallowing the next
        thing a person types.
        """
        if (sdk.session.get(key) or {}).get("phase") == "approving_request":
            return False
        request_id = sdk.frontend.pending_input(key)
        if not request_id:
            return False
        request = self._approvals.get(key) or {}
        value = text
        if (request.get("type") or "boolean") == "boolean":
            value = _parse_approval(text)
            if value is None:
                self._send_text(key, html.escape(
                    "Error: Approval needs yes or no."))
                return True
        if sdk.frontend.resolve(key, value, str(request_id)):
            self._approvals.pop(key, None)
            # One voice for all three ways in — tapped button, typed reply here,
            # typed reply the state machine coerced. A bare "Received." said the
            # same thing for a grant and a refusal, which is the one distinction
            # anybody reading it back needs.
            self._acknowledge_approval(sdk, key, request, value)
        return True

    def _deliver_file(self, sdk, item: dict):
        """Download one attachment into scratch and hand it over.

        Straight to disk rather than through memory: one wire message holds
        about 11 MB and Telegram allows 50, so bytes that crossed the boundary
        could not carry the larger half of what the transport accepts. The
        kernel allocated the path and takes the file from it, moving it into
        the attachment cache — a watched directory, so the pipeline indexes it
        like any other incoming file.
        """
        suffix = sdk.path.suffix(item["name"])
        temp = sdk.fs.temp(suffix=suffix)
        self._loop.run_until_complete(item["file"].download_to_drive(temp))
        return sdk.frontend.submit_attachment(
            item["key"], temp, extension=suffix.lstrip("."),
            file_name=item["name"], caption=item.get("caption") or "",
            is_photo=bool(item.get("is_photo")), ingest=True)

    # ──────────────────────────────────────────────────────────────────
    # Update handlers. These run inside a poll slice and must not make
    # Requests: they queue, and ``_deliver`` does the talking.
    # ──────────────────────────────────────────────────────────────────

    async def _on_text(self, update, _ctx):
        """One incoming text message."""
        if not self._allowed(update) or not update.message:
            return
        text = re.sub(r"^/([A-Za-z0-9_]+)@[^\s]+", r"/\1",
                      (update.message.text or "").strip())
        if not text:
            return
        # Queued before the typing action, not after: the action is a network
        # round trip, and awaiting it first would put a few hundred
        # milliseconds between the person pressing send and the turn starting.
        self._queue.append({"kind": "text", "key": self._key_for(update),
                            "text": text})
        await self._show_typing(update.message.chat)

    async def _on_attachment(self, update, _ctx):
        """One incoming photo, document, voice note or audio file."""
        if not self._allowed(update) or not update.message:
            return
        message = update.message
        handle, file_name, size, is_photo = None, "attachment", 0, False
        if message.photo:
            handle, file_name, is_photo = (await message.photo[-1].get_file(),
                                           "photo.jpg", True)
        elif message.document:
            handle = await message.document.get_file()
            file_name = message.document.file_name or "document"
            size = message.document.file_size or 0
        elif message.voice:
            handle = await message.voice.get_file()
            file_name, size = "voice.ogg", message.voice.file_size or 0
        elif message.audio:
            handle = await message.audio.get_file()
            file_name = message.audio.file_name or "audio.mp3"
            size = message.audio.file_size or 0
        if handle is None:
            return
        if size > _MAX_FILE_SIZE:
            await message.reply_text("File too large (50 MB limit).")
            return
        self._queue.append({
            "kind": "file", "key": self._key_for(update), "file": handle,
            "name": file_name, "caption": message.caption or "",
            "is_photo": is_photo,
        })
        await self._show_typing(message.chat)

    async def _on_callback(self, update, _ctx):
        """One inline-button press."""
        query = update.callback_query
        if not query:
            return
        await query.answer()
        entry = self._callbacks.pop(query.data or "", None)
        if entry is None:
            return
        key, value = entry
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except Exception:
            pass
        self._last_keyboard.pop(key, None)
        item = {"kind": "callback", "key": key, "text": value}
        if value.startswith("approval:"):
            request_id, answer = value.split(":", 2)[1:]
            item["approval"] = (request_id, answer)
        self._queue.append(item)

    async def _on_error(self, _update, ctx):
        """Swallow transient network errors; log the rest once."""
        from telegram.error import NetworkError, TimedOut

        error = getattr(ctx, "error", None)
        if isinstance(error, (NetworkError, TimedOut)):
            # The updater retries on its own, and the child's log handler
            # collapses the repeats it makes on the way.
            return
        self._sdk.log(f"Telegram handler error: {error}", "error")

    async def _show_typing(self, chat):
        """A single typing action, so the gap before the turn starts is not dead."""
        try:
            from telegram.constants import ChatAction

            await chat.send_action(ChatAction.TYPING)
        except Exception:
            pass

    def _allowed(self, update) -> bool:
        """Whether an update came from the one user this bot answers to."""
        if not self._allowed_user:
            return True
        user = getattr(update, "effective_user", None)
        return bool(user and user.id == self._allowed_user)

    # ──────────────────────────────────────────────────────────────────
    # Sessions.
    # ──────────────────────────────────────────────────────────────────

    def session_key(self, sdk, ctx):
        """Name the session a transport context belongs to.

        The kernel calls this with plain data or with nothing — a live PTB
        ``Update`` cannot cross a boundary. ``_key_for`` is the internal
        counterpart that does hold one.
        """
        if isinstance(ctx, dict):
            user = ctx.get("user_id") or 0
            chat = ctx.get("chat_id") or user
            if user or chat:
                return self._remember(user, chat, ctx.get("thread_id") or 0)
        return self._default_key()

    def _key_for(self, update) -> str:
        """The per-user, per-chat, per-thread key for one live update."""
        user = getattr(getattr(update, "effective_user", None), "id", 0) or 0
        chat = getattr(getattr(update, "effective_chat", None), "id",
                       None) or user
        thread = getattr(getattr(update, "effective_message", None),
                         "message_thread_id", None) or 0
        return self._remember(user, chat, thread)

    def _remember(self, user, chat, thread) -> str:
        """Build a session key and memoize the chat it maps to."""
        key = f"telegram:{user}:{chat}:{thread}"
        if chat:
            self._chat_by_session[key] = int(chat)
        return key

    def _default_key(self) -> str:
        """The session the allowed user talks to the bot in.

        Answered without an update because the kernel asks before any message
        has arrived — restoring the last conversation at start, for one.
        """
        return f"telegram:{self._allowed_user}:{self._allowed_user}:0"

    def _chat_id(self, session_key: str):
        """Recover the Telegram chat behind a session key."""
        if session_key in self._chat_by_session:
            return self._chat_by_session[session_key]
        try:
            chat = int(str(session_key).split(":")[2])
        except (IndexError, ValueError):
            return None
        self._chat_by_session[session_key] = chat
        return chat

    # ──────────────────────────────────────────────────────────────────
    # Rendering. Called between polls, so the loop is idle and a send can
    # simply be awaited.
    # ──────────────────────────────────────────────────────────────────

    def render(self, sdk, session_key: str, kind: str, payload):
        """Show one thing to the person behind a session."""
        if self._app is None or self._loop is None:
            return
        if kind == "messages":
            self._clear_keyboard(session_key)
            chat = self._chat_id(session_key)
            for message in payload or []:
                if message:
                    self._deliver_message(chat, message)
        elif kind == "attachments":
            self._clear_keyboard(session_key)
            self._await(self._send_media(self._chat_id(session_key),
                                         list(payload or [])))
        elif kind == "form_field":
            # The turn is live but waiting on the user — typing would mislead.
            # It resumes on the next turn-started event.
            self._typing(session_key, False)
            form = payload or {}
            self._send_text(session_key, self._prompt(form),
                            markup=self._enum_markup(session_key, form))
        elif kind == "approval":
            request = payload or {}
            self._approvals[session_key] = request
            self._typing(session_key, False)
            title = request.get("title") or "Approval requested"
            body = _md_to_tg_html(
                sdk, f"{title}\n\n{request.get('body') or ''}".strip())
            self._send_text(session_key, body,
                            markup=self._approval_markup(session_key, request))
        elif kind == "buttons":
            self._send_text(session_key, "Choose:",
                            markup=self._buttons_markup(session_key,
                                                        list(payload or [])))
        elif kind == "error":
            self._clear_keyboard(session_key)
            message = (payload.get("message") if isinstance(payload, dict)
                       else payload)
            self._send_text(session_key, html.escape(f"Error: {message}"))
        elif kind == "typing":
            self._typing(session_key, bool(payload))
        elif kind == "tool_status":
            self._tool_status(session_key, payload or {})
        elif kind == "stream_delta":
            self._stream_delta(session_key, payload or {})

    def _await(self, coro):
        """Run one coroutine to completion. Rendering must never raise.

        A frontend that cannot show something is not the turn's problem — the
        kernel carries on either way — so a failed send is logged and dropped,
        which is the same policy the host applies around this call anyway.
        """
        try:
            return self._loop.run_until_complete(coro)
        except Exception as exc:
            self._sdk.log(f"Telegram send failed: {exc}", "warning")
            return None

    def _schedule(self, coro):
        """Start a coroutine that outlives this call.

        It makes progress during subsequent poll slices. Used for the two
        things that are genuinely long-running — a streamed reply and a typing
        pulse — and nothing else.
        """
        return self._loop.create_task(coro)

    # ── messages ──────────────────────────────────────────────────────

    def _deliver_message(self, chat_id, text: str) -> None:
        """Queue one outgoing chat message."""
        if chat_id:
            self._await(self._deliver_message_async(chat_id, text))

    async def _deliver_message_async(self, chat_id, text: str) -> None:
        """Deliver one message: rich Markdown first, HTML pipeline fallback."""
        chunks = _chunks(_compact_detail_cards(self._sdk, text),
                         self._max_chars())
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
            except Exception as exc:
                if self._rich_refused(exc):
                    self._rich = False
                    self._sdk.log("Rich Messages unavailable; using HTML "
                                  "rendering from now on.")
                else:
                    self._sdk.log(f"sendRichMessage failed ({exc}); HTML "
                                  f"fallback for this message.", "warning")
        for chunk in chunks[sent:]:
            await self._send_text_async(chat_id,
                                        _md_to_tg_html(self._sdk, chunk), True)

    def _max_chars(self) -> int:
        """Telegram's per-message character cap, as declared."""
        return int(self.capabilities.get("max_message_chars")
                   or _MAX_MESSAGE_CHARS)

    # ── Rich Messages (Bot API 10.1) ──────────────────────────────────
    # InputRichMessage accepts raw Markdown, parsed server-side into
    # headings/tables/lists/code — no local conversion needed.
    # python-telegram-bot has no typed support yet (python-telegram-bot#5261),
    # so calls go through the typed method when present and PTB's raw request
    # layer otherwise.

    def _rich_capable(self) -> bool:
        """Whether Rich Message endpoints look reachable.

        Optimistic before the transport is up, and downgraded to False the
        first time the API refuses.
        """
        if self._rich is None:
            bot = getattr(self._app, "bot", None)
            if bot is None:
                return True
            self._rich = bool(hasattr(bot, "send_rich_message")
                              or hasattr(bot, "do_api_request"))
        return self._rich

    async def _rich_request(self, endpoint: str, payload: dict) -> None:
        """Call a Rich Message endpoint (typed PTB method or raw layer)."""
        bot = self._app.bot
        snake = re.sub(r"(?<!^)(?=[A-Z])", "_", endpoint).lower()
        method = getattr(bot, snake, None)
        if method is not None:
            import telegram

            rich = getattr(telegram, "InputRichMessage", None)
            if rich is not None:
                payload = {**payload,
                           "rich_message": rich(**payload["rich_message"])}
            await method(**payload)
            return
        await bot.do_api_request(endpoint, api_kwargs=payload)

    @staticmethod
    def _rich_refused(exc) -> bool:
        """True when the error means Rich Messages don't exist here at all."""
        text = str(exc).lower()
        return "not found" in text or "unknown method" in text

    # ── streaming ─────────────────────────────────────────────────────

    def _stream_delta(self, session_key: str, payload: dict) -> None:
        """Feed streamed agent text into the per-stream tracker."""
        stream_key = (session_key, payload.get("stream_id") or "")
        if payload.get("done"):
            tracker = self._streams.get(stream_key)
            if tracker:
                tracker.finish(payload.get("final_text"),
                               bool(payload.get("aborted")))
            return
        delta = payload.get("delta") or ""
        if not delta:
            return
        tracker = self._streams.get(stream_key)
        if tracker is not None:
            tracker.feed(delta)
            return
        chat_id = self._chat_id(session_key)
        if not chat_id:
            return
        tracker = StreamTracker(max_chars=self._max_chars() - 96)
        self._streams[stream_key] = tracker
        tracker.feed(delta)
        self._schedule(self._stream_pump(stream_key, chat_id, tracker))

    @staticmethod
    def _draft_id_for(stream_id: str) -> int:
        """Derive a stable non-zero draft id from the kernel's stream id."""
        try:
            return (int(stream_id.rpartition("_")[2], 16) & 0x7FFFFFFF) or 1
        except ValueError:
            return (abs(hash(stream_id)) & 0x7FFFFFFF) or 1

    async def _stream_pump(self, stream_key, chat_id, tracker):
        """Own one streamed reply, preferring native draft streaming.

        Mode ladder, downgrading in place when a call is refused:

        1. ``rich``  — ``sendRichMessageDraft`` (Bot API 10.1): partial
           Markdown streams with live rich formatting.
        2. ``draft`` — ``sendMessageDraft`` (9.3+): plain-text native typing
           animation.
        3. ``edit``  — legacy placeholder message edited on a throttle.

        Drafts are ephemeral 30s previews in private chats, so the reply is
        still delivered by ``_finalize_stream`` as a real message (the host
        suppressed the whole-message copy, so this pump IS the delivery path).
        Flood control: ``RetryAfter`` backs off and keeps buffering; a hard
        failure stops rendering but the final is still delivered.
        """
        from telegram.error import BadRequest, RetryAfter

        has_plain_draft = getattr(self._app.bot, "send_message_draft",
                                  None) is not None
        mode = ("rich" if self._rich_capable()
                else "draft" if has_plain_draft else "edit")
        draft_id = self._draft_id_for(stream_key[1])
        if mode != "edit":
            # Drafts are a dedicated streaming channel — much tighter cadence
            # than message edits without flirting with flood limits.
            tracker.edit_interval, tracker.burst_chars = 0.35, 64
        message_id = None
        next_allowed = 0.0
        broken = False

        def downgrade(reason):
            """Step down the ladder after a refusal."""
            nonlocal mode
            if mode == "rich":
                if self._rich_refused(reason):
                    self._rich = False
                mode = "draft" if has_plain_draft else "edit"
            else:
                mode = "edit"
            if mode == "edit":
                tracker.edit_interval, tracker.burst_chars = 1.75, 300
            self._sdk.log(f"Telegram stream downgraded to '{mode}' "
                          f"({reason}).")

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
                                    await self._rich_request(
                                        "sendRichMessageDraft", {
                                            "chat_id": chat_id,
                                            "draft_id": draft_id,
                                            "rich_message": {
                                                "markdown": current},
                                        })
                                else:
                                    await self._app.bot.send_message_draft(
                                        chat_id=chat_id, draft_id=draft_id,
                                        text=current)
                                tracker.mark_rendered(current, now)
                        else:
                            for head in finals:
                                if message_id is None:
                                    await self._app.bot.send_message(
                                        chat_id, head,
                                        disable_notification=True)
                                else:
                                    await self._app.bot.edit_message_text(
                                        head, chat_id=chat_id,
                                        message_id=message_id)
                                    # Head finalized; the tail gets a fresh
                                    # message.
                                    message_id = None
                            if current is not None:
                                if message_id is None:
                                    sent = await self._app.bot.send_message(
                                        chat_id, current + self._STREAM_CURSOR,
                                        disable_notification=True)
                                    message_id = sent.message_id
                                else:
                                    await self._app.bot.edit_message_text(
                                        current + self._STREAM_CURSOR,
                                        chat_id=chat_id, message_id=message_id)
                                tracker.mark_rendered(current, now)
                    except RetryAfter as exc:
                        next_allowed = time.time() + float(
                            getattr(exc, "retry_after", 3) or 3) + 0.5
                    except BadRequest as exc:
                        if mode in {"rich", "draft"}:
                            downgrade(exc)
                            continue
                        if current is not None:
                            # "message is not modified"
                            tracker.mark_rendered(current, now)
                    except Exception as exc:
                        if mode == "rich" and self._rich_refused(exc):
                            downgrade(exc)
                            continue
                        self._sdk.log(
                            f"Telegram stream render failed; deferring to "
                            f"final delivery: {exc}", "warning")
                        broken = True
                await asyncio.sleep(0.12 if mode != "edit" else 0.3)
            await self._finalize_stream(chat_id, message_id, tracker, aborted,
                                        final_text)
        except Exception as exc:
            self._sdk.log(f"Telegram stream pump crashed: {exc}", "error")
        finally:
            self._streams.pop(stream_key, None)

    async def _finalize_stream(self, chat_id, message_id, tracker, aborted,
                               final_text):
        """Bring the streamed message(s) to their final state."""
        remainder = tracker.remainder()
        if aborted:
            # Whatever follows (a compaction retry answer, "Cancelled.")
            # arrives as a normal whole message; just drop the cursor or the
            # empty placeholder. Draft modes leave nothing to clean up — the
            # ephemeral draft expires on its own.
            if message_id is not None:
                try:
                    if remainder:
                        await self._app.bot.edit_message_text(
                            remainder, chat_id=chat_id, message_id=message_id)
                    else:
                        await self._app.bot.delete_message(chat_id, message_id)
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
            try:
                await self._app.bot.edit_message_text(
                    remainder or final_text or "", chat_id=chat_id,
                    message_id=message_id)
            except Exception as exc:
                self._sdk.log(f"Telegram stream tail finalize failed: {exc}",
                              "warning")
            return
        # Common case: re-render the whole reply as HTML into the streamed
        # message, spilling extra chunks into fresh messages.
        chunks = _chunks(_md_to_tg_html(self._sdk, final_text or remainder),
                         self._max_chars()) or [""]
        first, rest = chunks[0], chunks[1:]
        delivered = False
        if first:
            for text, mode in ((first, "HTML"), (html.unescape(first), None)):
                try:
                    await self._app.bot.edit_message_text(
                        text, chat_id=chat_id, message_id=message_id,
                        parse_mode=mode)
                    delivered = True
                    break
                except Exception as exc:
                    if "not modified" in str(exc).lower():
                        delivered = True
                        break
            if not delivered:
                # Streamed message unusable — send everything fresh.
                rest = [first, *rest]
        for chunk in rest:
            try:
                await self._app.bot.send_message(chat_id, chunk,
                                                 parse_mode="HTML")
            except Exception:
                try:
                    await self._app.bot.send_message(chat_id,
                                                     html.unescape(chunk))
                except Exception as exc:
                    self._sdk.log(f"Telegram stream final chunk send failed: "
                                  f"{exc}", "warning")

    # ── typing ────────────────────────────────────────────────────────

    def _typing(self, session_key: str, on: bool) -> None:
        """Persistent typing pulse tracking the agent-turn lifecycle.

        Driven by turn-started/completed through the host, so the indicator
        stays on for the whole logical turn — including while a subagent
        barrier holds it open — and drops the moment the turn truly ends.
        Idempotent per session.
        """
        if self._loop is None or self._app is None:
            return
        if not on:
            stop = self._typing_stops.pop(session_key, None)
            if stop is not None:
                stop.set()
            return
        if session_key in self._typing_stops:
            return
        chat_id = self._chat_id(session_key)
        if not chat_id:
            return
        stop = asyncio.Event()
        self._typing_stops[session_key] = stop
        self._schedule(self._typing_pulse(session_key, chat_id, stop))

    async def _typing_pulse(self, session_key: str, chat_id, stop) -> None:
        """Refresh the typing action every 4s (Telegram expires it at ~5s)."""
        from telegram.constants import ChatAction

        try:
            while not stop.is_set():
                try:
                    await self._app.bot.send_chat_action(chat_id,
                                                         ChatAction.TYPING)
                    await asyncio.wait_for(stop.wait(), 4)
                except asyncio.TimeoutError:
                    pass
                except Exception:
                    return
        finally:
            if self._typing_stops.get(session_key) is stop:
                self._typing_stops.pop(session_key, None)

    # ── tool status ───────────────────────────────────────────────────

    def _tool_status(self, session_key: str, payload: dict) -> None:
        """Keep Telegram's single progress banner in sync with tool events."""
        chat_id = self._chat_id(session_key)
        if not chat_id:
            return
        key = f"{session_key}:{payload.get('call_id')}"
        name = (payload.get("tool_name") or payload.get("command_name")
                or "call")
        text = (_command_call(name, payload.get("args"))
                if payload.get("kind") == "command" else name)
        # The tool's declared `narration`, already collapsed and capped by the
        # kernel so both events carry the identical string — the banner is
        # edited in place, and a blurb that changed between started and
        # finished would rewrite the line under the reader.
        blurb = payload.get("narration") or ""
        status = payload.get("status")
        if status == "started":
            self._await(self._send_tool_started(chat_id, key, name, text, blurb))
        elif status == "progressed":
            self._await(self._progress_tool_message(chat_id, key, name, text, blurb))
        else:
            self._await(self._finish_tool_message(
                key, chat_id, name, text, bool(payload.get("ok")),
                payload.get("error"), blurb))

    async def _send_tool_started(self, chat_id, key, name, text, blurb=""):
        """Create the hourglass status message for a new tool or command call."""
        sent = await self._app.bot.send_message(
            chat_id, _banner("⋯", text, blurb),
            parse_mode="HTML", disable_notification=True)
        self._tool_messages[key] = (chat_id, sent.message_id, name, text, blurb)

    async def _progress_tool_message(self, chat_id, key, name, text, blurb=""):
        """Update the existing banner without sending a new message."""
        entry = self._tool_messages.get(key)
        if not entry:
            return await self._send_tool_started(chat_id, key, name, text, blurb)
        self._tool_messages[key] = (entry[0], entry[1], name, text, blurb)
        try:
            await self._app.bot.edit_message_text(
                _banner("⋯", text, blurb), chat_id=entry[0],
                message_id=entry[1], parse_mode="HTML")
        except Exception:
            pass
        return None

    async def _finish_tool_message(self, key, chat_id, name, text, ok, error, blurb=""):
        """Finalize the banner with success or failure text."""
        entry = self._tool_messages.pop(key, None)
        display = entry[3] if entry else text
        # Prefer what the started banner actually showed, so the line the
        # reader has been looking at is the one that gets a tick.
        shown = entry[4] if entry and len(entry) > 4 else blurb
        body = _banner("✓" if ok else "✕", display, shown)
        if error and not ok:
            body += f" ({html.escape(str(error))})"
        if entry:
            try:
                await self._app.bot.edit_message_text(
                    body, chat_id=entry[0], message_id=entry[1],
                    parse_mode="HTML")
                return
            except Exception:
                pass
        await self._app.bot.send_message(chat_id, body, parse_mode="HTML",
                                         disable_notification=True)

    # ── text and media sends ──────────────────────────────────────────

    def _send_text(self, session_key: str, text: str, use_html: bool = True,
                   markup=None) -> None:
        """Send a text payload to the chat behind a session key."""
        chat_id = self._chat_id(session_key)
        if chat_id:
            self._await(self._send_text_async(chat_id, text, use_html, markup,
                                              session_key))

    async def _send_text_async(self, chat_id, text: str, use_html: bool,
                               markup=None, session_key: str = ""):
        """Send one text payload, chunking and clearing old keyboards."""
        session_key = session_key or next(
            (k for k, v in self._chat_by_session.items() if v == chat_id), "")
        if session_key:
            await self._clear_keyboard_async(session_key)
        for chunk in _chunks(text, self._max_chars()):
            try:
                sent = await self._app.bot.send_message(
                    chat_id, chunk, parse_mode="HTML" if use_html else None,
                    reply_markup=markup)
            except Exception:
                sent = await self._app.bot.send_message(
                    chat_id, html.unescape(chunk), reply_markup=markup)
            if session_key and markup:
                self._last_keyboard[session_key] = (chat_id, sent.message_id)
            markup = None

    async def _send_media(self, chat_id, paths):
        """Send a batch of files using the best available media method."""
        if not chat_id:
            return
        from telegram import (InputMediaAudio, InputMediaDocument,
                              InputMediaPhoto, InputMediaVideo)

        builders = {
            "photo": lambda p: InputMediaPhoto(
                prepare_photo_bytes(self._sdk, p)),
            "video": lambda p: InputMediaVideo(file_bytes(self._sdk, p)),
            "audio": lambda p: InputMediaAudio(
                file_bytes(self._sdk, p), title=self._sdk.path.stem(p)),
            "document": lambda p: InputMediaDocument(
                file_bytes(self._sdk, p), filename=self._sdk.path.name(p)),
        }

        async def one(path, method):
            """Send one file with the API method chosen for it."""
            if method == "photo":
                await self._app.bot.send_photo(
                    chat_id, photo=prepare_photo_bytes(self._sdk, path))
            elif method == "video":
                await self._app.bot.send_video(
                    chat_id, video=file_bytes(self._sdk, path))
            elif method == "audio":
                await self._app.bot.send_audio(
                    chat_id, audio=file_bytes(self._sdk, path),
                    title=self._sdk.path.stem(path))
            else:
                await self._app.bot.send_document(
                    chat_id, document=file_bytes(self._sdk, path),
                    filename=self._sdk.path.name(path))

        for action in prepare_media_actions(
                self._sdk, paths,
                int(self.capabilities.get("max_upload_size")
                    or _MAX_FILE_SIZE)):
            try:
                if action.method == "media_group":
                    media = [builders[method_for(self._sdk, path,
                                                 action.group_type)](path)
                             for path in action.files]
                    await self._app.bot.send_media_group(chat_id, media)
                elif action.method == "text":
                    await self._app.bot.send_message(chat_id,
                                                     action.text_content,
                                                     parse_mode="HTML")
                else:
                    await one(action.files[0], action.method)
            except Exception as exc:
                self._sdk.log(f"Failed to send Telegram attachment: {exc}",
                              "error")
                await self._app.bot.send_message(
                    chat_id, f"Failed to send attachment: {exc}")

    # ── keyboards ─────────────────────────────────────────────────────

    def _prompt(self, form: dict) -> str:
        """Build the visible Telegram prompt for a form field.

        Form prompts can carry markdown tables (the /packages overview, say),
        and inline keyboards cannot ride on Rich Messages — so this always
        goes through the HTML converter, which aligns tables into <pre>.
        """
        field = form.get("field") or {}
        display = form.get("display") or {}
        prompt = (display.get("prompt") or field.get("prompt")
                  or field.get("name") or "Input required")
        bits = [_md_to_tg_html(self._sdk, str(prompt))]
        assist = display.get("assist")
        if assist:
            bits.append(f"<i>{html.escape(str(assist))}</i>")
        return "\n".join(bits)

    def _enum_markup(self, key: str, form: dict):
        """Build inline-keyboard choices for an enum-backed form field."""
        field = form.get("field") or {}
        display = form.get("display") or {}
        choices = display.get("choices") or [
            {"value": value, "label": str(value)}
            for value in (field.get("enum") or [])]
        try:
            cols = max(1, int(field.get("columns") or 1))
        except (TypeError, ValueError):
            cols = 1
        buttons = [self._button(str(c.get("label") or c.get("value")), key,
                                str(c.get("value"))) for c in choices]
        rows = [buttons[i:i + cols] for i in range(0, len(buttons), cols)]
        if display.get("allow_back"):
            rows.append([self._button("⟵ Back", key, "/back")])
        if display.get("allow_skip", field.get("required") is False):
            rows.append([self._button("⟶ Skip", key, "/skip")])
        if display.get("allow_cancel", True):
            rows.append([self._button("✕ Cancel", key, "/cancel")])
        return self._markup(rows)

    def _approval_markup(self, key: str, request: dict):
        """Build inline-keyboard controls for an approval request.

        ``enum`` and ``enum_labels`` pair by index: the value is what the
        callback answers with, the label is the only part meant to be read.
        Rendering the value put "allow" and "always:api.search.brave.com" on
        the buttons — the internal spelling, which is deliberately written for
        a ledger row months later rather than for a person mid-decision.
        """
        request_id = request.get("id") or "pending"
        is_boolean = (request.get("type") or "boolean") == "boolean"
        if request.get("enum"):
            values = request["enum"]
            labels = request.get("enum_labels") or []
            rows = [[self._button(
                        str(labels[index]) if index < len(labels) else str(value),
                        key, f"approval:{request_id}:{value}")]
                    for index, value in enumerate(values)]
            if not is_boolean:
                rows.append([self._button("✕ Cancel", key, "/cancel")])
            return self._markup(rows)
        if is_boolean:
            return self._markup([[
                self._button("Approve", key, f"approval:{request_id}:allow"),
                self._button("Deny", key, f"approval:{request_id}:deny")]])
        return self._markup([[self._button("✕ Cancel", key, "/cancel")]])

    def _buttons_markup(self, key: str, buttons):
        """Build inline-keyboard markup for a generic button list."""
        return self._markup([[self._button(
            str(b.get("label") or b.get("text") or b.get("value") or "Option"),
            key,
            str(b.get("value") or b.get("text") or b.get("label") or ""))]
            for b in buttons])

    def _button(self, label: str, key: str, value: str):
        """Create one callback-backed button and remember its payload."""
        from telegram import InlineKeyboardButton

        token = "bf:" + uuid.uuid4().hex[:16]
        self._callbacks[token] = (key, value)
        return InlineKeyboardButton(label[:64], callback_data=token)

    @staticmethod
    def _markup(rows):
        """Build an inline keyboard when there are rows to show."""
        from telegram import InlineKeyboardMarkup

        return InlineKeyboardMarkup(rows) if rows else None

    def _clear_keyboard(self, key: str) -> None:
        """Clear the last inline keyboard shown for a session."""
        self._await(self._clear_keyboard_async(key))

    async def _clear_keyboard_async(self, key: str):
        """Remove the last inline keyboard from Telegram, if it still exists."""
        entry = self._last_keyboard.pop(key, None)
        if not entry or self._app is None:
            return
        try:
            await self._app.bot.edit_message_reply_markup(
                chat_id=entry[0], message_id=entry[1], reply_markup=None)
        except Exception:
            pass
