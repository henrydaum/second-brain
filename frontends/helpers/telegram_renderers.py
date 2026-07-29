"""Telegram rendering helpers — pure logic plus the file reads it needs.

Split out of the frontend for two reasons that both still hold under the
sandbox. Telegram's media rules are fiddly enough (photo size caps, aspect
ratios, media-group batching, Google proxy files) to be worth testing on their
own, and ``StreamTracker`` is the throttle/rollover logic behind streamed
replies, which is exactly the kind of thing that should be checkable without a
bot token.

**Everything that touches disk takes ``sdk``.** These functions used to accept
a ``Path`` and read it; a path is now just a string the kernel understands, and
the bytes come back through ``fs.read_bytes``. The one exception is Pillow,
which does its own decoding — but it decodes a ``BytesIO`` we already hold, not
a file it opens, so the boundary is unbroken even though the library is
foreign.

``StreamTracker`` lost its lock. It had one because the agent thread fed it
while the Telegram event loop drained it; inside a box those are the same
thread — ``poll``, ``render`` and the event-loop slices all run on the child's
single serving thread — so the lock guarded nothing and cost a little.
"""

dependencies_files = []
dependencies_pip = ['Pillow']

import html
import io
import json
from dataclasses import dataclass, field

from PIL import Image

PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
AUDIO_EXTENSIONS = {".mp3", ".ogg", ".wav", ".flac", ".m4a", ".aac"}
UNSUPPORTED_IMAGE_EXTENSIONS = {".heic", ".heif", ".tiff", ".tif", ".svg"}
_GOOGLE_LINK_MAP = {
    ".gdoc": "https://docs.google.com/document/d/{doc_id}",
    ".gsheet": "https://docs.google.com/spreadsheets/d/{doc_id}",
    ".gslides": "https://docs.google.com/presentation/d/{doc_id}",
    ".gdraw": "https://docs.google.com/drawings/d/{doc_id}",
    ".gform": "https://docs.google.com/forms/d/{doc_id}",
}
_INLINE_TEXT_MAX = 3000
_MEDIA_GROUP_MAX = 10
_PHOTO_MAX_SIZE = 10 * 1024 * 1024
_PHOTO_MAX_DIMENSION_SUM = 10_000
_PHOTO_MAX_RATIO = 20

# One ``fs.read_bytes`` answer has to fit in one wire message, so a file larger
# than that is read in windows and joined. Comfortably under the kernel's cap
# rather than exactly at it: the ceiling is derived from the protocol and may
# move, and being wrong in this direction only costs an extra round trip.
_READ_WINDOW = 4 * 1024 * 1024


@dataclass
class SendAction:
    """One Telegram send: which API method, and what to feed it."""
    method: str
    files: list = field(default_factory=list)
    group_type: str = ""
    text_content: str = ""


def stat(sdk, path):
    """Size in bytes, or None when the path is not a readable file.

    ``fs.list`` pointed at a file answers for that file alone, which is the
    documented way to ask a stat-shaped question without building a glob out
    of a filename.
    """
    try:
        entries = sdk.fs.list(path, details=True)
    except sdk.Failed:
        return None
    entry = (entries or [None])[0]
    if not isinstance(entry, dict) or entry.get("is_dir"):
        return None
    return entry.get("size")


def read_all_bytes(sdk, path) -> bytes:
    """Every byte of a file, in wire-sized windows.

    A 50 MB video cannot come back in one answer — one message caps around
    11 MB — so this walks it. The loop ends on a short read rather than on a
    size fetched up front, which is one fewer Request and stays correct if the
    file is still being written.
    """
    chunks, offset = [], 0
    while True:
        chunk = sdk.fs.read_bytes(path, offset=offset, length=_READ_WINDOW)
        if not chunk:
            break
        chunks.append(chunk)
        offset += len(chunk)
        if len(chunk) < _READ_WINDOW:
            break
    return b"".join(chunks)


def file_bytes(sdk, path) -> io.BytesIO:
    """A file as a named buffer, which is what python-telegram-bot uploads."""
    buf = io.BytesIO(read_all_bytes(sdk, path))
    buf.name = sdk.path.name(path)
    return buf


def prepare_photo_bytes(sdk, path) -> io.BytesIO:
    """A buffer Telegram will accept as a *photo*, resizing if it must.

    Telegram refuses photos over 10 MB, over 10000 total pixels of width plus
    height, or more than 20:1 in either direction. A file failing any of those
    is re-encoded as JPEG at progressively smaller scales until it fits.
    """
    data = read_all_bytes(sdk, path)
    img = Image.open(io.BytesIO(data))
    w, h = img.size
    ratio = max(w, h) / max(1, min(w, h))
    if (len(data) <= _PHOTO_MAX_SIZE and w + h <= _PHOTO_MAX_DIMENSION_SUM
            and ratio <= _PHOTO_MAX_RATIO):
        buf = io.BytesIO(data)
        buf.name = sdk.path.name(path)
        return buf
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    dimension_scale = min(1, _PHOTO_MAX_DIMENSION_SUM / (w + h))
    for scale in [dimension_scale, dimension_scale * 0.75,
                  dimension_scale * 0.5, dimension_scale * 0.35,
                  dimension_scale * 0.25]:
        if scale <= 0:
            continue
        resized = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        resized.save(buf, format="JPEG", quality=85)
        rw, rh = resized.size
        resized_ratio = max(rw, rh) / max(1, min(rw, rh))
        if (buf.tell() <= _PHOTO_MAX_SIZE
                and rw + rh <= _PHOTO_MAX_DIMENSION_SUM
                and resized_ratio <= _PHOTO_MAX_RATIO):
            buf.seek(0)
            buf.name = sdk.path.stem(path) + ".jpg"
            return buf
    raise ValueError(
        f"{sdk.path.name(path)} could not be resized within Telegram's "
        f"photo limits")


def _google_link(sdk, path):
    """The web URL behind a Google Drive proxy file, or None."""
    template = _GOOGLE_LINK_MAP.get(sdk.path.suffix(path).lower())
    if not template:
        return None
    try:
        data = json.loads(sdk.fs.read(path))
        doc_id = (data.get("doc_id")
                  or data.get("url", "").split("/d/")[-1].split("/")[0])
        return template.format(doc_id=doc_id) if doc_id else None
    except (sdk.Failed, ValueError, AttributeError) as exc:
        sdk.log(f"Could not read Google proxy file "
                f"{sdk.path.name(path)}: {exc}", "warning")
        return None


def _classify(sdk, path, size) -> str:
    """Which Telegram send method a file wants."""
    ext = sdk.path.suffix(path).lower()
    if ext in _GOOGLE_LINK_MAP:
        return "google_link"
    if ext in PHOTO_EXTENSIONS:
        return "photo"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    if ext in UNSUPPORTED_IMAGE_EXTENSIONS:
        return "document"
    try:
        modality = sdk.parse.modality(ext)
    except sdk.Failed:
        return "document"
    return ("text" if modality == "text" and size <= _INLINE_TEXT_MAX
            else "document")


def prepare_media_actions(sdk, paths, max_file_size: int = 50 * 1024 * 1024):
    """Plan the sends for a batch of outgoing files.

    Photos and videos ride together in one media group, audio in another,
    documents in a third; small text files are inlined as ``<pre>`` instead of
    being sent as attachments nobody wants to download; Google proxy files
    become links. Anything over the upload cap is named in a trailing note
    rather than dropped silently.
    """
    photo_video, audio, documents, text_actions, skipped = [], [], [], [], []
    for path in paths:
        size = stat(sdk, path)
        if size is None:
            continue
        name = sdk.path.name(path)
        if size > max_file_size:
            skipped.append(
                f"{name} ({size / 1024 / 1024:.1f} MB exceeds 50 MB limit)")
            continue
        category = _classify(sdk, path, size)
        if category == "google_link":
            url = _google_link(sdk, path)
            if url:
                text_actions.append(SendAction(
                    "text",
                    text_content=f'<a href="{html.escape(url)}">'
                                 f'{html.escape(name)}</a>'))
            else:
                skipped.append(f"{name} (could not extract Google link)")
        elif category in {"photo", "video"}:
            photo_video.append(path)
        elif category == "audio":
            audio.append(path)
        elif category == "text":
            try:
                escaped = html.escape(sdk.fs.read(path))
            except sdk.Failed:
                documents.append(path)
                continue
            header = f"<b>{html.escape(name)}</b>\n<pre>"
            footer = "</pre>"
            available = 4096 - len(header) - len(footer)
            body = (escaped[:available - 20] + "\n... (truncated)"
                    if len(escaped) > available else escaped)
            text_actions.append(
                SendAction("text", text_content=header + body + footer))
        else:
            documents.append(path)

    actions = (_build_group_actions(sdk, photo_video, "photo_video")
               + _build_group_actions(sdk, audio, "audio")
               + _build_group_actions(sdk, documents, "document")
               + text_actions)
    if skipped:
        actions.append(SendAction(
            "text",
            text_content="Skipped files:\n" + "\n".join(
                f"- {item}" for item in skipped)))
    return actions


def method_for(sdk, path, group_type: str) -> str:
    """The single-file send method a grouped file would use on its own."""
    if group_type == "photo_video":
        return ("video" if sdk.path.suffix(path).lower() in VIDEO_EXTENSIONS
                else "photo")
    return "audio" if group_type == "audio" else "document"


def _build_group_actions(sdk, files, group_type: str):
    """Batch files into media groups of at most ten; singles are sent alone."""
    if not files:
        return []
    actions = []
    for i in range(0, len(files), _MEDIA_GROUP_MAX):
        chunk = files[i:i + _MEDIA_GROUP_MAX]
        if len(chunk) == 1:
            actions.append(
                SendAction(method_for(sdk, chunk[0], group_type), [chunk[0]]))
        else:
            actions.append(SendAction("media_group", chunk, group_type))
    return actions


class StreamTracker:
    """Pure-logic accumulator for one streamed agent reply.

    Fed by ``render`` as deltas arrive and drained by the frontend's stream
    pump. Deliberately transport-free, so the throttle and rollover logic is
    testable without python-telegram-bot.

    Contract with the pump:

    - ``should_edit(now)`` — is there unseen text AND has the edit throttle
      (interval or char burst) been satisfied?
    - ``take_render()`` — returns ``(finalized_heads, current_text)``. Heads
      are popped permanently: each rolls into its own finalized message when
      the buffer exceeds ``max_chars`` (Telegram's 4096 cap, with headroom for
      the cursor suffix). ``current_text`` is a snapshot the pump must confirm
      via ``mark_rendered`` after a successful edit — a throttled or failed
      edit is simply retried on the next pass.
    """

    def __init__(self, max_chars: int = 4000, edit_interval: float = 1.75,
                 burst_chars: int = 300):
        self.max_chars = max_chars
        self.edit_interval = edit_interval
        self.burst_chars = burst_chars
        self._pending = ""      # text not yet finalized into an earlier message
        self._rendered = ""     # what the current message is confirmed to show
        self._last_edit = 0.0
        self.rolled = False     # at least one size-cap rollover happened
        self.done = False
        self.aborted = False
        self.final_text = None

    def feed(self, delta: str) -> None:
        """Append streamed text."""
        self._pending += delta

    def finish(self, final_text, aborted: bool) -> None:
        """Mark the stream complete."""
        self.done, self.aborted, self.final_text = True, aborted, final_text

    def state(self):
        """Return ``(done, aborted, final_text)``."""
        return self.done, self.aborted, self.final_text

    def remainder(self) -> str:
        """Plain text belonging to the current (last) streamed message."""
        return self._pending

    def should_edit(self, now: float) -> bool:
        """Whether the pump should render this pass (dirty + throttle)."""
        if self._pending == self._rendered:
            return False
        grown = len(self._pending) - len(self._rendered)
        return ((now - self._last_edit) >= self.edit_interval
                or grown >= self.burst_chars)

    def take_render(self):
        """Pop finalized rollover heads and snapshot the current text."""
        finals = []
        while len(self._pending) > self.max_chars:
            split = self._pending.rfind("\n", self.max_chars // 2,
                                        self.max_chars)
            if split < 0:
                split = self.max_chars
            finals.append(self._pending[:split])
            self._pending = self._pending[split:].lstrip("\n")
            self._rendered = ""
            self.rolled = True
        current = self._pending if self._pending != self._rendered else None
        return finals, current

    def mark_rendered(self, text: str, now: float) -> None:
        """Confirm a successful edit so the throttle restarts from it."""
        self._rendered = text
        self._last_edit = now
