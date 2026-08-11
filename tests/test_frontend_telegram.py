"""Tests for the store Telegram frontend and its renderers helper.

Both files live on the ``store`` branch rather than in the kernel tree, so they
are materialized here the same way ``test_frontend_mcp.py`` does its subject —
except that a *worktree* on the store branch is preferred over the committed
ref when one exists. Mid-migration the interesting version of a file is the one
being edited, and a test that silently checks the last commit instead is a test
that passes while the work is broken.

What is covered is the part that has no bot in it: the markdown-to-Telegram-HTML
pipeline, message chunking, the streamed-reply tracker, and the media planner —
plus the conformance verdict, which is what decides whether the plugin loads at
all. ``python-telegram-bot`` is only touched inside ``start`` and the send
paths, so none of it is needed to run these.
"""

import subprocess
import sys
import types
from pathlib import Path

import pytest

# These exercise behaviour of a file on the store branch, so a kernel change
# cannot break them and they do not belong in the kernel's default run. The
# claims the *kernel* makes about this frontend -- conformance, the
# declarations the bridge reads, resolved isolation -- live in
# tests/test_store_frontend_contracts.py and still run by default.
pytestmark = pytest.mark.store

# Aliases the guest package under the bare name ``guest``, which is how plugin
# source resolves its imports both in-process and in a child.
import sandbox  # noqa: F401
from guest.loader import load_member, unload_box
from guest.sdk import _Markdown, _Path

_REPO = Path(__file__).resolve().parents[1]
_FRONTEND_REL = "frontends/frontend_telegram.py"
_HELPER_REL = "frontends/helpers/telegram_renderers.py"
_BOX = "telegram_under_test"


def _store_worktree():
    """A checkout of the store branch, if this clone has one."""
    proc = subprocess.run(
        ["git", "-C", str(_REPO), "worktree", "list", "--porcelain"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        encoding="utf-8", check=False)
    root = None
    for line in proc.stdout.splitlines():
        if line.startswith("worktree "):
            root = Path(line.split(" ", 1)[1])
        elif line.strip() in {"branch refs/heads/store", "branch store"}:
            return root
    return None


def _store_source(relative: str):
    """The store's copy of one file, from a worktree or from a ref."""
    worktree = _store_worktree()
    if worktree is not None and (worktree / relative).is_file():
        return (worktree / relative).read_text(encoding="utf-8")
    for ref in ("store", "origin/store"):
        proc = subprocess.run(
            ["git", "-C", str(_REPO), "show", f"{ref}:{relative}"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            encoding="utf-8", check=False)
        if proc.returncode == 0:
            return proc.stdout
    return None


@pytest.fixture(scope="module")
def sources():
    """Both files' text, or a skip when no store branch is reachable."""
    frontend = _store_source(_FRONTEND_REL)
    helper = _store_source(_HELPER_REL)
    if frontend is None or helper is None:
        pytest.skip("the Telegram package is not present on a local store ref")
    return {_FRONTEND_REL: frontend, _HELPER_REL: helper}


@pytest.fixture(scope="module")
def tree(sources, tmp_path_factory):
    """The two files laid out as an installed plugin tree."""
    root = tmp_path_factory.mktemp("telegram_store")
    for relative, text in sources.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def module(tree):
    """The frontend, imported as a member of its box.

    Loaded through the guest loader rather than plain importlib, because the
    plugin's ``from .telegram_renderers import ...`` only resolves inside the
    synthetic package a box installs — and getting that wrong is precisely the
    mistake this migration had to avoid, since the box is flat and the helper
    ships in a subfolder.
    """
    pytest.importorskip("PIL", reason="the renderers helper imports Pillow")
    loaded = load_member(
        tree / _FRONTEND_REL, box_name=_BOX,
        root=tree / "frontends", extra_roots=[tree / "frontends" / "helpers"])
    yield loaded
    unload_box(_BOX)


@pytest.fixture(scope="module")
def renderers(module):
    """The helper module, as the frontend sees it."""
    return sys.modules[f"box_{_BOX}.telegram_renderers"]


# ──────────────────────────────────────────────────────────────────────
# Conformance: the verdict that decides whether it loads at all.
# ──────────────────────────────────────────────────────────────────────









# ──────────────────────────────────────────────────────────────────────
# The markdown pipeline.
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def sdk():
    """Enough of an SDK for the pure rendering paths: the real helpers."""
    return types.SimpleNamespace(md=_Markdown, path=_Path)


def test_fenced_code_becomes_pre(module, sdk):
    html = module._md_to_tg_html(sdk, "before\n```python\nx = 1\n```after")
    assert '<pre><code class="language-python">x = 1</code></pre>' in html
    assert "before" in html and "after" in html


def test_bold_italic_and_inline_code(module, sdk):
    html = module._md_to_tg_html(sdk, "**b** *i* `c` & <tag>")
    assert "<b>b</b>" in html
    assert "<i>i</i>" in html
    assert "<code>c</code>" in html
    # Everything outside a span is escaped, or Telegram rejects the message.
    assert "&lt;tag&gt;" in html
    assert "&amp;" in html


def test_tables_render_as_aligned_pre(module, sdk):
    table = "| Name | Size |\n| --- | --- |\n| a | 1 |\n| bbbb | 22 |\n"
    html = module._md_to_tg_html(sdk, table)
    assert html.startswith("<pre>") and html.rstrip().endswith("</pre>")
    # Padded to a common width, which is the whole reason it goes through
    # md.align_tables rather than md.plain.
    assert "Name  Size" in html


def test_blockquotes_survive_as_blockquotes(module, sdk):
    html = module._md_to_tg_html(sdk, "> quoted **text**\n> more\n")
    assert "<blockquote>" in html
    assert "<b>text</b>" in html


def test_detail_cards_compact_into_a_fence(module, sdk):
    """A two-column table with an empty second header is a card, not data."""
    card = "| Plugin |  |\n| --- | --- |\n| name | telegram |\n"
    out = module._compact_detail_cards(sdk, card)
    assert out.startswith("```\n")
    assert "Plugin" in out


def test_real_tables_are_left_alone(module, sdk):
    """Both headers filled means it is data, and Telegram renders it natively."""
    table = "| Name | Size |\n| --- | --- |\n| a | 1 |\n"
    assert module._compact_detail_cards(sdk, table) == table


def test_chunking_splits_on_newlines_under_the_cap(module):
    body = "\n".join("line %d" % i for i in range(2000))
    chunks = module._chunks(body, 4096)
    assert len(chunks) > 1
    assert all(len(chunk) <= 4096 for chunk in chunks)
    # Nothing is lost and nothing is duplicated; only the split newlines go.
    assert "".join(chunks).replace("\n", "") == body.replace("\n", "")


def test_short_text_is_one_chunk_and_empty_is_none(module):
    assert module._chunks("hi", 4096) == ["hi"]
    assert module._chunks("", 4096) == []


def test_a_line_longer_than_the_cap_is_still_split(module):
    """No newline to split on must not mean one oversized message."""
    chunks = module._chunks("x" * 9000, 4096)
    assert all(len(chunk) <= 4096 for chunk in chunks)
    assert "".join(chunks) == "x" * 9000


# ──────────────────────────────────────────────────────────────────────
# Approvals and the command banner.
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("yes", True), ("Y", True), ("approve", True), ("1", True),
    ("no", False), ("DENY", False), ("0", False), ("/cancel", False),
    ("maybe", None), ("", None),
])
def test_typed_approvals_parse(module, text, expected):
    assert module._parse_approval(text) is expected


def _frontend(module, remembered=None):
    """The frontend without running ``__init__``, for the pure decisions."""
    made = module.TelegramFrontend.__new__(module.TelegramFrontend)
    made._approvals = dict(remembered or {})
    return made


def test_an_enum_approval_is_answered_with_its_own_value(module):
    """A tapped button must answer in the shape that frame accepts.

    Every sandbox Request dialog is typed ``string`` with an enum, and the
    button carries one of its values. Coercing that to ``True`` — right for
    the boolean gate a command declares, wrong here — meant the state machine
    validated ``True`` against ``["allow", "deny"]``, refused it, and left the
    frame up. The plugin on the other side waited out its dialog timeout and
    was denied, while the person saw a dialog they had already answered.
    """
    sandboxed = _frontend(module, {"s1": {
        "type": "string",
        "enum": ["allow", "always:api.example.com", "deny"],
        "enum_labels": ["Allow once", "Always allow api.example.com", "Deny"],
    }})

    assert sandboxed._approval_value("s1", "allow") == "allow"
    assert sandboxed._approval_value("s1", "deny") == "deny"
    # A remembering option's value carries colons of its own; the callback
    # split keeps them, and nothing downstream may reinterpret them.
    assert sandboxed._approval_value(
        "s1", "always:api.example.com") == "always:api.example.com"


def test_a_boolean_approval_still_answers_with_a_boolean(module):
    """The command gate declares ``type="boolean"``, whose lenient parser
    wants a bool — so the historical mapping has to survive for it."""
    gate = _frontend(module, {"s1": {"type": "boolean"}})

    assert gate._approval_value("s1", "allow") is True
    assert gate._approval_value("s1", "deny") is False
    # Nothing remembered — a restart, or a dialog this box never rendered.
    assert _frontend(module)._approval_value("s1", "allow") is True


def test_the_command_banner_quotes_arguments_with_spaces(module):
    assert module._command_call("packages", {"action": "install"}) == (
        "/packages install")
    assert module._command_call(
        "/config", {"key": "a b"}) == '/config "a b"'
    # None means "not collected", and a banner showing it would be noise.
    assert module._command_call("clear", {"a": None, "b": "x"}) == "/clear x"


# ──────────────────────────────────────────────────────────────────────
# StreamTracker — the throttle and rollover behind a streamed reply.
# ──────────────────────────────────────────────────────────────────────

def test_a_tracker_holds_back_until_the_throttle_allows(renderers):
    tracker = renderers.StreamTracker(edit_interval=10.0, burst_chars=100)
    tracker.feed("short")
    # The pump passes a wall clock, so ``last_edit`` starting at 0 means the
    # first pass is always past the interval however small the text.
    assert tracker.should_edit(1e6) is True
    tracker.mark_rendered("short", 100.0)
    tracker.feed("!")
    assert tracker.should_edit(100.1) is False   # neither interval nor burst
    assert tracker.should_edit(111.0) is True    # interval satisfied


def test_a_burst_beats_the_interval(renderers):
    tracker = renderers.StreamTracker(edit_interval=10.0, burst_chars=10)
    tracker.mark_rendered("", 100.0)
    tracker.feed("x" * 20)
    assert tracker.should_edit(100.1) is True


def test_nothing_new_is_never_worth_an_edit(renderers):
    tracker = renderers.StreamTracker()
    tracker.feed("done")
    tracker.mark_rendered("done", 0.0)
    assert tracker.should_edit(1e6) is False


def test_oversize_text_rolls_into_finalized_heads(renderers):
    tracker = renderers.StreamTracker(max_chars=100)
    tracker.feed("a" * 60 + "\n" + "b" * 90)
    heads, current = tracker.take_render()
    assert len(heads) == 1
    assert heads[0] == "a" * 60
    assert current == "b" * 90
    assert tracker.rolled is True
    assert len(tracker.remainder()) <= 100


def test_a_rollover_with_no_newline_splits_at_the_cap(renderers):
    tracker = renderers.StreamTracker(max_chars=50)
    tracker.feed("z" * 130)
    heads, current = tracker.take_render()
    assert [len(head) for head in heads] == [50, 50]
    assert current == "z" * 30


def test_finish_is_reported_atomically(renderers):
    tracker = renderers.StreamTracker()
    assert tracker.state() == (False, False, None)
    tracker.finish("the whole reply", False)
    assert tracker.state() == (True, False, "the whole reply")


# ──────────────────────────────────────────────────────────────────────
# The media planner, over a fake SDK. This is the half of the helper that
# was rewritten from pathlib to Requests, so it is the half worth driving.
# ──────────────────────────────────────────────────────────────────────

class _FakeFS:
    """``fs.list`` and ``fs.read`` over a real directory."""

    def __init__(self, failed):
        self._failed = failed

    def list(self, path, pattern="*", details=False, **_kw):
        target = Path(path)
        if not target.is_file():
            raise self._failed(f"no such file: {path}")
        info = target.stat()
        return [{"path": str(target), "name": target.name, "is_dir": False,
                 "mtime": info.st_mtime_ns, "size": info.st_size}]

    def read(self, path):
        return Path(path).read_text(encoding="utf-8", errors="replace")

    def read_bytes(self, path, offset=0, length=0):
        data = Path(path).read_bytes()[offset:]
        return data[:length] if length else data


@pytest.fixture
def media_sdk():
    """A stub SDK using the real path and markdown helpers."""
    class Failed(Exception):
        """Stands in for sdk.Failed."""

    text_extensions = {".txt", ".md", ".py", ".csv"}
    return types.SimpleNamespace(
        Failed=Failed,
        fs=_FakeFS(Failed),
        path=_Path,
        md=_Markdown,
        parse=types.SimpleNamespace(
            modality=lambda ext: ("text" if ext in text_extensions
                                  else "unknown")),
        log=lambda *a, **k: None,
    )


def _write(root, name, size=16):
    path = root / name
    path.write_bytes(b"x" * size)
    return str(path)


def test_photos_and_videos_ride_in_one_media_group(renderers, media_sdk,
                                                   tmp_path):
    paths = [_write(tmp_path, "a.jpg"), _write(tmp_path, "b.png"),
             _write(tmp_path, "c.mp4")]
    actions = renderers.prepare_media_actions(media_sdk, paths)
    assert [a.method for a in actions] == ["media_group"]
    assert actions[0].group_type == "photo_video"
    assert len(actions[0].files) == 3


def test_a_lone_file_is_sent_by_its_own_method(renderers, media_sdk, tmp_path):
    actions = renderers.prepare_media_actions(
        media_sdk, [_write(tmp_path, "only.mp3")])
    assert [a.method for a in actions] == ["audio"]


def test_media_groups_cap_at_ten(renderers, media_sdk, tmp_path):
    paths = [_write(tmp_path, f"p{i}.jpg") for i in range(23)]
    actions = renderers.prepare_media_actions(media_sdk, paths)
    assert [len(a.files) for a in actions] == [10, 10, 3]


def test_small_text_files_are_inlined_rather_than_attached(renderers,
                                                           media_sdk,
                                                           tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello <world>", encoding="utf-8")
    actions = renderers.prepare_media_actions(media_sdk, [str(path)])
    assert [a.method for a in actions] == ["text"]
    assert "<pre>" in actions[0].text_content
    assert "hello &lt;world&gt;" in actions[0].text_content


def test_a_large_text_file_is_sent_as_a_document(renderers, media_sdk,
                                                 tmp_path):
    path = tmp_path / "big.txt"
    path.write_text("y" * 50_000, encoding="utf-8")
    actions = renderers.prepare_media_actions(media_sdk, [str(path)])
    assert [a.method for a in actions] == ["document"]


def test_oversized_files_are_named_not_dropped(renderers, media_sdk, tmp_path):
    small = _write(tmp_path, "ok.jpg")
    big = _write(tmp_path, "huge.bin", size=2048)
    actions = renderers.prepare_media_actions(media_sdk, [small, big],
                                              max_file_size=1024)
    assert actions[-1].method == "text"
    assert "huge.bin" in actions[-1].text_content
    assert "Skipped files" in actions[-1].text_content


def test_a_missing_path_is_skipped_silently(renderers, media_sdk, tmp_path):
    actions = renderers.prepare_media_actions(
        media_sdk, [str(tmp_path / "gone.jpg")])
    assert actions == []


def test_google_proxy_files_become_links(renderers, media_sdk, tmp_path):
    path = tmp_path / "plan.gdoc"
    path.write_text('{"doc_id": "abc123"}', encoding="utf-8")
    actions = renderers.prepare_media_actions(media_sdk, [str(path)])
    assert [a.method for a in actions] == ["text"]
    assert "docs.google.com/document/d/abc123" in actions[0].text_content


def test_unreadable_google_proxy_files_are_reported(renderers, media_sdk,
                                                    tmp_path):
    path = tmp_path / "broken.gsheet"
    path.write_text("not json", encoding="utf-8")
    actions = renderers.prepare_media_actions(media_sdk, [str(path)])
    assert "could not extract Google link" in actions[-1].text_content


def test_reading_walks_a_file_larger_than_one_window(renderers, media_sdk,
                                                     tmp_path, monkeypatch):
    """The whole reason ``fs.read_bytes`` grew a window: 50 MB, 11 MB frames."""
    monkeypatch.setattr(renderers, "_READ_WINDOW", 64)
    path = tmp_path / "big.bin"
    payload = bytes(range(256)) * 5          # 1280 bytes, 20 windows
    path.write_bytes(payload)
    assert renderers.read_all_bytes(media_sdk, str(path)) == payload


def test_reading_an_empty_file_terminates(renderers, media_sdk, tmp_path):
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")
    assert renderers.read_all_bytes(media_sdk, str(path)) == b""


# ──────────────────────────────────────────────────────────────────────
# Albums. Telegram sends a media group as several updates, and a submitted
# attachment hands the turn to the agent — so delivering them as they land
# sent the first photo and answered "Still working" about the rest.
# ──────────────────────────────────────────────────────────────────────

class _Handle:
    """A telegram File, as far as this frontend uses one."""

    def __init__(self, recorded):
        self._recorded = recorded

    def download_to_drive(self, path):
        """The transport writing straight to the path the kernel allocated."""
        self._recorded.append(path)


class _FakeFrontendSDK:
    """The inbound half of the SDK, recording what was submitted."""

    class Failed(Exception):
        """Stands in for sdk.Failed."""

    def __init__(self, tmp_path):
        self.submits = []
        self.downloads = []
        self.logs = []
        self._tmp = tmp_path
        self._made = 0
        self.path = _Path
        self.fs = types.SimpleNamespace(temp=self._temp)
        self.frontend = types.SimpleNamespace(
            submit_attachments=self._submit_attachments,
            submit_text=lambda key, text: self.submits.append(
                ("text", key, text)),
            pending_input=lambda key: None)
        self.session = types.SimpleNamespace(get=lambda key: {})

    def log(self, message, level="info"):
        self.logs.append((level, message))

    def _temp(self, suffix=""):
        self._made += 1
        return str(self._tmp / f"scratch{self._made}{suffix}")

    def _submit_attachments(self, key, files, caption="", ingest=False):
        self.submits.append(("files", key, list(files), caption, ingest))
        return True


@pytest.fixture
def telegram(module, tmp_path):
    """A frontend instance with a loop that runs nothing and an SDK that lies."""
    def run(done):
        """Stand in for the loop: nothing here needs a slice to actually pass."""
        if hasattr(done, "close"):        # a real coroutine, e.g. poll's sleep
            done.close()
        return done

    frontend = module.TelegramFrontend()
    frontend._loop = types.SimpleNamespace(run_until_complete=run)
    return frontend, _FakeFrontendSDK(tmp_path)


def _photo(key="telegram:1:1:0", name="photo.jpg", group="g1", caption="",
           recorded=None):
    return {"kind": "file", "key": key, "file": _Handle(recorded or []),
            "name": name, "caption": caption, "is_photo": True,
            "group": group}


def test_an_album_is_submitted_as_one_message(telegram):
    """Three photos, one submit — which is the only arrangement that works.

    Three submits would be three ``send_attachment`` actions, and the second
    arrives at a session the first just handed to the agent.
    """
    frontend, sdk = telegram
    for index in range(3):
        frontend._deliver(sdk, _photo(name=f"p{index}.jpg",
                                      caption="which is sharpest?" if not index
                                      else ""))

    assert sdk.submits == [], "nothing goes until the album stops arriving"

    frontend._albums[("telegram:1:1:0", "g1")]["last"] -= frontend._ALBUM_WAIT
    assert frontend.poll(sdk) is True

    kind, key, files, caption, ingest = sdk.submits[0]
    assert (kind, key, ingest) == ("files", "telegram:1:1:0", True)
    assert [one["file_name"] for one in files] == ["p0.jpg", "p1.jpg",
                                                   "p2.jpg"]
    # Hoisted: it is the album's line, not a label on the photo that carried it.
    assert caption == "which is sharpest?"
    assert [one["caption"] for one in files] == ["", "", ""]


def test_a_lone_file_still_goes_straight_through(telegram):
    """No media group, no waiting — the ordinary case is unchanged."""
    frontend, sdk = telegram

    frontend._deliver(sdk, _photo(group="", name="only.jpg",
                                  caption="look at this"))

    kind, key, files, caption, _ingest = sdk.submits[0]
    assert (kind, len(files)) == ("files", 1)
    assert files[0]["file_name"] == "only.jpg"
    # A single file's caption stays on the file, which is what the one-file
    # form has always sent.
    assert (files[0]["caption"], caption) == ("look at this", "")


def test_anything_else_for_that_chat_sends_the_album_first(telegram):
    """What a person sent in an order should arrive in it."""
    frontend, sdk = telegram
    frontend._deliver(sdk, _photo(name="p0.jpg"))
    frontend._deliver(sdk, {"kind": "text", "key": "telegram:1:1:0",
                            "text": "what do you make of those?"})

    assert [one[0] for one in sdk.submits] == ["files", "text"]


def test_another_chats_album_is_left_alone(telegram):
    """Flushing is per chat: one person typing must not post another's photos."""
    frontend, sdk = telegram
    frontend._deliver(sdk, _photo(key="telegram:2:2:0", name="theirs.jpg"))
    frontend._deliver(sdk, {"kind": "text", "key": "telegram:1:1:0",
                            "text": "hello"})

    assert [one[0] for one in sdk.submits] == ["text"]
    assert ("telegram:2:2:0", "g1") in frontend._albums


def test_a_still_arriving_album_is_not_sent_early(telegram):
    """The window is measured from the *last* photo, not the first."""
    frontend, sdk = telegram
    frontend._deliver(sdk, _photo(name="p0.jpg"))
    frontend._albums[("telegram:1:1:0", "g1")]["last"] -= frontend._ALBUM_WAIT
    frontend._deliver(sdk, _photo(name="p1.jpg"))

    assert frontend._flush_albums(sdk) is False
    assert sdk.submits == []


def test_stopping_says_what_it_dropped(telegram):
    """Photos vanishing without a word is what this whole path avoids."""
    frontend, sdk = telegram
    frontend._deliver(sdk, _photo(name="p0.jpg"))

    frontend.stop(sdk)

    assert any("unsent album" in message for _level, message in sdk.logs)
    assert frontend._albums == {}
