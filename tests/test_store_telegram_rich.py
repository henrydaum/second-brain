"""Tests for the store Telegram frontend's Rich Message capability logic.

Materializes ``frontends/frontend_telegram.py`` + its renderers helper from
the local store ref as a small package (the frontend uses a relative helper
import), with a stub PIL so no Pillow is needed. Covers the capability
probe, the rich/basic agent-prompt switch, draft-id derivation, and the
"rich refused" classifier — the transport calls themselves are not driven.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO = Path(__file__).resolve().parents[1]


def _store_source(rel: str) -> str | None:
    for ref in ("store", "origin/store"):
        proc = subprocess.run(
            ["git", "-C", str(_REPO), "show", f"{ref}:{rel}"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", check=False)
        if proc.returncode == 0:
            return proc.stdout
    return None


@pytest.fixture(scope="module")
def frontend_cls(tmp_path_factory):
    frontend_src = _store_source("frontends/frontend_telegram.py")
    helper_src = _store_source("frontends/helpers/telegram_renderers.py")
    if frontend_src is None or helper_src is None:
        pytest.skip("telegram frontend not present on a local store ref")
    if "PIL" not in sys.modules:  # the helper imports Pillow at top level
        pil = types.ModuleType("PIL")
        pil.Image = types.ModuleType("PIL.Image")
        sys.modules["PIL"] = pil
    root = tmp_path_factory.mktemp("tg_frontend")
    pkg = root / "tg_store_pkg"
    (pkg / "helpers").mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "helpers" / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "frontend_telegram.py").write_text(frontend_src, encoding="utf-8")
    (pkg / "helpers" / "telegram_renderers.py").write_text(helper_src, encoding="utf-8")
    sys.path.insert(0, str(root))
    try:
        import importlib
        module = importlib.import_module("tg_store_pkg.frontend_telegram")
    finally:
        sys.path.remove(str(root))
    return module.TelegramFrontend


def _frontend(frontend_cls, bot_attrs: dict | None):
    fe = frontend_cls()
    if bot_attrs is None:
        fe.app = None
    else:
        fe.app = SimpleNamespace(bot=SimpleNamespace(**bot_attrs))
    return fe


def test_rich_capable_with_raw_request_layer(frontend_cls):
    fe = _frontend(frontend_cls, {"do_api_request": lambda *a, **k: None})
    assert fe._rich_capable() is True


def test_rich_incapable_on_ancient_ptb(frontend_cls):
    fe = _frontend(frontend_cls, {})  # neither typed method nor raw layer
    assert fe._rich_capable() is False
    # ...and the agent prompt advertises only basic formatting.
    prompt = fe.agent_prompt_for(None)
    assert "do NOT render" in prompt


def test_rich_prompt_advertises_full_markdown(frontend_cls):
    fe = _frontend(frontend_cls, {"do_api_request": lambda *a, **k: None})
    prompt = fe.agent_prompt_for(None)
    assert "tables" in prompt and "headings" in prompt
    assert "do NOT render" not in prompt


def test_prompt_downgrades_after_runtime_refusal(frontend_cls):
    fe = _frontend(frontend_cls, {"do_api_request": lambda *a, **k: None})
    assert fe._rich_capable() is True
    fe._rich = False  # what _deliver_message_async does on "Not Found"
    assert "do NOT render" in fe.agent_prompt_for(None)


def test_optimistic_before_transport_up_but_not_cached(frontend_cls):
    fe = _frontend(frontend_cls, None)  # app not started yet
    assert fe._rich_capable() is True
    assert fe._rich is None  # undetermined — re-probed once the bot exists


def test_draft_id_is_stable_and_nonzero(frontend_cls):
    a = frontend_cls._draft_id_for("st_ab12cd34ef56")
    assert a == frontend_cls._draft_id_for("st_ab12cd34ef56")
    assert a > 0
    assert frontend_cls._draft_id_for("st_0000000000000000") > 0  # never zero
    assert frontend_cls._draft_id_for("not-hex!") > 0


@pytest.fixture(scope="module")
def tg_module(frontend_cls):
    return sys.modules["tg_store_pkg.frontend_telegram"]


_TABLE_MD = "Files:\n\n| Name | Count |\n| --- | --- |\n| Tools | 16 |\n| Frontends | 2 |\n\nPick one."


def test_html_fallback_renders_tables_as_pre(tg_module):
    out = tg_module._md_to_tg_html(_TABLE_MD)

    assert "<pre>" in out and "</pre>" in out
    pre = out.split("<pre>")[1].split("</pre>")[0]
    assert "|" not in pre  # aligned columns, not raw markdown pipes
    lines = pre.split("\n")
    assert lines[0].startswith("Name")
    assert lines[2].index("16") == lines[3].index("2")
    assert out.startswith("Files:")
    assert out.endswith("Pick one.")


def test_html_fallback_renders_blockquotes(tg_module):
    out = tg_module._md_to_tg_html("Preview:\n\n> user: hi there\n> assistant: hello\n\nPick an action.")

    assert "<blockquote>user: hi there\nassistant: hello</blockquote>" in out
    assert out.startswith("Preview:")
    assert out.endswith("Pick an action.")


def test_detail_cards_compact_to_code_blocks_but_data_tables_stay(tg_module):
    card = "| timekeeper |  |\n| --- | --- |\n| Status | Loaded |"
    out = tg_module._compact_detail_cards(f"Pick:\n\n{card}\n\ntail")
    assert "```" in out and "| Status | Loaded |" not in out
    assert "timekeeper" in out

    data = "| Name | Count |\n| --- | --- |\n| Tools | 16 |"
    assert tg_module._compact_detail_cards(data) == data  # real table untouched


def test_form_prompt_renders_markdown(frontend_cls, tg_module):
    fe = _frontend(frontend_cls, {})
    prompt = fe._prompt({"field": {"prompt": _TABLE_MD}})
    assert "<pre>" in prompt
    assert "| Tools | 16 |" not in prompt


def test_banner_respects_config_toggle(frontend_cls):
    fe = _frontend(frontend_cls, {})
    fe.loop = object()
    fe._chat_by_session["s"] = 42
    scheduled = []
    fe._send_nowait = lambda coro: (scheduled.append(coro), coro.close())

    fe.config = {"telegram_pin_banner": False}
    fe.render_conversation_banner("s", {"title": "FIFA Briefings"})
    assert scheduled == []

    fe.config = {}  # default: enabled
    fe.render_conversation_banner("s", {"title": "FIFA Briefings"})
    assert len(scheduled) == 1


class _FakeBot:
    def __init__(self, fail_edit=False):
        self.fail_edit = fail_edit
        self.edits, self.sent, self.pins, self.unpins = [], [], [], []

    async def edit_message_text(self, text, chat_id=None, message_id=None):
        if self.fail_edit:
            raise RuntimeError("message to edit not found")
        self.edits.append((chat_id, message_id, text))

    async def send_message(self, chat_id, text, disable_notification=True):
        self.sent.append((chat_id, text))
        return SimpleNamespace(message_id=99)

    async def pin_chat_message(self, chat_id, message_id, disable_notification=True):
        self.pins.append((chat_id, message_id))

    async def unpin_chat_message(self, chat_id, message_id=None):
        self.unpins.append((chat_id, message_id))


def _banner_frontend(frontend_cls, monkeypatch, persisted=None, fail_edit=False):
    import asyncio

    fe = frontend_cls()
    bot = _FakeBot(fail_edit=fail_edit)
    fe.app = SimpleNamespace(bot=bot)
    fe.config = {"telegram_banner_messages": persisted or {}}
    saved = {}
    monkeypatch.setattr("config.config_manager.load_plugin_config", lambda: {})
    monkeypatch.setattr("config.config_manager.save_plugin_config", lambda values: saved.update(values))
    return fe, bot, saved, asyncio.run


def test_banner_restart_with_same_title_does_not_repin(frontend_cls, monkeypatch):
    # A fresh instance (restart) sees the persisted banner and does nothing —
    # this is the bug where every restart pinned a duplicate banner.
    fe, bot, _saved, run = _banner_frontend(
        frontend_cls, monkeypatch, persisted={"42": [7, "FIFA Briefings"]})

    run(fe._update_banner(42, "FIFA Briefings"))

    assert bot.sent == [] and bot.pins == [] and bot.edits == []


def test_banner_retitle_edits_in_place_and_persists(frontend_cls, monkeypatch):
    fe, bot, saved, run = _banner_frontend(
        frontend_cls, monkeypatch, persisted={"42": [7, "Old Title"]})

    run(fe._update_banner(42, "New Title"))

    assert bot.edits == [(42, 7, "\U0001f4ac New Title")]
    assert bot.pins == []
    assert saved["telegram_banner_messages"] == {"42": [7, "New Title"]}


def test_banner_edit_failure_replaces_and_unpins_stale(frontend_cls, monkeypatch):
    # The old pinned message is gone (e.g. user deleted it): pin the fresh
    # banner and unpin the stale id rather than accumulating pins.
    fe, bot, saved, run = _banner_frontend(
        frontend_cls, monkeypatch, persisted={"42": [7, "Old Title"]}, fail_edit=True)

    run(fe._update_banner(42, "New Title"))

    assert bot.pins == [(42, 99)]
    assert bot.unpins == [(42, 7)]
    assert saved["telegram_banner_messages"] == {"42": [99, "New Title"]}


def test_banner_map_tolerates_json_string_and_junk(frontend_cls):
    fe = frontend_cls()
    fe.config = {"telegram_banner_messages": '{"42": [7, "Title"], "bad": "junk"}'}
    assert fe._banner_map() == {42: (7, "Title")}


def test_rich_refused_classifier(frontend_cls):
    fe = _frontend(frontend_cls, {})
    assert fe._rich_refused(Exception("404 Not Found"))
    assert fe._rich_refused(Exception("Unknown method"))
    assert not fe._rich_refused(Exception("Bad Request: message is too long"))
