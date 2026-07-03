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


def test_form_prompt_renders_markdown(frontend_cls, tg_module):
    fe = _frontend(frontend_cls, {})
    prompt = fe._prompt({"field": {"prompt": _TABLE_MD}})
    assert "<pre>" in prompt
    assert "| Tools | 16 |" not in prompt


def test_rich_refused_classifier(frontend_cls):
    fe = _frontend(frontend_cls, {})
    assert fe._rich_refused(Exception("404 Not Found"))
    assert fe._rich_refused(Exception("Unknown method"))
    assert not fe._rich_refused(Exception("Bad Request: message is too long"))
