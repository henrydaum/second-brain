from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import plugins.frontends.frontend_web as fw
import plugins.tools.tool_execute_skill as execute_mod
from canvas.runtime import CanvasRuntime
from events.event_bus import bus
from events.event_channels import CANVAS_CHANGED


def test_push_wakes_stream_waiter_and_preserves_order():
	fe, key, seen = fw.WebFrontend(), "web:s", []
	t = threading.Thread(target=lambda: seen.extend(fe._wait_stream_events(key, timeout=1)), daemon=True)
	t.start()
	time.sleep(0.05)
	fe._push(key, {"type": "status", "content": "one"})
	fe._push(key, {"type": "status", "content": "two"})
	t.join(1)
	assert [e["content"] for e in seen] == ["one", "two"]


def test_active_stream_keeps_post_drain_from_stealing_events():
	fe, key = fw.WebFrontend(), "web:s"
	fe._stream_open(key)
	try:
		fe._push(key, {"type": "status", "content": "stream me"})
		assert fe._drain(key) == []
		assert fe._wait_stream_events(key, timeout=0)[0]["content"] == "stream me"
	finally:
		fe._stream_close(key)


def test_web_app_uses_eventsource_without_fast_polling():
	text = Path("plugins/frontends/web/app.js").read_text(encoding="utf-8")
	assert "new EventSource" in text
	assert "setInterval(poll, 1200)" not in text


def test_out_of_credits_uses_chat_notice_not_popup():
	text = Path("plugins/frontends/web/app.js").read_text(encoding="utf-8")
	html = Path("plugins/frontends/web/index.html").read_text(encoding="utf-8")
	assert "/app.js?v=31" in html
	assert "const stick = atBottom();" in text and "bottom(stick);" in text
	assert 'ev.error?.code === "out_of_credits"' in text
	assert "next_refill_seconds" in text and 'link.href = "/account"' in text
	assert 'ev.type === "credit_denied"' in text and 'details?.action === "render"' in text and 'ev.action === "render"' in text
	assert "openPaywall" not in text and "outOfMessages" not in html


def test_canvas_change_event_reaches_web_before_final_message():
	key = "web:s"
	fe = fw.WebFrontend()
	fe.config = {}
	fe.runtime = SimpleNamespace(
		sessions={key: SimpleNamespace()},
		skill_registry=SimpleNamespace(get_record=lambda slug: SimpleNamespace(name=slug, kind="background", controls=[])),
	)
	snap = {"path": "render.png", "chain": [{"slug": "base", "kind": "background", "controls": {}}], "size": 512, "palette_id": "default"}
	fe.on_bus_canvas_changed({"session_key": key, "canvas": snap})
	fe.render_messages(key, ["done"])
	assert [e["type"] for e in fe._drain(key, force=True)] == ["hero_image", "message"]


def test_execute_skill_emits_canvas_changed_after_render(monkeypatch):
	events = []
	unsub = bus.subscribe(CANVAS_CHANGED, events.append)
	try:
		monkeypatch.setattr(execute_mod, "render_canvas", lambda *a, **k: SimpleNamespace(
			image_path=Path("render.png"), pool_hash="abc123", seed=7, cache_hit=False, warning=None, warning_message=None,
		))

		class Reg:
			def get(self, slug):
				return SimpleNamespace(slug=slug, kind="background", controls=[])
			get_record = get

		ctx = SimpleNamespace(session_key="web:s", canvas=CanvasRuntime(), skill_registry=Reg(), db=None, config={}, services={})
		result = execute_mod.ExecuteSkill().run(ctx, slug="base")
		assert result.success
		assert events and events[-1]["canvas"]["path"] == "render.png"
	finally:
		unsub()
