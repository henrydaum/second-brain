from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import canvas.render as canvas_render
import plugins.frontends.frontend_web as fw
from canvas.runtime import CanvasRuntime
from events.event_bus import bus
from events.event_channels import CANVAS_CHANGED
from plugins.tools.tool_execute_skill import ExecuteSkill


class Registry:
	kinds = {"voronoi": "background", "grain": "filter", "menger": "object"}
	def get(self, slug):
		return SimpleNamespace(slug=slug, name=slug, kind=self.kinds[slug], controls=[], code="")
	get_record = get


def run_case(live: bool) -> dict:
	old_dir, old_run, calls, fe, unsub = canvas_render.RENDERS_DIR, canvas_render.run_skill, [], None, None
	tmp = Path(tempfile.mkdtemp(prefix="live_canvas_bench_"))
	def fake_run_skill(skill, *, output_image_path, **_):
		calls.append(skill.slug)
		Path(output_image_path).parent.mkdir(parents=True, exist_ok=True)
		Image.new("RGBA", (8, 8), (32, 64, 96, 255)).save(output_image_path)
		return {"ok": True}
	try:
		canvas_render.RENDERS_DIR = tmp
		canvas_render.run_skill = fake_run_skill
		key, cr, reg = "web:bench", CanvasRuntime(), Registry()
		ctx = SimpleNamespace(session_key=key, canvas=cr, skill_registry=reg, db=None, config={}, services={})
		if live:
			fe = fw.WebFrontend()
			fe.runtime = SimpleNamespace(sessions={key: SimpleNamespace()}, skill_registry=reg)
			unsub = bus.subscribe(CANVAS_CHANGED, fe.on_bus_canvas_changed)
		t0, steps = time.perf_counter(), []
		for slug in ("voronoi", "grain", "menger"):
			s0 = time.perf_counter()
			ExecuteSkill().run(ctx, slug=slug)
			steps.append((slug, (time.perf_counter() - s0) * 1000))
		events = fe._drain(key, force=True) if fe else []
		return {"mode": "live" if live else "baseline", "total_ms": (time.perf_counter() - t0) * 1000, "steps": steps, "run_skill_calls": calls, "events": len(events)}
	finally:
		if unsub:
			unsub()
		canvas_render.RENDERS_DIR, canvas_render.run_skill = old_dir, old_run
		shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
	for result in (run_case(False), run_case(True)):
		print(f"{result['mode']}: {result['total_ms']:.2f} ms, skill calls={len(result['run_skill_calls'])}, events={result['events']}")
		for slug, ms in result["steps"]:
			print(f"  {slug}: {ms:.2f} ms")
