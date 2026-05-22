from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from PIL import Image

from plugins.helpers.palettes import get_palette
from plugins.services.service_skill_worker_pool import SkillWorkerBusy, SkillWorkerPoolService
from plugins.skills.helpers.skill_runner import SkillRunError, run_skill
from plugins.skills.helpers.skill_store import Skill


GOOD = """
from plugins.BaseSkill import BaseSkill
import numpy as np

class WarmSkill(BaseSkill):
    name = "Warm"
    description = "Warm worker smoke."
    kind = "background"

    def run(self, canvas):
        arr = np.zeros((canvas.size, canvas.size, 3), dtype=np.uint8)
        arr[..., 0] = 64
        arr[..., 1] = 128
        canvas.commit_array(arr)
"""

BAD = """
from plugins.BaseSkill import BaseSkill

class BadSkill(BaseSkill):
    name = "Bad"
    description = "Fails."
    kind = "background"

    def run(self, canvas):
        raise ValueError("boom")
"""


def _skill(code=GOOD):
    return Skill("warm", "", "Warm", "Warm worker smoke.", "background", "test", code, 1.0)


@pytest.fixture
def out_dir():
    path = Path(".skill_worker_pool_test")
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def pool():
    svc = SkillWorkerPoolService({
        "skill_worker_pool_enabled": True,
        "skill_worker_min_idle": 1,
        "skill_worker_max_workers": 2,
        "skill_worker_queue_timeout_s": 5,
    })
    svc.load()
    try:
        yield svc
    finally:
        svc.unload()


def test_run_skill_uses_warm_worker_pool(pool, out_dir):
    out = out_dir / "out.png"
    run_skill(_skill(), params={}, palette=get_palette("japandi"), size=24, seed=1, input_image_path=None, output_image_path=out, worker_pool=pool)
    assert Image.open(out).convert("RGBA").getpixel((0, 0)) == (64, 128, 0, 255)


def test_worker_failure_retires_and_replacement_runs(pool, out_dir):
    with pytest.raises(SkillRunError) as ei:
        run_skill(_skill(BAD), params={}, palette=get_palette("japandi"), size=24, seed=1, input_image_path=None, output_image_path=out_dir / "bad.png", worker_pool=pool)
    assert ei.value.diagnostic.get("error_type") == "ValueError"
    run_skill(_skill(), params={}, palette=get_palette("japandi"), size=24, seed=2, input_image_path=None, output_image_path=out_dir / "good.png", worker_pool=pool)
    assert (out_dir / "good.png").is_file()


def test_pool_busy_does_not_cold_fallback(out_dir):
    svc = SkillWorkerPoolService({"skill_worker_pool_enabled": True, "skill_worker_min_idle": 0, "skill_worker_max_workers": 1, "skill_worker_queue_timeout_s": 0.1})
    svc._loaded = True
    assert svc._active_sem.acquire(blocking=False)
    try:
        with pytest.raises(SkillWorkerBusy):
            svc.run_job(job_path=str(out_dir / "unused.json"), timeout_s=1, memory_mb=128)
    finally:
        svc._active_sem.release()
        svc.unload()
