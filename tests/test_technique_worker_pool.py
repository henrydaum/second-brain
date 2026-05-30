from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from PIL import Image

from plugins.helpers.palettes import get_palette
from plugins.services.service_technique_worker_pool import TechniqueWorkerBusy, TechniqueWorkerPoolService
from plugins.techniques.helpers.technique_runner import TechniqueRunError, run_technique
from plugins.techniques.helpers.technique_store import Technique


GOOD = """
from plugins.BaseTechnique import BaseTechnique
import numpy as np

class WarmTechnique(BaseTechnique):
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
from plugins.BaseTechnique import BaseTechnique

class BadTechnique(BaseTechnique):
    name = "Bad"
    description = "Fails."
    kind = "background"

    def run(self, canvas):
        raise ValueError("boom")
"""


def _technique(code=GOOD):
    return Technique("warm", "", "Warm", "Warm worker smoke.", "background", "test", code, 1.0)


@pytest.fixture
def out_dir():
    path = Path(".technique_worker_pool_test")
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def pool():
    svc = TechniqueWorkerPoolService({
        "technique_worker_pool_enabled": True,
        "technique_worker_min_idle": 1,
        "technique_worker_max_workers": 2,
        "technique_worker_queue_timeout_s": 5,
    })
    svc.load()
    try:
        yield svc
    finally:
        svc.unload()


def test_run_technique_uses_warm_worker_pool(pool, out_dir):
    out = out_dir / "out.png"
    run_technique(_technique(), params={}, palette=get_palette("japandi"), size=24, seed=1, input_image_path=None, output_image_path=out, worker_pool=pool)
    assert Image.open(out).convert("RGBA").getpixel((0, 0)) == (64, 128, 0, 255)


def test_worker_failure_retires_and_replacement_runs(pool, out_dir):
    with pytest.raises(TechniqueRunError) as ei:
        run_technique(_technique(BAD), params={}, palette=get_palette("japandi"), size=24, seed=1, input_image_path=None, output_image_path=out_dir / "bad.png", worker_pool=pool)
    assert ei.value.diagnostic.get("error_type") == "ValueError"
    run_technique(_technique(), params={}, palette=get_palette("japandi"), size=24, seed=2, input_image_path=None, output_image_path=out_dir / "good.png", worker_pool=pool)
    assert (out_dir / "good.png").is_file()


def test_pool_busy_does_not_cold_fallback(out_dir):
    svc = TechniqueWorkerPoolService({"technique_worker_pool_enabled": True, "technique_worker_min_idle": 0, "technique_worker_max_workers": 1, "technique_worker_queue_timeout_s": 0.1})
    svc._loaded = True
    assert svc._active_sem.acquire(blocking=False)
    try:
        with pytest.raises(TechniqueWorkerBusy):
            svc.run_job(job_path=str(out_dir / "unused.json"), timeout_s=1, memory_mb=128)
    finally:
        svc._active_sem.release()
        svc.unload()
