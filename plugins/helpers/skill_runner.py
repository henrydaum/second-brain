"""Subprocess sandbox for executing skill code.

Spawns a child python interpreter in isolated mode (-I), pipes a JSON job
description over stdin, and waits for the child to write a result image to
the agreed output path. Hard wall-clock timeout in the parent; AST validation
re-checked in the child before exec.

This module never touches PIL on the parent side beyond reading the final
output; it stays a thin process boss so the import surface is small.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from plugins.helpers.palettes import Palette, get_palette, palette_exists
from plugins.helpers.skill_store import Skill, assert_valid

logger = logging.getLogger("SkillRunner")

DEFAULT_TIMEOUT_S = 30.0


class SkillRunError(RuntimeError):
    pass


def run_skill(
    skill: Skill,
    *,
    params: dict,
    palette: Palette,
    size: int,
    seed: int,
    input_image_path: Path | None,
    output_image_path: Path,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict:
    """Execute a skill in a sandboxed subprocess. Returns a small status dict."""
    assert_valid(skill.code)
    if skill.kind == "transform" and (input_image_path is None or not Path(input_image_path).is_file()):
        raise SkillRunError("transform skills require a current canvas image; create one first")

    output_image_path = Path(output_image_path)
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    # Pre-delete any stale output so we can detect missing commits.
    if output_image_path.exists():
        try:
            output_image_path.unlink()
        except OSError:
            pass

    job = {
        "code": skill.code,
        "kind": skill.kind,
        "params": dict(params or {}),
        "palette": palette.to_dict(),
        "size": int(size),
        "seed": int(seed),
        "input_image_path": str(input_image_path) if input_image_path else None,
        "output_image_path": str(output_image_path),
    }

    entry = Path(__file__).with_name("skill_sandbox_entry.py")
    # PYTHONPATH so the child can import plugins.helpers.palettes / numpy / PIL from the parent env.
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    project_root = Path(__file__).resolve().parents[2]
    extra = str(project_root)
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = extra + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = extra

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(job, f)
        job_path = f.name

    cmd = [sys.executable, "-I", "-B", str(entry), job_path]
    t0 = time.time()
    try:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, cwd=str(project_root),
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.communicate(timeout=2.0)
            except Exception:
                pass
            raise SkillRunError(f"skill '{skill.slug}' exceeded {timeout_s:.0f}s timeout")
    finally:
        try:
            os.unlink(job_path)
        except OSError:
            pass

    elapsed = time.time() - t0
    if proc.returncode != 0:
        err = (stderr or b"").decode("utf-8", errors="replace").strip()
        out = (stdout or b"").decode("utf-8", errors="replace").strip()
        msg = err.splitlines()[-1] if err else out or "unknown error"
        logger.error("Skill '%s' failed (rc=%s).\nSTDERR:\n%s\nSTDOUT:\n%s", skill.slug, proc.returncode, err or "(empty)", out or "(empty)")
        raise SkillRunError(f"skill '{skill.slug}' failed: {msg}")
    if not output_image_path.is_file():
        raise SkillRunError(f"skill '{skill.slug}' did not commit an image")

    return {
        "slug": skill.slug,
        "duration_s": elapsed,
        "output_image_path": str(output_image_path),
        "stdout": (stdout or b"").decode("utf-8", errors="replace"),
    }


def make_chain_entry(skill: Skill, params: dict, seed: int, controls: dict | None = None) -> dict:
    return {
        "slug": skill.slug,
        "kind": skill.kind,
        "params": dict(params or {}),
        "controls": dict(controls or {}),
        "seed": int(seed),
    }


def default_controls(skill: Skill) -> dict:
    """Build the initial control-value dict from a skill's declared schema."""
    values: dict = {}
    for c in (skill.controls or []):
        ctype = c.get("type")
        if ctype == "pan":
            values[c["x_param"]] = c.get("x_default", 0.0)
            values[c["y_param"]] = c.get("y_default", 0.0)
        elif ctype == "button":
            # Buttons trigger actions; no default value lives in the entry.
            continue
        elif ctype == "palette":
            # Filled in by the caller (seeded from canvas state).
            continue
        else:
            if "default" in c:
                values[c["name"]] = c["default"]
    return values


def resolve_entry(entry: dict, *, fallback_palette: Palette) -> tuple[dict, Palette]:
    """Merge an entry's controls onto its params and resolve its palette."""
    params = dict(entry.get("params") or {})
    controls = dict(entry.get("controls") or {})
    palette = fallback_palette
    palette_id = controls.pop("palette", None) or params.pop("palette", None)
    if isinstance(palette_id, str) and palette_exists(palette_id):
        palette = get_palette(palette_id)
    # Pan controls live under their x_param/y_param names already; same for sliders/enums/bools.
    params.update(controls)
    return params, palette


def replay_chain(
    chain: list[dict],
    *,
    palette: Palette,
    size: int,
    output_image_path: Path,
    workdir: Path,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    skill_loader,
) -> dict:
    """Replay a chain of (creation, *transforms) into output_image_path.

    skill_loader(slug) -> Skill | None — caller supplies the lookup so the runner
    doesn't have to depend on skill_store directly.
    """
    if not chain:
        raise SkillRunError("nothing to replay")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    current_input: Path | None = None
    final = Path(output_image_path)

    for idx, entry in enumerate(chain):
        slug = entry.get("slug")
        skill = skill_loader(slug) if slug else None
        if skill is None:
            raise SkillRunError(f"chain references missing skill '{slug}'")
        step_out = final if idx == len(chain) - 1 else (workdir / f"_replay_{idx}.png")
        merged_params, step_palette = resolve_entry(entry, fallback_palette=palette)
        run_skill(
            skill,
            params=merged_params,
            palette=step_palette,
            size=size,
            seed=int(entry.get("seed") or 0),
            input_image_path=current_input,
            output_image_path=step_out,
            timeout_s=timeout_s,
        )
        current_input = step_out
    return {"steps": len(chain), "output_image_path": str(final)}
