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
from plugins.skills.helpers.skill_store import Skill, assert_valid

logger = logging.getLogger("SkillRunner")

DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MEMORY_MB = 768


class SkillRunError(RuntimeError):
    """Raised when a skill fails. `diagnostic` carries structured fields
    (error_type, message, skill_lineno, skill_line, hint) when available —
    populated from the sandbox sidecar, or built directly for validation /
    timeout failures."""

    def __init__(self, message: str, diagnostic: dict | None = None):
        super().__init__(message)
        self.diagnostic = diagnostic or {}


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
    memory_mb: int = DEFAULT_MEMORY_MB,
) -> dict:
    """Execute a skill in a sandboxed subprocess. Returns a small status dict."""
    try:
        assert_valid(skill.code)
    except Exception as e:
        msg = str(e)
        if "disallowed import" in msg:
            hint = "Only math, random, colorsys, numpy, PIL.*, and `from plugins.BaseSkill import BaseSkill` are importable. Remove the disallowed import."
            raise SkillRunError(
                f"skill '{skill.slug}' failed validation: {msg}\n  hint: {hint}",
                diagnostic={"error_type": "ValidationError", "message": msg, "hint": hint},
            )
        raise SkillRunError(
            f"skill '{skill.slug}' failed validation: {msg}",
            diagnostic={"error_type": "ValidationError", "message": msg},
        )
    if skill.kind == "transform" and (input_image_path is None or not Path(input_image_path).is_file()):
        raise SkillRunError(
            "transform skills require a current canvas image; create one first",
            diagnostic={
                "error_type": "TransformWithoutCanvas",
                "message": "transform skills require a current canvas image; create one first",
                "hint": "Run a creation skill first, then chain this transform.",
            },
        )

    output_image_path = Path(output_image_path)
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
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
        "memory_mb": int(memory_mb),
    }

    entry = Path(__file__).with_name("skill_sandbox_entry.py")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    project_root = Path(__file__).resolve().parents[3]
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
            raise SkillRunError(
                f"skill '{skill.slug}' exceeded {timeout_s:.0f}s timeout",
                diagnostic={
                    "error_type": "Timeout",
                    "message": f"exceeded {timeout_s:.0f}s timeout",
                    "hint": "Vectorize with numpy or reduce iteration counts; per-pixel Python loops at full resolution always time out.",
                },
            )
    finally:
        try:
            os.unlink(job_path)
        except OSError:
            pass

    elapsed = time.time() - t0
    sidecar = _read_sidecar(output_image_path)
    if proc.returncode != 0:
        err = (stderr or b"").decode("utf-8", errors="replace").strip()
        out = (stdout or b"").decode("utf-8", errors="replace").strip()
        logger.error("Skill '%s' failed (rc=%s).\nSTDERR:\n%s\nSTDOUT:\n%s", skill.slug, proc.returncode, err or "(empty)", out or "(empty)")
        diag = dict(sidecar) if sidecar else {"error_type": "SandboxFailure", "message": (err.splitlines()[-1] if err else (out or "unknown error"))}
        raise SkillRunError(_format_error(skill.slug, sidecar, err, out), diagnostic=diag)
    if not output_image_path.is_file():
        raise SkillRunError(
            f"skill '{skill.slug}' did not commit an image",
            diagnostic={
                "error_type": "MissingCommit",
                "message": "run() returned without calling canvas.commit(image)",
                "hint": "Every code path through run() must end with canvas.commit(image).",
            },
        )

    result = {
        "slug": skill.slug,
        "duration_s": elapsed,
        "output_image_path": str(output_image_path),
        "stdout": (stdout or b"").decode("utf-8", errors="replace"),
    }
    if sidecar and sidecar.get("warning"):
        result["warning"] = sidecar.get("warning")
        result["warning_message"] = sidecar.get("message") or ""
    return result


def _read_sidecar(output_image_path: Path) -> dict | None:
    p = Path(str(output_image_path) + ".err.json")
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    finally:
        try: p.unlink()
        except OSError: pass
    return data if isinstance(data, dict) else None


def _format_error(slug: str, sidecar: dict | None, stderr: str, stdout: str) -> str:
    """Build a rich, agent-readable error message from a sandbox diagnostic."""
    if sidecar and sidecar.get("error_type"):
        parts = [f"skill '{slug}' failed: {sidecar['error_type']}: {sidecar.get('message', '')}"]
        lineno = sidecar.get("skill_lineno")
        line = (sidecar.get("skill_line") or "").strip()
        if lineno and line:
            parts.append(f"  at line {lineno}: {line}")
        elif lineno:
            parts.append(f"  at line {lineno}")
        if sidecar.get("hint"):
            parts.append(f"  hint: {sidecar['hint']}")
        return "\n".join(parts)
    msg = stderr.splitlines()[-1] if stderr else stdout or "unknown error"
    return f"skill '{slug}' failed: {msg}"


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
            continue
        elif ctype == "palette":
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
    on_step=None,
) -> dict:
    """Replay a chain of (creation, *transforms) into output_image_path.

    Each step consults the layer cache (``skill_cache``). On a hit the
    cached PNG is copied/used as the step's output and no subprocess
    runs. On a miss the subprocess renders, the result is cached, and
    the seed is added to the pool. Entries with ``seed=None`` sample
    from the seed pool first and only mint a fresh seed if the pool is
    empty.

    Mutates ``chain`` in place to record the resolved seed on each
    entry (so the caller can persist it to ``cs.canvas``).

    skill_loader(slug) -> Skill | None — caller supplies the lookup.
    """
    if not chain:
        raise SkillRunError("nothing to replay")
    from plugins.skills.helpers import skill_cache
    import shutil

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    current_input: Path | None = None
    final = Path(output_image_path)
    cache_hits = 0

    for idx, entry in enumerate(chain):
        slug = entry.get("slug")
        skill = skill_loader(slug) if slug else None
        if skill is None:
            raise SkillRunError(f"chain references missing skill '{slug}'")
        step_out = final if idx == len(chain) - 1 else (workdir / f"_replay_{idx}.png")
        merged_params, step_palette = resolve_entry(entry, fallback_palette=palette)

        # ── cache lookup ──
        code_sha = skill_cache.code_version(skill.code)
        in_hash = skill_cache.image_hash(current_input)
        pkey = skill_cache.pool_key(
            slug=slug, code_sha=code_sha, merged_params=merged_params,
            palette_id=getattr(step_palette, "id", "") or "",
            size=int(size), input_hash=in_hash,
        )
        raw_seed = entry.get("seed")
        # seed=None means "let the pool decide" (RunSkill path); an explicit
        # int means use it as-is (Regenerate, randomize-button, replay).
        if raw_seed is None:
            seed = skill_cache.sample_seed(pkey)
            if seed is None:
                seed = skill_cache.mint_seed()
        else:
            seed = int(raw_seed)
        entry["seed"] = seed  # record back so cs.canvas persists it

        ckey = skill_cache.cache_key(pkey, seed)
        cached = skill_cache.get(ckey)
        if cached is not None:
            try:
                shutil.copyfile(cached, step_out)
                current_input = step_out
                cache_hits += 1
                if on_step is not None:
                    try: on_step(idx + 1, len(chain))
                    except Exception: pass
                continue
            except OSError:
                logger.exception("cache copy-out failed; falling through to render")

        # ── miss: render, then cache ──
        run_skill(
            skill,
            params=merged_params,
            palette=step_palette,
            size=size,
            seed=int(seed),
            input_image_path=current_input,
            output_image_path=step_out,
            timeout_s=timeout_s,
        )
        try:
            skill_cache.put(
                ckey, step_out,
                skill_slug=slug, size=int(size),
                palette_id=getattr(step_palette, "id", "") or "",
                seed=int(seed), pool_key_=pkey,
            )
            skill_cache.add_seed(pkey, int(seed))
        except Exception:
            logger.exception("cache write failed (non-fatal)")

        current_input = step_out
        if on_step is not None:
            try: on_step(idx + 1, len(chain))
            except Exception: pass
    return {"steps": len(chain), "cache_hits": cache_hits, "output_image_path": str(final)}
