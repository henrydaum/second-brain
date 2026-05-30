"""Subprocess sandbox for executing technique code.

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
import threading
import time
from pathlib import Path

try:
    import psutil
except ImportError:
    psutil = None

from plugins.helpers.palettes import Palette, get_palette, palette_exists
from plugins.services.service_technique_worker_pool import TechniqueWorkerBusy, TechniqueWorkerJobTimeout
from plugins.techniques.helpers.technique_store import Technique, assert_valid

logger = logging.getLogger("TechniqueRunner")

DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MEMORY_MB = 768


class TechniqueRunError(RuntimeError):
    """Raised when a technique fails. `diagnostic` carries structured fields
    (error_type, message, technique_lineno, technique_line, hint) when available —
    populated from the sandbox sidecar, or built directly for validation /
    timeout failures."""

    def __init__(self, message: str, diagnostic: dict | None = None):
        super().__init__(message)
        self.diagnostic = diagnostic or {}


def run_technique(
    technique: Technique,
    *,
    params: dict,
    palette: Palette,
    size: int,
    seed: int,
    input_image_path: Path | None,
    output_image_path: Path,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    memory_mb: int = DEFAULT_MEMORY_MB,
    png_compress_level: int = 1,
    worker_pool=None,
) -> dict:
    """Execute a technique in a sandboxed subprocess. Returns a small status dict."""
    try:
        assert_valid(technique.code)
    except Exception as e:
        msg = str(e)
        if "disallowed import" in msg:
            hint = "Only math, random, colorsys, numpy, PIL.*, and `from plugins.BaseTechnique import BaseTechnique` are importable. Remove the disallowed import."
            raise TechniqueRunError(
                f"technique '{technique.slug}' failed validation: {msg}\n  hint: {hint}",
                diagnostic={"error_type": "ValidationError", "message": msg, "hint": hint},
            )
        raise TechniqueRunError(
            f"technique '{technique.slug}' failed validation: {msg}",
            diagnostic={"error_type": "ValidationError", "message": msg},
        )
    if technique.kind in ("filter", "object") and (input_image_path is None or not Path(input_image_path).is_file()):
        raise TechniqueRunError(
            f"{technique.kind} techniques require a current canvas image; run a background first",
            diagnostic={
                "error_type": "MissingPriorCanvas",
                "message": f"{technique.kind} techniques require a current canvas image; run a background first",
                "hint": f"Run a background technique first, then chain this {technique.kind}.",
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
        "code": technique.code,
        "kind": technique.kind,
        "params": dict(params or {}),
        "palette": palette.to_dict(),
        "size": int(size),
        "seed": int(seed),
        "input_image_path": str(input_image_path) if input_image_path else None,
        "output_image_path": str(output_image_path),
        "memory_mb": int(memory_mb),
        "png_compress_level": max(0, min(9, int(png_compress_level))),
    }

    entry = Path(__file__).with_name("technique_sandbox_entry.py")
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
    mem_cap_bytes = max(64, int(memory_mb)) * 1024 * 1024
    killed_for_memory = {"flag": False, "peak": 0}
    returncode = 1
    try:
        if worker_pool is not None and getattr(worker_pool, "enabled", True):
            try:
                pooled = worker_pool.run_job(
                    job_path=job_path, timeout_s=timeout_s, memory_mb=memory_mb,
                )
            except TechniqueWorkerBusy as e:
                raise TechniqueRunError(
                    "renderer busy; retry shortly",
                    diagnostic={"error_type": "RendererBusy", "message": str(e), "hint": "Wait a few seconds and retry; the global technique worker pool is at capacity."},
                )
            except TechniqueWorkerJobTimeout:
                raise TechniqueRunError(
                    f"technique '{technique.slug}' exceeded {timeout_s:.0f}s timeout",
                    diagnostic={
                        "error_type": "Timeout",
                        "message": f"exceeded {timeout_s:.0f}s timeout",
                        "hint": "Vectorize with numpy or reduce iteration counts; per-pixel Python loops at full resolution always time out.",
                    },
                )
            stdout, stderr, returncode = pooled.stdout, pooled.stderr, pooled.returncode
            killed_for_memory.update(flag=pooled.memory_killed, peak=pooled.peak_bytes)
        else:
            proc = subprocess.Popen(
                cmd, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=env, cwd=str(project_root),
            )
            watchdog_stop = threading.Event()
            if psutil is not None:
                def _watchdog():
                    try:
                        p = psutil.Process(proc.pid)
                    except psutil.Error:
                        return
                    while not watchdog_stop.wait(0.2):
                        try:
                            rss = p.memory_info().rss
                            for child in p.children(recursive=True):
                                try:
                                    rss += child.memory_info().rss
                                except psutil.Error:
                                    pass
                        except psutil.Error:
                            return
                        if rss > killed_for_memory["peak"]:
                            killed_for_memory["peak"] = rss
                        if rss > mem_cap_bytes:
                            killed_for_memory["flag"] = True
                            try:
                                proc.kill()
                            except OSError:
                                pass
                            return
                wd = threading.Thread(target=_watchdog, name=f"technique-mem-watchdog-{proc.pid}", daemon=True)
                wd.start()
            else:
                wd = None
            try:
                stdout, stderr = proc.communicate(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.communicate(timeout=2.0)
                except Exception:
                    pass
                raise TechniqueRunError(
                    f"technique '{technique.slug}' exceeded {timeout_s:.0f}s timeout",
                    diagnostic={
                        "error_type": "Timeout",
                        "message": f"exceeded {timeout_s:.0f}s timeout",
                        "hint": "Vectorize with numpy or reduce iteration counts; per-pixel Python loops at full resolution always time out.",
                    },
                )
            finally:
                watchdog_stop.set()
                if wd is not None:
                    wd.join(timeout=1.0)
            returncode = proc.returncode
        if killed_for_memory["flag"]:
            peak_mb = killed_for_memory["peak"] / (1024 * 1024)
            raise TechniqueRunError(
                f"technique '{technique.slug}' exceeded {memory_mb} MB memory cap (peak ~{peak_mb:.0f} MB)",
                diagnostic={
                    "error_type": "MemoryCap",
                    "message": f"exceeded {memory_mb} MB memory cap (peak ~{peak_mb:.0f} MB)",
                    "hint": f"Reduce array sizes or canvas resolution; allocations like np.zeros((N, N, 4), dtype=uint8) need 4*N*N bytes, so N=10000 already uses ~400 MB. Stay under {memory_mb} MB total.",
                },
            )
    finally:
        try:
            os.unlink(job_path)
        except OSError:
            pass

    elapsed = time.time() - t0
    sidecar = _read_sidecar(output_image_path)
    if returncode != 0:
        err = (stderr or b"").decode("utf-8", errors="replace").strip()
        out = (stdout or b"").decode("utf-8", errors="replace").strip()
        logger.error("Technique '%s' failed (rc=%s).\nSTDERR:\n%s\nSTDOUT:\n%s", technique.slug, returncode, err or "(empty)", out or "(empty)")
        diag = dict(sidecar) if sidecar else {"error_type": "SandboxFailure", "message": (err.splitlines()[-1] if err else (out or "unknown error"))}
        raise TechniqueRunError(_format_error(technique.slug, sidecar, err, out), diagnostic=diag)
    if not output_image_path.is_file():
        raise TechniqueRunError(
            f"technique '{technique.slug}' did not commit an image",
            diagnostic={
                "error_type": "MissingCommit",
                "message": "run() returned without calling canvas.commit(image)",
                "hint": "Every code path through run() must end with canvas.commit(image).",
            },
        )

    result = {
        "slug": technique.slug,
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
        parts = [f"technique '{slug}' failed: {sidecar['error_type']}: {sidecar.get('message', '')}"]
        lineno = sidecar.get("technique_lineno")
        line = (sidecar.get("technique_line") or "").strip()
        if lineno and line:
            parts.append(f"  at line {lineno}: {line}")
        elif lineno:
            parts.append(f"  at line {lineno}")
        if sidecar.get("hint"):
            parts.append(f"  hint: {sidecar['hint']}")
        return "\n".join(parts)
    msg = stderr.splitlines()[-1] if stderr else stdout or "unknown error"
    return f"technique '{slug}' failed: {msg}"


def make_chain_entry(technique: Technique, params: dict, seed: int, controls: dict | None = None) -> dict:
    return {
        "slug": technique.slug,
        "kind": technique.kind,
        "params": dict(params or {}),
        "controls": dict(controls or {}),
        "seed": int(seed),
    }


def default_controls(technique: Technique) -> dict:
    """Build the initial control-value dict from a technique's declared schema."""
    values: dict = {}
    for c in (technique.controls or []):
        ctype = c.get("type")
        if ctype == "pan":
            values[c["x_param"]] = c.get("x_default", 0.0)
            values[c["y_param"]] = c.get("y_default", 0.0)
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
