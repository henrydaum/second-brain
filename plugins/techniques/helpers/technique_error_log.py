"""Persistent log of technique execution failures.

Writes one row per `execute_technique` failure into the `technique_errors` table —
slug, error type, message, the failing line, the corrective hint, and the
params that triggered it. Conversation content is never stored.

Mined offline to surface recurring failure modes; those then become advice
folded into the generative-art encyclopedia (§13 Common pitfalls). Mirrors
the minimal-API shape of `technique_scoring.record_event()`.
"""

from __future__ import annotations

import json
import logging
import time

logger = logging.getLogger("TechniqueErrorLog")


def record_error(
    db,
    slug: str,
    params: dict | None,
    diagnostic: dict | None,
    session_key: str | None = None,
) -> None:
    """Persist a technique failure. Silent no-op if `db` is missing — the caller's
    error path must not depend on logging succeeding."""
    if db is None or not slug:
        return
    diag = diagnostic or {}
    error_type = str(diag.get("error_type") or "UnknownError")
    message = diag.get("message")
    lineno = diag.get("technique_lineno")
    line = diag.get("technique_line")
    hint = diag.get("hint")
    try:
        params_json = json.dumps(params, default=str) if params else None
    except Exception:
        params_json = None
    try:
        with db.lock:
            db.conn.execute(
                "INSERT INTO technique_errors (ts, slug, error_type, message, technique_lineno, technique_line, hint, params_json, session_key) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    time.time(),
                    str(slug),
                    error_type,
                    (str(message) if message is not None else None),
                    (int(lineno) if isinstance(lineno, int) else None),
                    (str(line) if line is not None else None),
                    (str(hint) if hint is not None else None),
                    params_json,
                    (str(session_key) if session_key else None),
                ),
            )
            db.conn.commit()
    except Exception:
        logger.exception("record_error failed: slug=%s", slug)
