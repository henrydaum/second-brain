"""Persistence operations for integer-credit promo codes."""

import secrets
import time


def create(db, credits: int, note: str = "", code: str = "") -> dict:
    credits = int(credits or 0)
    if credits <= 0:
        raise ValueError("credits must be a positive integer.")
    code = code.strip() or secrets.token_urlsafe(9)
    with db.lock:
        if db.conn.execute("SELECT 1 FROM web_promo_codes WHERE code = ?", (code,)).fetchone():
            raise ValueError(f"Code '{code}' already exists.")
        db.conn.execute(
            "INSERT INTO web_promo_codes (code, kind, credits, max_uses, uses, created_at, note) VALUES (?, 'credits', ?, 1, 0, ?, ?)",
            (code, credits, time.time(), note.strip()),
        )
        db.conn.commit()
    return {"code": code, "kind": "credits", "credits": credits, "note": note.strip()}


def list_all(db) -> list[dict]:
    with db.lock:
        rows = db.conn.execute("SELECT code, kind, credits, max_uses, uses, created_at, note FROM web_promo_codes ORDER BY created_at DESC").fetchall()
    return [dict(row) for row in rows]


def delete(db, code: str) -> bool:
    with db.lock:
        cur = db.conn.execute("DELETE FROM web_promo_codes WHERE code = ?", (code.strip(),))
        db.conn.commit()
    return (cur.rowcount or 0) > 0


def redeem(db, credits_service, code: str, account_id: str) -> dict:
    code = (code or "").strip()
    if not code:
        return {"ok": False, "error": "Enter a code."}
    if not account_id:
        return {"ok": False, "error": "Sign in first to redeem a promo code.", "need_auth": True}
    with db.lock:
        row = db.conn.execute("SELECT kind, credits, max_uses, uses FROM web_promo_codes WHERE code = ?", (code,)).fetchone()
        if not row:
            return {"ok": False, "error": "Code not found."}
        if (row["uses"] or 0) >= (row["max_uses"] or 1):
            return {"ok": False, "error": "This code has already been used."}
        user = db.conn.execute("SELECT user_id FROM web_users WHERE account_id = ?", (account_id,)).fetchone()
        if not user:
            return {"ok": False, "error": "Account not found."}
        if row["kind"] != "credits":
            return {"ok": False, "error": "Unknown code kind."}
        amount = int(row["credits"] or 0)
        db.conn.execute("UPDATE web_promo_codes SET uses = uses + 1 WHERE code = ?", (code,))
        db.conn.commit()
    if credits_service:
        credits_service.grant(db, user["user_id"], amount, "promo")
    return {"ok": True, "granted": f"{amount} credits"}
