"""
Manage promo codes for the web demo.

Used (by Henry, from his REPL) to mint single-use codes that grant either
unlimited tier or a credits boost. Marked background_safe=False so it can't
run from a background driver — only from an attended session.
"""

import logging
import secrets
import time

from plugins.BaseTool import BaseTool, ToolResult

logger = logging.getLogger("ManagePromoCodes")


class ManagePromoCodes(BaseTool):
    """Manage promo codes."""
    name = "manage_promo_codes"
    description = (
        "Create, list, or delete promo codes for the web demo. "
        "Use action='create_unlimited' for an unlimited-tier code, "
        "'create_credits' (with credits=N) for a credit grant, "
        "'list' to see existing codes, or 'delete' with code=... to remove one."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create_unlimited", "create_credits", "list", "delete"],
                "description": "Which operation to perform.",
            },
            "credits": {"type": "integer", "description": "Credits to grant (for create_credits)."},
            "note": {"type": "string", "description": "Human-readable note attached to the code."},
            "code": {"type": "string", "description": "Code to delete (for action=delete)."},
        },
        "required": ["action"],
    }
    background_safe = False

    def run(self, context, **kwargs) -> ToolResult:
        """Run manage promo codes."""
        action = (kwargs.get("action") or "").strip()
        db = getattr(getattr(context, "runtime", None), "db", None) or getattr(context, "db", None)
        if db is None:
            return ToolResult.failed("Database not available.")
        note = (kwargs.get("note") or "").strip()

        if action == "create_unlimited":
            code = _mint_code()
            with db.lock:
                db.conn.execute(
                    "INSERT INTO web_promo_codes (code, kind, credits, max_uses, uses, created_at, note) "
                    "VALUES (?, 'unlimited', NULL, 1, 0, ?, ?)",
                    (code, time.time(), note),
                )
                db.conn.commit()
            return ToolResult(data={"code": code, "kind": "unlimited", "note": note}, llm_summary=f"Created unlimited promo code: {code}")

        if action == "create_credits":
            amt = int(kwargs.get("credits") or 0)
            if amt <= 0:
                return ToolResult.failed("credits must be a positive integer.")
            code = _mint_code()
            with db.lock:
                db.conn.execute(
                    "INSERT INTO web_promo_codes (code, kind, credits, max_uses, uses, created_at, note) "
                    "VALUES (?, 'credits', ?, 1, 0, ?, ?)",
                    (code, amt, time.time(), note),
                )
                db.conn.commit()
            return ToolResult(data={"code": code, "kind": "credits", "credits": amt, "note": note}, llm_summary=f"Created {amt}-credit promo code: {code}")

        if action == "list":
            with db.lock:
                rows = db.conn.execute(
                    "SELECT code, kind, credits, max_uses, uses, created_at, note FROM web_promo_codes ORDER BY created_at DESC"
                ).fetchall()
            items = [dict(r) for r in rows]
            return ToolResult(data={"codes": items}, llm_summary=f"{len(items)} promo code(s).")

        if action == "delete":
            code = (kwargs.get("code") or "").strip()
            if not code:
                return ToolResult.failed("code is required for delete.")
            with db.lock:
                cur = db.conn.execute("DELETE FROM web_promo_codes WHERE code = ?", (code,))
                db.conn.commit()
            deleted = (cur.rowcount or 0) > 0
            return ToolResult(data={"deleted": deleted, "code": code}, llm_summary=("Deleted." if deleted else "No matching code."))

        return ToolResult.failed(f"Unknown action: {action}")


def _mint_code() -> str:
    return secrets.token_urlsafe(9)
