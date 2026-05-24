"""
Manage promo codes for the web demo.

Used (by Henry, from his REPL) to mint single-use codes that grant integer
credit boosts. Marked background_safe=False so it can't
run from a background driver — only from an attended session.
"""

from billing import promo_codes
from plugins.BaseTool import BaseTool, ToolResult


class ManagePromoCodes(BaseTool):
    """Manage promo codes."""
    name = "manage_promo_codes"
    description = (
        "Create, list, or delete promo codes for the web demo. "
        "Use op='create_credits' (with credits=N) for a credit grant, "
        "'list' to see existing codes, or 'delete' with code=... to remove one."
    )
    parameters = {
        "type": "object",
        "properties": {
            "op": {
                "type": "string",
                "enum": ["create_credits", "list", "delete"],
                "description": "Which operation to perform.",
            },
            "credits": {"type": "integer", "description": "Credits to grant (for create_credits)."},
            "note": {"type": "string", "description": "Human-readable note attached to the code."},
            "code": {"type": "string", "description": "On create: custom code to use (leave blank for random). On delete: the code to remove."},
        },
        "required": ["op"],
    }
    background_safe = False

    def run(self, context, **kwargs) -> ToolResult:
        """Run manage promo codes."""
        op = (kwargs.get("op") or "").strip()
        db = getattr(getattr(context, "runtime", None), "db", None) or getattr(context, "db", None)
        if db is None:
            return ToolResult.failed("Database not available.")
        if op == "create_credits":
            try:
                item = promo_codes.create(db, kwargs.get("credits"), kwargs.get("note") or "", kwargs.get("code") or "")
            except ValueError as e:
                return ToolResult.failed(str(e))
            return ToolResult(data=item, llm_summary=f"Created {item['credits']}-credit promo code: {item['code']}")

        if op == "list":
            items = promo_codes.list_all(db)
            return ToolResult(data={"codes": items}, llm_summary=f"{len(items)} promo code(s).")

        if op == "delete":
            code = (kwargs.get("code") or "").strip()
            if not code:
                return ToolResult.failed("code is required for delete.")
            deleted = promo_codes.delete(db, code)
            return ToolResult(data={"deleted": deleted, "code": code}, llm_summary=("Deleted." if deleted else "No matching code."))

        return ToolResult.failed(f"Unknown op: {op}")
