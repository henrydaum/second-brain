"""Small text-file CRUD tool for repo-native editing."""

from pathlib import Path

from paths import DATA_DIR, ROOT_DIR
from plugins.BaseTool import BaseTool, ToolResult

ROOTS = tuple(p.resolve() for p in (ROOT_DIR, DATA_DIR))


def _path(raw: str) -> tuple[Path | None, str | None]:
    raw = (raw or "").strip()
    if not raw:
        return None, "path is required."
    p = Path(raw)
    p = (p if p.is_absolute() else ROOT_DIR / p).resolve()
    return (p, None) if any(p == r or r in p.parents for r in ROOTS) else (None, f"Path is outside allowed roots: {p}")


class EditFile(BaseTool):
    name = "edit_file"
    description = (
        "Create, overwrite, exact-replace, append to, or delete a UTF-8 text file. "
        "Use read_file first for non-trivial edits, then replace exact text. "
        "Paths may be absolute or relative to the project root; edits are limited "
        "to the project root and Second Brain data directory."
    )
    parameters = {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["create", "overwrite", "replace", "append", "delete"], "description": "File operation to perform."},
            "path": {"type": "string", "description": "Target file path."},
            "content": {"type": "string", "description": "Text for create, overwrite, or append."},
            "old_text": {"type": "string", "description": "Exact text to replace."},
            "new_text": {"type": "string", "description": "Replacement text."},
            "replace_all": {"type": "boolean", "description": "Replace every occurrence instead of requiring exactly one match."},
        },
        "required": ["operation", "path"],
    }
    requires_services = []
    max_calls = 20
    background_safe = False

    def run(self, context, **kwargs) -> ToolResult:
        op = (kwargs.get("operation") or "").strip().lower()
        p, err = _path(kwargs.get("path", ""))
        if err:
            return ToolResult.failed(err)
        try:
            if op == "delete":
                if not p.exists():
                    return ToolResult.failed(f"File not found: {p}")
                if not p.is_file():
                    return ToolResult.failed(f"Not a file: {p}")
                p.unlink()
                return ToolResult(data={"path": str(p), "operation": op}, llm_summary=f"Deleted {p}.")
            if op in {"create", "overwrite", "append"}:
                text = kwargs.get("content")
                if text is None:
                    return ToolResult.failed("content is required for create, overwrite, and append.")
                if op == "create" and p.exists():
                    return ToolResult.failed(f"File already exists: {p}")
                p.parent.mkdir(parents=True, exist_ok=True)
                prior = p.read_text(encoding="utf-8") if op == "append" and p.exists() else ""
                p.write_text((prior + text) if op == "append" else text, encoding="utf-8")
                verb = {"create": "Created", "overwrite": "Overwrote", "append": "Appended to"}[op]
                return ToolResult(data={"path": str(p), "operation": op}, llm_summary=f"{verb} {p}.")
            if op == "replace":
                old, new = kwargs.get("old_text"), kwargs.get("new_text")
                if old in (None, ""):
                    return ToolResult.failed("old_text is required for replace.")
                if new is None:
                    return ToolResult.failed("new_text is required for replace.")
                if not p.is_file():
                    return ToolResult.failed(f"File not found: {p}")
                text = p.read_text(encoding="utf-8")
                count = text.count(old)
                if count == 0:
                    return ToolResult.failed("old_text was not found.")
                if count > 1 and not kwargs.get("replace_all"):
                    return ToolResult.failed(f"old_text appears {count} times; pass replace_all=true or make it unique.")
                p.write_text(text.replace(old, new, -1 if kwargs.get("replace_all") else 1), encoding="utf-8")
                return ToolResult(data={"path": str(p), "operation": op, "replacements": count if kwargs.get("replace_all") else 1}, llm_summary=f"Replaced text in {p}.")
            return ToolResult.failed("operation must be create, overwrite, replace, append, or delete.")
        except UnicodeDecodeError:
            return ToolResult.failed(f"Cannot edit binary or non-UTF-8 file: {p}")
        except Exception as e:
            return ToolResult.failed(f"Edit failed: {e}")
