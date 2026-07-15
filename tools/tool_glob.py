"""Find files by name pattern on disk."""

dependencies_files = ['tools/helpers/file_walk.py']
dependencies_pip = []

from plugins.BaseTool import BaseTool, ToolResult
from .helpers.file_walk import compile_glob, iter_files, match_rel, mtime_sorted, resolve_root

DEFAULT_LIMIT = 100
MAX_LIMIT = 500


class GlobFiles(BaseTool):
    """Glob files."""
    name = "glob"
    description = (
        "Find files by glob pattern. '*' and '?' never cross directories; use '**' "
        "for any depth ('*.py' = top-level files only, 'src/**/*.ts' = any depth under "
        "src). Searches the project root by default; paths may be absolute or relative "
        "to it. Results are newest-first. Skips junk directories (.git, node_modules, "
        "__pycache__, ...). Use grep to search file contents instead."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern, e.g. '*.py' (top level) or '**/*.py' (any depth)."},
            "path": {"type": "string", "description": "Directory to search. Absolute or relative to the project root. Defaults to the project root."},
            "limit": {"type": "integer", "description": "Max files to return. Default 100, max 500."},
        },
        "required": ["pattern"],
    }
    requires_services = []
    max_calls = 10
    background_safe = True

    def run(self, context, **kwargs) -> ToolResult:
        """Run glob."""
        pattern = (kwargs.get("pattern") or "").strip()
        if not pattern:
            return ToolResult.failed("No pattern provided.")
        limit = max(1, min(int(kwargs.get("limit") or DEFAULT_LIMIT), MAX_LIMIT))

        root, err = resolve_root(kwargs.get("path"))
        if err:
            return ToolResult.failed(err)
        if not root.is_dir():
            return ToolResult.failed(f"Not a directory: {root}")

        compiled = compile_glob(pattern)
        files, scan_truncated = iter_files(root)
        matches = mtime_sorted([f for f in files if match_rel(f, root, compiled)])
        truncated = len(matches) > limit
        matches = matches[:limit]

        rels = [f.relative_to(root).as_posix() for f in matches]
        lines = [f"Glob '{pattern}' under {root} — {len(rels)} file(s)."]
        if not rels:
            lines.append("No files matched.")
        else:
            lines.append("")
            lines.extend(rels)
        if truncated:
            lines.append(f"(showing first {limit} newest files; more exist — narrow the pattern or raise limit)")
        if scan_truncated:
            lines.append("(file scan hit the enumeration cap — narrow 'path')")

        return ToolResult(
            True,
            data={"root": str(root), "pattern": pattern, "results": rels,
                  "truncated": truncated, "scan_truncated": scan_truncated},
            llm_summary="\n".join(lines),
        )
