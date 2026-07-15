"""Live-tree content search — regex over files on disk.

Complements lexical_search (which queries the indexed corpus): grep reads
the actual files under the project root / data directory right now, so it
sees uncommitted and unindexed content.
"""

dependencies_files = ['tools/helpers/file_walk.py']
dependencies_pip = []

import re
from pathlib import Path

from plugins.BaseTool import BaseTool, ToolResult
from .helpers.file_walk import (
    MAX_FILE_BYTES,
    compile_glob,
    is_binary,
    iter_files,
    match_rel,
    mtime_sorted,
    resolve_root,
)

DEFAULT_LIMIT = 100
MAX_LIMIT = 500
MAX_CONTEXT = 10
MAX_CHARS = 20_000  # summary cap, mirrors read_file


class Grep(BaseTool):
    """Grep."""
    name = "grep"
    description = (
        "Search file contents on disk with a Python regular expression (re syntax, "
        "not PCRE — escape literal braces etc.). Searches the project root by default; "
        "paths may be absolute or relative to it. Filter files with 'glob' "
        "('*.py' = top level only, '**/*.py' = any depth). Skips binary and very large "
        "files and well-known junk directories (.git, node_modules, __pycache__, ...). "
        "Use lexical_search instead when you want ranked search over indexed content."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Python re regular expression to search for."},
            "path": {"type": "string", "description": "File or directory to search. Absolute or relative to the project root. Defaults to the project root."},
            "glob": {"type": "string", "description": "Glob filter for files, e.g. '*.py' (top level) or '**/*.py' (any depth)."},
            "output_mode": {"type": "string", "enum": ["files_with_matches", "content", "count"], "description": "files_with_matches (default): matching file paths. content: matching lines with line numbers. count: match counts per file."},
            "case_insensitive": {"type": "boolean", "description": "Case-insensitive matching. Default false."},
            "context_lines": {"type": "integer", "description": "Lines of context around each match (content mode only, max 10). Default 0."},
            "multiline": {"type": "boolean", "description": "Let the pattern span lines ('.' matches newlines too). Default false."},
            "limit": {"type": "integer", "description": "Max results (files, lines, or count rows). Default 100, max 500."},
        },
        "required": ["pattern"],
    }
    requires_services = []
    max_calls = 10
    background_safe = True

    def run(self, context, **kwargs) -> ToolResult:
        """Run grep."""
        pattern = (kwargs.get("pattern") or "").strip()
        if not pattern:
            return ToolResult.failed("No pattern provided.")

        flags = re.IGNORECASE if kwargs.get("case_insensitive") else 0
        multiline = bool(kwargs.get("multiline"))
        if multiline:
            flags |= re.DOTALL
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return ToolResult.failed(f"Invalid regex: {e}")

        mode = kwargs.get("output_mode") or "files_with_matches"
        if mode not in {"files_with_matches", "content", "count"}:
            return ToolResult.failed(f"Unknown output_mode: {mode}")
        context_lines = max(0, min(int(kwargs.get("context_lines") or 0), MAX_CONTEXT))
        limit = max(1, min(int(kwargs.get("limit") or DEFAULT_LIMIT), MAX_LIMIT))

        root, err = resolve_root(kwargs.get("path"))
        if err:
            return ToolResult.failed(err)
        if not root.exists():
            return ToolResult.failed(f"Path not found: {root}")

        glob_filter = None
        raw_glob = (kwargs.get("glob") or "").strip()
        if raw_glob:
            glob_filter = compile_glob(raw_glob)

        scan_truncated = False
        if root.is_file():
            files, base = [root], root.parent
        else:
            files, scan_truncated = iter_files(root)
            base = root
            if glob_filter is not None:
                files = [f for f in files if match_rel(f, base, glob_filter)]
            files = mtime_sorted(files)

        results = []          # per-mode payloads
        skipped_binary = 0
        skipped_large = 0
        truncated = False

        for f in files:
            try:
                if f.stat().st_size > MAX_FILE_BYTES:
                    skipped_large += 1
                    continue
            except OSError:
                continue
            if is_binary(f):
                skipped_binary += 1
                continue
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rel = self._rel(f, base)

            if mode == "files_with_matches":
                if regex.search(text):
                    results.append(rel)
            elif mode == "count":
                n = sum(1 for _ in regex.finditer(text))
                if n:
                    results.append((rel, n))
            else:  # content
                self._content_matches(regex, multiline, text, rel, context_lines, results, limit)

            if len(results) >= limit:
                truncated = len(results) > limit or f is not files[-1]
                results = results[:limit]
                break

        summary = self._summary(pattern, root, mode, results, truncated,
                                scan_truncated, skipped_binary, skipped_large, limit)
        return ToolResult(
            True,
            data={
                "root": str(root), "mode": mode, "results": results,
                "truncated": truncated, "scan_truncated": scan_truncated,
                "skipped_binary": skipped_binary, "skipped_large": skipped_large,
            },
            llm_summary=summary,
        )

    @staticmethod
    def _rel(path: Path, base: Path) -> str:
        """Root-relative posix path for display."""
        try:
            return path.relative_to(base).as_posix()
        except ValueError:
            return str(path)

    @staticmethod
    def _content_matches(regex, multiline, text, rel, context_lines, results, limit):
        """Append content-mode result groups for one file."""
        lines = text.splitlines()
        if multiline:
            hit_lines = sorted({text.count("\n", 0, m.start()) for m in regex.finditer(text)})
        else:
            hit_lines = [i for i, line in enumerate(lines) if regex.search(line)]
        for i in hit_lines:
            lo = max(0, i - context_lines)
            hi = min(len(lines), i + context_lines + 1)
            group = [
                f"{rel}:{n + 1}{':' if n == i else '-'} {lines[n]}"
                for n in range(lo, hi)
            ]
            results.append("\n".join(group))
            if len(results) >= limit:
                return

    @staticmethod
    def _summary(pattern, root, mode, results, truncated, scan_truncated,
                 skipped_binary, skipped_large, limit) -> str:
        """Build the model-facing markdown summary."""
        lines = [f"Searched {root} for /{pattern}/ — {len(results)} result(s)."]
        if not results:
            lines.append("No matches found.")
        elif mode == "files_with_matches":
            lines.append("")
            lines.extend(results)
        elif mode == "count":
            from plugins.frontends.helpers.formatters import md_table
            total = sum(n for _, n in results)
            lines.append("")
            lines.append(md_table(["file", "matches"], [[rel, str(n)] for rel, n in results]))
            lines.append(f"Total: {total} match(es).")
        else:
            lines.append("")
            lines.append("\n--\n".join(results))
        if truncated:
            lines.append(f"(showing first {limit}; more matches exist — narrow the pattern or raise limit)")
        if scan_truncated:
            lines.append("(file scan hit the enumeration cap — narrow 'path' or 'glob')")
        if skipped_binary or skipped_large:
            lines.append(f"(skipped {skipped_binary} binary and {skipped_large} oversized files)")
        text = "\n".join(lines)
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS] + "\n... (output capped at 20000 chars — narrow the search or lower limit)"
        return text
