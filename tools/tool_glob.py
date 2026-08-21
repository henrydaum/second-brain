"""Find files by name pattern on disk.

Sandboxed. The walk, the junk-directory pruning, the symlink guard and the
newest-first ordering moved behind ``fs.list``'s recursive shape — they belong
to every caller, not to whichever tool wanted them first, and none of them can
be done from inside a box anyway.
"""

dependencies_files = []
dependencies_pip = []
requests = ["fs.list", "paths.get", "session.get"]

from guest.bases import BaseTool

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

    # This text reads the permission mode and nothing else, so it goes stale
    # only when the session does — not on every file the agent writes, which is
    # what the default rung would charge for it.
    agent_prompt_refresh = "session"

    def agent_prompt(self, sdk):
        """Point lockdown toward the mediated directory-inspection path."""
        if (sdk.session.get() or {}).get("mode") != "lockdown":
            return ""
        return (
            "## Inspecting directories in lockdown\n"
            "Use glob for directory discovery and filename inspection instead "
            "of shell traversal. Narrow `path` and `pattern` as you learn the "
            "tree."
        )

    def run(self, sdk, **kwargs):
        """Run glob."""
        pattern = (kwargs.get("pattern") or "").strip()
        if not pattern:
            return sdk.fail("No pattern provided.")

        try:
            limit = int(kwargs.get("limit") or DEFAULT_LIMIT)
        except (TypeError, ValueError):
            limit = DEFAULT_LIMIT
        limit = max(1, min(limit, MAX_LIMIT))

        project = sdk.paths.get("project")
        root = sdk.path.absolute((kwargs.get("path") or "").strip() or project,
                                 base=project)

        try:
            found = sdk.fs.list(root, pattern=pattern, recursive=True,
                                files_only=True, sort="mtime", limit=limit)
        except sdk.Denied as refused:
            return sdk.fail(str(refused))
        except sdk.Failed as failed:
            return sdk.fail(failed.error)

        # The Request answers in absolute paths; a listing reads better
        # relative to the root that was searched.
        entries = found.get("entries") or []
        rels = [_relative(sdk, entry, root) for entry in entries]

        lines = [f"Glob '{pattern}' under {root} — {len(rels)} file(s)."]
        if not rels:
            lines.append("No files matched.")
        else:
            lines.append("")
            lines.extend(rels)
        if found.get("truncated"):
            lines.append(f"(showing first {limit} newest files; more exist — narrow the pattern or raise limit)")
        if found.get("scan_truncated"):
            lines.append("(file scan hit the enumeration cap — narrow 'path')")

        return sdk.ok(
            {"root": root, "pattern": pattern, "results": rels,
             "truncated": bool(found.get("truncated")),
             "scan_truncated": bool(found.get("scan_truncated"))},
            llm_summary="\n".join(lines))


def _relative(sdk, path, root) -> str:
    """``path`` as a posix path under ``root``, or unchanged if it is not."""
    if not sdk.path.within(path, root):
        return str(path)
    trimmed = str(path)[len(str(root)):].lstrip("\\/")
    return trimmed.replace("\\", "/") or str(path)
