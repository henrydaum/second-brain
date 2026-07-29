"""Live-tree content search — regex over files on disk.

Complements lexical_search (which queries the indexed corpus): grep reads the
actual files under the project root right now, so it sees uncommitted and
unindexed content.

Sandboxed, and most of it went away in the move. Walking the tree, pruning
junk directories, skipping binaries, compiling the regex and shelling out to
ripgrep all used to live here; they now live behind the ``fs.search`` Request,
because none of them can be done from inside a box — a guest-side search costs
one round trip per file, and ``shutil.which("rg")`` plus ``subprocess.run`` is
exactly the unmediated reach the sandbox exists to remove. What is left is
this tool's actual job: a parameter schema the model can use, and a summary it
can read.
"""

dependencies_files = []
dependencies_pip = []
requests = ["fs.search", "paths.get"]

from guest.bases import BaseTool

DEFAULT_LIMIT = 100
MAX_LIMIT = 500
MAX_CONTEXT = 10
MAX_CHARS = 20_000  # summary cap, mirrors read_file

# This tool's vocabulary, and the Request's. They differ because
# "files_with_matches" is grep's word and the model has been trained on it.
MODES = {"files_with_matches": "files", "content": "content", "count": "count"}


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

    def run(self, sdk, **kwargs):
        """Run grep."""
        pattern = (kwargs.get("pattern") or "").strip()
        if not pattern:
            return sdk.fail("No pattern provided.")

        mode = kwargs.get("output_mode") or "files_with_matches"
        if mode not in MODES:
            return sdk.fail(f"Unknown output_mode: {mode}")

        context_lines = _clamp(kwargs.get("context_lines"), 0, 0, MAX_CONTEXT)
        limit = _clamp(kwargs.get("limit"), DEFAULT_LIMIT, 1, MAX_LIMIT)
        project = sdk.paths.get("project")
        root = sdk.path.absolute((kwargs.get("path") or "").strip() or project,
                                 base=project)

        try:
            found = sdk.fs.search(
                pattern, root=root, glob=(kwargs.get("glob") or "").strip(),
                regex=True, mode=MODES[mode],
                case_insensitive=bool(kwargs.get("case_insensitive")),
                multiline=bool(kwargs.get("multiline")),
                context_lines=context_lines, limit=limit)
        except sdk.Denied as refused:
            return sdk.fail(str(refused))
        except sdk.Failed as failed:
            # A regex the engine will not compile arrives here, and it is the
            # model's to fix — so the reason travels back verbatim.
            return sdk.fail(failed.error)

        return sdk.ok(found, llm_summary=_summary(sdk, pattern, mode, found, limit))


def _clamp(raw, fallback, low, high):
    """A caller-supplied integer, bounded, tolerating anything unusable."""
    try:
        value = int(raw) if raw is not None else fallback
    except (TypeError, ValueError):
        value = fallback
    return max(low, min(value, high))


def _summary(sdk, pattern, mode, found, limit) -> str:
    """Build the model-facing markdown summary."""
    results = found.get("results") or []
    lines = [f"Searched {found.get('root')} for /{pattern}/ — {len(results)} result(s)."]
    if not results:
        lines.append("No matches found.")
    elif mode == "files_with_matches":
        lines.append("")
        lines.extend(results)
    elif mode == "count":
        total = sum(int(n) for _rel, n in results)
        # md.table opens with its own blank line, so the table starts its own
        # block and GFM parsers do not fold it into the sentence above.
        lines.append(sdk.md.table(["file", "matches"],
                                  [[rel, str(n)] for rel, n in results]))
        lines.append(f"Total: {total} match(es).")
    else:
        lines.append("")
        lines.append("\n--\n".join(results))

    if found.get("truncated"):
        lines.append(f"(showing first {limit}; more matches exist — narrow the pattern or raise limit)")
    if found.get("scan_truncated"):
        lines.append("(file scan hit the enumeration cap — narrow 'path' or 'glob')")
    skipped_binary = found.get("skipped_binary") or 0
    skipped_large = found.get("skipped_large") or 0
    if skipped_binary or skipped_large:
        lines.append(f"(skipped {skipped_binary} binary and {skipped_large} oversized files)")

    text = "\n".join(lines)
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + "\n... (output capped at 20000 chars — narrow the search or lower limit)"
    return text
