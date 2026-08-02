"""Extract archives and feed their child files back into the pipeline.

Container parsing is crossable: the parser performs extraction in its own
sandbox and returns only the resulting child paths.  Each successful per-path
result exposes those paths as ``discovered_paths``, so the orchestrator
registers and processes them normally, including nested archives.
"""

dependencies_files = ["parsers/parse_container.py"]
dependencies_pip = []

import time

from guest.bases import BaseTask
from guest.parsing import basename, suffix


_FORMATS = {
    ".zip": "zip",
    ".tar": "tar",
    ".gz": "tar",
    ".bz2": "tar",
    ".7z": "7z",
    ".rar": "rar",
    ".eml": "eml",
}


class ExtractContainer(BaseTask):
    """Extract container files and discover their contents."""

    name = "extract_container"
    modalities = ["container"]
    reads = []
    writes = ["extracted_containers"]
    requires_services = []
    requests = ["parse.file"]
    output_schema = """
        CREATE TABLE IF NOT EXISTS extracted_containers (
            path TEXT PRIMARY KEY,
            archive_format TEXT,
            file_count INTEGER,
            extract_dir TEXT,
            extracted_at REAL
        );
    """
    batch_size = 2
    max_workers = 2
    timeout = 300

    def run(self, sdk, paths):
        """Extract each archive without failing the rest of the batch."""
        now = time.time()
        outcomes = []

        for path in paths:
            try:
                children = list(sdk.parse.file(path, "container") or [])
            except sdk.Failed as failed:
                sdk.log(
                    f"container parse failed for {basename(path)}: {failed}",
                    level="warning",
                )
                outcomes.append({"ok": False, "error": f"parse failed: {failed}"})
                continue

            archive_format = _FORMATS.get(suffix(path), suffix(path).lstrip("."))
            extract_dir = _common_parent(children)
            sdk.log(
                f"extracted {len(children)} files from {basename(path)} "
                f"({archive_format or 'unknown'} archive)"
            )
            outcomes.append({
                "ok": True,
                "data": [{
                    "path": path,
                    "archive_format": archive_format,
                    "file_count": len(children),
                    "extract_dir": extract_dir,
                    "extracted_at": now,
                }],
                "discovered_paths": children,
            })

        return sdk.ok(per_path=outcomes)


def _common_parent(paths):
    """Best shared parent for display and bookkeeping, using path strings only."""
    normalized = [str(path).replace("\\", "/").rstrip("/") for path in paths]
    if not normalized:
        return ""
    parents = [path.rsplit("/", 1)[0] if "/" in path else "" for path in normalized]
    common = parents[0].split("/")
    for parent in parents[1:]:
        parts = parent.split("/")
        matched = 0
        for left, right in zip(common, parts):
            if left != right:
                break
            matched += 1
        common = common[:matched]
        if not common:
            return ""
    return "/".join(common)
