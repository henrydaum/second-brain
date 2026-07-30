"""Attachment parsing helpers for container and archive inputs."""


dependencies_files = []
dependencies_pip = ['py7zr', 'rarfile']

import email
import hashlib
import tarfile
import time
import zipfile

from guest.parsing import ParseResult, basename, register

# zipfile/tarfile read and extract archives themselves, which cannot be turned
# into Requests — and an archive is untrusted input by definition. The process
# boundary is what actually contains a malicious or malformed one.
isolation = "subprocess"


# Returns a list of child paths (folders or files)

"""
Container parsers.

Returns ParseResult(modality="container", output=list[str]).

The output is a list of absolute paths to extracted child files.
These paths live under the app's data directory (DATA_DIR/extracted),
not in the user's original folder. The user's filesystem is never modified.

Extraction directory structure:
    <DATA_DIR>/extracted/<hash_of_archive_path>/
        ├── report.pdf
        ├── images/
        │   ├── photo1.jpg
        │   └── photo2.png
        └── data.csv

After extraction, the calling task (or orchestrator) feeds these paths
back into the crawler's register_paths() function, and they enter the
system as first-class files. Nested archives (a ZIP inside a ZIP) get
their own container task and extract recursively through the task system.

Security notes:
    - Zip bombs: extraction has a max total size limit.
    - Path traversal: all extracted paths are resolved and validated
      to stay within the extraction directory.
    - Symlinks: not followed during extraction.

Supports: ZIP, TAR, GZ, BZ2, 7Z (if py7zr installed), EML
"""


# Safety limits
MAX_EXTRACT_SIZE = 2 * 1024 * 1024 * 1024  # 2 GB total extracted
MAX_FILES = 10_000                           # max files per archive
# One extraction directory per archive, per process. The old version keyed a
# stable directory under DATA_DIR off the archive's hash so re-parsing reused
# it; sandboxed code cannot name a location outside its scratch space, so the
# cache lives here instead — same effect within a run, nothing left behind
# after one.
_EXTRACT_DIRS: dict = {}


def _extract_dir(sdk, archive_path: str) -> str:
    """
    Get a stable extraction directory for an archive.
    Same archive path always extracts to the same directory,
    so re-parsing doesn't create duplicates.
    """
    key = hashlib.md5(archive_path.encode()).hexdigest()[:12]
    if key not in _EXTRACT_DIRS:
        _EXTRACT_DIRS[key] = sdk.fs.temp(directory=True)
    return _EXTRACT_DIRS[key]


def _validate_path(member_path: str, dest: str) -> bool:
    """Ensure an extracted path stays within the destination directory.

    Zip-slip defence, done on the string rather than with ``realpath``: a
    parser has no business resolving paths on disk, and a member naming ``..``
    or an absolute location is rejected before anything is written.
    """
    member = (member_path or "").replace("\\", "/")
    if member.startswith("/") or ":" in member.split("/", 1)[0]:
        return False
    parts = [p for p in member.split("/") if p and p != "."]
    return ".." not in parts


def _collect_paths(sdk, dest: str) -> list[str]:
    """Every *file* the extraction produced, however deep.

    ``fs.list`` reports directories alongside files and there is no Request
    that tells them apart, so they are subtracted instead: anything that is a
    parent of another entry is a directory. That costs nothing extra, unlike
    probing each path. An extracted directory that is empty has no children to
    give it away and would survive — archives rarely carry one, and a caller
    that tries to parse it gets an ordinary failure.
    """
    entries = sdk.fs.list(dest, pattern="**/*")
    normalized = [e.replace("\\", "/") for e in entries]
    parents = {p.rsplit("/", 1)[0] for p in normalized if "/" in p}
    return [original for original, norm in zip(entries, normalized)
            if norm not in parents]


# ===================================================================
# ZIP
# ===================================================================

def parse_zip(sdk, path: str, config: dict = None) -> ParseResult:
    """Extract a ZIP archive and return child paths."""
    try:
        if not zipfile.is_zipfile(path):
            return ParseResult.failed("Not a valid ZIP file", modality="container")

        t0 = time.time()
        dest = _extract_dir(sdk, path)
        max_size = config.get("max_extract_size", MAX_EXTRACT_SIZE)
        max_files = config.get("max_files", MAX_FILES)
        total_size = 0
        file_count = 0

        with zipfile.ZipFile(path, "r") as zf:
            for info in zf.infolist():
                # Skip directories
                if info.is_dir():
                    continue

                # Path traversal check
                if not _validate_path(info.filename, dest):
                    sdk.log(f"Skipping suspicious path: {info.filename}", level="warning")
                    continue

                # Size limit
                total_size += info.file_size
                if total_size > max_size:
                    sdk.log(f"Extraction size limit reached for {path}", level="warning")
                    break

                # File count limit
                file_count += 1
                if file_count > max_files:
                    sdk.log(f"File count limit reached for {path}", level="warning")
                    break

                zf.extract(info, dest)

        children = _collect_paths(sdk, dest)
        sdk.log(
            f"ZIP extracted: {basename(path)} — {len(children)} files in {time.time() - t0:.2f}s"
        , level="debug")

        return ParseResult(
            modality="container",
            output=children,
            metadata={
                "archive_format": "zip",
                "file_count": len(children),
                "extract_dir": dest,
            },
        )
    except Exception as e:
        sdk.log(f"Failed to parse {path}: {e}", level="debug")
        return ParseResult.failed(str(e), modality="container")


register(".zip", "container", parse_zip)


# ===================================================================
# TAR (including .tar.gz, .tar.bz2)
# ===================================================================

def parse_tar(sdk, path: str, config: dict = None) -> ParseResult:
    """Extract a TAR archive (optionally compressed) and return child paths."""
    try:
        if not tarfile.is_tarfile(path):
            return ParseResult.failed("Not a valid TAR file", modality="container")

        dest = _extract_dir(sdk, path)
        max_size = config.get("max_extract_size", MAX_EXTRACT_SIZE)
        max_files = config.get("max_files", MAX_FILES)
        total_size = 0
        file_count = 0

        with tarfile.open(path, "r:*") as tf:
            for member in tf:
                # Skip directories and non-files (symlinks, devices, etc.)
                if not member.isfile():
                    continue

                # Path traversal check
                if not _validate_path(member.name, dest):
                    sdk.log(f"Skipping suspicious path: {member.name}", level="warning")
                    continue

                # Size limit
                total_size += member.size
                if total_size > max_size:
                    sdk.log(f"Extraction size limit reached for {path}", level="warning")
                    break

                # File count limit
                file_count += 1
                if file_count > max_files:
                    sdk.log(f"File count limit reached for {path}", level="warning")
                    break

                tf.extract(member, dest, set_attrs=False)

        children = _collect_paths(sdk, dest)

        return ParseResult(
            modality="container",
            output=children,
            metadata={
                "archive_format": "tar",
                "file_count": len(children),
                "extract_dir": dest,
            },
        )
    except Exception as e:
        sdk.log(f"Failed to parse {path}: {e}", level="debug")
        return ParseResult.failed(str(e), modality="container")


register([".tar", ".gz", ".bz2"], "container", parse_tar)


# ===================================================================
# 7Z
# ===================================================================

def parse_7z(sdk, path: str, config: dict = None) -> ParseResult:
    """Extract a 7-Zip archive and return child paths."""
    try:
        import py7zr
    except ImportError:
        sdk.log("py7zr not installed", level="debug")
        return ParseResult.failed("py7zr not installed", modality="container")

    try:
        dest = _extract_dir(sdk, path)

        with py7zr.SevenZipFile(path, mode="r") as archive:
            archive.extractall(path=dest)

        children = _collect_paths(sdk, dest)

        return ParseResult(
            modality="container",
            output=children,
            metadata={
                "archive_format": "7z",
                "file_count": len(children),
                "extract_dir": dest,
            },
        )
    except Exception as e:
        sdk.log(f"Failed to parse {path}: {e}", level="debug")
        return ParseResult.failed(str(e), modality="container")


register(".7z", "container", parse_7z)


# ===================================================================
# RAR
# ===================================================================

def parse_rar(sdk, path: str, config: dict = None) -> ParseResult:
    """Extract a RAR archive and return child paths."""
    try:
        import rarfile
    except ImportError:
        sdk.log("rarfile not installed", level="debug")
        return ParseResult.failed("rarfile not installed", modality="container")

    try:
        dest = _extract_dir(sdk, path)

        with rarfile.RarFile(path, "r") as rf:
            rf.extractall(dest)

        children = _collect_paths(sdk, dest)

        return ParseResult(
            modality="container",
            output=children,
            metadata={
                "archive_format": "rar",
                "file_count": len(children),
                "extract_dir": dest,
            },
        )
    except Exception as e:
        sdk.log(f"Failed to parse {path}: {e}", level="debug")
        return ParseResult.failed(str(e), modality="container")


register(".rar", "container", parse_rar)


# ===================================================================
# EML (Email)
#
# Emails are containers: a text body plus zero or more attachments.
# The body is extracted as a .txt file, attachments keep their
# original filenames. All land in the extraction directory.
# ===================================================================

def parse_eml(sdk, path: str, config: dict = None) -> ParseResult:
    """Extract an email's body and attachments as child files."""
    try:
        dest = _extract_dir(sdk, path)

        msg = email.message_from_string(sdk.fs.read(path))

        children = []
        skipped = []          # binary attachments; see the note below

        # Extract body
        body_parts = []
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                disposition = str(part.get("Content-Disposition", ""))

                # Text body (not an attachment)
                if content_type == "text/plain" and "attachment" not in disposition:
                    charset = part.get_content_charset() or "utf-8"
                    payload = part.get_payload(decode=True)
                    if payload:
                        body_parts.append(payload.decode(charset, errors="ignore"))

                # Attachments.
                #
                # LIMITATION: an attachment is arbitrary bytes and
                # ``sdk.fs.write`` takes text, so only ones that decode as
                # text are written out. Binary attachments are counted and
                # named in the metadata but not extracted — a ``fs.write_bytes``
                # Request would close this, and until then failing loudly in
                # the metadata beats writing a corrupted file.
                elif "attachment" in disposition or part.get_filename():
                    filename = basename(
                        part.get_filename() or f"attachment_{len(children)}")
                    if not _validate_path(filename, dest):
                        continue
                    payload = part.get_payload(decode=True)
                    if not payload:
                        continue
                    try:
                        text = payload.decode(
                            part.get_content_charset() or "utf-8")
                    except (UnicodeDecodeError, LookupError):
                        skipped.append(filename)
                        continue
                    filepath = f"{dest.rstrip('/')}/{filename}"
                    sdk.fs.write(filepath, text)
                    children.append(filepath)
        else:
            # Simple non-multipart email
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or "utf-8"
                body_parts.append(payload.decode(charset, errors="ignore"))

        # Save body as a text file
        if body_parts:
            body_path = f"{dest.rstrip('/')}/email_body.txt"
            sdk.fs.write(body_path, "\n\n".join(body_parts))
            children.insert(0, body_path)

        # Metadata from email headers
        metadata = {
            "archive_format": "eml",
            "file_count": len(children),
            "skipped_binary_attachments": skipped,
            "extract_dir": dest,
            "subject": msg.get("Subject", ""),
            "from": msg.get("From", ""),
            "to": msg.get("To", ""),
            "date": msg.get("Date", ""),
        }

        return ParseResult(
            modality="container",
            output=children,
            metadata=metadata,
        )
    except Exception as e:
        sdk.log(f"Failed to parse {path}: {e}", level="debug")
        return ParseResult.failed(str(e), modality="container")


register(".eml", "container", parse_eml)
