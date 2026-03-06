import logging
import email
import hashlib
import os
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path
from Stage_1.ParseResult import ParseResult
import Stage_1.registry as registry

logger = logging.getLogger("ParseContainer")

# Returns a list of child paths (folders or files)

"""
Container parsers.

Returns ParseResult(modality="container", output=list[str]).

The output is a list of absolute paths to extracted child files.
These paths live in a temp directory managed by the system, not
in the user's original folder. The user's filesystem is never modified.

Extraction directory structure:
    /tmp/DataRefinery/extracted/<hash_of_archive_path>/
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
EXTRACT_BASE = os.path.join(tempfile.gettempdir(), "DataRefinery", "extracted")


def _extract_dir(archive_path: str) -> str:
    """
    Get a stable extraction directory for an archive.
    Same archive path always extracts to the same directory,
    so re-parsing doesn't create duplicates.
    """
    path_hash = hashlib.md5(archive_path.encode()).hexdigest()[:12]
    dest = os.path.join(EXTRACT_BASE, path_hash)
    os.makedirs(dest, exist_ok=True)
    return dest


def _validate_path(member_path: str, dest: str) -> bool:
    """Ensure an extracted path stays within the destination directory."""
    resolved = os.path.realpath(os.path.join(dest, member_path))
    return resolved.startswith(os.path.realpath(dest))


def _collect_paths(dest: str) -> list[str]:
    """Walk the extraction directory and return all file paths."""
    paths = []
    for root, _, files in os.walk(dest):
        for f in files:
            paths.append(os.path.join(root, f))
    return paths


# ===================================================================
# ZIP
# ===================================================================

def parse_zip(path: str, config: dict, services: dict = None) -> ParseResult:
    """Extract a ZIP archive and return child paths."""
    try:
        if not zipfile.is_zipfile(path):
            return ParseResult.failed("Not a valid ZIP file", modality="container")

        dest = _extract_dir(path)
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
                    logger.warning(f"Skipping suspicious path: {info.filename}")
                    continue

                # Size limit
                total_size += info.file_size
                if total_size > max_size:
                    logger.warning(f"Extraction size limit reached for {path}")
                    break

                # File count limit
                file_count += 1
                if file_count > max_files:
                    logger.warning(f"File count limit reached for {path}")
                    break

                zf.extract(info, dest)

        children = _collect_paths(dest)

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
        logger.debug(f"Failed to parse {path}: {e}")
        return ParseResult.failed(str(e), modality="container")


registry.register(".zip", "container", parse_zip)


# ===================================================================
# TAR (including .tar.gz, .tar.bz2)
# ===================================================================

def parse_tar(path: str, config: dict, services: dict = None) -> ParseResult:
    """Extract a TAR archive (optionally compressed) and return child paths."""
    try:
        if not tarfile.is_tarfile(path):
            return ParseResult.failed("Not a valid TAR file", modality="container")

        dest = _extract_dir(path)
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
                    logger.warning(f"Skipping suspicious path: {member.name}")
                    continue

                # Size limit
                total_size += member.size
                if total_size > max_size:
                    logger.warning(f"Extraction size limit reached for {path}")
                    break

                # File count limit
                file_count += 1
                if file_count > max_files:
                    logger.warning(f"File count limit reached for {path}")
                    break

                tf.extract(member, dest, set_attrs=False)

        children = _collect_paths(dest)

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
        logger.debug(f"Failed to parse {path}: {e}")
        return ParseResult.failed(str(e), modality="container")


registry.register([".tar", ".gz", ".bz2"], "container", parse_tar)


# ===================================================================
# 7Z
# ===================================================================

def parse_7z(path: str, config: dict, services: dict = None) -> ParseResult:
    """Extract a 7-Zip archive and return child paths."""
    try:
        import py7zr
    except ImportError:
        logger.debug("py7zr not installed")
        return ParseResult.failed("py7zr not installed", modality="container")

    try:
        dest = _extract_dir(path)

        with py7zr.SevenZipFile(path, mode="r") as archive:
            archive.extractall(path=dest)

        children = _collect_paths(dest)

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
        logger.debug(f"Failed to parse {path}: {e}")
        return ParseResult.failed(str(e), modality="container")


registry.register(".7z", "container", parse_7z)


# ===================================================================
# RAR
# ===================================================================

def parse_rar(path: str, config: dict, services: dict = None) -> ParseResult:
    """Extract a RAR archive and return child paths."""
    try:
        import rarfile
    except ImportError:
        logger.debug("rarfile not installed")
        return ParseResult.failed("rarfile not installed", modality="container")

    try:
        dest = _extract_dir(path)

        with rarfile.RarFile(path, "r") as rf:
            rf.extractall(dest)

        children = _collect_paths(dest)

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
        logger.debug(f"Failed to parse {path}: {e}")
        return ParseResult.failed(str(e), modality="container")


registry.register(".rar", "container", parse_rar)


# ===================================================================
# EML (Email)
#
# Emails are containers: a text body plus zero or more attachments.
# The body is extracted as a .txt file, attachments keep their
# original filenames. All land in the extraction directory.
# ===================================================================

def parse_eml(path: str, config: dict, services: dict = None) -> ParseResult:
    """Extract an email's body and attachments as child files."""
    try:
        dest = _extract_dir(path)

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            msg = email.message_from_file(f)

        children = []

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

                # Attachments
                elif "attachment" in disposition or part.get_filename():
                    filename = part.get_filename() or f"attachment_{len(children)}"
                    # Sanitize filename
                    filename = Path(filename).name
                    filepath = os.path.join(dest, filename)
                    payload = part.get_payload(decode=True)
                    if payload:
                        with open(filepath, "wb") as out:
                            out.write(payload)
                        children.append(filepath)
        else:
            # Simple non-multipart email
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or "utf-8"
                body_parts.append(payload.decode(charset, errors="ignore"))

        # Save body as a text file
        if body_parts:
            body_path = os.path.join(dest, "email_body.txt")
            with open(body_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(body_parts))
            children.insert(0, body_path)

        # Metadata from email headers
        metadata = {
            "archive_format": "eml",
            "file_count": len(children),
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
        logger.debug(f"Failed to parse {path}: {e}")
        return ParseResult.failed(str(e), modality="container")


registry.register(".eml", "container", parse_eml)