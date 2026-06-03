"""PDF parsers (PyMuPDF / ``fitz``).

Packaged separately from the kernel text parser because PyMuPDF is a heavy
binary dependency. Registers text, image, and tabular extraction for ``.pdf``.
The ``fitz`` import is lazy so the module degrades to "not installed" rather
than failing the parser-discovery scan when PyMuPDF is absent.
"""

import logging
import time
from pathlib import Path

from plugins.services.helpers.ParseResult import ParseResult
from plugins.services.helpers import parser_registry as registry
from plugins.services.helpers.parsing_utils import clean_text, max_chars

logger = logging.getLogger("ParsePDF")


def parse_pdf_text(path: str, config: dict, services: dict = None) -> ParseResult:
    """Extract text from a PDF. Detects embedded images and tables."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.debug("PyMuPDF not installed")
        return ParseResult.failed("PyMuPDF not installed", modality="text")

    try:
        t0 = time.time()
        limit = max_chars(config)

        with fitz.open(path) as doc:
            text_parts = []
            current_len = 0
            image_count = 0
            has_tables = False

            for page in doc:
                page_text = page.get_text()
                text_parts.append(page_text)
                current_len += len(page_text)

                # Image detection (cheap — reads page structure, not pixels)
                image_count += len(page.get_images(full=False))

                if not has_tables:
                    tables = page.find_tables()
                    if tables.tables:
                        has_tables = True

                if current_len > limit:
                    break

            text = clean_text("".join(text_parts)[:limit])

            also_contains = []
            if image_count > 0:
                also_contains.append("image")
            if has_tables:
                also_contains.append("tabular")

            # A PDF with almost no text but multiple pages is likely scanned.
            # Flag it so the orchestrator queues OCR via the image modality.
            is_scanned = len(text.strip()) < 50 and len(doc) > 0
            if is_scanned and "image" not in also_contains:
                also_contains.append("image")

            metadata = {
                "char_count": len(text),
                "page_count": len(doc),
                "image_count": image_count,
                "has_images": image_count > 0,
                "has_tables": has_tables,
                "is_scanned": is_scanned,
            }

        logger.debug(
            f"PDF parsed: {Path(path).name} — {metadata['page_count']} pages, "
            f"{len(text)} chars in {time.time() - t0:.2f}s"
        )
        return ParseResult(
            modality="text",
            output=text,
            metadata=metadata,
            also_contains=also_contains,
        )
    except Exception as e:
        logger.debug(f"Failed to parse {path}: {e}")
        return ParseResult.failed(str(e), modality="text")


registry.register(".pdf", "text", parse_pdf_text)


def parse_pdf_image(path: str, config: dict, services: dict = None) -> ParseResult:
    """Extract embedded images from a PDF as PIL.Image objects."""
    try:
        import fitz
        from PIL import Image
        import io
    except ImportError as e:
        logger.debug(f"Missing dependency: {e}")
        return ParseResult.failed(f"Missing dependency: {e}", modality="image")

    try:
        with fitz.open(path) as doc:
            images = []
            max_images = config.get("max_images", 50)

            for page in doc:
                if len(images) >= max_images:
                    break
                for img_info in page.get_images(full=True):
                    if len(images) >= max_images:
                        break
                    xref = img_info[0]
                    try:
                        base_image = doc.extract_image(xref)
                        img = Image.open(io.BytesIO(base_image["image"]))
                        images.append(img)
                    except Exception:
                        continue

        if not images:
            return ParseResult.failed(
                "No extractable images found in PDF", modality="image"
            )

        return ParseResult(
            modality="image",
            output=images,
            metadata={"image_count": len(images), "source_format": "pdf"},
        )
    except Exception as e:
        logger.debug(f"Failed to parse {path}: {e}")
        return ParseResult.failed(str(e), modality="image")


registry.register(".pdf", "image", parse_pdf_image)


def parse_pdf_tables(path: str, config: dict, services: dict = None) -> ParseResult:
    """Extract tables from a PDF as DataFrames."""
    try:
        import fitz
        import pandas as pd  # noqa: F401  (table.to_pandas needs pandas installed)
    except ImportError as e:
        logger.debug(f"Missing dependency: {e}")
        return ParseResult.failed(f"Missing dependency: {e}", modality="tabular")

    try:
        with fitz.open(path) as doc:
            all_tables = {}
            table_idx = 0

            for page_num, page in enumerate(doc):
                found = page.find_tables()
                for table in found.tables:
                    df = table.to_pandas()
                    if df.empty:
                        continue
                    key = f"page_{page_num + 1}_table_{table_idx + 1}"
                    all_tables[key] = df
                    table_idx += 1

        if not all_tables:
            return ParseResult.failed(
                "No extractable tables found in PDF", modality="tabular"
            )

        table_meta = {}
        total_rows = 0
        for name, df in all_tables.items():
            total_rows += len(df)
            table_meta[name] = {
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            }

        return ParseResult(
            modality="tabular",
            output=all_tables,
            metadata={
                "total_rows": total_rows,
                "table_count": len(all_tables),
                "table_names": list(all_tables.keys()),
                "tables": table_meta,
                "source_format": "pdf",
            },
        )
    except Exception as e:
        logger.debug(f"Failed to parse {path}: {e}")
        return ParseResult.failed(str(e), modality="tabular")


registry.register(".pdf", "tabular", parse_pdf_tables)
