"""PDF parsing service for page-level text extraction."""

import logging
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def extract_pages(file_path: str) -> list[dict[str, Any]]:
    """Extract text content from each page of a PDF file.

    Args:
        file_path: Absolute path to the PDF file on disk.

    Returns:
        A list of dicts, each containing ``page_number`` (1-indexed)
        and the extracted ``text`` for that page.

    Raises:
        ValueError: If the PDF is corrupt, has zero pages, or contains
            no extractable text across all pages.
    """
    path = Path(file_path)
    if not path.exists():
        raise ValueError(f"File not found: {file_path}")

    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        logger.error("Failed to open PDF: %s — %s", file_path, exc)
        raise ValueError(f"Corrupt or unreadable PDF: {path.name}") from exc

    if doc.page_count == 0:
        doc.close()
        raise ValueError(f"PDF has zero pages: {path.name}")

    pages: list[dict[str, Any]] = []
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text = page.get_text("text").strip()
        pages.append({
            "page_number": page_num + 1,
            "text": text,
        })

    doc.close()

    # Check if any page yielded text
    has_text = any(p["text"] for p in pages)
    if not has_text:
        logger.warning("No extractable text found in %s", path.name)
        raise ValueError(
            f"No extractable text found in PDF: {path.name}. "
            "The file may be image-based or empty."
        )

    logger.info(
        "Extracted %d page(s) from %s",
        len(pages),
        path.name,
    )
    return pages
