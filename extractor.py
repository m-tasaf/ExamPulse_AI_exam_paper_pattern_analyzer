"""File inspection and MIME detection helpers for past-year exam question papers.

Supports:
- PDFs (native digital text and scanned/diagram pages)
- Images (JPEG, PNG, WEBP) for handwritten/photographed question papers or diagrams
- Plain text (.txt)
"""

from __future__ import annotations

import io
import re
from typing import BinaryIO

from pypdf import PdfReader

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MIME_MAP = {
    ".pdf": "application/pdf",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".txt": "text/plain",
}


def get_file_mime_type(filename: str) -> str:
    """Return appropriate MIME type based on file extension."""
    lower = filename.lower()
    for ext, mime in MIME_MAP.items():
        if lower.endswith(ext):
            return mime
    return "application/octet-stream"


def is_image_file(filename: str) -> bool:
    """Check if the filename corresponds to a supported image format."""
    lower = filename.lower()
    return any(lower.endswith(ext) for ext in IMAGE_EXTENSIONS)


def is_pdf_file(filename: str) -> bool:
    """Check if filename is a PDF."""
    return filename.lower().endswith(".pdf")


def extract_text_from_pdf(file_obj: bytes | BinaryIO) -> str:
    """Extract readable text from a PDF if present (best-effort).

    Args:
        file_obj: Raw PDF bytes or a binary file-like object.

    Returns:
        Extracted text joined across all pages, or empty string if scanned/image-only.
    """
    stream = io.BytesIO(file_obj) if isinstance(file_obj, (bytes, bytearray)) else file_obj
    try:
        reader = PdfReader(stream)
    except Exception:
        return ""

    pages_text: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        cleaned = text.strip()
        if cleaned:
            pages_text.append(cleaned)
    return "\n\n".join(pages_text).strip()


def infer_year_from_filename(filename: str, default_year: int = 2024) -> int:
    """Infer a 4-digit exam year (2000-2035) from a filename, or return default_year."""
    match = re.search(r"\b(20[0-3]\d)\b", filename)
    if match:
        return int(match.group(1))
    return default_year
