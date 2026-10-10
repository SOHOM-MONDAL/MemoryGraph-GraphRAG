"""PDF parsing module using PyMuPDF (fitz) with optional OCR fallback."""

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union
import pymupdf as fitz  # Modern PyMuPDF import

from services.pdf.cleaner import clean_text

logger = logging.getLogger(__name__)

PDF_MAGIC_BYTES = b"%PDF-"


class PDFParseError(Exception):
    """Raised when PDF validation or parsing fails."""
    pass


@dataclass
class PDFPage:
    """Represents an extracted page from a PDF."""
    page_number: int  # 1-indexed
    text: str
    raw_char_count: int = 0


def validate_pdf_bytes(content: bytes, max_mb: int = 25) -> None:
    """Validate PDF header magic bytes and size constraints."""
    if not content:
        raise PDFParseError("Uploaded file is empty.")

    max_bytes = max_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise PDFParseError(f"File size exceeds maximum allowed limit of {max_mb} MB.")

    if not content.startswith(PDF_MAGIC_BYTES):
        raise PDFParseError("Invalid file: Missing '%PDF-' magic bytes header.")


def parse_pdf(
    source: Union[bytes, str, Path],
    max_mb: int = 25,
) -> List[PDFPage]:
    """Parse PDF page by page using PyMuPDF.

    Args:
        source: Raw PDF bytes or file path.
        max_mb: Maximum allowed file size in MB.

    Returns:
        List of parsed PDFPage objects with cleaned text.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise PDFParseError(f"File not found: {path}")
        content = path.read_bytes()
    else:
        content = source

    validate_pdf_bytes(content, max_mb=max_mb)

    pages: List[PDFPage] = []

    try:
        doc = fitz.open(stream=content, filetype="pdf")
    except Exception as e:
        raise PDFParseError(f"PyMuPDF failed to open PDF document: {str(e)}") from e

    try:
        for page_idx in range(len(doc)):
            page_num = page_idx + 1
            page = doc[page_idx]
            extracted_text = page.get_text("text") or ""

            # If extracted text is minimal or empty, attempt OCR fallback if pytesseract is installed
            if len(extracted_text.strip()) < 10:
                extracted_text = _try_ocr_fallback(page, default=extracted_text)

            cleaned = clean_text(extracted_text)
            pages.append(
                PDFPage(
                    page_number=page_num,
                    text=cleaned,
                    raw_char_count=len(extracted_text),
                )
            )
    finally:
        doc.close()

    return pages


def _try_ocr_fallback(page: fitz.Page, default: str) -> str:
    """Attempt pytesseract OCR fallback if page has no extracted text."""
    try:
        import pytesseract
        from PIL import Image

        pix = page.get_pixmap(dpi=150)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        ocr_text = pytesseract.image_to_string(img)
        if ocr_text and ocr_text.strip():
            logger.info("OCR successfully extracted %d characters from page %d", len(ocr_text), page.number + 1)
            return ocr_text
    except Exception as e:
        logger.debug("OCR fallback not available or failed for page %d: %s", page.number + 1, str(e))

    return default
