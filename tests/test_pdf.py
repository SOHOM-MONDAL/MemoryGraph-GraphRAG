"""Tests for PDF parsing, validation, and text cleaning."""

import pytest
import fitz  # PyMuPDF

from services.pdf.cleaner import clean_text
from services.pdf.parser import PDFParseError, parse_pdf, validate_pdf_bytes


def create_sample_pdf(pages_text: list[str]) -> bytes:
    """Helper to create an in-memory valid PDF using PyMuPDF."""
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        # Insert text at top-left
        page.insert_text(fitz.Point(50, 72), text, fontsize=11)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_clean_text_normalizes_whitespace_and_hyphenation():
    dirty_text = "This is a demon-\nstration of hy-\nphenated words and   multiple   spaces.\n\n\n\nNew paragraph."
    cleaned = clean_text(dirty_text)
    assert "demonstration" in cleaned
    assert "hyphenated" in cleaned
    assert "multiple spaces." in cleaned
    assert "New paragraph." in cleaned
    assert "\n\n\n\n" not in cleaned


def test_validate_pdf_bytes_success():
    valid_pdf = create_sample_pdf(["Hello world"])
    # Should not raise
    validate_pdf_bytes(valid_pdf, max_mb=5)


def test_validate_pdf_bytes_rejects_empty():
    with pytest.raises(PDFParseError, match="empty"):
        validate_pdf_bytes(b"", max_mb=5)


def test_validate_pdf_bytes_rejects_non_pdf():
    with pytest.raises(PDFParseError, match="magic bytes"):
        validate_pdf_bytes(b"NOT_A_PDF_FILE_HEADER", max_mb=5)


def test_validate_pdf_bytes_rejects_oversized():
    fake_header = b"%PDF-1.4" + b"A" * 1024 * 1024 * 2
    with pytest.raises(PDFParseError, match="maximum allowed limit"):
        validate_pdf_bytes(fake_header, max_mb=1)


def test_parse_pdf_extracts_all_pages():
    pages_content = [
        "First page content with important information.",
        "Second page content with additional details.",
    ]
    pdf_bytes = create_sample_pdf(pages_content)
    parsed_pages = parse_pdf(pdf_bytes)

    assert len(parsed_pages) == 2
    assert parsed_pages[0].page_number == 1
    assert "First page" in parsed_pages[0].text
    assert parsed_pages[1].page_number == 2
    assert "Second page" in parsed_pages[1].text
