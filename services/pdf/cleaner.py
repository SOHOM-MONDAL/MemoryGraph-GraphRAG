"""Text cleaning utilities for parsed documents."""

import re


def clean_text(text: str) -> str:
    """Clean extracted document text.

    - Strips non-printable and null characters
    - Fixes hyphenated line breaks (e.g., 'trans-\nfer' -> 'transfer')
    - Replaces odd unicode whitespace and tabs with standard space
    - Preserves paragraph separations while collapsing extraneous newlines
    """
    if not text:
        return ""

    # Remove null bytes and control chars except newlines and tabs
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

    # Rejoin words broken across line breaks with hyphens
    cleaned = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", cleaned)

    # Standardize CRLF to LF
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

    # Replace runs of horizontal whitespace with single space
    cleaned = re.sub(r"[ \t]+", " ", cleaned)

    # Collapse runs of 3+ newlines to 2 (paragraph break)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    # Trim leading/trailing whitespace per line
    lines = [line.strip() for line in cleaned.split("\n")]
    cleaned = "\n".join(lines).strip()

    return cleaned
