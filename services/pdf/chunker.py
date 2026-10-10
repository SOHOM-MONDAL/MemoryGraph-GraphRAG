"""Sentence- and paragraph-aware document chunker."""

import datetime
import re
from typing import List, Optional
from pydantic import BaseModel, Field

from services.pdf.parser import PDFPage


class PDFChunk(BaseModel):
    """Schema representing an individual document chunk."""

    chunk_id: str
    document_id: str
    user_id: str
    page_number: int
    chunk_index: int
    text: str
    embedding: Optional[List[float]] = None
    created_at: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences respecting punctuation boundaries."""
    if not text:
        return []
    # Match sentences ending with ., !, ? followed by whitespace or quotes
    sentence_end_pattern = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
    raw_sentences = sentence_end_pattern.split(text)
    cleaned = [s.strip() for s in raw_sentences if s.strip()]
    return cleaned if cleaned else [text.strip()]


def chunk_document(
    pages: List[PDFPage],
    document_id: str,
    user_id: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> List[PDFChunk]:
    """Chunk document pages while preserving page context and sentence boundaries.

    Args:
        pages: List of extracted PDFPage objects.
        document_id: ID of the parent document.
        user_id: User identifier for isolation.
        chunk_size: Target maximum characters per chunk.
        chunk_overlap: Target character overlap between consecutive chunks.

    Returns:
        List of PDFChunk objects.
    """
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 4)

    all_chunks: List[PDFChunk] = []
    global_chunk_idx = 0

    for page in pages:
        page_text = page.text.strip()
        if not page_text:
            continue

        paragraphs = [p.strip() for p in page_text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [page_text]

        current_sentences: List[str] = []
        current_len = 0

        for para in paragraphs:
            sentences = split_into_sentences(para)
            for sentence in sentences:
                sentence_len = len(sentence)

                # If single sentence exceeds chunk size, split it into slices
                if sentence_len > chunk_size:
                    if current_sentences:
                        chunk_text = " ".join(current_sentences).strip()
                        if chunk_text:
                            all_chunks.append(
                                PDFChunk(
                                    chunk_id=f"{document_id}_p{page.page_number}_c{global_chunk_idx}",
                                    document_id=document_id,
                                    user_id=user_id,
                                    page_number=page.page_number,
                                    chunk_index=global_chunk_idx,
                                    text=chunk_text,
                                )
                            )
                            global_chunk_idx += 1
                        current_sentences = []
                        current_len = 0

                    start = 0
                    while start < sentence_len:
                        end = min(start + chunk_size, sentence_len)
                        slice_text = sentence[start:end].strip()
                        if slice_text:
                            all_chunks.append(
                                PDFChunk(
                                    chunk_id=f"{document_id}_p{page.page_number}_c{global_chunk_idx}",
                                    document_id=document_id,
                                    user_id=user_id,
                                    page_number=page.page_number,
                                    chunk_index=global_chunk_idx,
                                    text=slice_text,
                                )
                            )
                            global_chunk_idx += 1
                        if end >= sentence_len:
                            break
                        start = max(start + 1, end - chunk_overlap)
                    continue

                if current_len + sentence_len + 1 > chunk_size and current_sentences:
                    chunk_text = " ".join(current_sentences).strip()
                    if chunk_text:
                        all_chunks.append(
                            PDFChunk(
                                chunk_id=f"{document_id}_p{page.page_number}_c{global_chunk_idx}",
                                document_id=document_id,
                                user_id=user_id,
                                page_number=page.page_number,
                                chunk_index=global_chunk_idx,
                                text=chunk_text,
                            )
                        )
                        global_chunk_idx += 1

                    # Compute overlap carryover
                    overlap_sentences: List[str] = []
                    overlap_len = 0
                    for s in reversed(current_sentences):
                        if overlap_len + len(s) + 1 <= chunk_overlap:
                            overlap_sentences.insert(0, s)
                            overlap_len += len(s) + 1
                        else:
                            break

                    current_sentences = overlap_sentences + [sentence]
                    current_len = sum(len(s) for s in current_sentences) + len(current_sentences) - 1
                else:
                    current_sentences.append(sentence)
                    current_len += sentence_len + 1

        # Emit any trailing sentences for the page
        if current_sentences:
            chunk_text = " ".join(current_sentences).strip()
            if chunk_text:
                all_chunks.append(
                    PDFChunk(
                        chunk_id=f"{document_id}_p{page.page_number}_c{global_chunk_idx}",
                        document_id=document_id,
                        user_id=user_id,
                        page_number=page.page_number,
                        chunk_index=global_chunk_idx,
                        text=chunk_text,
                    )
                )
                global_chunk_idx += 1

    return all_chunks
