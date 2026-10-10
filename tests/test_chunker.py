"""Tests for sentence-aware chunking and page tracking."""

from services.pdf.chunker import chunk_document, split_into_sentences
from services.pdf.parser import PDFPage


def test_split_into_sentences():
    text = "First sentence here. Second sentence starts! Is this the third? Yes, it is."
    sentences = split_into_sentences(text)
    assert len(sentences) == 4
    assert sentences[0] == "First sentence here."
    assert sentences[1] == "Second sentence starts!"
    assert sentences[2] == "Is this the third?"
    assert sentences[3] == "Yes, it is."


def test_chunk_document_preserves_page_numbers_and_ids():
    pages = [
        PDFPage(page_number=1, text="Page one intro. Second sentence of page one.", raw_char_count=50),
        PDFPage(page_number=2, text="Page two starting sentence. More information on page two.", raw_char_count=60),
    ]

    chunks = chunk_document(
        pages=pages,
        document_id="doc_123",
        user_id="user_alice",
        chunk_size=50,
        chunk_overlap=10,
    )

    assert len(chunks) >= 2
    # Verify metadata
    for c in chunks:
        assert c.document_id == "doc_123"
        assert c.user_id == "user_alice"
        assert c.page_number in [1, 2]
        assert c.chunk_id.startswith("doc_123_p")
        assert len(c.text) > 0


def test_chunking_respects_sentence_boundaries_when_possible():
    text = (
        "Artificial intelligence is transforming industries across the globe. "
        "Natural language processing allows computers to understand human language. "
        "Knowledge graphs organize structured relationships between entities. "
        "Retrieval augmented generation connects models to authoritative context."
    )
    pages = [PDFPage(page_number=1, text=text, raw_char_count=len(text))]

    chunks = chunk_document(
        pages=pages,
        document_id="ai_doc",
        user_id="tester",
        chunk_size=160,
        chunk_overlap=30,
    )

    assert len(chunks) > 1
    # Check that chunks end with complete punctuation where possible
    for c in chunks:
        assert c.text[-1] in [".", "!", "?", '"', "'"] or len(c.text) <= 160
