"""Knowledge Graph entity and relationship extractor."""

import json
import logging
import re
from typing import List, Optional

from services.graph.models import GraphTriplet
from services.llm.base import LLMProvider

logger = logging.getLogger("memorygraph.graph.extractor")

EXTRACTION_SYSTEM_PROMPT = (
    "You are an expert Knowledge Graph Information Extraction system. "
    "Given a passage of text, extract factual relationships as a JSON array of triplets. "
    "Each triplet must have exact keys: 'subject', 'predicate', 'object'. "
    "Keep subjects and objects concise (1-4 words, entities/concepts/tools). "
    "Keep predicates concise (e.g. 'uses', 'consists of', 'processes', 'stores', 'connects to', 'implements'). "
    "Output ONLY the valid JSON array without any markdown formatting or extra text."
)

# Common relational indicator patterns for fast heuristic / offline fallback
RELATION_PATTERNS = [
    # X is / are / was Y
    (re.compile(r"([A-Z][A-Za-z0-9_-]+(?:\s+[A-Za-z0-9_-]+){0,3})\s+(?:is|are|was|were)\s+(?:an?|the)?\s*([A-Za-z0-9_-]+(?:\s+[A-Za-z0-9_-]+){0,3})", re.IGNORECASE), "is a"),
    # X uses / utilizes / employs Y
    (re.compile(r"([A-Z][A-Za-z0-9_-]+(?:\s+[A-Za-z0-9_-]+){0,2})\s+(?:uses|utilizes|employs|supports)\s+([A-Za-z0-9_-]+(?:\s+[A-Za-z0-9_-]+){0,3})", re.IGNORECASE), "uses"),
    # X connects to / communicates with Y
    (re.compile(r"([A-Z][A-Za-z0-9_-]+(?:\s+[A-Za-z0-9_-]+){0,2})\s+(?:connects to|communicates with|publishes to)\s+([A-Za-z0-9_-]+(?:\s+[A-Za-z0-9_-]+){0,3})", re.IGNORECASE), "connects to"),
    # X stores / persists / caches Y
    (re.compile(r"([A-Z][A-Za-z0-9_-]+(?:\s+[A-Za-z0-9_-]+){0,2})\s+(?:stores|persists|caches|maintains)\s+([A-Za-z0-9_-]+(?:\s+[A-Za-z0-9_-]+){0,3})", re.IGNORECASE), "stores"),
    # X extracts / processes / parses Y
    (re.compile(r"([A-Z][A-Za-z0-9_-]+(?:\s+[A-Za-z0-9_-]+){0,2})\s+(?:extracts|processes|parses|chunks)\s+([A-Za-z0-9_-]+(?:\s+[A-Za-z0-9_-]+){0,3})", re.IGNORECASE), "processes"),
    # X provides / generates / returns Y
    (re.compile(r"([A-Z][A-Za-z0-9_-]+(?:\s+[A-Za-z0-9_-]+){0,2})\s+(?:provides|generates|returns|produces)\s+([A-Za-z0-9_-]+(?:\s+[A-Za-z0-9_-]+){0,3})", re.IGNORECASE), "provides"),
]


def clean_entity(text: str) -> str:
    """Normalize extracted entity string."""
    cleaned = re.sub(r"[^\w\s-]", "", text).strip()
    return cleaned.title() if len(cleaned) > 1 else cleaned


def fallback_heuristic_extraction(
    text: str,
    document_id: str,
    chunk_id: Optional[str] = None,
) -> List[GraphTriplet]:
    """Extract triplets using linguistic patterns and named entity candidates when LLM is unavailable."""
    triplets: List[GraphTriplet] = []
    seen = set()

    sentences = re.split(r"(?<=[.!?])\s+", text)
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        for pattern, default_pred in RELATION_PATTERNS:
            match = pattern.search(sent)
            if match:
                subj = clean_entity(match.group(1))
                obj = clean_entity(match.group(2))
                if subj and obj and subj.lower() != obj.lower() and len(subj) >= 2 and len(obj) >= 2:
                    key = (subj.lower(), default_pred.lower(), obj.lower())
                    if key not in seen:
                        seen.add(key)
                        triplets.append(
                            GraphTriplet(
                                subject=subj,
                                predicate=default_pred,
                                object=obj,
                                document_id=document_id,
                                chunk_id=chunk_id,
                                confidence=0.8,
                            )
                        )

    # If pattern matching didn't yield triplets, extract co-occurring technical/capitalized terms
    if not triplets:
        words = re.findall(r"\b[A-Z][A-Za-z0-9_]{2,}\b", text)
        unique_terms = []
        for w in words:
            cleaned = clean_entity(w)
            if cleaned and cleaned not in unique_terms:
                unique_terms.append(cleaned)

        for i in range(len(unique_terms) - 1):
            s = unique_terms[i]
            o = unique_terms[i + 1]
            if s.lower() != o.lower():
                triplets.append(
                    GraphTriplet(
                        subject=s,
                        predicate="related to",
                        object=o,
                        document_id=document_id,
                        chunk_id=chunk_id,
                        confidence=0.6,
                    )
                )

    return triplets[:8]  # Limit to most prominent per chunk


def extract_triplets(
    text: str,
    document_id: str,
    chunk_id: Optional[str] = None,
    llm: Optional[LLMProvider] = None,
) -> List[GraphTriplet]:
    """Extract entity-relation triplets from document chunk text using LLM or fallback heuristics.

    Args:
        text: Raw text of document chunk.
        document_id: Source document ID.
        chunk_id: Optional source chunk identifier.
        llm: Optional LLMProvider to execute structured extraction.

    Returns:
        List of extracted GraphTriplet instances.
    """
    if not text or not text.strip():
        return []

    # Attempt LLM extraction first if provider is passed and responsive
    if llm and hasattr(llm, "health") and llm.health():
        prompt = f"Text Passage:\n\"\"\"{text[:1000]}\"\"\"\n\nExtract all entity-relation triplets as a JSON array:"
        try:
            raw_output = llm.generate(prompt=prompt, system=EXTRACTION_SYSTEM_PROMPT)
            # Find JSON array in output
            json_match = re.search(r"\[.*\]", raw_output, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                triplets = []
                for item in parsed:
                    if isinstance(item, dict) and "subject" in item and "predicate" in item and "object" in item:
                        s = clean_entity(str(item["subject"]))
                        p = str(item["predicate"]).strip().lower()
                        o = clean_entity(str(item["object"]))
                        if s and p and o:
                            triplets.append(
                                GraphTriplet(
                                    subject=s,
                                    predicate=p,
                                    object=o,
                                    document_id=document_id,
                                    chunk_id=chunk_id,
                                    confidence=0.95,
                                )
                            )
                if triplets:
                    return triplets
        except Exception as e:
            logger.debug("LLM triplet extraction skipped or failed: %s. Using heuristic fallback.", str(e))

    # Fallback to linguistic and heuristic extractor
    return fallback_heuristic_extraction(text=text, document_id=document_id, chunk_id=chunk_id)
