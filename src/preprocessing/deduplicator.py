"""Content deduplication module across scraped page chunks."""

import logging
import re
from typing import List, Set
from src.models import PageChunk

logger = logging.getLogger(__name__)


def _normalize_block(block: str) -> str:
    """Normalize a text block into a canonical string for fingerprinting."""
    lowered = block.lower()
    cleaned = re.sub(r"[^\w\s]", "", lowered)
    normalized = re.sub(r"\s+", " ", cleaned).strip()
    return normalized


def deduplicate_chunks(chunks: List[PageChunk]) -> List[PageChunk]:
    """
    Deduplicate redundant text blocks across multiple PageChunks deterministically.

    Preserves first occurrence of each unique text block and retains chunk source URLs.
    """
    seen_fingerprints: Set[str] = set()
    deduplicated_chunks: List[PageChunk] = []

    for chunk in chunks:
        if not chunk.cleaned_content:
            continue

        # Split content into paragraphs/blocks by line breaks
        raw_blocks = chunk.cleaned_content.split("\n")
        unique_blocks: List[str] = []

        for block in raw_blocks:
            stripped = block.strip()
            if not stripped:
                unique_blocks.append("")
                continue

            fingerprint = _normalize_block(stripped)

            # Skip fingerprint deduplication if block is a short markdown header, but deduplicate longer duplicate text blocks
            if fingerprint and len(fingerprint) > 10 and not stripped.startswith("#"):
                if fingerprint in seen_fingerprints:
                    continue
                seen_fingerprints.add(fingerprint)

            unique_blocks.append(stripped)

        # Reassemble cleaned content
        deduped_content = "\n".join(unique_blocks)
        # Collapse multiple consecutive newlines
        deduped_content = re.sub(r"\n{3,}", "\n\n", deduped_content).strip()

        if deduped_content:
            deduplicated_chunks.append(
                PageChunk(
                    source_url=chunk.source_url,
                    cleaned_content=deduped_content,
                    section_title=chunk.section_title,
                    token_count=chunk.token_count,
                )
            )

    return deduplicated_chunks
