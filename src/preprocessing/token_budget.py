"""Relevance-aware token budget management module using tiktoken."""

import logging
import re
from typing import List, Tuple, Optional
import tiktoken

from config import get_settings
from src.models import PageChunk

logger = logging.getLogger(__name__)

# Keywords for block and chunk relevance scoring
RELEVANCE_KEYWORDS = [
    "company",
    "about",
    "product",
    "products",
    "solution",
    "solutions",
    "customer",
    "customers",
    "audience",
    "team",
    "leadership",
    "founder",
    "founders",
    "ceo",
    "contact",
    "sales",
    "support",
    "pricing",
]

# Encoder singleton
_ENCODER = None


def get_token_encoder():
    """Return tiktoken encoder instance."""
    global _ENCODER
    if _ENCODER is None:
        try:
            _ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _ENCODER = tiktoken.encoding_for_model("gpt-4o-mini")
    return _ENCODER


def count_tokens(text: str) -> int:
    """Calculate exact token count of text using tiktoken."""
    if not text:
        return 0
    try:
        encoder = get_token_encoder()
        return len(encoder.encode(text))
    except Exception:
        # Fallback estimation (~4 chars per token)
        return max(1, len(text) // 4)


def score_content_block(text: str, source_url: str) -> float:
    """
    Calculate relevance score for a text block based on keyword signals and source URL.
    """
    score = 0.0
    text_lower = text.lower()
    url_lower = source_url.lower()

    # URL path score
    for kw in RELEVANCE_KEYWORDS:
        if kw in url_lower:
            score += 3.0

    # Header score (lines starting with #)
    for line in text_lower.splitlines():
        if line.startswith("#"):
            for kw in RELEVANCE_KEYWORDS:
                if kw in line:
                    score += 5.0

    # Text keyword density score
    for kw in RELEVANCE_KEYWORDS:
        matches = len(re.findall(r"\b" + re.escape(kw) + r"\b", text_lower))
        score += matches * 1.5

    return score


def enforce_token_budget(
    chunks: List[PageChunk], max_tokens: Optional[int] = None
) -> Tuple[List[PageChunk], int]:
    """
    Prioritize high-signal page content blocks and cap total tokens strictly under max_tokens.

    Returns:
        Tuple[List[PageChunk], int]: Budget-compliant chunks and total token count.
    """
    settings = get_settings()
    budget = max_tokens or settings.max_context_tokens

    if not chunks:
        return [], 0

    # 1. Break chunks into block items with source URL and relevance score
    blocks: List[Tuple[float, int, str, str]] = []  # (score, original_index, source_url, content)
    index_counter = 0

    for chunk in chunks:
        if not chunk.cleaned_content:
            continue

        # Split into section blocks by headers or paragraph breaks
        raw_blocks = re.split(r"\n(?=#|\n)", chunk.cleaned_content)
        for blk in raw_blocks:
            stripped = blk.strip()
            if not stripped:
                continue
            blk_score = score_content_block(stripped, chunk.source_url)
            blocks.append((blk_score, index_counter, chunk.source_url, stripped))
            index_counter += 1

    if not blocks:
        return [], 0

    # 2. Sort blocks by score descending, preserving original index for tie-breaking
    sorted_blocks = sorted(blocks, key=lambda b: (-b[0], b[1]))

    selected_blocks: List[Tuple[int, str, str]] = []  # (original_index, source_url, content)
    current_tokens = 0

    # 3. Accumulate blocks under budget limit
    for score, orig_idx, source_url, content in sorted_blocks:
        formatted_header = f"[Source: {source_url}]\n"
        blk_text = f"{formatted_header}{content}\n\n"
        blk_tokens = count_tokens(blk_text)

        if current_tokens + blk_tokens <= budget:
            selected_blocks.append((orig_idx, source_url, content))
            current_tokens += blk_tokens
        else:
            # Partial truncation for last block if space remains
            remaining_tokens = budget - current_tokens
            if remaining_tokens > 15:
                header_tokens = count_tokens(formatted_header)
                allowed_tokens = max(1, remaining_tokens - header_tokens)

                encoder = get_token_encoder()
                content_tokens = encoder.encode(content)
                truncated_token_ids = content_tokens[:allowed_tokens]
                truncated_content = encoder.decode(truncated_token_ids).rstrip()

                candidate_text = f"{formatted_header}{truncated_content}\n\n"
                candidate_tokens = count_tokens(candidate_text)

                # Trim token by token if formatting added overflow tokens
                while candidate_tokens > remaining_tokens and len(truncated_token_ids) > 1:
                    truncated_token_ids = truncated_token_ids[:-1]
                    truncated_content = encoder.decode(truncated_token_ids).rstrip()
                    candidate_text = f"{formatted_header}{truncated_content}\n\n"
                    candidate_tokens = count_tokens(candidate_text)

                if candidate_tokens <= remaining_tokens and truncated_content:
                    selected_blocks.append((orig_idx, source_url, truncated_content))
                    current_tokens += candidate_tokens
            break

    # 4. Re-sort selected blocks by original index to maintain logical reading order
    selected_blocks.sort(key=lambda b: b[0])

    # 5. Group selected blocks by source_url into PageChunk models
    url_to_contents: dict = {}
    for _, source_url, content in selected_blocks:
        if source_url not in url_to_contents:
            url_to_contents[source_url] = []
        url_to_contents[source_url].append(content)

    final_chunks: List[PageChunk] = []
    total_tokens = 0

    for source_url, contents in url_to_contents.items():
        combined_text = "\n\n".join(contents)
        chunk_tokens = count_tokens(f"[Source: {source_url}]\n{combined_text}")
        total_tokens += chunk_tokens
        final_chunks.append(
            PageChunk(
                source_url=source_url,
                cleaned_content=combined_text,
                token_count=chunk_tokens,
            )
        )

    return final_chunks, total_tokens
