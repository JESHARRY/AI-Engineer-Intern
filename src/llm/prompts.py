"""System prompts and context formatting for LLM structured extraction."""

from typing import List
from src.models import PageChunk

SYSTEM_PROMPT = """You are a precise data extraction AI agent specializing in B2B lead enrichment.
Your task is to extract structured company information strictly from the provided website content.

CRITICAL INSTRUCTIONS (EVIDENCE-GROUNDED EXTRACTION):
1. Use ONLY the supplied website content. Do NOT use prior knowledge, external assumptions, or guesses.
2. Do NOT invent, assume, or fabricate names, job titles, email addresses, or LinkedIn URLs.
3. If specific information is missing or unavailable in the supplied text:
   - For strings (e.g., company_overview, target_audience_icp), return an empty string "".
   - For lists (e.g., contact_points, team_members), return an empty list [].
   - For optional fields (e.g., linkedin_url), return null.
4. Field Definitions:
   - company_overview: Exactly 2 concise sentences describing what the company does, based strictly on the content.
   - target_audience_icp: Concise description of the target audience or Ideal Customer Profile (ICP).
   - contact_points: List of generic/public company contact email addresses (e.g., sales@, support@, info@, help@) explicitly found in the content.
   - team_members: List of key leadership or team members explicitly named in the text, including name, role/title, and linkedin_url if present.
5. Do NOT calculate or output any confidence score.
"""


def format_context_for_llm(chunks: List[PageChunk]) -> str:
    """
    Format a list of PageChunk objects into a clean, source-attributed prompt context string.
    """
    if not chunks:
        return "No website content available."

    formatted_sections: List[str] = []
    for chunk in chunks:
        if not chunk.cleaned_content:
            continue
        header = f"SOURCE URL: {chunk.source_url}"
        section = f"{header}\n\nCONTENT:\n{chunk.cleaned_content.strip()}"
        formatted_sections.append(section)

    if not formatted_sections:
        return "No website content available."

    return "\n\n" + ("=" * 40) + "\n\n".join(formatted_sections)
