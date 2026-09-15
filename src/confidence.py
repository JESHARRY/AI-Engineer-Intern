"""Deterministic confidence scoring module for extracted domain data.

Calculates a data_confidence_score strictly between 0.0 and 1.0 based on
field completeness, contact evidence, and crawl coverage, independent of the LLM.
"""

import re
from typing import List, Optional
from pydantic import BaseModel, Field

from src.models import LeadEnrichmentResult, TeamMember


class ConfidenceInput(BaseModel):
    """Internal input structure for deterministic confidence scoring."""

    company_overview: str = ""
    target_audience_icp: str = ""
    contact_points: List[str] = Field(default_factory=list)
    team_members: List[TeamMember] = Field(default_factory=list)
    pages_scraped_count: int = 0
    is_blocked_or_failed: bool = False


# Regex for basic email format validation
_EMAIL_REGEX = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")


def calculate_confidence_score(scoring_input: ConfidenceInput) -> float:
    """
    Calculate a deterministic confidence score strictly between 0.0 and 1.0.

    Scoring Matrix:
    1. Company Overview (Max +0.25):
       - Valid overview summary present = +0.25
    2. Target Audience / ICP (Max +0.20):
       - Clear ICP summary present = +0.20
    3. Leadership Coverage (Max +0.25):
       - 2+ valid team members (with name & role) = +0.25
       - 1 valid team member (with name & role) = +0.15
       - 0 team members = +0.00
    4. Contact Points & LinkedIn Evidence (Max +0.15):
       - At least one public contact email present = +0.10
       - At least one LinkedIn profile URL present = +0.05
    5. Crawl Coverage (Max +0.15):
       - Homepage + 2 or more subpages (pages_scraped_count >= 3) = +0.15
       - Homepage + 1 subpage (pages_scraped_count == 2) = +0.10
       - Homepage only (pages_scraped_count == 1) = +0.05
       - Blocked or failed crawl = +0.00

    Returns:
        float: Deterministic confidence score clamped to [0.0, 1.0].
    """
    # If the crawl was completely blocked or failed with 0 pages, score is 0.0
    if scoring_input.is_blocked_or_failed and scoring_input.pages_scraped_count == 0:
        return 0.0

    score = 0.0

    # 1. Company Overview (+0.25)
    if scoring_input.company_overview and scoring_input.company_overview.strip():
        score += 0.25

    # 2. ICP (+0.20)
    if scoring_input.target_audience_icp and scoring_input.target_audience_icp.strip():
        score += 0.20

    # 3. Leadership Coverage (+0.25 / +0.15 / +0.00)
    valid_members = [
        m
        for m in scoring_input.team_members
        if m.name and m.name.strip() and m.role and m.role.strip()
    ]
    if len(valid_members) >= 2:
        score += 0.25
    elif len(valid_members) == 1:
        score += 0.15

    # 4. Public Contacts & LinkedIn Evidence
    has_email = any(
        _EMAIL_REGEX.search(item)
        for item in scoring_input.contact_points
        if item and item.strip()
    )
    if has_email:
        score += 0.10

    has_linkedin = any(
        m.linkedin_url and "linkedin.com" in m.linkedin_url.lower()
        for m in scoring_input.team_members
    ) or any(
        item and "linkedin.com" in item.lower()
        for item in scoring_input.contact_points
    )
    if has_linkedin:
        score += 0.05

    # 5. Crawl Coverage (+0.15 / +0.10 / +0.05 / +0.00)
    if not scoring_input.is_blocked_or_failed:
        if scoring_input.pages_scraped_count >= 3:
            score += 0.15
        elif scoring_input.pages_scraped_count == 2:
            score += 0.10
        elif scoring_input.pages_scraped_count == 1:
            score += 0.05

    # Clamp final score strictly to [0.0, 1.0] and round to 2 decimal places
    final_score = round(min(max(score, 0.0), 1.0), 2)
    return final_score


def compute_result_confidence(
    result: LeadEnrichmentResult,
    pages_scraped_count: int,
    is_blocked_or_failed: bool = False,
) -> float:
    """Helper function to calculate confidence score directly from a LeadEnrichmentResult."""
    scoring_input = ConfidenceInput(
        company_overview=result.company_overview,
        target_audience_icp=result.target_audience_icp,
        contact_points=result.contact_points,
        team_members=result.team_members,
        pages_scraped_count=pages_scraped_count,
        is_blocked_or_failed=is_blocked_or_failed,
    )
    return calculate_confidence_score(scoring_input)
