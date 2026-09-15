"""Data models for Autonomous Lead Enrichment Agent using Pydantic v2."""

from typing import List, Optional
from pydantic import BaseModel, Field


class TeamMember(BaseModel):
    """Extracted key leadership or team member model."""

    name: str = Field(description="Full name of the team or leadership member")
    role: str = Field(description="Role, position, or job title within the company")
    linkedin_url: Optional[str] = Field(
        default=None, description="LinkedIn profile URL if discoverable"
    )


class PageChunk(BaseModel):
    """Preprocessed DOM content chunk with source URL attribution."""

    source_url: str = Field(description="Source URL where this content was scraped")
    cleaned_content: str = Field(description="Scrubbed textual/markdown content")
    section_title: Optional[str] = Field(
        default=None, description="Discovered page or section title"
    )
    token_count: int = Field(
        default=0, ge=0, description="Estimated token count for this chunk"
    )


class ScrapedPage(BaseModel):
    """Raw scraped web page model holding HTTP metadata and uncleaned HTML."""

    source_url: str = Field(description="Normalized target URL requested")
    final_url: str = Field(description="Final URL reached after redirects")
    status_code: Optional[int] = Field(default=None, description="HTTP response status code")
    page_title: Optional[str] = Field(default=None, description="HTML document title")
    raw_html: str = Field(default="", description="Uncleaned raw DOM HTML content")
    error_message: Optional[str] = Field(default=None, description="Error detail if page fetch failed")
    blocked_by_bot: bool = Field(default=False, description="True if bot block signature was detected")


class LeadEnrichmentResult(BaseModel):
    """Structured extraction output model for a company domain."""

    company_overview: str = Field(
        default="", description="Concise 2-sentence company summary"
    )
    target_audience_icp: str = Field(
        default="", description="Target audience or Ideal Customer Profile (ICP)"
    )
    contact_points: List[str] = Field(
        default_factory=list,
        description="Public contact email addresses or contact methods",
    )
    team_members: List[TeamMember] = Field(
        default_factory=list, description="Key leadership and team members"
    )
    data_confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Deterministic confidence score between 0.0 and 1.0",
    )


class DomainResult(BaseModel):
    """Top-level pipeline execution result for a single domain with error isolation."""

    domain: str = Field(description="Target company domain name")
    status: str = Field(
        default="success",
        description="Execution status: 'success', 'partial', 'error', or 'bot_blocked'",
    )
    extracted_data: Optional[LeadEnrichmentResult] = Field(
        default=None, description="Extracted lead data if available"
    )
    error_message: Optional[str] = Field(
        default=None, description="Error detail if execution failed"
    )
    blocked_by_bot: bool = Field(
        default=False, description="True if domain crawl was blocked by bot protection"
    )
    pages_scraped_count: int = Field(
        default=0, ge=0, description="Number of successfully scraped pages"
    )
    scraped_urls: List[str] = Field(
        default_factory=list, description="List of URLs successfully scraped"
    )
    scraped_pages: List[ScrapedPage] = Field(
        default_factory=list, description="List of raw scraped page payloads for preprocessing"
    )
