"""Shared pytest fixtures for the Lead Enrichment Agent test suite."""

import pytest
from src.models import LeadEnrichmentResult, TeamMember
from src.confidence import ConfidenceInput


@pytest.fixture
def sample_team_members() -> list[TeamMember]:
    """Provide sample valid team members."""
    return [
        TeamMember(
            name="Alice Smith",
            role="Chief Executive Officer",
            linkedin_url="https://linkedin.com/in/alicesmith",
        ),
        TeamMember(
            name="Bob Jones",
            role="Head of Engineering",
            linkedin_url="https://linkedin.com/in/bobjones",
        ),
    ]


@pytest.fixture
def complete_lead_result(sample_team_members: list[TeamMember]) -> LeadEnrichmentResult:
    """Provide a fully populated LeadEnrichmentResult instance."""
    return LeadEnrichmentResult(
        company_overview="Postman is a leading API platform for building and using APIs. It simplifies each step of the API lifecycle and streamlines collaboration.",
        target_audience_icp="Software engineers, API developers, product managers, and enterprise engineering teams.",
        contact_points=["help@postman.com", "press@postman.com"],
        team_members=sample_team_members,
        data_confidence_score=0.0,
    )
