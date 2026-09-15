"""Unit tests for the deterministic confidence scorer module."""

import pytest
from src.confidence import ConfidenceInput, calculate_confidence_score, compute_result_confidence
from src.models import LeadEnrichmentResult, TeamMember


def test_1_fully_complete_result(complete_lead_result: LeadEnrichmentResult):
    """Test 1: Fully complete result produces exact score of 1.0."""
    score = compute_result_confidence(
        result=complete_lead_result,
        pages_scraped_count=3,
        is_blocked_or_failed=False,
    )
    # Overview (0.25) + ICP (0.20) + 2 Team Members (0.25) + Email (0.10) + LinkedIn (0.05) + Crawl 3 pages (0.15) = 1.00
    assert score == 1.00


def test_2_overview_and_icp_only():
    """Test 2: Result with overview and ICP only on a single homepage."""
    inp = ConfidenceInput(
        company_overview="Example company overview summary.",
        target_audience_icp="Developers and small businesses.",
        contact_points=[],
        team_members=[],
        pages_scraped_count=1,
        is_blocked_or_failed=False,
    )
    # Overview (0.25) + ICP (0.20) + Homepage only (0.05) = 0.50
    score = calculate_confidence_score(inp)
    assert score == 0.50


def test_3_one_team_member():
    """Test 3: Single valid team member contributes +0.15."""
    member = TeamMember(name="John Doe", role="Founder")
    inp = ConfidenceInput(
        team_members=[member],
        pages_scraped_count=0,
        is_blocked_or_failed=False,
    )
    score = calculate_confidence_score(inp)
    assert score == 0.15


def test_4_two_or_more_team_members():
    """Test 4: Two or more valid team members contribute +0.25."""
    members = [
        TeamMember(name="John Doe", role="Founder"),
        TeamMember(name="Jane Doe", role="CTO"),
    ]
    inp = ConfidenceInput(
        team_members=members,
        pages_scraped_count=0,
        is_blocked_or_failed=False,
    )
    score = calculate_confidence_score(inp)
    assert score == 0.25


def test_5_contact_email_only():
    """Test 5: Public contact email contributes +0.10."""
    inp = ConfidenceInput(
        contact_points=["contact@example.com"],
        pages_scraped_count=0,
        is_blocked_or_failed=False,
    )
    score = calculate_confidence_score(inp)
    assert score == 0.10


def test_6_linkedin_only():
    """Test 6: Discovered LinkedIn URL contributes +0.05."""
    member = TeamMember(name="John", role="Dev", linkedin_url="https://linkedin.com/in/johndoe")
    inp = ConfidenceInput(
        team_members=[member],
        pages_scraped_count=0,
        is_blocked_or_failed=False,
    )
    # 1 member (+0.15) + LinkedIn (+0.05) = 0.20
    score = calculate_confidence_score(inp)
    assert score == 0.20

    # Explicit LinkedIn in contact_points without member role
    inp_contact = ConfidenceInput(
        contact_points=["https://linkedin.com/company/example"],
        pages_scraped_count=0,
        is_blocked_or_failed=False,
    )
    score_contact = calculate_confidence_score(inp_contact)
    assert score_contact == 0.05


def test_7_homepage_only():
    """Test 7: Crawl of homepage only contributes +0.05."""
    inp = ConfidenceInput(
        pages_scraped_count=1,
        is_blocked_or_failed=False,
    )
    score = calculate_confidence_score(inp)
    assert score == 0.05


def test_8_homepage_plus_one_subpage():
    """Test 8: Crawl of homepage + 1 subpage contributes +0.10."""
    inp = ConfidenceInput(
        pages_scraped_count=2,
        is_blocked_or_failed=False,
    )
    score = calculate_confidence_score(inp)
    assert score == 0.10


def test_9_homepage_plus_two_or_more_subpages():
    """Test 9: Crawl of homepage + 2 or more subpages contributes +0.15."""
    inp = ConfidenceInput(
        pages_scraped_count=3,
        is_blocked_or_failed=False,
    )
    score = calculate_confidence_score(inp)
    assert score == 0.15


def test_10_blocked_or_failed_crawl():
    """Test 10: Blocked or failed crawl with 0 pages scraped produces 0.0."""
    inp = ConfidenceInput(
        company_overview="Partial content",
        pages_scraped_count=0,
        is_blocked_or_failed=True,
    )
    score = calculate_confidence_score(inp)
    assert score == 0.0


def test_11_score_always_remains_within_bounds():
    """Test 11: Score is strictly bounded within [0.0, 1.0]."""
    # Test lower bound
    empty_inp = ConfidenceInput()
    assert calculate_confidence_score(empty_inp) == 0.0

    # Test upper bound with redundant data
    max_inp = ConfidenceInput(
        company_overview="Overview sentence one. Overview sentence two.",
        target_audience_icp="Ideal customer profile target.",
        contact_points=["admin@domain.com", "support@domain.com", "https://linkedin.com/company/domain"],
        team_members=[
            TeamMember(name="A", role="R1", linkedin_url="https://linkedin.com/in/a"),
            TeamMember(name="B", role="R2", linkedin_url="https://linkedin.com/in/b"),
            TeamMember(name="C", role="R3", linkedin_url="https://linkedin.com/in/c"),
        ],
        pages_scraped_count=10,
        is_blocked_or_failed=False,
    )
    score = calculate_confidence_score(max_inp)
    assert 0.0 <= score <= 1.0
    assert score == 1.00


def test_12_score_is_deterministic():
    """Test 12: Two identical inputs produce the exact same confidence score."""
    inp1 = ConfidenceInput(
        company_overview="Company overview string.",
        target_audience_icp="Developers",
        contact_points=["info@company.com"],
        team_members=[TeamMember(name="Jane", role="CEO")],
        pages_scraped_count=2,
    )
    inp2 = ConfidenceInput(
        company_overview="Company overview string.",
        target_audience_icp="Developers",
        contact_points=["info@company.com"],
        team_members=[TeamMember(name="Jane", role="CEO")],
        pages_scraped_count=2,
    )
    score1 = calculate_confidence_score(inp1)
    score2 = calculate_confidence_score(inp2)
    assert score1 == score2
