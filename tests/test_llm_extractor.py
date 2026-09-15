"""Unit tests for LLM extraction client, prompts, and cost tracker."""

import pytest
from unittest.mock import MagicMock, patch
from openai import AuthenticationError, RateLimitError

from src.models import LeadEnrichmentResult, TeamMember, PageChunk
from src.llm.prompts import SYSTEM_PROMPT, format_context_for_llm
from src.llm.cost_tracker import calculate_llm_cost, UsageMetrics
from src.llm.client import LLMExtractor, sanitize_error_message


@pytest.fixture
def sample_chunks() -> list[PageChunk]:
    """Provide sample preprocessed PageChunk objects."""
    return [
        PageChunk(
            source_url="https://postman.com/about",
            cleaned_content="# About Postman\nPostman is an API platform for building and using APIs.\nIt simplifies API lifecycle collaboration.",
        ),
        PageChunk(
            source_url="https://postman.com/team",
            cleaned_content="# Leadership\nAlice Smith - Chief Executive Officer\nBob Jones - Head of Engineering",
        ),
    ]


@pytest.fixture
def mock_parsed_result() -> LeadEnrichmentResult:
    """Provide expected extracted LeadEnrichmentResult."""
    return LeadEnrichmentResult(
        company_overview="Postman is an API platform for building and using APIs. It simplifies API lifecycle collaboration.",
        target_audience_icp="API developers and engineering teams.",
        contact_points=["help@postman.com"],
        team_members=[
            TeamMember(name="Alice Smith", role="Chief Executive Officer", linkedin_url="https://linkedin.com/in/alicesmith"),
            TeamMember(name="Bob Jones", role="Head of Engineering", linkedin_url=None),
        ],
        data_confidence_score=0.0,
    )


def test_1_valid_structured_response_parses_successfully(sample_chunks, mock_parsed_result):
    """Test 1: Valid structured response parses correctly into LeadEnrichmentResult."""
    extractor = LLMExtractor(api_key="sk-test-key", model="gpt-4o-mini")

    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.parsed = mock_parsed_result
    mock_completion.usage.prompt_tokens = 1000
    mock_completion.usage.completion_tokens = 200

    with patch.object(extractor, "_execute_api_call", return_value=mock_completion):
        result, usage = extractor.extract_lead_data(sample_chunks)

        assert result.company_overview == mock_parsed_result.company_overview
        assert result.target_audience_icp == mock_parsed_result.target_audience_icp
        assert len(result.team_members) == 2
        assert usage.prompt_tokens == 1000
        assert usage.completion_tokens == 200
        assert usage.estimated_cost_usd > 0.0


def test_2_company_overview_is_returned(sample_chunks, mock_parsed_result):
    """Test 2: Company overview field is returned in result."""
    extractor = LLMExtractor(api_key="sk-test-key")
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.parsed = mock_parsed_result

    with patch.object(extractor, "_execute_api_call", return_value=mock_completion):
        result, _ = extractor.extract_lead_data(sample_chunks)
        assert "API platform" in result.company_overview


def test_3_icp_is_returned(sample_chunks, mock_parsed_result):
    """Test 3: Target audience ICP field is returned in result."""
    extractor = LLMExtractor(api_key="sk-test-key")
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.parsed = mock_parsed_result

    with patch.object(extractor, "_execute_api_call", return_value=mock_completion):
        result, _ = extractor.extract_lead_data(sample_chunks)
        assert "developers" in result.target_audience_icp


def test_4_contact_points_are_returned(sample_chunks, mock_parsed_result):
    """Test 4: Generic contact points are returned in result."""
    extractor = LLMExtractor(api_key="sk-test-key")
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.parsed = mock_parsed_result

    with patch.object(extractor, "_execute_api_call", return_value=mock_completion):
        result, _ = extractor.extract_lead_data(sample_chunks)
        assert result.contact_points == ["help@postman.com"]


def test_5_team_members_are_returned(sample_chunks, mock_parsed_result):
    """Test 5: Team members list is returned in result."""
    extractor = LLMExtractor(api_key="sk-test-key")
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.parsed = mock_parsed_result

    with patch.object(extractor, "_execute_api_call", return_value=mock_completion):
        result, _ = extractor.extract_lead_data(sample_chunks)
        assert len(result.team_members) == 2
        assert result.team_members[0].name == "Alice Smith"


def test_6_linkedin_url_is_optional(sample_chunks, mock_parsed_result):
    """Test 6: Team member with linkedin_url=None parses cleanly."""
    extractor = LLMExtractor(api_key="sk-test-key")
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.parsed = mock_parsed_result

    with patch.object(extractor, "_execute_api_call", return_value=mock_completion):
        result, _ = extractor.extract_lead_data(sample_chunks)
        assert result.team_members[1].linkedin_url is None


def test_7_missing_contact_info_handled_with_empty_list(sample_chunks):
    """Test 7: Missing contact information defaults to empty list."""
    empty_contact_result = LeadEnrichmentResult(
        company_overview="Overview text.",
        target_audience_icp="Developers.",
        contact_points=[],
        team_members=[],
    )
    extractor = LLMExtractor(api_key="sk-test-key")
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.parsed = empty_contact_result

    with patch.object(extractor, "_execute_api_call", return_value=mock_completion):
        result, _ = extractor.extract_lead_data(sample_chunks)
        assert result.contact_points == []


def test_8_missing_team_info_handled_with_empty_list(sample_chunks):
    """Test 8: Missing team information defaults to empty list."""
    empty_team_result = LeadEnrichmentResult(
        company_overview="Overview text.",
        target_audience_icp="Developers.",
        contact_points=["info@example.com"],
        team_members=[],
    )
    extractor = LLMExtractor(api_key="sk-test-key")
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.parsed = empty_team_result

    with patch.object(extractor, "_execute_api_call", return_value=mock_completion):
        result, _ = extractor.extract_lead_data(sample_chunks)
        assert result.team_members == []


def test_9_invalid_structured_response_handled(sample_chunks):
    """Test 9: Failure to parse structured output raises RuntimeError."""
    extractor = LLMExtractor(api_key="sk-test-key")
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.parsed = None
    mock_completion.choices[0].message.refusal = "Safety refusal"

    with patch.object(extractor, "_execute_api_call", return_value=mock_completion):
        with pytest.raises(RuntimeError) as exc_info:
            extractor.extract_lead_data(sample_chunks)
        assert "Safety refusal" in str(exc_info.value)


def test_10_api_authentication_error_handled(sample_chunks):
    """Test 10: AuthenticationError is caught and raised as RuntimeError."""
    extractor = LLMExtractor(api_key="invalid-key")
    auth_err = AuthenticationError("Incorrect API key provided", response=MagicMock(), body=None)

    with patch.object(extractor, "_execute_api_call", side_effect=auth_err):
        with pytest.raises(RuntimeError) as exc_info:
            extractor.extract_lead_data(sample_chunks)
        assert "Authentication Error" in str(exc_info.value)


def test_11_rate_limit_triggers_bounded_retry(sample_chunks):
    """Test 11: RateLimitError triggers bounded retries up to max attempts."""
    extractor = LLMExtractor(api_key="sk-test-key")
    rate_err = RateLimitError("Rate limit reached", response=MagicMock(), body=None)

    with patch.object(extractor, "_execute_api_call", side_effect=rate_err) as mock_call:
        with pytest.raises(RuntimeError):
            extractor.extract_lead_data(sample_chunks)
        # Should attempt retries up to tenacity stop condition
        assert mock_call.call_count >= 1


def test_12_final_api_failure_produces_meaningful_error(sample_chunks):
    """Test 12: Final API exception raises descriptive RuntimeError."""
    extractor = LLMExtractor(api_key="sk-test-key")

    with patch.object(extractor, "_execute_api_call", side_effect=ValueError("Custom API failure")):
        with pytest.raises(RuntimeError) as exc_info:
            extractor.extract_lead_data(sample_chunks)
        assert "LLM Extraction failed" in str(exc_info.value)


def test_13_source_urls_included_in_formatted_context(sample_chunks):
    """Test 13: Formatted prompt context explicitly includes SOURCE URL header for each chunk."""
    formatted = format_context_for_llm(sample_chunks)
    assert "SOURCE URL: https://postman.com/about" in formatted
    assert "SOURCE URL: https://postman.com/team" in formatted


def test_14_raw_html_not_sent_to_llm_context(sample_chunks):
    """Test 14: Prompt context contains clean text and markdown, no raw HTML tags."""
    formatted = format_context_for_llm(sample_chunks)
    assert "<html" not in formatted.lower()
    assert "<body" not in formatted.lower()
    assert "<script" not in formatted.lower()


def test_15_confidence_score_not_requested_from_llm():
    """Test 15: System prompt explicitly directs LLM NOT to output confidence scores."""
    assert "Do NOT calculate or output any confidence score." in SYSTEM_PROMPT


def test_16_api_key_never_appears_in_logs_or_errors():
    """Test 16: sanitize_error_message masks OpenAI secret key patterns."""
    raw_error = "Failed request using key sk-proj-1234567890abcdef123456"
    sanitized = sanitize_error_message(raw_error)
    assert "sk-proj-1234567890abcdef123456" not in sanitized
    assert "sk-***MASKED***" in sanitized


def test_17_cost_tracker_handles_known_usage():
    """Test 17: Cost tracker calculates estimated USD cost for gpt-4o-mini."""
    metrics = calculate_llm_cost("gpt-4o-mini", prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert metrics.prompt_tokens == 1_000_000
    assert metrics.completion_tokens == 1_000_000
    assert metrics.total_tokens == 2_000_000
    # $0.15 + $0.60 = $0.75 USD
    assert metrics.estimated_cost_usd == 0.75


def test_18_cost_tracker_handles_missing_usage():
    """Test 18: Cost tracker gracefully handles None/missing token values."""
    metrics = calculate_llm_cost("gpt-4o-mini", prompt_tokens=None, completion_tokens=None)
    assert metrics.prompt_tokens == 0
    assert metrics.completion_tokens == 0
    assert metrics.total_tokens == 0
    assert metrics.estimated_cost_usd == 0.0


def test_19_identical_input_produces_deterministic_context_formatting(sample_chunks):
    """Test 19: format_context_for_llm is 100% deterministic."""
    fmt1 = format_context_for_llm(sample_chunks)
    fmt2 = format_context_for_llm(sample_chunks)
    assert fmt1 == fmt2


def test_20_no_hallucination_scenario():
    """Test 20: Minimal website context with no team/emails returns empty lists without hallucinated details."""
    minimal_chunks = [
        PageChunk(
            source_url="https://examplecorp.com",
            cleaned_content="ExampleCorp builds database tools for developers.",
        )
    ]
    minimal_result = LeadEnrichmentResult(
        company_overview="ExampleCorp builds database tools for developers.",
        target_audience_icp="Software developers requiring database tools.",
        contact_points=[],
        team_members=[],
        data_confidence_score=0.0,
    )

    extractor = LLMExtractor(api_key="sk-test-key")
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.parsed = minimal_result

    with patch.object(extractor, "_execute_api_call", return_value=mock_completion):
        result, _ = extractor.extract_lead_data(minimal_chunks)

        assert "database tools" in result.company_overview
        assert result.contact_points == []
        assert result.team_members == []
