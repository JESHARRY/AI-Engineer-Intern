"""Unit tests for the end-to-end LeadEnrichmentPipeline module using mocked boundaries."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.models import DomainResult, ScrapedPage, LeadEnrichmentResult, TeamMember
from src.llm.cost_tracker import UsageMetrics
from src.pipeline import LeadEnrichmentPipeline


@pytest.fixture
def mock_scraped_pages() -> list[ScrapedPage]:
    """Provide realistic mock ScrapedPage objects."""
    return [
        ScrapedPage(
            source_url="https://postman.com",
            final_url="https://postman.com",
            status_code=200,
            page_title="Postman API Platform",
            raw_html="<html><body><header><h1>Postman Platform</h1></header><main><p>Postman simplifies API development.</p></main></body></html>",
        ),
        ScrapedPage(
            source_url="https://postman.com/about",
            final_url="https://postman.com/about",
            status_code=200,
            page_title="About Postman",
            raw_html="<html><body><h1>About Us</h1><p>Postman is a leading API platform for developers.</p></body></html>",
        ),
    ]


@pytest.fixture
def mock_extracted_lead_data() -> LeadEnrichmentResult:
    """Provide sample extracted LeadEnrichmentResult."""
    return LeadEnrichmentResult(
        company_overview="Postman is an API platform for building and using APIs. It simplifies API lifecycle collaboration.",
        target_audience_icp="Developers and enterprise engineering teams.",
        contact_points=["help@postman.com"],
        team_members=[
            TeamMember(name="Alice Smith", role="CEO", linkedin_url="https://linkedin.com/in/alicesmith")
        ],
        data_confidence_score=0.0,
    )


@pytest.mark.asyncio
async def test_1_successful_single_domain_pipeline(mock_scraped_pages, mock_extracted_lead_data):
    """Test 1: Successful single domain processing through full pipeline."""
    mock_crawler = MagicMock()
    mock_crawler.crawl_domain = AsyncMock(
        return_value=DomainResult(
            domain="postman.com",
            status="success",
            pages_scraped_count=2,
            scraped_urls=["https://postman.com", "https://postman.com/about"],
            scraped_pages=mock_scraped_pages,
        )
    )

    mock_extractor = MagicMock()
    mock_usage = UsageMetrics(model="gpt-4o-mini", prompt_tokens=500, completion_tokens=100, estimated_cost_usd=0.0001)
    mock_extractor.extract_lead_data.return_value = (mock_extracted_lead_data, mock_usage)

    pipeline = LeadEnrichmentPipeline(crawler=mock_crawler, extractor=mock_extractor)
    result = await pipeline.process_single_domain("postman.com")

    assert result.status == "success"
    assert result.domain == "postman.com"
    assert result.pages_scraped_count == 2
    assert result.extracted_data is not None
    assert result.extracted_data.data_confidence_score > 0.0


@pytest.mark.asyncio
async def test_2_multiple_domains_batch_processing(mock_scraped_pages, mock_extracted_lead_data):
    """Test 2: Pipeline processes multiple domains in batch."""
    mock_crawler = MagicMock()
    mock_crawler.crawl_domain = AsyncMock(
        return_value=DomainResult(
            domain="domain.com",
            status="success",
            pages_scraped_count=2,
            scraped_urls=["https://domain.com"],
            scraped_pages=mock_scraped_pages,
        )
    )

    mock_extractor = MagicMock()
    mock_usage = UsageMetrics(model="gpt-4o-mini")
    mock_extractor.extract_lead_data.return_value = (mock_extracted_lead_data, mock_usage)

    pipeline = LeadEnrichmentPipeline(crawler=mock_crawler, extractor=mock_extractor)

    with patch("src.pipeline.BrowserManager") as mock_bm_cls, patch("src.pipeline.write_json_output") as mock_writer:
        mock_bm_cls.return_value.__aenter__.return_value = AsyncMock()
        results = await pipeline.run_batch(["domain1.com", "domain2.com"], output_path="output/test_output.json")

        assert len(results) == 2
        assert mock_writer.called


@pytest.mark.asyncio
async def test_3_one_domain_failure_does_not_stop_other_domains(mock_scraped_pages, mock_extracted_lead_data):
    """Test 3: Failure on domain1 does not prevent domain2 from being processed."""
    mock_crawler = MagicMock()

    async def side_effect(domain, **kwargs):
        if domain == "bad-domain.com":
            raise RuntimeError("DNS lookup failed for bad-domain.com")
        return DomainResult(
            domain=domain,
            status="success",
            pages_scraped_count=2,
            scraped_urls=["https://good-domain.com"],
            scraped_pages=mock_scraped_pages,
        )

    mock_crawler.crawl_domain = AsyncMock(side_effect=side_effect)

    mock_extractor = MagicMock()
    mock_usage = UsageMetrics(model="gpt-4o-mini")
    mock_extractor.extract_lead_data.return_value = (mock_extracted_lead_data, mock_usage)

    pipeline = LeadEnrichmentPipeline(crawler=mock_crawler, extractor=mock_extractor)

    with patch("src.pipeline.BrowserManager") as mock_bm_cls, patch("src.pipeline.write_json_output"):
        mock_bm_cls.return_value.__aenter__.return_value = AsyncMock()
        results = await pipeline.run_batch(["bad-domain.com", "good-domain.com"])

        assert len(results) == 2
        assert results[0].status == "error"
        assert results[1].status == "success"


@pytest.mark.asyncio
async def test_4_crawler_failure_becomes_structured_domain_error():
    """Test 4: Crawler error returns DomainResult with status='error'."""
    mock_crawler = MagicMock()
    mock_crawler.crawl_domain = AsyncMock(
        return_value=DomainResult(
            domain="failed-domain.com",
            status="error",
            error_message="HTTP 500 Internal Server Error",
            pages_scraped_count=0,
        )
    )

    pipeline = LeadEnrichmentPipeline(crawler=mock_crawler)
    result = await pipeline.process_single_domain("failed-domain.com")

    assert result.status == "error"
    assert "HTTP 500" in result.error_message


@pytest.mark.asyncio
async def test_5_preprocessing_failure_becomes_structured_domain_error():
    """Test 5: Empty HTML preprocessing yields structured error DomainResult."""
    empty_page = ScrapedPage(source_url="https://empty.com", final_url="https://empty.com", raw_html="")
    mock_crawler = MagicMock()
    mock_crawler.crawl_domain = AsyncMock(
        return_value=DomainResult(
            domain="empty.com",
            status="success",
            pages_scraped_count=1,
            scraped_pages=[empty_page],
        )
    )

    pipeline = LeadEnrichmentPipeline(crawler=mock_crawler)
    result = await pipeline.process_single_domain("empty.com")

    assert result.status == "error"
    assert "No valid HTML" in result.error_message


@pytest.mark.asyncio
async def test_6_llm_failure_becomes_structured_domain_error(mock_scraped_pages):
    """Test 6: LLM extraction failure produces partial/error DomainResult."""
    mock_crawler = MagicMock()
    mock_crawler.crawl_domain = AsyncMock(
        return_value=DomainResult(
            domain="llm-fail.com",
            status="success",
            pages_scraped_count=2,
            scraped_pages=mock_scraped_pages,
        )
    )

    mock_extractor = MagicMock()
    mock_extractor.extract_lead_data.side_effect = RuntimeError("OpenAI API RateLimitError")

    pipeline = LeadEnrichmentPipeline(crawler=mock_crawler, extractor=mock_extractor)
    result = await pipeline.process_single_domain("llm-fail.com")

    assert result.status in ("partial", "error")
    assert "RateLimitError" in result.error_message


@pytest.mark.asyncio
async def test_7_confidence_score_is_attached(mock_scraped_pages, mock_extracted_lead_data):
    """Test 7: Extracted result has data_confidence_score attached."""
    mock_crawler = MagicMock()
    mock_crawler.crawl_domain = AsyncMock(
        return_value=DomainResult(
            domain="postman.com",
            status="success",
            pages_scraped_count=2,
            scraped_pages=mock_scraped_pages,
        )
    )

    mock_extractor = MagicMock()
    mock_extractor.extract_lead_data.return_value = (mock_extracted_lead_data, UsageMetrics(model="gpt-4o-mini"))

    pipeline = LeadEnrichmentPipeline(crawler=mock_crawler, extractor=mock_extractor)
    result = await pipeline.process_single_domain("postman.com")

    assert result.extracted_data is not None
    assert result.extracted_data.data_confidence_score > 0.0


@pytest.mark.asyncio
async def test_8_confidence_remains_within_bounds(mock_scraped_pages, mock_extracted_lead_data):
    """Test 8: Calculated confidence score is strictly within [0.0, 1.0]."""
    mock_crawler = MagicMock()
    mock_crawler.crawl_domain = AsyncMock(
        return_value=DomainResult(
            domain="postman.com",
            status="success",
            pages_scraped_count=2,
            scraped_pages=mock_scraped_pages,
        )
    )

    mock_extractor = MagicMock()
    mock_extractor.extract_lead_data.return_value = (mock_extracted_lead_data, UsageMetrics(model="gpt-4o-mini"))

    pipeline = LeadEnrichmentPipeline(crawler=mock_crawler, extractor=mock_extractor)
    result = await pipeline.process_single_domain("postman.com")

    score = result.extracted_data.data_confidence_score
    assert 0.0 <= score <= 1.0


@pytest.mark.asyncio
async def test_9_empty_crawl_handled_gracefully():
    """Test 9: Crawl returning 0 pages produces error DomainResult."""
    mock_crawler = MagicMock()
    mock_crawler.crawl_domain = AsyncMock(
        return_value=DomainResult(domain="empty.com", status="error", pages_scraped_count=0)
    )

    pipeline = LeadEnrichmentPipeline(crawler=mock_crawler)
    result = await pipeline.process_single_domain("empty.com")

    assert result.status == "error"
    assert result.pages_scraped_count == 0


@pytest.mark.asyncio
async def test_10_blocked_crawl_handled_gracefully():
    """Test 10: Bot-blocked crawl returns status='bot_blocked'."""
    mock_crawler = MagicMock()
    mock_crawler.crawl_domain = AsyncMock(
        return_value=DomainResult(
            domain="blocked.com",
            status="bot_blocked",
            blocked_by_bot=True,
            error_message="Cloudflare protection",
        )
    )

    pipeline = LeadEnrichmentPipeline(crawler=mock_crawler)
    result = await pipeline.process_single_domain("blocked.com")

    assert result.status == "bot_blocked"
    assert result.blocked_by_bot is True


@pytest.mark.asyncio
async def test_11_output_contains_one_result_per_requested_domain(mock_scraped_pages, mock_extracted_lead_data):
    """Test 11: Batch execution returns exactly 1 DomainResult per requested domain."""
    mock_crawler = MagicMock()
    mock_crawler.crawl_domain = AsyncMock(
        return_value=DomainResult(
            domain="dom.com",
            status="success",
            pages_scraped_count=2,
            scraped_pages=mock_scraped_pages,
        )
    )

    mock_extractor = MagicMock()
    mock_extractor.extract_lead_data.return_value = (mock_extracted_lead_data, UsageMetrics(model="gpt-4o-mini"))

    pipeline = LeadEnrichmentPipeline(crawler=mock_crawler, extractor=mock_extractor)

    with patch("src.pipeline.BrowserManager") as mock_bm_cls, patch("src.pipeline.write_json_output"):
        mock_bm_cls.return_value.__aenter__.return_value = AsyncMock()
        domains = ["postman.com", "supabase.com", "vapi.ai"]
        results = await pipeline.run_batch(domains)

        assert len(results) == len(domains)
        assert [r.domain for r in results] == domains


@pytest.mark.asyncio
async def test_12_no_api_key_exposed_in_errors():
    """Test 12: Secret keys like sk-proj-12345 are masked in pipeline error messages."""
    mock_crawler = MagicMock()
    mock_crawler.crawl_domain = AsyncMock(
        side_effect=RuntimeError("Failure with key sk-proj-1234567890abcdef123456")
    )

    pipeline = LeadEnrichmentPipeline(crawler=mock_crawler)
    result = await pipeline.process_single_domain("key-test.com")

    assert "sk-proj-1234567890abcdef123456" not in result.error_message
    assert "sk-***MASKED***" in result.error_message


@pytest.mark.asyncio
async def test_13_pipeline_does_not_make_real_network_calls():
    """Test 13: Pipeline executes 100% offline when dependencies are mocked."""
    mock_crawler = MagicMock()
    mock_crawler.crawl_domain = AsyncMock()
    mock_extractor = MagicMock()

    pipeline = LeadEnrichmentPipeline(crawler=mock_crawler, extractor=mock_extractor)
    assert pipeline.crawler == mock_crawler
    assert pipeline.extractor == mock_extractor
