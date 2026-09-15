"""Unit tests for browser manager and website crawler modules."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.models import ScrapedPage, DomainResult
from src.scraper.crawler import (
    normalize_domain,
    is_internal_url,
    normalize_url,
    score_link,
    detect_bot_block,
    extract_ranked_internal_links,
    CompanyCrawler,
)


def create_mock_browser_manager():
    """Helper to create an AsyncMock BrowserManager."""
    mock_bm = AsyncMock()
    mock_page = AsyncMock()
    mock_bm.get_page.return_value = mock_page
    return mock_bm


def test_1_domain_normalization():
    """Test 1: Normalizing raw domains into HTTPS base URLs."""
    assert normalize_domain("postman.com") == "https://postman.com"
    assert normalize_domain("http://supabase.com/") == "https://supabase.com"
    assert normalize_domain("  vapi.ai/  ") == "https://vapi.ai"


def test_2_internal_vs_external_url_detection():
    """Test 2: Detecting whether URLs belong to the target domain or subdomains."""
    assert is_internal_url("https://postman.com/about", "postman.com") is True
    assert is_internal_url("https://sub.postman.com/team", "postman.com") is True
    assert is_internal_url("https://twitter.com/postman", "postman.com") is False
    assert is_internal_url("https://github.com/supabase/supabase", "supabase.com") is False


def test_3_url_normalization_and_deduplication():
    """Test 3: Normalizing relative paths, trailing slashes, and fragment anchors."""
    base = "https://postman.com"
    norm1 = normalize_url("about/", base)
    norm2 = normalize_url("/about#team", base)
    norm3 = normalize_url("about?ref=nav", base)

    assert norm1 == "https://postman.com/about"
    assert norm2 == "https://postman.com/about"
    assert norm3 == "https://postman.com/about"


def test_4_relevant_link_ranking():
    """Test 4: Relevant URL paths rank higher than generic or terms pages."""
    about_score = score_link("/about", "About Us")
    pricing_score = score_link("/pricing", "Pricing Plans")
    terms_score = score_link("/terms", "Terms of Service")

    assert about_score > terms_score
    assert pricing_score > terms_score
    assert about_score >= 15.0  # 10 path + 5 anchor


def test_5_anchor_text_ranking():
    """Test 5: Anchor text containing target keywords increases rank."""
    score_with_anchor = score_link("/company/info", "Our Leadership")
    score_without_anchor = score_link("/company/info", "Click Here")

    assert score_with_anchor > score_without_anchor
    assert score_without_anchor == 20.0  # company (10) + info (10)
    assert score_with_anchor == 25.0     # company (10) + info (10) + leadership anchor (5)


def test_6_irrelevant_links_rank_lower():
    """Test 6: Irrelevant pages (blog, legal, terms, privacy) receive penalized scores."""
    blog_score = score_link("/blog/how-to-build-apis", "Read Blog")
    legal_score = score_link("/legal/privacy-policy", "Privacy Policy")

    assert blog_score <= 0.0
    assert legal_score <= 0.0


def test_7_max_subpages_is_respected():
    """Test 7: Extracted subpages do not exceed MAX_SUBPAGES limit (max_subpages - 1)."""
    html = """
    <html>
      <body>
        <a href="/about">About Us</a>
        <a href="/team">Our Team</a>
        <a href="/pricing">Pricing</a>
        <a href="/contact">Contact</a>
        <a href="/careers">Careers</a>
        <a href="/solutions">Solutions</a>
      </body>
    </html>
    """
    subpages = extract_ranked_internal_links(html, "https://postman.com", max_subpages=5)
    # Total selected subpages must be at most max_subpages - 1 = 4
    assert len(subpages) <= 4
    assert len(subpages) == 4


def test_8_external_links_are_excluded():
    """Test 8: External links are filtered out during link discovery."""
    html = """
    <html>
      <body>
        <a href="/about">About Us</a>
        <a href="https://twitter.com/postman">Twitter Profile</a>
        <a href="https://github.com/postman">GitHub Repo</a>
      </body>
    </html>
    """
    subpages = extract_ranked_internal_links(html, "https://postman.com", max_subpages=5)
    assert "https://twitter.com/postman" not in subpages
    assert "https://github.com/postman" not in subpages
    assert "https://postman.com/about" in subpages


def test_9_non_navigational_protocols_excluded():
    """Test 9: mailto, tel, and javascript links return None during normalization."""
    base = "https://postman.com"
    assert normalize_url("mailto:support@postman.com", base) is None
    assert normalize_url("tel:+18005550199", base) is None
    assert normalize_url("javascript:void(0)", base) is None
    assert normalize_url("data:image/png;base64,123", base) is None


def test_10_bot_block_detection():
    """Test 10: Bot block detection accurately identifies challenge pages."""
    assert detect_bot_block(403, "Access Denied", "Cloudflare security check") is True
    assert detect_bot_block(503, "Just a moment...", "Checking your browser before accessing") is True
    assert detect_bot_block(200, "Postman API Platform", "<html>Welcome to Postman</html>") is False


@pytest.mark.asyncio
async def test_11_timeout_handling_in_crawler():
    """Test 11: Timeout error on page fetch is caught and returned as structured result."""
    crawler = CompanyCrawler(max_subpages=3)
    mock_bm = create_mock_browser_manager()

    # Mock fetch_page returning a timeout ScrapedPage
    timeout_page = ScrapedPage(
        source_url="https://postman.com",
        final_url="https://postman.com",
        error_message="Fetch failed: Timeout 15000ms exceeded",
    )

    with patch.object(crawler, "fetch_page", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = timeout_page
        res = await crawler.crawl_domain("postman.com", browser_manager=mock_bm)

        assert res.status == "error"
        assert res.pages_scraped_count == 0
        assert "Timeout" in res.error_message


@pytest.mark.asyncio
async def test_12_http_error_handling():
    """Test 12: HTTP 404/500 errors produce structured DomainResult."""
    crawler = CompanyCrawler(max_subpages=3)
    mock_bm = create_mock_browser_manager()

    error_page = ScrapedPage(
        source_url="https://postman.com",
        final_url="https://postman.com",
        status_code=404,
        error_message="HTTP 404 Not Found",
    )

    with patch.object(crawler, "fetch_page", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = error_page
        res = await crawler.crawl_domain("postman.com", browser_manager=mock_bm)

        assert res.status == "error"
        assert res.pages_scraped_count == 0


@pytest.mark.asyncio
async def test_13_subpage_failure_does_not_halt_crawl():
    """Test 13: Failure on one subpage does not stop other subpages from being crawled."""
    crawler = CompanyCrawler(max_subpages=3)
    mock_bm = create_mock_browser_manager()

    homepage = ScrapedPage(
        source_url="https://postman.com",
        final_url="https://postman.com",
        status_code=200,
        page_title="Postman API Platform",
        raw_html='<html><body><a href="/about">About</a><a href="/team">Team</a></body></html>',
    )
    about_page = ScrapedPage(
        source_url="https://postman.com/about",
        final_url="https://postman.com/about",
        status_code=200,
        page_title="About Postman",
        raw_html="<html><body>About us text</body></html>",
    )
    team_page = ScrapedPage(
        source_url="https://postman.com/team",
        final_url="https://postman.com/team",
        error_message="Timeout loading team page",
    )

    async def side_effect(page_obj, url):
        if "about" in url:
            return about_page
        elif "team" in url:
            return team_page
        return homepage

    with patch.object(crawler, "fetch_page", side_effect=side_effect):
        res = await crawler.crawl_domain("postman.com", browser_manager=mock_bm)

        assert res.status == "partial"
        assert res.pages_scraped_count == 2  # homepage + about
        assert len(res.scraped_pages) == 3


def test_14_empty_or_no_relevant_links_handled_gracefully():
    """Test 14: HTML with no internal links returns empty candidate list without error."""
    html_no_links = "<html><body><h1>No links here</h1></body></html>"
    subpages = extract_ranked_internal_links(html_no_links, "https://postman.com", max_subpages=5)
    assert subpages == []


@pytest.mark.asyncio
async def test_15_redirect_and_final_url_handling():
    """Test 15: Redirected URLs update final_url correctly in ScrapedPage."""
    crawler = CompanyCrawler(max_subpages=2)
    mock_bm = create_mock_browser_manager()

    redirected_page = ScrapedPage(
        source_url="https://postman.com",
        final_url="https://www.postman.com/home",
        status_code=200,
        page_title="Postman Home",
        raw_html="<html><body>Home page</body></html>",
    )

    with patch.object(crawler, "fetch_page", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = redirected_page
        res = await crawler.crawl_domain("postman.com", browser_manager=mock_bm)

        assert res.status == "success"
        assert res.scraped_pages[0].source_url == "https://postman.com"
        assert res.scraped_pages[0].final_url == "https://www.postman.com/home"
