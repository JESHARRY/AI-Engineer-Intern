"""Website crawler with dynamic link ranking, bot-block detection, and error isolation."""

import logging
import re
from typing import List, Optional, Tuple, Dict, Set
from urllib.parse import urlparse, urljoin, unquote

from bs4 import BeautifulSoup

from config import get_settings
from src.models import DomainResult, ScrapedPage
from src.scraper.browser import BrowserManager

logger = logging.getLogger(__name__)

# Target keywords for relevance scoring
RELEVANCE_KEYWORDS = [
    "about",
    "company",
    "team",
    "leadership",
    "founder",
    "founders",
    "contact",
    "pricing",
    "product",
    "products",
    "solutions",
    "customers",
    "careers",
    "press",
    "info",
    "who-we-are",
]

# Irrelevant path patterns to penalize
IRRELEVANT_PATTERNS = [
    r"/blog(/.*)?$",
    r"/docs(/.*)?$",
    r"/legal(/.*)?$",
    r"/terms(/.*)?$",
    r"/privacy(/.*)?$",
    r"/cookie(/.*)?$",
    r"/login(/.*)?$",
    r"/signup(/.*)?$",
    r"/auth(/.*)?$",
    r"/status(/.*)?$",
    r"/changelog(/.*)?$",
    r"/rss(/.*)?$",
]

# Ignored static file extensions
IGNORED_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".pdf",
    ".css",
    ".js",
    ".ico",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".zip",
    ".tar",
    ".gz",
    ".mp4",
    ".mp3",
    ".avi",
    ".mov",
)

# Bot block content signatures
BOT_BLOCK_SIGNATURES = [
    "access denied",
    "cloudflare",
    "checking your browser",
    "verify you are human",
    "just a moment",
    "security check",
    "pardon our interruption",
    "captcha",
    "please enable cookies",
    "attention required",
]


def normalize_domain(domain: str) -> str:
    """Normalize a raw domain string into a valid base HTTPS URL."""
    cleaned = domain.strip().lower()
    cleaned = re.sub(r"^https?://", "", cleaned)
    cleaned = cleaned.rstrip("/")
    return f"https://{cleaned}"


def is_internal_url(url: str, base_domain: str) -> bool:
    """Check if a URL belongs to the same domain or subdomain."""
    try:
        parsed_target = urlparse(url)
        parsed_base = urlparse(normalize_domain(base_domain))

        target_host = parsed_target.netloc.lower().split(":")[0]
        base_host = parsed_base.netloc.lower().split(":")[0]

        # Strip 'www.' for host comparison
        target_clean = re.sub(r"^www\.", "", target_host)
        base_clean = re.sub(r"^www\.", "", base_host)

        if not target_clean:
            return False

        return target_clean == base_clean or target_clean.endswith("." + base_clean)
    except Exception:
        return False


def normalize_url(raw_url: str, base_url: str) -> Optional[str]:
    """
    Sanitize and normalize a link URL relative to a base URL.
    Returns None if the link is invalid, non-HTTP, or an asset URL.
    """
    if not raw_url or not raw_url.strip():
        return None

    stripped = raw_url.strip()

    # Reject non-navigational protocols
    if stripped.startswith(("mailto:", "tel:", "javascript:", "data:", "blob:", "ftp:")):
        return None

    try:
        # Resolve relative URL against base
        joined = urljoin(base_url, stripped)
        parsed = urlparse(joined)

        # Only HTTP / HTTPS allowed
        if parsed.scheme.lower() not in ("http", "https"):
            return None

        # Check ignored file extensions
        path_lower = parsed.path.lower()
        if any(path_lower.endswith(ext) for ext in IGNORED_EXTENSIONS):
            return None

        # Reconstruct normalized URL (drop query & fragment for clean page deduplication)
        norm_scheme = parsed.scheme.lower()
        norm_netloc = parsed.netloc.lower()
        norm_path = parsed.path.rstrip("/")
        if not norm_path:
            norm_path = "/"

        normalized = f"{norm_scheme}://{norm_netloc}{norm_path}"
        return normalized
    except Exception:
        return None


def score_link(path: str, anchor_text: str) -> float:
    """
    Calculate a deterministic relevance score for a link based on path and anchor text.

    Scoring:
    - Path segment exact match with keyword: +10.0
    - Path segment substring match with keyword: +7.0
    - Anchor text keyword match: +5.0
    - Irrelevant path pattern: -10.0
    """
    score = 0.0
    path_lower = path.lower()
    anchor_lower = anchor_text.lower().strip()

    # Penalize known irrelevant patterns
    for pattern in IRRELEVANT_PATTERNS:
        if re.search(pattern, path_lower):
            score -= 10.0
            break

    # Score URL path segments
    path_segments = [seg for seg in path_lower.split("/") if seg]
    for seg in path_segments:
        for kw in RELEVANCE_KEYWORDS:
            if seg == kw:
                score += 10.0
            elif kw in seg:
                score += 7.0

    # Score anchor text
    if anchor_lower:
        for kw in RELEVANCE_KEYWORDS:
            if kw in anchor_lower:
                score += 5.0

    return max(score, -10.0)


def detect_bot_block(
    status_code: Optional[int], page_title: Optional[str], raw_html: Optional[str]
) -> bool:
    """
    Detect potential bot blocks conservatively based on HTTP status codes and content signatures.
    """
    title_lower = (page_title or "").lower()
    html_lower = (raw_html or "").lower()

    # HTTP 403 or 429 status code + signature or empty body
    if status_code in (403, 429):
        if not raw_html or len(raw_html.strip()) < 500:
            return True
        if any(sig in title_lower or sig in html_lower for sig in BOT_BLOCK_SIGNATURES):
            return True

    # HTTP 503 with Cloudflare / challenge signature
    if status_code == 503 and any(
        sig in title_lower or sig in html_lower for sig in ("cloudflare", "checking your browser", "just a moment")
    ):
        return True

    # Page title strongly indicates bot block
    if any(title_lower == sig for sig in ("just a moment...", "access denied", "attention required!")):
        return True

    return False


def extract_ranked_internal_links(
    raw_html: str, base_url: str, max_subpages: int
) -> List[str]:
    """
    Parse HTML, extract internal links, score them deterministically, and return top ranked URLs.
    """
    if not raw_html or not raw_html.strip():
        return []

    soup = BeautifulSoup(raw_html, "html.parser")
    domain = urlparse(base_url).netloc

    scored_links: Dict[str, float] = {}

    for a_tag in soup.find_all("a", href=True):
        raw_href = a_tag.get("href")
        anchor_text = a_tag.get_text(separator=" ", strip=True)

        norm_url = normalize_url(raw_href, base_url)
        if not norm_url or not is_internal_url(norm_url, domain):
            continue

        # Skip link if it's identical to the base homepage URL
        if norm_url.rstrip("/") == base_url.rstrip("/"):
            continue

        url_path = urlparse(norm_url).path
        score = score_link(url_path, anchor_text)

        # Keep highest score for duplicate URLs
        if norm_url not in scored_links or score > scored_links[norm_url]:
            scored_links[norm_url] = score

    # Sort links by score in descending order
    sorted_candidates = sorted(
        scored_links.items(), key=lambda item: item[1], reverse=True
    )

    # Filter candidates: prefer positive scores, but fallback to non-negative if needed
    positive_candidates = [url for url, sc in sorted_candidates if sc > 0.0]
    if positive_candidates:
        return positive_candidates[: max_subpages - 1]

    neutral_candidates = [url for url, sc in sorted_candidates if sc >= 0.0]
    return neutral_candidates[: max_subpages - 1]


class CompanyCrawler:
    """Crawler for company domain homepages and relevant subpages."""

    def __init__(
        self,
        browser_manager: Optional[BrowserManager] = None,
        max_subpages: Optional[int] = None,
    ):
        settings = get_settings()
        self.browser_manager = browser_manager
        self.max_subpages = max_subpages or settings.max_subpages

    async def fetch_page(self, page_obj, url: str) -> ScrapedPage:
        """
        Fetch a single web page using Playwright and return a ScrapedPage model.
        Catches timeouts and network exceptions cleanly.
        """
        status_code: Optional[int] = None
        final_url: str = url
        page_title: Optional[str] = None
        raw_html: str = ""
        error_msg: Optional[str] = None
        is_blocked: bool = False

        try:
            response = await page_obj.goto(url, wait_until="domcontentloaded")
            if response:
                status_code = response.status
                final_url = response.url

            page_title = await page_obj.title()
            raw_html = await page_obj.content()

            # Check bot block
            if detect_bot_block(status_code, page_title, raw_html):
                logger.warning(f"[WARNING] Bot block detected for {url} (Status: {status_code})")
                is_blocked = True
                error_msg = f"Bot protection challenge detected (HTTP {status_code})"

        except Exception as e:
            error_msg = f"Fetch failed: {str(e)}"
            logger.warning(f"Error fetching page {url}: {e}")

        return ScrapedPage(
            source_url=url,
            final_url=final_url,
            status_code=status_code,
            page_title=page_title,
            raw_html=raw_html,
            error_message=error_msg,
            blocked_by_bot=is_blocked,
        )

    async def crawl_domain(
        self, domain: str, browser_manager: Optional[BrowserManager] = None
    ) -> DomainResult:
        """
        Crawl a company domain: homepage first, discover & score internal subpages, and crawl top subpages.
        Enforces error isolation so failures on individual subpages do not crash the crawl.
        """
        bm = browser_manager or self.browser_manager
        own_bm = False
        if not bm:
            bm = BrowserManager()
            own_bm = True

        scraped_pages: List[ScrapedPage] = []
        scraped_urls: List[str] = []
        homepage_url = normalize_domain(domain)

        try:
            if own_bm:
                await bm.start()

            page_obj = await bm.get_page()
            try:
                homepage_res = await self.fetch_page(page_obj, homepage_url)
            finally:
                await page_obj.close()

            scraped_pages.append(homepage_res)
            if not homepage_res.error_message and not homepage_res.blocked_by_bot:
                scraped_urls.append(homepage_res.final_url)

            # Check homepage bot block or critical failure
            if homepage_res.blocked_by_bot:
                return DomainResult(
                    domain=domain,
                    status="bot_blocked",
                    error_message=homepage_res.error_message,
                    blocked_by_bot=True,
                    pages_scraped_count=0,
                    scraped_urls=[],
                    scraped_pages=scraped_pages,
                )

            if homepage_res.error_message and not homepage_res.raw_html:
                return DomainResult(
                    domain=domain,
                    status="error",
                    error_message=homepage_res.error_message,
                    blocked_by_bot=False,
                    pages_scraped_count=0,
                    scraped_urls=[],
                    scraped_pages=scraped_pages,
                )

            # Discover subpages
            subpage_urls = extract_ranked_internal_links(
                raw_html=homepage_res.raw_html,
                base_url=homepage_res.final_url or homepage_url,
                max_subpages=self.max_subpages,
            )

            # Crawl top subpages with isolated exception handling
            for sub_url in subpage_urls:
                sub_page_obj = None
                try:
                    sub_page_obj = await bm.get_page()
                    sub_res = await self.fetch_page(sub_page_obj, sub_url)
                    scraped_pages.append(sub_res)
                    if not sub_res.error_message and not sub_res.blocked_by_bot:
                        scraped_urls.append(sub_res.final_url)
                except Exception as sub_exc:
                    logger.warning(f"Failed to fetch subpage {sub_url}: {sub_exc}")
                    scraped_pages.append(
                        ScrapedPage(
                            source_url=sub_url,
                            final_url=sub_url,
                            error_message=str(sub_exc),
                        )
                    )
                finally:
                    if sub_page_obj:
                        try:
                            await sub_page_obj.close()
                        except Exception:
                            pass

            # Determine overall domain crawl status
            successful_pages = [p for p in scraped_pages if not p.error_message]
            status = "success"
            if len(successful_pages) < len(scraped_pages):
                status = "partial"
            if len(successful_pages) == 0:
                status = "error"

            return DomainResult(
                domain=domain,
                status=status,
                pages_scraped_count=len(successful_pages),
                scraped_urls=scraped_urls,
                scraped_pages=scraped_pages,
            )

        except Exception as exc:
            logger.error(f"Unhandled error crawling domain {domain}: {exc}")
            return DomainResult(
                domain=domain,
                status="error",
                error_message=str(exc),
                pages_scraped_count=len(scraped_urls),
                scraped_urls=scraped_urls,
                scraped_pages=scraped_pages,
            )
        finally:
            if own_bm:
                await bm.close()
