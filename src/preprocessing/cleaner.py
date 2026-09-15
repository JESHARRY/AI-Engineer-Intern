"""DOM cleaner and text extraction module."""

import logging
import re
from typing import Optional
from bs4 import BeautifulSoup, Comment
from src.models import PageChunk, ScrapedPage

logger = logging.getLogger(__name__)

# Tags to strip completely along with their contents
UNWANTED_TAGS = [
    "script",
    "style",
    "noscript",
    "svg",
    "iframe",
    "canvas",
    "template",
    "nav",
    "footer",
]

# Common attribute patterns for UI boilerplate (cookie banners, popups, consent)
BOILERPLATE_ATTR_PATTERNS = re.compile(
    r"(cookie|consent|privacy-banner|gdpr|popup|modal-backdrop|newsletter-signup)",
    re.IGNORECASE,
)

# Block level HTML tags that warrant line breaks
BLOCK_TAGS = ["p", "div", "section", "article", "header", "main", "li", "tr", "br"]


def clean_dom_to_text(raw_html: str) -> str:
    """
    Parse raw HTML, strip non-content/boilerplate nodes, and extract clean text.

    - Removes script, style, svg, iframe, canvas, template, nav, footer tags.
    - Removes boilerplate UI elements (cookie banners, consent popups).
    - Preserves headers, article, main, section, and body content.
    - Normalizes excessive whitespace and converts headers to Markdown (# H1, ## H2).
    """
    if not raw_html or not raw_html.strip():
        return ""

    try:
        soup = BeautifulSoup(raw_html, "html.parser")

        # 1. Remove HTML comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        # 2. Remove unwanted non-content elements
        for tag_name in UNWANTED_TAGS:
            for element in soup.find_all(tag_name):
                element.decompose()

        # 3. Remove conservative boilerplate UI elements by id/class
        for element in soup.find_all(True):
            element_id = element.get("id") or ""
            element_class = " ".join(element.get("class") or [])
            if BOILERPLATE_ATTR_PATTERNS.search(element_id) or BOILERPLATE_ATTR_PATTERNS.search(element_class):
                element.decompose()

        # 4. Format headings as markdown headers
        for lvl in range(1, 7):
            for h in soup.find_all(f"h{lvl}"):
                text = h.get_text(strip=True)
                if text:
                    h.replace_with(f"\n\n{'#' * lvl} {text}\n\n")

        # 5. Insert newlines around block tags so inline tags (b, i, span, a) remain intact
        for b_tag in soup.find_all(BLOCK_TAGS):
            b_tag.insert_before("\n")
            b_tag.insert_after("\n")

        # 6. Extract text from body or root
        body = soup.find("body") or soup
        text_content = body.get_text()

        # 7. Normalize whitespace (collapse multiple spaces per line & excessive blank lines)
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text_content.splitlines()]
        cleaned_lines = []
        blank_count = 0
        for line in lines:
            if not line:
                blank_count += 1
                if blank_count <= 2:  # allow max 2 consecutive blank lines
                    cleaned_lines.append("")
            else:
                blank_count = 0
                cleaned_lines.append(line)

        result_text = "\n".join(cleaned_lines).strip()
        return result_text
    except Exception as e:
        logger.warning(f"Error scrubbing HTML: {e}")
        return ""


def clean_page(scraped_page: ScrapedPage) -> PageChunk:
    """Transform a raw ScrapedPage into a cleaned PageChunk with preserved source URL."""
    cleaned_text = clean_dom_to_text(scraped_page.raw_html)
    return PageChunk(
        source_url=scraped_page.final_url or scraped_page.source_url,
        cleaned_content=cleaned_text,
        section_title=scraped_page.page_title,
        token_count=0,
    )
