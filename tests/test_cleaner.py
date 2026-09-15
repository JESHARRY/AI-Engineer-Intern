"""Unit tests for DOM cleaner, deduplicator, and token budget modules."""

import pytest
from src.models import PageChunk, ScrapedPage
from src.preprocessing.cleaner import clean_dom_to_text, clean_page
from src.preprocessing.deduplicator import deduplicate_chunks
from src.preprocessing.token_budget import (
    count_tokens,
    score_content_block,
    enforce_token_budget,
)


def test_1_script_tags_removed():
    """Test 1: Script tags and their contents are completely stripped."""
    html = "<html><body><h1>Title</h1><script>var x = 1; alert(x);</script><p>Content</p></body></html>"
    text = clean_dom_to_text(html)
    assert "var x" not in text
    assert "alert" not in text
    assert "Title" in text
    assert "Content" in text


def test_2_style_tags_removed():
    """Test 2: Style tags and CSS rules are stripped."""
    html = "<html><body><style>body { background: red; }</style><p>Visible Text</p></body></html>"
    text = clean_dom_to_text(html)
    assert "background" not in text
    assert "Visible Text" in text


def test_3_svg_elements_removed():
    """Test 3: SVG markup is stripped."""
    html = "<html><body><div><svg><path d='M0 0h24v24H0z'/></svg><p>Real Text</p></div></body></html>"
    text = clean_dom_to_text(html)
    assert "path d=" not in text
    assert "Real Text" in text


def test_4_iframe_canvas_template_removed():
    """Test 4: iframe, canvas, and template elements are stripped."""
    html = """
    <html>
      <body>
        <iframe>Ad Frame</iframe>
        <canvas id="myCanvas"></canvas>
        <template><p>Template Content</p></template>
        <p>Main Body Text</p>
      </body>
    </html>
    """
    text = clean_dom_to_text(html)
    assert "Ad Frame" not in text
    assert "myCanvas" not in text
    assert "Template Content" not in text
    assert "Main Body Text" in text


def test_5_navigation_boilerplate_removed():
    """Test 5: Navigation nav elements are stripped."""
    html = "<html><body><nav><a href='/'>Home</a><a href='/about'>About</a></nav><p>Main Content</p></body></html>"
    text = clean_dom_to_text(html)
    assert "Home" not in text
    assert "Main Content" in text


def test_6_footer_boilerplate_removed():
    """Test 6: Footer elements are stripped."""
    html = "<html><body><main><p>Body Content</p></main><footer><p>© 2026 Company Inc</p></footer></body></html>"
    text = clean_dom_to_text(html)
    assert "© 2026 Company Inc" not in text
    assert "Body Content" in text


def test_7_meaningful_content_remains():
    """Test 7: Main, article, and section elements are preserved."""
    html = """
    <html>
      <body>
        <main>
          <article>
            <h2>Article Title</h2>
            <p>Article description and company profile.</p>
          </article>
        </main>
      </body>
    </html>
    """
    text = clean_dom_to_text(html)
    assert "Article Title" in text
    assert "Article description and company profile." in text


def test_8_header_content_not_blindly_removed():
    """Test 8: Header elements containing main titles/summaries are not stripped."""
    html = "<html><body><header><h1>Postman Platform</h1></header><p>Overview text</p></body></html>"
    text = clean_dom_to_text(html)
    assert "Postman Platform" in text
    assert "Overview text" in text


def test_9_whitespace_normalized():
    """Test 9: Excessive blank lines and whitespace are normalized."""
    html = "<html><body><p>Line 1</p><br/><br/><br/><br/><p>Line 2</p></body></html>"
    text = clean_dom_to_text(html)
    assert "\n\n\n\n" not in text
    assert "Line 1" in text
    assert "Line 2" in text


def test_10_html_tags_do_not_appear_in_cleaned_text():
    """Test 10: HTML tags are completely removed from final text."""
    html = "<html><body><div><p>Hello <b>World</b>, <i>Welcome</i>!</p></div></body></html>"
    text = clean_dom_to_text(html)
    assert "<b>" not in text
    assert "</b>" not in text
    assert "<p>" not in text
    assert "Hello World, Welcome!" in text


def test_11_source_url_preserved():
    """Test 11: clean_page retains original source_url in PageChunk."""
    page = ScrapedPage(
        source_url="https://supabase.com/about",
        final_url="https://supabase.com/about",
        raw_html="<html><body><h1>About Supabase</h1><p>Supabase is an open source Firebase alternative.</p></body></html>",
    )
    chunk = clean_page(page)
    assert chunk.source_url == "https://supabase.com/about"
    assert "Supabase is an open source Firebase alternative." in chunk.cleaned_content


def test_12_duplicate_content_across_pages_removed():
    """Test 12: Duplicate paragraph text appearing on multiple pages is removed from subsequent pages."""
    chunk1 = PageChunk(
        source_url="https://postman.com/about",
        cleaned_content="Postman is an API platform for developers.\n\nOur mission is to simplify API development.",
    )
    chunk2 = PageChunk(
        source_url="https://postman.com/team",
        cleaned_content="Postman is an API platform for developers.\n\nMeet our engineering team.",
    )

    deduped = deduplicate_chunks([chunk1, chunk2])
    assert len(deduped) == 2
    assert "Postman is an API platform for developers." in deduped[0].cleaned_content
    # Duplicate sentence should be omitted from chunk2
    assert "Postman is an API platform for developers." not in deduped[1].cleaned_content
    assert "Meet our engineering team." in deduped[1].cleaned_content


def test_13_unique_content_remains():
    """Test 13: Unique content blocks across chunks are preserved."""
    chunk1 = PageChunk(source_url="https://vapi.ai", cleaned_content="Vapi is an AI voice platform.")
    chunk2 = PageChunk(source_url="https://vapi.ai/pricing", cleaned_content="Pricing starts at $0.05/min.")

    deduped = deduplicate_chunks([chunk1, chunk2])
    assert len(deduped) == 2
    assert "Vapi is an AI voice platform." in deduped[0].cleaned_content
    assert "Pricing starts at $0.05/min." in deduped[1].cleaned_content


def test_14_relevant_page_higher_score():
    """Test 14: Pages with target keywords in URL receive higher relevance scores."""
    about_score = score_content_block("Company description.", "https://postman.com/about")
    misc_score = score_content_block("Company description.", "https://postman.com/random-page")

    assert about_score > misc_score


def test_15_relevant_sections_higher_priority():
    """Test 15: Content blocks with header keywords (Leadership, Pricing) receive higher scores."""
    leadership_text = "# Leadership Team\nOur founders and executive officers."
    generic_text = "Some general text about office locations."

    score_lead = score_content_block(leadership_text, "https://postman.com/team")
    score_gen = score_content_block(generic_text, "https://postman.com/team")

    assert score_lead > score_gen


def test_16_token_budget_respected():
    """Test 16: enforce_token_budget produces output with token count <= max_tokens."""
    chunk = PageChunk(
        source_url="https://supabase.com",
        cleaned_content="Supabase provides Postgres database, Authentication, Instant APIs, and Realtime subscriptions.\n\n" * 50,
    )
    final_chunks, total_tokens = enforce_token_budget([chunk], max_tokens=100)
    assert total_tokens <= 100


def test_17_token_count_never_exceeds_max_tokens():
    """Test 17: Final token count never exceeds budget even with massive text input."""
    massive_text = "Word " * 5000
    chunk = PageChunk(source_url="https://example.com", cleaned_content=massive_text)

    final_chunks, total_tokens = enforce_token_budget([chunk], max_tokens=500)
    assert total_tokens <= 500


def test_18_content_below_budget_not_truncated():
    """Test 18: Small content below budget limit is retained in full."""
    small_text = "Postman simplifies API development for over 30 million developers."
    chunk = PageChunk(source_url="https://postman.com", cleaned_content=small_text)

    final_chunks, total_tokens = enforce_token_budget([chunk], max_tokens=2000)
    assert len(final_chunks) == 1
    assert "30 million developers" in final_chunks[0].cleaned_content


def test_19_large_content_safely_reduced():
    """Test 19: Large multi-block content is safely reduced to fit budget cap."""
    chunk1 = PageChunk(
        source_url="https://postman.com/about",
        cleaned_content="# About Postman\nPostman is an API platform.\n\n" * 20,
    )
    chunk2 = PageChunk(
        source_url="https://postman.com/pricing",
        cleaned_content="# Pricing Plans\nFree tier, Basic tier, Enterprise tier.\n\n" * 20,
    )

    final_chunks, total_tokens = enforce_token_budget([chunk1, chunk2], max_tokens=200)
    assert total_tokens <= 200
    assert len(final_chunks) >= 1


def test_20_empty_html_does_not_crash():
    """Test 20: Processing empty HTML string returns empty text without crashing."""
    assert clean_dom_to_text("") == ""
    assert clean_dom_to_text("   ") == ""


def test_21_malformed_html_does_not_crash():
    """Test 21: Malformed or unclosed HTML parses safely without crashing."""
    malformed = "<div class='unclosed'><h2>Heading<p>Unclosed paragraph<div>Nested"
    text = clean_dom_to_text(malformed)
    assert "Heading" in text
    assert "Unclosed paragraph" in text


def test_22_unicode_and_non_english_handled_safely():
    """Test 22: UTF-8 Unicode characters (accented, Asian, emojis) clean safely."""
    unicode_html = "<html><body><h1>Société & 🚀 Tech</h1><p>日本語の文章与中文内容</p></body></html>"
    text = clean_dom_to_text(unicode_html)
    assert "Société" in text
    assert "🚀" in text
    assert "日本語の文章与中文内容" in text


def test_23_output_is_deterministic():
    """Test 23: Preprocessing functions produce identical outputs for identical inputs."""
    html = "<html><body><main><h1>Company Title</h1><p>Description text.</p></main></body></html>"
    res1 = clean_dom_to_text(html)
    res2 = clean_dom_to_text(html)
    assert res1 == res2

    chunk = PageChunk(source_url="https://test.com", cleaned_content=res1)
    b1, t1 = enforce_token_budget([chunk], max_tokens=500)
    b2, t2 = enforce_token_budget([chunk], max_tokens=500)
    assert t1 == t2
    assert b1[0].cleaned_content == b2[0].cleaned_content
