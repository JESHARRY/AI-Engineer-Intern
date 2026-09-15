"""End-to-end Autonomous Lead Enrichment Pipeline orchestrator."""

import logging
from typing import List, Optional
from config import get_settings
from src.models import DomainResult, LeadEnrichmentResult
from src.confidence import compute_result_confidence
from src.scraper.browser import BrowserManager
from src.scraper.crawler import CompanyCrawler
from src.preprocessing.cleaner import clean_page
from src.preprocessing.deduplicator import deduplicate_chunks
from src.preprocessing.token_budget import enforce_token_budget
from src.llm.client import LLMExtractor, sanitize_error_message
from src.output.writer import write_json_output

logger = logging.getLogger(__name__)


class LeadEnrichmentPipeline:
    """Orchestrates end-to-end domain crawling, preprocessing, LLM extraction, and confidence scoring."""

    def __init__(
        self,
        crawler: Optional[CompanyCrawler] = None,
        extractor: Optional[LLMExtractor] = None,
    ):
        self.crawler = crawler or CompanyCrawler()
        self.extractor = extractor or LLMExtractor()

    async def process_single_domain(
        self, domain: str, browser_manager: Optional[BrowserManager] = None
    ) -> DomainResult:
        """
        Process a single domain through Crawl -> Clean -> Deduplicate -> Budget -> LLM -> Score.
        Isolated so any step failure returns a structured DomainResult without crashing batch.
        """
        logger.info(f"--- Starting enrichment pipeline for domain: {domain} ---")

        try:
            # 1. Crawl Domain
            crawl_res = await self.crawler.crawl_domain(domain, browser_manager=browser_manager)

            # Handle bot block or complete crawl failure
            if crawl_res.blocked_by_bot or crawl_res.status == "bot_blocked":
                logger.warning(f"Domain {domain} was blocked by bot protection.")
                return DomainResult(
                    domain=domain,
                    status="bot_blocked",
                    error_message=crawl_res.error_message or "Blocked by website bot protection",
                    blocked_by_bot=True,
                    pages_scraped_count=0,
                    scraped_urls=[],
                    scraped_pages=crawl_res.scraped_pages,
                )

            if crawl_res.pages_scraped_count == 0 or not crawl_res.scraped_pages:
                logger.warning(f"Crawl failed for domain {domain}: {crawl_res.error_message}")
                return DomainResult(
                    domain=domain,
                    status="error",
                    error_message=crawl_res.error_message or "Failed to fetch any pages for domain",
                    pages_scraped_count=0,
                    scraped_urls=[],
                    scraped_pages=crawl_res.scraped_pages,
                )

            # 2. Preprocess Scraped Pages (Clean -> Deduplicate -> Relevance Budget)
            successful_pages = [p for p in crawl_res.scraped_pages if not p.error_message and p.raw_html]

            if not successful_pages:
                return DomainResult(
                    domain=domain,
                    status="error",
                    error_message="No valid HTML content retrieved from scraped pages",
                    pages_scraped_count=0,
                    scraped_urls=[],
                    scraped_pages=crawl_res.scraped_pages,
                )

            raw_chunks = [clean_page(p) for p in successful_pages]
            deduped_chunks = deduplicate_chunks(raw_chunks)
            budget_chunks, token_count = enforce_token_budget(deduped_chunks)

            if not budget_chunks:
                return DomainResult(
                    domain=domain,
                    status="error",
                    error_message="Preprocessing resulted in empty context",
                    pages_scraped_count=len(successful_pages),
                    scraped_urls=crawl_res.scraped_urls,
                    scraped_pages=crawl_res.scraped_pages,
                )

            logger.info(f"Preprocessing completed for {domain}: {len(budget_chunks)} chunks, ~{token_count} tokens")

            # 3. LLM Extraction
            try:
                extracted_lead_data, usage = self.extractor.extract_lead_data(budget_chunks)
                logger.info(f"LLM extraction succeeded for {domain}. Usage: {usage.total_tokens} tokens (${usage.estimated_cost_usd:.6f})")
            except Exception as llm_exc:
                clean_err = sanitize_error_message(str(llm_exc))
                logger.error(f"LLM extraction failed for domain {domain}: {clean_err}")
                return DomainResult(
                    domain=domain,
                    status="partial" if len(successful_pages) > 0 else "error",
                    error_message=f"LLM Extraction Error: {clean_err}",
                    pages_scraped_count=len(successful_pages),
                    scraped_urls=crawl_res.scraped_urls,
                    scraped_pages=crawl_res.scraped_pages,
                )

            # 4. Deterministic Confidence Scoring
            confidence_score = compute_result_confidence(
                result=extracted_lead_data,
                pages_scraped_count=len(successful_pages),
                is_blocked_or_failed=False,
            )
            extracted_lead_data.data_confidence_score = confidence_score
            logger.info(f"Calculated deterministic confidence score for {domain}: {confidence_score:.2f}")

            # Determine final status
            final_status = "success" if crawl_res.status == "success" else "partial"

            return DomainResult(
                domain=domain,
                status=final_status,
                extracted_data=extracted_lead_data,
                pages_scraped_count=len(successful_pages),
                scraped_urls=crawl_res.scraped_urls,
                scraped_pages=crawl_res.scraped_pages,
            )

        except Exception as exc:
            clean_err = sanitize_error_message(str(exc))
            logger.error(f"Pipeline processing failed for domain {domain}: {clean_err}")
            return DomainResult(
                domain=domain,
                status="error",
                error_message=clean_err,
            )

    async def run_batch(
        self, domains: List[str], output_path: str = "output/output.json"
    ) -> List[DomainResult]:
        """
        Run enrichment pipeline across a batch of domains with strict domain isolation.
        """
        results: List[DomainResult] = []

        async with BrowserManager() as bm:
            for domain in domains:
                try:
                    res = await self.process_single_domain(domain, browser_manager=bm)
                except Exception as exc:
                    clean_msg = sanitize_error_message(str(exc))
                    logger.error(f"Pipeline error processing domain {domain}: {clean_msg}")
                    res = DomainResult(
                        domain=domain,
                        status="error",
                        error_message=clean_msg,
                    )
                results.append(res)

        # Write output JSON
        write_json_output(results, output_path=output_path)
        return results
