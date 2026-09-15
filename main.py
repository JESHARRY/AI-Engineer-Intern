"""Main entry point for Autonomous Lead Enrichment Agent."""

import asyncio
import logging
import os
import sys
from rich.console import Console
from rich.table import Table

from config import get_settings
from src.pipeline import LeadEnrichmentPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")
console = Console()

# Target test domains from assignment
TARGET_DOMAINS = ["postman.com", "supabase.com", "vapi.ai"]


def print_summary_table(results):
    """Render a beautiful rich summary table for processed domains."""
    table = Table(title="Autonomous Lead Enrichment Agent - Results Summary")
    table.add_column("Domain", style="cyan", no_wrap=True)
    table.add_column("Status", style="bold")
    table.add_column("Pages Scraped", justify="right")
    table.add_column("Confidence", justify="right", style="green")
    table.add_column("Company Overview / Error", style="dim")

    for res in results:
        status_color = (
            "green"
            if res.status == "success"
            else "yellow"
            if res.status == "partial"
            else "red"
        )
        status_str = f"[{status_color}]{res.status.upper()}[/{status_color}]"

        conf_str = "N/A"
        summary_str = res.error_message or "No summary available"

        if res.extracted_data:
            conf_str = f"{res.extracted_data.data_confidence_score:.2f}"
            summary_str = res.extracted_data.company_overview or summary_str

        table.add_row(
            res.domain,
            status_str,
            str(res.pages_scraped_count),
            conf_str,
            summary_str[:80] + "..." if len(summary_str) > 80 else summary_str,
        )

    console.print(table)


async def async_main():
    """Asynchronous main CLI launcher."""
    settings = get_settings()

    console.print("[bold blue]Autonomous Lead Enrichment Agent[/bold blue]")
    console.print(f"Target Domains: {', '.join(TARGET_DOMAINS)}")
    console.print(f"Default Model: {settings.openai_model}")

    if not settings.openai_api_key or not settings.openai_api_key.strip():
        console.print(
            "\n[bold red]WARNING: OPENAI_API_KEY is not set in environment or .env file.[/bold red]"
        )
        console.print("Please copy .env.example to .env and provide your OpenAI API key to run live extractions.")
        sys.exit(1)

    pipeline = LeadEnrichmentPipeline()
    console.print("\n[yellow]Starting domain batch execution...[/yellow]\n")

    results = await pipeline.run_batch(TARGET_DOMAINS, output_path="output/output.json")

    console.print("\n")
    print_summary_table(results)
    console.print("\n[bold green]Output saved to: output/output.json[/bold green]")


def main():
    """Main CLI entry point."""
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Process interrupted by user. Exiting...[/yellow]")
        sys.exit(0)


if __name__ == "__main__":
    main()
