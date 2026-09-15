# Development Plan: Autonomous Lead Enrichment Agent

## 1. Requirement Analysis & Core Principles

The objective is to build a **robust, modular, resilient Autonomous Lead Enrichment Agent suitable for technical evaluation** that takes a list of company domains (specifically `postman.com`, `supabase.com`, and `vapi.ai`), fetches their homepages and dynamically discovered subpages, preprocesses content with source attribution and relevance-aware token budgeting, and uses OpenAI's API with Pydantic structured output to extract company profiles, ICP, public contacts, leadership members (with names, roles, LinkedIn URLs), and a deterministic confidence score.

### Core Architectural Principles & Invariants
1. **Pipeline Domain Isolation Invariant**: Domain isolation is an explicit system invariant. A failure for one domain (404, 500, timeout, bot block, API error) must **never** terminate processing of other domains. Every domain yields a successful, partial, or structured error result.
2. **Dynamic Link Discovery & Ranking**: Discover internal links on homepages and rank them using URL paths, anchor text, and relevance signals (keywords like `about`, `team`, `company`, `contact`, `pricing`, `product`, `leadership`, `founders`, etc.) rather than relying on strict URL whitelists.
3. **Graceful Bot-Block Handling (No Circumvention)**: Detect HTTP 403/429/Cloudflare challenge blocks, log the warning, attempt single normal navigation retry, and if still blocked, return a structured partial/error result and continue processing other domains. No CAPTCHA solving or anti-bot circumvention.
4. **Deterministic Confidence Score (Independent of LLM)**: Calculate `data_confidence_score` (0.0 to 1.0) deterministically after extraction and validation. The LLM does not generate or influence the confidence score.
5. **Evidence-Grounded Extraction & Source Traceability**: Enforce strict system prompts requiring every extracted detail to be grounded in the provided text (return empty values/lists if unavailable). Retain source URL on internal `PageChunk(url=..., content=...)` objects for debugging traceability.
6. **Relevance-Aware Token Budgeting**: Prioritize content chunks from high-value pages (Company, Team, Contact, Pricing, ICP) before capping total context tokens.
7. **Dedicated Output Writer Module**: Isolate output formatting and persistence (`output/output.json`) into `src/output/writer.py`.
8. **Current SDK Integration**: Use the installed OpenAI Python SDK's standard Pydantic/structured-output mechanism (`client.beta.chat.completions.parse` / `response_format`).
9. **Simple Async Orchestration**: Sequential or light async domain orchestration without complex worker queues, event buses, or multi-agent overhead.
10. **Core-First Scope**: No search engine/LinkedIn discovery, LangGraph, Browser-Use, or complex multi-agent frameworks until the core pipeline is fully working and verified.

---

## 2. System Architecture

```
                                  [ Domain Input List ]
                                           │
                                           ▼
                              ┌─────────────────────────┐
                              │    Pipeline Engine      │
                              │ (Domain Isolation Loop) │
                              └────────────┬────────────┘
                                           │
       ┌───────────────────────────────────┼───────────────────────────────────┐
       │                                   │                                   │
       ▼                                   ▼                                   ▼
┌───────────────────────┐       ┌───────────────────────┐       ┌───────────────────────┐
│ Dynamic Scraper       │       │ Preprocessing Engine  │       │ LLM Extractor         │
│ - Playwright Async    │ ────> │ - DOM Scrubbing       │ ────> │ - OpenAI SDK          │
│ - Dynamic Link Ranker │       │ - Source Chunking     │       │ - Pydantic Parse      │
│ - Bot-Block Detect    │       │ - Relevance Budgeting │       │ - Evidence Grounding  │
└───────────────────────┘       └───────────────────────┘       └───────────┬───────────┘
                                                                            │
                                                                            ▼
                                                                ┌───────────────────────┐
                                                                │ Confidence Scorer     │
                                                                │ - Deterministic Rules │
                                                                │ - Post-Extract Matrix │
                                                                └───────────┬───────────┘
                                                                            │
                                                                            ▼
                                                                ┌───────────────────────┐
                                                                │ Output Writer         │
                                                                │ - src/output/writer.py│
                                                                │ - output/output.json  │
                                                                └───────────────────────┘
```

---

## 3. Project Directory Structure

```
AI Engineer Intern_Assessment/
├── .env.example
├── .gitignore
├── README.md
├── DEVELOPMENT_PLAN.md         # Updated staged implementation reference
├── pyproject.toml
├── requirements.txt
├── main.py                     # CLI entrypoint
├── config.py                   # Environment settings & Pydantic Config
├── output/                     # Generated results directory
│   └── output.json             # Primary output deliverable
├── src/
│   ├── __init__.py
│   ├── models.py               # Pydantic schemas (TeamMember, LeadEnrichmentResult, PageChunk, etc.)
│   ├── confidence.py           # Deterministic completeness & evidence-based confidence scorer
│   ├── scraper/
│   │   ├── __init__.py
│   │   ├── browser.py          # Playwright async browser context manager
│   │   └── crawler.py          # Homepage fetching, link ranking & subpage crawling
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── cleaner.py          # DOM scrubbing & clean text/markdown extraction
│   │   ├── deduplicator.py     # Content deduplication & source chunk management
│   │   └── token_budget.py     # Relevance-aware token prioritization & budget capping
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py           # OpenAI SDK wrapper with Pydantic structured output parse
│   │   ├── prompts.py          # System prompts for evidence-grounded extraction
│   │   └── cost_tracker.py     # Token usage and estimated API cost calculation
│   ├── output/
│   │   ├── __init__.py
│   │   └── writer.py           # Dedicated output writer for JSON/CSV persistence
│   └── pipeline.py             # Main pipeline orchestrator (Domain Isolation Loop)
└── tests/
    ├── __init__.py
    ├── conftest.py             # Shared fixtures and mock HTML payloads
    ├── test_cleaner.py         # Unit tests for preprocessing & token budget
    ├── test_crawler.py         # Unit tests for link ranking & discovery
    ├── test_confidence.py      # Unit tests for confidence score calculator
    ├── test_llm_extractor.py   # Unit tests for LLM parser & mock responses
    ├── test_writer.py          # Unit tests for output writer
    └── test_pipeline.py       # Integration tests for end-to-end execution
```

---

## 4. Module Responsibilities

| Module | File Path | Responsibility |
| :--- | :--- | :--- |
| **Config** | `config.py` | Load `.env`, manage constants (timeouts, max subpages, token limit cap, model choice). |
| **Models** | `src/models.py` | Define Pydantic models: `TeamMember`, `LeadEnrichmentResult`, `PageChunk`, `ScrapedPage`, `DomainResult`. |
| **Confidence Scorer** | `src/confidence.py` | Calculate deterministic `data_confidence_score` (0.0 to 1.0) post-extraction based on completeness matrix. Independent of LLM. |
| **Browser Manager** | `src/scraper/browser.py` | Manage Playwright async browser context, user-agent headers, and clean teardown. |
| **Crawler** | `src/scraper/crawler.py` | Fetch homepage, score/rank internal links via path & anchor text signals, crawl top subpages, detect bot blocks. |
| **DOM Cleaner** | `src/preprocessing/cleaner.py` | Scrub scripts, styles, SVGs, navbars, footers, boilerplate; extract clean body text chunks with source URL retention. |
| **Deduplicator** | `src/preprocessing/deduplicator.py` | Deduplicate repeating text blocks across pages while preserving `PageChunk` source URLs. |
| **Token Budgeter** | `src/preprocessing/token_budget.py` | Rank content chunks by page relevance signals and assemble context under maximum token cap (e.g., 8,000 tokens). |
| **LLM Client** | `src/llm/client.py` | Invoke OpenAI SDK with supported Pydantic structured output; handle API retries via `tenacity`. |
| **Prompts** | `src/llm/prompts.py` | Craft system prompts enforcing evidence-grounded factual extraction (return empty list/null when missing). |
| **Cost Tracker** | `src/llm/cost_tracker.py` | Calculate token usage and estimated API cost per domain and total batch run. |
| **Output Writer** | `src/output/writer.py` | Format, serialize, and write pipeline results to `output/output.json`. |
| **Pipeline** | `src/pipeline.py` | Enforce domain isolation loop (Scrape -> Clean -> LLM -> Score -> Write). Never fails batch run on domain error. |
| **CLI Entry Point** | `main.py` | Parse arguments, render `rich` status progress and summary tables. |

---

## 5. Confidence Score Methodology (Post-Extraction & LLM-Independent)

The `data_confidence_score` $S \in [0.0, 1.0]$ is computed deterministically in `src/confidence.py` **after extraction and validation**, entirely independent of the LLM:

$$S = S_{\text{overview}} + S_{\text{icp}} + S_{\text{leadership}} + S_{\text{contacts}} + S_{\text{coverage}}$$

### Scoring Matrix:
1. **Company Overview Quality ($S_{\text{overview}}$, Max 0.25)**:
   - Valid 2-sentence summary extracted: `+0.25`
   - Missing/Empty: `+0.00`
2. **Target Audience / ICP Quality ($S_{\text{icp}}$, Max 0.20)**:
   - Clear ICP defined: `+0.20`
   - Missing/Empty: `+0.00`
3. **Leadership / Team Coverage ($S_{\text{leadership}}$, Max 0.25)**:
   - 2 or more team members extracted with name & role: `+0.25`
   - 1 team member extracted with name & role: `+0.15`
   - 0 team members: `+0.00`
4. **Public Contacts & LinkedIn Evidence ($S_{\text{contacts}}$, Max 0.15)**:
   - At least 1 public contact email present: `+0.10`
   - At least 1 LinkedIn profile URL present for leadership: `+0.05`
5. **Crawl & Source Coverage ($S_{\text{coverage}}$, Max 0.15)**:
   - Homepage + 2 or more relevant subpages successfully scraped: `+0.15`
   - Homepage + 1 subpage scraped: `+0.10`
   - Homepage only: `+0.05`
   - Scrape blocked/failed: `+0.00`

---

## 6. Pipeline Domain Isolation & Bot-Block Strategy

### Domain Isolation Pipeline Invariant
Domain isolation is guaranteed via explicit `try/except` domain isolation blocks:
```python
for domain in domains:
    try:
        result = await process_single_domain(domain)
    except Exception as exc:
        logger.error(f"Unhandled error processing {domain}: {exc}")
        result = DomainResult(domain=domain, status="error", error_message=str(exc))
    results.append(result)
```
A failure on one domain will **never** stop processing of other domains.

### Graceful Bot-Block Handling Protocol
1. **Detection**: Check status code (403, 429, 503) or challenge text ("Just a moment...", Cloudflare frame).
2. **Handling**: Single standard retry -> if still blocked, log warning, set `blocked_by_bot = True`, return structured partial/error result, and continue batch. No CAPTCHA or anti-bot circumvention.

---

## 7. Dynamic Subpage Discovery & Source Traceability

- **Subpage Ranking**: Homepage links scored by path & anchor text keywords (`about`, `team`, `company`, `contact`, `pricing`, `product`, `founders`, `leadership`, `who-we-are`, `careers`).
- **Internal Source Attribution**: `PageChunk(url=..., section=..., content=...)` retains exact source URLs for debugging and auditability.
- **Relevance-Aware Token Budget**: High-relevance content chunks prioritized before capping tokens (~8,000 max via `tiktoken`).

---

## 8. Staged Implementation Plan

### Stage 1: Project Setup, Data Models & Independent Confidence Engine
- Initialize repo, `pyproject.toml`, `requirements.txt`, `.env.example`, `.gitignore`.
- Implement `src/models.py` (`TeamMember`, `LeadEnrichmentResult`, `PageChunk`, `DomainResult`).
- Implement `src/confidence.py` (post-extraction deterministic completeness scorer) and `tests/test_confidence.py`.
- Implement `config.py`.

### Stage 2: Browser Scraping Engine & Dynamic Link Ranking
- Implement Playwright manager (`src/scraper/browser.py`).
- Implement crawler (`src/scraper/crawler.py`) with dynamic link ranking and bot-block detection.
- Write `tests/test_crawler.py`.

### Stage 3: Source-Attributed Preprocessing & Token Budgeting
- Implement DOM cleaner (`src/preprocessing/cleaner.py`) generating `PageChunk` with source URLs.
- Implement deduplicator & token budgeter (`src/preprocessing/token_budget.py`).
- Write `tests/test_cleaner.py`.

### Stage 4: Evidence-Grounded LLM Extractor & Cost Tracker
- Implement OpenAI SDK structured output parser (`src/llm/client.py`).
- Implement evidence-grounded prompt templates (`src/llm/prompts.py`).
- Implement cost tracker (`src/llm/cost_tracker.py`) and `tests/test_llm_extractor.py`.

### Stage 5: Dedicated Output Writer, Pipeline Orchestrator & CLI
- Implement `src/output/writer.py` (saves to `output/output.json`).
- Implement `src/pipeline.py` enforcing strict domain isolation loop.
- Build `main.py` CLI with `rich` console output.
- Generate primary deliverable `output/output.json` for target domains (`postman.com`, `supabase.com`, `vapi.ai`).

### Stage 6: Testing, Documentation & Final Verification
- Run complete test suite via `pytest`.
- Write comprehensive `README.md`.
- Final verification of all deliverables.
