# Autonomous Lead Enrichment Agent

A modular and resilient Python-based agent that enriches company leads from public web data.

The application accepts a list of company domains, dynamically crawls their public websites, extracts and cleans relevant content, and uses an OpenAI LLM with structured Pydantic outputs to generate company intelligence.

---

## Objective

The agent automates the following workflow:

```text
Company Domains
      ↓
Playwright Web Crawler
      ↓
Homepage + Relevant Subpages
      ↓
HTML / DOM Cleaning
      ↓
Content Deduplication
      ↓
Relevance Ranking + Token Budgeting
      ↓
OpenAI Structured Extraction
      ↓
Pydantic Validation
      ↓
Deterministic Confidence Scoring
      ↓
Structured JSON Output
