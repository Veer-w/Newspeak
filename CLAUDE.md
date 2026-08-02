# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Newspeak curates the top AI/ML news each day: it ingests articles from RSS feeds and Hacker News, deduplicates them (within a run *and* against previously sent stories), uses Google Gemini to rank and summarize the best few (`NEWSLETTER_SIZE`, default 7), renders an HTML newsletter, and emails it to subscribers. It runs as a GitHub Actions cron job (`.github/workflows/newsletter.yml`, every 2 days at 6:00 AM UTC) and also exposes a FastAPI HTTP interface.

## Commands

All commands run through `uv` and require `PYTHONPATH=src` (the package lives under `src/newspeak`).

```bash
uv sync                                                          # install deps + sync venv

# Pipeline runs (main.py)
PYTHONPATH=src uv run python src/newspeak/main.py --dry-run --mock-llm   # safe: no Gemini calls, no email; writes last_newsletter.html
PYTHONPATH=src uv run python src/newspeak/main.py --dry-run              # live Gemini, writes HTML instead of emailing
PYTHONPATH=src uv run python src/newspeak/main.py                        # full production run (live Gemini + live email)
PYTHONPATH=src uv run python src/newspeak/main.py --serve                # start FastAPI server (default 127.0.0.1:8000)

# Tests
PYTHONPATH=src uv run pytest                                     # full suite
PYTHONPATH=src uv run pytest tests/test_pipeline.py             # single file
PYTHONPATH=src uv run pytest tests/test_pipeline.py::test_name  # single test
```

Requires Python 3.14 (`.python-version`). There is no configured linter/formatter.

## Configuration

All config comes from environment variables, loaded via `python-dotenv` from a `.env` file (see README for the full list). Key toggles: `GEMINI_API_KEY` (ranking), `RECIPIENTS` (comma-separated), and email delivery which prefers `RESEND_API_KEY` and falls back to `SMTP_*`. `RSS_FEEDS`, `KEYWORDS`, and `HN_STORIES_LIMIT` override the defaults hardcoded in `config.py`.

## Architecture

The pipeline is a linear flow assembled from swappable, abstract-based components. `main.py` (CLI) and `api.py` (HTTP) are both thin entrypoints that select concrete implementations based on config/flags, then hand them to the same `run_newsletter_pipeline`.

**Pipeline stages** (`pipeline.py`): `ingest_all_sources` → `deduplicate_articles` → `filter_new_articles` (cross-run) → `LLMProvider.rank_and_summarize` → `EmailDelivery.send_newsletter` → `record_sent` (cross-run).

- **Ingestion** runs RSS and HN sources concurrently via `asyncio.gather(..., return_exceptions=True)` — one source failing never aborts the run. Blocking/sync SDK calls (Gemini, Resend, SMTP) are wrapped in `loop.run_in_executor` to stay non-blocking. Sources return raw `Article` objects.
- **Deduplication** is pure logic: exact-URL dedup, then title-based Jaccard similarity (threshold 0.65). Near-duplicates keep the entry with the longer description. The Jaccard/`clean_text` helpers live in `text.py` (re-exported from `pipeline.py`).
- **Cross-run dedup** (`history.py`): before ranking, `filter_new_articles` drops candidates already delivered in a prior run — matched by exact URL, or by a title Jaccard-similar (≥0.65) to a previously sent one. State persists in `sent_history.json` (path/retention via `HISTORY_FILE`/`HISTORY_RETENTION_DAYS`, default 30 days); the GitHub Actions workflow commits it back to the repo so it survives between ephemeral runs. Only a **real** successful delivery records history (`record_history=not dry_run`), so dry-runs/previews never poison it.
- **LLM ranking** (`llm/provider.py`) uses Gemini structured output (`response_schema`) to return `NewsItem`s; results are sorted by score and truncated to `llm_top_n`, then a diversity gate trims to `NEWSLETTER_SIZE` (default 7). The LLM prompt *also* does its own relevance filtering and dedup — so both the pure dedup step and Gemini are curation layers.
- **Delivery** (`delivery/email.py`) renders a single inline-CSS Jinja2 template (`HTML_TEMPLATE`) with autoescape on (LLM output is untrusted), then sends.

**Provider abstraction pattern**: Both LLM and delivery are ABCs with multiple implementations.
- `LLMProvider` → `GeminiProvider` | `HeuristicRankingProvider` (LLM-free, pure) | `OllamaProvider` (local) | `MockLLMProvider`, plus `FallbackProvider(primary, backup)` which delegates to `backup` when `primary` returns `[]`.
- LLM selection is centralized in **`build_llm_provider(config, mock)`** (`llm/__init__.py`) — the single source of truth, driven by `LLM_BACKEND` (`gemini`|`ollama`|`heuristic`). Both `main.py` and `api.py` call it; don't re-add per-entrypoint selection logic. Default `gemini` is wrapped as `FallbackProvider(Gemini, Heuristic)` so a rate-limited/quota-exhausted Gemini still ships a newsletter.
- `EmailDelivery` → `ResendDelivery` | `SMTPDelivery` | `MockDelivery` (writes to a local HTML file; used for dry runs). Delivery selection is *still* duplicated in `main.py`/`api.py`.

**LLM cost/free-tier controls** (all pure, in `llm/`): `select_top_candidates` (pre-ranks by `score_article` so the LLM gets the best `LLM_MAX_CANDIDATES`, not an arbitrary slice); `GeminiProvider` caps candidates, truncates descriptions, omits URLs from the prompt, and retries `429`/`503` with backoff. The LLM returns only an article `id`; `build_news_items` maps it back to the trusted `Article` so url/title/source can't be hallucinated.

**Data models** (`types.py`) are all frozen Pydantic models: `Article` (raw ingested), `NewsItem` (curated/ranked, doubles as the Gemini response schema), `Newsletter`.

**Entrypoint note**: `main()` keeps `argparse` and `uvicorn.run()` (which manages its own event loop) *outside* `asyncio.run()`; only the pipeline path enters `asyncio.run()`. This avoids "event loop is already running" in `--serve` mode.

**FastAPI endpoints** (`api.py`): `GET /health`, `GET /preview` (renders HTML directly, `?mock_llm=`), `POST /trigger` (runs full pipeline, `?mock_llm=&dry_run=`).

**Do not commit 
