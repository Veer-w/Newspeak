import asyncio
from datetime import datetime, timezone
import logging
from typing import Sequence
from newspeak.types import Article, NewsItem
from newspeak.sources import aggregate_rss_feeds, fetch_hn_stories
from newspeak.llm import LLMProvider, select_top_candidates, enforce_source_diversity
from newspeak.delivery import EmailDelivery
from newspeak.config import Config

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Helper to clean titles/descriptions for similarity matching."""
    return "".join(c for c in text.lower() if c.isalnum() or c.isspace())


def get_jaccard_similarity(text1: str, text2: str) -> float:
    """Pure Jaccard Similarity calculation between two text strings."""
    words1 = set(clean_text(text1).split())
    words2 = set(clean_text(text2).split())
    if not words1 or not words2:
        return 0.0
    return len(words1 & words2) / len(words1 | words2)


def deduplicate_articles(articles: Sequence[Article], similarity_threshold: float = 0.65) -> Sequence[Article]:
    """
    Pure function that deduplicates articles based on exact URL and title Jaccard similarity.
    When a near-duplicate is found with a longer description, it replaces the existing entry.
    """
    # Phase 1: exact URL deduplication — keep the one with the longer description
    unique_by_url: dict[str, Article] = {}
    for art in articles:
        if art.url not in unique_by_url:
            unique_by_url[art.url] = art
        else:
            existing = unique_by_url[art.url]
            if len(art.description) > len(existing.description):
                unique_by_url[art.url] = art

    # Phase 2: title similarity deduplication
    unique_articles: list[Article] = []
    for art in unique_by_url.values():
        is_duplicate = False
        for existing in unique_articles:
            similarity = get_jaccard_similarity(art.title, existing.title)
            if similarity >= similarity_threshold:
                is_duplicate = True
                # If the new article has a more detailed description, replace in-place
                if len(art.description) > len(existing.description):
                    idx = unique_articles.index(existing)
                    unique_articles[idx] = art
                break

        if not is_duplicate:
            unique_articles.append(art)

    return unique_articles


async def ingest_all_sources(config: Config) -> Sequence[Article]:
    """
    Asynchronously triggers all source fetchers concurrently.
    Uses return_exceptions=True so a failure in one source never kills the other.
    """
    logger.info("Starting ingestion of all sources...")

    rss_task = aggregate_rss_feeds(config.rss_feeds, max_per_feed=config.rss_max_per_feed)
    hn_task = fetch_hn_stories(config.hn_stories_limit, config.keywords)

    results = await asyncio.gather(rss_task, hn_task, return_exceptions=True)

    rss_results: Sequence[Article] = []
    hn_results: Sequence[Article] = []

    if isinstance(results[0], BaseException):
        logger.error(f"RSS ingestion raised an exception: {results[0]}")
    else:
        rss_results = results[0]

    if isinstance(results[1], BaseException):
        logger.error(f"HN ingestion raised an exception: {results[1]}")
    else:
        hn_results = results[1]

    combined = list(rss_results) + list(hn_results)
    logger.info(f"Ingested {len(combined)} total raw articles ({len(rss_results)} RSS, {len(hn_results)} HN).")
    return combined


async def run_newsletter_pipeline(
    config: Config,
    llm_provider: LLMProvider,
    delivery_client: EmailDelivery
) -> bool:
    """
    Coordinates the entire end-to-end pipeline:
    Ingest -> Deduplicate -> Rank & Summarize (LLM) -> Format & Deliver.
    """
    logger.info("Triggering Newspeak pipeline...")

    # Step 1: Ingestion
    raw_articles = await ingest_all_sources(config)
    if not raw_articles:
        logger.error("No articles ingested from any source. Pipeline aborted.")
        return False

    # Step 2: Deduplication (Pure logic)
    curated_articles = deduplicate_articles(raw_articles)
    logger.info(f"Deduplicated raw list down to {len(curated_articles)} unique candidates.")

    # Step 2.5: Heuristic pre-ranking — send the LLM the best N candidates (eases token load).
    candidates = select_top_candidates(curated_articles, config.keywords, config.llm_max_candidates)

    # Step 3: LLM Evaluation (Ranking & Summarization) — returns up to llm_top_n items.
    top_news = await llm_provider.rank_and_summarize(candidates)
    if not top_news:
        logger.error("LLM failed to return ranked news. Pipeline aborted.")
        return False

    # Step 3.5: Source diversity — cap items per publisher and trim to the final 10, so the
    # digest isn't dominated by a single high-volume source (e.g. arXiv). This is the single
    # authoritative diversity gate; it runs regardless of which LLM backend produced the list.
    top_news = list(enforce_source_diversity(top_news, config.max_per_source))[:10]

    logger.info(f"Successfully curated top {len(top_news)} news items.")

    # Step 4: Formatting & Delivery — use UTC to match the daily UTC cron schedule.
    date_str = datetime.now(timezone.utc).strftime("%B %d, %Y")
    success = await delivery_client.send_newsletter(
        date_str=date_str,
        items=top_news,
        recipients=config.recipients
    )

    if success:
        logger.info("Newspeak newsletter pipeline completed successfully!")
    else:
        logger.error("Newspeak newsletter delivery failed.")

    return success
