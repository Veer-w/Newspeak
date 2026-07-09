import asyncio
from datetime import datetime
import logging
from typing import Sequence
from newspeak.types import Article, NewsItem
from newspeak.sources import aggregate_rss_feeds, fetch_hn_stories
from newspeak.llm import LLMProvider
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
    """Pure function that deduplicates a list of articles based on URL and title similarity."""
    unique_by_url: dict[str, Article] = {}
    for art in articles:
        if art.url not in unique_by_url:
            unique_by_url[art.url] = art
        else:
            # If URL is duplicate, keep the one with longer description
            existing = unique_by_url[art.url]
            if len(art.description) > len(existing.description):
                unique_by_url[art.url] = art

    # Now deduplicate by title similarity
    unique_articles: list[Article] = []
    for art in unique_by_url.values():
        is_duplicate = False
        for existing in unique_articles:
            # Check URL similarity (e.g. same article on different feeds)
            if art.url == existing.url:
                is_duplicate = True
                break
                
            # Check title similarity
            similarity = get_jaccard_similarity(art.title, existing.title)
            if similarity >= similarity_threshold:
                is_duplicate = True
                # Keep the one with the more detailed description or preferred source
                if len(art.description) > len(existing.description):
                    # Replace the existing duplicate with this higher-detail article
                    idx = unique_articles.index(existing)
                    unique_articles[idx] = art
                break
                
        if not is_duplicate:
            unique_articles.append(art)
            
    return unique_articles


async def ingest_all_sources(config: Config) -> Sequence[Article]:
    """Asynchronously triggers all source fetchers and returns a combined list of raw articles."""
    logger.info("Starting ingestion of all sources...")
    
    # Trigger RSS and HN concurrently
    rss_task = aggregate_rss_feeds(config.rss_feeds)
    hn_task = fetch_hn_stories(config.hn_stories_limit, config.keywords)
    
    rss_results, hn_results = await asyncio.gather(rss_task, hn_task)
    
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
    
    This function separates external configurations and providers from the pipeline logic.
    """
    logger.info("Triggering Newspeak pipeline...")
    
    # Step 1: Ingestion
    raw_articles = await ingest_all_sources(config)
    if not raw_articles:
        logger.error("No articles ingested. Pipeline aborted.")
        return False
        
    # Step 2: Deduplication (Pure logic)
    curated_articles = deduplicate_articles(raw_articles)
    logger.info(f"Deduplicated raw list down to {len(curated_articles)} unique candidates.")
    
    # Step 3: LLM Evaluation (Ranking & Summarization)
    top_news = await llm_provider.rank_and_summarize(curated_articles)
    if not top_news:
        logger.error("Gemini failed to return ranked news. Pipeline aborted.")
        return False
        
    logger.info(f"Successfully curated top {len(top_news)} news items.")
    
    # Step 4: Formatting & Delivery
    date_str = datetime.now().strftime("%B %d, %Y")
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
