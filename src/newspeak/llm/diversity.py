import logging
from typing import Sequence
from urllib.parse import urlparse
from newspeak.types import NewsItem

logger = logging.getLogger(__name__)


def source_domain(url: str) -> str:
    """Pure: normalize a URL to a publisher-grouping key (registrable-ish domain).

    Groups items by publisher so the diversity cap counts sources, not feeds. The two
    arXiv feeds emit different `source` strings ("cs.AI updates…" vs "cs.LG updates…")
    but share the arxiv.org domain, so grouping must key on the URL, not the source name.
    Strips a leading "www." and common feed subdomains, then keeps the last two labels
    (e.g. rss.arxiv.org -> arxiv.org, feeds.venturebeat.com -> venturebeat.com).
    """
    netloc = urlparse(url).netloc.lower()
    if not netloc:
        return url.lower()
    # Drop port if present.
    netloc = netloc.split(":", 1)[0]
    labels = netloc.split(".")
    if labels and labels[0] in ("www", "rss", "feeds", "feed", "blog", "news"):
        labels = labels[1:]
    # Keep the last two labels as the grouping key (good enough for our feed set).
    return ".".join(labels[-2:]) if len(labels) >= 2 else ".".join(labels)


def enforce_source_diversity(
    items: Sequence[NewsItem],
    max_per_source: int,
) -> Sequence[NewsItem]:
    """Pure: cap how many items each publisher (by domain) may contribute.

    Walks the items in their given order (callers pass them already sorted by score, best
    first) and greedily keeps each one while its domain is under `max_per_source`. Skipped
    items are appended afterwards in their original order, so if there aren't enough
    diverse sources to fill the list the highest-scored leftovers still backfill the tail.
    """
    if max_per_source <= 0:
        return list(items)

    counts: dict[str, int] = {}
    kept: list[NewsItem] = []
    overflow: list[NewsItem] = []
    for item in items:
        domain = source_domain(item.url)
        if counts.get(domain, 0) < max_per_source:
            counts[domain] = counts.get(domain, 0) + 1
            kept.append(item)
        else:
            overflow.append(item)

    if overflow:
        logger.info(
            f"Diversity cap ({max_per_source}/source) deferred {len(overflow)} over-represented "
            f"items; backfilling from {len(counts)} distinct sources."
        )
    return kept + overflow
