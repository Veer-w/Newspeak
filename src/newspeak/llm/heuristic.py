import re
import logging
from typing import Sequence
from newspeak.types import Article, NewsItem
from newspeak.llm.provider import LLMProvider, _truncate

logger = logging.getLogger(__name__)

# Sources we trust more for "verified" AI/ML news, matched against the feed/source name.
_AUTHORITATIVE_SOURCE_HINTS = (
    "arxiv", "hugging face", "huggingface", "apple", "deepmind", "openai",
    "anthropic", "google", "nvidia", "hacker news",
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _keyword_hits(text: str, keywords: Sequence[str]) -> int:
    """Pure: count how many distinct keywords appear as whole words in the text."""
    if not keywords:
        return 0
    pattern = "|".join(re.escape(kw) for kw in keywords)
    return len(set(m.group(0).lower() for m in re.finditer(rf"\b(?:{pattern})\b", text, re.IGNORECASE)))


def score_article(article: Article, keywords: Sequence[str]) -> float:
    """Pure heuristic relevance score (1.0–10.0) used for pre-ranking and as an LLM-free backup.

    Rough signals only — this is a deterministic stand-in for the LLM editor, not a
    replacement for it. Higher is more relevant/substantial.
    """
    score = 5.0

    # AI/ML keyword relevance — strong signal in the title, weaker in the body.
    score += min(_keyword_hits(article.title, keywords), 4) * 1.0
    score += min(_keyword_hits(article.description, keywords), 5) * 0.3

    # Substance: articles with real content beat one-liners; near-empty ones are penalized.
    desc_len = len(article.description)
    if desc_len >= 200:
        score += 1.0
    elif desc_len < 40:
        score -= 1.5

    # Source authority.
    source_lower = article.source.lower()
    if any(hint in source_lower for hint in _AUTHORITATIVE_SOURCE_HINTS):
        score += 1.0

    # Clamp to the same 1.0–10.0 scale the LLM uses.
    return max(1.0, min(10.0, score))


def select_top_candidates(
    articles: Sequence[Article],
    keywords: Sequence[str],
    limit: int,
) -> Sequence[Article]:
    """Pure: pick the `limit` highest-scoring candidates to send to the LLM.

    Eases free-tier token load by sending the LLM the BEST N candidates rather than an
    arbitrary first N. Ordering is deterministic (score desc, then original order for ties).
    """
    if limit <= 0 or len(articles) <= limit:
        return list(articles)
    ranked = sorted(
        enumerate(articles),
        key=lambda pair: (score_article(pair[1], keywords), -pair[0]),
        reverse=True,
    )
    selected = [art for _, art in ranked[:limit]]
    logger.info(f"Pre-ranked {len(articles)} candidates down to the top {limit} by heuristic score.")
    return selected


def _summarize(article: Article) -> str:
    """Pure: derive a short summary from the article's own text (no LLM)."""
    text = article.description.strip()
    if not text:
        return f"Reported by {article.source}. See the original for details."
    sentences = _SENTENCE_SPLIT_RE.split(text)
    summary = " ".join(sentences[:2]).strip()
    return _truncate(summary, 320)


class HeuristicRankingProvider(LLMProvider):
    """Zero-cost, network-free LLM backup.

    Ranks by score_article and summarizes from the article's own text. Always available
    (works on GitHub Actions with no API key), so the newsletter still ships when the
    primary LLM is rate-limited or quota-exhausted — just without LLM-written summaries.
    """

    def __init__(self, keywords: Sequence[str], top_n: int = 10):
        self.keywords = keywords
        self.top_n = top_n

    async def rank_and_summarize(self, articles: Sequence[Article]) -> Sequence[NewsItem]:
        if not articles:
            logger.warning("No articles provided to the heuristic ranker.")
            return []

        logger.info(f"Heuristic ranking {len(articles)} articles (LLM-free backup)...")
        scored = sorted(articles, key=lambda a: score_article(a, self.keywords), reverse=True)

        return [
            NewsItem(
                title=art.title,
                url=art.url,
                summary=_summarize(art),
                score=round(score_article(art, self.keywords), 1),
                reason=f"Selected by heuristic relevance ranking from {art.source}.",
                source=art.source,
            )
            for art in scored[: self.top_n]
        ]
