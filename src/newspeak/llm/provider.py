import asyncio
from abc import ABC, abstractmethod
from typing import Sequence
import logging
import json
from pydantic import BaseModel, Field
from google import genai
from google.genai import types as genai_types
from google.genai.errors import APIError
from newspeak.types import Article, NewsItem

logger = logging.getLogger(__name__)


class RankedItemSchema(BaseModel):
    """A single ranking decision from the LLM.

    The model only references the source article by ID and generates the score/summary/
    reason — it never emits the url, title, or source, so those can't be hallucinated.
    """
    id: int = Field(description="The integer ID of the selected source article from the candidate list")
    score: float = Field(description="Relevance and impact score from 1.0 (lowest) to 10.0 (highest)")
    summary: str = Field(description="A concise 2-sentence summary explaining the news and its technical significance")
    reason: str = Field(description="A brief explanation of why this is top news in AI/ML")


class GeminiNewsRankingSchema(BaseModel):
    """Pydantic schema for structured output shared by all LLM ranking backends."""
    items: list[RankedItemSchema]


# Free-tier defaults: keep the request well under per-minute token limits.
DEFAULT_MAX_CANDIDATES = 150      # cap how many articles we send in one request
MAX_DESCRIPTION_CHARS = 500       # trim long feed/abstract text (arXiv abstracts are big)
_RETRYABLE_STATUS = (429, 503)    # rate-limited / model overloaded — safe to retry

SYSTEM_INSTRUCTION = (
    "You are a Senior Editor for a premier AI/ML newsletter. Your job is to select the most "
    "significant, impactful, and 100% verified AI/ML developments from the raw list of ingested "
    "articles provided, and to assemble a balanced, readable digest.\n\n"
    "Strict Guidelines:\n"
    "1. Accuracy is paramount. Ensure the news is based on real developments — research papers, "
    "model/product releases, or credible industry reporting. Avoid clickbait, hype, speculative "
    "rumors, or purely promotional pieces.\n"
    "2. Relevance: Filter out non-AI general tech news. Keep the full breadth of AI/ML: research "
    "(ML, deep learning, NLP, computer vision, generative models), model and product launches, "
    "hardware breakthroughs (e.g., TPUs/GPUs), agentic workflows, and notable AI industry or "
    "community news. AI industry news is in scope — do not discard it in favor of papers.\n"
    "3. Diversity: Assemble a MIX of content types and sources. Do not fill the list with research "
    "papers, and do not let any single source or publisher dominate — favor variety across "
    "papers, releases, and industry/community news so the reader gets a rounded picture of the day.\n"
    "4. Deduplication: If multiple articles cover the same event, select the single best article "
    "with the highest score and detail, then discard the duplicates.\n"
    "5. Output format: Return the requested items in the JSON structure. For each item, return the "
    "integer `id` of the source article exactly as given — never invent or alter URLs, titles, or IDs.\n"
    "6. Summary: Write a clear, dense, 2-sentence summary of the development. Do not fluff."
)


def _truncate(text: str, limit: int) -> str:
    """Pure helper: trim text to a character budget, appending an ellipsis when cut."""
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def format_candidates(candidates: Sequence[Article], max_desc_chars: int = MAX_DESCRIPTION_CHARS) -> str:
    """Pure: render candidates into the ID-tagged prompt block shared by all LLM backends.

    The URL is intentionally omitted — the model returns an ID and we look the URL up
    ourselves (see build_news_items), so sending URLs just wastes tokens.
    """
    blocks = [
        (
            f"ID: {i}\n"
            f"Title: {art.title}\n"
            f"Source: {art.source}\n"
            f"Description: {_truncate(art.description, max_desc_chars)}\n"
            "----------------------------------------"
        )
        for i, art in enumerate(candidates)
    ]
    return "\n".join(blocks)


def build_ranking_prompt(
    candidates: Sequence[Article],
    max_desc_chars: int = MAX_DESCRIPTION_CHARS,
    top_n: int = 10,
) -> str:
    """Pure: build the user prompt for ranking the given candidates.

    Asks for the top `top_n` items (a diversity gate downstream trims to the final 10, so
    a larger `top_n` gives it backfill candidates from under-represented sources).
    """
    return (
        f"Here is the list of candidates to rank and summarize:\n\n"
        f"{format_candidates(candidates, max_desc_chars)}\n\n"
        f"Please select the top {top_n} articles by their `id`, score them on a scale of 1.0 to 10.0, "
        "write a summary and reason for each, and output the result in the specified structured schema. "
        "Aim for a diverse mix of sources and content types (research, releases, industry news) rather "
        "than many entries of the same kind."
    )


def build_news_items(
    ranked: Sequence[RankedItemSchema],
    articles: Sequence[Article],
) -> Sequence[NewsItem]:
    """Pure function: maps LLM ranking decisions back onto the trusted source articles by ID.

    url, title, and source always come from the original ingested Article, so a
    hallucinated or altered link can never reach the newsletter. IDs outside the
    candidate range are dropped.
    """
    items: list[NewsItem] = []
    for r in ranked:
        if r.id < 0 or r.id >= len(articles):
            logger.warning(f"LLM returned out-of-range article ID {r.id}; skipping.")
            continue
        art = articles[r.id]
        items.append(
            NewsItem(
                title=art.title,
                url=art.url,
                summary=r.summary,
                score=r.score,
                reason=r.reason,
                source=art.source,
            )
        )
    return items


class LLMProvider(ABC):
    """Abstract Base Class for modular LLM integrations."""

    @abstractmethod
    async def rank_and_summarize(self, articles: Sequence[Article]) -> Sequence[NewsItem]:
        """Rank candidates, select the top 10 AI/ML news, and generate summaries."""
        pass


class GeminiProvider(LLMProvider):
    """Concrete LLM Provider using Google Gemini client via google-genai SDK."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash",
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        max_retries: int = 3,
        retry_base_delay: float = 10.0,
        top_n: int = 10,
    ):
        if not api_key:
            raise ValueError("GEMINI_API_KEY must not be empty when constructing GeminiProvider.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.max_candidates = max_candidates
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.top_n = top_n

    async def rank_and_summarize(self, articles: Sequence[Article]) -> Sequence[NewsItem]:
        if not articles:
            logger.warning("No articles provided to rank and summarize.")
            return []

        # Defensive cap so the request stays within free-tier per-minute token limits.
        # IDs index into this capped list, so build_news_items maps against `candidates`.
        candidates = list(articles[: self.max_candidates])
        if len(articles) > len(candidates):
            logger.info(
                f"Capping {len(articles)} candidates to {len(candidates)} for the LLM request (free-tier token budget)."
            )

        user_content = build_ranking_prompt(candidates, top_n=self.top_n)

        def call_gemini():
            return self.client.models.generate_content(
                model=self.model_name,
                contents=user_content,
                config=genai_types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=GeminiNewsRankingSchema,
                    temperature=0.1,
                )
            )

        # Retry loop: free-tier requests can hit transient 429 (rate limit) / 503
        # (overloaded) responses. Back off and retry those; fail fast on anything else.
        for attempt in range(1, self.max_retries + 1):
            try:
                loop = asyncio.get_running_loop()
                logger.info(
                    f"Sending {len(candidates)} articles to Gemini for ranking "
                    f"(attempt {attempt}/{self.max_retries})..."
                )
                response = await loop.run_in_executor(None, call_gemini)

                # Guard against empty/blocked responses
                if not response.text:
                    logger.error("Gemini returned an empty response (possibly blocked or quota exceeded).")
                    return []

                result_json = json.loads(response.text)
                parsed_response = GeminiNewsRankingSchema.model_validate(result_json)

                # Map the LLM's ID-based ranking back onto the trusted source articles.
                news_items = build_news_items(parsed_response.items, candidates)

                # Sort by score descending and return the top N (the pipeline's diversity
                # gate trims this to the final 10, backfilling under-represented sources).
                sorted_items = sorted(news_items, key=lambda x: x.score, reverse=True)
                return sorted_items[: self.top_n]

            except APIError as e:
                code = getattr(e, "code", None)
                if code in _RETRYABLE_STATUS and attempt < self.max_retries:
                    delay = self.retry_base_delay * attempt
                    logger.warning(
                        f"Gemini returned {code} (rate-limited/overloaded); retrying in {delay:.0f}s "
                        f"(attempt {attempt}/{self.max_retries})..."
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.error(f"Gemini API Error (code {code}): {e}")
                return []
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Gemini JSON response: {e}")
                return []
            except Exception as e:
                logger.error(f"Failed to generate summaries via Gemini: {e}")
                return []

        logger.error(f"Gemini ranking failed after {self.max_retries} attempts (persistent rate limiting).")
        return []


class FallbackProvider(LLMProvider):
    """Tries a primary provider; if it yields no items, falls back to a backup.

    GeminiProvider swallows API errors and returns [] on failure, so an empty result is
    the signal to fall back (e.g. persistent rate limiting or quota exhaustion). This
    guarantees a newsletter still ships even when the primary LLM is unavailable.
    """

    def __init__(self, primary: LLMProvider, backup: LLMProvider):
        self.primary = primary
        self.backup = backup

    async def rank_and_summarize(self, articles: Sequence[Article]) -> Sequence[NewsItem]:
        items = await self.primary.rank_and_summarize(articles)
        if items:
            return items
        logger.warning(
            f"Primary provider {type(self.primary).__name__} returned no items; "
            f"falling back to {type(self.backup).__name__}."
        )
        return await self.backup.rank_and_summarize(articles)


class MockLLMProvider(LLMProvider):
    """Mock LLM Provider for local development and testing without API costs."""

    def __init__(self, top_n: int = 10):
        self.top_n = top_n

    async def rank_and_summarize(self, articles: Sequence[Article]) -> Sequence[NewsItem]:
        logger.info("Mock LLM Provider: Mocking ranking and summarization...")
        mock_items = []
        for i, art in enumerate(articles[: self.top_n]):
            mock_items.append(
                NewsItem(
                    title=art.title,
                    url=art.url,
                    summary=f"[MOCK SUMMARY] This is a mock 2-sentence summary of the article from {art.source}. It represents a significant technical breakthrough in the field of AI.",
                    score=10.0 - (i * 0.5),
                    reason=f"Selected for testing from {art.source}.",
                    source=art.source
                )
            )
        return mock_items
