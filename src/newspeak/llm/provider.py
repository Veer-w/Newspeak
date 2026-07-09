from abc import ABC, abstractmethod
from typing import Sequence
import logging
import json
from pydantic import BaseModel
from google import genai
from google.genai import types as genai_types
from google.genai.errors import APIError
from newspeak.types import Article, NewsItem

logger = logging.getLogger(__name__)

class GeminiNewsRankingSchema(BaseModel):
    """Pydantic schema for structured output from Gemini API."""
    items: list[NewsItem]


class LLMProvider(ABC):
    """Abstract Base Class for modular LLM integrations."""
    
    @abstractmethod
    async def rank_and_summarize(self, articles: Sequence[Article]) -> Sequence[NewsItem]:
        """Rank candidates, select the top 10 AI/ML news, and generate summaries."""
        pass


class GeminiProvider(LLMProvider):
    """Concrete LLM Provider using Google Gemini client via google-genai SDK."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    async def rank_and_summarize(self, articles: Sequence[Article]) -> Sequence[NewsItem]:
        if not articles:
            logger.warning("No articles provided to rank and summarize.")
            return []

        # Prepare prompt
        formatted_articles = []
        for i, art in enumerate(articles):
            formatted_articles.append(
                f"ID: {i}\n"
                f"Title: {art.title}\n"
                f"Source: {art.source}\n"
                f"URL: {art.url}\n"
                f"Description: {art.description}\n"
                "----------------------------------------"
            )
        
        articles_context = "\n".join(formatted_articles)

        system_instruction = (
            "You are a Senior Editor for a premier AI/ML newsletter. Your job is to select "
            "the top 10 most technically significant, impactful, and 100% verified news developments "
            "from the raw list of ingested articles provided.\n\n"
            "Strict Guidelines:\n"
            "1. Accuracy is paramount. Ensure the news is 100% true and based on actual technical developments, "
            "releases, or scientific papers. Avoid clickbait, hype, speculative rumors, or promotional pieces.\n"
            "2. Relevance: Filter out general tech news. Focus purely on machine learning, deep learning, NLP, "
            "computer vision, generative models, hardware breakthroughs (e.g., TPUs/GPUs), and agentic workflows.\n"
            "3. Deduplication: If multiple articles cover the same event, select the single best article with "
            "the highest score and technical details, then discard the duplicates.\n"
            "4. Output format: You must return exactly up to 10 items in the requested JSON structure.\n"
            "5. Summary: Write a clear, dense, 2-sentence summary of the technical contribution. Do not fluff."
        )

        user_content = (
            f"Here is the list of candidates to rank and summarize:\n\n{articles_context}\n\n"
            "Please select the top 10 articles, score them on a scale of 1.0 to 10.0, summarize them "
            "and output the result in the specified structured schema."
        )

        try:
            # Call Gemini API synchronously since client is sync, but we wrap in asyncio executor or call directly
            # Run in executor to prevent blocking the event loop
            import asyncio
            loop = asyncio.get_event_loop()
            
            def call_gemini():
                return self.client.models.generate_content(
                    model=self.model_name,
                    contents=user_content,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        response_schema=GeminiNewsRankingSchema,
                        temperature=0.1,  # Lower temperature for deterministic ranking/summarization
                    )
                )

            logger.info("Sending articles to Gemini for ranking and summarization...")
            response = await loop.run_in_executor(None, call_gemini)
            
            # Parse response
            result_json = json.loads(response.text)
            parsed_response = GeminiNewsRankingSchema.model_validate(result_json)
            
            # Sort items by score descending
            sorted_items = sorted(parsed_response.items, key=lambda x: x.score, reverse=True)
            return sorted_items[:10]

        except APIError as e:
            logger.error(f"Gemini API Error: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to generate summaries via Gemini: {e}")
            return []


class MockLLMProvider(LLMProvider):
    """Mock LLM Provider for local development and testing without API costs."""

    async def rank_and_summarize(self, articles: Sequence[Article]) -> Sequence[NewsItem]:
        logger.info("Mock LLM Provider: Mocking ranking and summarization...")
        # Take up to 10 articles, map them directly to NewsItems with placeholder summaries
        mock_items = []
        for i, art in enumerate(articles[:10]):
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
