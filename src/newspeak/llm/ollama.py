import json
import logging
from typing import Sequence
import httpx
from newspeak.types import Article, NewsItem
from newspeak.llm.provider import (
    LLMProvider,
    GeminiNewsRankingSchema,
    SYSTEM_INSTRUCTION,
    build_ranking_prompt,
    build_news_items,
)

logger = logging.getLogger(__name__)

# Local models have small context windows and run on CPU, so keep the request small.
OLLAMA_MAX_CANDIDATES = 40
OLLAMA_MAX_DESCRIPTION_CHARS = 300


class OllamaProvider(LLMProvider):
    """LLM provider backed by a local Ollama server (http://localhost:11434 by default).

    Intended for LOCAL runs — running/previewing the newsletter on your own machine
    without spending Gemini's free-tier quota. Not suitable for GitHub Actions, whose
    runners are CPU-only and would be far too slow.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        host: str = "http://localhost:11434",
        max_candidates: int = OLLAMA_MAX_CANDIDATES,
        timeout: float = 300.0,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.max_candidates = max_candidates
        self.timeout = timeout

    async def rank_and_summarize(self, articles: Sequence[Article]) -> Sequence[NewsItem]:
        if not articles:
            logger.warning("No articles provided to the Ollama ranker.")
            return []

        candidates = list(articles[: self.max_candidates])
        user_content = build_ranking_prompt(candidates, max_desc_chars=OLLAMA_MAX_DESCRIPTION_CHARS)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
            # Ollama structured outputs: constrain generation to our JSON schema.
            "format": GeminiNewsRankingSchema.model_json_schema(),
            "options": {"temperature": 0.1},
        }

        try:
            logger.info(f"Sending {len(candidates)} articles to local Ollama model '{self.model}'...")
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(f"{self.host}/api/chat", json=payload)

            if response.status_code != 200:
                logger.error(f"Ollama returned HTTP {response.status_code}: {response.text[:200]}")
                return []

            content = response.json().get("message", {}).get("content", "")
            if not content:
                logger.error("Ollama returned an empty response.")
                return []

            parsed = GeminiNewsRankingSchema.model_validate(json.loads(content))
            news_items = build_news_items(parsed.items, candidates)
            return sorted(news_items, key=lambda x: x.score, reverse=True)[:10]

        except httpx.HTTPError as e:
            logger.error(f"Failed to reach Ollama at {self.host} (is it running?): {e}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Ollama JSON response: {e}")
            return []
        except Exception as e:
            logger.error(f"Ollama ranking failed: {e}")
            return []
