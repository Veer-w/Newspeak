import logging
from newspeak.config import Config
from newspeak.llm.provider import (
    LLMProvider,
    GeminiProvider,
    MockLLMProvider,
    FallbackProvider,
    RankedItemSchema,
    GeminiNewsRankingSchema,
    build_news_items,
)
from newspeak.llm.heuristic import (
    HeuristicRankingProvider,
    select_top_candidates,
    score_article,
)
from newspeak.llm.ollama import OllamaProvider

logger = logging.getLogger(__name__)


def build_llm_provider(config: Config, mock: bool = False) -> LLMProvider:
    """Select and assemble the ranking provider from config (single source of truth).

    Backends (via LLM_BACKEND):
      - "gemini" (default): Gemini primary with the zero-cost heuristic as auto-backup.
      - "ollama": local Ollama primary with heuristic backup (local runs only).
      - "heuristic": LLM-free heuristic only.
    `mock=True` overrides everything with the offline MockLLMProvider.
    """
    if mock:
        return MockLLMProvider()

    backend = (config.llm_backend or "gemini").strip().lower()

    if backend == "heuristic":
        logger.info("LLM backend: heuristic (LLM-free).")
        return HeuristicRankingProvider(config.keywords)

    if backend == "ollama":
        logger.info(f"LLM backend: local Ollama (model={config.ollama_model}), heuristic backup.")
        return FallbackProvider(
            primary=OllamaProvider(model=config.ollama_model, host=config.ollama_host),
            backup=HeuristicRankingProvider(config.keywords),
        )

    # Default: Gemini, with the always-available heuristic as an automatic backup so the
    # newsletter still ships if Gemini is rate-limited / quota-exhausted.
    if not config.gemini_api_key:
        raise ValueError(
            "GEMINI_API_KEY is not set. Provide it, use --mock-llm, or set "
            "LLM_BACKEND=heuristic (or LLM_BACKEND=ollama for a local model)."
        )
    logger.info("LLM backend: Gemini primary, heuristic backup.")
    return FallbackProvider(
        primary=GeminiProvider(api_key=config.gemini_api_key, max_candidates=config.llm_max_candidates),
        backup=HeuristicRankingProvider(config.keywords),
    )


__all__ = [
    "LLMProvider",
    "GeminiProvider",
    "MockLLMProvider",
    "FallbackProvider",
    "HeuristicRankingProvider",
    "OllamaProvider",
    "build_llm_provider",
    "select_top_candidates",
    "score_article",
    "build_news_items",
    "RankedItemSchema",
    "GeminiNewsRankingSchema",
]
