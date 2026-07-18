import os
from typing import Sequence
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load .env file if it exists (useful for local development)
load_dotenv()

def _get_env_int(key: str, default: int) -> int:
    val = os.getenv(key, "")
    if not val.strip():
        return default
    try:
        return int(val)
    except ValueError:
        return default


DEFAULT_RSS_FEEDS = (
    # Research (high-volume — kept in check by RSS_MAX_PER_FEED + the per-source cap).
    "https://rss.arxiv.org/rss/cs.AI",  # arXiv Artificial Intelligence
    "https://rss.arxiv.org/rss/cs.LG",  # arXiv Machine Learning
    # Industry news / press.
    "https://techcrunch.com/category/artificial-intelligence/feed/",  # TechCrunch AI
    "https://venturebeat.com/category/ai/feed/",  # VentureBeat AI
    "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",  # The Verge AI
    "https://feeds.arstechnica.com/arstechnica/technology-lab",  # Ars Technica (tech/AI)
    # Lab / product blogs.
    "https://machinelearning.apple.com/rss.xml",  # Apple Machine Learning Blog
    "https://openai.com/blog/rss.xml",  # OpenAI Blog
    "https://deepmind.google/blog/rss.xml",  # Google DeepMind Blog
    "https://huggingface.co/blog/feed.xml",  # Hugging Face Blog
)

DEFAULT_KEYWORDS = (
    "ai", "ml", "machine learning", "llm", "neural", "deep learning", "transformer",
    "artificial intelligence", "pytorch", "tensorflow", "gpt", "claude", "gemini",
    "llama", "generative ai", "diffusion", "rag", "fine-tuning", "agentic", "llms",
    # Word-boundary matching (see sources/hn.py) means brand names must be listed
    # explicitly — "ai" no longer matches inside "OpenAI".
    "openai", "anthropic", "deepmind", "mistral", "cohere", "huggingface", "hugging face",
    "nvidia", "chatgpt", "copilot", "agent", "agents", "multimodal", "embeddings", "quantization",
)

class Config(BaseModel):
    """Frozen config loading and holding environment variables for Newspeak."""
    model_config = {"frozen": True}

    gemini_api_key: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    
    # Email Delivery Settings (Supports Resend or SMTP)
    resend_api_key: str = Field(default_factory=lambda: os.getenv("RESEND_API_KEY", ""))
    
    smtp_server: str = Field(default_factory=lambda: os.getenv("SMTP_SERVER", ""))
    smtp_port: int = Field(default_factory=lambda: _get_env_int("SMTP_PORT", 587))
    smtp_username: str = Field(default_factory=lambda: os.getenv("SMTP_USERNAME", ""))
    smtp_password: str = Field(default_factory=lambda: os.getenv("SMTP_PASSWORD", ""))
    smtp_from: str = Field(default_factory=lambda: os.getenv("SMTP_FROM", "newspeak@localhost"))

    # Email provider selection: "auto" (default), "resend", or "smtp".
    # Set EMAIL_PROVIDER=smtp to force SMTP even when RESEND_API_KEY is also set.
    email_provider: str = Field(default_factory=lambda: os.getenv("EMAIL_PROVIDER", "auto"))

    # Recipients (comma-separated list in env)
    recipients: Sequence[str] = Field(
        default_factory=lambda: [
            email.strip() 
            for email in os.getenv("RECIPIENTS", "").split(",") 
            if email.strip()
        ]
    )

    # Ingestion Config
    rss_feeds: Sequence[str] = Field(
        default_factory=lambda: [
            feed.strip() 
            for feed in os.getenv("RSS_FEEDS", "").split(",") 
            if feed.strip()
        ] or list(DEFAULT_RSS_FEEDS)
    )
    
    keywords: Sequence[str] = Field(
        default_factory=lambda: [
            kw.strip().lower() 
            for kw in os.getenv("KEYWORDS", "").split(",") 
            if kw.strip()
        ] or list(DEFAULT_KEYWORDS)
    )

    # Max stories to pull from Hacker News API
    hn_stories_limit: int = Field(default_factory=lambda: _get_env_int("HN_STORIES_LIMIT", 100))

    # Max articles taken from any single RSS feed per run. Stops a high-volume feed
    # (arXiv publishes 50-150 papers/day) from flooding the candidate pool and crowding
    # out news/blog sources before ranking.
    rss_max_per_feed: int = Field(default_factory=lambda: _get_env_int("RSS_MAX_PER_FEED", 20))

    # Max items any single publisher (grouped by URL domain) may occupy in the final
    # top 10. The diversity gate backfills freed slots with the next-best items from
    # under-represented sources, so the newsletter isn't 10/10 from one source (e.g. arXiv).
    max_per_source: int = Field(default_factory=lambda: _get_env_int("MAX_PER_SOURCE", 3))

    # How many ranked items the LLM/heuristic returns before the diversity trim. Must be
    # comfortably larger than 10 so the diversity gate has backfill candidates.
    llm_top_n: int = Field(default_factory=lambda: _get_env_int("LLM_TOP_N", 25))

    # Max candidates sent to the LLM in one request. Lower this if you hit free-tier
    # per-minute token limits; raise it (with a paid key) for broader coverage.
    llm_max_candidates: int = Field(default_factory=lambda: _get_env_int("LLM_MAX_CANDIDATES", 150))

    # Ranking backend: "gemini" (default, + heuristic backup), "ollama" (local), or
    # "heuristic" (LLM-free). See newspeak.llm.build_llm_provider.
    llm_backend: str = Field(default_factory=lambda: os.getenv("LLM_BACKEND", "gemini"))

    # Local Ollama settings (only used when LLM_BACKEND=ollama; intended for local runs).
    ollama_model: str = Field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.2"))
    ollama_host: str = Field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))


def load_config() -> Config:
    """Pure-like loader function for config."""
    return Config()
