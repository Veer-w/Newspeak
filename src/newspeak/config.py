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
    "https://rss.arxiv.org/rss/cs.AI",  # arXiv Artificial Intelligence
    "https://rss.arxiv.org/rss/cs.LG",  # arXiv Machine Learning
    "https://hnrss.github.io/active",  # Hacker News active threads (often high quality)
    "https://techcrunch.com/category/artificial-intelligence/feed/",  # TechCrunch AI
    "https://machinelearning.apple.com/rss.xml",  # Apple Machine Learning Blog
    "https://subconscious.substack.com/feed",  # Substack newsletters
)

DEFAULT_KEYWORDS = (
    "ai", "ml", "machine learning", "llm", "neural", "deep learning", "transformer",
    "artificial intelligence", "pytorch", "tensorflow", "gpt", "claude", "gemini",
    "llama", "generative ai", "diffusion", "rag", "fine-tuning", "agentic", "llms"
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
    hn_stories_limit: int = Field(default_factory=lambda: _get_env_int("HN_STORIES_LIMIT", 50))


def load_config() -> Config:
    """Pure-like loader function for config."""
    return Config()
