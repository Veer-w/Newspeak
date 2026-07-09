from pydantic import BaseModel, Field

class Article(BaseModel):
    """Represents a raw article ingested from RSS or API feeds."""
    model_config = {"frozen": True}

    title: str
    url: str
    description: str
    source: str
    published_at: str | None = None

class NewsItem(BaseModel):
    """Represents a curated, ranked, and summarized news item from Gemini."""
    model_config = {"frozen": True}

    title: str = Field(description="The original headline of the article")
    url: str = Field(description="The source URL of the article")
    summary: str = Field(description="A concise 2-sentence summary explaining the news and its technical significance")
    score: float = Field(description="Relevance and impact score from 1.0 (lowest) to 10.0 (highest)")
    reason: str = Field(description="A brief explanation of why this is top news in AI/ML")
    source: str = Field(description="The source name of the article")

class Newsletter(BaseModel):
    """Represents the final newsletter payload containing the top news and recipients list."""
    model_config = {"frozen": True}

    date: str
    items: list[NewsItem]
    recipients: list[str]
