import asyncio
import pytest
from newspeak.types import Article, NewsItem
from newspeak.pipeline import clean_text, get_jaccard_similarity, deduplicate_articles
from newspeak.sources.rss import parse_feed_content, strip_html
from newspeak.sources.hn import contains_keywords
from newspeak.llm.provider import RankedItemSchema, build_news_items, LLMProvider, FallbackProvider
from newspeak.llm.heuristic import score_article, select_top_candidates, HeuristicRankingProvider

def test_clean_text() -> None:
    assert clean_text("Hello, World!") == "hello world"
    assert clean_text("GPT-4o & Gemini 2.5 Flash") == "gpt4o  gemini 25 flash"


def test_jaccard_similarity() -> None:
    # Exact match
    assert get_jaccard_similarity("Machine Learning", "machine learning") == 1.0
    
    # Partial match
    # words: {'introducing', 'the', 'new', 'model'} vs {'the', 'new', 'model', 'released'}
    # intersection: {'the', 'new', 'model'} (size 3)
    # union: {'introducing', 'the', 'new', 'model', 'released'} (size 5)
    # similarity: 3/5 = 0.6
    assert get_jaccard_similarity("Introducing the new model", "The new model released") == 0.6
    
    # No match
    assert get_jaccard_similarity("Apple iPhone", "Neural networks") == 0.0


def test_deduplicate_articles() -> None:
    art1 = Article(
        title="OpenAI Releases GPT-5",
        url="https://openai.com/gpt-5",
        description="A major new AI model from OpenAI.",
        source="OpenAI Blog"
    )
    art2 = Article(
        title="OpenAI Releases GPT-5 Model Today",
        url="https://techcrunch.com/openai-gpt-5",
        description="OpenAI officially announced GPT-5 today, bringing agentic capabilities.",
        source="TechCrunch"
    )
    art3 = Article(
        title="Google DeepMind announces AlphaFold 3",
        url="https://deepmind.google/alphafold3",
        description="A breakthrough in protein folding structure prediction.",
        source="Google DeepMind"
    )
    
    # art1 and art2 have very similar titles, art2 has a longer description.
    # Deduplication should keep art2 and art3.
    candidates = [art1, art2, art3]
    result = deduplicate_articles(candidates, similarity_threshold=0.6)
    
    assert len(result) == 2
    urls = {a.url for a in result}
    assert "https://techcrunch.com/openai-gpt-5" in urls
    assert "https://deepmind.google/alphafold3" in urls


def test_parse_rss_feed() -> None:
    rss_xml = b"""<?xml version="1.0" encoding="UTF-8" ?>
    <rss version="2.0">
    <channel>
        <title>AI News Feed</title>
        <link>https://ainews.example.com</link>
        <description>AI News</description>
        <item>
            <title>New AI Chip Released</title>
            <link>https://ainews.example.com/chip</link>
            <description>A new neural processing unit was announced today.</description>
            <pubDate>Thu, 09 Jul 2026 12:00:00 GMT</pubDate>
        </item>
    </channel>
    </rss>
    """
    articles = parse_feed_content(rss_xml, "https://ainews.example.com/feed")
    assert len(articles) == 1
    assert articles[0].title == "New AI Chip Released"
    assert articles[0].url == "https://ainews.example.com/chip"
    assert articles[0].source == "AI News Feed"


def test_parse_atom_feed() -> None:
    atom_xml = b"""<?xml version="1.0" encoding="utf-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
        <title>HF Papers</title>
        <link href="https://huggingface.co/papers"/>
        <entry>
            <title>Llama 4 Technical Report</title>
            <link rel="alternate" type="text/html" href="https://huggingface.co/papers/1234.5678"/>
            <summary>The technical documentation detailing Llama 4 pretraining.</summary>
            <updated>2026-07-09T18:00:00Z</updated>
        </entry>
    </feed>
    """
    articles = parse_feed_content(atom_xml, "https://huggingface.co/papers/feed")
    assert len(articles) == 1
    assert articles[0].title == "Llama 4 Technical Report"
    assert articles[0].url == "https://huggingface.co/papers/1234.5678"
    assert articles[0].source == "HF Papers"


def test_strip_html_decodes_entities() -> None:
    # Tags removed and residual (double-escaped) HTML entities decoded.
    assert strip_html("<p>AT&amp;T ships &#39;AI&#39; chip</p>") == "AT&T ships 'AI' chip"


def test_contains_keywords_word_boundary() -> None:
    keywords = ["ai", "ml", "rag", "openai"]
    # Whole-word matches
    assert contains_keywords("New AI model released", keywords) is True
    assert contains_keywords("OpenAI ships GPT-6", keywords) is True
    # Substrings of unrelated words must NOT match
    assert contains_keywords("Please check your email today", keywords) is False
    assert contains_keywords("Improving html storage layers", keywords) is False


def test_build_news_items_maps_by_id_and_drops_out_of_range() -> None:
    articles = [
        Article(title="Real Title A", url="https://a.example/x", description="d", source="Src A"),
        Article(title="Real Title B", url="https://b.example/y", description="d", source="Src B"),
    ]
    ranked = [
        RankedItemSchema(id=1, score=9.0, summary="s-b", reason="r-b"),
        # LLM tries to reference a non-existent / hallucinated article id -> dropped
        RankedItemSchema(id=99, score=8.0, summary="s-x", reason="r-x"),
        RankedItemSchema(id=0, score=7.0, summary="s-a", reason="r-a"),
    ]
    items = build_news_items(ranked, articles)

    assert len(items) == 2
    # url/title/source come from the trusted Article, not the model
    assert items[0].url == "https://b.example/y"
    assert items[0].title == "Real Title B"
    assert items[0].summary == "s-b"
    assert items[1].url == "https://a.example/x"


_KW = ["ai", "ml", "llm", "openai", "transformer"]


def test_score_article_prefers_relevant_and_substantial() -> None:
    relevant = Article(
        title="OpenAI releases new transformer LLM",
        url="https://a/x",
        description="A" * 250,
        source="Hacker News",
    )
    irrelevant = Article(
        title="A quiet walk in the park",
        url="https://b/y",
        description="short",
        source="Some Blog",
    )
    assert score_article(relevant, _KW) > score_article(irrelevant, _KW)
    # Score stays within the shared 1.0–10.0 scale.
    assert 1.0 <= score_article(irrelevant, _KW) <= 10.0
    assert 1.0 <= score_article(relevant, _KW) <= 10.0


def test_select_top_candidates_keeps_best_and_respects_limit() -> None:
    strong = Article(title="New AI LLM from OpenAI", url="https://a/x", description="B" * 250, source="arXiv")
    weak = Article(title="Cooking tips", url="https://b/y", description="", source="Blog")
    result = select_top_candidates([weak, strong], _KW, limit=1)
    assert len(result) == 1
    assert result[0].url == "https://a/x"
    # limit >= len is a no-op passthrough.
    assert len(select_top_candidates([weak, strong], _KW, limit=5)) == 2


def test_heuristic_provider_returns_items_without_llm() -> None:
    articles = [
        Article(title=f"AI model {i} from OpenAI", url=f"https://x/{i}", description="C" * 120, source="arXiv")
        for i in range(15)
    ]
    items = asyncio.run(HeuristicRankingProvider(_KW).rank_and_summarize(articles))
    assert len(items) == 10  # capped at top 10
    assert all(isinstance(it, NewsItem) and it.summary for it in items)


class _EmptyProvider(LLMProvider):
    async def rank_and_summarize(self, articles):
        return []


class _StubProvider(LLMProvider):
    def __init__(self, items):
        self._items = items

    async def rank_and_summarize(self, articles):
        return self._items


def test_fallback_uses_backup_when_primary_empty() -> None:
    backup_item = NewsItem(title="t", url="https://u", summary="s", score=7.0, reason="r", source="src")
    fb = FallbackProvider(primary=_EmptyProvider(), backup=_StubProvider([backup_item]))
    result = asyncio.run(fb.rank_and_summarize([]))
    assert result == [backup_item]


def test_fallback_skips_backup_when_primary_succeeds() -> None:
    primary_item = NewsItem(title="p", url="https://p", summary="s", score=9.0, reason="r", source="src")
    fb = FallbackProvider(primary=_StubProvider([primary_item]), backup=_EmptyProvider())
    result = asyncio.run(fb.rank_and_summarize([]))
    assert result == [primary_item]


def test_email_provider_smtp_forces_smtp_over_resend() -> None:
    from newspeak.config import Config
    from newspeak.delivery import build_delivery_client, SMTPDelivery, ResendDelivery

    # Both Resend and SMTP configured; EMAIL_PROVIDER=smtp must pick SMTP.
    cfg = Config(
        email_provider="smtp",
        resend_api_key="re_123",
        smtp_server="smtp.gmail.com",
        smtp_from="me@gmail.com",
    )
    assert isinstance(build_delivery_client(cfg), SMTPDelivery)

    # "auto" with a Resend key prefers Resend (the old default behavior).
    cfg_auto = Config(email_provider="auto", resend_api_key="re_123", smtp_server="smtp.gmail.com")
    assert isinstance(build_delivery_client(cfg_auto), ResendDelivery)

    # EMAIL_PROVIDER=smtp with no SMTP server is a clear configuration error.
    with pytest.raises(ValueError):
        build_delivery_client(Config(email_provider="smtp", resend_api_key="re_123"))
