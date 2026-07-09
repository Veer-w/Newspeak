import pytest
from newspeak.types import Article
from newspeak.pipeline import clean_text, get_jaccard_similarity, deduplicate_articles
from newspeak.sources.rss import parse_feed_content

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
