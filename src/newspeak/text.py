"""Shared pure text helpers used by both in-run dedup (pipeline) and cross-run dedup
(history). Kept in their own module so neither importer has to depend on the other."""


def clean_text(text: str) -> str:
    """Helper to clean titles/descriptions for similarity matching."""
    return "".join(c for c in text.lower() if c.isalnum() or c.isspace())


def get_jaccard_similarity(text1: str, text2: str) -> float:
    """Pure Jaccard Similarity calculation between two text strings."""
    words1 = set(clean_text(text1).split())
    words2 = set(clean_text(text2).split())
    if not words1 or not words2:
        return 0.0
    return len(words1 & words2) / len(words1 | words2)
