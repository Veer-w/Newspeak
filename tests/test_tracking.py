from newspeak.types import Article, NewsItem
from newspeak.tracking import (
    sign,
    build_tracking_url,
    tracking_urls_for,
    update_reputation,
    reputation_weights,
    reputation_bonus,
    count_by_source,
    _empty_reputation,
    MIN_SENDS,
    WEIGHT_MIN,
    WEIGHT_MAX,
)
from newspeak.llm.heuristic import score_article


def _news(url: str, score: float = 5.0) -> NewsItem:
    return NewsItem(title="t", url=url, summary="s", score=score, reason="r", source="src")


# ---- signing (must match worker/tracker.js sign()) --------------------------------

def test_sign_is_stable_known_vector() -> None:
    # Frozen expected value; worker/tracker.js MUST produce the same 16 hex chars.
    assert sign("https://ex.com/a|ex.com", "secret123") == "4e44f0f1c39c990e"


def test_build_tracking_url_shape_and_signature() -> None:
    url = build_tracking_url("https://ex.com/a?b=1", "ex.com", "https://w.dev", "secret")
    assert url.startswith("https://w.dev/r?")
    assert "u=https%3A%2F%2Fex.com%2Fa%3Fb%3D1" in url
    assert "s=ex.com" in url
    assert f"sig={sign('https://ex.com/a?b=1|ex.com', 'secret')}" in url


# ---- link wrapping toggling -------------------------------------------------------

class _Cfg:
    def __init__(self, base="", secret=""):
        self.tracking_base_url = base
        self.tracking_secret = secret

    @property
    def tracking_enabled(self) -> bool:
        return bool(self.tracking_base_url and self.tracking_secret)


def test_tracking_urls_none_when_disabled() -> None:
    items = [_news("https://a.com/x")]
    assert tracking_urls_for(items, _Cfg()) is None


def test_tracking_urls_when_enabled() -> None:
    items = [_news("https://techcrunch.com/x"), _news("https://arxiv.org/abs/1")]
    urls = tracking_urls_for(items, _Cfg("https://w.dev", "sec"))
    assert urls is not None and len(urls) == 2
    assert urls[0].startswith("https://w.dev/r?") and "s=techcrunch.com" in urls[0]


# ---- reputation math --------------------------------------------------------------

def test_update_reputation_delta_decay_and_weights() -> None:
    prev = _empty_reputation()
    sent = {"techcrunch.com": 4, "arxiv.org": 4}
    rep = update_reputation(prev, {"techcrunch.com": 30, "arxiv.org": 5}, sent, decay=0.9)
    src = rep["sources"]
    # Clicks folded in as delta from cursor 0; cursor advances to the cumulative value.
    assert src["techcrunch.com"]["clicks"] == 30.0
    assert src["techcrunch.com"]["cursor"] == 30
    # High-CTR source gets a positive (clamped) weight; low-CTR gets negative.
    assert src["techcrunch.com"]["weight"] > 0
    assert src["arxiv.org"]["weight"] < 0
    assert WEIGHT_MIN <= src["arxiv.org"]["weight"] <= WEIGHT_MAX


def test_update_reputation_cursor_prevents_double_counting() -> None:
    prev = _empty_reputation()
    r1 = update_reputation(prev, {"a.com": 10}, {"a.com": 3}, decay=1.0)
    # Same cumulative total on the next run ⇒ zero new clicks (delta from cursor).
    r2 = update_reputation(r1, {"a.com": 10}, {"a.com": 3}, decay=1.0)
    assert r2["sources"]["a.com"]["clicks"] == 10.0  # unchanged (no decay, no new delta)
    assert r2["sources"]["a.com"]["sends"] == 6.0     # sends accumulated across both runs


def test_weight_zero_until_min_sends() -> None:
    prev = _empty_reputation()
    rep = update_reputation(prev, {"a.com": 100}, {"a.com": MIN_SENDS - 1}, decay=1.0)
    # Not enough exposure yet ⇒ no weight regardless of clicks.
    assert rep["sources"]["a.com"]["weight"] == 0.0


def test_reputation_bonus_and_score_article_use_weights() -> None:
    weights = {"techcrunch.com": 1.2, "arxiv.org": -0.8}
    assert reputation_bonus("https://techcrunch.com/x", weights) == 1.2
    assert reputation_bonus("https://unknown.com/x", weights) == 0.0
    assert reputation_bonus("https://x.com/x", None) == 0.0

    art_tc = Article(title="AI model", url="https://techcrunch.com/x", description="d" * 100, source="TechCrunch")
    base = score_article(art_tc, ["ai"])
    boosted = score_article(art_tc, ["ai"], weights)
    assert boosted == min(10.0, base + 1.2)


def test_count_by_source() -> None:
    items = [_news("https://arxiv.org/a"), _news("https://arxiv.org/b"), _news("https://techcrunch.com/c")]
    assert count_by_source(items) == {"arxiv.org": 2, "techcrunch.com": 1}
