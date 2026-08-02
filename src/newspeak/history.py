"""Cross-run article history: remembers which stories were already sent so the same
news doesn't reappear in later newsletters.

The pipeline runs on an ephemeral GitHub Actions runner, so state can't live in memory
between runs. Instead we persist a small JSON file (committed back to the repo by the
workflow) recording every delivered item's URL + title + timestamp. Before ranking, we
drop any candidate already present in that history — by exact URL, or by a title that is
Jaccard-similar to a previously sent one (which catches the same event reported by a
different source under a different URL).
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Sequence
from newspeak.types import Article, NewsItem
from newspeak.text import get_jaccard_similarity

logger = logging.getLogger(__name__)

# Same threshold the in-run title dedup uses (see pipeline.deduplicate_articles).
DEFAULT_SIMILARITY_THRESHOLD = 0.65


def _parse_ts(value: str) -> datetime | None:
    """Pure: parse an ISO-8601 timestamp, assuming UTC when no offset is present."""
    try:
        dt = datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def load_history(path: str | Path) -> list[dict]:
    """Read the sent-article history file. Returns [] if it's missing or unreadable."""
    p = Path(path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Could not read history file {p}: {e}; treating history as empty.")
        return []
    if not isinstance(data, list):
        logger.warning(f"History file {p} is not a JSON list; ignoring it.")
        return []
    return [r for r in data if isinstance(r, dict)]


def prune_history(records: Sequence[dict], retention_days: int, now: datetime | None = None) -> list[dict]:
    """Pure: drop records older than `retention_days` so the file stays bounded.

    Records whose timestamp is missing or unparseable are kept (fail-safe: better to keep
    suppressing a possible repeat than to silently forget it). retention_days <= 0 disables
    pruning entirely.
    """
    if retention_days <= 0:
        return list(records)
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=retention_days)
    kept: list[dict] = []
    for r in records:
        ts = _parse_ts(r.get("sent_at", ""))
        if ts is None or ts >= cutoff:
            kept.append(r)
    return kept


def filter_new_articles(
    articles: Sequence[Article],
    history: Sequence[dict],
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> Sequence[Article]:
    """Pure: drop articles that were already sent in a previous run.

    An article is suppressed when its URL exactly matches a previously sent URL, or when
    its title is Jaccard-similar (>= threshold) to a previously sent title.
    """
    if not history:
        return list(articles)
    seen_urls = {r.get("url") for r in history if r.get("url")}
    seen_titles = [r["title"] for r in history if r.get("title")]

    fresh: list[Article] = []
    dropped = 0
    for art in articles:
        if art.url in seen_urls:
            dropped += 1
            continue
        if any(get_jaccard_similarity(art.title, t) >= similarity_threshold for t in seen_titles):
            dropped += 1
            continue
        fresh.append(art)

    if dropped:
        logger.info(f"Cross-run dedup: suppressed {dropped} article(s) already sent in a previous run.")
    return fresh


def build_records(items: Sequence[NewsItem], now: datetime | None = None) -> list[dict]:
    """Pure: turn delivered NewsItems into history records stamped with the send time."""
    ts = (now or datetime.now(timezone.utc)).isoformat()
    return [{"url": it.url, "title": it.title, "sent_at": ts} for it in items]


def record_sent(
    path: str | Path,
    items: Sequence[NewsItem],
    retention_days: int,
    now: datetime | None = None,
) -> None:
    """Append newly delivered items to the history file, pruning entries past retention."""
    if not items:
        return
    now = now or datetime.now(timezone.utc)
    updated = prune_history(list(load_history(path)) + build_records(items, now), retention_days, now)
    p = Path(path)
    try:
        p.write_text(json.dumps(updated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        logger.info(
            f"Recorded {len(items)} sent item(s) to {p} ({len(updated)} total within retention)."
        )
    except OSError as e:
        logger.error(f"Failed to write history file {p}: {e}")
