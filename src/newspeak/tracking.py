"""Click-through tracking + per-source reputation (option "c").

Reader engagement is captured by a tiny Cloudflare Worker (see `worker/tracker.js`):
each article link in the email is rewritten to point at the Worker, which records the
click in KV and 302-redirects to the real article. Because readers click *between*
cron runs, the Worker is the always-on receiver; each run we pull its cumulative counts
via `/stats`, fold them into a decayed per-source click-through rate, and turn that into
a small bounded ranking `weight`. State lives in `source_reputation.json` (committed
back by the workflow) — there is no database.

Everything here is inert unless `config.tracking_enabled` (a Worker origin + signing
secret are both set), so the default/tested behavior is unchanged.
"""

import hmac
import json
import logging
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Mapping, Sequence
from urllib.parse import quote, urlencode
import httpx
from newspeak.types import Article, NewsItem
from newspeak.llm.diversity import source_domain

logger = logging.getLogger(__name__)

# --- Ranking-weight tuning (see update_reputation) --------------------------------
MIN_SENDS = 3          # ignore a source until we've shown it enough to be meaningful
WEIGHT_GAIN = 2.0      # how strongly relative CTR maps to a score bonus
WEIGHT_MIN = -1.0      # clamp: a disliked source can lose at most this much
WEIGHT_MAX = 1.5       # clamp: a loved source can gain at most this much
_EPS = 1e-6


# ============================ Signed link wrapping ================================

def sign(payload: str, secret: str) -> str:
    """Truncated HMAC-SHA256 hex digest. Must match the Worker's `sign()` exactly."""
    return hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), sha256).hexdigest()[:16]


def build_tracking_url(dest: str, source_key: str, base: str, secret: str) -> str:
    """Wrap `dest` in a signed redirect through the tracking Worker.

    The signature covers `dest|source_key`, so the Worker only counts (and redirects)
    links we actually minted — it can't be used as an open redirect or to inflate stats.
    """
    sig = sign(f"{dest}|{source_key}", secret)
    query = urlencode({"u": dest, "s": source_key, "sig": sig}, quote_via=quote)
    return f"{base}/r?{query}"


def tracking_urls_for(items: Sequence[NewsItem], config) -> list[str] | None:
    """Per-item signed tracking URLs aligned to `items`, or None when tracking is off."""
    if not config.tracking_enabled:
        return None
    return [
        build_tracking_url(it.url, source_domain(it.url), config.tracking_base_url, config.tracking_secret)
        for it in items
    ]


# ============================ Reputation persistence =============================

def _empty_reputation() -> dict:
    return {"updated_at": None, "sources": {}}


def load_reputation(path: str | Path) -> dict:
    """Read source_reputation.json, returning an empty structure if missing/unreadable."""
    p = Path(path)
    if not p.exists():
        return _empty_reputation()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Could not read reputation file {p}: {e}; starting fresh.")
        return _empty_reputation()
    if not isinstance(data, dict) or not isinstance(data.get("sources"), dict):
        logger.warning(f"Reputation file {p} has an unexpected shape; starting fresh.")
        return _empty_reputation()
    return data


def save_reputation(path: str | Path, reputation: dict) -> None:
    """Persist the reputation structure as pretty JSON."""
    p = Path(path)
    try:
        p.write_text(json.dumps(reputation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        logger.info(f"Updated source reputation ({len(reputation.get('sources', {}))} sources) at {p}.")
    except OSError as e:
        logger.error(f"Failed to write reputation file {p}: {e}")


# ============================ Reputation math (pure) =============================

def update_reputation(
    prev: dict,
    click_totals: Mapping[str, int],
    sent_counts: Mapping[str, int],
    decay: float,
    now: datetime | None = None,
) -> dict:
    """Fold this run's clicks (cumulative from KV) and sends into decayed per-source stats.

    KV click counts are cumulative, so we keep a per-source `cursor` (the last cumulative
    value we consumed) and take the positive delta since then. Clicks and sends both decay
    each run so recent behavior dominates, then the click-through rate (Laplace-smoothed)
    becomes a bounded `weight` relative to the mean CTR across established sources.
    """
    now = now or datetime.now(timezone.utc)
    sources: dict[str, dict] = {d: dict(v) for d, v in prev.get("sources", {}).items()}

    domains = set(sources) | set(click_totals) | set(sent_counts)
    for d in domains:
        s = sources.get(d, {"clicks": 0.0, "sends": 0.0, "cursor": 0, "weight": 0.0})
        cursor = int(s.get("cursor", 0))
        cumulative = int(click_totals.get(d, cursor))  # unreported ⇒ no new clicks
        delta = max(0, cumulative - cursor)
        s["clicks"] = round(float(s.get("clicks", 0.0)) * decay + delta, 4)
        s["sends"] = round(float(s.get("sends", 0.0)) * decay + int(sent_counts.get(d, 0)), 4)
        s["cursor"] = max(cumulative, cursor)  # monotonic: never rewind the cursor
        sources[d] = s

    # Weights are relative to the mean CTR of sources with enough exposure to trust.
    established = {d: s for d, s in sources.items() if s["sends"] >= MIN_SENDS}
    ctr = {d: (s["clicks"] + 1.0) / (s["sends"] + 2.0) for d, s in established.items()}
    mean_ctr = sum(ctr.values()) / len(ctr) if ctr else 0.0

    for d, s in sources.items():
        if d in ctr and mean_ctr > 0:
            raw = WEIGHT_GAIN * (ctr[d] - mean_ctr) / (mean_ctr + _EPS)
            s["weight"] = round(max(WEIGHT_MIN, min(WEIGHT_MAX, raw)), 4)
        else:
            s["weight"] = 0.0

    return {"updated_at": now.isoformat(), "sources": sources}


def reputation_weights(reputation: dict) -> dict[str, float]:
    """Flatten to a {source_domain: weight} map for the ranking layer."""
    return {d: float(s.get("weight", 0.0)) for d, s in reputation.get("sources", {}).items()}


def reputation_bonus(url: str, weights: Mapping[str, float] | None) -> float:
    """Ranking bonus (± bounded) for `url`'s publisher; 0.0 when unknown or disabled."""
    if not weights:
        return 0.0
    return weights.get(source_domain(url), 0.0)


# ============================ Stats fetch (best-effort) =========================

async def fetch_click_counts(config) -> dict[str, int]:
    """Pull cumulative per-source click counts from the Worker's /stats endpoint.

    Best-effort: any failure (network, auth, bad JSON) returns {} so a tracking outage
    never blocks the newsletter — this run just skips the reputation update.
    """
    if not config.tracking_enabled or not config.tracking_stats_token:
        return {}
    url = f"{config.tracking_base_url}/stats"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params={"token": config.tracking_stats_token})
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning(f"Could not fetch click stats from {url}: {e}; skipping reputation update.")
        return {}

    if not isinstance(data, dict):
        logger.warning(f"Unexpected /stats payload from {url}; skipping reputation update.")
        return {}
    counts: dict[str, int] = {}
    for key, value in data.items():
        domain = key[4:] if key.startswith("clk:") else key  # tolerate raw KV keys
        try:
            counts[domain] = int(value)
        except (TypeError, ValueError):
            continue
    return counts


def count_by_source(items: Sequence[NewsItem | Article]) -> dict[str, int]:
    """Tally how many delivered items came from each publisher domain."""
    counts: dict[str, int] = {}
    for it in items:
        d = source_domain(it.url)
        counts[d] = counts.get(d, 0) + 1
    return counts
