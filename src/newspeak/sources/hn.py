import asyncio
import re
import httpx
import logging
from typing import Sequence
from newspeak.types import Article

logger = logging.getLogger(__name__)

HN_BASE_URL = "https://hacker-news.firebaseio.com/v0"

async def fetch_story(client: httpx.AsyncClient, story_id: int) -> Article | None:
    """Fetches details of a single Hacker News item and maps to Article if it's a story with a URL."""
    try:
        url = f"{HN_BASE_URL}/item/{story_id}.json"
        response = await client.get(url, timeout=5.0)
        if response.status_code != 200:
            return None

        data = response.json()
        # Guard against unexpected response shape (dict, None, etc.)
        if not isinstance(data, dict) or data.get("type") != "story":
            return None

        title = data.get("title", "").strip()
        story_url = data.get("url", "").strip()
        text = data.get("text", "").strip()  # Ask HN post body text

        if not title or not story_url:
            # Include Ask-HN posts with no external link but a title, using HN URL
            if title and data.get("id"):
                story_url = f"https://news.ycombinator.com/item?id={data['id']}"
            else:
                return None

        description = text or f"Hacker News story with {data.get('score', 0)} points and {data.get('descendants', 0)} comments."

        # Use None for missing timestamps instead of "0"
        ts = data.get("time")
        published_at = str(ts) if ts else None

        return Article(
            title=title,
            url=story_url,
            description=description,
            source="Hacker News",
            published_at=published_at
        )
    except Exception as e:
        logger.debug(f"Failed to fetch HN story {story_id}: {e}")
        return None


def contains_keywords(text: str, keywords: Sequence[str]) -> bool:
    """Pure check: does the text contain any keyword as a whole word (case-insensitive)?

    Uses word boundaries so short keywords like "ai"/"ml"/"rag" don't match substrings of
    unrelated words ("email", "html", "storage"). The compiled pattern is cached by the re
    module across calls since the keyword list is stable within a run.
    """
    if not keywords:
        return False
    pattern = "|".join(re.escape(kw) for kw in keywords)
    return re.search(rf"\b(?:{pattern})\b", text, re.IGNORECASE) is not None


async def fetch_hn_stories(limit: int, keywords: Sequence[str]) -> Sequence[Article]:
    """Fetches top Hacker News stories, filters by keyword, and maps to Articles."""
    logger.info("Fetching Hacker News top stories...")
    async with httpx.AsyncClient() as client:
        try:
            top_stories_url = f"{HN_BASE_URL}/topstories.json"
            response = await client.get(top_stories_url, timeout=10.0)
            if response.status_code != 200:
                logger.error(f"Failed to fetch HN top stories list: HTTP {response.status_code}")
                return []

            data = response.json()
            # Guard against unexpected response type (e.g. Firebase error dict or null)
            if not isinstance(data, list):
                logger.error(f"HN API returned unexpected data type: {type(data).__name__}. Expected list.")
                return []

            story_ids = data[:limit]

            # Fetch all story details concurrently, tolerate individual failures
            tasks = [fetch_story(client, story_id) for story_id in story_ids]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            valid_stories = [
                r for r in results
                if r is not None and not isinstance(r, BaseException)
            ]

            # Match against title AND body text — an AI story is often only signalled in
            # the post body (Ask-HN) or subtitle, not the headline, so title-only filtering
            # dropped most of HN's AI content.
            relevant_stories = [
                story for story in valid_stories
                if contains_keywords(f"{story.title} {story.description}", keywords)
            ]

            logger.info(f"Fetched {len(valid_stories)} HN stories, {len(relevant_stories)} are AI-related.")
            return relevant_stories

        except Exception as e:
            logger.error(f"Error fetching Hacker News stories: {e}")
            return []
