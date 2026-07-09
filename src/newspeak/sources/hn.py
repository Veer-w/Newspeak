import asyncio
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
        if not data or data.get("type") != "story":
            return None
        
        title = data.get("title", "").strip()
        story_url = data.get("url", "").strip()
        text = data.get("text", "").strip()  # Ask HN text
        
        if not title or not story_url:
            # Skip ask stories without external links, or we can use HN link as the URL
            if title and data.get("id"):
                story_url = f"https://news.ycombinator.com/item?id={data['id']}"
            else:
                return None

        # Build description using text if available, otherwise blank
        description = text or f"Hacker News story with {data.get('score', 0)} points and {data.get('descendants', 0)} comments."
        
        return Article(
            title=title,
            url=story_url,
            description=description,
            source="Hacker News",
            published_at=str(data.get("time", ""))
        )
    except Exception as e:
        logger.debug(f"Failed to fetch HN story {story_id}: {e}")
        return None


def contains_keywords(title: str, keywords: Sequence[str]) -> bool:
    """Pure check to see if title contains any keyword (case insensitive)."""
    title_lower = title.lower()
    return any(kw in title_lower for kw in keywords)


async def fetch_hn_stories(limit: int, keywords: Sequence[str]) -> Sequence[Article]:
    """Fetches top Hacker News stories, filters by keyword, and maps to Articles."""
    logger.info("Fetching Hacker News top stories...")
    async with httpx.AsyncClient() as client:
        try:
            # Fetch top stories list
            top_stories_url = f"{HN_BASE_URL}/topstories.json"
            response = await client.get(top_stories_url, timeout=10.0)
            if response.status_code != 200:
                logger.error(f"Failed to fetch HN top stories list: {response.status_code}")
                return []
            
            story_ids = response.json()[:limit]
            
            # Fetch details for all stories concurrently
            tasks = [fetch_story(client, story_id) for story_id in story_ids]
            results = await asyncio.gather(*tasks)
            
            # Filter and parse
            valid_stories = [r for r in results if r is not None]
            
            # Filter by AI/ML keywords to keep it relevant
            relevant_stories = [
                story for story in valid_stories 
                if contains_keywords(story.title, keywords)
            ]
            
            logger.info(f"Fetched {len(valid_stories)} HN stories, {len(relevant_stories)} are AI-related.")
            return relevant_stories
            
        except Exception as e:
            logger.error(f"Error fetching Hacker News stories: {e}")
            return []
