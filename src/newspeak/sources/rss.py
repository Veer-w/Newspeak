import xml.etree.ElementTree as ET
import httpx
from typing import Sequence
import logging
from newspeak.types import Article

logger = logging.getLogger(__name__)

def parse_rss_item(item: ET.Element, source_name: str) -> Article | None:
    """Parses a single RSS 2.0 <item> element into an Article."""
    try:
        title_el = item.find("title")
        link_el = item.find("link")
        desc_el = item.find("description")
        pub_el = item.find("pubDate")

        title = title_el.text.strip() if title_el is not None and title_el.text else ""
        link = link_el.text.strip() if link_el is not None and link_el.text else ""
        desc = desc_el.text.strip() if desc_el is not None and desc_el.text else ""
        pub_date = pub_el.text.strip() if pub_el is not None and pub_el.text else None

        # Clean HTML tags from description if any
        if desc.startswith("<![CDATA[") and desc.endswith("]]>"):
            desc = desc[9:-3]

        if not title or not link:
            return None

        return Article(
            title=title,
            url=link,
            description=desc,
            source=source_name,
            published_at=pub_date
        )
    except Exception as e:
        logger.warning(f"Error parsing RSS item: {e}")
        return None


def parse_atom_entry(entry: ET.Element, namespaces: dict[str, str], source_name: str) -> Article | None:
    """Parses a single Atom <entry> element into an Article."""
    try:
        title_el = entry.find("atom:title", namespaces)
        # Atom link can have multiple attributes, we look for rel="alternate" or first link
        link_el = entry.find("atom:link[@rel='alternate']", namespaces)
        if link_el is None:
            link_el = entry.find("atom:link", namespaces)
        
        desc_el = entry.find("atom:summary", namespaces)
        if desc_el is None:
            desc_el = entry.find("atom:content", namespaces)
            
        pub_el = entry.find("atom:published", namespaces)
        if pub_el is None:
            pub_el = entry.find("atom:updated", namespaces)

        title = title_el.text.strip() if title_el is not None and title_el.text else ""
        link = link_el.attrib.get("href", "").strip() if link_el is not None else ""
        desc = desc_el.text.strip() if desc_el is not None and desc_el.text else ""
        pub_date = pub_el.text.strip() if pub_el is not None and pub_el.text else None

        if not title or not link:
            return None

        return Article(
            title=title,
            url=link,
            description=desc,
            source=source_name,
            published_at=pub_date
        )
    except Exception as e:
        logger.warning(f"Error parsing Atom entry: {e}")
        return None


def parse_feed_content(xml_content: bytes, url: str) -> Sequence[Article]:
    """Pure parser that maps XML bytes into a sequence of Articles."""
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        logger.error(f"XML parse error for {url}: {e}")
        return []

    # Infer a readable source name from the URL or feed title
    source_name = url.split("//")[-1].split("/")[0]
    title_el = root.find(".//title")
    if title_el is None:
        title_el = root.find(".//{http://www.w3.org/2005/Atom}title")
    if title_el is not None and title_el.text:
        source_name = title_el.text.strip()

    # Detect if RSS or Atom
    # RSS typically has <channel>
    channel = root.find("channel")
    if channel is not None:
        items = channel.findall("item")
        parsed = [parse_rss_item(item, source_name) for item in items]
        return [p for p in parsed if p is not None]

    # Atom namespace detection
    # ElementTree tags will contain namespace e.g. {http://www.w3.org/2005/Atom}feed
    if "Atom" in root.tag or root.tag.endswith("feed"):
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)
        # If default namespace isn't registering, try search by tag suffix
        if not entries:
            entries = root.findall(".//{http://www.w3.org/2005/Atom}entry")
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            
        parsed = [parse_atom_entry(entry, ns, source_name) for entry in entries]
        return [p for p in parsed if p is not None]

    # Fallback to finding 'item' or 'entry' anywhere
    fallback_items = root.findall(".//item")
    if fallback_items:
        parsed = [parse_rss_item(item, source_name) for item in fallback_items]
        return [p for p in parsed if p is not None]

    logger.warning(f"Unknown feed format for {url}")
    return []


async def fetch_feed(client: httpx.AsyncClient, url: str) -> Sequence[Article]:
    """Fetches a single feed and parses it. Returns empty list on failure."""
    try:
        logger.info(f"Fetching feed: {url}")
        # Standard user-agent to avoid getting blocked
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Newspeak/1.0"}
        response = await client.get(url, headers=headers, timeout=10.0, follow_redirects=True)
        if response.status_code == 200:
            return parse_feed_content(response.content, url)
        else:
            logger.error(f"Failed to fetch {url}: Status {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"Exception fetching {url}: {e}")
        return []


async def aggregate_rss_feeds(urls: Sequence[str]) -> Sequence[Article]:
    """Orchestrates concurrent fetching of multiple RSS/Atom feeds."""
    async with httpx.AsyncClient() as client:
        # Fetch feeds concurrently
        import asyncio
        tasks = [fetch_feed(client, url) for url in urls]
        results = await asyncio.gather(*tasks)
        # Flatten the list of lists
        return [article for sublist in results for article in sublist]
