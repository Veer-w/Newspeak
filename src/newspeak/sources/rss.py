import asyncio
import xml.etree.ElementTree as ET
import html
import re
from urllib.parse import urlparse
import httpx
from typing import Sequence
import logging
from newspeak.types import Article

logger = logging.getLogger(__name__)

# Compiled regex for stripping HTML tags from feed descriptions
_HTML_TAG_RE = re.compile(r'<[^>]+>')

def strip_html(text: str) -> str:
    """Pure function: removes HTML tags and decodes HTML entities from a text string.

    ElementTree decodes the XML entity layer once, but feeds that double-escape HTML
    (common with WordPress/TechCrunch) leave a residual layer like "&amp;"/"&#39;" —
    html.unescape resolves those. Idempotent on already-clean text.
    """
    return html.unescape(_HTML_TAG_RE.sub('', text)).strip()

def _parse_source_name(url: str, root: ET.Element) -> str:
    """Infers a readable feed source name from XML title elements or falls back to netloc."""
    # Try RSS <title> first, then Atom {namespace}title
    title_el = root.find(".//title")
    if title_el is None:
        title_el = root.find(".//{http://www.w3.org/2005/Atom}title")
    if title_el is not None and title_el.text and title_el.text.strip():
        return title_el.text.strip()
    # Fallback to URL netloc
    parsed = urlparse(url)
    return parsed.netloc or url


def parse_rss_item(item: ET.Element, source_name: str) -> Article | None:
    """Parses a single RSS 2.0 <item> element into an Article."""
    try:
        title_el = item.find("title")
        link_el = item.find("link")
        desc_el = item.find("description")
        pub_el = item.find("pubDate")

        title = html.unescape(title_el.text).strip() if title_el is not None and title_el.text else ""
        link = link_el.text.strip() if link_el is not None and link_el.text else ""
        raw_desc = desc_el.text.strip() if desc_el is not None and desc_el.text else ""
        pub_date = pub_el.text.strip() if pub_el is not None and pub_el.text else None

        # Strip HTML tags from descriptions — feeds often include raw HTML
        desc = strip_html(raw_desc)

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
        # Atom link: prefer rel="alternate", fallback to first link
        link_el = entry.find("atom:link[@rel='alternate']", namespaces)
        if link_el is None:
            link_el = entry.find("atom:link", namespaces)

        desc_el = entry.find("atom:summary", namespaces)
        if desc_el is None:
            desc_el = entry.find("atom:content", namespaces)

        pub_el = entry.find("atom:published", namespaces)
        if pub_el is None:
            pub_el = entry.find("atom:updated", namespaces)

        title = html.unescape(title_el.text).strip() if title_el is not None and title_el.text else ""
        link = link_el.attrib.get("href", "").strip() if link_el is not None else ""
        raw_desc = desc_el.text.strip() if desc_el is not None and desc_el.text else ""
        pub_date = pub_el.text.strip() if pub_el is not None and pub_el.text else None

        # Strip HTML tags from descriptions
        desc = strip_html(raw_desc)

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

    source_name = _parse_source_name(url, root)

    ATOM_NS = "http://www.w3.org/2005/Atom"

    # Detect RSS (has <channel>)
    channel = root.find("channel")
    if channel is not None:
        items = channel.findall("item")
        parsed = [parse_rss_item(item, source_name) for item in items]
        return [p for p in parsed if p is not None]

    # Detect Atom: namespaced root tag is {http://...}feed or bare "feed"
    if root.tag == f"{{{ATOM_NS}}}feed" or root.tag == "feed":
        ns = {"atom": ATOM_NS}
        entries = root.findall("atom:entry", ns)
        if not entries:
            # Fallback: search by fully-qualified tag name
            entries = root.findall(f".//{{{ATOM_NS}}}entry")
        parsed = [parse_atom_entry(entry, ns, source_name) for entry in entries]
        return [p for p in parsed if p is not None]

    # Fallback: search anywhere for <item> elements
    fallback_items = root.findall(".//item")
    if fallback_items:
        parsed = [parse_rss_item(item, source_name) for item in fallback_items]
        return [p for p in parsed if p is not None]

    logger.warning(f"Unknown feed format for {url}")
    return []


async def fetch_feed(client: httpx.AsyncClient, url: str) -> Sequence[Article]:
    """Fetches a single feed and parses it. Returns empty list on any failure."""
    try:
        logger.info(f"Fetching feed: {url}")
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Newspeak/1.0"}
        # Explicit per-field timeout to guard against slow/hanging servers
        timeout = httpx.Timeout(connect=5.0, read=15.0, write=5.0, pool=5.0)
        response = await client.get(url, headers=headers, timeout=timeout, follow_redirects=True)
        if response.status_code == 200:
            return parse_feed_content(response.content, url)
        else:
            logger.error(f"Failed to fetch {url}: HTTP {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"Exception fetching {url}: {e}")
        return []


async def aggregate_rss_feeds(urls: Sequence[str]) -> Sequence[Article]:
    """Orchestrates concurrent fetching of multiple RSS/Atom feeds."""
    async with httpx.AsyncClient() as client:
        tasks = [fetch_feed(client, url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        articles: list[Article] = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                logger.error(f"Feed fetch raised exception for {urls[i]}: {result}")
            else:
                articles.extend(result)
        return articles
