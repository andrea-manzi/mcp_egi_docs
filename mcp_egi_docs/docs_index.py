"""Fetch, parse and search-index the documentation at docs.egi.eu."""

from __future__ import annotations

import asyncio
import logging
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify

logger = logging.getLogger(__name__)

BASE_URL = "https://docs.egi.eu"
SITEMAP_URL = f"{BASE_URL}/sitemap.xml"

# Sitemap entries that are not real documentation pages.
EXCLUDED_PATHS = {"/search/", "/_footer/"}

USER_AGENT = "mcp-egi-docs/0.1 (docs.egi.eu reader)"
FETCH_CONCURRENCY = 8
REQUEST_TIMEOUT = 20.0

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-_]*")
_BLANK_LINES_RE = re.compile(r"\n{3,}")

# BM25 parameters.
_K1 = 1.5
_B = 0.75
_TITLE_BOOST = 3.0
_PHRASE_BONUS = 2.0


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def normalize_path(path: str) -> str:
    """Normalize user input into an absolute site path like '/users/getting-started/'."""
    p = path.strip()
    if "://" in p:
        p = urlparse(p).path
    if not p.startswith("/"):
        p = "/" + p
    if not p.endswith("/"):
        p += "/"
    return p


@dataclass
class DocPage:
    path: str
    url: str
    title: str
    text: str
    markdown: str


async def fetch_page(client: httpx.AsyncClient, path: str) -> DocPage:
    """Fetch one documentation page and return it parsed."""
    url = urljoin(BASE_URL, path)
    resp = await client.get(url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # The Docsy theme wraps the actual documentation in <div class="td-content">.
    content = soup.find("div", class_="td-content")
    if content is None:
        content = soup.find("main") or soup.body or soup

    for tag in content.find_all(["script", "style", "nav", "img"]):
        tag.decompose()

    h1 = content.find("h1")
    if h1:
        title = h1.get_text(strip=True)
        # The title is reported separately, drop it from the page body.
        h1.decompose()
    elif soup.title:
        title = soup.title.get_text(strip=True)
    else:
        title = path

    # Make links absolute so the markdown is usable outside the site.
    for a in content.find_all("a", href=True):
        a["href"] = urljoin(url, a["href"])

    text = content.get_text(" ", strip=True)
    markdown = markdownify(str(content), heading_style="ATX").strip()
    markdown = _BLANK_LINES_RE.sub("\n\n", markdown)

    return DocPage(path=path, url=url, title=title, text=text, markdown=markdown)


class DocsIndex:
    """In-memory full-text index of the whole docs.egi.eu site."""

    def __init__(self) -> None:
        self.pages: dict[str, DocPage] = {}
        self.total_urls = 0
        self.building = False
        self.built_at: float | None = None
        self._build_task: asyncio.Task | None = None
        self._body_tf: dict[str, Counter] = {}
        self._title_tf: dict[str, Counter] = {}
        self._df: Counter = Counter()
        self._len_sum = 0
        self._idf: dict[str, float] = {}
        self._avg_len = 1.0

    @property
    def status_line(self) -> str:
        if self.building:
            total = self.total_urls or "?"
            return f"index still building ({len(self.pages)}/{total} pages loaded)"
        return f"{len(self.pages)} pages indexed"

    # ---- index building -------------------------------------------------

    def start(self) -> None:
        """Start (or restart after a failure) the background index build."""
        if self._build_task is None or self._build_task.done():
            self._build_task = asyncio.create_task(self.build())

    async def build(self) -> None:
        self.building = True
        try:
            async with httpx.AsyncClient(
                headers={"User-Agent": USER_AGENT},
                timeout=REQUEST_TIMEOUT,
                follow_redirects=True,
            ) as client:
                paths = await self._fetch_sitemap(client)
                self.total_urls = len(paths)
                sem = asyncio.Semaphore(FETCH_CONCURRENCY)
                await asyncio.gather(
                    *(self._fetch_and_store(client, sem, p) for p in paths)
                )
            self._compute_idf()
            self.built_at = time.time()
            logger.info("Index built: %d pages", len(self.pages))
        finally:
            self.building = False

    async def _fetch_sitemap(self, client: httpx.AsyncClient) -> list[str]:
        resp = await client.get(SITEMAP_URL)
        resp.raise_for_status()
        locs = re.findall(r"<loc>(.*?)</loc>", resp.text)
        paths = []
        for loc in locs:
            # The sitemap uses relative locations like /users/getting-started/.
            path = normalize_path(urlparse(urljoin(BASE_URL, loc.strip())).path)
            if path not in EXCLUDED_PATHS:
                paths.append(path)
        return list(dict.fromkeys(paths))

    async def _fetch_and_store(
        self, client: httpx.AsyncClient, sem: asyncio.Semaphore, path: str
    ) -> None:
        async with sem:
            try:
                page = await fetch_page(client, path)
            except Exception as exc:
                logger.warning("Skipping %s: %s", path, exc)
                return
            self.pages[path] = page
            body_tokens = tokenize(page.text)
            title_tokens = tokenize(page.title)
            self._body_tf[path] = Counter(body_tokens)
            self._title_tf[path] = Counter(title_tokens)
            self._len_sum += len(body_tokens)
            for term in set(body_tokens) | set(title_tokens):
                self._df[term] += 1
            if len(self.pages) % 25 == 0:
                logger.info("Indexed %d pages...", len(self.pages))

    def _compute_idf(self) -> None:
        n = max(len(self.pages), 1)
        self._idf = {
            t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in self._df.items()
        }
        self._avg_len = self._len_sum / n

    # ---- queries ----------------------------------------------------------

    def get_page(self, path: str) -> DocPage | None:
        return self.pages.get(normalize_path(path))

    def search(self, query: str, limit: int = 10) -> list[tuple[float, DocPage, str]]:
        """BM25-ranked full-text search. Returns (score, page, snippet) tuples."""
        qterms = tokenize(query)
        if not qterms:
            return []

        phrase = " ".join(qterms)
        scored: list[tuple[float, DocPage]] = []
        for path, page in self.pages.items():
            body_tf = self._body_tf[path]
            title_tf = self._title_tf[path]
            dl = sum(body_tf.values()) or 1
            score = 0.0
            for term in qterms:
                idf = self._idf.get(term, 1.0)
                f = body_tf.get(term, 0)
                if f:
                    denom = f + _K1 * (1 - _B + _B * dl / self._avg_len)
                    score += idf * (f * (_K1 + 1)) / denom
                if title_tf.get(term):
                    score += _TITLE_BOOST * idf * title_tf[term]
            if phrase in page.text.lower():
                score += _PHRASE_BONUS
            if score > 0:
                scored.append((score, page))

        scored.sort(key=lambda sp: (-sp[0], sp[1].path))
        return [
            (score, page, _make_snippet(page, qterms)) for score, page in scored[:limit]
        ]

    def sections(self) -> dict[str, dict[str, int]]:
        """Map of top-level section -> subsection -> number of pages."""
        tree: dict[str, dict[str, int]] = {}
        for path in self.pages:
            segs = [s for s in path.split("/") if s]
            if not segs:
                continue
            top = segs[0]
            sub = segs[1] if len(segs) > 1 else "(top level)"
            tree.setdefault(top, {}).setdefault(sub, 0)
            tree[top][sub] += 1
        return tree


def _make_snippet(page: DocPage, qterms: list[str], width: int = 240) -> str:
    text = re.sub(r"\s+", " ", page.text)
    if len(text) <= width:
        return text
    low = text.lower()
    pos = -1
    for term in qterms:
        i = low.find(term)
        if i != -1 and (pos == -1 or i < pos):
            pos = i
    if pos == -1:
        return text[:width].strip() + "…"
    start = max(0, pos - width // 3)
    end = min(len(text), start + width)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return prefix + text[start:end].strip() + suffix
