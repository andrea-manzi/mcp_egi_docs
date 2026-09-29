"""MCP server exposing the EGI documentation (docs.egi.eu)."""

from __future__ import annotations

import argparse
import logging
from contextlib import asynccontextmanager

import httpx
from mcp.server.fastmcp import FastMCP

from . import docs_index
from .docs_index import (
    BASE_URL,
    REQUEST_TIMEOUT,
    USER_AGENT,
    DocsIndex,
    DocPage,
    normalize_path,
)

logging.basicConfig(level=logging.INFO)
for noisy in ("httpx", "httpcore", "mcp"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

index = DocsIndex()


@asynccontextmanager
async def lifespan(server: FastMCP):
    # Warm up the search index in the background as soon as a client connects.
    index.start()
    yield {}


mcp = FastMCP(
    "egi-docs",
    instructions=(
        "Access to the EGI Federation documentation hosted at docs.egi.eu. "
        "Use search_docs to find pages by keywords, then get_doc_page to read "
        "the full content of a page, or list_sections to browse the site structure."
    ),
    lifespan=lifespan,
)


@mcp.tool()
async def search_docs(query: str, max_results: int = 8) -> str:
    """Full-text search over all docs.egi.eu documentation pages.

    query: keywords or a phrase, e.g. 'obtain check-in access token'.
    Returns matching pages ranked by relevance, with title, URL and a snippet.
    """
    index.start()  # no-op if the build is already running or done
    if not index.pages:
        return (
            "The documentation index is not ready yet "
            f"({index.status_line}). Please try again in a few seconds."
        )

    results = index.search(query, limit=max(1, min(max_results, 25)))
    if not results:
        return f"No documentation pages match '{query}' ({index.status_line})."

    lines = [
        f"{len(results)} result(s) for \"{query}\" ({index.status_line}):\n"
    ]
    for i, (_score, page, snippet) in enumerate(results, 1):
        lines.append(f"{i}. {page.title}\n   {page.url}\n   {snippet}\n")
    return "\n".join(lines)


@mcp.tool()
async def get_doc_page(path: str, max_chars: int = 20000) -> str:
    """Read the full content of a docs.egi.eu documentation page as markdown.

    path: the page path, e.g. '/users/getting-started/' (a full URL works too).
    max_chars: the returned content is truncated at this many characters.
    """
    max_chars = max(1000, min(max_chars, 100000))
    page: DocPage | None = index.get_page(path)
    if page is None:
        norm = normalize_path(path)
        try:
            async with httpx.AsyncClient(
                headers={"User-Agent": USER_AGENT},
                timeout=REQUEST_TIMEOUT,
                follow_redirects=True,
            ) as client:
                page = await docs_index.fetch_page(client, norm)
        except httpx.HTTPStatusError as exc:
            return (
                f"Could not fetch {norm}: HTTP {exc.response.status_code}.\n"
                "Use search_docs to find valid page paths."
            )
        except httpx.HTTPError as exc:
            return f"Could not fetch {norm}: {exc}"

    body = page.markdown
    suffix = ""
    if len(body) > max_chars:
        body = body[:max_chars].rstrip()
        suffix = (
            "\n\n---\n*[Truncated: the full page is "
            f"{len(page.markdown)} characters; call again with a larger "
            "max_chars to see more.]*"
        )
    return f"# {page.title}\n\nSource: {page.url}\n\n{body}{suffix}"


@mcp.tool()
async def list_sections() -> str:
    """Show the documentation structure of docs.egi.eu.

    Returns the top-level sections (users, providers, internal, ...) with
    their subsections and the number of pages in each.
    """
    index.start()  # no-op if the build is already running or done
    if not index.pages:
        return (
            "The documentation index is not ready yet "
            f"({index.status_line}). Please try again in a few seconds."
        )

    tree = index.sections()
    lines = [f"Documentation structure of {BASE_URL} ({index.status_line}):\n"]
    for top in sorted(tree):
        total = sum(tree[top].values())
        lines.append(f"/{top}/ — {total} page(s)")
        for sub in sorted(tree[top]):
            lines.append(f"  - {sub} ({tree[top][sub]})")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MCP server that serves the EGI documentation (docs.egi.eu)"
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http", "sse"],
        default="stdio",
        help="MCP transport to use (default: stdio)",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="HTTP bind host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="HTTP bind port (default: 8000)"
    )
    args = parser.parse_args()

    if args.transport != "stdio":
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        logger.info(
            "Serving MCP over %s at http://%s:%d%s",
            args.transport,
            args.host,
            args.port,
            mcp.settings.streamable_http_path,
        )
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
