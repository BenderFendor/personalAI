"""Tool implementations for web search, news, URL fetching, and more."""

import json
import logging
import math
from datetime import datetime
from typing import Dict, Any, Optional, Callable, List
import requests
import trafilatura
from trafilatura.settings import use_config
from ddgs import DDGS
from rich.console import Console
from rich.markup import escape
from rich.progress import Progress, SpinnerColumn, TextColumn
import wikipediaapi
import arxiv
import fitz  # PyMuPDF

try:
    from utils.llama_cpp_provider import LlamaCppProvider
except Exception:
    LlamaCppProvider = None

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Handles execution of all tool functions."""

    def __init__(self, config: Dict[str, Any], console: Console):
        """Initialize tool executor.

        Args:
            config: Configuration dictionary
            console: Rich console for output
        """
        self.config = config
        self.console = console
        self._tools = self._register_tools()
        self._llama_cpp_provider: Optional[LlamaCppProvider] = None

    def _get_llama_cpp_provider(self) -> LlamaCppProvider:
        """Get or create the LlamaCppProvider instance."""
        if self._llama_cpp_provider is None:
            if LlamaCppProvider is None:
                raise RuntimeError("llama.cpp provider is not available.")

            base_url = self.config.get("llama_cpp_base_url", "http://localhost:8080")
            model = self.config.get("model", "qwen3")

            self._llama_cpp_provider = LlamaCppProvider(
                model=model,
                base_url=base_url,
                temperature=0.7,
            )
        return self._llama_cpp_provider

    def _register_tools(self) -> Dict[str, Callable[..., str]]:
        """Register all available tool functions.

        Returns:
            Dictionary mapping tool names to functions
        """
        return {
            "web_search": self.web_search,
            "search_and_fetch": self.search_and_fetch,
            "news_search": self.news_search,
            "fetch_url_content": self.fetch_url_content,
            "calculate": self.calculate,
            "get_current_time": self.get_current_time,
            "search_wikipedia": self.search_wikipedia,
            "search_arxiv": self.search_arxiv,
            "deep_research": self.deep_research,
            "search_academic": self.search_academic,
            "search_pubmed": self.search_pubmed,
        }

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool function by name.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments

        Returns:
            Tool execution result as string
        """
        if tool_name not in self._tools:
            logger.warning("tool.execute missing tool=%s", tool_name)
            return f"Error: Tool '{tool_name}' not found"

        try:
            func = self._tools[tool_name]
            logger.debug("tool.execute start tool=%s args=%s", tool_name, arguments)

            # Normalize common argument aliases to make models more forgiving
            if tool_name == "news_search":
                arguments = self._normalize_news_search_args(arguments)
            elif tool_name == "web_search":
                arguments = self._normalize_web_search_args(arguments)

            result = func(**arguments)
            logger.debug("tool.execute done tool=%s", tool_name)
            return result
        except Exception as e:
            logger.exception("tool.execute error tool=%s", tool_name)
            return f"Error executing {tool_name}: {str(e)}"

    # -------------------------
    # Argument normalization
    # -------------------------
    def _normalize_news_search_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Map common aliases to the expected news_search signature.

        Expected: keywords: str, region: str = "us-en", safesearch: str = "moderate",
        timelimit: Optional[str] = None ("d","w","m"), max_results: int = 10
        """
        normalized = dict(args) if isinstance(args, dict) else {}

        # Alias 'query' -> 'keywords'
        if "keywords" not in normalized and "query" in normalized:
            normalized["keywords"] = normalized.pop("query")

        # Date filters to DuckDuckGo 'timelimit'
        # Accept 'date_filter' or 'date_range'
        date_val = None
        for k in ("timelimit", "date_filter", "date_range"):
            if k in normalized:
                date_val = normalized.get(k)
                # remove alias keys after reading
                if k != "timelimit":
                    normalized.pop(k, None)
                break
        if date_val is not None and "timelimit" not in normalized:
            mapping = {
                "d": "d",
                "day": "d",
                "past_day": "d",
                "last_day": "d",
                "1d": "d",
                "w": "w",
                "week": "w",
                "past_week": "w",
                "last_week": "w",
                "7d": "w",
                "m": "m",
                "month": "m",
                "past_month": "m",
                "last_month": "m",
                "30d": "m",
            }
            key = str(date_val).strip().lower()
            normalized["timelimit"] = mapping.get(key, None)
            if normalized["timelimit"] is None:
                normalized.pop("timelimit", None)  # leave unset if unknown

        # Safe search aliases
        for alias in ("safeSearch", "safe_search"):
            if alias in normalized and "safesearch" not in normalized:
                normalized["safesearch"] = normalized.pop(alias)

        # max results alias
        for alias in ("maxResults", "max_results"):
            if alias in normalized:
                try:
                    normalized["max_results"] = int(normalized[alias])
                except Exception:
                    pass
                if alias != "max_results":
                    normalized.pop(alias, None)
                break

        # region alias: accept 'us' and normalize to 'us-en'
        if "region" in normalized:
            val = str(normalized["region"]).lower()
            if val in {"us", "uk"}:
                normalized["region"] = f"{val}-en"

        return normalized

    def _normalize_web_search_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Map simple aliases for web_search: 'q'->'query', 'num_iterations'->'iterations'."""
        normalized = dict(args) if isinstance(args, dict) else {}
        if "query" not in normalized and "q" in normalized:
            normalized["query"] = normalized.pop("q")
        if "iterations" not in normalized and "num_iterations" in normalized:
            normalized["iterations"] = normalized.pop("num_iterations")
        # Clamp iterations if provided as string
        if "iterations" in normalized:
            try:
                normalized["iterations"] = int(normalized["iterations"])
            except Exception:
                normalized["iterations"] = 1
        return normalized

    def web_search(self, query: str, iterations: int = 1) -> str:
        """Execute web search with optional iterations.

        Args:
            query: Search query
            iterations: Number of search iterations (1-5)

        Returns:
            Formatted search results
        """
        if not self.config.get("web_search_enabled", True):
            return "Web search is disabled"

        iterations = max(1, min(iterations, 5))  # Clamp between 1 and 5
        logger.debug("web_search start query=%s iterations=%s", query, iterations)

        try:
            ddgs = DDGS()
            results = ddgs.text(
                query, max_results=self.config.get("max_search_results", 20)
            )
            logger.debug("web_search results=%s", len(results) if results else 0)

            if not results:
                return f"No search results found for '{query}'"

            output = f"Search results for '{query}'"

            if iterations > 1:
                output += f" (Iteration 1 of {iterations} - Analyze these results and refine your next search)"

            output += ":\n\n"

            for i, result in enumerate(results, 1):
                output += f"{i}. {result.get('title', 'No title')}\n"
                output += f"   URL: {result.get('href', 'No URL')}\n"
                output += f"   {result.get('body', 'No description')}\n\n"

            if iterations > 1:
                output += (
                    f"\n[ITERATION GUIDANCE]\n"
                    f"You requested {iterations} search iterations.\n"
                    f"After analyzing these results:\n"
                    f"- Identify gaps in information\n"
                    f"- Refine your search query\n"
                    f"- Call web_search() again with the improved query\n"
                    f"- Repeat {iterations - 1} more time(s)\n"
                    f"Each search should build on the previous results.\n"
                )

            return output

        except Exception as e:
            error_msg = f"Error performing web search: {str(e)}"
            self.console.print(f"[red]{escape(error_msg)}[/red]")
            return error_msg

    # -------------------------
    # Composite: search_and_fetch
    # -------------------------
    def search_and_fetch(
        self,
        query: str,
        max_search_results: int = 15,
        max_fetch_pages: int = 5,
        similarity_threshold: float = 0.55,
        diversity_lambda: float = 0.4,
        fetch_concurrency: int = 3,
        include_chunks: bool = False,
        searxng_engines: Optional[List[str]] = None,
    ) -> str:
        """Search then fetch, chunk, and index most relevant pages.

        Steps:
        1. Perform web search (SearXNG if enabled, DuckDuckGo fallback) for up to max_search_results.
        2. Rank URLs by semantic similarity + MMR diversity (if display helper available).
        3. Filter by similarity_threshold; select top-N up to max_fetch_pages.
        4. Fetch pages (bounded sequentially for now; future: async).
        5. Chunk & index via WebSearchRAG (auto_index) if available.
        6. Return structured summary with citations i.e having the page's name using direct quotes and snippets and exercepts when needed & optional chunk previews.
        """
        if not self.config.get("web_search_enabled", True):
            return "Web search is disabled"

        try:
            max_search_results = max(3, min(max_search_results, 50))
            max_fetch_pages = max(1, min(max_fetch_pages, 10))
            fetch_concurrency = max(1, min(fetch_concurrency, 8))
            diversity_lambda = max(0.0, min(diversity_lambda, 1.0))
            similarity_threshold = max(0.0, min(similarity_threshold, 1.0))

            # Use SearXNG if enabled, fallback to DuckDuckGo
            logger.debug(
                "search_and_fetch start query=%s max_search_results=%s max_fetch_pages=%s",
                query,
                max_search_results,
                max_fetch_pages,
            )
            use_searxng = self.config.get("use_searxng", False)
            raw_results = []

            if use_searxng:
                try:
                    engines = searxng_engines or self.config.get(
                        "searxng_default_engines"
                    )
                    raw_results = self._searxng_search(
                        query, max_search_results, engines
                    )
                    if raw_results:
                        self.console.print(f"[dim]Using SearXNG meta-search[/dim]")
                except Exception as e:
                    self.console.print(
                        f"[yellow]SearXNG unavailable, falling back to DuckDuckGo: {e}[/yellow]"
                    )
                    raw_results = []

            # Fallback to DuckDuckGo
            if not raw_results:
                raw_results = self._ddg_search(query, max_search_results)

            if not raw_results:
                return f"No search results found for '{query}'"
            logger.debug("search_and_fetch raw_results=%s", len(raw_results))

            # Build basic list for ranking
            candidates = []
            for r in raw_results:
                candidates.append(
                    {
                        "url": r.get("href", ""),
                        "title": r.get("title", "Untitled"),
                        "snippet": r.get("body", ""),
                    }
                )

            # Use DisplayHelper ranking if available via console; fallback to simple ordering
            ranked = candidates
            try:
                # Lazy import to avoid circular
                from utils.display import DisplayHelper  # type: ignore

                # We instantiate a temporary helper ONLY for ranking (won't impact main ChatBot state)
                helper = DisplayHelper(self.console)
                # Reuse existing semantic ranking util; treat snippet as context
                # It expects a plain text block; we'll synthesize one
                synthetic_block = "\n".join(
                    f"{c['title']} {c['url']} {c['snippet']}" for c in candidates
                )
                ranked_urls = helper.extract_and_rank_urls(
                    synthetic_block, query, threshold=0.0
                )
                # Map back preserving order; ranked_urls already sorted by score desc
                url_to_score = {u["url"]: u["score"] for u in ranked_urls}
                ranked = [c for c in candidates if c["url"] in url_to_score]
                # Attach scores
                for c in ranked:
                    c["score"] = url_to_score.get(c["url"], 0.0)
                # Apply threshold
                ranked = [
                    c for c in ranked if c.get("score", 0.0) >= similarity_threshold
                ]
                # Diversity (simple MMR-like re-ranking if we still have many)
                # No external embedding model: keep top-K by score as a practical default
                ranked = sorted(
                    ranked, key=lambda x: x.get("score", 0.0), reverse=True
                )[:max_fetch_pages]
            except Exception:
                # Fallback: top-K raw order
                ranked = ranked[:max_fetch_pages]

            if not ranked:
                return f"No qualifying results (threshold {similarity_threshold}) for '{query}'"
            logger.debug("search_and_fetch ranked_results=%s", len(ranked))

            show_chunks = include_chunks or self.config.get("show_chunk_previews", True)
            summary_lines = [f"Composite search_and_fetch for '{query}'", ""]
            summary_lines.append("[RANKED URL SELECTION]")
            for idx, c in enumerate(ranked, 1):
                score = c.get("score")
                score_txt = f" (score={score:.2f})" if score is not None else ""
                summary_lines.append(f"{idx}. {c['title']} - {c['url']}{score_txt}")
                if c["snippet"]:
                    snippet = c["snippet"][:160].replace("\n", " ")
                    summary_lines.append(
                        f"    Snippet: {snippet}{'...' if len(c['snippet']) > 160 else ''}"
                    )

            # Fetch each URL
            fetched_blocks = []
            for idx, c in enumerate(ranked, 1):
                url = c["url"]
                if not url:
                    continue
                fetch_result = self.fetch_url_content(url=url, max_length=4000)
                if fetch_result.startswith("Error:"):
                    fetched_blocks.append(f"[{idx}] {url}\n   ERROR: {fetch_result}")
                    continue
                fetched_blocks.append(
                    f"[{idx}] {url}\n{fetch_result[:600]}{'...' if len(fetch_result) > 600 else ''}"
                )

            summary_lines.append("")
            summary_lines.append("[FETCHED CONTENT PREVIEWS]")
            summary_lines.extend(fetched_blocks)
            summary_lines.append("")
            summary_lines.append("CITATION SOURCES:")
            for i, c in enumerate(ranked, 1):
                summary_lines.append(f"[{i}] {c['title']} ({c['url']})")

            return "\n".join(summary_lines)
        except Exception as e:
            error_msg = f"Error executing search_and_fetch: {str(e)}"
            self.console.print(f"[red]{escape(error_msg)}[/red]")
            return error_msg

    def news_search(
        self,
        keywords: str,
        region: str = "us-en",
        safesearch: str = "moderate",
        timelimit: Optional[str] = None,
        max_results: int = 10,
        auto_fetch: bool = False,
        max_fetch_pages: int = 5,
        similarity_threshold: float = 0.35,
        include_chunks: bool = False,
    ) -> str:
        """Search for recent news articles.

        Args:
            keywords: Keywords to search for
            region: Region code (e.g., "us-en", "uk-en")
            safesearch: Safe search level ("on", "moderate", "off")
            timelimit: Time filter ("d", "w", "m", or None)
            max_results: Maximum number of results (1-50)
            auto_fetch: Enable automatic fetching, chunking, and RAG indexing
            max_fetch_pages: Maximum pages to fetch when auto_fetch=True (1-10)
            similarity_threshold: Minimum similarity for fetching (0.0-1.0)
            include_chunks: Force include chunk previews

        Returns:
            Formatted news search results
        """
        if not self.config.get("web_search_enabled", True):
            return "News search is disabled"

        try:
            logger.debug(
                "news_search start keywords=%s max_results=%s auto_fetch=%s",
                keywords,
                max_results,
                auto_fetch,
            )
            max_results = max(1, min(max_results, 50))
            max_fetch_pages = max(1, min(max_fetch_pages, 10))
            similarity_threshold = max(0.0, min(similarity_threshold, 1.0))

            ddgs = DDGS()
            results = ddgs.news(
                query=keywords,
                region=region,
                safesearch=safesearch,
                timelimit=timelimit,
                max_results=max_results,
            )

            if not results:
                return f"No news articles found for '{keywords}'"

            # Simple mode (backward compatible)
            if not auto_fetch:
                output = f"News search results for '{keywords}'"
                if timelimit:
                    time_desc = {
                        "d": "past day",
                        "w": "past week",
                        "m": "past month",
                    }.get(timelimit, "")
                    output += f" ({time_desc})"
                output += f" - Found {len(results)} articles:\n\n"

                for i, article in enumerate(results, 1):
                    output += f"{i}. {article.get('title', 'No title')}\n"
                    output += f"   Source: {article.get('source', 'Unknown source')}\n"
                    output += f"   Date: {article.get('date', 'Unknown date')}\n"
                    output += f"   URL: {article.get('url', 'No URL')}\n"

                    body = article.get("body", "")
                    if body:
                        body_preview = body[:300] + "..." if len(body) > 300 else body
                        output += f"   Summary: {body_preview}\n"

                    if article.get("image"):
                        output += f"   Image: {article['image']}\n"

                    output += "\n"

                output += "\n[TIP] Use fetch_url_content(url='...') to read the full article text from any URL above.\n"
                output += "[TIP] Or use auto_fetch=true to automatically fetch and index top articles.\n"

                return output

            # Auto-fetch mode: semantic ranking + fetching + RAG indexing
            # Build candidates for ranking
            candidates = []
            for article in results:
                candidates.append(
                    {
                        "url": article.get("url", ""),
                        "title": article.get("title", "No title"),
                        "snippet": article.get("body", ""),
                        "source": article.get("source", "Unknown source"),
                        "date": article.get("date", "Unknown date"),
                        "image": article.get("image", ""),
                    }
                )

            # Semantic ranking
            ranked = candidates
            ranking_succeeded = False
            try:
                from utils.display import DisplayHelper

                helper = DisplayHelper(self.console)
                # Create synthetic block for ranking
                synthetic_block = "\n".join(
                    f"{c['title']} {c['url']} {c['snippet']}" for c in candidates
                )
                ranked_urls = helper.extract_and_rank_urls(
                    synthetic_block, keywords, threshold=0.0
                )

                if ranked_urls:
                    # Map scores back
                    url_to_score = {u["url"]: u["score"] for u in ranked_urls}
                    ranked = [c for c in candidates if c["url"] in url_to_score]
                    for c in ranked:
                        c["score"] = url_to_score.get(c["url"], 0.0)

                    # Apply threshold and sort
                    ranked = [
                        c for c in ranked if c.get("score", 0.0) >= similarity_threshold
                    ]
                    ranked = sorted(
                        ranked, key=lambda x: x.get("score", 0.0), reverse=True
                    )[:max_fetch_pages]
                    ranking_succeeded = True

                    # Debug output
                    self.console.print(
                        f"[dim]Semantic ranking: {len(ranked)} articles above threshold {similarity_threshold}[/dim]"
                    )
            except Exception as e:
                self.console.print(
                    f"[dim yellow]Warning: Semantic ranking failed ({str(e)[:50]}), using raw order[/dim yellow]"
                )

            # Fallback: use raw order if ranking failed or no results above threshold
            if not ranked:
                if ranking_succeeded:
                    self.console.print(
                        f"[dim yellow]No articles above threshold {similarity_threshold}, using top {max_fetch_pages} by raw order[/dim yellow]"
                    )
                ranked = candidates[:max_fetch_pages]
                # Mark as raw order (no score)
                for c in ranked:
                    c["score"] = None

            # Build output
            show_chunks = include_chunks or self.config.get("show_chunk_previews", True)
            summary_lines = [f"News search with auto-fetch for '{keywords}'"]
            if timelimit:
                time_desc = {"d": "past day", "w": "past week", "m": "past month"}.get(
                    timelimit, ""
                )
                summary_lines[0] += f" ({time_desc})"
            summary_lines.append("")
            summary_lines.append("[RANKED NEWS ARTICLES]")

            for idx, c in enumerate(ranked, 1):
                score = c.get("score")
                score_txt = f" (score={score:.2f})" if score is not None else ""
                summary_lines.append(f"{idx}. {c['title']}{score_txt}")
                summary_lines.append(f"    Source: {c['source']} | Date: {c['date']}")
                summary_lines.append(f"    URL: {c['url']}")
                if c["snippet"]:
                    snippet = c["snippet"][:200].replace("\n", " ")
                    summary_lines.append(
                        f"    Summary: {snippet}{'...' if len(c['snippet']) > 200 else ''}"
                    )
                summary_lines.append("")

            # Fetch articles
            # TODO: Add async fetching with asyncio + aiohttp for parallel concurrency
            # (see search_and_fetch pattern - use fetch_concurrency parameter)
            fetched_blocks = []
            for idx, c in enumerate(ranked, 1):
                url = c["url"]
                if not url:
                    continue

                self.console.print(
                    f"[dim]Fetching article {idx}/{len(ranked)}: {c['title'][:50]}...[/dim]"
                )
                # Fetch full content for RAG indexing (max allowed: 20000 chars)
                fetch_result = self.fetch_url_content(url=url, max_length=20000)

                if fetch_result.startswith("Error:"):
                    fetched_blocks.append(
                        f"[{idx}] {c['title']} - {url}\n   ERROR: {fetch_result}"
                    )
                    continue

                preview = fetch_result[:600].replace("\n", " ")
                fetched_blocks.append(
                    f"[{idx}] {c['title']} - {c['source']} ({c['date']})\n"
                    f"    {url}\n"
                    f"    {preview}{'...' if len(fetch_result) > 600 else ''}"
                )

            summary_lines.append("[FETCHED CONTENT PREVIEWS]")
            summary_lines.extend(fetched_blocks)
            summary_lines.append("")
            summary_lines.append("CITATION SOURCES:")
            for i, c in enumerate(ranked, 1):
                summary_lines.append(
                    f"[{i}] {c['title']} - {c['source']} ({c['date']}) [{c['url']}]"
                )

            return "\n".join(summary_lines)

        except Exception as e:
            error_msg = f"Error performing news search: {str(e)}"
            self.console.print(f"[red]{escape(error_msg)}[/red]")
            return error_msg

    def fetch_url_content(self, url: str, max_length: int = 5000) -> str:
        """Fetch and extract clean text content from a URL.

        Args:
            url: URL to fetch content from
            max_length: Maximum character length (500-20000)

        Returns:
            Extracted text content
        """
        if not url.startswith(("http://", "https://")):
            return "Error: Invalid URL format. URL must start with http:// or https://"

        # Check for unsupported content types/URLs
        unsupported_patterns = [
            "youtube.com",
            "youtu.be",  # Video platforms
            "vimeo.com",
            "dailymotion.com",
            "twitch.tv",
            "tiktok.com",
            ".mp4",
            ".avi",
            ".mov",
            ".wmv",
            ".flv",  # Video files
            ".mp3",
            ".wav",
            ".ogg",
            ".flac",  # Audio files
            ".pdf",  # PDFs (could be supported separately in future)
            ".zip",
            ".rar",
            ".tar",
            ".gz",  # Archives
            ".exe",
            ".dmg",
            ".apk",  # Executables
        ]

        url_lower = url.lower()
        for pattern in unsupported_patterns:
            if pattern in url_lower:
                content_type = (
                    "video"
                    if any(
                        vid in pattern
                        for vid in ["youtube", "youtu", "vimeo", "mp4", "avi", "mov"]
                    )
                    else "unsupported content"
                )
                return f"Error: Cannot extract text from {content_type}. URL contains {pattern}. Please search for text-based articles or documentation instead."

        try:
            max_length = max(500, min(max_length, 20000))
            logger.debug(
                "fetch_url_content start url=%s max_length=%s", url, max_length
            )

            config = use_config()
            config.set("DEFAULT", "EXTRACTION_TIMEOUT", "10")

            self.console.print(f"[dim]Fetching content from {url[:60]}...[/dim]")

            # First check content type with a HEAD request
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
                head_response = requests.head(
                    url, timeout=5, headers=headers, allow_redirects=True
                )
                content_type = head_response.headers.get("content-type", "").lower()

                # Check if content type is not HTML/text
                if content_type and not any(
                    ct in content_type
                    for ct in ["text/html", "text/plain", "application/xhtml"]
                ):
                    if "video" in content_type or "audio" in content_type:
                        return f"Error: Cannot extract text from media content (Content-Type: {content_type}). Please search for text-based articles."
                    elif "pdf" in content_type:
                        return f"Error: PDF content detected. PDF text extraction not currently supported."
                    elif any(
                        binary in content_type
                        for binary in [
                            "image",
                            "application/octet-stream",
                            "application/zip",
                        ]
                    ):
                        return f"Error: Binary content detected (Content-Type: {content_type}). Cannot extract text."
            except:
                # If HEAD request fails, continue with normal fetch attempt
                pass

            downloaded = trafilatura.fetch_url(url)

            if not downloaded:
                # Fallback to requests
                try:
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    }
                    response = requests.get(url, timeout=10, headers=headers)
                    response.raise_for_status()
                    downloaded = response.text
                except Exception as req_error:
                    return f"Error fetching URL: Could not download content. Error: {str(req_error)}"

            content = trafilatura.extract(
                downloaded,
                include_formatting=False,
                include_links=False,
                include_images=False,
                include_tables=True,
                config=config,
                favor_recall=True,
            )

            if not content:
                return f"Error: Could not extract readable content from {url}"

            if len(content) > max_length:
                content = (
                    content[:max_length]
                    + f"\n\n[Content truncated at {max_length} characters. Original: {len(content)} chars]"
                )
            logger.debug("fetch_url_content extracted_length=%s", len(content))

            output = f"Content extracted from {url}:\n"
            output += "=" * 60 + "\n\n"
            output += content
            output += "\n\n" + "=" * 60
            output += f"\n[Extracted {len(content)} characters]"

            return output

        except Exception as e:
            error_msg = f"Error fetching content from {url}: {str(e)}"
            self.console.print(f"[red]{escape(error_msg)}[/red]")
            return error_msg

    def calculate(self, expression: str) -> str:
        """Perform mathematical calculation.

        Args:
            expression: Mathematical expression to evaluate

        Returns:
            Calculation result
        """
        try:
            allowed_names = {
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
                "sum": sum,
                "pow": pow,
                "sqrt": math.sqrt,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "pi": math.pi,
                "e": math.e,
            }
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return f"Result: {result}"
        except Exception as e:
            return f"Error calculating: {str(e)}"

    def get_current_time(self) -> str:
        """Get current date and time.

        Returns:
            Formatted current date and time
        """
        now = datetime.now()
        return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"

    def search_wikipedia(
        self, query: str, top_k: int = 2, max_chars: int = 3000, auto_index: bool = True
    ) -> str:
        """Search Wikipedia for encyclopedic information.

        Args:
            query: Search query
            top_k: Number of articles to retrieve (1-5)
            max_chars: Maximum characters per article summary
            auto_index: Whether to index full articles into RAG

        Returns:
            Formatted Wikipedia search results with inline URL citations
        """
        try:
            logger.debug(
                "search_wikipedia start query=%s top_k=%s max_chars=%s",
                query,
                top_k,
                max_chars,
            )
            top_k = max(1, min(top_k, 5))
            max_chars = max(500, min(max_chars, 10000))

            # Initialize Wikipedia API with user agent
            wiki = wikipediaapi.Wikipedia(
                language="en",
                user_agent="PersonalAI-Chatbot/1.0 (Educational RAG project)",
            )

            # Search for pages - wikipediaapi doesn't have direct search, so we use the page method
            # For better search, we'll try to get the page directly first
            page = wiki.page(query)

            results = []
            if page.exists():
                results.append(page)
            else:
                # If direct match fails, try common variations
                variations = [
                    query.title(),  # Title case
                    query.lower(),  # Lower case
                    query.upper(),  # Upper case
                    query.replace(" ", "_"),  # Underscores
                ]
                for variant in variations:
                    page = wiki.page(variant)
                    if page.exists():
                        results.append(page)
                        break

            if not results:
                return f"No Wikipedia articles found for '{query}'. Try being more specific or using full proper nouns."
            logger.debug("search_wikipedia results=%s", len(results))

            output_lines = [f"Wikipedia search results for '{escape(query)}':\n"]
            indexed_chunks = 0

            for idx, page in enumerate(results[:top_k], 1):
                # Extract summary
                summary = page.summary[:max_chars]
                if len(page.summary) > max_chars:
                    summary += "..."

                # Format output with inline citation
                output_lines.append(f"{idx}. **{escape(page.title)}**")
                output_lines.append(f"   URL: {page.fullurl}")
                output_lines.append(f"   Summary: {escape(summary)}\n")

            # Add citation format guidance
            output_lines.append("\nINLINE CITATIONS:")
            for idx, page in enumerate(results[:top_k], 1):
                output_lines.append(
                    f"[{idx}] {page.title} - Wikipedia ({page.fullurl})"
                )

            output_lines.append(
                "\nWhen answering, cite sources inline like: 'Quantum computing uses qubits [1]...'"
            )

            return "\n".join(output_lines)

        except Exception as e:
            error_msg = f"Error searching Wikipedia: {str(e)}"
            self.console.print(f"[red]{escape(error_msg)}[/red]")
            return error_msg

    def search_arxiv(
        self,
        query: str,
        max_results: int = 3,
        get_full_text: bool = False,
        sort_by: str = "relevance",
        auto_index: bool = True,
    ) -> str:
        """Search arXiv for academic papers.

        Args:
            query: Search query
            max_results: Maximum number of papers (1-10)
            get_full_text: Whether to download and parse full PDFs
            sort_by: Sort order ('relevance', 'lastUpdatedDate', 'submittedDate')
            auto_index: Whether to index papers into RAG

        Returns:
            Formatted arXiv search results with inline URL citations
        """
        try:
            logger.debug(
                "search_arxiv start query=%s max_results=%s get_full_text=%s",
                query,
                max_results,
                get_full_text,
            )
            max_results = max(1, min(max_results, 10))

            # Map sort parameter
            sort_map = {
                "relevance": arxiv.SortCriterion.Relevance,
                "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
                "submittedDate": arxiv.SortCriterion.SubmittedDate,
            }
            sort_criterion = sort_map.get(sort_by, arxiv.SortCriterion.Relevance)

            # Search arXiv
            self.console.print(f"[dim]Searching arXiv for: {escape(query)}...[/dim]")
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=sort_criterion,
                sort_order=arxiv.SortOrder.Descending,
            )

            results = list(search.results())

            if not results:
                return f"No arXiv papers found for '{query}'"

            output_lines = [f"arXiv search results for '{escape(query)}':\n"]
            indexed_chunks = 0

            for idx, paper in enumerate(results, 1):
                # Format authors
                authors = ", ".join([author.name for author in paper.authors[:3]])
                if len(paper.authors) > 3:
                    authors += f" et al. ({len(paper.authors)} total)"

                # Extract content - abstract by default, full PDF if requested
                content = paper.summary

                if get_full_text:
                    self.console.print(
                        f"[dim]Downloading PDF: {escape(paper.title[:50])}...[/dim]"
                    )
                    try:
                        # Download PDF to temporary location
                        pdf_path = paper.download_pdf(dirpath="/tmp")

                        # Extract text using PyMuPDF
                        doc = fitz.open(pdf_path)
                        full_text = ""
                        for page_num in range(
                            min(len(doc), 20)
                        ):  # Limit to first 20 pages
                            page = doc[page_num]
                            # Cast to str to satisfy static typing; PyMuPDF returns str for default mode
                            full_text += str(page.get_text())
                        doc.close()

                        if full_text.strip():
                            content = full_text
                            self.console.print(
                                f"[dim]Extracted {len(full_text)} characters from PDF[/dim]"
                            )
                        else:
                            self.console.print(
                                f"[dim yellow]Warning: PDF extraction yielded no text, using abstract[/dim yellow]"
                            )
                    except Exception as pdf_error:
                        self.console.print(
                            f"[dim yellow]Warning: PDF download failed: {escape(str(pdf_error)[:50])}, using abstract[/dim yellow]"
                        )

                # Format output with inline citation
                output_lines.append(f"{idx}. **{escape(paper.title)}**")
                output_lines.append(f"   Authors: {escape(authors)}")
                output_lines.append(
                    f"   Published: {paper.published.strftime('%Y-%m-%d')}"
                )
                output_lines.append(f"   arXiv ID: {paper.entry_id.split('/')[-1]}")
                output_lines.append(f"   URL: {paper.entry_id}")
                output_lines.append(f"   PDF: {paper.pdf_url}")
                output_lines.append(f"   Categories: {', '.join(paper.categories)}")

                # Show abstract preview (even if we have full text)
                abstract_preview = paper.summary[:400]
                if len(paper.summary) > 400:
                    abstract_preview += "..."
                output_lines.append(f"   Abstract: {escape(abstract_preview)}\n")

            # Add citation format guidance
            output_lines.append("\nINLINE CITATIONS:")
            for idx, paper in enumerate(results, 1):
                authors_short = ", ".join([author.name for author in paper.authors[:2]])
                if len(paper.authors) > 2:
                    authors_short += " et al."
                year = paper.published.strftime("%Y")
                output_lines.append(
                    f"[{idx}] {authors_short} ({year}). {paper.title}. arXiv:{paper.entry_id.split('/')[-1]} ({paper.entry_id})"
                )

            output_lines.append(
                "\nWhen answering, cite sources inline like: 'Transformers achieve state-of-the-art results [1]...'"
            )

            return "\n".join(output_lines)

        except Exception as e:
            error_msg = f"Error searching arXiv: {str(e)}"
            self.console.print(f"[red]{escape(error_msg)}[/red]")
            return error_msg

    # =========================================================================
    # DEEP RESEARCH - Multi-step recursive research agent
    # =========================================================================

    def deep_research(
        self,
        topic: str,
        depth: int = 3,
        breadth: int = 4,
        quality_threshold: float = 0.75,
        include_academic: bool = True,
    ) -> str:
        """
        Performs recursive research with planning, execution, evaluation, and synthesis.

        Args:
            topic: Main research subject
            depth: How many recursive iterations (1-5)
            breadth: Number of sub-queries per iteration (2-6)
            quality_threshold: Minimum quality score to stop early (0.0-1.0)
            include_academic: Include academic sources (Semantic Scholar, arXiv)

        Returns:
            Comprehensive research report with citations
        """
        self.console.print(
            f"[bold magenta]Deep research started: {escape(topic)}[/bold magenta]"
        )

        # Clamp parameters
        depth = max(1, min(depth, 5))
        breadth = max(2, min(breadth, 6))
        quality_threshold = max(0.0, min(quality_threshold, 1.0))

        # State tracking
        research_state = {
            "topic": topic,
            "current_depth": 0,
            "max_depth": depth,
            "findings": [],
            "indexed_sources": set(),
            "quality_scores": [],
        }

        # Recursive research loop
        while research_state["current_depth"] < research_state["max_depth"]:
            current_depth = research_state["current_depth"]
            self.console.print(
                f"\n[cyan]Research depth: {current_depth + 1}/{depth}[/cyan]"
            )

            # STEP 1: PLANNING - Generate sub-queries
            self.console.print(f"[dim]Generating search queries...[/dim]")
            sub_queries = self._generate_sub_queries(
                topic=topic,
                previous_findings=research_state["findings"],
                breadth=breadth,
                depth_level=current_depth,
            )

            self.console.print(f"[dim]Sub-queries: {sub_queries}[/dim]")

            # STEP 2: EXECUTION - Multi-source search for each sub-query
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                search_task = progress.add_task(
                    f"[cyan]Investigating {len(sub_queries)} sub-topics...",
                    total=len(sub_queries),
                )

                for i, query in enumerate(sub_queries):
                    self.console.print(
                        f"\n[bold blue][{i + 1}/{len(sub_queries)}] Searching: {query}[/bold blue]"
                    )

                    # Multi-source search
                    findings = self._execute_multi_source_search(
                        query=query, include_academic=include_academic
                    )

                    # Show a preview of what was found
                    if findings and len(findings) > 100:
                        preview = findings[:300].replace("\n", " ")
                        self.console.print(f"[dim]Found: {preview}...[/dim]")

                    research_state["findings"].append(
                        {"query": query, "content": findings, "depth": current_depth}
                    )
                    progress.update(search_task, advance=1)

            # STEP 3: EVALUATION - Assess research quality
            self.console.print(f"\n[dim]Evaluating research quality...[/dim]")
            quality_score = self._evaluate_research_quality(
                topic=topic, findings=research_state["findings"]
            )
            research_state["quality_scores"].append(quality_score)

            self.console.print(
                f"[yellow]Quality Score: {quality_score:.2f}/1.0[/yellow]"
            )

            # STEP 4: ESCALATION CHECK - Stop if quality threshold met
            if quality_score >= quality_threshold:
                self.console.print(
                    "[green]Quality threshold reached. Stopping early.[/green]"
                )
                break

            research_state["current_depth"] += 1

        # STEP 5: SYNTHESIS - Generate final report
        self.console.print(
            f"\n[bold magenta]Synthesizing research report...[/bold magenta]"
        )
        report = self._synthesize_report(
            topic=topic,
            findings=research_state["findings"],
            depth_reached=research_state["current_depth"],
        )

        return report

    def _generate_sub_queries(
        self, topic: str, previous_findings: List[Dict], breadth: int, depth_level: int
    ) -> List[str]:
        """
        Uses LLM to decompose topic into searchable sub-questions.
        Adapts based on previous findings (recursive refinement).
        """
        # Context from previous iteration (if any)
        context = ""
        if previous_findings:
            recent = previous_findings[-breadth:]
            context = "\n".join(
                [f"- {f['query']}: {f['content'][:200]}..." for f in recent]
            )

        planning_prompt = f"""You are a research planning assistant. Break down the following research topic into {breadth} distinct, specific sub-questions for web search investigation.

Research Topic: {topic}
Current Depth: {depth_level + 1}

{f"Previous Findings (use to identify gaps):\n{context}" if context else ""}

Requirements:
- Each sub-question must be a practical web search query (not overly academic)
- Avoid redundancy with previous queries
- Focus on different aspects: definition/overview, tools/software, best practices, examples, community resources
- Make queries specific and searchable

Return a JSON object with a "queries" array containing {breadth} search queries.
Example format: {{"queries": ["query 1", "query 2", "query 3"]}}"""

        try:
            provider = self._get_llama_cpp_provider()
            response = provider.chat(
                messages=[{"role": "user", "content": planning_prompt}],
                stream=False,
            )

            # OpenAI-compatible format
            content = ""
            if isinstance(response, dict):
                choices = response.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")

            raw_content = content
            self.console.print(f"[dim]Planning response: {raw_content[:200]}...[/dim]")

            parsed = json.loads(raw_content)

            # Try multiple JSON formats
            queries = []
            if isinstance(parsed, dict):
                # Try common keys
                for key in [
                    "queries",
                    "sub_queries",
                    "questions",
                    "search_queries",
                    "list",
                ]:
                    if key in parsed and isinstance(parsed[key], list):
                        queries = [str(q) for q in parsed[key] if q]
                        break
                # If no list found, try to find any list value
                if not queries:
                    for v in parsed.values():
                        if isinstance(v, list) and len(v) > 0:
                            queries = [str(q) for q in v if q]
                            break
            elif isinstance(parsed, list):
                queries = [str(q) for q in parsed if q]

            if queries:
                return queries[:breadth]

            raise ValueError(f"Could not extract queries from: {raw_content[:100]}")

        except Exception as e:
            self.console.print(f"[yellow]Planning fallback: {e}[/yellow]")
            # Generate practical fallback queries based on topic
            base_queries = [
                f"{topic} guide",
                f"{topic} tools software",
                f"{topic} best practices",
                f"{topic} examples tutorials",
                f"{topic} community resources",
                f"how to {topic}",
            ]
            return base_queries[:breadth]

    def _execute_multi_source_search(
        self, query: str, include_academic: bool = True
    ) -> str:
        """
        Executes search across multiple sources (web + academic).
        Uses multiple fallback strategies to ensure results.
        """
        results = []
        got_web_results = False

        # Try search_and_fetch first with lower threshold for niche topics
        try:
            web_result = self.search_and_fetch(
                query=query,
                max_search_results=10,
                max_fetch_pages=3,
                similarity_threshold=0.3,  # Lower threshold for niche topics
            )
            # Check if we actually got results
            if (
                web_result
                and "No qualifying results" not in web_result
                and "No search results" not in web_result
            ):
                results.append(f"## Web Sources\n{web_result}")
                got_web_results = True
        except Exception as e:
            self.console.print(f"[yellow]search_and_fetch error: {e}[/yellow]")

        # Fallback to basic web_search if search_and_fetch failed
        if not got_web_results:
            try:
                self.console.print(f"[dim]Falling back to basic web search...[/dim]")
                web_result = self.web_search(query=query, iterations=1)
                if web_result and "No search results" not in web_result:
                    results.append(f"## Web Sources\n{web_result}")
                    got_web_results = True
            except Exception as e:
                self.console.print(f"[yellow]Web search fallback error: {e}[/yellow]")

        # Always try Wikipedia for conceptual/definitional queries
        try:
            # Extract key terms for Wikipedia (remove common words)
            wiki_query = " ".join(
                [
                    w
                    for w in query.split()
                    if len(w) > 3
                    and w.lower()
                    not in [
                        "what",
                        "are",
                        "the",
                        "how",
                        "does",
                        "overview",
                        "concepts",
                        "recent",
                        "developments",
                    ]
                ]
            )
            if wiki_query:
                wiki_result = self.search_wikipedia(
                    query=wiki_query, top_k=1, max_chars=2000
                )
                if wiki_result and "No Wikipedia articles found" not in wiki_result:
                    results.append(f"## Wikipedia\n{wiki_result}")
        except Exception as e:
            self.console.print(f"[dim]Wikipedia search skipped: {e}[/dim]")

        if not include_academic:
            return "\n\n".join(results) if results else f"No results found for: {query}"

        # Only search academic sources if query seems academically relevant
        # Check for explicit academic indicators
        academic_indicators = [
            "research",
            "study",
            "paper",
            "journal",
            "empirical",
            "theory",
            "methodology",
            "analysis",
            "evidence",
            "algorithm",
            "framework",
            "neural network",
            "machine learning",
        ]

        query_lower = query.lower()
        is_academic = any(ind in query_lower for ind in academic_indicators)

        # Avoid academic search for clearly non-academic topics
        non_academic_indicators = [
            "setup",
            "how to",
            "best",
            "tutorial",
            "guide",
            "tips",
            "coding setup",
            "development environment",
        ]
        is_practical = any(ind in query_lower for ind in non_academic_indicators)

        if is_academic and not is_practical:
            try:
                academic_result = self.search_academic(query, limit=3)
                if (
                    academic_result
                    and "No academic papers found" not in academic_result
                ):
                    results.append(f"## Academic Sources\n{academic_result}")
            except Exception as e:
                self.console.print(f"[dim]Academic search skipped: {e}[/dim]")

        # For medical/biomedical queries only
        medical_keywords = [
            "disease",
            "treatment",
            "drug",
            "clinical",
            "patient",
            "diagnosis",
            "therapy",
            "cancer",
            "covid",
            "vaccine",
        ]

        if any(kw in query_lower for kw in medical_keywords):
            try:
                pubmed_result = self.search_pubmed(query, limit=3)
                if pubmed_result and "No PubMed articles found" not in pubmed_result:
                    results.append(f"## Biomedical Sources\n{pubmed_result}")
            except Exception as e:
                self.console.print(f"[dim]PubMed search skipped: {e}[/dim]")

        # arXiv for technical CS/physics topics only
        arxiv_keywords = [
            "neural",
            "deep learning",
            "transformer",
            "quantum",
            "optimization",
        ]

        if any(kw in query_lower for kw in arxiv_keywords):
            try:
                arxiv_result = self.search_arxiv(query, max_results=2)
                if arxiv_result and "No arXiv papers found" not in arxiv_result:
                    results.append(f"## arXiv Papers\n{arxiv_result}")
            except Exception as e:
                self.console.print(f"[dim]arXiv search skipped: {e}[/dim]")

        return "\n\n".join(results) if results else f"No results found for: {query}"

    def _evaluate_research_quality(self, topic: str, findings: List[Dict]) -> float:
        """
        Uses LLM to assess if current research sufficiently addresses the topic.
        Returns quality score (0.0-1.0).
        """
        # Prepare findings summary
        findings_text = "\n\n".join(
            [
                f"Sub-topic: {f['query']}\nFindings: {f['content'][:500]}..."
                for f in findings[-6:]  # Last 6 findings
            ]
        )

        evaluation_prompt = f"""You are a research quality evaluator. Assess if the following findings adequately address the research topic.

Research Topic: {topic}

Findings:
{findings_text}

Evaluate on these criteria:
1. Coverage: Are multiple aspects of the topic addressed?
2. Depth: Is there sufficient detail in each aspect?
3. Credibility: Are findings from reliable sources?
4. Coherence: Do findings fit together logically?

Return ONLY a JSON object with:
{{
  "score": <float 0.0-1.0>,
  "reasoning": "<brief explanation>",
  "gaps": ["<missing aspect 1>", "<missing aspect 2>"]
}}"""

        try:
            provider = self._get_llama_cpp_provider()
            response = provider.chat(
                messages=[{"role": "user", "content": evaluation_prompt}],
                stream=False,
            )

            # OpenAI-compatible format
            content = ""
            if isinstance(response, dict):
                choices = response.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")

            result = json.loads(content)
            score = float(result.get("score", 0.5))

            self.console.print(
                f"[dim]Evaluation: {result.get('reasoning', 'N/A')}[/dim]"
            )
            if result.get("gaps"):
                self.console.print(
                    f"[dim]Gaps identified: {', '.join(result['gaps'])}[/dim]"
                )

            return max(0.0, min(1.0, score))  # Clamp to [0, 1]

        except Exception as e:
            self.console.print(
                f"[yellow]Evaluation error: {e}. Using default score.[/yellow]"
            )
            return 0.5  # Neutral score on error

    def _synthesize_report(
        self, topic: str, findings: List[Dict], depth_reached: int
    ) -> str:
        """
        Generates a comprehensive, citation-rich research report.
        """
        # Retrieve relevant context from web searches

        # Organize findings by depth
        findings_by_depth = {}
        for f in findings:
            d = f.get("depth", 0)
            if d not in findings_by_depth:
                findings_by_depth[d] = []
            findings_by_depth[d].append(f)

        findings_text = ""
        for depth_level, items in sorted(findings_by_depth.items()):
            findings_text += f"\n## Research Phase {depth_level + 1}\n"
            for item in items:
                findings_text += f"### {item['query']}\n{item['content'][:600]}...\n\n"

        synthesis_prompt = f"""You are an expert research synthesizer. Create a comprehensive report on the following topic using the provided research findings.

Topic: {topic}
Depth Reached: {depth_reached + 1} iterations

Research Findings:
{findings_text}

Instructions:
1. Write a well-structured report with:
   - Executive Summary
   - Key Findings (organized by theme, not chronologically)
   - Detailed Analysis
   - Limitations & Future Research
2. **Cite sources** inline using [Source: URL] format
3. Synthesize conflicting information objectively
4. Highlight areas with insufficient data
5. Use academic tone but remain accessible
6. Minimum 2-4 unique sources cited

Format: Markdown"""

        try:
            provider = self._get_llama_cpp_provider()
            response = provider.chat(
                messages=[{"role": "user", "content": synthesis_prompt}],
                stream=False,
            )

            # OpenAI-compatible format
            content = ""
            if isinstance(response, dict):
                choices = response.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")

            report = content
        except Exception as e:
            self.console.print(f"[red]Synthesis error: {e}[/red]")
            report = f"# Research Report: {topic}\n\nError generating synthesis. Raw findings below:\n\n{findings_text}"

        # Add metadata footer
        report += f"\n\n---\n**Research Metadata**\n"
        report += f"- Topic: {topic}\n"
        report += f"- Research Depth: {depth_reached + 1} iterations\n"
        report += f"- Total Sub-Queries: {len(findings)}\n"

        return report

    # =========================================================================
    # ACADEMIC SEARCH TOOLS
    # =========================================================================

    def search_academic(
        self,
        query: str,
        limit: int = 10,
        year_filter: Optional[str] = None,
        fields_of_study: Optional[List[str]] = None,
    ) -> str:
        """
        Searches Semantic Scholar for academic papers and auto-indexes results.

        Args:
            query: Search query
            limit: Number of papers to retrieve (1-100)
            year_filter: Year range (e.g., "2020-2024" or "2023-")
            fields_of_study: List of fields (e.g., ["Computer Science", "Medicine"])

        Returns:
            Formatted academic results with citations
        """
        url = "https://api.semanticscholar.org/graph/v1/paper/search"

        # Build query params
        params = {
            "query": query,
            "limit": min(limit, 100),
            "fields": "title,authors,year,abstract,url,citationCount,openAccessPdf,venue,publicationDate,fieldsOfStudy",
        }

        if year_filter:
            params["year"] = year_filter

        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)

        try:
            logger.debug("search_academic start query=%s limit=%s", query, limit)
            self.console.print(f"[dim]Querying Semantic Scholar: {escape(query)}[/dim]")

            headers = {"User-Agent": "PersonalAI-Chatbot/1.0 (Educational RAG project)"}
            # Add API key if available (optional but recommended for higher rate limits)
            if self.config.get("semantic_scholar_api_key"):
                headers["x-api-key"] = self.config["semantic_scholar_api_key"]

            response = requests.get(url, params=params, headers=headers, timeout=15)

            if response.status_code == 429:
                return "Rate limit exceeded for Semantic Scholar. Please try again later or use an API key."

            if not response.ok:
                return f"Academic search failed: HTTP {response.status_code}"

            data = response.json()

            if not data.get("data"):
                return f"No academic papers found for: {query}"

            # Format and index results
            formatted_results = []
            indexed_chunks = 0

            for paper in data["data"][:limit]:
                # Extract key info
                title = paper.get("title", "Unknown Title")
                year = paper.get("year", "N/A")
                authors_list = paper.get("authors", [])
                authors = ", ".join(
                    [a.get("name", "Unknown") for a in authors_list[:3]]
                )
                if len(authors_list) > 3:
                    authors += " et al."

                citation_count = paper.get("citationCount", 0)
                abstract = (
                    paper.get("abstract", "No abstract available.")
                    or "No abstract available."
                )
                paper_url = paper.get("url", "")
                pdf_info = paper.get("openAccessPdf")
                pdf_url = pdf_info.get("url", "No PDF") if pdf_info else "No PDF"
                venue = paper.get("venue", "Unknown Venue") or "Unknown Venue"

                # Format for display
                formatted = f"""
**{title}** ({year})
*Authors:* {authors}
*Venue:* {venue} | *Citations:* {citation_count}
*Abstract:* {abstract[:400]}{"..." if len(abstract) > 400 else ""}
*URL:* {paper_url}
*PDF:* {pdf_url}
"""
                formatted_results.append(formatted)

            result_text = "\n---\n".join(formatted_results)
            return f"Found {len(formatted_results)} academic papers:\n\n{result_text}"

        except Exception as e:
            return f"Error in academic search: {str(e)}"

    def search_pubmed(
        self, query: str, limit: int = 10, sort: str = "relevance"
    ) -> str:
        """
        Searches PubMed for biomedical literature using NCBI E-utilities.

        Args:
            query: Search query (supports PubMed query syntax)
            limit: Number of results (1-50)
            sort: Sort order ("relevance" or "date")

        Returns:
            Formatted PubMed results
        """
        try:
            logger.debug("search_pubmed start query=%s limit=%s", query, limit)
            self.console.print(f"[dim]Querying PubMed: {escape(query)}[/dim]")

            limit = max(1, min(limit, 50))

            # Use NCBI E-utilities API (no BioPython dependency)
            base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

            # Get email from config for NCBI compliance
            email = self.config.get("pubmed_email", "personalai@example.com")

            # Step 1: Search for IDs
            search_url = f"{base_url}/esearch.fcgi"
            search_params = {
                "db": "pubmed",
                "term": query,
                "retmax": limit,
                "retmode": "json",
                "sort": "relevance" if sort == "relevance" else "pub_date",
                "email": email,
            }

            search_response = requests.get(search_url, params=search_params, timeout=15)
            if not search_response.ok:
                return f"PubMed search failed: HTTP {search_response.status_code}"

            search_data = search_response.json()
            id_list = search_data.get("esearchresult", {}).get("idlist", [])

            if not id_list:
                return f"No PubMed articles found for: {query}"

            # Step 2: Fetch article details
            fetch_url = f"{base_url}/efetch.fcgi"
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(id_list),
                "retmode": "xml",
                "email": email,
            }

            fetch_response = requests.get(fetch_url, params=fetch_params, timeout=15)
            if not fetch_response.ok:
                return f"PubMed fetch failed: HTTP {fetch_response.status_code}"

            # Parse XML response
            import xml.etree.ElementTree as ET

            root = ET.fromstring(fetch_response.text)

            formatted_results = []
            indexed_chunks = 0

            for article in root.findall(".//PubmedArticle"):
                medline = article.find(".//MedlineCitation")
                if medline is None:
                    continue

                # Extract article info
                article_elem = medline.find(".//Article")
                if article_elem is None:
                    continue

                title_elem = article_elem.find(".//ArticleTitle")
                title = title_elem.text if title_elem is not None else "Unknown Title"

                abstract_elem = article_elem.find(".//Abstract/AbstractText")
                abstract = (
                    abstract_elem.text
                    if abstract_elem is not None
                    else "No abstract available."
                )

                pmid_elem = medline.find(".//PMID")
                pmid = pmid_elem.text if pmid_elem is not None else "N/A"

                journal_elem = article_elem.find(".//Journal/Title")
                journal = (
                    journal_elem.text if journal_elem is not None else "Unknown Journal"
                )

                # Extract year
                year_elem = article_elem.find(".//Journal/JournalIssue/PubDate/Year")
                year = year_elem.text if year_elem is not None else "N/A"

                # Extract authors
                authors = []
                for author in article_elem.findall(".//AuthorList/Author")[:3]:
                    last_name = author.find("LastName")
                    initials = author.find("Initials")
                    if last_name is not None:
                        name = last_name.text
                        if initials is not None:
                            name += f" {initials.text}"
                        authors.append(name)

                author_count = len(article_elem.findall(".//AuthorList/Author"))
                authors_str = ", ".join(authors)
                if author_count > 3:
                    authors_str += " et al."

                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

                formatted = f"""
**{title}** ({year})
*Authors:* {authors_str}
*Journal:* {journal} | *PMID:* {pmid}
*Abstract:* {abstract[:400] if abstract else "No abstract"}{"..." if abstract and len(abstract) > 400 else ""}
*URL:* {url}
"""
                formatted_results.append(formatted)

            result_text = "\n---\n".join(formatted_results)
            return f"Found {len(formatted_results)} PubMed articles:\n\n{result_text}"

        except Exception as e:
            return f"Error in PubMed search: {str(e)}"

    # =========================================================================
    # SearXNG Integration
    # =========================================================================

    def _searxng_search(
        self, query: str, max_results: int, engines: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Queries SearXNG meta-search API.
        Returns list of search results with URL, title, content.
        Returns empty list on any error (caller should fallback to DuckDuckGo).
        """
        searxng_url = self.config.get("searxng_url", "http://localhost:8080")
        searxng_timeout = self.config.get("searxng_timeout", 5)  # Short timeout

        params = {"q": query, "format": "json", "pageno": 1}

        if engines:
            params["engines"] = ",".join(engines)

        try:
            # Use a short connect timeout to fail fast if SearXNG is down
            response = requests.get(
                f"{searxng_url}/search",
                params=params,
                timeout=(2, searxng_timeout),  # (connect_timeout, read_timeout)
            )

            if not response.ok:
                raise Exception(f"HTTP {response.status_code}")

            data = response.json()
            results = []

            for item in data.get("results", [])[:max_results]:
                results.append(
                    {
                        "href": item.get("url", ""),
                        "title": item.get("title", ""),
                        "body": item.get("content", ""),
                        "engines": item.get("engines", []),
                    }
                )

            return results

        except requests.exceptions.ConnectTimeout:
            self.console.print(
                f"[dim yellow]SearXNG connection timeout (is it running at {searxng_url}?)[/dim yellow]"
            )
            return []
        except requests.exceptions.ConnectionError:
            self.console.print(
                f"[dim yellow]SearXNG unavailable at {searxng_url}[/dim yellow]"
            )
            return []
        except requests.exceptions.ReadTimeout:
            self.console.print(f"[dim yellow]SearXNG read timeout[/dim yellow]")
            return []
        except Exception as e:
            self.console.print(f"[dim yellow]SearXNG error: {e}[/dim yellow]")
            return []

    def _ddg_search(self, query: str, max_results: int) -> List[Dict]:
        """
        Fallback to DuckDuckGo (existing implementation).
        """
        try:
            ddgs = DDGS()
            results = list(ddgs.text(query, max_results=max_results))
            return results
        except Exception as e:
            self.console.print(f"[red]DuckDuckGo search failed: {e}[/red]")
            return []
