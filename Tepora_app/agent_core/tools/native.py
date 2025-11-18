from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, List

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from requests.adapters import HTTPAdapter
from urllib.parse import urlparse
from urllib3.util.retry import Retry

from .. import config

logger = logging.getLogger(__name__)


class GoogleCustomSearchInput(BaseModel):
    query: str = Field(description="検索クエリ")


class GoogleCustomSearchTool(BaseTool):
    name: str = "native_google_search"
    description: str = "Google Custom Search APIを使用してWeb検索を実行し、複数の結果を返します。"
    args_schema: type[BaseModel] = GoogleCustomSearchInput
    session: Any = Field(None, exclude=True)

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.timeout = (10, 30)
        return session

    def _perform_search(self, query: str) -> str:
        if not config.GOOGLE_SEARCH_ENABLED:
            return "Error: Google Custom Search is disabled (API keys not configured)."

        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": config.GOOGLE_CUSTOM_SEARCH_API_KEY,
            "cx": config.GOOGLE_CUSTOM_SEARCH_ENGINE_ID,
            "q": query,
            "num": min(config.GOOGLE_CUSTOM_SEARCH_MAX_RESULTS, 10),
            "safe": "active",
        }
        headers = {
            "User-Agent": "AI-Agent/1.0 (Google Custom Search Tool)",
            "Accept": "application/json",
        }

        logger.info("Executing Google Custom Search for query: %s", query)
        start_time = time.time()
        try:
            response = self.session.get(url, params=params, headers=headers, timeout=(10, 30))
            elapsed_time = time.time() - start_time
            logger.info("Search completed in %.2f seconds", elapsed_time)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.Timeout as exc:  # noqa: PERF203
            logger.error("Google Custom Search API timeout: %s", exc)
            return "Error: Search request timed out. Please try again later."
        except requests.exceptions.ConnectionError as exc:
            logger.error("Google Custom Search API connection error: %s", exc)
            return "Error: Connection failed. Please check your internet connection and try again."
        except requests.exceptions.HTTPError as exc:
            status_code = exc.response.status_code if exc.response else "N/A"
            logger.error("Google Custom Search API HTTP error: Status %s", status_code)
            if status_code == 429:
                return "Error: Rate limit exceeded. Please wait a moment and try again."
            if status_code == 403:
                return "Error: API access denied. Please check your API key and permissions."
            return f"Error: HTTP error occurred with status code {status_code}."
        except requests.exceptions.RequestException as exc:
            logger.error("Google Custom Search API request failed: %s", exc)
            return f"Error: Failed to perform search: {exc}"
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse Google API response: %s", exc)
            return "Error: Invalid response from Google API. Please try again."
        except Exception as exc:  # noqa: BLE001
            logger.error("Unexpected error in Google Custom Search: %s", exc)
            return f"Error: Unexpected error occurred: {exc}"

        if "error" in data:
            error_info = data["error"]
            error_message = f"Google API Error: {error_info.get('code', 'Unknown')} - {error_info.get('message', 'Unknown error')}"
            logger.error(error_message)
            return f"Error: {error_message}"

        items = data.get("items")
        if not items:
            return f"No search results found for: {query}"

        results = [
            {
                "title": item.get("title", "No title"),
                "link": item.get("link", "No link"),
                "snippet": item.get("snippet", "No description"),
            }
            for item in items
        ]
        logger.info("Successfully retrieved %d search results", len(results))
        return json.dumps(
            {
                "query": query,
                "total_results": len(results),
                "results": results,
                "search_time": f"{elapsed_time:.2f}s",
            },
            ensure_ascii=False,
            indent=2,
        )

    def _run(self, query: str) -> str:
        return self._perform_search(query)

    async def _arun(self, query: str) -> str:
        return await asyncio.to_thread(self._perform_search, query)

    def __del__(self) -> None:  # noqa: D401
        if hasattr(self, "session"):
            self.session.close()


class WebFetchInput(BaseModel):
    url: str = Field(description="内容を取得したいWebページのURL")


class WebFetchTool(BaseTool):
    name: str = "native_web_fetch"
    description: str = "指定されたURLのWebページにアクセスし、主要なテキストコンテンツを抽出して返します。HTMLタグは除去されます。"
    args_schema: type[BaseModel] = WebFetchInput

    def _create_session(self) -> requests.Session:
        session = requests.Session()
        retry_strategy = Retry(total=2, backoff_factor=1, status_forcelist=[429, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update({"User-Agent": "AI-Agent/1.0 (WebFetchTool)"})
        return session

    def _validate_url(self, url: str) -> str | None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return "Error: URL must include a valid http/https scheme and host."
        return None

    def _fetch_content(self, url: str) -> str:
        validation_error = self._validate_url(url)
        if validation_error:
            logger.warning("WebFetch validation failed for URL '%s': %s", url, validation_error)
            return validation_error

        with self._create_session() as session:
            try:
                logger.info("Fetching content from URL: %s", url)
                response = session.get(url, timeout=(10, 20))
                response.raise_for_status()
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to retrieve URL %s: %s", url, exc, exc_info=True)
                return f"Error: Failed to retrieve content from {url}. Reason: {exc}"

        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            return f"Error: URL is not an HTML page. Content-Type: {content_type}"

        soup = BeautifulSoup(response.text, "html.parser")
        for element in soup(["script", "style", "header", "footer", "nav", "aside"]):
            element.decompose()

        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        cleaned_text = "\n".join(chunk for chunk in chunks if chunk)
        return cleaned_text[:16000]

    def _run(self, url: str) -> str:
        return self._fetch_content(url)

    async def _arun(self, url: str) -> str:
        return await asyncio.to_thread(self._fetch_content, url)


def load_native_tools() -> List[BaseTool]:
    logger.info("Loading native tools...")
    tools: List[BaseTool] = []

    if config.GOOGLE_SEARCH_ENABLED:
        try:
            google_search_tool = GoogleCustomSearchTool()
            google_search_tool.name = "native_google_search"
            google_search_tool.description = (
                "Search the web with Google Custom Search API and return multiple results (list of findings). This is a native tool."
            )
            tools.append(google_search_tool)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load Google Custom Search tool: %s", exc, exc_info=True)

    if tools:
        tools.append(WebFetchTool())
    else:
        # Even if search is disabled, web fetch can still be useful.
        tools.append(WebFetchTool())

    for tool in tools:
        logger.info("Native tool available: %s", tool.name)

    return tools
