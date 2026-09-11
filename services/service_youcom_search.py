"""You.com search service providing web search, smart search, and content extraction.

This service interfaces with You.com's API endpoints to provide real-time web search
capabilities. It supports both web search and smart search modes, with automatic
content extraction for URLs. The API key is handled securely through the SDK's
secret system.
"""

dependencies_files = []
dependencies_pip = []

import json
import re
import urllib.parse
from typing import Dict, Any, Optional

from guest.bases import BaseService

_USER_AGENT = "SecondBrain-YoucomSearch/1.0"


class YoucomSearchProvider(BaseService):
    """You.com search provider with web search, smart search, and content extraction."""

    name = "youcom_search_provider"
    description = "Search the web and extract content using You.com's premium API."
    shared = True
    requests = ["net.http", "config.read"]
    exports = [
        "web_search",
        "smart_search", 
        "extract_content",
        "has_api_key",
    ]
    config_settings = [
        ("You.com API Key", "secret_ydc_api_key",
         "API key for You.com search services (YDC_API_KEY). "
         "Stored as a secret: plugins receive an opaque handle and the kernel "
         "substitutes the real key into outbound requests.",
         "",
         {"type": "text"}),
    ]

    # You.com API endpoints
    WEB_SEARCH_URL = "https://api.you.com/smart/search"
    SMART_SEARCH_URL = "https://api.you.com/smart/search"  
    CONTENT_URL = "https://api.you.com/smart/search"

    def start(self, sdk):
        """Service startup - no persistent connections needed."""
        return True

    def stop(self, sdk):
        """Service shutdown - no cleanup needed."""
        return None

    # ── API key management ──────────────────────────────────────────

    def _get_api_key(self, sdk) -> str:
        """Get the API key handle from config."""
        return str(sdk.config.read("secret_ydc_api_key") or "").strip()

    def has_api_key(self, sdk) -> bool:
        """Check if API key is configured."""
        return bool(self._get_api_key(sdk))

    # ── HTTP utilities ──────────────────────────────────────────────

    def _clean_text(self, value: str, limit: Optional[int] = None) -> str:
        """Clean and optionally truncate text."""
        if not value:
            return ""
        text = value.replace("\\n", " ").replace("\\r", " ").strip()
        text = " ".join(text.split())  # Normalize whitespace
        if limit and len(text) > limit:
            text = text[:max(0, limit - 3)] + "..."
        return text

    def _make_headers(self, api_key: str) -> Dict[str, str]:
        """Create headers for You.com API requests."""
        return {
            "X-API-Key": api_key,
            "Content-Type": "application/json",
            "User-Agent": _USER_AGENT,
        }

    def _http_request(self, sdk, url: str, params: Dict[str, Any]) -> tuple[Optional[dict], Optional[dict]]:
        """Make HTTP request to You.com API.
        
        Returns (response_data, error_dict). Exactly one will be set.
        Errors are returned as dicts to survive service.call boundary.
        """
        api_key = self._get_api_key(sdk)
        if not api_key:
            return None, {"http_status": 401, "error": "No API key configured"}

        try:
            # You.com expects POST requests with JSON body
            body = json.dumps(params)
            response = sdk.net.http(
                url,
                method="POST", 
                headers=self._make_headers(api_key),
                body=body
            )
            
            status = int(response.get("status", 0))
            response_text = response.get("body", "")
            
            if status >= 400:
                return None, {
                    "http_status": status,
                    "error": self._clean_text(response_text, 500)
                }
                
            if not response_text:
                return None, {"http_status": status, "error": "Empty response"}
                
            try:
                data = json.loads(response_text)
                return data, None
            except json.JSONDecodeError as e:
                return None, {
                    "http_status": status,
                    "error": f"Invalid JSON response: {e}"
                }
                
        except Exception as e:
            return None, {"error": f"Request failed: {e}"}

    # ── search implementations ──────────────────────────────────────

    def web_search(self, sdk, query: str, count: int = 6, country: str = "", 
                   safesearch: str = "moderate") -> Dict[str, Any]:
        """Execute web search via You.com API.
        
        Returns dict with query, count, hits, and potentially error info.
        """
        params = {
            "query": query,
            "type": "search",  # Regular web search
            "count": count,
            "safesearch": safesearch,
        }
        
        if country:
            params["country"] = country.upper()

        data, error = self._http_request(sdk, self.WEB_SEARCH_URL, params)
        if error:
            return {"query": query, "count": 0, "hits": [], **error}

        # Extract search results from You.com response
        hits = []
        if isinstance(data, dict):
            results = data.get("hits", [])
            for item in results[:count]:
                if not isinstance(item, dict):
                    continue
                    
                hit = {
                    "title": self._clean_text(item.get("title", ""), 200),
                    "url": item.get("url", ""),
                    "description": self._clean_text(item.get("description", ""), 300),
                }
                
                # Handle snippets if present
                snippets = item.get("snippets")
                if isinstance(snippets, list):
                    hit["snippets"] = [self._clean_text(s, 200) for s in snippets[:3]]
                
                hits.append(hit)

        return {
            "query": query,
            "count": len(hits),
            "hits": hits,
            "raw_response": data
        }

    def smart_search(self, sdk, query: str, country: str = "", 
                     safesearch: str = "moderate") -> Dict[str, Any]:
        """Execute smart search (chat/answer mode) via You.com API.
        
        Returns dict with query, answer, sources, and potentially error info.
        """
        params = {
            "query": query,
            "type": "chat",  # Smart/chat search mode  
            "safesearch": safesearch,
        }
        
        if country:
            params["country"] = country.upper()

        data, error = self._http_request(sdk, self.SMART_SEARCH_URL, params) 
        if error:
            return {"query": query, "answer": "", "sources": [], **error}

        # Extract answer and sources from You.com response
        answer = ""
        sources = []
        
        if isinstance(data, dict):
            # You.com chat responses typically have an answer field
            answer = self._clean_text(data.get("answer", ""), 4000)
            
            # Extract sources/citations
            raw_sources = data.get("sources", []) or data.get("citations", [])
            for source in raw_sources[:8]:
                if isinstance(source, dict):
                    source_item = {
                        "title": self._clean_text(source.get("title") or source.get("name", ""), 200),
                        "url": source.get("url", ""),
                    }
                    if source_item["url"]:
                        sources.append(source_item)

        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "raw_response": data
        }

    def extract_content(self, sdk, url: str, max_chars: int = 15000) -> Dict[str, Any]:
        """Extract content from a specific URL.
        
        Returns dict with url, title, content, status, and potentially error info.
        """
        params = {
            "query": url,
            "type": "news",  # Content extraction mode
        }

        data, error = self._http_request(sdk, self.CONTENT_URL, params)
        if error:
            return {"url": url, "title": "", "content": "", "status": 0, **error}

        # Extract content from You.com response
        title = ""
        content = ""
        status = 200
        
        if isinstance(data, dict):
            # You.com content responses vary, try multiple fields
            title = self._clean_text(data.get("title", ""), 300)
            
            # Try different content fields
            content_text = (
                data.get("content") or 
                data.get("text") or
                data.get("body") or
                ""
            )
            content = self._clean_text(content_text, max_chars)

        truncated = len(content) >= max_chars
        if truncated:
            content += "\\n\\n[content truncated]"

        return {
            "url": url,
            "title": title,
            "content": content,
            "status": status,
            "truncated": truncated,
            "raw_response": data
        }