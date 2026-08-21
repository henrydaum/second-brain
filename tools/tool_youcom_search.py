"""Search the web using You.com's API with intelligent mode selection.

You.com provides premium web search with real-time results, smart search,
and content extraction. This tool automatically selects the best mode
based on query type and provides rich, structured results with citations.
"""

dependencies_files = ['services/service_youcom_search.py']
dependencies_pip = []

import re

from guest.bases import BaseTool

_URL_RE = re.compile(r"^(https?://|www\.)\S+$", re.IGNORECASE)
_PROVIDER = "youcom_search_provider"


class YoucomSearch(BaseTool):
    """You.com web search with intelligent mode selection."""

    name = "youcom_search"
    description = (
        "Search the public web using You.com's premium search API. "
        "Automatically selects between web search, smart search, and content extraction "
        "based on query type. Provides real-time results with citations and source URLs. "
        "If 'query' is a URL, the content is extracted and returned as cleaned text."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for, or a URL to extract content from.",
            },
            "mode": {
                "type": "string", 
                "description": "Search mode. 'auto' intelligently selects web/smart search based on query type.",
                "enum": ["auto", "web", "smart"],
                "default": "auto",
            },
            "count": {
                "type": "integer",
                "description": "Max results for web search mode. Default 6, max 20.",
                "default": 6,
            },
            "country": {
                "type": "string",
                "description": "Optional 2-letter country code for localized results (e.g., US, GB, DE).",
            },
            "safesearch": {
                "type": "string",
                "description": "Content filtering level.",
                "enum": ["strict", "moderate", "off"],
                "default": "moderate",
            },
            "narration": {
                "type": "string",
                "description": (
                    "A few words on what you are looking for and why, shown to "
                    "the user beside the call. E.g. 'checking latest AI research updates'."
                ),
            },
        },
        "required": ["query"],
    }
    requires_services = ["youcom_search_provider"]
    requests = ["service.call"]

    # ── provider interaction ────────────────────────────────────────

    def _call(self, sdk, method, **kwargs):
        """Call a provider method, returning dict or empty dict on failure.
        
        HTTP errors are returned in the dict as 'http_status' rather than raised,
        since that's what survives service.call and allows fallback logic.
        """
        return sdk.services.call(_PROVIDER, method, **kwargs) or {}

    def _refused(self, data) -> str:
        """Extract error reason from API response, or empty string if none."""
        status = (data or {}).get("http_status")
        if not status:
            return ""
        detail = (data or {}).get("error") or ""
        return f"HTTP {status}" + (f": {detail[:500]}" if detail else "")

    # ── query analysis ─────────────────────────────────────────────

    def _looks_question_like(self, query):
        """Determine if query reads like a question vs keywords."""
        q = query.lower().strip()
        question_starters = (
            "what", "why", "how", "when", "where", "who", "which", "compare",
            "explain", "summarize", "define", "describe", "analyze"
        )
        return (
            q.endswith("?") or 
            q.startswith(question_starters) or 
            len(query.split()) >= 8 or
            " vs " in q or " versus " in q
        )

    # ── result formatting ──────────────────────────────────────────

    def _format_web_results(self, sdk, data, prefix=""):
        """Format web search results as numbered list with descriptions."""
        results = data.get("hits", [])
        query = data.get("query", "")

        if not results:
            summary = f"No web results found for '{query}'."
            if prefix:
                summary = prefix + "\n\n" + summary
            return sdk.ok({"mode": "web", **data}, llm_summary=summary)

        lines = [f"Found {len(results)} web result(s) for '{query}':"]
        for index, item in enumerate(results, start=1):
            title = item.get("title") or "(no title)"
            url = item.get("url") or ""
            lines.append(f"{index}. {title} — {url}")
            
            description = item.get("description") or item.get("snippets", [])
            if isinstance(description, list) and description:
                # Join multiple snippets
                desc_text = " ".join(description[:2])  # Use first 2 snippets
            elif isinstance(description, str):
                desc_text = description
            else:
                desc_text = ""
                
            if desc_text:
                lines.append(f"   {desc_text}")

        summary = "\n".join(lines)
        if prefix:
            summary = prefix + "\n\n" + summary

        return sdk.ok({"mode": "web", **data}, llm_summary=summary)

    def _format_smart_results(self, sdk, data):
        """Format smart search results with answer and sources."""
        query = data.get("query", "")
        answer = data.get("answer", "")
        sources = data.get("sources", [])

        if not answer:
            # Fall back to web results if no smart answer
            return self._format_web_results(sdk, data, prefix="Smart search unavailable, showing web results")

        lines = [f"You.com smart answer for '{query}':", "", answer]
        
        if sources:
            lines.append("")
            lines.append("Sources:")
            for index, source in enumerate(sources, start=1):
                title = source.get("title") or source.get("name") or "(untitled)"
                url = source.get("url") or ""
                if url:
                    lines.append(f"{index}. {title} — {url}")

        return sdk.ok({"mode": "smart", **data}, llm_summary="\n".join(lines))

    def _format_content_results(self, sdk, data, url):
        """Format URL content extraction results."""
        title = data.get("title", "")
        content = data.get("content", "")
        status = data.get("status", 200)
        
        header = f"Extracted content from {url} (status {status})"
        if title:
            header += f"\nTitle: {title}"
            
        if not content:
            return sdk.fail(f"No readable content found at {url}")
            
        summary = header + "\n\n" + content
        if data.get("truncated"):
            summary += "\n\n[content truncated]"
            
        return sdk.ok({"mode": "content", **data}, llm_summary=summary)

    # ── main tool implementation ───────────────────────────────────

    def run(self, sdk, **kwargs):
        """Execute You.com search with intelligent mode selection."""
        query = (kwargs.get("query") or "").strip()
        if not query:
            return sdk.fail("Missing required parameter: query")

        # Handle URL content extraction
        if _URL_RE.match(query):
            return self._extract_content(sdk, query)

        # Parameter processing
        mode = (kwargs.get("mode") or "auto").strip().lower()
        if mode not in {"auto", "web", "smart"}:
            mode = "auto"

        try:
            count = int(kwargs.get("count", 6))
        except (TypeError, ValueError):
            count = 6
        count = max(1, min(count, 20))

        country = (kwargs.get("country") or "").strip().upper()
        safesearch = (kwargs.get("safesearch") or "moderate").strip().lower()
        if safesearch not in {"strict", "moderate", "off"}:
            safesearch = "moderate"

        # Check API key availability
        has_api_key = bool(self._call(sdk, "has_api_key"))
        if not has_api_key:
            return sdk.fail(
                "You.com API key not configured. Set YDC_API_KEY in service settings "
                "or configure oauth authentication."
            )

        # Intelligent mode selection
        chosen_mode = mode
        if mode == "auto":
            chosen_mode = "smart" if self._looks_question_like(query) else "web"

        # Execute search
        if chosen_mode == "smart":
            return self._smart_search(sdk, query, country, safesearch, mode != "auto")
        else:
            return self._web_search(sdk, query, count, country, safesearch)

    def _web_search(self, sdk, query, count, country, safesearch):
        """Execute web search and format results."""
        params = {
            "query": query,
            "count": count,
            "safesearch": safesearch,
        }
        if country:
            params["country"] = country

        data = self._call(sdk, "web_search", **params)
        if (why := self._refused(data)):
            return sdk.fail(f"You.com web search failed: {why}")

        return self._format_web_results(sdk, data)

    def _smart_search(self, sdk, query, country, safesearch, explicit_mode):
        """Execute smart search with fallback to web search."""
        params = {
            "query": query,
            "safesearch": safesearch,
        }
        if country:
            params["country"] = country

        data = self._call(sdk, "smart_search", **params)
        if (why := self._refused(data)):
            if explicit_mode:
                return sdk.fail(f"You.com smart search failed: {why}")
            # Auto mode: fall back to web search
            return self._web_search(sdk, query, 6, country, safesearch)

        return self._format_smart_results(sdk, data)

    def _extract_content(self, sdk, query):
        """Extract content from URL."""
        # Normalize URL
        url = query if query.lower().startswith(("http://", "https://")) else "https://" + query
        
        data = self._call(sdk, "extract_content", url=url)
        if (why := self._refused(data)):
            return sdk.fail(f"You.com content extraction failed: {why}")

        return self._format_content_results(sdk, data, url)