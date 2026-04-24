from plugins.BaseTool import BaseTool, ToolResult

import re
import urllib.error

_URL_RE = re.compile(r"^(https?://|www\.)\S+$", re.IGNORECASE)


class WebSearch(BaseTool):
    name = "web_search"
    description = (
        "Search the public web for information that is not already available in the "
        "local file system, especially current facts, external references, or verification. "
        "Uses Brave search by default and can use Brave Answers when mode='answers' or mode='auto'. "
        "If 'query' is a URL (http://, https://, or www.), the page is fetched and its cleaned text is returned."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for on the public web.",
            },
            "mode": {
                "type": "string",
                "description": "Search mode. 'auto' uses Answers for question-like queries and Search otherwise.",
                "enum": ["auto", "search", "answers"],
                "default": "auto",
            },
            "count": {
                "type": "integer",
                "description": "Max results for search mode. Default 5, max 20.",
                "default": 5,
            },
            "country": {
                "type": "string",
                "description": "Optional 2-letter country code such as US, GB, or DE.",
            },
            "search_lang": {
                "type": "string",
                "description": "Optional language code such as en, de, or fr.",
                "default": "en",
            },
            "safesearch": {
                "type": "string",
                "description": "Safe search level for search mode.",
                "enum": ["off", "moderate", "strict"],
                "default": "moderate",
            },
            "freshness": {
                "type": "string",
                "description": "Optional freshness filter for search mode such as pd, pw, pm, or py.",
            },
        },
        "required": ["query"],
    }
    requires_services = ["web_search_provider"]
    agent_enabled = True
    max_calls = 5

    def _looks_question_like(self, query):
        q = query.lower().strip()
        starters = (
            "what", "why", "how", "when", "where", "who", "which", "compare", "explain", "summarize"
        )
        return q.endswith("?") or q.startswith(starters) or len(query.split()) >= 8

    def _format_search_result(self, data, engine="brave", prefix=""):
        results = data.get("results", [])
        query = data.get("query", "")
        label = "DuckDuckGo" if engine == "duckduckgo" else "search"

        lines = [f"Found {len(results)} {label} result(s) for '{query}':"]
        for i, item in enumerate(results, start=1):
            lines.append(f"{i}. {item.get('title') or '(no title)'} — {item.get('url') or ''}")
            if item.get("description"):
                lines.append(f"   {item['description']}")

        summary = "\n".join(lines) if results else f"No {label} results found for '{query}'."
        if prefix:
            summary = prefix + "\n\n" + summary

        return ToolResult(
            success=True,
            data={"mode": "search", **({"engine": engine} if engine == "duckduckgo" else {}), **data},
            llm_summary=summary,
        )

    def _format_answers_result(self, data):
        query = data.get("query", "")
        answer = data.get("answer", "")
        sources = data.get("sources", [])

        lines = [f"Brave answer for '{query}':", answer]
        if sources:
            lines.append("Sources:")
            for i, c in enumerate(sources, start=1):
                lines.append(f"{i}. {(c.get('title') or '(untitled source)')} — {c.get('url')}")

        return ToolResult(
            success=True,
            data={"mode": "answers", **data},
            llm_summary="\n".join(lines),
        )

    def run(self, context, **kwargs):
        query = (kwargs.get("query") or "").strip()
        if not query:
            return ToolResult.failed("Missing required parameter: query")

        svc = context.services.get("web_search_provider")
        if not svc or not svc.loaded:
            return ToolResult.failed("web_search_provider service is not available.")

        if _URL_RE.match(query):
            url = query if query.lower().startswith(("http://", "https://")) else "https://" + query
            try:
                data = svc.fetch_url(url)
            except urllib.error.HTTPError as e:
                return ToolResult.failed(f"Fetch HTTP error {e.code} for {url}")
            except urllib.error.URLError as e:
                return ToolResult.failed(f"Fetch connection error for {url}: {e}")
            except Exception as e:
                return ToolResult.failed(f"Fetch failed for {url}: {e}")

            header = f"Fetched {data['final_url']} (status {data['status']}, {data['content_type'] or 'unknown type'})"
            if data["title"]:
                header += f"\nTitle: {data['title']}"
            summary = header + "\n\n" + data["text"]
            if data["truncated"]:
                summary += "\n\n[content truncated]"
            return ToolResult(success=True, data={"mode": "fetch", **data}, llm_summary=summary)

        try:
            count = int(kwargs.get("count", 5))
        except Exception:
            count = 5
        count = max(1, min(count, 20))

        mode = (kwargs.get("mode") or "auto").strip().lower()
        if mode not in {"auto", "search", "answers"}:
            mode = "auto"

        country = (kwargs.get("country") or "").strip()
        search_lang = (kwargs.get("search_lang") or "en").strip() or "en"
        safesearch = (kwargs.get("safesearch") or "moderate").strip().lower() or "moderate"
        if safesearch not in {"off", "moderate", "strict"}:
            safesearch = "moderate"
        freshness = (kwargs.get("freshness") or "").strip()

        chosen_mode = mode
        if mode == "auto":
            chosen_mode = "answers" if self._looks_question_like(query) else "search"

        if chosen_mode == "answers" and not svc.has_answers_key():
            chosen_mode = "search" if mode == "auto" else None
            if chosen_mode is None:
                return ToolResult.failed(
                    "Brave Answers API key not configured. Set brave_answers_api_key in service settings or environment."
                )

        if chosen_mode == "search" and not svc.has_search_key():
            try:
                data = svc.duckduckgo_search(query, count, search_lang)
                return self._format_search_result(data, engine="duckduckgo",
                                                  prefix="No Brave API key configured — using DuckDuckGo fallback.")
            except Exception as e:
                return ToolResult.failed(f"DuckDuckGo fallback failed: {e}")

        try:
            if chosen_mode == "answers":
                try:
                    data = svc.answers(query, country, search_lang)
                    return self._format_answers_result(data)
                except urllib.error.HTTPError as e:
                    body = svc._read_http_error_body(e)
                    if mode == "auto":
                        if svc.has_search_key():
                            data = svc.search(query, count, country, search_lang, safesearch, freshness)
                            return self._format_search_result(data,
                                                              prefix="Brave Answers was unavailable, so I used Brave Search instead.")
                        try:
                            data = svc.duckduckgo_search(query, count, search_lang)
                            return self._format_search_result(data, engine="duckduckgo",
                                                              prefix="Brave APIs unavailable — using DuckDuckGo fallback.")
                        except Exception:
                            pass
                    return ToolResult.failed(f"Brave Answers HTTP error {e.code}: {body[:500]}")

            data = svc.search(query, count, country, search_lang, safesearch, freshness)
            return self._format_search_result(data)
        except urllib.error.HTTPError as e:
            body = svc._read_http_error_body(e)
            return ToolResult.failed(f"Brave API HTTP error {e.code}: {body[:500]}")
        except urllib.error.URLError as e:
            return ToolResult.failed(f"Brave API connection error: {e}")
        except Exception as e:
            return ToolResult.failed(f"Web search failed: {e}")
