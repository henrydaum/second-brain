from Stage_3.BaseTool import BaseTool, ToolResult

import os
import json
import gzip
import html
import re
import urllib.parse
import urllib.request
import urllib.error


class WebSearch(BaseTool):
    name = "web_search"
    description = (
        "Search the public web with Brave. Simple by default: returns readable, ranked results. "
        "Can also call Brave Answers for grounded answer generation when mode='answers' or mode='auto'."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for.",
            },
            "mode": {
                "type": "string",
                "description": "Search mode. 'auto' uses Answers for question-like prompts and Search otherwise.",
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
    requires_services = []
    agent_enabled = True
    max_calls = 5

    # Add your keys here, or set BRAVE_SEARCH_API_KEY / BRAVE_ANSWERS_API_KEY in the environment.
    BRAVE_SEARCH_API_KEY = "BSASQYgVl16NvB9KOa9UDr0d7WonZSw"
    BRAVE_ANSWERS_API_KEY = "BSAour0nJF2KDA2YmpvVKH3gLlQv_JR"

    SEARCH_API_URL = "https://api.search.brave.com/res/v1/web/search"
    ANSWERS_API_URL = "https://api.search.brave.com/res/v1/chat/completions"

    def _get_search_key(self):
        return (self.BRAVE_SEARCH_API_KEY or os.getenv("BRAVE_SEARCH_API_KEY") or os.getenv("BRAVE_API_KEY") or "").strip()

    def _get_answers_key(self):
        return (self.BRAVE_ANSWERS_API_KEY or os.getenv("BRAVE_ANSWERS_API_KEY") or "").strip()

    def _clean_text(self, value, limit=None):
        text = (value or "").replace("\n", " ").replace("\r", " ").strip()
        text = " ".join(text.split())
        if limit and len(text) > limit:
            text = text[: max(0, limit - 3)] + "..."
        return text

    def _looks_question_like(self, query):
        q = query.lower().strip()
        starters = (
            "what", "why", "how", "when", "where", "who", "which", "compare", "explain", "summarize"
        )
        return q.endswith("?") or q.startswith(starters) or len(query.split()) >= 8

    def _headers(self, api_key, json_body=False):
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": api_key,
            "User-Agent": "SecondBrain-WebSearch/3.0",
        }
        if json_body:
            headers["Content-Type"] = "application/json"
        return headers

    def _decode_raw(self, raw, headers):
        encoding = (headers.get("Content-Encoding") or "").lower().strip() if headers else ""
        if encoding == "gzip":
            raw = gzip.decompress(raw)
        return raw.decode("utf-8", errors="replace")

    def _read_json_response(self, request, timeout=30):
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            text = self._decode_raw(raw, response.headers)
            return json.loads(text)

    def _read_http_error_body(self, error):
        try:
            raw = error.read()
            headers = getattr(error, "headers", {})
            return self._decode_raw(raw, headers)
        except Exception:
            return ""

    # ── DuckDuckGo fallback (no API key needed) ──────────────────────

    DDG_URL = "https://html.duckduckgo.com/html/"

    def _duckduckgo_search(self, query, count, search_lang):
        """Scrape DuckDuckGo HTML-only endpoint as a keyless fallback."""
        params = urllib.parse.urlencode({"q": query, "kl": search_lang or "en"})
        payload = params.encode("utf-8")
        request = urllib.request.Request(
            self.DDG_URL, data=payload, method="POST",
            headers={
                "User-Agent": "SecondBrain-WebSearch/3.0",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            raw = response.read()
            encoding = (response.headers.get("Content-Encoding") or "").lower()
            if encoding == "gzip":
                raw = gzip.decompress(raw)
            page = raw.decode("utf-8", errors="replace")

        # Parse result blocks: <a class="result__a" href="...">title</a>
        # and <a class="result__snippet" ...>description</a>
        results = []
        for m in re.finditer(
            r'<a\s+rel="nofollow"\s+class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
            page, re.DOTALL,
        ):
            raw_url, raw_title = m.group(1), m.group(2)
            # DDG wraps URLs in a redirect — extract the real one
            url_match = re.search(r'uddg=([^&]+)', raw_url)
            url = urllib.parse.unquote(url_match.group(1)) if url_match else raw_url
            title = self._clean_text(re.sub(r"<[^>]+>", "", raw_title), 200)
            results.append({"title": title, "url": url, "description": ""})

        # Grab snippets
        snippets = re.findall(
            r'<a\s+class="result__snippet"[^>]*>(.*?)</a>', page, re.DOTALL,
        )
        for i, snippet in enumerate(snippets):
            if i < len(results):
                results[i]["description"] = self._clean_text(
                    html.unescape(re.sub(r"<[^>]+>", "", snippet)), 300,
                )

        results = results[:count]

        lines = [f"Found {len(results)} DuckDuckGo result(s) for '{query}':"]
        for i, item in enumerate(results, start=1):
            lines.append(f"{i}. {item.get('title') or '(no title)'} — {item.get('url') or ''}")
            if item.get("description"):
                lines.append(f"   {item['description']}")

        return ToolResult(
            success=True,
            data={
                "mode": "search",
                "engine": "duckduckgo",
                "query": query,
                "count": len(results),
                "results": results,
            },
            llm_summary=("\n".join(lines) if results else f"No DuckDuckGo results found for '{query}'."),
        )

    # ── Brave Search ────────────────────────────────────────────────

    def _search(self, api_key, query, count, country, search_lang, safesearch, freshness):
        params = {
            "q": query,
            "count": count,
            "search_lang": search_lang,
            "safesearch": safesearch,
        }
        if country:
            params["country"] = country
        if freshness:
            params["freshness"] = freshness

        url = f"{self.SEARCH_API_URL}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(url, headers=self._headers(api_key), method="GET")
        data = self._read_json_response(request)

        web = data.get("web", {}) if isinstance(data, dict) else {}
        results = web.get("results", []) if isinstance(web, dict) else []

        normalized = []
        for item in results[:count]:
            if not isinstance(item, dict):
                continue
            normalized.append({
                "title": self._clean_text(item.get("title", ""), 200),
                "url": item.get("url", ""),
                "display_url": item.get("meta_url", {}).get("display_url", "") if isinstance(item.get("meta_url"), dict) else "",
                "description": self._clean_text(item.get("description", ""), 300),
                "age": item.get("age", ""),
                "language": item.get("language", ""),
            })

        lines = [f"Found {len(normalized)} search result(s) for '{query}':"]
        for i, item in enumerate(normalized, start=1):
            lines.append(f"{i}. {item.get('title') or '(no title)'} — {item.get('url') or ''}")
            if item.get("description"):
                lines.append(f"   {item['description']}")

        return ToolResult(
            success=True,
            data={
                "mode": "search",
                "query": query,
                "count": len(normalized),
                "results": normalized,
                "raw": data,
            },
            llm_summary=("\n".join(lines) if normalized else f"No search results found for '{query}'."),
        )

    def _answers(self, api_key, query, country, search_lang):
        body = {
            "model": "brave",
            "stream": False,
            "messages": [
                {"role": "user", "content": query}
            ],
        }
        if country:
            body["country"] = country.lower()
        if search_lang:
            body["language"] = search_lang.lower()

        payload = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            self.ANSWERS_API_URL,
            headers=self._headers(api_key, json_body=True),
            data=payload,
            method="POST",
        )
        data = self._read_json_response(request)

        answer_text = ""
        choices = data.get("choices") if isinstance(data, dict) else None
        if isinstance(choices, list) and choices:
            first = choices[0] if isinstance(choices[0], dict) else {}
            message = first.get("message", {}) if isinstance(first, dict) else {}
            content = message.get("content") if isinstance(message, dict) else ""
            if isinstance(content, str):
                answer_text = self._clean_text(content, 6000)
            elif isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text") or item.get("content") or ""
                        if isinstance(text, str) and text.strip():
                            parts.append(text)
                answer_text = self._clean_text("\n\n".join(parts), 6000)

        citations = []

        def harvest(obj):
            if isinstance(obj, dict):
                maybe_url = obj.get("url")
                maybe_title = obj.get("title") or obj.get("name") or obj.get("source") or ""
                if isinstance(maybe_url, str) and maybe_url.startswith("http"):
                    citations.append({
                        "title": self._clean_text(maybe_title, 200),
                        "url": maybe_url,
                    })
                for v in obj.values():
                    harvest(v)
            elif isinstance(obj, list):
                for v in obj:
                    harvest(v)

        harvest(data)

        deduped_citations = []
        seen = set()
        for c in citations:
            url = c.get("url") or ""
            if url and url not in seen:
                seen.add(url)
                deduped_citations.append(c)
            if len(deduped_citations) >= 8:
                break

        if not answer_text:
            answer_text = self._clean_text(json.dumps(data, ensure_ascii=False), 2500)

        lines = [f"Brave answer for '{query}':", answer_text]
        if deduped_citations:
            lines.append("Sources:")
            for i, c in enumerate(deduped_citations, start=1):
                lines.append(f"{i}. {(c.get('title') or '(untitled source)')} — {c.get('url')}")

        return ToolResult(
            success=True,
            data={
                "mode": "answers",
                "query": query,
                "answer": answer_text,
                "sources": deduped_citations,
                "raw": data,
            },
            llm_summary="\n".join(lines),
        )

    def run(self, context, **kwargs):
        query = (kwargs.get("query") or "").strip()
        if not query:
            return ToolResult.failed("Missing required parameter: query")

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

        if chosen_mode == "answers":
            answers_key = self._get_answers_key()
            if not answers_key:
                if mode == "auto":
                    chosen_mode = "search"
                else:
                    return ToolResult.failed(
                        "Brave Answers API key not configured. Set BRAVE_ANSWERS_API_KEY in the plugin or environment."
                    )

        if chosen_mode == "search":
            search_key = self._get_search_key()
            if not search_key:
                # No Brave key — fall back to DuckDuckGo
                try:
                    result = self._duckduckgo_search(query, count, search_lang)
                    result.llm_summary = "No Brave API key configured — using DuckDuckGo fallback.\n\n" + result.llm_summary
                    return result
                except Exception as e:
                    return ToolResult.failed(f"DuckDuckGo fallback failed: {e}")

        try:
            if chosen_mode == "answers":
                try:
                    return self._answers(self._get_answers_key(), query, country, search_lang)
                except urllib.error.HTTPError as e:
                    body = self._read_http_error_body(e)
                    if mode == "auto":
                        search_key = self._get_search_key()
                        if search_key:
                            fallback = self._search(search_key, query, count, country, search_lang, safesearch, freshness)
                            fallback.llm_summary = "Brave Answers was unavailable, so I used Brave Search instead.\n\n" + fallback.llm_summary
                            return fallback
                        # No Brave Search key either — try DuckDuckGo
                        try:
                            fallback = self._duckduckgo_search(query, count, search_lang)
                            fallback.llm_summary = "Brave APIs unavailable — using DuckDuckGo fallback.\n\n" + fallback.llm_summary
                            return fallback
                        except Exception:
                            pass
                    return ToolResult.failed(f"Brave Answers HTTP error {e.code}: {body[:500]}")

            return self._search(self._get_search_key(), query, count, country, search_lang, safesearch, freshness)
        except urllib.error.HTTPError as e:
            body = self._read_http_error_body(e)
            return ToolResult.failed(f"Brave API HTTP error {e.code}: {body[:500]}")
        except urllib.error.URLError as e:
            return ToolResult.failed(f"Brave API connection error: {e}")
        except Exception as e:
            return ToolResult.failed(f"Web search failed: {e}")
