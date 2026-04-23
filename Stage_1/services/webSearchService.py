import os
import json
import gzip
import html
import re
import urllib.parse
import urllib.request
import urllib.error
import logging

from Stage_1.BaseService import BaseService

logger = logging.getLogger("WebSearchService")


class WebSearchProvider(BaseService):
    """Provides web search via Brave Search, Brave Answers, and DuckDuckGo fallback."""

    model_name = "web_search_provider"
    shared = True

    config_settings = [
        ("Brave Search API Key", "brave_search_api_key",
         "API key for Brave Web Search.",
         "",
         {"type": "text"}),

        ("Brave Answers API Key", "brave_answers_api_key",
         "API key for Brave Answers (grounded answer generation).",
         "",
         {"type": "text"}),
    ]

    SEARCH_API_URL = "https://api.search.brave.com/res/v1/web/search"
    ANSWERS_API_URL = "https://api.search.brave.com/res/v1/chat/completions"
    DDG_URL = "https://html.duckduckgo.com/html/"

    def __init__(self, search_key="", answers_key=""):
        super().__init__()
        self._search_key = search_key
        self._answers_key = answers_key

    def _load(self) -> bool:
        self._loaded = True
        return True

    def unload(self):
        self._loaded = False

    # ── Key access ──────────────────────────────────────────────────

    def get_search_key(self):
        return (self._search_key or os.getenv("BRAVE_SEARCH_API_KEY") or os.getenv("BRAVE_API_KEY") or "").strip()

    def get_answers_key(self):
        return (self._answers_key or os.getenv("BRAVE_ANSWERS_API_KEY") or "").strip()

    def has_search_key(self):
        return bool(self.get_search_key())

    def has_answers_key(self):
        return bool(self.get_answers_key())

    # ── HTTP helpers ────────────────────────────────────────────────

    def _clean_text(self, value, limit=None):
        text = (value or "").replace("\n", " ").replace("\r", " ").strip()
        text = " ".join(text.split())
        if limit and len(text) > limit:
            text = text[: max(0, limit - 3)] + "..."
        return text

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

    # ── Public search methods ───────────────────────────────────────
    # These return plain dicts. The tool layer wraps them in ToolResult.

    def search(self, query, count=5, country="", search_lang="en", safesearch="moderate", freshness=""):
        """Brave Web Search. Returns dict with keys: results, query, count, raw."""
        api_key = self.get_search_key()
        if not api_key:
            raise ValueError("No Brave Search API key configured.")

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

        return {"query": query, "count": len(normalized), "results": normalized, "raw": data}

    def answers(self, query, country="", search_lang="en"):
        """Brave Answers. Returns dict with keys: answer, sources, query, raw."""
        api_key = self.get_answers_key()
        if not api_key:
            raise ValueError("No Brave Answers API key configured.")

        body = {
            "model": "brave",
            "stream": False,
            "messages": [{"role": "user", "content": query}],
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

        deduped = []
        seen = set()
        for c in citations:
            url = c.get("url") or ""
            if url and url not in seen:
                seen.add(url)
                deduped.append(c)
            if len(deduped) >= 8:
                break

        if not answer_text:
            answer_text = self._clean_text(json.dumps(data, ensure_ascii=False), 2500)

        return {"query": query, "answer": answer_text, "sources": deduped, "raw": data}

    def fetch_url(self, url, max_chars=20000, timeout=20):
        """Fetch a URL and return cleaned text. Returns dict: url, final_url, status, content_type, title, text, truncated."""
        request = urllib.request.Request(
            url, method="GET",
            headers={
                "User-Agent": "SecondBrain-WebSearch/3.0",
                "Accept": "text/html,application/xhtml+xml,application/xml,text/plain,application/json;q=0.9,*/*;q=0.8",
                "Accept-Encoding": "gzip",
            },
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            text = self._decode_raw(raw, response.headers)
            final_url = response.geturl()
            status = response.status
            content_type = (response.headers.get("Content-Type") or "").lower()

        title = ""
        if "html" in content_type or "<html" in text[:2000].lower():
            m = re.search(r"<title[^>]*>(.*?)</title>", text, re.DOTALL | re.IGNORECASE)
            if m:
                title = self._clean_text(html.unescape(re.sub(r"<[^>]+>", "", m.group(1))), 300)
            body = re.sub(r"(?is)<(script|style|noscript|svg|head)[^>]*>.*?</\1>", " ", text)
            body = re.sub(r"(?is)<[^>]+>", " ", body)
            body = html.unescape(body)
            body = re.sub(r"[ \t]+", " ", body)
            body = re.sub(r"\n\s*\n+", "\n\n", body).strip()
        else:
            body = text

        truncated = len(body) > max_chars
        if truncated:
            body = body[:max_chars] + "\n\n[...truncated]"

        return {
            "url": url,
            "final_url": final_url,
            "status": status,
            "content_type": content_type,
            "title": title,
            "text": body,
            "truncated": truncated,
        }

    def duckduckgo_search(self, query, count=5, search_lang="en"):
        """DuckDuckGo fallback. Returns dict with keys: results, query, count."""
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

        results = []
        for m in re.finditer(
            r'<a\s+rel="nofollow"\s+class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
            page, re.DOTALL,
        ):
            raw_url, raw_title = m.group(1), m.group(2)
            url_match = re.search(r'uddg=([^&]+)', raw_url)
            url = urllib.parse.unquote(url_match.group(1)) if url_match else raw_url
            title = self._clean_text(re.sub(r"<[^>]+>", "", raw_title), 200)
            results.append({"title": title, "url": url, "description": ""})

        snippets = re.findall(
            r'<a\s+class="result__snippet"[^>]*>(.*?)</a>', page, re.DOTALL,
        )
        for i, snippet in enumerate(snippets):
            if i < len(results):
                results[i]["description"] = self._clean_text(
                    html.unescape(re.sub(r"<[^>]+>", "", snippet)), 300,
                )

        return {"query": query, "count": len(results[:count]), "results": results[:count]}


def build_services(config: dict) -> dict:
    return {
        "web_search_provider": WebSearchProvider(
            search_key=config.get("brave_search_api_key", ""),
            answers_key=config.get("brave_answers_api_key", ""),
        ),
    }
