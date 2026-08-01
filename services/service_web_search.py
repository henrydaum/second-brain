"""Web search over Brave, with a keyless DuckDuckGo fallback.

Every effect this service has is one Request — ``net.http`` — and it holds no
credential at any point. The two API keys are declared ``secret_*``, so
``sdk.config.read`` hands back ``<secret:name>`` handles; the handles go
straight into the request headers and the kernel swaps in the real key on the
way out, after the policy function has already decided. The service can
therefore neither log a key nor put one in an error message, which is the
property that matters for code nobody reviews closely.

Reaching ``api.search.brave.com`` at all depends on the user having put that
host in ``net_allowed_hosts``; without it every search raises an approval
dialog naming the host. That is the intended shape — a person decides what the
app may talk to, and this file cannot widen it.

This file used to carry ~55 lines proposing those hosts to the user on the
first search, through a ``config.write`` it did not own so the kernel would
raise the dialog. All of it is gone: the kernel's own approval dialog now
offers "Always allow api.search.brave.com" beside "Allow once", writing the
same setting. Asking at the moment of the refusal is strictly better than
asking beforehand — the person is already looking at the request that needs
it, and no plugin has to know the setting exists.
"""

dependencies_files = []
dependencies_pip = []

import html
import json
import re
import urllib.parse

from guest.bases import BaseService

_UA = "SecondBrain-WebSearch/3.0"


class WebSearchProvider(BaseService):
    """Brave Search, Brave Answers, and a DuckDuckGo fallback."""

    name = "web_search_provider"
    description = "Search the public web and fetch pages as cleaned text."
    shared = True
    requests = ["net.http", "config.read"]
    exports = [
        "search",
        "answers",
        "fetch_url",
        "duckduckgo_search",
        "has_search_key",
        "has_answers_key",
    ]
    config_settings = [
        ("Brave Search API Key", "secret_brave_search_api_key",
         "API key for Brave Web Search. Stored as a secret: plugins receive "
         "an opaque handle and the kernel substitutes the real key into the "
         "outbound request.",
         "",
         {"type": "text"}),

        ("Brave Answers API Key", "secret_brave_answers_api_key",
         "API key for Brave Answers (grounded answer generation). Stored as a "
         "secret, same as the search key.",
         "",
         {"type": "text"}),
    ]

    SEARCH_API_URL = "https://api.search.brave.com/res/v1/web/search"
    ANSWERS_API_URL = "https://api.search.brave.com/res/v1/chat/completions"
    DDG_URL = "https://html.duckduckgo.com/html/"

    def start(self, sdk):
        """Nothing to open — every call reads its key and asks the kernel."""
        return True

    def stop(self, sdk):
        """No connection, no thread, nothing to tear down."""
        return None

    # ── keys ────────────────────────────────────────────────────────
    #
    # These return the *handle*, not the key. A handle is a plain string that
    # only means anything inside ``net.http``, so it is safe to hold, compare
    # and pass around — and an empty one is how "not configured" reads. The
    # environment-variable fallbacks the native version carried (BRAVE_API_KEY
    # and friends) are gone: an env var declares nothing, cannot be a
    # ``secret_*``, and would have to be revealed in plaintext to be used.

    def _search_key(self, sdk) -> str:
        """The search key's handle, or ""."""
        return str(sdk.config.read("secret_brave_search_api_key") or "").strip()

    def _answers_key(self, sdk) -> str:
        """The answers key's handle, or ""."""
        return str(
            sdk.config.read("secret_brave_answers_api_key") or "").strip()

    def has_search_key(self, sdk) -> bool:
        """Whether a Brave Search key is configured."""
        return bool(self._search_key(sdk))

    def has_answers_key(self, sdk) -> bool:
        """Whether a Brave Answers key is configured."""
        return bool(self._answers_key(sdk))

    # ── HTTP ────────────────────────────────────────────────────────

    def _clean_text(self, value, limit=None):
        """Collapse whitespace, and truncate if a limit is given."""
        text = (value or "").replace("\n", " ").replace("\r", " ").strip()
        text = " ".join(text.split())
        if limit and len(text) > limit:
            text = text[: max(0, limit - 3)] + "..."
        return text

    def _headers(self, key: str, json_body: bool = False) -> dict:
        """Brave's headers, carrying the key as a handle.

        No ``Accept-Encoding: gzip``. The native version asked for gzip and
        decompressed it itself; ``net.http`` answers with decoded text, so
        asking for a compressed body would only produce mojibake — and there is
        no ``gzip`` left in this file to undo it with.
        """
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": key,
            "User-Agent": _UA,
        }
        if json_body:
            headers["Content-Type"] = "application/json"
        return headers

    def _json(self, sdk, url, *, headers=None, method="GET", body=None):
        """One request, as ``(payload, refusal)`` — exactly one of them set.

        A refusal is **returned, not raised**, and that is a boundary
        requirement rather than a style choice: an export's return value
        crosses ``service.call`` as data, but an exception crosses as a message
        string and nothing else. The native version raised ``HTTPError`` and
        the tool caught it by type to decide whether to fall back to
        DuckDuckGo — a decision that cannot be made from a flattened string. So
        the status travels in the answer where the caller can branch on it.

        An HTTP error status is an ordinary answer here, with the body
        attached, because Brave says *why* it refused in that body: a 429 means
        wait, a 401 means the key is wrong, and only the body separates either
        from a bad parameter.
        """
        answer = sdk.net.http(url, method=method, headers=headers or {},
                              body=body)
        status = int(answer.get("status") or 0)
        text = answer.get("body") or ""
        if status >= 400:
            return None, {"http_status": status,
                          "error": self._clean_text(text, 500)}
        try:
            return json.loads(text), None
        except ValueError as exc:
            return None, {"http_status": status,
                          "error": f"unreadable response: {exc}"}

    # ── public search methods ───────────────────────────────────────
    # Plain dicts out. The tool layer decides how to render them.

    def search(self, sdk, query, count=5, country="", search_lang="en",
               safesearch="moderate", freshness=""):
        """Brave Web Search. Returns {query, count, results, raw}."""
        key = self._search_key(sdk)
        if not key:
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
        data, refusal = self._json(sdk, url, headers=self._headers(key))
        if refusal is not None:
            return {"query": query, "count": 0, "results": [], **refusal}

        web = data.get("web", {}) if isinstance(data, dict) else {}
        results = web.get("results", []) if isinstance(web, dict) else []

        normalized = []
        for item in results[:count]:
            if not isinstance(item, dict):
                continue
            meta = item.get("meta_url")
            normalized.append({
                "title": self._clean_text(item.get("title", ""), 200),
                "url": item.get("url", ""),
                "display_url": (meta.get("display_url", "")
                                if isinstance(meta, dict) else ""),
                "description": self._clean_text(item.get("description", ""),
                                                300),
                "age": item.get("age", ""),
                "language": item.get("language", ""),
            })

        return {"query": query, "count": len(normalized),
                "results": normalized, "raw": data}

    def answers(self, sdk, query, country="", search_lang="en"):
        """Brave Answers. Returns {query, answer, sources, raw}."""
        key = self._answers_key(sdk)
        if not key:
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

        data, refusal = self._json(
            sdk, self.ANSWERS_API_URL, method="POST",
            headers=self._headers(key, json_body=True), body=json.dumps(body))
        if refusal is not None:
            return {"query": query, "answer": "", "sources": [], **refusal}

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
            """Collect every {title, url} pair anywhere in the response."""
            if isinstance(obj, dict):
                maybe_url = obj.get("url")
                maybe_title = (obj.get("title") or obj.get("name")
                               or obj.get("source") or "")
                if isinstance(maybe_url, str) and maybe_url.startswith("http"):
                    citations.append({
                        "title": self._clean_text(maybe_title, 200),
                        "url": maybe_url,
                    })
                for value in obj.values():
                    harvest(value)
            elif isinstance(obj, list):
                for value in obj:
                    harvest(value)

        harvest(data)

        deduped = []
        seen = set()
        for citation in citations:
            url = citation.get("url") or ""
            if url and url not in seen:
                seen.add(url)
                deduped.append(citation)
            if len(deduped) >= 8:
                break

        if not answer_text:
            answer_text = self._clean_text(
                json.dumps(data, ensure_ascii=False), 2500)

        return {"query": query, "answer": answer_text, "sources": deduped,
                "raw": data}

    def fetch_url(self, sdk, url, max_chars=20000):
        """Fetch a page as cleaned text.

        Returns {url, final_url, status, content_type, title, text,
        truncated}. ``final_url`` is the requested URL: ``net.http`` follows
        redirects but does not report where it landed, and inventing a
        different answer would be worse than a truthful one.
        """
        answer = sdk.net.http(url, headers={
            "User-Agent": _UA,
            "Accept": "text/html,application/xhtml+xml,application/xml,"
                      "text/plain,application/json;q=0.9,*/*;q=0.8",
        })
        status = int(answer.get("status") or 0)
        text = answer.get("body") or ""
        headers = answer.get("headers") or {}
        content_type = (headers.get("content-type") or "").lower()

        title = ""
        if "html" in content_type or "<html" in text[:2000].lower():
            match = re.search(r"<title[^>]*>(.*?)</title>", text,
                              re.DOTALL | re.IGNORECASE)
            if match:
                title = self._clean_text(
                    html.unescape(re.sub(r"<[^>]+>", "", match.group(1))), 300)
            body = re.sub(
                r"(?is)<(script|style|noscript|svg|head)[^>]*>.*?</\1>",
                " ", text)
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
            "final_url": url,
            "status": status,
            "content_type": content_type,
            "title": title,
            "text": body,
            "truncated": truncated,
        }

    def duckduckgo_search(self, sdk, query, count=5, search_lang="en"):
        """Keyless fallback. Returns {query, count, results}."""
        payload = urllib.parse.urlencode(
            {"q": query, "kl": search_lang or "en"})
        answer = sdk.net.http(
            self.DDG_URL, method="POST", body=payload,
            headers={
                "User-Agent": _UA,
                "Content-Type": "application/x-www-form-urlencoded",
            })
        status = int(answer.get("status") or 0)
        if status >= 400:
            return {"query": query, "count": 0, "results": [],
                    "http_status": status,
                    "error": self._clean_text(answer.get("body") or "", 500)}
        page = answer.get("body") or ""

        results = []
        for match in re.finditer(
            r'<a\s+rel="nofollow"\s+class="result__a"[^>]*href="([^"]*)"'
            r'[^>]*>(.*?)</a>',
            page, re.DOTALL,
        ):
            raw_url, raw_title = match.group(1), match.group(2)
            url_match = re.search(r'uddg=([^&]+)', raw_url)
            url = (urllib.parse.unquote(url_match.group(1))
                   if url_match else raw_url)
            title = self._clean_text(re.sub(r"<[^>]+>", "", raw_title), 200)
            results.append({"title": title, "url": url, "description": ""})

        snippets = re.findall(
            r'<a\s+class="result__snippet"[^>]*>(.*?)</a>', page, re.DOTALL)
        for index, snippet in enumerate(snippets):
            if index < len(results):
                results[index]["description"] = self._clean_text(
                    html.unescape(re.sub(r"<[^>]+>", "", snippet)), 300)

        return {"query": query, "count": len(results[:count]),
                "results": results[:count]}
