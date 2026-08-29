"""Search the public web, or fetch one page as text.

All of the routing, none of the effects: every network call belongs to
``web_search_provider``, reached through ``sdk.services.call``. That Request is
SAFE because the callee spends *this* chain's grant — the service's own
``net.http`` is classified with this tool still in the chain — so the tool
needs no egress declaration and cannot acquire one by asking.
"""


dependencies_files = ['services/service_web_search.py']
dependencies_pip = []

import re

from guest.bases import BaseTool

_URL_RE = re.compile(r"^(https?://|www\.)\S+$", re.IGNORECASE)
_PROVIDER = "web_search_provider"

#: How many page links to show beside the files. Enough to navigate a site
#: with, far short of enough to bury the thing the caller came for.
PAGE_SAMPLE = 25


class WebSearch(BaseTool):
    """Web search."""

    name = "web_search"
    description = (
        "Search the public web for information that is not already available in the "
        "local file system, especially current facts, external references, or verification. "
        "Uses Brave search by default and can use Brave Answers when mode='answers' or mode='auto'. "
        "DuckDuckGo is used as a fallback. If 'query' is a URL (http://, https://, or www.), "
        "the page is fetched and its cleaned text is returned; add mode='links' to get the "
        "URLs the page links to instead of its text, which is how you find a file to download "
        "when the link is on a button rather than in the prose."
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
                "description": "Search mode. 'auto' uses Answers for question-like queries and Search otherwise. 'links' only applies when 'query' is a URL: it lists what the page links to instead of reading it.",
                "enum": ["auto", "search", "answers", "links"],
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
            "narration": {
                "type": "string",
                "description": (
                    "A few words on what you are looking for and why, shown to "
                    "the user beside the call. E.g. 'checking today's AI "
                    "datacenter coverage'."
                ),
            },
        },
        "required": ["query"],
    }
    # Spelled out rather than ``[_PROVIDER]``: declarations are read by AST,
    # so a name is read as nothing at all.
    requires_services = ["web_search_provider"]
    requests = ["service.call"]

    # ── talking to the provider ─────────────────────────────────────

    def _call(self, sdk, method, **kwargs):
        """One provider method, as a dict.

        A refused API call comes back *in* the dict as ``http_status`` rather
        than as an exception, because that is what survives ``service.call``.
        A genuine transport failure (no reply at all) still raises
        ``sdk.Failed``, which callers let propagate.
        """
        return sdk.services.call(_PROVIDER, method, **kwargs) or {}

    def _refused(self, data) -> str:
        """The reason an API said no, or "" if it did not.

        One reader, because three call sites make the same decision: whether
        to fall back or to report. ``http_status`` is only present when the
        provider got an error status back.
        """
        status = (data or {}).get("http_status")
        if not status:
            return ""
        detail = (data or {}).get("error") or ""
        return f"HTTP {status}" + (f": {detail[:500]}" if detail else "")

    # ── rendering ───────────────────────────────────────────────────

    def _looks_question_like(self, query):
        """Whether a query reads like a question rather than keywords."""
        q = query.lower().strip()
        starters = (
            "what", "why", "how", "when", "where", "who", "which", "compare",
            "explain", "summarize",
        )
        return q.endswith("?") or q.startswith(starters) or len(query.split()) >= 8

    def _format_search_result(self, sdk, data, engine="brave", prefix=""):
        """Search results as a numbered list."""
        results = data.get("results", [])
        query = data.get("query", "")
        label = "DuckDuckGo" if engine == "duckduckgo" else "search"

        lines = [f"Found {len(results)} {label} result(s) for '{query}':"]
        for index, item in enumerate(results, start=1):
            lines.append(
                f"{index}. {item.get('title') or '(no title)'} — "
                f"{item.get('url') or ''}")
            if item.get("description"):
                lines.append(f"   {item['description']}")

        summary = ("\n".join(lines) if results
                   else f"No {label} results found for '{query}'.")
        if prefix:
            summary = prefix + "\n\n" + summary

        payload = {"mode": "search", **data}
        if engine == "duckduckgo":
            payload["engine"] = engine
        return sdk.ok(payload, llm_summary=summary)

    def _format_answers_result(self, sdk, data):
        """A grounded answer with its sources."""
        query = data.get("query", "")
        answer = data.get("answer", "")
        sources = data.get("sources", [])

        lines = [f"Brave answer for '{query}':", answer]
        if sources:
            lines.append("Sources:")
            for index, source in enumerate(sources, start=1):
                lines.append(
                    f"{index}. {source.get('title') or '(untitled source)'} — "
                    f"{source.get('url')}")

        return sdk.ok({"mode": "answers", **data},
                      llm_summary="\n".join(lines))

    # ── the tool ────────────────────────────────────────────────────

    def run(self, sdk, **kwargs):
        """Run web search."""
        query = (kwargs.get("query") or "").strip()
        if not query:
            return sdk.fail("Missing required parameter: query")

        if _URL_RE.match(query):
            if (kwargs.get("mode") or "").strip().lower() == "links":
                return self._links(sdk, query)
            return self._fetch(sdk, query)

        try:
            count = int(kwargs.get("count", 5))
        except (TypeError, ValueError):
            count = 5
        count = max(1, min(count, 20))

        mode = (kwargs.get("mode") or "auto").strip().lower()
        if mode == "links":
            # Reached only when the query was not a URL, and there is nothing
            # sensible to fall back to: silently searching instead would
            # answer a different question than the one asked.
            return sdk.fail(
                "mode='links' needs a URL as the query — it lists what one "
                "page links to. Search for the page first, then pass its URL.")
        if mode not in {"auto", "search", "answers"}:
            mode = "auto"

        country = (kwargs.get("country") or "").strip()
        search_lang = (kwargs.get("search_lang") or "en").strip() or "en"
        safesearch = (kwargs.get("safesearch") or "moderate").strip().lower()
        if safesearch not in {"off", "moderate", "strict"}:
            safesearch = "moderate"
        freshness = (kwargs.get("freshness") or "").strip()

        has_answers = bool(self._call(sdk, "has_answers_key"))
        has_search = bool(self._call(sdk, "has_search_key"))

        chosen = mode
        if mode == "auto":
            chosen = "answers" if self._looks_question_like(query) else "search"

        if chosen == "answers" and not has_answers:
            if mode != "auto":
                return sdk.fail(
                    "Brave Answers API key not configured. Set "
                    "secret_brave_answers_api_key in the service settings.")
            chosen = "search"

        if chosen == "search" and not has_search:
            return self._duckduckgo(
                sdk, query, count, search_lang,
                prefix="No Brave API key configured — using DuckDuckGo "
                       "fallback.")

        if chosen == "answers":
            data = self._call(sdk, "answers", query=query, country=country,
                              search_lang=search_lang)
            if not (why := self._refused(data)):
                return self._format_answers_result(sdk, data)
            # Answers is the flakier of the two endpoints, so in ``auto`` its
            # refusal is a reason to try something else rather than to give up
            # — the user asked a question, not for a particular API.
            if mode != "auto":
                return sdk.fail(f"Brave Answers {why}")
            if not has_search:
                return self._duckduckgo(
                    sdk, query, count, search_lang,
                    prefix="Brave APIs unavailable — using DuckDuckGo "
                           "fallback.")
            data = self._call(sdk, "search", query=query, count=count,
                              country=country, search_lang=search_lang,
                              safesearch=safesearch, freshness=freshness)
            if (why := self._refused(data)):
                return sdk.fail(f"Brave Search {why}")
            return self._format_search_result(
                sdk, data,
                prefix="Brave Answers was unavailable, so I used Brave Search "
                       "instead.")

        data = self._call(sdk, "search", query=query, count=count,
                          country=country, search_lang=search_lang,
                          safesearch=safesearch, freshness=freshness)
        if (why := self._refused(data)):
            return sdk.fail(f"Brave Search {why}")
        return self._format_search_result(sdk, data)

    def _fetch(self, sdk, query):
        """The query was a URL, so read the page instead of searching."""
        url = (query if query.lower().startswith(("http://", "https://"))
               else "https://" + query)
        data = self._call(sdk, "fetch_url", url=url)
        status = int(data.get("status") or 0)
        if status >= 400:
            return sdk.fail(f"Fetch HTTP error {status} for {url}")

        header = (f"Fetched {data.get('final_url') or url} (status {status}, "
                  f"{data.get('content_type') or 'unknown type'})")
        if data.get("title"):
            header += f"\nTitle: {data['title']}"
        summary = header + "\n\n" + (data.get("text") or "")
        if data.get("truncated"):
            summary += "\n\n[content truncated]"
        return sdk.ok({"mode": "fetch", **data}, llm_summary=summary)

    def _links(self, sdk, query):
        """List what a page links to, rather than what it says.

        **Two sections, because a page links to two different kinds of thing.**
        A link with a file extension points at something downloadable; one
        without points at another page, and every site carries dozens of those
        in its header and footer. Reported as one flat list they came back at
        roughly ten to one against, and the caller had to read 200 rows to find
        the fourteen it wanted — which is what actually happened the first time
        this ran against an image site.

        So files come first and in full, and pages are sampled with a count of
        the rest. Pages are not noise and are not dropped: following one is how
        you get from a downloads index to a release page. They are just never
        what you are looking at *this* list for.

        A table, because the caller is choosing one row out of many and the
        three things it chooses on — the link text, the tag that held it, the
        extension — do not survive being flattened into prose.
        """
        url = (query if query.lower().startswith(("http://", "https://"))
               else "https://" + query)
        data = self._call(sdk, "page_links", url=url)
        status = int(data.get("status") or 0)
        if status >= 400:
            return sdk.fail(f"Fetch HTTP error {status} for {url}")
        if (why := data.get("error")):
            return sdk.fail(f"Could not read {url}: {why}")

        links = data.get("links") or []
        if not links and not (200 <= status < 300):
            # A 3xx we could not follow, or a 2xx-adjacent refusal. Saying
            # "this page has no links" would be a claim about a page nobody
            # was shown — sites answer 307 to an unrecognised client, and the
            # status is the only part of that worth reporting.
            return sdk.fail(
                f"{url} answered {status} rather than a page, so there is "
                f"nothing to list. The site may be refusing this client.")
        if not links:
            return sdk.ok(
                {"mode": "links", **data},
                llm_summary=(
                    f"{data.get('final_url') or url} links to nothing "
                    f"reachable. Either the page is empty of links or they "
                    f"are built by JavaScript, which cannot be read this way "
                    f"— try the site's own download or API page."))

        where = data.get("final_url") or url
        files = [link for link in links if link.get("extension")]
        pages = [link for link in links if not link.get("extension")]

        parts = [f"{len(files)} file link(s) and {len(pages)} page link(s) "
                 f"on {where}"
                 + (f" ({data['title']})" if data.get("title") else "")
                 + ("; the list was truncated." if data.get("truncated")
                    else ".")]

        if files:
            parts.append(sdk.md.table(
                ["URL", "Link text", "Ext", "Tag"],
                [[link["url"], link.get("text") or "",
                  link.get("extension") or "", link.get("kind") or ""]
                 for link in files]))
        else:
            parts.append(
                "No link on this page names a file. The downloads are "
                "probably on a page this one links to, or are built by "
                "JavaScript and cannot be read this way.")

        if pages:
            shown = pages[:PAGE_SAMPLE]
            listed = "\n".join(
                f"- {link.get('text') or '(no text)'} — {link['url']}"
                for link in shown)
            heading = "Pages linked from here"
            if len(pages) > len(shown):
                heading += (f" (first {len(shown)} of {len(pages)}; ask for "
                            f"one of these by URL to go deeper)")
            parts.append(f"{heading}:\n{listed}")

        return sdk.ok({"mode": "links", **data}, llm_summary="\n\n".join(parts))

    def _duckduckgo(self, sdk, query, count, search_lang, prefix=""):
        """The keyless fallback, and the reason we are using it."""
        data = self._call(sdk, "duckduckgo_search", query=query, count=count,
                          search_lang=search_lang)
        if (why := self._refused(data)):
            return sdk.fail(f"DuckDuckGo fallback failed: {why}")
        return self._format_search_result(sdk, data, engine="duckduckgo",
                                          prefix=prefix)
