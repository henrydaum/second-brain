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

#: Which attribute carries a URL, per tag. The set is small on purpose: these
#: are the places a *file* is named, which is a narrower question than "every
#: URL on the page" and keeps a link list short enough to read.
_LINK_ATTRS = {
    "a": ("href",),
    "img": ("src", "data-src"),
    "source": ("src",),
    "video": ("src", "poster"),
    "audio": ("src",),
    "embed": ("src",),
    "object": ("data",),
    "link": ("href",),
}

#: ``<meta property="og:image" content="...">`` and friends. Social-card
#: metadata is often the only place a page names its own primary asset in
#: full resolution, so it is worth the four lines.
_META_KEYS = ("og:image", "og:video", "og:audio", "twitter:image")

#: Redirect statuses ``net.http`` hands back rather than following, when the
#: hop leaves the host it was classified for.
_REDIRECTS = (301, 302, 303, 307, 308)

#: How interesting a link is, by the tag that held it. Having a file
#: extension separates files from pages; this separates the *file* somebody
#: put there to be taken from the file the site needs to render itself.
#:
#: Ordering by extension alone put a favicon, an apple-touch-icon and a logo
#: above the installers on python.org — all four are files, and only one kind
#: is what anybody came for. The tag is what tells them apart, structurally
#: and on every site: an ``<a>`` is something a person clicks, embedded media
#: is what the page shows, and ``<link>``/``<meta>`` is chrome the browser
#: reads and nobody looks at.
#: Extensions that name a *page*, not a file. They have to be listed because
#: the files/pages split reads the extension, and these are the ones where the
#: extension lies about which side of it the link falls on: ``.php`` is a
#: script that renders a document, ``.html`` is the document. Left in the file
#: group they dominated it on any server that spells its URLs this way —
#: archive.org put six ``search.php`` links above the video the page is about.
_PAGE_EXTENSIONS = frozenset({
    ".html", ".htm", ".xhtml", ".shtml", ".php", ".php3", ".php4", ".php5",
    ".asp", ".aspx", ".jsp", ".jspx", ".cgi", ".pl", ".do", ".action",
})

_RANK = {"a": 0,
         "video": 1, "audio": 1, "source": 1, "embed": 1, "object": 1,
         "img": 2,
         "link": 3, "meta": 3}


class _Links:
    """Pull every URL a page names out of its HTML.

    Regex rather than ``html.parser``: this file already strips tags that way
    for :meth:`fetch_url`, the input is arbitrary broken markup that a strict
    parser raises on more often than it helps, and nothing here needs a tree —
    the question is "which attributes hold URLs", which is flat.

    **This is the answer to links that are not in the text.** ``fetch_url``
    returns a page rendered down to prose, which is right for reading and
    throws away every ``href`` — so a download button, a "get the PDF" link,
    an image gallery all vanish before the agent sees them. They were never
    hidden; they were in the markup the cleaner discarded.

    What it does not reach is a URL that does not exist until JavaScript
    builds one. No amount of parsing finds those, and the honest answer is to
    say so rather than to return a shorter list without comment.
    """

    #: One tag with its attributes, unparsed.
    TAG = re.compile(r"<\s*(a|img|source|video|audio|embed|object|link)\b([^>]*)>",
                     re.IGNORECASE)
    ATTR = re.compile(r"([\w:-]+)\s*=\s*(\"[^\"]*\"|\'[^\']*\'|[^\s>]+)")
    META = re.compile(r"<\s*meta\b([^>]*)>", re.IGNORECASE)
    #: The visible text of an anchor, which is usually what the button said.
    ANCHOR = re.compile(r"<a\b[^>]*>(.*?)</a>", re.IGNORECASE | re.DOTALL)

    @staticmethod
    def _attrs(raw: str) -> dict:
        """One tag's attributes as a dict, lowercased keys, unquoted values."""
        found = {}
        for match in _Links.ATTR.finditer(raw or ""):
            value = match.group(2).strip()
            if value[:1] in ('"', "'"):
                value = value[1:-1]
            found[match.group(1).lower()] = html.unescape(value.strip())
        return found

    @staticmethod
    def _labels(markup: str) -> dict:
        """href -> the anchor's own visible text.

        Read separately from the tag scan because the text sits *between* the
        tags rather than in an attribute. It is what tells an agent which of
        nine links on a release page is the one it wants, so it is worth a
        second pass.
        """
        labels = {}
        for match in _Links.ANCHOR.finditer(markup or ""):
            opening = match.group(0)
            attrs = _Links._attrs(opening[:opening.find(">")])
            href = attrs.get("href", "")
            if not href:
                continue
            text = html.unescape(re.sub(r"<[^>]+>", " ", match.group(1)))
            text = re.sub(r"\s+", " ", text).strip()
            if text and href not in labels:
                labels[href] = text[:120]
        return labels

    @staticmethod
    def extract(markup: str, base: str) -> list:
        """Every URL the page names, absolute, in document order.

        Deduplicated on the resolved URL, keeping the first occurrence — which
        is the one whose label came from the page rather than from a repeated
        footer.
        """
        markup = markup or ""
        labels = _Links._labels(markup)
        found, seen = [], set()

        def add(raw, kind, label=""):
            raw = (raw or "").strip()
            if not raw or raw.startswith(("#", "javascript:", "data:", "mailto:", "tel:")):
                return
            try:
                absolute = urllib.parse.urljoin(base, raw)
            except ValueError:
                return
            if not absolute.lower().startswith(("http://", "https://")):
                return
            if absolute in seen:
                return
            seen.add(absolute)
            path = urllib.parse.urlsplit(absolute).path
            _, _, tail = path.rpartition("/")
            extension = ""
            if "." in tail[1:]:
                extension = "." + tail.rsplit(".", 1)[1].lower()
                if len(extension) > 8 or not extension[1:].isalnum():
                    extension = ""
            found.append({"url": absolute, "text": label, "kind": kind,
                          "extension": extension})

        for match in _Links.TAG.finditer(markup):
            tag = match.group(1).lower()
            attrs = _Links._attrs(match.group(2))
            # ``<link rel=stylesheet>`` is machinery, not content; the ones
            # worth keeping name a document or an icon.
            if tag == "link" and attrs.get("rel", "").lower() not in (
                    "alternate", "icon", "shortcut icon", "apple-touch-icon"):
                continue
            for attribute in _LINK_ATTRS.get(tag, ()):
                if (value := attrs.get(attribute)):
                    add(value, tag, labels.get(value, ""))
            # ``srcset`` is a comma-separated list with size descriptors.
            for candidate in (attrs.get("srcset") or "").split(","):
                add(candidate.strip().split(" ")[0], tag)

        for match in _Links.META.finditer(markup):
            attrs = _Links._attrs(match.group(1))
            key = (attrs.get("property") or attrs.get("name") or "").lower()
            if key in _META_KEYS:
                add(attrs.get("content"), "meta")

        return found


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
        "page_links",
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

    def page_links(self, sdk, url, limit=200):
        """Fetch a page and answer with the URLs it names, not its prose.

        The counterpart to :meth:`fetch_url`, which renders a page down to
        text and discards every ``href`` on the way. Both are one fetch; they
        differ only in what is kept, which is why they live side by side
        rather than one growing a flag.

        Returns ``{url, final_url, status, content_type, title, links,
        file_count, page_count, truncated}``, each link ``{url, text, kind,
        extension}``. Links carrying a file extension come first — see below.

        Redirects are followed here, one call at a time. ``net.http`` hands
        back a cross-host 3xx rather than following it, so each hop is a fresh
        policy decision — and a link extractor that gave up at the first
        redirect would be useless, since a shortened or canonical URL is
        exactly the kind a person pastes.
        """
        for _hop in range(5):
            answer = sdk.net.http(url, headers={
                "User-Agent": _UA,
                "Accept": "text/html,application/xhtml+xml,application/xml;"
                          "q=0.9,*/*;q=0.8",
            })
            status = int(answer.get("status") or 0)
            location = (answer.get("headers") or {}).get("location") or ""
            if status in _REDIRECTS and location:
                url = urllib.parse.urljoin(url, location)
                continue
            break
        else:
            return {"url": url, "final_url": url, "status": 0,
                    "content_type": "", "title": "", "links": [],
                    "truncated": False, "error": "too many redirects"}

        headers = answer.get("headers") or {}
        markup = answer.get("body") or ""
        content_type = (headers.get("content-type") or "").lower()

        title = ""
        if (match := re.search(r"<title[^>]*>(.*?)</title>", markup,
                               re.DOTALL | re.IGNORECASE)):
            title = self._clean_text(
                html.unescape(re.sub(r"<[^>]+>", "", match.group(1))), 300)

        links = _Links.extract(markup, url)
        # A link with a file extension points at a *thing*; one without points
        # at another page. Every site's header and footer is the second kind
        # and there are dozens of them, so ordering by that one bit is what
        # makes the answer readable — and it is what makes ``limit`` safe,
        # since the rows that fall off the end are now navigation rather than
        # the file the caller came for. Files are then ranked by ``_RANK``,
        # which is a stable sort, so document order survives inside a tier.
        def is_file(link):
            extension = link["extension"]
            return bool(extension) and extension not in _PAGE_EXTENSIONS

        files = sorted((link for link in links if is_file(link)),
                       key=lambda link: _RANK.get(link["kind"], 2))
        pages = [link for link in links if not is_file(link)]
        ordered = files + pages
        return {
            "url": url,
            "final_url": url,
            "status": status,
            "content_type": content_type,
            "title": title,
            "links": ordered[:limit],
            "file_count": len(files),
            "page_count": len(pages),
            "truncated": len(links) > limit or bool(answer.get("truncated")),
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
