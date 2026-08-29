"""Fetch a file from the internet into the agent's own tree.

The capability this restores is larger than it looks. ``sdk.net.http`` answers
with *decoded text* under a size cap, so until ``to_file`` existed there was no
route by which an image, a PDF, an archive or a track could reach the agent at
all — every parser the store ships for those formats was unreachable except for
files a person had already handed over. This is the other half of them.

**Where it lands is not a choice.** Everything under ``workspace`` is
free-write (the authoring grant), so a download into ``workspace/downloads``
costs one dialog for the *host* and nothing for the write. Point it somewhere
else and the kernel asks about the destination as well, which is a question the
agent cannot usefully answer on the user's behalf — so the folder is fixed and
``filename`` names only the file.

**Naming is done here, not by the kernel**, which writes to whatever path it is
handed. Three sources in order — the ``Content-Disposition`` header, the URL's
own last segment, then a guess from ``Content-Type`` — because the extension is
not decoration: ``parse.modality`` routes on it, so a ``.pdf`` saved as
``download`` is a file nothing can open. The URL's suffix is deliberately *not*
trusted over the server's content type, since a URL ending ``.php`` may answer
with a JPEG and frequently does.

Redirects are half-handled by the kernel: hops inside one host are followed
there, and a hop to a *different* host comes back as a 3xx because that host
has not been through the gate. So the loop below exists for the cross-host case
only, and each turn of it is a fresh policy decision — which is the point, not
an inconvenience.
"""

dependencies_files = []
dependencies_pip = []
requests = ["net.http", "paths.get", "parse.modality", "fs.list", "fs.move"]

from urllib.parse import unquote, urljoin, urlsplit

from guest.bases import BaseTool

#: Where downloads go, inside the tree the agent may write to freely.
FOLDER = "downloads"

#: Cross-host redirects only — the kernel already followed the same-host ones.
MAX_HOPS = 4

#: Enough to name the common cases. An unknown type gets no extension rather
#: than a wrong one: no suffix routes to "unknown", which is honest, where
#: ``.bin`` would route to "unknown" while looking like a decision.
EXTENSIONS = {
    "image/jpeg": ".jpg", "image/png": ".png", "image/gif": ".gif",
    "image/webp": ".webp", "image/svg+xml": ".svg", "image/bmp": ".bmp",
    "image/tiff": ".tif", "image/x-icon": ".ico", "image/heic": ".heic",
    "audio/mpeg": ".mp3", "audio/mp3": ".mp3", "audio/wav": ".wav",
    "audio/x-wav": ".wav", "audio/flac": ".flac", "audio/ogg": ".ogg",
    "audio/mp4": ".m4a", "audio/aac": ".aac", "audio/opus": ".opus",
    "video/mp4": ".mp4", "video/webm": ".webm", "video/quicktime": ".mov",
    "video/x-matroska": ".mkv", "video/x-msvideo": ".avi",
    "application/pdf": ".pdf",
    "application/zip": ".zip", "application/x-tar": ".tar",
    "application/gzip": ".gz", "application/x-7z-compressed": ".7z",
    "application/vnd.rar": ".rar",
    "application/json": ".json", "application/xml": ".xml",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/msword": ".doc", "application/vnd.ms-excel": ".xls",
    "text/plain": ".txt", "text/html": ".html", "text/csv": ".csv",
    "text/markdown": ".md",
}

#: Anything a filename may not contain on the stricter of the two platforms,
#: plus the separators, since only the last segment of a name is wanted.
FORBIDDEN = '<>:"/\\|?*'


def _content_type(headers) -> str:
    """The bare media type, without the charset and other parameters."""
    return str((headers or {}).get("content-type", "")).split(";")[0].strip().lower()


def _disposition_name(headers) -> str:
    """The filename a server asked for, if it asked for one.

    Handles both spellings — ``filename*=UTF-8''x`` first, since a server
    sending both means the plain one to be the fallback.
    """
    raw = str((headers or {}).get("content-disposition", ""))
    for marker in ("filename*=", "filename="):
        if marker not in raw:
            continue
        value = raw.split(marker, 1)[1].split(";")[0].strip().strip('"\'')
        if marker == "filename*=" and "''" in value:
            value = value.split("''", 1)[1]
        if value:
            return unquote(value)
    return ""


def _clean(name: str) -> str:
    """One safe filename component, or "".

    Takes the last segment and drops everything a path could be built out of,
    so a server naming ``../../config.json`` names ``config.json`` and gets no
    further. The kernel refuses its own files anyway; this is the cheaper half
    of that, done before a request is made.
    """
    name = str(name or "").strip().replace("\\", "/").split("/")[-1]
    name = "".join(character for character in name
                   if character not in FORBIDDEN and character.isprintable())
    name = name.strip(" .")
    return name[:120]


def _named(url: str, headers, override: str) -> str:
    """What to call the file, in the order the module docstring gives."""
    media = _content_type(headers)
    suggested = _clean(override) or _clean(_disposition_name(headers))
    if not suggested:
        suggested = _clean(unquote(urlsplit(url).path.rsplit("/", 1)[-1]))

    known = EXTENSIONS.get(media, "")
    if not suggested:
        return "download" + (known or "")
    has_suffix = "." in suggested[1:]
    # The server's own content type beats a suffix the URL happened to carry:
    # a link ending ``.php`` that answers with a JPEG is an ordinary thing on
    # the web, and the extension is what decides whether anything can open it.
    if known and not suggested.lower().endswith(known):
        if not has_suffix or (media and media != "application/octet-stream"
                              and known not in (".txt", ".html")):
            return suggested + known if not has_suffix else suggested.rsplit(".", 1)[0] + known
    return suggested


def _exists(sdk, path) -> bool:
    """Whether something is already at ``path``.

    ``fs.list`` pointed at a file answers for that file alone — the way a box
    asks a question ``Path.exists()`` would answer, since it has no pathlib.
    """
    try:
        return bool(sdk.fs.list(path))
    except sdk.Failed:
        return False


def _free_path(sdk, folder: str, name: str) -> str:
    """``folder/name``, with a counter if that is taken.

    Overwriting silently would be the worst of the options: two downloads of
    different things under one name, and no way to tell from the answer which
    one is on disk.
    """
    candidate = sdk.path.join(folder, name)
    if not _exists(sdk, candidate):
        return candidate
    stem, suffix = sdk.path.stem(name), sdk.path.suffix(name)
    for number in range(2, 100):
        candidate = sdk.path.join(folder, f"{stem} ({number}){suffix}")
        if not _exists(sdk, candidate):
            return candidate
    return sdk.path.join(folder, f"{stem} ({number}){suffix}")


class Download(BaseTool):
    """Fetch a URL to a file the agent can then work on."""

    name = "download"
    description = (
        "Download a file from a URL and save it under workspace/downloads, "
        "returning the path. Use this for anything that is not a web page: "
        "images, audio, video, PDFs, spreadsheets, documents, archives, "
        "datasets. Reading a page's text is web_search's job; this is for the "
        "bytes behind a link. The user is asked to approve each new host. "
        "Fails if the file is larger than the user's download limit."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Direct URL of the file itself, not the page "
                               "linking to it.",
            },
            "filename": {
                "type": "string",
                "description": "Optional name to save it as. Leave this out "
                               "unless the user asked for a particular name — "
                               "the server usually names it better, and the "
                               "extension matters.",
            },
            "show": {
                "type": "boolean",
                "description": "Show the file to the user in chat once it is "
                               "downloaded. Set this when they asked to see "
                               "it; leave it out when you are downloading "
                               "something to work on yourself.",
            },
            "narration": {
                "type": "string",
                "description": "A few words on what you are downloading and "
                               "why, shown to the user. E.g. 'grabbing the "
                               "dataset CSV they linked'.",
            },
        },
        "required": ["url"],
    }

    agent_prompt = (
        "## Downloading files\n"
        "download saves a URL to workspace/downloads and gives you the "
        "path. It is the only way to get a non-text file — an image, audio, "
        "video, PDF, spreadsheet, archive — onto disk where you can read or "
        "parse it. It needs the URL of the file itself; a page URL downloads "
        "the HTML. When the user should see what you downloaded, pass "
        "show=true rather than describing it."
    )
    def run(self, sdk, **kwargs):
        """Download one URL, following cross-host redirects as fresh decisions."""
        url = str(kwargs.get("url") or "").strip()
        if not url:
            return sdk.fail("A url is required.")
        if not url.lower().startswith(("http://", "https://")):
            return sdk.fail(f"{url!r} is not an http(s) URL.")

        folder = sdk.path.join(sdk.paths.get("workspace"), FOLDER)

        for _hop in range(MAX_HOPS + 1):
            try:
                answer = sdk.net.http(url, to_file=self._provisional(sdk, folder, url))
            except sdk.Denied as error:
                # Worth catching: the user said no to this host, and retrying
                # is asking the same person the same question.
                return sdk.fail(
                    f"The user declined the request to download from {url}. "
                    f"({error}) Do not retry.")

            status = int(answer.get("status") or 0)
            location = (answer.get("headers") or {}).get("location") or ""
            if status in (301, 302, 303, 307, 308) and location:
                # The kernel already followed everything inside one host, so
                # this is a new host and the next call is a new decision.
                url = urljoin(answer.get("final_url") or url, location)
                continue
            if status >= 400:
                detail = (answer.get("body") or "").strip()[:300]
                return sdk.fail(f"{url} answered {status}."
                                + (f" {detail}" if detail else ""))
            if status < 200 or status >= 300:
                return sdk.fail(f"{url} answered {status}, which is not a file.")
            return self._settled(sdk, folder, answer, kwargs)

        return sdk.fail(f"{url} redirected across more than {MAX_HOPS} hosts.")

    def _provisional(self, sdk, folder, url):
        """A destination to hand the kernel before the headers are known.

        The kernel writes where it is told, and the headers that name a file
        properly arrive *with* the reply — so a destination has to be chosen
        from the URL alone and revisited afterwards. :meth:`_settled` renames
        when the server had a better idea, which is one extra Request in the
        cases that need it and none in the cases that do not. The alternative,
        streaming to a scratch name and always moving, pays that cost every
        time to avoid a name that is usually already right.
        """
        return sdk.path.join(folder, _named(url, {}, ""))

    def _settled(self, sdk, folder, answer, kwargs):
        """Report a finished download, renaming it if the server named it."""
        path = answer.get("path") or ""
        headers = answer.get("headers") or {}
        final_url = answer.get("final_url") or ""
        wanted = _named(final_url or "", headers, kwargs.get("filename") or "")

        if wanted and wanted != sdk.path.name(path):
            destination = _free_path(sdk, folder, wanted)
            try:
                sdk.fs.move(path, destination)
                path = destination
            except sdk.Failed:
                # The bytes are on disk under a workable name; a failed rename
                # is cosmetic and must not lose the download.
                pass

        suffix = sdk.path.suffix(path)
        detail = self._modality(sdk, suffix)
        size = int(answer.get("bytes") or 0)
        media = _content_type(headers)

        summary = (f"Downloaded {size:,} bytes to {path}"
                   + (f" ({media})" if media else "") + ".")
        if detail.get("known"):
            summary += f" It parses as {detail['modality']}."
        elif suffix:
            summary += (f" No parser is installed for {suffix} — "
                        f"read it as bytes, or install one.")

        return sdk.ok(
            {"path": path, "bytes": size, "content_type": media,
             "modality": detail.get("modality", "unknown"),
             "parseable": bool(detail.get("known")), "final_url": final_url},
            llm_summary=summary,
            attachments=[path] if kwargs.get("show") else None)

    def _modality(self, sdk, suffix):
        """What the kernel makes of this extension, or nothing.

        Asked rather than assumed, because the answer depends on which parser
        packages are installed — which this tool cannot know and the agent
        most wants to be told, since it decides whether the next step is
        ``parse.file`` or a bytes read.
        """
        if not suffix:
            return {}
        try:
            return sdk.parse.modality(suffix, detail=True) or {}
        except sdk.Failed:
            return {}
