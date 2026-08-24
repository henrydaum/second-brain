"""
Info tool — the agent's lookup for how Second Brain works and what is here.

This exists to make the system prompt short. Guidance that every turn paid for
whether or not it was needed — the slash-command catalog, per-tool usage notes,
the SQL rules, the pointer at a 23 KB SDK reference — is answered here instead,
on the turns that actually ask.

The trade is a round trip against a permanent per-call cost, and it only pays
when the roster is good enough to decide from. So every listing leads with the
name and one line, and the detail is a second call. A roster entry that does
not say enough is the failure mode worth watching: the agent does not error, it
simply never drills in, and nothing anywhere reports that.

**Enumerated, not searched.** Every kind but ``sdk`` and ``docs`` is a closed
list of tens of items, and an enum cannot miss where a query can — an agent
handed nothing concludes the thing does not exist. The two markdown kinds are
the exception because a document is navigated rather than enumerated, and there
matching is deliberately generous: a slug, an SDK namespace, a word from a
heading, or a near miss all resolve, and an ambiguous query answers with the
candidates rather than with the longest one.

Everything it reports comes from a Request the kernel already answers, which is
why this is a store package and not kernel code: nothing here needs standing
knowledge the kernel does not already have.
"""

dependencies_files = []
dependencies_pip = []
requests = ["fs.read", "fs.list", "fs.exists", "paths.get", "db.query",
            "tool.list", "command.list", "service.list",
            "task.list", "task.graph", "plugin.list"]

import difflib
import re

from guest.bases import BaseTool

#: The closed vocabulary. Order is the order the agent_prompt lists them in:
#: what to read before writing code first, then what exists on this machine.
KINDS = ("sdk", "docs", "templates", "tools", "commands", "services",
         "tasks", "frontends", "scripts", "database")

#: A safety net, not a routine trim. Chapters are leaves (see _chapters) and
#: templates are the authoring contract — half a contract is worse than none —
#: so nothing should reach this in normal use. If something does, the note says
#: which call to make instead.
MAX_CHARS = 20000

#: Families that live under a tree root, and the filename prefix each one uses.
#: Mirrors trees.py; kept here because a guest cannot import it.
FAMILY_PREFIX = {
    "tools": "tool_",
    "commands": "command_",
    "services": "service_",
    "tasks": "task_",
    "frontends": "frontend_",
    "parsers": "parse_",
    "llm": "llm_",
}

#: Trees searched when resolving one plugin's path, agent-first. The agent
#: means the file it wrote, so workspace wins — the same order
#: isolation.resolve_script uses and the reverse of discovery precedence.
TREES = ("workspace", "installed", "bundled")


# ── Markdown chaptering ───────────────────────────────────────────────

def _slug(title):
    """A heading as a stable lookup key."""
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", title.lower())).strip("-")


def _chapters(text):
    """Split markdown into leaf chapters at ``##`` and ``###``.

    Every heading's body runs to the *next heading of any level*, which is what
    makes each chapter a leaf: ``## The Request reference`` becomes its own
    three-line preamble rather than swallowing the sixteen ``###`` families
    under it. That is the whole point — a chapter has to be small enough that
    fetching one is cheaper than fetching the file.

    Fenced code is tracked because it has to be. SDK.md contains Python
    comments at column zero inside ``python`` fences (``# helper_words.py``),
    and a regex that does not know it is inside a fence reads them as headings.
    """
    lines = text.split("\n")
    heads = []
    fenced = False
    for index, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        match = re.match(r"^(#{2,3})\s+(.+?)\s*$", line)
        if match:
            heads.append((index, len(match.group(1)), match.group(2)))

    found = []
    for position, (index, level, title) in enumerate(heads):
        end = heads[position + 1][0] if position + 1 < len(heads) else len(lines)
        body = "\n".join(lines[index:end]).strip()
        found.append({
            "level": level,
            "title": title,
            "slug": _slug(title),
            "body": body,
            # Which SDK namespaces this chapter documents. The cheapest useful
            # index available and the one an agent actually queries by: it
            # thinks "where is sdk.fs written up", not "what is that section
            # called". Files and processes is unfindable by its title.
            "namespaces": sorted(set(re.findall(r"sdk\.([a-z_]+)\.", body))),
            "lines": end - index,
        })
    return found


def _ranked(chapters, pattern):
    """Chapters containing ``pattern``, densest first, near-ties kept.

    Presence alone is useless for anything an SDK guide mentions in passing:
    ``sdk.fs`` appears in eleven chapters and is *documented* in one, so a
    membership test answered with a menu where the reference was wanted.
    Density separates them — a chapter that documents a call uses it fifteen
    times, one that references it uses it once — and where nothing leads, the
    ambiguity is real and every candidate is offered.
    """
    scored = sorted(((len(re.findall(pattern, chapter["body"])), chapter)
                     for chapter in chapters), key=lambda pair: -pair[0])
    scored = [(count, chapter) for count, chapter in scored if count]
    if not scored:
        return []
    best = scored[0][0]
    if best <= 1:
        return [chapter for _, chapter in scored]
    return [chapter for count, chapter in scored if count * 2 >= best]


def _match(chapters, query):
    """Chapters matching ``query``, narrowest reading first, or [].

    Five passes, each returning whole rather than merging, so an exact slug is
    never diluted by a fuzzy near-miss. The two middle passes are what make
    this usable from an agent's side: it asks for what it wants to *call*
    ("fs", "spawn"), not for what the section happens to be titled — "Files
    and processes" and "Orchestrating subagents" are unfindable by name.
    """
    wanted = (query or "").strip().lower().lstrip("#")
    if not wanted:
        return []
    escaped = re.escape(wanted)
    for candidates in (
        [c for c in chapters if c["slug"] == wanted],
        _ranked(chapters, r"sdk\." + escaped + r"\."),          # a namespace
        _ranked(chapters, r"sdk\.[a-z_]+\." + escaped + r"\b"),  # a method
        [c for c in chapters
         if wanted in c["title"].lower() or wanted in c["slug"]],
    ):
        if candidates:
            return candidates
    close = difflib.get_close_matches(
        wanted, [c["slug"] for c in chapters], n=5, cutoff=0.5)
    return [c for c in chapters if c["slug"] in close]


def _toc(chapters, label):
    """A chapter list, indented by heading level, with what each covers."""
    lines = [f"# {label} — {len(chapters)} sections", ""]
    for chapter in chapters:
        indent = "  " * (chapter["level"] - 2)
        note = ""
        if chapter["namespaces"]:
            note = "  (sdk." + ", sdk.".join(chapter["namespaces"][:6]) + ")"
        lines.append(f"{indent}{chapter['slug']}  — {chapter['title']}"
                     f" [{chapter['lines']} lines]{note}")
    return "\n".join(lines)


def _render_chapter(chapter, source):
    """One chapter's body, capped, saying where it came from."""
    body = chapter["body"]
    if len(body) > MAX_CHARS:
        body = body[:MAX_CHARS] + (
            f"\n\n[truncated at {MAX_CHARS} characters — "
            f"read {source} directly for the rest]")
    return f"{body}\n\n---\nFrom {source}"


def _ambiguous(matches, kind, query):
    """Several chapters matched: name them rather than concatenating them."""
    lines = [f"{len(matches)} sections match '{query}'. "
             f"Ask for one by its key:", ""]
    for chapter in matches:
        lines.append(f"  info(\"{kind}\", \"{chapter['slug']}\")"
                     f"  — {chapter['title']} [{chapter['lines']} lines]")
    return "\n".join(lines)


# ── Paths ─────────────────────────────────────────────────────────────

def _join(*parts):
    """Join path segments with forward slashes.

    Not ``os.path``: the validator refuses ``os`` (it is an effect module) and
    every kernel path handler accepts forward slashes on Windows. Separators
    inside a segment are normalized too, or a Windows tree root joined to a
    relative path answers ``C:\\Users\\...\\workspace/scripts/x.py``, which is
    a path the agent then copies into a message.
    """
    cleaned = [str(part).replace("\\", "/").rstrip("/") for part in parts if part]
    return "/".join(part for part in cleaned if part)


def _locate(sdk, family, filename):
    """The first tree holding ``family/filename``, or ""."""
    for tree in TREES:
        try:
            root = sdk.paths.get(tree)
        except sdk.Failed:
            continue
        candidate = _join(root, family, filename)
        try:
            if sdk.fs.exists(candidate):
                return candidate
        except sdk.Failed:
            continue
    return ""


def _plugin_path(sdk, family, name):
    """Where one registered plugin's source lives, or "" if not resolvable."""
    prefix = FAMILY_PREFIX.get(family, "")
    return _locate(sdk, family, f"{prefix}{name}.py") if prefix else ""


# ── Rendering helpers ─────────────────────────────────────────────────

def _first_line(text, limit=140):
    """The first sentence of a description, for a roster line."""
    flat = " ".join((text or "").split())
    if not flat:
        return "(no description)"
    cut = flat.split(". ")[0].rstrip(".")
    if len(cut) > limit:
        cut = cut[:limit].rsplit(" ", 1)[0] + "…"
    return cut


def _roster(kind, rows, render):
    """A listing plus the one line that says how to get the detail."""
    if not rows:
        return f"No {kind} on this machine."
    lines = [f"# {kind} — {len(rows)}", ""]
    lines += [render(row) for row in rows]
    lines += ["", f'Detail: info("{kind}", "<name>")']
    return "\n".join(lines)


def _pick(rows, name, kind):
    """One row by name, or a message naming the near misses."""
    for row in rows:
        if row.get("name") == name:
            return row, ""
    names = [row.get("name", "") for row in rows]
    close = difflib.get_close_matches(name, names, n=4, cutoff=0.4)
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    return None, (f"No {kind[:-1]} named '{name}'.{hint} "
                  f'Call info("{kind}") for the full list.')


def _arguments(schema):
    """A JSON-schema parameter block as one readable line per argument."""
    properties = (schema or {}).get("properties")
    if not isinstance(properties, dict) or not properties:
        return "  (no arguments)"
    required = set((schema or {}).get("required") or [])
    lines = []
    for name, spec in properties.items():
        spec = spec if isinstance(spec, dict) else {}
        mark = " (required)" if name in required else ""
        lines.append(f"  - {name}: {spec.get('type', '?')}{mark}")
        note = " ".join((spec.get("description") or "").split())
        if note:
            lines.append(f"      {note}")
        if spec.get("enum"):
            lines.append(f"      one of: {', '.join(str(v) for v in spec['enum'])}")
    return "\n".join(lines)


def _settings(rows):
    """Editable config settings, already redacted by the kernel."""
    if not rows:
        return ""
    lines = ["", "Settings (change with /config):"]
    for row in rows:
        current = row.get("current")
        shown = "unset" if current in (None, "") else str(current)
        lines.append(f"  - {row.get('key')} = {shown}"
                     f"  — {_first_line(row.get('description'), 100)}")
    return "\n".join(lines)


class Info(BaseTool):
    """One lookup for documentation and live catalogs alike."""

    name = "info"
    description = (
        "Look up how Second Brain works and what is installed on this machine. "
        "Pass a kind alone to list what exists; add a name for one item's "
        "detail. Read 'sdk' and 'templates' before writing any script or "
        "plugin, and check the relevant kind before concluding something is "
        "missing. Read-only, and changes nothing."
    )
    parameters = {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                # Spelled out rather than built from KINDS: the kernel reads
                # this without importing the file, so one computed value
                # discards the whole *entire* declaration and the tool loads
                # with no schema at all. Keep this list and KINDS in step.
                "enum": ["sdk", "docs", "templates", "tools", "commands",
                         "services", "tasks", "frontends", "scripts",
                         "database"],
                "description": "What to look up.",
            },
            "name": {
                "type": "string",
                "description": (
                    "One item from that kind's list. For 'sdk' and 'docs' this "
                    "is a section key, an SDK namespace such as 'fs' or 'db', "
                    "or words from a heading. Omit to see what is available."
                ),
            },
        },
        "required": ["kind"],
    }

    # A plain string, so it is read from this file without a box call. This is
    # the pointer that the shortened system prompt leans on; it lists the kinds
    # because an agent that cannot see the menu never opens it. It leaves with
    # the package, which is the reason it lives here and not in the kernel's
    # static prompt.
    agent_prompt = (
        "## Looking things up\n"
        "`info(kind)` lists what exists; `info(kind, name)` returns the detail. "
        "Use it instead of guessing, and before saying something is not "
        "installed.\n"
        "- `sdk` — the SDK reference by section; read before writing any code\n"
        "- `templates` — the authoring contract for one plugin family\n"
        "- `docs` — README and everything in docs/\n"
        "- `tools` — what you can call, and each one's arguments\n"
        "- `commands` — slash commands; these are the user's to run, not yours\n"
        "- `services` — background capabilities, and whether each is loaded\n"
        "- `tasks` — the pipeline, task status and what is scheduled\n"
        "- `frontends` — transports, and whether each is enabled\n"
        "- `scripts` — scripts already written, by path\n"
        "- `database` — tables and their schemas"
    )

    def run(self, sdk, kind, name=""):
        """Dispatch to one kind. Everything answers with markdown text."""
        kind = (kind or "").strip().lower()
        name = (name or "").strip()
        if kind not in KINDS:
            return sdk.fail(f"Unknown kind '{kind}'. One of: {', '.join(KINDS)}.")
        return getattr(self, f"_{kind}")(sdk, name)

    # ── Documentation ────────────────────────────────────────────────

    def _sdk(self, sdk, name):
        """The SDK reference, by section."""
        source = "docs/SDK.md"
        chapters = _chapters(self._read_doc(sdk, source))
        if not name:
            return (_toc(chapters, "docs/SDK.md") + "\n\n"
                    'One section: info("sdk", "<key>"). A namespace works too '
                    '— info("sdk", "fs") finds where sdk.fs.* is written up.')
        matches = _match(chapters, name)
        if not matches:
            return (f"No SDK section matches '{name}'.\n\n"
                    + _toc(chapters, "docs/SDK.md"))
        if len(matches) > 1:
            return _ambiguous(matches, "sdk", name)
        return _render_chapter(matches[0], source)

    def _docs(self, sdk, name):
        """README and docs/, by file then by section."""
        files = self._doc_files(sdk)
        if not name:
            lines = ["# docs — README.md and docs/", ""]
            for relative, size in files:
                lines.append(f"  {relative}"
                             + (f"  [{size // 1024} KB]" if size >= 1024 else ""))
            lines += ["", 'A file\'s sections: info("docs", "permissions"). '
                          'A section directly: info("docs", "<words from its '
                          'heading>").']
            return "\n".join(lines)

        # A filename match answers with that file's contents page rather than
        # the file: PERMISSIONS_MAP.md alone is 22 KB, and handing back the
        # whole of it is the cost this tool exists to avoid.
        wanted = name.lower()
        for relative, _ in files:
            stem = relative.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
            if wanted == stem or wanted in stem or stem in wanted:
                chapters = _chapters(self._read_doc(sdk, relative))
                if not chapters:
                    return _render_chapter(
                        {"body": self._read_doc(sdk, relative)}, relative)
                return _toc(chapters, relative)

        matched = []
        for relative, _ in files:
            for chapter in _match(_chapters(self._read_doc(sdk, relative)), name):
                chapter["source"] = relative
                matched.append(chapter)
        if not matched:
            listing = "\n".join(f"  {relative}" for relative, _ in files)
            return (f"Nothing in the docs matches '{name}'. Files:\n{listing}")
        if len(matched) > 1:
            lines = [f"{len(matched)} sections match '{name}':", ""]
            for chapter in matched:
                lines.append(f"  {chapter['source']}  #{chapter['slug']}"
                             f"  — {chapter['title']}")
            lines += ["", 'Ask for one: info("docs", "<key>").']
            return "\n".join(lines)
        return _render_chapter(matched[0], matched[0]["source"])

    def _templates(self, sdk, name):
        """The authoring contract for one plugin family, verbatim."""
        root = sdk.paths.get("project")
        listing = sdk.fs.list(_join(root, "templates"), pattern="*_template.py")
        stems = sorted(
            entry.rsplit("/", 1)[-1].rsplit("\\", 1)[-1][:-len("_template.py")]
            for entry in self._entries(listing)
            if str(entry).endswith("_template.py"))
        if not name:
            lines = ["# templates", ""]
            lines += [f"  {stem}" for stem in stems]
            lines += ["", 'One template, in full: info("templates", "tool"). '
                          "Read it before writing that kind of file — code "
                          "written from memory does not load."]
            return "\n".join(lines)

        wanted = name.lower().replace("_template", "").rstrip("s")
        for stem in stems:
            if stem.lower().rstrip("s") == wanted or wanted in stem.lower():
                relative = f"templates/{stem}_template.py"
                text = sdk.fs.read(_join(root, relative))
                if len(text) > MAX_CHARS:
                    text = text[:MAX_CHARS] + f"\n# [truncated — read {relative}]"
                return f"```python\n{text}\n```\n\n---\nFrom {relative}"
        return (f"No template for '{name}'. Available: {', '.join(stems)}.")

    def _read_doc(self, sdk, relative):
        """One documentation file, by repo-relative path."""
        return sdk.fs.read(_join(sdk.paths.get("project"), relative))

    def _doc_files(self, sdk):
        """``(relative_path, size)`` for README.md and every docs/*.md."""
        root = sdk.paths.get("project")
        found = []
        listing = sdk.fs.list(_join(root, "docs"), pattern="*.md", details=True)
        for entry in self._entries(listing):
            if isinstance(entry, dict):
                path = str(entry.get("path") or entry.get("name") or "")
                size = int(entry.get("size") or 0)
            else:
                path, size = str(entry), 0
            leaf = path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
            if leaf.endswith(".md"):
                found.append((f"docs/{leaf}", size))
        found.sort()
        return [("README.md", 0)] + found

    @staticmethod
    def _entries(listing):
        """The rows of an fs.list answer, whichever shape it came back in.

        Passing ``details`` or a walking argument switches the answer from a
        bare list to ``{"root", "entries", ...}``. Both shapes are read here so
        a caller does not have to remember which one it asked for.
        """
        if isinstance(listing, dict):
            return listing.get("entries") or []
        return listing or []

    # ── Live catalogs ────────────────────────────────────────────────

    def _tools(self, sdk, name):
        """What the current scope can call."""
        rows = sdk.tools.list(details=True)
        if not name:
            return _roster("tools", rows, lambda row:
                           f"  {row.get('name')}  — "
                           f"{_first_line(row.get('description'))}")
        row, problem = _pick(rows, name, "tools")
        if row is None:
            return problem
        path = _plugin_path(sdk, "tools", name)
        parts = [f"# {row.get('name')}", "",
                 " ".join((row.get("description") or "").split()), "",
                 "Arguments:", _arguments(row.get("parameters"))]
        if row.get("requires_services"):
            parts += ["", "Needs services: "
                      + ", ".join(row["requires_services"])]
        if path:
            parts += ["", f"Source: {path}"]
        parts.append(_settings(row.get("config_settings")))
        return "\n".join(part for part in parts if part is not None)

    def _commands(self, sdk, name):
        """Slash commands. The user runs these; the agent cannot."""
        rows = sdk.commands.list(details=True, visible=True)
        if not name:
            listing = _roster("commands", rows, lambda row:
                              f"  /{row.get('name')}  — "
                              f"{_first_line(row.get('description'))}")
            return (listing + "\n\nThese are user-invoked. Writing '/name' in "
                    "a reply sends text and executes nothing — refer the user "
                    "to the command instead.")
        row, problem = _pick(rows, name.lstrip("/"), "commands")
        if row is None:
            return problem
        parts = [f"# /{row.get('name')}", "",
                 " ".join((row.get("description") or "").split()), "",
                 f"Category: {row.get('category') or 'Other'}"]
        steps = row.get("form") or []
        if steps:
            parts += ["", "Asks the user for:"]
            parts += [f"  - {step.get('name')}"
                      f"{'' if step.get('required') else ' (optional)'}"
                      for step in steps]
        path = _plugin_path(sdk, "commands", row.get("name"))
        if path:
            parts += ["", f"Source: {path}"]
        parts += ["", "The user runs this, not you."]
        return "\n".join(parts)

    def _services(self, sdk, name):
        """Background capabilities and whether each is loaded."""
        rows = sdk.services.list(details=True)
        if not name:
            return _roster("services", rows, lambda row:
                           f"  {row.get('name')} "
                           f"[{'loaded' if row.get('loaded') else 'unloaded'}]"
                           f"  — {_first_line(row.get('description'))}")
        row, problem = _pick(rows, name, "services")
        if row is None:
            return problem
        parts = [f"# {row.get('name')}", "",
                 " ".join((row.get("description") or "").split()), "",
                 f"Loaded: {bool(row.get('loaded'))}",
                 f"Lifecycle: {row.get('lifecycle') or 'unknown'}"]
        path = _plugin_path(sdk, "services", row.get("name"))
        if path:
            parts += ["", f"Source: {path}"]
        parts.append(_settings(row.get("config_settings")))
        return "\n".join(part for part in parts if part is not None)

    def _tasks(self, sdk, name):
        """The pipeline, task status, and what is scheduled."""
        rows = sdk.tasks.list(details=True)
        if not name:
            if not rows:
                return ("No tasks installed. The pipeline substrate boots and "
                        "idles until a pipeline package is installed.")
            lines = [f"# tasks — {len(rows)}", ""]
            for row in rows:
                counts = row.get("counts") or {}
                # Spelled out and zeroes dropped. Initials collided — two of
                # the count keys begin with "p", so every task read
                # "D:6 F:0 P:0 P:0" and the two P's meant different things.
                state = " ".join(f"{key}={value}"
                                 for key, value in sorted(counts.items())
                                 if value)
                flags = " [paused]" if row.get("paused") else ""
                if row.get("schedule_count"):
                    flags += f" [{row['schedule_count']} scheduled]"
                lines.append(f"  {row.get('name')} ({row.get('trigger')})"
                             f"{flags}  {state}"
                             f"  — {_first_line(row.get('description'))}")
            graph = sdk.tasks.graph()
            if graph:
                lines += ["", str(graph)]
            lines += ["", 'Detail: info("tasks", "<name>")']
            return "\n".join(lines)
        row, problem = _pick(rows, name, "tasks")
        if row is None:
            return problem
        parts = [f"# {row.get('name')}", "",
                 " ".join((row.get("description") or "").split()), "",
                 f"Trigger: {row.get('trigger')}",
                 f"Runs: {row.get('counts') or {}}",
                 f"Paused: {bool(row.get('paused'))}",
                 f"Scheduled jobs: {row.get('schedule_count') or 0}"]
        if row.get("trigger_channels"):
            parts.append("Channels: " + ", ".join(row["trigger_channels"]))
        if row.get("requires_services"):
            parts.append("Needs services: " + ", ".join(row["requires_services"]))
        path = _plugin_path(sdk, "tasks", row.get("name"))
        if path:
            parts += ["", f"Source: {path}"]
        parts.append(_settings(row.get("config_settings")))
        return "\n".join(part for part in parts if part is not None)

    def _frontends(self, sdk, name):
        """Transports, and whether each is enabled."""
        rows = sdk.plugins.list(source="registered", category="frontends",
                                details=True)
        if not name:
            return _roster("frontends", rows, lambda row:
                           f"  {row.get('name')} "
                           f"[{'running' if row.get('loaded') else 'stopped'}"
                           f"{'' if row.get('available') else ', not installed'}]"
                           f"  — {_first_line(row.get('description'))}")
        row, problem = _pick(rows, name, "frontends")
        if row is None:
            return problem
        parts = [f"# {row.get('name')}", "",
                 " ".join((row.get("description") or "").split()), "",
                 f"Installed: {bool(row.get('available'))}",
                 f"Running: {bool(row.get('loaded'))}"]
        path = _plugin_path(sdk, "frontends", row.get("name"))
        if path:
            parts += ["", f"Source: {path}"]
        parts.append(_settings(row.get("config_settings")))
        return "\n".join(part for part in parts if part is not None)

    def _scripts(self, sdk, name):
        """Scripts already written, across every tree.

        A script has no prefix, no base class and no entry point — the
        directory is the whole declaration — so there is no registry to ask.
        The trees are walked instead.
        """
        found = []
        for tree in TREES:
            try:
                root = sdk.paths.get(tree)
            except sdk.Failed:
                continue
            directory = _join(root, "scripts")
            try:
                if not sdk.fs.exists(directory):
                    continue
                listing = sdk.fs.list(directory, pattern="*.py")
            except sdk.Failed:
                continue
            for entry in self._entries(listing):
                leaf = str(entry).rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
                if leaf.endswith(".py") and not leaf.startswith("_"):
                    found.append((leaf[:-3], tree, _join(directory, leaf)))
        if not found:
            return ("No scripts yet. Write one to "
                    f"{_join(sdk.paths.get('workspace'), 'scripts')} and run it "
                    'with run_script — start from info("templates", "script").')
        if not name:
            lines = [f"# scripts — {len(found)}", ""]
            lines += [f"  {stem} ({tree})  {path}" for stem, tree, path in found]
            lines += ["", 'Source of one: info("scripts", "<name>")']
            return "\n".join(lines)
        for stem, tree, path in found:
            if stem.lower() == name.lower().replace(".py", ""):
                text = sdk.fs.read(path)
                if len(text) > MAX_CHARS:
                    text = text[:MAX_CHARS] + f"\n# [truncated — read {path}]"
                return f"```python\n{text}\n```\n\n---\nFrom {path}"
        stems = [stem for stem, _, _ in found]
        return f"No script named '{name}'. Available: {', '.join(stems)}."

    def _database(self, sdk, name):
        """Tables and their schemas.

        ``sqlite_master`` is readable on purpose — the kernel's own error text
        recommends it for schema questions. It is the *write* path that refuses
        the schema table, which is not what this asks for.
        """
        if not name:
            rows = sdk.db.query(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name")
            names = [row["name"] for row in rows]
            lines = [f"# database — {len(names)} tables", ""]
            lines += [f"  {table}" for table in names]
            lines += [
                "", 'Schema of one: info("database", "conversations")', "",
                "Two rules the kernel enforces, so ignoring them is a refusal "
                "rather than a result:",
                "  1. Read rows you own through a `my_` name — "
                "`SELECT * FROM my_conversations`, never `conversations`. "
                "Same for `my_action_ledger`.",
                "  2. Kernel tables are not writable through SQL. A table your "
                "own plugin created with CREATE stays freely writable.",
            ]
            return "\n".join(lines)

        rows = sdk.db.query(
            "SELECT name, sql FROM sqlite_master WHERE type = 'table' "
            "AND name = ?", [name.replace("my_", "", 1)])
        if not rows:
            everything = sdk.db.query(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name")
            names = [row["name"] for row in everything]
            close = difflib.get_close_matches(name, names, n=4, cutoff=0.4)
            hint = f" Did you mean: {', '.join(close)}?" if close else ""
            return f"No table named '{name}'.{hint}"

        parts = [f"# {rows[0]['name']}", "",
                 f"```sql\n{rows[0].get('sql') or '(no schema recorded)'}\n```"]
        indexes = sdk.db.query(
            "SELECT name, sql FROM sqlite_master WHERE type = 'index' "
            "AND tbl_name = ? AND sql IS NOT NULL ORDER BY name",
            [rows[0]["name"]])
        if indexes:
            parts += ["", "Indexes:"]
            parts += [f"  {row['name']}" for row in indexes]
        if rows[0]["name"] in ("conversations", "action_ledger"):
            parts += ["", f"Read this one as `my_{rows[0]['name']}` — the base "
                      "table holds every user's rows and is refused."]
        if rows[0]["name"] == "conversation_messages":
            parts += ["", "`role` does not say who wrote a row: the kernel "
                      "writes `role='user'` rows nobody typed, and each "
                      "carries a non-empty `author`. For what the user "
                      "actually said, add `AND COALESCE(author, '') = ''`."]
        return "\n".join(parts)
