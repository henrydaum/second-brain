"""Every string the agent can see, in one file, with its source.

Agent-facing text is spread across three populations that are edited in
different places and shipped on different schedules: the kernel's own prompt
and loop messages, the assembled prompt as a model actually receives it, and
the ``name``/``description``/``parameters``/``agent_prompt`` declarations that
each store plugin carries. Nothing had ever shown them together, which is why
editing the agent's voice meant grepping for a half-remembered sentence.

Two sources, and both are needed for different reasons.

**Live** answers what the model is being told *right now* on this machine —
the three blocks verbatim, a map saying which part of them came from where,
and the schema of every registered tool. It is the ground truth and it is also
incomplete by construction: it can only show what is installed here.

**AST** answers what could be said at all. It reads the store branch without
installing anything and never imports what it reads, so a tool nobody has
installed still shows up. It is the only view that covers the ~43 KB of store
declarations, which is where most agent-facing text actually lives.

The two halves differ on a *dynamic* ``agent_prompt`` — a method whose text
depends on live state. The live half renders it for real, by calling into the
plugin's box exactly as the prompt builder does; that is why this script wires
a runtime (see ``live_view``), and without one every such contribution comes
back empty and vanishes from the dump with nothing said. The AST half cannot,
so it marks the declaration and names the refresh cue instead.

A dev script rather than a command or an SDK script, for two reasons that are
not preference. A sandboxed script runs with ``sandbox/`` as its cwd and
cannot import the kernel at all, so ``agent.system_prompt`` is out of reach. A
command only sees what is installed on the machine running it, which excludes
exactly the store text this exists to show.

    python dev/dump_agent_text.py                 # live + store, to agent_text.txt
    python dev/dump_agent_text.py --no-live       # AST only; boots nothing
    python dev/dump_agent_text.py --out x.txt
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import prompt_cues  # noqa: E402 — needs the path above

#: Kernel modules that put words in front of the agent. Curated rather than
#: globbed: "does this string reach a model" is not decidable from the source,
#: and a whole-tree scan drowns the twenty sentences that matter in several
#: hundred log lines. Adding a module here is the one maintenance cost this
#: script has, and a missing one shows up as text you cannot find in the dump.
KERNEL_SITES = (
    ("agent/system_prompt.py", "prompt assembly, section headers, status lines"),
    ("runtime/conversation_loop.py", "tool-budget, empty-response, compaction, cancellation"),
    ("runtime/subagents.py", "background-agent note, timeout and failure reports"),
    ("runtime/agent_scope.py", "profile scope notes"),
    ("runtime/notifications.py", "notification prompt block"),
    # ``runtime/hooks.py`` is deliberately absent: every string in it is a log
    # line. A doorman's note is authored by the *plugin* and reaches the agent
    # through conversation_loop's gate, which is listed above.
    ("runtime/dispatch.py", "the [SYSTEM NOTE] wrapper for a user's slash command"),
    ("agent/tool_registry.py", "tool dispatch failures the model reads"),
    ("bundled/services/service_compactor.py", "the compaction prompt"),
    # Refusals reach the model, not only the ledger: interpreter._settle hands
    # ``Decision.reason`` back as the failure a plugin sees. So policy's
    # sentences are agent-facing even though they are written as audit text.
    # ``approval.py`` is deliberately absent — that is the dialog's ``say``
    # half, which only a person ever reads, and it belongs in the user dump.
    ("sandbox/policy.py", "refusal reasons handed back to the model"),
)

#: Declarations a plugin carries that the agent reads. ``name`` is included
#: because a tool's name is the first thing a model matches on, and renaming
#: one is as much an authoring decision as rewriting its description.
PLUGIN_DECLS = ("name", "description", "parameters", "agent_prompt",
                "agent_prompt_refresh", "llm_summary")

#: Store families worth scanning. ``parsers/`` is omitted deliberately — a
#: parser has no agent-facing surface at all; it is reached by extension.
STORE_FAMILIES = ("tools", "tasks", "services", "commands", "frontends")

#: Below this a literal is a key, a separator or a format fragment rather than
#: something anybody would translate or rewrite.
MIN_PROSE = 25

RULE = "=" * 78
THIN = "-" * 78


# ── Provenance ────────────────────────────────────────────────────────

def _git(*args, cwd=ROOT) -> str:
    """One git command's stdout, or "" — never raises, this is a dump."""
    try:
        done = subprocess.run(["git", "-C", str(cwd), *args],
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                              text=True, encoding="utf-8", check=False)
        return done.stdout.strip() if done.returncode == 0 else ""
    except OSError:
        return ""


def store_worktree() -> Path | None:
    """A checked-out ``store`` worktree, if the clone has one.

    Preferred over ``git show store:`` for the reason the test suite prefers
    it: a worktree holds what is being *edited*, and the whole point of this
    dump is to look at text before committing a change to it.
    """
    listing = _git("worktree", "list", "--porcelain")
    path = None
    for line in listing.splitlines():
        if line.startswith("worktree "):
            path = Path(line[len("worktree "):].strip())
        elif line.strip() == "branch refs/heads/store" and path is not None:
            return path
    return None


def store_files() -> list[tuple[str, str]]:
    """``(relative_path, source)`` for every store plugin, worktree or ref.

    Falling back to the ref matters: a clone with no store worktree is the
    normal case for anyone but the person who set this repo up, and a dump
    that silently omitted the largest population would be worse than none.
    """
    found: list[tuple[str, str]] = []
    worktree = store_worktree()
    if worktree is not None:
        for family in STORE_FAMILIES:
            directory = worktree / family
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("*.py")):
                found.append((f"{family}/{path.name}",
                              path.read_text(encoding="utf-8", errors="replace")))
        if found:
            return found

    for ref in ("store", "origin/store"):
        listing = _git("ls-tree", "-r", "--name-only", ref)
        if not listing:
            continue
        for relative in sorted(listing.splitlines()):
            family = relative.split("/")[0]
            if family not in STORE_FAMILIES or not relative.endswith(".py"):
                continue
            if relative.count("/") != 1:  # helpers/ hold no declarations
                continue
            source = _git("show", f"{ref}:{relative}")
            if source:
                found.append((relative, source))
        if found:
            return found
    return found


# ── AST extraction ────────────────────────────────────────────────────

def _docstrings(tree: ast.AST) -> set[str]:
    return {ast.get_docstring(node, clean=False)
            for node in ast.walk(tree)
            if isinstance(node, (ast.Module, ast.FunctionDef,
                                 ast.AsyncFunctionDef, ast.ClassDef))}


def _logged_lines(tree: ast.AST) -> set[int]:
    """Line numbers of strings that are arguments to a logging call.

    The single most effective filter available. A log message and a message to
    the model are both prose in the same file, and nothing in the string tells
    them apart — but the call they sit inside does, and roughly two thirds of
    the long literals in ``conversation_loop.py`` are log lines.
    """
    lines: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        name = getattr(target, "attr", None) or getattr(target, "id", None)
        root = getattr(getattr(target, "value", None), "id", "")
        if root in ("logger", "logging", "log") or name in ("debug", "warning",
                                                            "exception"):
            for child in ast.walk(node):
                if isinstance(child, ast.Constant) and isinstance(child.value, str):
                    lines.add(child.lineno)
    return lines


#: Keyword arguments whose value only a *person* ever reads. ``Decision``
#: carries two strings on purpose: ``reason`` goes to the ledger and is handed
#: to the model as the refusal, while ``say`` is the human half the approval
#: dialog prints. Collecting both put "Deleted rows are not recoverable." in a
#: dump of what the agent sees, which no model is ever shown.
HUMAN_KWARGS = ("say",)


def _kwarg_lines(tree: ast.AST, keywords: tuple[str, ...]) -> set[int]:
    """Lines of string constants passed as one of ``keywords``."""
    lines: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg not in keywords:
                continue
            for child in ast.walk(keyword.value):
                if isinstance(child, ast.Constant) and isinstance(child.value, str):
                    lines.add(child.lineno)
    return lines


def _enclosing(tree: ast.AST) -> dict[int, str]:
    """Line number → enclosing function, so an entry says where it lives."""
    where: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", node.lineno)
            for line in range(node.lineno, end + 1):
                where.setdefault(line, node.name)
    return where


def _declared_cue(scope) -> str:
    """The ``agent_prompt_refresh`` sibling of a dynamic prompt, if any.

    The one part of a dynamic contribution that is readable without running
    anything, and the part that says how much of the prompt it churns.
    """
    for node in getattr(scope, "body", []):
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(t, ast.Name) and t.id == "agent_prompt_refresh"
               for t in node.targets):
            value = _evaluate(node.value)
            if isinstance(value, str) and value:
                return value
    return "write (the default)"


def kernel_strings(relative: str) -> list[tuple[int, str, str]]:
    """``(line, enclosing_function, text)`` for one kernel module."""
    path = ROOT / relative
    if not path.is_file():
        return []
    source = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    docs, logged, where = _docstrings(tree), _logged_lines(tree), _enclosing(tree)
    human = _kwarg_lines(tree, HUMAN_KWARGS)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        text = node.value
        if (text in docs or node.lineno in logged or node.lineno in human
                or len(text) < MIN_PROSE
                or " " not in text.strip()
                or text.strip().upper().startswith(("SELECT ", "INSERT ",
                                                    "UPDATE ", "DELETE ",
                                                    "CREATE "))):
            continue
        found.append((node.lineno, where.get(node.lineno, "<module>"), text))
    return sorted(found)


def plugin_declarations(source: str) -> list[tuple[int, str, str]]:
    """``(line, declaration, rendered)`` for one plugin file.

    Reads rather than imports, which is the same rule the package manager and
    the bridge follow — a store file must never be executed to be inspected.

    Scoped to module level and class bodies, because that is what a
    *declaration* is. Walking the whole tree picked up every local variable
    that happened to be called ``name`` — ``name = (raw or "").strip()`` inside
    a validator function is not agent-facing text, and ten of them were
    padding the dump with entries nobody could act on.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    scopes: list[ast.AST] = [tree]
    scopes += [node for node in tree.body if isinstance(node, ast.ClassDef)]
    found = []
    for scope in scopes:
        for node in scope.body:
            if isinstance(node, ast.Assign):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
                for name in names:
                    if name not in PLUGIN_DECLS:
                        continue
                    found.append((node.lineno, name,
                                  _render(_evaluate(node.value))))
            elif (isinstance(node, ast.FunctionDef)
                  and node.name == "agent_prompt"):
                # The dynamic shape. Its text depends on live state, so the
                # only honest thing to print is that it exists, where, and how
                # often it moves — the cue is the one part of a dynamic
                # contribution that *is* readable without running anything.
                cue = _declared_cue(scope)
                found.append((node.lineno, "agent_prompt",
                              f"<dynamic, refreshes on {cue}; see the "
                              "assembled prompt above for the installed "
                              "result>"))
    return sorted(found)


def _evaluate(node: ast.AST):
    """A declaration's value, structure preserved, non-literals left as source.

    ``literal_eval`` is all-or-nothing, and one computed leaf is enough to
    lose the whole thing: ``tool_ask_question`` builds its enum with
    ``sorted(TYPES)``, which turned a readable seven-argument schema into a
    single unreadable line. Descending by hand keeps every authored
    description and shows the one computed leaf as the expression it is.
    """
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Dict):
        return {_evaluate(k): _evaluate(v)
                for k, v in zip(node.keys, node.values) if k is not None}
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return [_evaluate(item) for item in node.elts]
    if isinstance(node, ast.JoinedStr):
        # An f-string's literal parts are the authored half; the holes are
        # runtime values nothing here can know.
        return "".join(part.value if isinstance(part, ast.Constant)
                       else "{…}" for part in node.values)
    return f"<{ast.unparse(node)}>"


def _render(value) -> str:
    """A declaration as readable text rather than as a repr."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return _render_schema(value)
    return repr(value)


def _render_schema(schema: dict, indent: str = "") -> str:
    """A JSON-schema parameter block as one line per argument.

    The raw dict is what the model gets, but it is unreadable at a glance and
    the argument *descriptions* inside it are authored prose that belongs in
    this dump as much as the tool description does.
    """
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return repr(schema)
    required = set(schema.get("required") or [])
    lines = []
    for name, spec in properties.items():
        spec = spec if isinstance(spec, dict) else {}
        kind = spec.get("type", "?")
        mark = " (required)" if name in required else ""
        lines.append(f"{indent}  - {name}: {kind}{mark}")
        note = (spec.get("description") or "").strip()
        if note:
            lines.append(f"{indent}      {note}")
    return "\n".join(lines) if lines else "(no arguments)"


# ── Live introspection ────────────────────────────────────────────────

def live_view() -> dict:
    """Boot discovery far enough to build the real prompt, or explain why not.

    Mirrors ``runtime.bootstrap`` rather than inventing a shortcut, so what
    lands in the file is the prompt this machine actually sends — profile
    scope, active model and all. Nothing is written: no frontend starts, no
    conversation opens, and services are adapted rather than loaded.

    **Mirroring it includes the context factory**, and for a long time this did
    not. A sandboxed plugin whose ``agent_prompt`` is a *method* is answered
    inside its box, and the SDK it gets there is built by
    ``runtime.context.kernel_context`` — which ``main.pyw`` installs via
    ``Sandbox.bind_context`` and which needs a runtime for ``sdk.session.get``
    to answer at all. Without it every session-reading prompt method failed
    with ``the runtime is not available in this kernel``, the bridge dropped
    the result, and the dump showed those plugins contributing *nothing*.

    That is the exact failure this file exists to make visible, arrived at from
    the other side: text you cannot find in the dump because the dump could not
    produce it. The runtime is constructed and never driven — no conversation
    is opened and nothing is written — which is enough for the SDK to answer.
    """
    import state_machine  # noqa: F401 — settles a circular import before use

    from config import config_manager
    from pipeline.database import Database
    from pipeline.orchestrator import Orchestrator
    from agent.tool_registry import ToolRegistry
    from agent.system_prompt import build_prompt_sections
    from plugins.plugin_discovery import (discover_commands, discover_services,
                                          discover_tasks, discover_tools)
    from plugins.command_registry import CommandRegistry
    from runtime.agent_scope import load_scope, resolve_agent_llm, scoped_registry
    from runtime.context import kernel_context, set_kernel_parts
    from runtime.conversation_runtime import ConversationRuntime
    from sandbox.bridge import get_sandbox

    config = config_manager.load()
    db = Database(config["db_path"])
    services = discover_services(config)
    orchestrator = Orchestrator(db, config, services)
    discover_tasks(orchestrator)
    tools = ToolRegistry(db, config, services)
    tools.orchestrator = orchestrator
    discover_tools(tools)
    commands = CommandRegistry()
    discover_commands(commands)

    # The wiring main.pyw and runtime.bootstrap do, in the same order: the
    # parts first, then the factory. A method-shaped agent_prompt is answered
    # inside a box, and this is what gives that box an SDK that can answer.
    set_kernel_parts(db=db, config=config, root_dir=ROOT,
                     services=services, orchestrator=orchestrator,
                     tool_registry=tools, command_registry=commands,
                     runtime=ConversationRuntime(db=db, services=services,
                                                 config=config))
    get_sandbox().bind_context(kernel_context)

    # The same two steps bootstrap takes, in the same order: a scope with
    # nothing in it is passed as None, and a scope with a tool filter narrows
    # the registry the prompt is built from — skip that and the dump lists
    # tools the profile actually hides.
    profile = config.get("active_agent_profile") or "default"
    scope = load_scope(profile, config)
    scope = scope if (scope.has_tool_filter or scope.prompt_suffix) else None
    for_prompt = scoped_registry(tools, scope, db=db) if scope else tools
    sections: list = []
    messages = build_prompt_sections(
        db, orchestrator, for_prompt, services, scope=scope,
        profile_name=profile, commands=commands, config=config,
        active_llm=resolve_agent_llm(profile, config, services),
        sections_out=sections)
    return {
        "profile": profile,
        "messages": messages,
        "sections": sections,
        "schemas": (for_prompt.get_all_schemas()
                    if hasattr(for_prompt, "get_all_schemas") else []),
        "tools": sorted(getattr(for_prompt, "tools", {})),
        "commands": sorted(c.name for c in commands.visible_commands()),
        "authored_by": _authored_by(for_prompt, services, orchestrator,
                                    commands, config, profile, scope),
    }


def _authored_by(registry, services, orchestrator, commands, config,
                 profile, scope) -> dict:
    """``heading -> (plugin, cue)`` for every contribution in the prompt.

    The dump could always show the text and never say whose it was, which is
    the one question worth asking before editing a line of it: a section the
    kernel writes is yours to rewrite, and a section a plugin contributes is
    not — it moves when that package is updated, and editing the prompt to
    match it is editing the wrong file.

    Each plugin is asked the same way ``_collect`` asks, so a heading here is
    a heading that really lands. Anything the map does not claim is kernel.
    """
    from agent.system_prompt import PromptContext, _in_scope

    context = PromptContext(
        db=None, services=services or {}, orchestrator=orchestrator,
        config=config or {}, scope=scope, profile_name=profile)
    found = {}
    for plugin in _in_scope(registry, services, orchestrator, commands,
                            None, None):
        raw = getattr(plugin, "agent_prompt", "")
        try:
            text = ((raw(context) if callable(raw) else raw) or "").strip()
        except Exception:
            text = ""
        if not text:
            continue
        # *Every* heading, not just the first. A contribution may carry
        # several — run_script writes "## Scripts — reach for these first"
        # and "## Scripts you have" — and recording only the first attributed
        # the rest to the kernel, which is the one thing this map exists to
        # get right. An unheaded contribution is claimed by its first line.
        headings = [line[3:].strip() for line in text.splitlines()
                    if line.startswith("## ")]
        for heading in headings or [text.splitlines()[0].strip()]:
            found[heading] = (getattr(plugin, "name", "?"),
                              prompt_cues.of(plugin))
    return found


#: The one block that is a file rather than an assembly. Reported whole,
#: because it is edited whole — splitting it into paragraphs would list
#: twenty rows nobody addresses individually.
FILE_BLOCK = "STATIC SYSTEM PROMPT"


def section_map(sections, authored_by) -> list:
    """``(block, heading, chars, source)`` for the assembled prompt.

    Reads the section list ``build_prompt_sections`` filled in rather than
    re-deriving it from the rendered text — see ``sections_out`` there for why
    parsing cannot get this right.

    A section is named by its ``##`` heading when it has one and by its first
    line when it does not, because the unheaded ones are real sections: the
    model line, the profile line and the clock. Those are exactly the parts
    with no name to write prose against, which is worth being able to see.
    """
    rows = []
    for block, text in sections:
        body = text.strip()
        if not body:
            continue
        if block == FILE_BLOCK:
            rows.append((block, "(the whole file)", len(body),
                         "kernel — system_prompt_static.md"))
            continue
        first = body.splitlines()[0].strip()
        heading = first[3:].strip() if first.startswith("## ") else first
        who = authored_by.get(heading)
        rows.append((block, heading, len(body),
                     f"{who[0]} ({who[1]})" if who else "kernel"))
    return rows


# ── Rendering ─────────────────────────────────────────────────────────

def _block(title: str, note: str = "") -> str:
    head = f"\n{RULE}\n{title}\n"
    if note:
        head += f"{note}\n"
    return head + RULE


def _quote(text: str, indent: str = "    ") -> str:
    """Indent a literal so prose containing blank lines stays one visual unit."""
    return "\n".join(indent + line if line else indent.rstrip()
                     for line in text.splitlines()) or f"{indent}(empty)"


def render(live: dict | None, store: list[tuple[str, str]]) -> str:
    out: list[str] = []
    add = out.append

    add(RULE)
    add("SECOND BRAIN — AGENT-FACING TEXT")
    add("Every string the agent can see, with the file and line it came from.")
    add(RULE)
    add(f"Generated:    {datetime.now().isoformat(timespec='seconds')}")
    add(f"Kernel:       {_git('rev-parse', '--short', 'HEAD')} "
        f"({_git('rev-parse', '--abbrev-ref', 'HEAD')})")
    worktree = store_worktree()
    add(f"Store:        {'worktree ' + str(worktree) if worktree else 'branch store'}")
    add(f"Store files:  {len(store)}")
    if live:
        add(f"Profile:      {live['profile']}")
        add(f"Installed:    {len(live['tools'])} tools, {len(live['commands'])} commands")
    else:
        add("Live view:    skipped (--no-live) — assembled prompt not shown")

    # ── 1. The assembled prompt ──
    if live:
        add(_block("1. THE ASSEMBLED PROMPT (live)",
                   "Verbatim, as the model receives it. Built by "
                   "agent/system_prompt.py:build_prompt_sections."))

        # The map first, because the question worth asking before editing any
        # of this is whose line it is. A kernel section is yours to rewrite; a
        # plugin's moves when that package is updated, and matching the prompt
        # to it edits the wrong file.
        rows = section_map(live["sections"], live["authored_by"])
        add(f"\n{THIN}\nSECTION MAP — where each part comes from\n{THIN}\n")
        head, sect = "block", "section"
        add(f"  {head:<21} {sect:<38} {'chars':>6}  authored by")
        add(f"  {'-' * 21} {'-' * 38} {'-' * 6}  {'-' * 26}")
        seen_block = None
        for block, heading, size, source in rows:
            shown = "" if block == seen_block else block
            seen_block = block
            add(f"  {shown:<21} {heading[:38]:<38} {size:>6}  {source}")
        kernel = sum(n for _, _, n, s in rows if s.startswith("kernel"))
        plugin = sum(n for _, _, n, s in rows if not s.startswith("kernel"))
        add(f"\n  kernel-authored: {kernel:,} chars    "
            f"plugin-contributed: {plugin:,} chars")
        add("  A cue in brackets is the plugin's declared agent_prompt_refresh"
            " — see prompt_cues.py.")

        for message in live["messages"]:
            add(f"\n{THIN}\nrole: {message['role']}  "
                f"({len(message['content'])} chars)\n{THIN}")
            add(message["content"])

        # ── 2. Tool schemas ──
        add(_block("2. TOOL SCHEMAS (live)",
                   "What each installed tool advertises to the model. "
                   "Authored in the plugin file."))
        for schema in live["schemas"]:
            fn = schema.get("function", schema)
            add(f"\n{THIN}\n{fn.get('name')}\n{THIN}")
            add(_quote((fn.get("description") or "").strip()))
            params = fn.get("parameters")
            if isinstance(params, dict):
                add("  arguments:")
                add(_render_schema(params, indent="  "))
        if not live["schemas"]:
            add("\n(no tools registered on this machine)")

    # ── 3. Kernel strings ──
    add(_block("3. KERNEL AGENT-FACING STRINGS (source)",
               "Authored text in the kernel modules that speak to the agent. "
               "Log lines are filtered out;\ndocstrings and SQL are excluded."))
    total = 0
    for relative, note in KERNEL_SITES:
        found = kernel_strings(relative)
        total += len(found)
        add(f"\n{THIN}\n{relative}  —  {note}\n{THIN}")
        if not found:
            add("    (nothing)")
            continue
        for line, function, text in found:
            add(f"\n  {relative}:{line}  in {function}()")
            add(_quote(text, "    "))
    add(f"\n  ({total} strings across {len(KERNEL_SITES)} modules)")

    # ── 4. Store declarations ──
    add(_block("4. STORE PLUGIN DECLARATIONS (source)",
               "Read from the store branch without installing anything — "
               "so uninstalled plugins\nappear too. This is where most "
               "agent-facing text lives."))
    if not store:
        add("\n(no store branch found — is `git worktree list` showing one?)")
    declared = 0
    for relative, source in store:
        found = plugin_declarations(source)
        if not found:
            continue
        declared += len(found)
        add(f"\n{THIN}\n{relative}\n{THIN}")
        for line, name, text in found:
            add(f"\n  {relative}:{line}  {name} =")
            add(_quote(text, "    "))
    add(f"\n  ({declared} declarations across {len(store)} files)")

    add(f"\n{RULE}\nEND\n{RULE}")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="agent_text.txt",
                        help="output file (default: agent_text.txt)")
    parser.add_argument("--no-live", action="store_true",
                        help="skip the live boot; AST sources only")
    args = parser.parse_args(argv)

    live = None
    if not args.no_live:
        try:
            live = live_view()
        except Exception as exc:  # noqa: BLE001 — a dump must still produce a file
            print(f"live view unavailable ({type(exc).__name__}: {exc}); "
                  f"continuing with source only", file=sys.stderr)

    text = render(live, store_files())
    out = Path(args.out)
    out.write_text(text, encoding="utf-8")
    print(f"{out}  ({len(text):,} chars, {len(text.splitlines()):,} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
