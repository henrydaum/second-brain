"""
Lexical Search tool.

BM25-ranked keyword search across the FTS5 ``lexical_index``. Searches all
indexed content (text chunks, OCR text, tabular text, and any future source
that writes into ``lexical_content``).

**The whole search happens in SQL, and five rows cross the boundary.** That is
not an optimization, it is what makes the tool expressible at all: ``db.query``
caps its answer, so a tool that ranked in Python would have to page the entire
corpus over the wire. FTS5 puts the index in the database and ``ORDER BY rank
LIMIT ?`` expresses the reduction, so the answer is what travels. The semantic
tool beside this one now works the same way, over ``vec_cosine``.
"""


# The index task is a real dependency, not a packaging convenience: this tool
# queries an FTS table that only ``task_lexical_index`` ever fills, so without
# it every search returns nothing and looks like an empty corpus.
dependencies_files = ['tools/helpers/SearchResult.py',
                      'tasks/task_lexical_index.py']
dependencies_pip = []
requests = ["db.query"]

import re

from guest.bases import BaseTool

from .helpers.SearchResult import SearchResult


class LexicalSearch(BaseTool):
    """Lexical search."""
    name = "lexical_search"
    description = (
        "Search for files by keyword using BM25-ranked full-text search. "
        "Searches across all indexed text content including text chunks, "
        "OCR results, and any other indexed sources.\n\n"
        "Supports FTS5 query syntax:\n"
        '- Phrases: "exact phrase"\n'
        "- Boolean: term1 AND term2, term1 OR term2, NOT term\n"
        "- Prefix: term*\n"
        "- Plain keywords: just type words and they are ANDed together"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query. Supports FTS5 syntax (phrases, AND/OR/NOT, prefix*).",
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum number of results to return. Default 5.",
                "default": 5,
            },
            "sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Filter by content source. Omit to search all sources. "
                    'Current values: "extracted", "ocr", "tabular". '
                    "Future values may include audio_transcript, video_subtitle, etc."
                ),
            },
            "folder": {
                "type": "string",
                "description": "Filter results to files under this folder path.",
            },
            "narration": {
                "type": "string",
                "description": (
                    "A few words on what you are looking for and why, shown to "
                    "the user beside the call. E.g. 'finding the notes about "
                    "the Berlin trip'."
                ),
            },
        },
        "required": ["query"],
    }
    requires_services = []

    def run(self, sdk, **kwargs):
        """Run lexical search."""
        query = (kwargs.get("query") or "").strip()
        top_k = max(1, int(kwargs.get("top_k") or 5))
        sources = kwargs.get("sources") or None
        # Stripped, because a folder filter is matched as a literal prefix and
        # a path is the one argument a person hand-enters or pastes. A single
        # trailing space turns the LIKE into one that can never match, and the
        # result is an empty search that looks exactly like an empty corpus.
        folder = (kwargs.get("folder") or "").strip() or None

        if not query:
            return sdk.fail("No query provided.")

        fts_query = _prepare_fts_query(query)
        if not fts_query:
            return sdk.fail("Query produced no searchable terms.")

        sql_parts = [
            "SELECT sc.path AS path, sc.chunk_index AS chunk_index,",
            "       sc.content AS content, sc.source AS source,",
            "       si.rank AS rank",
            "FROM lexical_index si",
            "JOIN lexical_content sc ON si.rowid = sc.rowid",
            "WHERE lexical_index MATCH ?",
        ]
        params = [fts_query]

        if sources:
            placeholders = ", ".join("?" for _ in sources)
            sql_parts.append(f"AND sc.source IN ({placeholders})")
            params.extend(sources)

        if folder:
            normalized = folder.replace("\\", "/").rstrip("/")
            sql_parts.append("AND (replace(sc.path, char(92), '/') = ?"
                             " OR replace(sc.path, char(92), '/') LIKE ?)")
            params.extend([normalized, normalized + "/%"])

        sql_parts.append("ORDER BY si.rank")
        sql_parts.append("LIMIT ?")
        params.append(top_k)

        sql = "\n".join(sql_parts)
        try:
            rows = sdk.db.query(sql, params)
        except sdk.Failed as failed:
            # Two very different failures arrive here. A missing lexical_index
            # means the indexing package is not installed — a fact about this
            # system, and there is nothing to retry. A *syntax* error means the
            # query was passed through as FTS5 and is not FTS5, which only
            # FTS5 could have told us; see ``_prepare_fts_query``. Ask again in
            # the language we are sure of rather than reporting a parser error
            # to somebody who was writing prose.
            literal = _literal_fts_query(query)
            if not _is_fts_syntax_error(failed.error) or literal == fts_query:
                return sdk.fail(f"Search failed: {failed.error}")
            sdk.log(f"query was not valid FTS5 ({failed.error}); searching for "
                    f"its words instead", level="debug")
            params[0] = literal
            try:
                rows = sdk.db.query(sql, params)
            except sdk.Failed as retried:
                return sdk.fail(f"Search failed: {retried.error}")

        if not rows:
            return sdk.ok([], llm_summary=f'No results found for "{query}".')

        modalities = _modalities(sdk, {row["path"] for row in rows})

        results = [
            SearchResult(
                path=row["path"],
                # FTS5 rank is negative and better when smaller; invert so
                # every stream agrees that higher means better, which is what
                # the hybrid tool's fusion assumes.
                score=-1.0 * float(row["rank"]),
                source=row["source"],
                stream="lexical",
                modality=modalities.get(row["path"], "unknown"),
                content=row["content"],
                chunk_index=int(row["chunk_index"] or 0),
            ).to_dict()
            for row in rows
        ]

        return sdk.ok(results,
                      llm_summary=_search_summary(query, results),
                      attachments=list({r["path"] for r in results}))


# --- Helpers, module level so the other two tools can import them ---


def _prepare_fts_query(query: str) -> str:
    """Prepare a query string for FTS5 MATCH.

    A query carrying explicit operators is passed through, so somebody who
    means ``"exact phrase" OR other*`` gets it. Otherwise the words are
    extracted and joined, which FTS5 implicitly ANDs.

    **The guess is unsound in one direction and that is handled at the call
    site, not here.** Deciding whether a string is FTS5 by looking for
    operator characters says yes to ordinary prose containing a quote or a
    star, and the result is not a worse ranking — it is a parser error, which
    is how ``what's the "memory" design`` came back as ``fts5: syntax error
    near "'"``. The apostrophe never survives the token path; it only survives
    the pass-through. Nothing here can be made to answer correctly, for the
    same reason the shell classifier could not: what parses as FTS5 is a
    question only FTS5 answers. So this stays a cheap guess, and a syntax
    error is taken as its refutation and retried literally.
    """
    has_operators = any(op in query
                        for op in ['"', " AND ", " OR ", " NOT ", "*"])
    if has_operators:
        return query

    return _literal_fts_query(query)


def _literal_fts_query(query: str) -> str:
    """The same words, with nothing left that FTS5 could read as syntax."""
    return " ".join(re.findall(r'\w+', query.lower()))


def _is_fts_syntax_error(message: str) -> bool:
    """Whether a failed query was rejected by the FTS5 *parser*.

    Narrow on purpose: an unindexed corpus, a missing table and a denied
    Request must all keep reporting what they are rather than being retried
    into a second, more confusing failure.
    """
    text = (message or "").lower()
    return "fts5" in text and ("syntax error" in text or "no such column" in text)


def _modalities(sdk, paths) -> dict:
    """Batch-fetch modality from the files table for a set of paths.

    Answers ``{}`` on failure rather than raising: a result with an unknown
    modality is still a usable result, and losing the whole search because a
    decoration query failed would be the wrong trade.
    """
    paths = list(paths)
    if not paths:
        return {}
    placeholders = ", ".join("?" for _ in paths)
    try:
        rows = sdk.db.query(
            f"SELECT path, modality FROM files WHERE path IN ({placeholders})",
            paths)
    except sdk.Failed as failed:
        sdk.log(f"modality lookup failed: {failed.error}", level="warning")
        return {}
    return {row["path"]: row["modality"] for row in rows}


def _search_summary(query: str, results: list) -> str:
    """Build the standardized LLM summary string for search results."""
    lines = [f'Found {len(results)} result(s) for "{query}":']
    for r in results[:5]:
        snippet = (r.get("content") or "").replace("\n", " ")[:120]
        lines.append(f'- {r["path"]} (score {r["score"]:.2f}): "{snippet}..."')
    if len(results) > 5:
        lines.append(f"[+{len(results) - 5} more]")
    return "\n".join(lines)
