"""
Lexical Search tool.

BM25-ranked keyword search across the FTS5 lexical_index. Searches all
indexed content (text chunks, OCR text, and any future modalities that
write to lexical_content with new source values).

Returns a list of SearchResult dicts, one per matching chunk.
"""

import logging
import os
import re

from Stage_3.BaseTool import BaseTool, ToolResult
from Stage_3.SearchResult import SearchResult

logger = logging.getLogger("LexicalSearch")


class LexicalSearch(BaseTool):
    name = "lexical_search"
    agent_enabled = False  # Superseded by hybrid_search; still callable internally
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
                "description": "Maximum number of results to return. Default 20.",
                "default": 20,
            },
            "sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Filter by content source. Omit to search all sources. "
                    'Current values: "extracted", "ocr". '
                    "Future values may include audio_transcript, video_subtitle, etc."
                ),
            },
            "folder": {
                "type": "string",
                "description": "Filter results to files under this folder path.",
            },
        },
        "required": ["query"],
    }
    requires_services = []

    def run(self, context, **kwargs):
        query = kwargs.get("query", "").strip()
        top_k = kwargs.get("top_k", 20)
        sources = kwargs.get("sources", None)
        folder = kwargs.get("folder", None)

        if not query:
            return ToolResult.failed("No query provided.")

        # --- 1. Build the FTS5 query ---
        # Clean the query for safe FTS5 matching.
        # If the user didn't use explicit FTS5 syntax, extract keywords.
        fts_query = self._prepare_fts_query(query)
        if not fts_query:
            return ToolResult.failed("Query produced no searchable terms.")

        # --- 2. Build SQL with optional filters ---
        sql_parts = [
            "SELECT sc.path, sc.chunk_index, sc.content, sc.source, si.rank",
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
            normalized_folder = os.path.normpath(folder)
            sql_parts.append("AND sc.path LIKE ? || '%'")
            params.append(normalized_folder)

        sql_parts.append("ORDER BY si.rank")
        sql_parts.append("LIMIT ?")
        params.append(top_k)

        sql = "\n".join(sql_parts)

        # --- 3. Execute ---
        try:
            with context.db.lock:
                cur = context.db.conn.execute(sql, params)
                rows = cur.fetchall()
        except Exception as e:
            logger.error(f"Lexical search failed: {e}")
            return ToolResult.failed(f"Search failed: {e}")

        if not rows:
            return ToolResult(
                data=[],
                metadata={"query": query, "fts_query": fts_query, "result_count": 0},
            )

        # --- 4. Look up modalities for result paths ---
        paths = list({row[0] for row in rows})
        modality_map = self._get_modalities(context.db, paths)

        # --- 5. Build SearchResult objects ---
        results = []
        for path, chunk_index, content, source, rank in rows:
            results.append(SearchResult(
                path=path,
                score=-1.0 * float(rank),  # FTS5 rank is negative; invert so higher = better
                source=source,
                stream="lexical",
                modality=modality_map.get(path, "unknown"),
                content=content,
                chunk_index=int(chunk_index),
            ).to_dict())

        return ToolResult(
            data=results,
            metadata={
                "query": query,
                "fts_query": fts_query,
                "result_count": len(results),
            },
        )

    # --- Helpers ---

    def _prepare_fts_query(self, query: str) -> str:
        """
        Prepare a query string for FTS5 MATCH.

        If the query contains explicit FTS5 operators (AND, OR, NOT, quotes, *),
        pass it through as-is. Otherwise, extract alphanumeric tokens and
        join them so FTS5 implicitly ANDs them.
        """
        has_operators = any(op in query for op in ['"', " AND ", " OR ", " NOT ", "*"])
        if has_operators:
            return query

        # Extract word tokens, ignore punctuation
        tokens = re.findall(r'\w+', query.lower())
        if not tokens:
            return ""
        return " ".join(tokens)

    def _get_modalities(self, db, paths: list) -> dict:
        """Batch-fetch modality from the files table for a list of paths."""
        if not paths:
            return {}

        placeholders = ", ".join("?" for _ in paths)
        sql = f"SELECT path, modality FROM files WHERE path IN ({placeholders})"

        try:
            with db.lock:
                cur = db.conn.execute(sql, paths)
                return {row[0]: row[1] for row in cur.fetchall()}
        except Exception as e:
            logger.error(f"Modality lookup failed: {e}")
            return {}
