"""
Semantic Search tool.

Vector similarity search across embedding tables. Searches each embedding
stream independently (text, image, and any future modalities), embeds the
query with the correct model per stream, and returns ranked results.

Results from different streams are NOT merged — each carries its stream tag
so the hybrid search tool can apply RRF fusion across them.

Adding a new modality = one entry in EMBEDDING_STREAMS.

**The ranking happens in SQL, and only the top few rows cross the boundary.**
The native version read every vector in the table into the tool and did the
arithmetic in numpy, which cannot survive a sandbox: ``db.query`` caps its
answer, and at a hundred thousand chunks the corpus is a couple of hundred
megabytes of JSON *per query*. So the reduction moved to where the vectors
already are. ``vec_cosine`` is a scalar function the kernel registers on its
SQLite connection — an operator, not a search: it knows nothing about
embeddings, models or streams, and everything else here is ordinary SQL
composed around it. The shape is deliberately the same one ``lexical_search``
has always had with FTS5.
"""


dependencies_files = ['services/service_embed.py',
                      'tools/helpers/SearchResult.py',
                      'tools/tool_lexical_search.py']
dependencies_pip = []
requests = ["db.query", "service.call"]

from guest.bases import BaseTool

from .helpers.SearchResult import SearchResult
from .tool_lexical_search import _search_summary


# =====================================================================
# STREAM REGISTRY
#
# Each entry defines one embedding stream: which table to search,
# which column is the per-file index, which embedder service to use,
# and where to find text content for the results.
#
# To add a new modality (e.g. audio):
#   1. Create an audio_embeddings table in a new task
#   2. Add an "audio" entry here
#   That's it — the tool picks it up automatically.
# =====================================================================

EMBEDDING_STREAMS = {
    "text": {
        "table": "text_embeddings",
        "index_col": "chunk_index",
        "service": "text_embedder",
        "source": "text_embedding",
        "content_table": "text_chunks",     # WHERE to get text content
        "content_join_col": "chunk_index",  # JOIN column (besides path)
        # How to embed the *query* for this stream. The text model has one
        # encoder; CLIP has two, and asking it to encode a query the way it
        # encodes an image would open the query string as a file.
        "query_method": "encode",
    },
    "image": {
        "table": "image_embeddings",
        "index_col": "image_index",
        "service": "image_embedder",
        "source": "image_embedding",
        "content_table": "ocr_text",    # OCR text for images (if available)
        "content_join_col": None,       # JOIN on path only (ocr_text has no index col)
        "query_method": "encode_text",  # CLIP's text tower, into the shared space
    },
    # "audio": {
    #     "table": "audio_embeddings",
    #     "index_col": "segment_index",
    #     "service": "audio_embedder",
    #     "source": "audio_embedding",
    #     "content_table": "audio_transcripts",
    #     "content_join_col": "segment_index",
    #     "query_method": "encode",
    # },
}

# Map stream names to the SearchResult field for the index column.
# If a stream's index_col doesn't match a SearchResult field name,
# add the mapping here.
INDEX_FIELD_MAP = {
    "chunk_index": "chunk_index",
    "image_index": "image_index",
    # "segment_index": "segment_index",  # future
}


class SemanticSearch(BaseTool):
    """Semantic search."""
    name = "semantic_search"
    description = (
        "Search for files by meaning using vector similarity. Embeds your "
        "query and compares it against stored embeddings (text, image, and "
        "any future modalities). Returns the most semantically similar results.\n\n"
        "Each embedding stream (text, image) is searched independently with "
        "its own model."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language query to search for.",
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum results per stream. Default 5.",
                "default": 5,
            },
            "streams": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Which embedding streams to search. Omit to search all available. "
                    'Current options: "text", "image".'
                ),
            },
            "folder": {
                "type": "string",
                "description": "Filter results to files under this folder path.",
            },
        },
        "required": ["query"],
    }
    requires_services = []  # Checked dynamically per stream
    background_safe = True

    def run(self, sdk, **kwargs):
        """Run semantic search."""
        query = (kwargs.get("query") or "").strip()
        top_k = max(1, int(kwargs.get("top_k") or 5))
        requested = kwargs.get("streams") or None
        folder = kwargs.get("folder") or None

        if not query:
            return sdk.fail("No query provided.")

        if requested:
            stream_names = [s for s in requested if s in EMBEDDING_STREAMS]
            if not stream_names:
                return sdk.fail("No valid streams requested. Available: "
                                f"{list(EMBEDDING_STREAMS)}")
        else:
            stream_names = list(EMBEDDING_STREAMS)

        results = []
        for name in stream_names:
            found = self._search_stream(sdk, name, query, top_k, folder)
            if found is None:
                sdk.log(f"skipping {name} stream: embedder unavailable",
                        level="debug")
                continue
            results.extend(found)

        return sdk.ok(results,
                      llm_summary=_search_summary(query, results),
                      attachments=list({r["path"] for r in results}))

    def _search_stream(self, sdk, stream_name, query, top_k, folder):
        """
        Search a single embedding stream. Returns a list of result dicts,
        or None if the stream's embedder isn't available.
        """
        config = EMBEDDING_STREAMS[stream_name]
        service = config["service"]

        # 1. The embedder, and which model it holds. ``describe`` answers both
        #    in one call, and it is the model's *own* id — the adapter is named
        #    after the service, so reading model_name off it would give
        #    "text_embedder" and match no stored row.
        try:
            described = sdk.services.call(service, "describe")
        except sdk.Failed:
            return None
        if not described or not described.get("loaded"):
            return None
        model_name = described.get("model_name")

        # 2. Encode the query. Slow — this is a model call.
        try:
            encoded = sdk.services.call(service, config["query_method"],
                                        inputs=query)
        except sdk.Failed as failed:
            sdk.log(f"failed to encode query for {stream_name}: {failed.error}",
                    level="error")
            return None
        vector = encoded[0] if encoded else None
        if not vector:
            sdk.log(f"{service} returned no vector for the query",
                    level="error")
            return None

        rows = self._rank(sdk, config, vector, model_name, top_k, folder)
        if not rows:
            return []

        index_field = INDEX_FIELD_MAP.get(config["index_col"],
                                          config["index_col"])
        return [
            SearchResult(**{
                "path": row["path"],
                "score": float(row["score"]),
                "source": config["source"],
                "stream": f"{stream_name}_semantic",
                "modality": row.get("modality") or "unknown",
                "content": row.get("content"),
                index_field: int(row["idx"] or 0),
            }).to_dict()
            for row in rows
        ]

    def _rank(self, sdk, config, vector, model_name, top_k, folder):
        """The whole search, as one statement.

        Tried twice at most: the content table belongs to a *different*
        package (``ocr_text`` exists only when the OCR task is installed), so
        a missing one is an uninstalled capability rather than an error. The
        fallback drops the join and returns results without snippets, which is
        the same degrade-quietly rule ``task_lexical_index`` follows per source.
        """
        try:
            return sdk.db.query(*_ranking_sql(config, vector, model_name,
                                              top_k, folder, content=True))
        except sdk.Failed as failed:
            if not config.get("content_table"):
                raise
            sdk.log(f"ranking without content from "
                    f"{config['content_table']}: {failed.error}",
                    level="debug")

        try:
            return sdk.db.query(*_ranking_sql(config, vector, model_name,
                                              top_k, folder, content=False))
        except sdk.Failed as failed:
            # Now it is the embeddings table itself, i.e. nothing has been
            # embedded into this stream yet.
            sdk.log(f"no {config['table']} to search: {failed.error}",
                    level="debug")
            return []


def _ranking_sql(config, vector, model_name, top_k, folder, content: bool):
    """Build (sql, params) for one stream's ranking query.

    The inner SELECT is where the work happens: it is the only place
    ``vec_cosine`` runs, over the base table alone, so the joins that decorate
    the answer never see more than ``top_k`` rows. ``length(embedding) = ?``
    drops vectors left behind by a different model *before* the arithmetic —
    the same dimension check the native version did in numpy, moved to where
    it costs nothing.
    """
    inner = [
        f"SELECT path, {config['index_col']} AS idx,",
        "       vec_cosine(embedding, ?) AS score",
        f"FROM {config['table']}",
        "WHERE model_name = ? AND length(embedding) = ?",
    ]
    params = [vector, model_name, len(vector)]

    if folder:
        normalized = str(folder).replace("\\", "/").rstrip("/")
        inner.append("AND (replace(path, char(92), '/') = ?"
                     " OR replace(path, char(92), '/') LIKE ?)")
        params.extend([normalized, normalized + "/%"])

    inner.append("ORDER BY score DESC")
    inner.append("LIMIT ?")
    params.append(top_k)

    select = ["t.path AS path", "t.idx AS idx", "t.score AS score",
              "f.modality AS modality"]
    joins = ["LEFT JOIN files f ON f.path = t.path"]

    content_table = config.get("content_table")
    if content and content_table:
        select.append("c.content AS content")
        join_col = config.get("content_join_col")
        on = "c.path = t.path"
        if join_col:
            on += f" AND c.{join_col} = t.idx"
        joins.append(f"LEFT JOIN {content_table} c ON {on}")

    sql = "\n".join([
        "SELECT " + ", ".join(select),
        "FROM (" + "\n".join(inner) + ") t",
        *joins,
        # A NULL score is a vector vec_cosine could not compare at all; SQLite
        # sorts those last under DESC, so they only appear when the stream has
        # fewer usable rows than asked for. Dropping them here is cheaper than
        # computing the function twice to filter in the inner query.
        "WHERE t.score IS NOT NULL",
        "ORDER BY t.score DESC",
    ])
    return sql, params
