"""
Semantic Search tool.

Vector similarity search across embedding tables. Searches each embedding
stream independently (text, image, and any future modalities), embeds the
query with the correct model per stream, and returns ranked results.

Results from different streams are NOT merged — each carries its stream tag
so the hybrid search tool can apply RRF fusion across them.

Adding a new modality = one entry in EMBEDDING_STREAMS.
"""

import logging
import os
import time

import numpy as np

from plugins.BaseTool import BaseTool, ToolResult
from plugins.tools.helpers.SearchResult import SearchResult
from plugins.tools.tool_lexical_search import _search_summary

logger = logging.getLogger("SemanticSearch")


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
    },
    "image": {
        "table": "image_embeddings",
        "index_col": "image_index",
        "service": "image_embedder",
        "source": "image_embedding",
        "content_table": "ocr_text",    # OCR text for images (if available)
        "content_join_col": None,       # JOIN on path only (ocr_text has no index col)
    },
    # "audio": {
    #     "table": "audio_embeddings",
    #     "index_col": "segment_index",
    #     "service": "audio_embedder",
    #     "source": "audio_embedding",
    #     "content_table": "audio_transcripts",
    #     "content_join_col": "segment_index",
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
    name = "semantic_search"
    description = (
        "Search for files by meaning using vector similarity. Embeds your "
        "query and compares it against stored embeddings (text, image, and "
        "any future modalities). Returns the most semantically similar results.\n\n"
        "Each embedding stream (text, image) is searched independently with "
        "its own model. Results are tagged by stream for downstream fusion."
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

    def run(self, context, **kwargs):
        query = kwargs.get("query", "").strip()
        top_k = kwargs.get("top_k", 5)
        requested_streams = kwargs.get("streams", None)
        folder = kwargs.get("folder", None)

        if not query:
            return ToolResult.failed("No query provided.")

        # Determine which streams to search
        if requested_streams:
            stream_names = [s for s in requested_streams if s in EMBEDDING_STREAMS]
            if not stream_names:
                available = list(EMBEDDING_STREAMS.keys())
                return ToolResult.failed(
                    f"No valid streams requested. Available: {available}"
                )
        else:
            stream_names = list(EMBEDDING_STREAMS.keys())

        # Search each stream
        all_results = []
        streams_searched = []
        streams_skipped = []

        for stream_name in stream_names:
            stream_results = self._search_stream(
                context, stream_name, query, top_k, folder
            )
            if stream_results is None:
                streams_skipped.append(stream_name)
            else:
                streams_searched.append(stream_name)
                all_results.extend(stream_results)

        paths = list({r["path"] for r in all_results})
        return ToolResult(
            data=all_results,
            llm_summary=_search_summary(query, all_results),
            attachment_paths=paths,
        )

    def _search_stream(self, context, stream_name, query, top_k, folder):
        """
        Search a single embedding stream. Returns list of result dicts,
        or None if the stream's embedder isn't available.
        """
        config = EMBEDDING_STREAMS[stream_name]
        table = config["table"]
        index_col = config["index_col"]
        service_name = config["service"]
        source = config["source"]
        stream_tag = f"{stream_name}_semantic"

        # 1. Get embedder service
        embedder = context.services.get(service_name)
        if not embedder or not embedder.loaded:
            logger.info(f"Skipping {stream_name} stream: {service_name} not loaded")
            return None

        # 2. Encode query — this calls the embedding model and can be slow
        logger.debug(f"Encoding query for {stream_name} stream...")
        try:
            query_vec = embedder.encode(query)
        except Exception as e:
            logger.error(f"Failed to encode query for {stream_name}: {e}")
            return None

        if query_vec is None:
            logger.error(f"Encoder returned None for {stream_name}")
            return None

        # Handle both 1D (single string) and 2D (list) encode results
        if query_vec.ndim == 2:
            query_vec = query_vec[0]

        # 3. Load embeddings from DB
        sql_parts = [f"SELECT path, {index_col}, embedding FROM {table}"]
        sql_parts.append(f"WHERE model_name = ?")
        params = [embedder.model_name]

        if folder:
            normalized_folder = os.path.normpath(folder)
            sql_parts.append("AND path LIKE ? || '%'")
            params.append(normalized_folder)

        sql = "\n".join(sql_parts)

        try:
            with context.db.lock:
                cur = context.db.conn.execute(sql, params)
                rows = cur.fetchall()
        except Exception as e:
            logger.error(f"Failed to load embeddings from {table}: {e}")
            return None

        if not rows:
            logger.info(f"No embeddings in {table} for model {embedder.model_name}")
            return []

        # 4. Deserialize embedding blobs into numpy vectors and stack into
        # a matrix. Skip any rows with dimension mismatches (stale embeddings
        # from a different model).
        paths = []
        indices = []
        valid_vecs = []

        for row in rows:
            path, idx, blob = row[0], row[1], row[2]
            if blob:
                vec = np.frombuffer(blob, dtype=np.float32)
                if vec.shape[0] == query_vec.shape[0]:
                    paths.append(path)
                    indices.append(idx)
                    valid_vecs.append(vec)

        if not valid_vecs:
            return []

        emb_matrix = np.vstack(valid_vecs)

        # 5. Cosine similarity — both query and stored vectors are pre-normalized
        # during encoding, so dot product == cosine similarity.
        t_sim = time.time()
        scores = np.dot(emb_matrix, query_vec)
        logger.debug(
            f"Similarity search over {len(valid_vecs)} vectors in {time.time() - t_sim:.3f}s"
        )

        # 6. Top-k
        k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[-k:][::-1]

        # 7. Fetch content for top results
        top_paths = [paths[i] for i in top_indices]
        top_idx_values = [indices[i] for i in top_indices]
        content_map = self._fetch_content(
            context.db, config, top_paths, top_idx_values
        )

        # 8. Look up modalities
        modality_map = self._get_modalities(context.db, list(set(top_paths)))

        # 9. Build SearchResult objects
        index_field = INDEX_FIELD_MAP.get(index_col, index_col)
        results = []

        for i in top_indices:
            path = paths[i]
            idx_value = indices[i]

            result_kwargs = {
                "path": path,
                "score": float(scores[i]),
                "source": source,
                "stream": stream_tag,
                "modality": modality_map.get(path, "unknown"),
                "content": content_map.get((path, idx_value)),
                index_field: int(idx_value),
            }
            results.append(SearchResult(**result_kwargs).to_dict())

        return results

    def _fetch_content(self, db, stream_config, paths, indices):
        """
        Fetch text content for a list of (path, index) pairs.
        Returns a dict mapping (path, index) -> content string.
        """
        content_table = stream_config.get("content_table")
        if not content_table or not paths:
            return {}

        join_col = stream_config.get("content_join_col")

        try:
            with db.lock:
                if join_col:
                    # JOIN on (path, index_col) — e.g. text_chunks
                    # Build a query for each (path, index) pair
                    placeholders = " OR ".join(
                        f"(path = ? AND {join_col} = ?)" for _ in paths
                    )
                    params = []
                    for p, idx in zip(paths, indices):
                        params.extend([p, idx])

                    sql = f"SELECT path, {join_col}, content FROM {content_table} WHERE {placeholders}"
                    cur = db.conn.execute(sql, params)
                    return {(row[0], row[1]): row[2] for row in cur.fetchall()}
                else:
                    # JOIN on path only — e.g. ocr_text (one row per path)
                    unique_paths = list(set(paths))
                    placeholders = ", ".join("?" for _ in unique_paths)
                    sql = f"SELECT path, content FROM {content_table} WHERE path IN ({placeholders})"
                    cur = db.conn.execute(sql, unique_paths)
                    path_content = {row[0]: row[1] for row in cur.fetchall()}

                    # Map back to (path, index) keys for uniform access
                    result = {}
                    for p, idx in zip(paths, indices):
                        if p in path_content:
                            result[(p, idx)] = path_content[p]
                    return result

        except Exception as e:
            logger.error(f"Content fetch from {content_table} failed: {e}")
            return {}

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
