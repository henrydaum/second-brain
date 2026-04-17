"""
Hybrid Search tool.

Fuses results from lexical and semantic search using Reciprocal Rank Fusion
(RRF). Calls the existing search tools via context.call_tool(), deduplicates
chunks into documents, applies RRF across all streams, and groups final
results by modality.

Modality-agnostic — works with whatever modalities the sub-tools return
(text, image, audio, tabular, etc.) without hardcoding any of them.
"""

import logging
import time
from collections import defaultdict

from Stage_3.BaseTool import BaseTool, ToolResult
from Stage_3.tools.tool_lexical_search import _search_summary

logger = logging.getLogger("HybridSearch")

# RRF constant — higher values give less weight to rank differences.
# 60 is the standard value from the original RRF paper.
RRF_K = 60


class HybridSearch(BaseTool):
    name = "hybrid_search"
    description = (
        "Search for files using both keyword and semantic similarity, then "
        "fuse the results for higher accuracy. Combines BM25 lexical search "
        "with vector similarity search.\n\n"
        "Results are ranked using Reciprocal Rank Fusion (RRF). Documents "
        'found by multiple methods are marked as "Hybrid" and ranked higher.'
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum total results to return. Default 5.",
                "default": 5,
            },
            "folder": {
                "type": "string",
                "description": "Filter results to files under this folder path.",
            },
            "modality": {
                "type": "string",
                "description": (
                    "Filter results to a specific file modality. "
                    'E.g. "text", "image". Omit to search all.'
                ),
            },
        },
        "required": ["query"],
    }
    requires_services = []

    def run(self, context, **kwargs):
        query = kwargs.get("query", "").strip()
        max_results = kwargs.get("max_results", 5)
        folder = kwargs.get("folder", None)
        modality = kwargs.get("modality", None)

        if not query:
            return ToolResult.failed("No query provided.")

        # --- 1. Fetch from sub-tools ---
        # Over-fetch to give RRF enough candidates to fuse meaningfully.
        fetch_limit = max(200, max_results * 10)

        lex_kwargs = {"query": query, "top_k": fetch_limit}
        sem_kwargs = {"query": query, "top_k": fetch_limit}
        if folder:
            lex_kwargs["folder"] = folder
            sem_kwargs["folder"] = folder
        if modality:
            # Map modality to the corresponding semantic embedding stream
            sem_kwargs["streams"] = [modality]

        t0 = time.time()
        lex_result = context.call_tool("lexical_search", **lex_kwargs)
        sem_result = context.call_tool("semantic_search", **sem_kwargs)
        logger.debug(f"Sub-tool fetch completed in {time.time() - t0:.2f}s")

        lex_data = lex_result.data if lex_result.success and lex_result.data else []
        sem_data = sem_result.data if sem_result.success and sem_result.data else []

        all_raw = lex_data + sem_data

        # Filter by modality if requested (lexical search doesn't filter by
        # modality natively, so we apply the filter here after the fact)
        if modality:
            all_raw = [r for r in all_raw if r.get("modality") == modality]

        if not all_raw:
            return ToolResult(data={}, llm_summary=f'No results found for "{query}".')

        # --- 2. Group by stream ---
        by_stream = defaultdict(list)
        for result in all_raw:
            by_stream[result["stream"]].append(result)

        # --- 3. Deduplicate within each stream (collapse chunks into docs) ---
        deduped_streams = {}
        for stream_name, results in by_stream.items():
            deduped_streams[stream_name] = _dedup_by_path(results)

        # --- 4. RRF across streams ---
        merged_docs, rrf_scores = _apply_rrf(deduped_streams)

        # --- 5. Sort globally and take max_results ---
        for path, doc in merged_docs.items():
            doc["score"] = rrf_scores[path]

        flat_results = sorted(
            merged_docs.values(),
            key=lambda d: d["score"],
            reverse=True,
        )[:max_results]

        logger.info(
            f"Hybrid search: {len(by_stream)} streams, "
            f"{len(merged_docs)} unique docs, {len(flat_results)} returned"
        )

        paths = [d["path"] for d in flat_results]
        return ToolResult(
            data=flat_results,
            llm_summary=_search_summary(query, flat_results),
            attachment_paths=paths,
        )


def _dedup_by_path(results):
    """
    Collapse multiple chunks of the same file into one document entry.
    A single PDF might have 50 matching chunks — we keep the best-scoring
    chunk's content as the representative snippet, and count total hits
    so the user knows how much of the document matched.
    """
    by_path = {}
    for res in results:
        path = res["path"]
        if path not in by_path:
            by_path[path] = dict(res)
            by_path[path]["num_hits"] = 1
        else:
            stored = by_path[path]
            stored["num_hits"] += 1
            if res["score"] > stored["score"]:
                _update_content(stored, res)
    return list(by_path.values())


def _apply_rrf(deduped_streams):
    """
    Apply Reciprocal Rank Fusion across all streams.

    RRF scores each document as: sum over streams of 1/(K + rank).
    Documents that appear in multiple streams accumulate higher scores,
    naturally boosting results found by both keyword AND vector search.

    Returns (merged_docs, rrf_scores) where:
      - merged_docs: path -> merged result dict
      - rrf_scores:  path -> cumulative RRF score
    """
    rrf_scores = {}
    merged_docs = {}

    for stream_name, docs in deduped_streams.items():
        docs.sort(key=lambda x: x["score"], reverse=True)
        result_type = "Lexical" if stream_name == "lexical" else "Semantic"

        for rank, doc in enumerate(docs):
            path = doc["path"]
            # RRF formula: each stream contributes 1/(K + rank + 1) per document
            rrf_scores[path] = rrf_scores.get(path, 0.0) + 1.0 / (RRF_K + rank + 1)

            if path not in merged_docs:
                merged_docs[path] = dict(doc)
                merged_docs[path]["result_type"] = result_type
            else:
                stored = merged_docs[path]

                # Mark as Hybrid if found via different retrieval methods
                if stored["result_type"] != result_type:
                    stored["result_type"] = "Hybrid"

                # Accumulate hits
                stored["num_hits"] += doc["num_hits"]

                # Keep the higher-scoring content
                if doc["score"] > stored["score"]:
                    _update_content(stored, doc)

                # Merge source tags
                existing = set(stored["source"].split(", "))
                incoming = set(doc["source"].split(", "))
                stored["source"] = ", ".join(sorted(existing | incoming))

    return merged_docs, rrf_scores


# Fields that represent the "display content" of a result.
# Updated in place when a higher-scoring chunk is found.
_CONTENT_FIELDS = ("content", "score", "chunk_index", "image_index")


def _update_content(target, source):
    """Overwrite display-content fields on target from source."""
    for field in _CONTENT_FIELDS:
        if field in source:
            target[field] = source[field]
