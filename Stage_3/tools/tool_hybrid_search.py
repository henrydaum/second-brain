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

logger = logging.getLogger("HybridSearch")

# RRF constant — higher values give less weight to rank differences.
# 60 is the standard value from the original RRF paper.
RRF_K = 60


class HybridSearch(BaseTool):
    name = "hybrid_search"
    description = (
        "Search for files using both keyword and semantic similarity, then "
        "fuse the results for higher accuracy. Combines BM25 lexical search "
        "with vector similarity search across all available embedding streams "
        "(text, image, and any future modalities).\n\n"
        "Results are ranked using Reciprocal Rank Fusion (RRF) and grouped "
        "by modality (text, image, etc.). Documents found by multiple methods "
        'are marked as "Hybrid" and ranked higher.'
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query.",
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum results per modality in the final output. Default 10.",
                "default": 10,
            },
            "folder": {
                "type": "string",
                "description": "Filter results to files under this folder path.",
            },
            "sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Filter lexical results by content source. "
                    'E.g. "extracted", "ocr".'
                ),
            },
            "streams": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Which semantic embedding streams to search. "
                    'E.g. "text", "image". Omit to search all.'
                ),
            },
        },
        "required": ["query"],
    }
    requires_services = []

    def run(self, context, **kwargs):
        query = kwargs.get("query", "").strip()
        top_k = kwargs.get("top_k", 10)
        folder = kwargs.get("folder", None)
        sources = kwargs.get("sources", None)
        streams = kwargs.get("streams", None)

        if not query:
            return ToolResult.failed("No query provided.")

        # --- 1. Fetch from sub-tools ---
        # Over-fetch to give RRF enough candidates to fuse meaningfully.
        fetch_limit = max(200, top_k * 10)

        lex_kwargs = {"query": query, "top_k": fetch_limit}
        sem_kwargs = {"query": query, "top_k": fetch_limit}
        if folder:
            lex_kwargs["folder"] = folder
            sem_kwargs["folder"] = folder
        if sources:
            lex_kwargs["sources"] = sources
        if streams:
            sem_kwargs["streams"] = streams

        t0 = time.time()
        lex_result = context.call_tool("lexical_search", **lex_kwargs)
        sem_result = context.call_tool("semantic_search", **sem_kwargs)
        logger.debug(f"Sub-tool fetch completed in {time.time() - t0:.2f}s")

        lex_data = lex_result.data if lex_result.success and lex_result.data else []
        sem_data = sem_result.data if sem_result.success and sem_result.data else []

        all_raw = lex_data + sem_data

        if not all_raw:
            return ToolResult(
                data={},
                metadata={"query": query, "result_count": 0},
            )

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

        # --- 5. Group by modality and take top_k ---
        by_modality = defaultdict(list)
        for path, doc in merged_docs.items():
            doc["score"] = rrf_scores[path]
            by_modality[doc["modality"]].append(doc)

        final = {}
        total = 0
        for modality, docs in by_modality.items():
            docs.sort(key=lambda x: x["score"], reverse=True)
            final[modality] = docs[:top_k]
            total += len(final[modality])

        logger.info(
            f"Hybrid search: {len(by_stream)} streams, "
            f"{len(merged_docs)} unique docs, {total} returned"
        )

        return ToolResult(
            data=final,
            metadata={
                "query": query,
                "streams_fused": list(by_stream.keys()),
                "unique_docs": len(merged_docs),
                "result_count": total,
            },
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
