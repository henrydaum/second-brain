"""
Hybrid Search tool.

Fuses results from lexical and semantic search using Reciprocal Rank Fusion
(RRF). Calls the existing search tools via ``sdk.tools.call``, deduplicates
chunks into documents, applies RRF across all streams, and groups final
results by modality.

Modality-agnostic — works with whatever modalities the sub-tools return
(text, image, audio, tabular, etc.) without hardcoding any of them.

Everything below the two sub-tool calls is arithmetic over dicts, which is why
this migrated almost unchanged: the fusion never touched the database or the
filesystem. ``tool.call`` is SAFE for the same reason a script is — the
callee's own Requests are classified with this tool still in the chain, so
routing through it launders nothing.
"""


dependencies_files = ['tools/tool_lexical_search.py',
                      'tools/tool_semantic_search.py']
dependencies_pip = []
requests = ["tool.call"]

from guest.bases import BaseTool

from .tool_lexical_search import _search_summary

# RRF constant — higher values give less weight to rank differences.
# 60 is the standard value from the original RRF paper.
RRF_K = 60


class HybridSearch(BaseTool):
    """Hybrid search."""
    name = "hybrid_search"
    description = (
        "Search indexed files using both keyword and semantic retrieval, then "
        "fuse the results for better ranking. Prefer this over lexical_search or semantic_search alone when retrieving local files or excerpts. Optional folder and modality filters can narrow the search."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for in the indexed local files.",
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
    dependencies_tools = ["lexical_search", "semantic_search"]
    agent_prompt = (
        "## Searching indexed files\n"
        "Three retrieval tools search the indexed corpus — your sync_directories plus dropped-in attachments. "
        "Files outside the index are not searchable; use read_file for a path you already know.\n"
        "- hybrid_search: fuses keyword + semantic ranking. Default choice for finding local files or excerpts.\n"
        "- lexical_search: exact keyword/identifier/code matching. Use for error strings, function names, rare terms.\n"
        "- semantic_search: meaning-based retrieval. Use for paraphrased or conceptual questions where exact wording won't match.\n"
        "Results are excerpts (chunks) grouped by document; follow up with read_file for full context."
    )

    def run(self, sdk, **kwargs):
        """Run hybrid search."""
        query = (kwargs.get("query") or "").strip()
        max_results = max(1, int(kwargs.get("max_results") or 5))
        folder = kwargs.get("folder") or None
        modality = kwargs.get("modality") or None

        if not query:
            return sdk.fail("No query provided.")

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

        all_raw = (_sub_search(sdk, "lexical_search", lex_kwargs)
                   + _sub_search(sdk, "semantic_search", sem_kwargs))

        # Filter by modality if requested (lexical search doesn't filter by
        # modality natively, so we apply the filter here after the fact)
        if modality:
            all_raw = [r for r in all_raw if r.get("modality") == modality]

        if not all_raw:
            return sdk.ok([], llm_summary=f'No results found for "{query}".')

        # --- Group by stream ---
        by_stream = {}
        for result in all_raw:
            by_stream.setdefault(result["stream"], []).append(result)

        # --- Deduplicate within each stream (collapse chunks into docs) ---
        deduped_streams = {name: _dedup_by_path(results)
                           for name, results in by_stream.items()}

        # --- RRF across streams ---
        merged_docs, rrf_scores = _apply_rrf(deduped_streams)

        # --- Sort globally and take max_results ---
        for path, doc in merged_docs.items():
            doc["score"] = rrf_scores[path]

        flat_results = sorted(merged_docs.values(),
                              key=lambda d: d["score"],
                              reverse=True)[:max_results]

        sdk.log(f"hybrid search: {len(by_stream)} streams, "
                f"{len(merged_docs)} unique docs, "
                f"{len(flat_results)} returned")

        return sdk.ok(flat_results,
                      llm_summary=_search_summary(query, flat_results),
                      attachments=[d["path"] for d in flat_results])


def _sub_search(sdk, name: str, kwargs) -> list:
    """One sub-tool's results, or an empty list.

    Either retriever may legitimately be missing — semantic search needs an
    embedder loaded, lexical search needs the FTS index built — and fusing one
    stream is still a useful answer. So a failure here degrades the ranking
    rather than failing the search, which is what the native version got from
    reading ``result.success`` off a returned envelope.
    """
    try:
        data = sdk.tools.call(name, **kwargs)
    except sdk.Failed as failed:
        sdk.log(f"{name} unavailable: {failed.error}", level="debug")
        return []
    return data if isinstance(data, list) else []


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
