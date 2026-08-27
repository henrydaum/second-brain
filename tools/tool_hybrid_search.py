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
requests = ["tool.call", "tool.list", "db.query"]

from guest.bases import BaseTool

from .tool_lexical_search import _search_summary

# RRF constant — higher values give less weight to rank differences.
# 60 is the standard value from the original RRF paper.
RRF_K = 60


class HybridSearch(BaseTool):
    """Hybrid search."""
    name = "hybrid_search"
    description = (
        "Search indexed files from the sync_directories using a hybrid lexical/semantic search algorithm. "
        "Optional folder and modality filters can narrow the search."
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
    dependencies_tools = ["lexical_search", "semantic_search"]
    # No cue declared: the document count below is live, so an indexing run has
    # to show up on the agent's next call. That is the default rung.

    def agent_prompt(self, sdk):
        """Name the retrieval tools in scope, and say when the corpus is empty.

        The static version this replaces asserted that three tools search "the
        indexed corpus" whether or not anything was in it, and whether or not
        all three were installed. The empty case is the expensive one: the
        tools return nothing, which reads exactly like the fact not existing,
        so an agent searches three ways and concludes it does not.
        """
        scope = set(sdk.tools.list() or [])
        indexed = self._indexed(sdk)
        if indexed == 0:
            return (
                "## Searching indexed files\n"
                "Nothing is indexed yet — the retrieval tools will return "
                "nothing. An empty result here means an empty corpus, not a "
                "missing fact. Use read_file for a path you already know."
            )
        held = (f"{indexed} documents indexed (sync_directories plus "
                "dropped-in attachments)." if indexed
                else "Search covers your sync_directories plus dropped-in "
                     "attachments.")
        lines = [
            "## Searching indexed files",
            f"{held} Files outside the index are not searchable; use "
            "read_file for a path you already know.",
        ]
        lines += [text for name, text in (
            ("hybrid_search",
             "- hybrid_search: keyword + semantic, fused. Default choice."),
            ("lexical_search",
             "- lexical_search: exact keywords, identifiers, error strings."),
            ("semantic_search",
             "- semantic_search: meaning-based, for paraphrased questions."),
        ) if name in scope]
        lines.append("Results are excerpts grouped by document; follow up with "
                     "read_file for full context.")
        return "\n".join(lines)

    @staticmethod
    def _indexed(sdk):
        """Distinct indexed documents, or None when the table is not there yet.

        None rather than 0 deliberately: asserting an empty corpus we could not
        verify would be the same wrong conclusion in the other direction.
        """
        try:
            rows = sdk.db.query(
                "SELECT COUNT(DISTINCT path) AS n FROM lexical_content")
        except sdk.Failed:
            return None
        return int((rows or [{}])[0].get("n") or 0)

    def run(self, sdk, **kwargs):
        """Run hybrid search."""
        query = (kwargs.get("query") or "").strip()
        max_results = max(1, int(kwargs.get("max_results") or 5))
        # Stripped here too, not only in the sub-tools: this is the argument
        # the agent and /tools actually fill in, and the value it forwards
        # should be the one it was asked about.
        folder = (kwargs.get("folder") or "").strip() or None
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

        lexical, lex_error = _sub_search(sdk, "lexical_search", lex_kwargs)
        semantic, sem_error = _sub_search(sdk, "semantic_search", sem_kwargs)
        all_raw = lexical + semantic

        # Filter by modality if requested (lexical search doesn't filter by
        # modality natively, so we apply the filter here after the fact)
        if modality:
            all_raw = [r for r in all_raw if r.get("modality") == modality]

        if not all_raw:
            # "No results" and "both retrievers are broken" used to be the same
            # sentence, which made every failure downstream undiagnosable: a
            # caller could not tell a corpus with no match from a search that
            # never ran. Naming the ones that failed costs a clause and is the
            # only place the information exists.
            broken = [note for note in (lex_error, sem_error) if note]
            if broken:
                sdk.log(f"hybrid search ran no retriever: {'; '.join(broken)}",
                        level="warning")
                return sdk.fail(
                    f'Could not search for "{query}" — {"; ".join(broken)}.')
            return sdk.ok([], llm_summary=f'No results found for "{query}".')

        if lex_error or sem_error:
            # One stream is still a useful answer, but a *ranking* built from
            # half the evidence should say so rather than look complete.
            sdk.log(f"hybrid search degraded: {lex_error or sem_error}",
                    level="warning")

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


def _sub_search(sdk, name: str, kwargs) -> tuple:
    """``(results, problem)`` for one sub-tool.

    Either retriever may legitimately be missing — semantic search needs an
    embedder loaded, lexical search needs the FTS index built — and fusing one
    stream is still a useful answer. So a failure here degrades the ranking
    rather than failing the search, which is what the native version got from
    reading ``result.success`` off a returned envelope.

    What changed is that it no longer degrades *invisibly*. This returned a
    bare list and logged at ``debug``, so a search with both retrievers down
    was indistinguishable from a corpus with nothing in it — from here, from
    the caller, and from the user reading "No results found". The problem
    string is the caller's to report; only the caller knows whether losing one
    stream mattered.
    """
    try:
        data = sdk.tools.call(name, **kwargs)
    except sdk.Failed as failed:
        sdk.log(f"{name} unavailable: {failed.error}", level="debug")
        return [], f"{name} unavailable: {failed.error}"
    return (data if isinstance(data, list) else []), ""


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
