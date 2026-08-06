"""The two ways a search query can be rejected by the thing it is handed to.

Same shape as ``test_store_attachment_tools``: kernel-side invariants that
happen to be *about* store files, with the store file as the input.

Both bugs here were found in one scheduled subagent run, and they share a
failure mode worth naming. Neither produced a bad *result* — a worse ranking,
a missing row — which is what a search is expected to risk. Each produced an
exception from a component that had every right to refuse: FTS5's parser, and
CLIP's positional embedding table. Both then surfaced through a background
agent nobody was watching, as a red line in a log.

Skips cleanly when no store ref is reachable.
"""

import ast
import re
import sqlite3

import pytest

# Aliases the guest package under the bare name ``guest``, which is how plugin
# source resolves its imports both in-process and in a child.
import sandbox  # noqa: F401
from tests.support import store_source

LEXICAL = "tools/tool_lexical_search.py"
EMBED = "services/service_embed.py"


def _source_or_skip(relative: str) -> str:
    text = store_source(relative)
    if text is None:
        pytest.skip(f"{relative} is not present on a local store ref")
    return text


def _functions(relative: str, *names) -> dict:
    """Module-level functions from a store file, without loading the plugin.

    The file imports ``guest.bases`` and, in the embedder's case, torch — so
    executing it whole would either need a box or a GPU. These functions are
    pure string handling and are exactly what is under test.
    """
    tree = ast.parse(_source_or_skip(relative))
    keep = [node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names]
    missing = set(names) - {node.name for node in keep}
    assert not missing, f"{relative} no longer defines {sorted(missing)}"
    namespace = {"re": re}
    exec(compile(ast.Module(body=keep, type_ignores=[]), relative, "exec"),
         namespace)
    return namespace


@pytest.fixture
def index():
    """A real FTS5 table, because the parser is the thing being asked."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        "CREATE VIRTUAL TABLE lexical_index USING fts5(content);"
        "INSERT INTO lexical_index VALUES ('the memory design doc');")
    return conn


def _match(conn, query):
    return conn.execute(
        "SELECT * FROM lexical_index WHERE lexical_index MATCH ?",
        [query]).fetchall()


# ──────────────────────────────────────────────────────────────────────
# FTS5: a guess about syntax, and the only thing that can refute it.
# ──────────────────────────────────────────────────────────────────────

def test_prose_containing_a_quote_is_not_valid_fts5(index):
    """The premise. Without this the retry below is solving nothing.

    ``_prepare_fts_query`` decides "is this FTS5?" by looking for operator
    characters, so an apostrophe anywhere in a query that also contains a
    double quote or a star reaches the parser raw. FTS5 reads ``'`` as a
    string delimiter and refuses.
    """
    fns = _functions(LEXICAL, "_prepare_fts_query", "_literal_fts_query",
                     "_is_fts_syntax_error")
    query = 'what\'s the "memory" design'

    with pytest.raises(sqlite3.OperationalError) as caught:
        _match(index, fns["_prepare_fts_query"](query))

    assert "syntax error" in str(caught.value)
    assert fns["_is_fts_syntax_error"](str(caught.value)), (
        "the retry is keyed off this message; if FTS5 rephrases it the tool "
        "goes back to reporting a parser error to somebody writing prose")


def test_the_literal_form_always_parses(index):
    """The fallback has to be something no query can turn back into syntax.

    Extracting ``\\w+`` leaves no quote, star, parenthesis or operator, so the
    retry cannot fail the way the first attempt did — which is what makes one
    retry enough rather than a loop.
    """
    fns = _functions(LEXICAL, "_literal_fts_query")
    for query in ['what\'s the "memory" design', "don't * stop", "a AND (b",
                  'NEAR("x" "y", 3', "-- ; DROP", "memory design"]:
        _match(index, fns["_literal_fts_query"](query) or "memory")


def test_deliberate_fts5_syntax_still_works(index):
    """The pass-through is worth keeping; it is only the guess that is unsound.

    A phrase query is the common case and must not be flattened into an AND of
    its words, which would rank differently and quietly.
    """
    fns = _functions(LEXICAL, "_prepare_fts_query")

    assert fns["_prepare_fts_query"]('"memory design"') == '"memory design"'
    assert len(_match(index, '"memory design"')) == 1
    assert not _match(index, '"design memory"'), "a phrase is ordered"


def test_only_a_parser_error_is_retried():
    """An unindexed corpus and a missing table must keep saying what they are.

    Retrying those would replace one honest failure with a second, more
    confusing one — and the second would be reported as if it were the cause.
    """
    fns = _functions(LEXICAL, "_is_fts_syntax_error")

    assert fns["_is_fts_syntax_error"]('fts5: syntax error near "\'"')
    assert not fns["_is_fts_syntax_error"]("no such table: lexical_index")
    assert not fns["_is_fts_syntax_error"]("denied: db.query")
    assert not fns["_is_fts_syntax_error"]("")


# ──────────────────────────────────────────────────────────────────────
# CLIP: a hard ceiling that raises instead of degrading.
# ──────────────────────────────────────────────────────────────────────

class _Tokenizer:
    """One token per word, which is enough to test the arithmetic."""

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, tokens, skip_special_tokens=False):
        return " ".join(tokens)


class _Model:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer


def _embedder(tokenizer=None):
    """An ``ImageEmbedder`` with just enough on it to call the trimmer.

    Built by hand rather than instantiated, because the class body imports
    torch. The method under test touches only ``self.model`` and
    ``self.model_name``.
    """
    source = _source_or_skip(EMBED)
    tree = ast.parse(source)
    method = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ImageEmbedder":
            method = next((item for item in node.body
                           if isinstance(item, ast.FunctionDef)
                           and item.name == "_fit_text_tower"), None)
    assert method is not None, "ImageEmbedder no longer trims its input"
    limit = next(node.value.value for node in tree.body
                 if isinstance(node, ast.Assign)
                 and getattr(node.targets[0], "id", "") == "_CLIP_TEXT_TOKENS")
    namespace = {"_CLIP_TEXT_TOKENS": limit}
    exec(compile(ast.Module(body=[method], type_ignores=[]), EMBED, "exec"),
         namespace)

    class Stub:
        model = _Model(tokenizer)
        model_name = "clip-ViT-B-32"
        _fit_text_tower = namespace["_fit_text_tower"]

    return Stub(), limit


class _Log:
    def log(self, *args, **kwargs):
        pass


def test_a_long_query_is_shortened_to_the_text_tower():
    """440 tokens into a 77-position table is the run that prompted this.

    CLIP raises rather than truncating, so the whole image half of a hybrid
    search failed — and it failed inside a scheduled subagent, where the only
    trace was a log line.
    """
    embedder, limit = _embedder(_Tokenizer())

    kept = embedder._fit_text_tower(_Log(), "word " * 440)

    assert len(kept.split()) == limit - 2, "BOS and EOS need a position each"


def test_a_short_query_is_returned_untouched():
    """Exactly, not merely equivalently — a round trip through a tokenizer
    normalises whitespace and case, and a query is the user's own words."""
    embedder, _ = _embedder(_Tokenizer())
    query = "  A  photo   of a Cat  "

    assert embedder._fit_text_tower(_Log(), query) == query


def test_an_unreachable_tokenizer_falls_back_pessimistically():
    """A model exposing no tokenizer must still not be handed 440 tokens.

    Words are not tokens and cannot be made into them, so the fallback takes
    half the budget: wrong in the direction that returns a result.
    """
    embedder, limit = _embedder(tokenizer=None)

    kept = embedder._fit_text_tower(_Log(), "word " * 440)

    assert len(kept.split()) == (limit - 2) // 2


def test_a_raising_tokenizer_falls_back_rather_than_propagating():
    """The trimmer sits in front of a foreign library; it cannot become the
    new reason the call fails."""
    class Broken:
        def encode(self, text, add_special_tokens=False):
            raise RuntimeError("no vocabulary loaded")

    embedder, limit = _embedder(Broken())

    kept = embedder._fit_text_tower(_Log(), "word " * 440)

    assert len(kept.split()) == (limit - 2) // 2
