"""CLIP-based visual descriptor lookup. The agent's eyes."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from PIL import Image

from paths import DATA_DIR

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

logger = logging.getLogger("ClipInspect")

VOCAB_PATH = Path(__file__).with_name("data") / "visual_vocab.json"
CACHE_DIR = DATA_DIR / "canvas" / "vocab_cache"


def _load_vocab() -> dict[str, list[str]]:
    try:
        return json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _vocab_signature(vocab: dict, model_name: str) -> str:
    payload = json.dumps({"v": vocab, "m": model_name}, sort_keys=True).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


def _embed_vocab(embedder, vocab: dict[str, list[str]]) -> dict[str, tuple[list[str], "np.ndarray"]]:
    """Embed each term in the vocabulary. Result: {category: (terms, matrix)}."""
    if np is None:
        return {}
    out = {}
    for category, terms in vocab.items():
        prompts = [f"an artwork that feels {t}" for t in terms]
        vecs = embedder.encode(prompts)
        if vecs is None:
            continue
        out[category] = (list(terms), np.asarray(vecs, dtype="float32"))
    return out


def _get_or_build_vocab_embeddings(embedder) -> dict[str, tuple[list[str], "np.ndarray"]]:
    """Cache vocab embeddings on the embedder instance + on-disk npz."""
    if np is None:
        return {}
    cached = getattr(embedder, "_vocab_cache", None)
    if cached:
        return cached

    vocab = _load_vocab()
    if not vocab:
        embedder._vocab_cache = {}
        return {}

    sig = _vocab_signature(vocab, getattr(embedder, "model_name", "?"))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{sig}.npz"

    if cache_path.exists():
        try:
            data = np.load(cache_path, allow_pickle=True)
            result = {}
            for category in vocab:
                terms_key = f"{category}__terms"
                vecs_key = f"{category}__vecs"
                if terms_key in data and vecs_key in data:
                    result[category] = (list(data[terms_key]), data[vecs_key])
            embedder._vocab_cache = result
            return result
        except Exception as e:
            logger.warning("vocab cache load failed: %s", e)

    if not getattr(embedder, "loaded", False):
        embedder.load()
    result = _embed_vocab(embedder, vocab)
    try:
        save_payload = {}
        for category, (terms, vecs) in result.items():
            save_payload[f"{category}__terms"] = np.array(terms, dtype=object)
            save_payload[f"{category}__vecs"] = vecs
        np.savez(cache_path, **save_payload)
    except Exception as e:
        logger.warning("vocab cache save failed: %s", e)
    embedder._vocab_cache = result
    return result


def inspect(context, image_path: str | Path, top_k: int = 3) -> dict[str, list[str]]:
    """Return top descriptors per category for the given image. Empty dict on failure."""
    if np is None:
        return {}
    services = getattr(context, "services", {}) or {}
    embedder = services.get("image_embedder")
    if not embedder:
        return {}
    try:
        if not getattr(embedder, "loaded", False):
            embedder.load()
        if not getattr(embedder, "loaded", False):
            return {}
        vocab = _get_or_build_vocab_embeddings(embedder)
        if not vocab:
            return {}
        img = Image.open(image_path).convert("RGB")
        img_vec = embedder.encode([img])
        if img_vec is None or len(img_vec) == 0:
            return {}
        v = np.asarray(img_vec[0], dtype="float32")
        v = v / (np.linalg.norm(v) or 1.0)
        result: dict[str, list[str]] = {}
        for category, (terms, mat) in vocab.items():
            sims = mat @ v
            order = np.argsort(-sims)[: max(1, top_k)]
            result[category] = [terms[int(i)] for i in order]
        return result
    except Exception as e:
        logger.warning("clip inspect failed: %s", e)
        return {}


def format_inspect(descriptors: dict[str, list[str]]) -> str:
    """One-line text summary suitable for an LLM tool result."""
    if not descriptors:
        return ""
    parts = [f"{cat}=[{', '.join(words)}]" for cat, words in descriptors.items()]
    return "Looks like: " + "; ".join(parts) + "."
