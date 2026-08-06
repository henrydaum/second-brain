"""Text and image embeddings, from sentence-transformers models.

Two services in one file, which is the shape this has always had and now has a
sharper reason for: a file declaring several services gets **one box** (see
``sandbox/bridge.py``), so the two embedders share a process — one import of
torch, one accelerator context, one set of CUDA kernels compiled. Two files
would cost all of that twice for models that are never used simultaneously
anyway, since a box serializes its calls.

They are separate *services* rather than one service with a ``kind`` argument
because the pipeline reaches them by name: ``task_embed_text`` declares
``requires_services = ["text_embedder"]`` and must be schedulable when only
the text model is loaded.

**Vectors cross as bytes.** ``encode`` answers with one raw float32 buffer per
input — exactly what goes into the ``embedding`` BLOB column — rather than a
list of Python floats. Bytes cross the boundary natively (``guest/protocol.py``
packs them), so nothing here encodes anything by hand, and the value a caller
hands to ``sdk.db.write`` is the value sqlite stores.
"""

dependencies_files = []
dependencies_pip = ['numpy', 'pillow', 'sentence-transformers', 'torch']

from guest.bases import BaseService

#: Width of CLIP's positional embedding table, and a hard ceiling rather than a
#: tuning knob: the text tower has exactly this many positions, so a longer
#: sequence raises instead of degrading. Every CLIP variant sentence-
#: transformers ships uses 77.
_CLIP_TEXT_TOKENS = 77

#: Model weights live at the **root of DATA_DIR**, one directory per model,
#: named after it — ``DATA_DIR/BAAI_bge-small-en-v1.5``.
#:
#: They used to sit beside the plugin file, which the tree layout turned into
#: ``installed/services/``. That is a *plugin root*: something globs
#: ``service_*.py`` there, the package manager installs and removes files
#: there, and a multi-gigabyte model directory in the middle of it is at best
#: clutter and at worst something an uninstall walks into. Weights are data,
#: not code, and DATA_DIR is where data goes.
#:
#: There is deliberately only one location now. A second "bundled" path
#: existed so weights could be shipped next to the plugin; moved out of the
#: services root it would be the same directory as this one, so it collapsed
#: rather than being dropped — pre-placing weights here still skips the
#: download exactly as it did.
WEIGHTS_NOTE = "DATA_DIR/<model_name_with_underscores>"


def _exists(sdk, path) -> bool:
    """Whether a path is there.

    ``sdk.fs.list`` *fails* on a missing path rather than answering with an
    empty list, and the SDK turns a failed Request into a raise — so
    ``if sdk.fs.list(p)`` does not test existence, it throws. This is the
    idiom the store's file-editing tools already use.
    """
    try:
        return bool(sdk.fs.list(path))
    except sdk.Failed:
        return False


class _SentenceTransformerEmbedder:
    """The half both embedders share.

    Deliberately not a ``BaseService`` subclass: the validator counts plugin
    classes by what they subclass, and a shared implementation that registered
    itself as a third service would be an empty service the kernel tries to
    load.
    """

    #: Set by each subclass — the config key naming its model.
    setting = ""
    #: Set by each subclass — the model to use when nothing is configured.
    fallback = ""

    def __init__(self):
        """Nothing is loaded until start()."""
        self.model = None
        self.model_name = self.fallback
        self.device = "cpu"
        self.chunk_size = 512

    # ── lifecycle ───────────────────────────────────────────────────

    def start(self, sdk):
        """Resolve the model, find or fetch its weights, and load it.

        Two things the native version did are gone, both because they were
        working around problems this file no longer has:

        - **The HuggingFace offline environment dance.** It set
          ``HF_HUB_OFFLINE`` and ``TRANSFORMERS_OFFLINE`` around loading,
          which reaches ``os.environ`` and is refused. It was also redundant:
          both load paths already pass ``local_files_only=True``, which is the
          actual guarantee the env vars were reaching for.
        - **The connectivity probe.** ``is_connected()`` opened a socket to
          8.8.8.8 to decide whether attempting a download was worthwhile.
          Attempting it and handling the failure is both simpler and more
          accurate — a machine with DNS and no route to HuggingFace passed the
          probe and failed the download anyway.
        """
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_name = str(
            sdk.config.read(self.setting) or self.fallback).strip() \
            or self.fallback
        self.chunk_size = int(sdk.config.read("embed_chunk_size") or 512)
        wants_cuda = bool(sdk.config.read("embed_use_cuda"))
        self.device = "cuda" if wants_cuda and torch.cuda.is_available() \
            else "cpu"

        # One location, at the root of DATA_DIR — see WEIGHTS_NOTE.
        weights = sdk.path.join(sdk.paths.get("data"),
                                self.model_name.replace("/", "_"))

        if _exists(sdk, weights):
            sdk.log(f"using local weights for {self.model_name}")
        elif not self._download(sdk, weights):
            return False

        self.model = SentenceTransformer(weights, device=self.device,
                                         local_files_only=True)
        self.model.max_seq_length = self.chunk_size
        sdk.log(f"{self.model_name} loaded on {self.device}")
        return True

    def _download(self, sdk, destination) -> bool:
        """Fetch the weights and save them where we will look next time.

        The download is the library's own network I/O, past the kernel's
        reach — it holds no credential and the host is HuggingFace's, which is
        the documented limit of what the boundary covers. A partial download
        is cleaned up, so a failure here does not poison the next attempt into
        loading half a model.
        """
        from sentence_transformers import SentenceTransformer

        sdk.log(f"downloading {self.model_name}; this may take a while")
        try:
            fetched = SentenceTransformer(self.model_name)
            fetched.save(destination)
            sdk.log(f"saved {self.model_name} to {destination}")
            return True
        except Exception as exc:
            sdk.log(f"download of {self.model_name} failed: {exc}",
                    level="error")
            try:
                sdk.fs.delete(destination)
            except Exception:
                pass    # nothing was written, or it is not ours to remove
            return False

    def stop(self, sdk):
        """Drop the model.

        No ``gc.collect()`` and no ``torch.cuda.empty_cache()``: this service
        is its process, and closing the box ends it. Both calls existed to
        reclaim memory inside a process that outlived the service.
        """
        self.model = None
        return None

    # ── exports ─────────────────────────────────────────────────────

    def describe(self, sdk):
        """Which model is loaded, and how wide its vectors are.

        Load-bearing rather than informational. The bridge names an adapter
        after the *service*, so ``embedder.model_name`` reads "text_embedder";
        but the value stored in the ``model_name`` column — and matched by
        ``WHERE model_name = ?`` when searching — has to be the model's own
        id, or a search finds nothing and cannot say why.
        """
        return {
            "model_name": self.model_name,
            "dim": self._dim(),
            "device": self.device,
            "loaded": self.model is not None,
        }

    def _dim(self):
        """Vector width, across the sentence-transformers rename.

        ``get_sentence_embedding_dimension`` was renamed to
        ``get_embedding_dimension`` and now warns. Prefer the new name;
        fall back to the old one for installs that predate it.
        """
        if self.model is None:
            return 0
        getter = (getattr(self.model, "get_embedding_dimension", None)
                  or self.model.get_sentence_embedding_dimension)
        return getter()

    def encode(self, sdk, inputs):
        """Embed a batch of strings, returning one float32 buffer per input.

        A single string is accepted and answered with a one-item list, so a
        caller never has to branch on how many it asked about.

        ``ImageEmbedder`` overrides this: its inputs are paths that have to
        become PIL objects before the model sees them, and the base class
        deliberately does not try to guess which it was handed. See that
        override for why guessing would fail silently.

        No ``except Exception`` around the encode. A raise here becomes a
        failed Result with the traceback intact; catching it to return ``None``
        cost the cause and left the caller reporting "embedder returned None",
        which is what the native version did and what made encode failures
        undiagnosable.
        """
        if self.model is None:
            raise RuntimeError(f"{self.model_name} is not loaded")

        batch = [inputs] if isinstance(inputs, str) else list(inputs)
        if not batch:
            return []
        return self._to_buffers(sdk, batch)

    def _to_buffers(self, sdk, batch) -> list:
        """Run the model and hand back raw float32 buffers.

        The vectors are L2-normalized, which is what makes a dot product a
        cosine similarity for whoever searches them later.
        """
        import numpy as np

        vectors = self.model.encode(batch, normalize_embeddings=True,
                                    convert_to_numpy=True)
        sdk.log(f"encoded {len(batch)} input(s) with {self.model_name}",
                level="debug")
        # ``astype`` rather than trusting the model's dtype: the stored column
        # is read back with ``np.frombuffer(..., dtype=np.float32)``, and a
        # model that answered in float16 would produce buffers of the wrong
        # length that only fail at search time.
        return [np.asarray(vector, dtype=np.float32).tobytes()
                for vector in vectors]


class TextEmbedder(_SentenceTransformerEmbedder, BaseService):
    """Embed text chunks for semantic search."""

    name = "text_embedder"
    description = "Embed text into vectors for semantic search."
    shared = True
    timeout = 300
    requests = ["config.read", "paths.get", "fs.list", "fs.delete"]
    exports = ["encode", "describe"]
    setting = "embed_text_model_name"
    fallback = "BAAI/bge-small-en-v1.5"
    config_settings = [
        ("Text Embedding Model", "embed_text_model_name",
         "SentenceTransformer model for text embeddings.",
         "BAAI/bge-small-en-v1.5",
         {"type": "text"}),

        ("GPU Acceleration", "embed_use_cuda",
         "Use the GPU for embedding. Provides a significant speed-up.",
         True,
         {"type": "bool"}),

        ("Chunk Size", "embed_chunk_size",
         "Size in tokens for text splitting. Smaller chunks store specific "
         "facts; larger chunks preserve more context.",
         512,
         {"type": "slider", "range": (64, 2048, 31), "is_float": False}),
    ]


class ImageEmbedder(_SentenceTransformerEmbedder, BaseService):
    """Embed images with CLIP, for searching pictures by description."""

    name = "image_embedder"
    description = "Embed images into vectors for semantic search."
    shared = True
    timeout = 300
    requests = ["config.read", "paths.get", "fs.list", "fs.delete"]
    exports = ["encode", "encode_text", "describe"]
    setting = "embed_image_model_name"
    fallback = "clip-ViT-B-32"
    config_settings = [
        ("Image Embedding Model", "embed_image_model_name",
         "CLIP model for image embeddings.",
         "clip-ViT-B-32",
         {"type": "text"}),
    ]

    def encode(self, sdk, inputs):
        """Embed a batch of image *paths*, one float32 buffer per image.

        Opening the files here rather than inheriting the text encode is not
        a convenience — it is the difference between working and silently
        not. CLIP is multimodal, and ``SentenceTransformer.encode`` decides
        what a batch *is* from the type of its members: hand it strings and it
        embeds them as text, cheerfully and without error. A batch of paths
        would therefore produce perfectly valid embeddings *of the filenames*,
        which nothing would flag, no test on shapes would catch, and which
        would surface only as image search that returns nonsense.

        Paths rather than image bytes because the caller has already written
        each image to scratch: a file is something both boxes can name, while
        a PIL object is precisely the thing that cannot cross between them.
        """
        if self.model is None:
            raise RuntimeError(f"{self.model_name} is not loaded")

        from PIL import Image

        batch = [inputs] if isinstance(inputs, str) else list(inputs)
        if not batch:
            return []

        opened = []
        try:
            for path in batch:
                image = Image.open(path)
                image.load()          # before the file handle goes away
                opened.append(image.convert("RGB")
                              if image.mode != "RGB" else image)
            return self._to_buffers(sdk, opened)
        finally:
            for image in opened:
                try:
                    image.close()
                except Exception:     # noqa: BLE001 - closing is best effort
                    pass

    def encode_text(self, sdk, inputs):
        """Embed *text* into the same space the images live in.

        CLIP has two towers writing into one shared space, which is the whole
        reason searching pictures by description works: a caption's vector
        lands near the vectors of images it describes. So a query needs the
        text tower while the corpus needed the image tower, and the two are
        reached from the same model by handing ``encode`` different types.

        This exists as a separate export because :meth:`encode` no longer
        guesses. It used to, and that was the bug — strings went to the text
        tower silently, so *indexing* a batch of image paths embedded the
        filenames. Splitting the two makes each caller say which it meant, and
        the failure mode of getting it wrong is now a missing file rather than
        an index full of plausible nonsense.
        """
        if self.model is None:
            raise RuntimeError(f"{self.model_name} is not loaded")

        batch = [inputs] if isinstance(inputs, str) else list(inputs)
        if not batch:
            return []
        return self._to_buffers(sdk, [self._fit_text_tower(sdk, text)
                                      for text in batch])

    def _fit_text_tower(self, sdk, text):
        """Shorten one string to something CLIP's text tower will accept.

        CLIP's positional embedding table is 77 wide and the model *raises*
        rather than truncating, so a long query does not degrade — it fails
        outright, and what failed here was a background subagent's image
        search that nobody was watching. Any query long enough to hit this is
        well past the point where CLIP is doing anything with the tail anyway:
        the text tower was trained on captions and collapses whatever it gets
        into a single vector.

        Truncation is by **token**, asked of the model's own tokenizer,
        because that is the only counter that agrees with the thing that
        raises. Word count is not a proxy — a path, a hash or a CJK sentence
        can be several tokens per word — so the word split is a last-resort
        fallback for a model whose tokenizer is not reachable, and it is
        deliberately pessimistic.
        """
        limit = _CLIP_TEXT_TOKENS - 2          # BOS and EOS take one each
        tokenizer = getattr(self.model, "tokenizer", None)
        encode = getattr(tokenizer, "encode", None)
        decode = getattr(tokenizer, "decode", None)
        try:
            tokens = encode(text, add_special_tokens=False) if encode else None
        except Exception:                      # noqa: BLE001 - foreign library
            tokens = None
        if tokens is not None and decode is not None:
            if len(tokens) <= limit:
                return text
            kept = decode(tokens[:limit], skip_special_tokens=True)
        else:
            words = text.split()
            if len(words) <= limit // 2:
                return text
            kept = " ".join(words[:limit // 2])
        sdk.log(f"query shortened to fit {self.model_name}'s "
                f"{_CLIP_TEXT_TOKENS}-token text tower", level="debug")
        return kept
