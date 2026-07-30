"""Local audio transcription, via faster-whisper.

The model is downloaded on first load and cached under DATA_DIR. Everything
this service touches outside its own process is a Request; the transcription
itself runs inside the box, on weights the box loaded, and nothing about the
audio ever crosses the boundary — the caller names a path and gets text back.

**A word about the deadline.** A resident box's per-call ceiling is the
declared ``timeout``, clamped by the kernel to 600 seconds, and it counts
*guest* time. Transcription makes no Requests while it runs, so it is charged
in full — and exceeding the ceiling does not merely fail the call, it marks
the box dead and the service has to be reloaded. In practice that is a limit
of roughly one to two hours of audio on CPU, less on a slow machine, and it
applies to the first-run model download too (which runs under the box's
*start* budget, the same 600 seconds). Splitting a long recording before
handing it over is the way around it.
"""

dependencies_files = []
dependencies_pip = ['faster-whisper']

import re

from guest.bases import BaseService


def _exists(sdk, path) -> bool:
    """Whether a path is there.

    ``sdk.fs.list`` *fails* on a missing path rather than answering with an
    empty list, and the SDK turns a failed Request into a raise — so
    ``if sdk.fs.list(p)`` does not test existence, it throws.
    """
    try:
        return bool(sdk.fs.list(path))
    except sdk.Failed:
        return False


class WhisperService(BaseService):
    """Transcribe audio files to text, on this machine."""

    name = "whisper"
    description = "Transcribe audio to text locally with faster-whisper."
    shared = True
    # The per-call ceiling, at the kernel's maximum. See the module docstring.
    timeout = 600
    requests = ["config.read", "paths.get", "fs.list"]
    exports = ["transcribe", "describe"]
    config_settings = [
        ("Whisper Model", "whisper_model_name",
         "faster-whisper model size. Larger is more accurate and slower: "
         "tiny, base, small, medium, large-v3.",
         "base",
         {"type": "text"}),

        ("GPU Acceleration", "whisper_use_cuda",
         "Use the GPU for transcription when one is available. Falls back to "
         "CPU on its own if CUDA cannot be used.",
         True,
         {"type": "bool"}),
    ]

    def __init__(self):
        """Nothing is loaded until start()."""
        self.model = None
        self.model_name = "base"
        self.device = "cpu"

    # ── lifecycle ───────────────────────────────────────────────────

    def start(self, sdk):
        """Load the model, downloading it on first use.

        CUDA is *attempted* rather than detected. The native version imported
        torch for one call — ``torch.cuda.is_available()`` — which meant a
        multi-gigabyte dependency to answer a question the failure already
        answers. faster-whisper runs on CTranslate2 and does not otherwise
        need torch at all, so trying and falling back is both cheaper and
        more accurate: a CUDA that exists but cannot be used reports
        available, and only the attempt finds out.
        """
        from faster_whisper import WhisperModel

        self.model_name = str(
            sdk.config.read("whisper_model_name") or "base").strip() or "base"
        wants_cuda = bool(sdk.config.read("whisper_use_cuda"))

        # The download root is created by huggingface_hub on the way past.
        # Asking the kernel to make it first would be a write outside scratch,
        # which raises an approval dialog at boot for a directory the library
        # is about to create anyway.
        download_root = sdk.path.join(sdk.paths.get("data"), "whisper")

        for device in (["cuda", "cpu"] if wants_cuda else ["cpu"]):
            try:
                self.model = WhisperModel(self.model_name, device=device,
                                          compute_type="auto",
                                          download_root=download_root)
                self.device = device
                sdk.log(f"whisper {self.model_name} loaded on {device}")
                return True
            except Exception as exc:
                sdk.log(f"whisper could not load on {device}: {exc}",
                        level="warning" if device == "cuda" else "error")
        return False

    def stop(self, sdk):
        """Drop the model.

        No ``gc.collect()``, and no ``torch.cuda.empty_cache()``. Both existed
        to reclaim memory inside a process that outlived the service; this
        service *is* its process, and closing the box ends it. The OS reclaims
        far more thoroughly than either call did.
        """
        self.model = None
        return None

    # ── exports ─────────────────────────────────────────────────────

    def describe(self, sdk):
        """What this service loaded.

        Needed because the bridge names an adapter after the *service*, so a
        caller reading ``whisper.model_name`` would get the string "whisper".
        The model actually used is recorded against every transcript, so it
        has to be askable.
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "loaded": self.model is not None,
        }

    def transcribe(self, sdk, audio_path):
        """Transcribe one file. Returns the full transcript, or "".

        Non-speech is filtered twice: Silero VAD plus a per-segment
        ``no_speech_prob`` threshold on the way through, then a heuristic pass
        over the result for Whisper's classic hallucinations.
        """
        if self.model is None:
            return ""
        if not _exists(sdk, audio_path):
            sdk.log(f"audio file not found: {audio_path}", level="warning")
            return ""

        name = sdk.path.name(audio_path)
        sdk.log(f"transcribing {name}")
        segments, info = self.model.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )

        kept = [
            segment.text.strip()
            for segment in segments
            if segment.no_speech_prob < 0.6 and segment.text.strip()
        ]
        text = " ".join(kept).strip()

        if text and _looks_like_hallucination(text):
            sdk.log(f"discarded likely-hallucinated transcript for {name}: "
                    f"{text[:80]!r}")
            text = ""

        sdk.log(f"transcribed {name}: {len(text)} chars, "
                f"language={info.language} ({info.language_probability:.0%})")
        return text


def _looks_like_hallucination(text: str) -> bool:
    """Heuristic for Whisper's classic non-speech hallucinations.

    Music, silence and ambient noise get transcribed as repeated
    YouTube-trained phrases — "Thank you." or "Thanks for watching." over and
    over. Two signals:

      1. The unique-word ratio is very low (one phrase repeated).
      2. The output is short and matches a known canned phrase.
    """
    stripped = text.strip()
    if not stripped:
        return False

    words = re.findall(r"\w+", stripped.lower())
    if not words:
        return False

    canned = {
        "thank you", "thank you.", "thanks for watching",
        "thanks for watching.", "you", "bye", "bye.", ".",
    }
    if stripped.lower() in canned:
        return True

    # Repetition: under 30% unique across at least 8 words is a strong signal.
    if len(words) >= 8:
        return len(set(words)) / len(words) < 0.3

    return False
