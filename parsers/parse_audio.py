"""Attachment parsing helpers for audio inputs."""


dependencies_files = []
dependencies_pip = ['librosa', 'numpy', 'soundfile']

# soundfile/librosa decode audio in C and read the file themselves, so their
# actions cannot be turned into Requests. A process boundary is what actually
# contains them — and a malformed file that kills a box is a failed parse
# rather than a dead kernel. That boundary is not declared here: the kernel
# derives it from the imports it can see (``sandbox/isolation.py``), because
# code being contained may not be the authority on its own containment.

import json
from guest.parsing import ParseResult, basename, register


# Returns a standardized np.ndarray + sample rate integer

"""
Audio parsers.

Returns ParseResult(modality="audio", audio=(np.ndarray, sample_rate)).

The numpy array is the waveform: a 1D float32 array of amplitude samples.
The int is the sample rate (e.g. 44100 for CD, 16000 for Whisper).

If soundfile isn't installed, falls back to metadata-only via ffprobe.
Tasks that need the waveform will fail gracefully with a clear error.
"""


def _probe_metadata(sdk, path: str) -> dict:
    """
    Extract audio metadata via ffprobe. Returns empty dict if unavailable.
    This is lightweight — no decoding, just reads the file header.

    Running a process is a Request, so the user is asked once for a parser
    that shells out — which is the right thing to be asked about.
    """
    try:
        result = sdk.proc.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format", "-show_streams",
                str(path),
            ],
            timeout=10,
        )

        if result["code"] != 0:
            return {}

        probe = json.loads(result["stdout"])
        fmt = probe.get("format", {})
        metadata = {
            "duration_seconds": float(fmt.get("duration", 0)),
            "format_name": fmt.get("format_name", ""),
            "bit_rate": int(fmt.get("bit_rate", 0)) if fmt.get("bit_rate") else 0,
        }

        # Find the audio stream
        for stream in probe.get("streams", []):
            if stream.get("codec_type") == "audio":
                metadata["codec"] = stream.get("codec_name", "")
                metadata["sample_rate"] = int(stream.get("sample_rate", 0))
                metadata["channels"] = int(stream.get("channels", 0))
                break

        return metadata

    except sdk.Denied:
        sdk.log("not allowed to run ffprobe", level="debug")
        return {}
    except Exception as e:
        sdk.log(f"ffprobe failed: {e}", level="debug")
        return {}


def parse_audio(sdk, path: str, config: dict = None) -> ParseResult:
    """
    Load an audio file as (np.ndarray, sample_rate).

    Uses soundfile as the primary loader (handles WAV, FLAC, OGG natively).
    Falls back to librosa for MP3 and other formats that need decoding.
    Falls back to metadata-only if neither is available.
    """
    metadata = _probe_metadata(sdk, path)

    # Target sample rate: Whisper uses 16000, music apps use 44100.
    # Default to the file's native rate (None = no resampling).
    target_sr = config.get("sample_rate", None)

    # Try soundfile first (fast, no resampling)
    try:
        import soundfile as sf
        import numpy as np

        data, sr = sf.read(path, dtype="float32")

        # Convert stereo to mono if needed
        if data.ndim > 1:
            data = data.mean(axis=1)

        metadata["sample_rate"] = sr
        metadata["samples"] = len(data)
        metadata["duration_seconds"] = len(data) / sr

        return ParseResult(
            modality="audio",
            output=(data, sr),
            metadata=metadata,
        )
    except Exception as sf_err:
        sdk.log(f"soundfile failed for {basename(path)}: {sf_err}", level="debug")

    # Fallback: librosa (handles MP3, M4A, etc. via ffmpeg)
    try:
        import librosa
        import numpy as np

        data, sr = librosa.load(path, sr=target_sr, mono=True)

        metadata["sample_rate"] = sr
        metadata["samples"] = len(data)
        metadata["duration_seconds"] = len(data) / sr

        return ParseResult(
            modality="audio",
            output=(data, sr),
            metadata=metadata,
        )
    except ImportError:
        sdk.log("Neither soundfile nor librosa available", level="debug")
    except Exception as lr_err:
        sdk.log(f"librosa failed for {basename(path)}: {lr_err}", level="debug")

    # Last resort: metadata only, no waveform
    if metadata:
        return ParseResult(
            modality="audio",
            output=None,
            metadata={**metadata, "warning": "No audio loader available — metadata only"},
        )

    return ParseResult.failed(
        "No audio loader available (install soundfile or librosa)",
        modality="audio",
    )


def parse_audio_text(sdk, path: str, config: dict = None) -> ParseResult:
    """
    Transcribe an audio file to text by delegating to the whisper service.

    This is the "text" rendering the attachment system asks for
    (parse_attachment calls parser.parse(path, "text")), so a voice note
    attached mid-conversation is transcribed synchronously and rides into
    the turn as parsed_text instead of a bare file pointer. Soft dependency:
    if the Whisper package isn't installed, this fails cleanly and the
    attachment falls back to pointer routing.
    """
    try:
        text = (sdk.services.call("whisper", "transcribe",
                                  audio_path=str(path)) or "").strip()
    except sdk.Failed as exc:
        # Not installed, not loaded, or it broke. All three are the same
        # answer here: there is no text rendering for this file, and the
        # attachment falls back to pointer routing.
        return ParseResult.failed(
            f"Whisper is unavailable — no text rendering for audio ({exc})",
            modality="text",
        )

    metadata = _probe_metadata(sdk, path)
    try:
        described = sdk.services.call("whisper", "describe") or {}
        metadata["model_name"] = described.get("model_name") or "unknown"
    except sdk.Failed:
        metadata["model_name"] = "unknown"
    limit = (config or {}).get("max_chars")
    if limit and len(text) > limit:
        text = text[:limit]
        metadata["truncated"] = True
    metadata["char_count"] = len(text)
    if not text:
        metadata["warning"] = "No speech detected"

    return ParseResult(
        modality="text",
        output=text or None,
        metadata=metadata,
    )


_AUDIO_EXTENSIONS = [
    ".mp3", ".wav", ".flac", ".m4a",
    ".aac", ".ogg", ".oga", ".wma", ".opus",
]

# "audio" must register first: the first modality registered for an
# extension becomes its default, and get_modality must keep answering
# "audio" so native-capable LLMs still get the raw file inlined.
register(_AUDIO_EXTENSIONS, "audio", parse_audio)
register(_AUDIO_EXTENSIONS, "text", parse_audio_text)
