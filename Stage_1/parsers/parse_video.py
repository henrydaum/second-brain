import logging
from pathlib import Path
from Stage_1.ParseResult import ParseResult
import Stage_1.registry as registry

logger = logging.getLogger(__name__)

# Returns a standardized av.Container object

"""
Video parsers.

Returns ParseResult(modality="video", output=av.Container).

av.Container is a lazy handle to the video file. Nothing is decoded
until a task explicitly pulls from a stream. Tasks pick what they need:

    container = result.output
    audio_stream = container.streams.audio[0]   # for transcription
    video_stream = container.streams.video[0]   # for frame extraction

The parser validates the file is a real video, extracts lightweight
metadata (duration, resolution, codecs, stream counts), and returns
the open container. The calling task is responsible for closing it
when done.

Also detects embedded audio and subtitle tracks via also_contains,
so the orchestrator can queue transcription or subtitle extraction
tasks as needed.

Requires: av (PyAV)
"""


def parse_video(path: str, config: dict) -> ParseResult:
    """
    Open a video file and return an av.Container handle.

    The container is lazy — no frames or audio are decoded until
    a task explicitly iterates a stream.
    """
    try:
        import av
    except ImportError:
        logger.debug("PyAV not installed")
        return ParseResult.failed("PyAV not installed", modality="video")

    try:
        container = av.open(path)

        # --- Metadata extraction (cheap, reads headers only) ---
        metadata = {}
        also_contains = []

        # Video stream info
        if container.streams.video:
            vs = container.streams.video[0]
            metadata["width"] = vs.codec_context.width
            metadata["height"] = vs.codec_context.height
            metadata["video_codec"] = vs.codec_context.name
            metadata["fps"] = float(vs.average_rate) if vs.average_rate else 0.0
            metadata["frame_count"] = vs.frames if vs.frames else 0

        # Duration
        if container.duration:
            metadata["duration_seconds"] = float(container.duration) / av.time_base
        elif container.streams.video:
            # Fallback: estimate from video stream
            vs = container.streams.video[0]
            if vs.duration and vs.time_base:
                metadata["duration_seconds"] = float(vs.duration * vs.time_base)

        # Audio stream detection
        metadata["audio_streams"] = len(container.streams.audio)
        if container.streams.audio:
            audio = container.streams.audio[0]
            metadata["audio_codec"] = audio.codec_context.name
            metadata["audio_sample_rate"] = audio.codec_context.sample_rate
            metadata["audio_channels"] = audio.codec_context.channels
            also_contains.append("audio")

        # Subtitle stream detection
        metadata["subtitle_streams"] = len(container.streams.subtitles)
        if container.streams.subtitles:
            also_contains.append("text")

        metadata["stream_count"] = len(container.streams)

        return ParseResult(
            modality="video",
            output=container,
            metadata=metadata,
            also_contains=also_contains,
        )
    except av.AVError as e:
        logger.debug(f"Failed to open video: {e}")
        return ParseResult.failed(f"Failed to open video: {e}", modality="video")
    except Exception as e:
        logger.debug(f"Failed to parse {path}: {e}")
        return ParseResult.failed(str(e), modality="video")


registry.register([
    ".mp4", ".mkv", ".avi", ".mov",
    ".webm", ".flv", ".wmv",
], "video", parse_video)