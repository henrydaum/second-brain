"""Audio-to-text parser: transcribe an audio attachment via the Whisper service.

This is the missing link in the attachment text path. The attachment system asks
the parser for ``parse(path, "text")`` and injects whatever text comes back into
the user message (the kernel's native -> parsed-text -> pointer routing). The
companion ``parse_audio`` helper registers audio extensions under the ``"audio"``
modality and returns a *waveform*, so on its own an audio attachment never yields
text and the agent only sees a file pointer.

Registering the same extensions under the ``"text"`` modality here closes that
gap: when Whisper is loaded, a voice note is transcribed at attachment time and
the transcript rides along in the user message. It degrades gracefully — if the
Whisper service is not loaded (or there is no detectable speech), it fails the
parse so the attachment falls back to the pointer blurb instead of injecting an
empty string. Heavy transcription only happens inside ``transcribe`` on the
Whisper service, so importing this helper stays cheap.
"""

dependencies_files = ['services/service_whisper.py']
dependencies_pip = []


from guest.parsing import ParseResult, basename, register


# Mirror parse_audio's extension set so every audio attachment can be transcribed.
AUDIO_EXTENSIONS = [
    ".mp3", ".wav", ".flac", ".m4a",
    ".aac", ".ogg", ".oga", ".wma", ".opus",
]


def parse_voice(sdk, path: str, config: dict = None) -> ParseResult:
    """Return the spoken contents of an audio file as text via the Whisper service."""
    try:
        text = (sdk.services.call("whisper", "transcribe",
                                  audio_path=str(path)) or "").strip()
    except sdk.Failed as e:
        # Not installed, not loaded, or it broke. One answer for all three:
        # leave the audio untranscribed and let the attachment fall back to
        # the pointer blurb.
        sdk.log(f"Transcription unavailable for {basename(path)}: {e}",
                level="warning")
        return ParseResult.failed(
            f"Whisper is unavailable — audio left untranscribed ({e}).",
            modality="text",
        )

    if not text:
        # No detectable speech (or filtered as a hallucination). Fail the parse so
        # the attachment falls back to the pointer blurb rather than empty text.
        return ParseResult.failed("No detectable speech in audio.", modality="text")

    return ParseResult(
        modality="text",
        output=text,
        metadata={"source": "whisper", "char_count": len(text)},
    )


register(AUDIO_EXTENSIONS, "text", parse_voice)
