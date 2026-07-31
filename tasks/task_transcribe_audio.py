"""Transcribe audio files with the Whisper service.

The thinnest task in the bundle, and deliberately so: the audio never moves.
The service holds the model, the task names a path, and a string comes back —
which is the shape every heavy modality wants and the reason ``whisper`` is a
service rather than a parser call.

No parser is provisioned here for the same reason. Decoding audio produces a
waveform, and a waveform is an *intermediate* on its way to text; the service
does that decode inside its own box, beside the model that consumes it.
"""

dependencies_files = ['services/service_whisper.py']
dependencies_pip = []

import time

from guest.bases import BaseTask
from guest.parsing import basename


class TranscribeAudio(BaseTask):
    """Turn speech into text, one file at a time."""

    name = "transcribe_audio"
    modalities = ["audio"]
    reads = []
    writes = ["audio_transcripts"]
    requires_services = ["whisper"]
    requests = ["service.call"]
    output_schema = """
        CREATE TABLE IF NOT EXISTS audio_transcripts (
            path TEXT PRIMARY KEY,
            content TEXT,
            char_count INTEGER,
            model_name TEXT,
            transcribed_at REAL
        );
    """
    batch_size = 4
    # Whisper saturates whatever it is given, so a second concurrent file
    # makes both slower rather than either faster.
    max_workers = 1
    timeout = 600

    def run(self, sdk, paths):
        """Transcribe each path.

        No "is the service loaded?" guard: the orchestrator's ``_services_ready``
        already refuses to dispatch a task whose ``requires_services`` are not
        loaded, so a guard here could only ever be checking something the
        kernel had checked a moment earlier.
        """
        # Asked once for the batch rather than per file. The bridge names an
        # adapter after the *service*, so reading ``model_name`` off it would
        # record the string "whisper" against every transcript instead of the
        # model that produced it.
        described = sdk.services.call("whisper", "describe") or {}
        model_name = described.get("model_name") or "unknown"

        now = time.time()
        outcomes = []

        for path in paths:
            try:
                text = (sdk.services.call("whisper", "transcribe",
                                          audio_path=path) or "").strip()
            except sdk.Failed as failed:
                outcomes.append({"ok": False, "error": str(failed)})
                continue

            sdk.log(f"transcribed {len(text)} chars from {basename(path)}"
                    if text else f"no speech detected in {basename(path)}")

            outcomes.append({
                "ok": True,
                "data": [{
                    "path": path,
                    "content": text,
                    "char_count": len(text),
                    "model_name": model_name,
                    "transcribed_at": now,
                }],
            })

        return sdk.ok(per_path=outcomes)
