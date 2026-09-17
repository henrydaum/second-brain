/**
 * Recording a voice note in the browser.
 *
 * This produces a *file*, not a transcript. Nothing here tries to understand
 * speech: the recording becomes an ordinary attachment, goes across by the same
 * route as a dragged PDF, and Second Brain's own audio parser is what turns it
 * into words. That keeps one transcription in the system rather than two that
 * can disagree, and it means the audio itself survives in the conversation —
 * a voice note is a record, not just a slow way of typing.
 *
 * **Why re-encode to WAV instead of sending what the browser recorded.** A
 * `MediaRecorder` hands back whatever container the browser felt like: Chrome
 * gives `audio/webm`, and `.webm` is a *video* extension as far as the kernel's
 * modality map is concerned (`parsing/registry.py`), so an audio parser would
 * never be offered it. WAV is unambiguous. It is also exactly what a speech
 * model wants — 16 kHz, one channel — so the file arrives already in the shape
 * the far end would have converted it to anyway.
 *
 * The cost is size: uncompressed 16 kHz mono is about 2 MB a minute. For a
 * voice note that is nothing next to the ~11 MB one Request carries, and
 * `uploadToHost` chunks it regardless.
 */

/** 16 kHz mono is what speech recognition runs at; more is thrown away. */
const SAMPLE_RATE = 16000;

export type Recording = {
  /** Stop, and answer with the finished file. */
  stop: () => Promise<File>;
  /** Stop and throw the audio away. */
  cancel: () => void;
};

/** Whether this browser can record at all. `mediaDevices` is missing outside a
 *  secure context — plain http on anything but localhost — so the button has
 *  something honest to key off rather than failing on click. */
export function canRecord(): boolean {
  return (
    typeof navigator !== "undefined" &&
    typeof navigator.mediaDevices?.getUserMedia === "function" &&
    typeof MediaRecorder !== "undefined"
  );
}

/**
 * Ask for the microphone and start recording.
 *
 * Rejects if permission is refused, which is the caller's to report — a mic
 * button that silently does nothing is the bug this whole change is about.
 */
export async function record(): Promise<Recording> {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const recorder = new MediaRecorder(stream);
  const chunks: Blob[] = [];

  recorder.ondataavailable = (event) => {
    if (event.data.size > 0) chunks.push(event.data);
  };
  recorder.start();

  // Releasing the tracks is what turns off the browser's recording indicator.
  // Skipping it leaves the tab looking like it is still listening, which is
  // both alarming and true.
  const release = () => {
    for (const track of stream.getTracks()) track.stop();
  };

  return {
    stop: () =>
      new Promise<File>((resolve, reject) => {
        recorder.onstop = () => {
          release();
          toWav(new Blob(chunks, { type: recorder.mimeType })).then(
            resolve,
            reject,
          );
        };
        recorder.onerror = () => {
          release();
          reject(new Error("The recording failed."));
        };
        if (recorder.state === "inactive") {
          release();
          reject(new Error("The recording had already stopped."));
        } else {
          recorder.stop();
        }
      }),

    cancel: () => {
      recorder.onstop = release;
      if (recorder.state !== "inactive") recorder.stop();
      else release();
    },
  };
}

/** Whatever the browser recorded, as a 16 kHz mono WAV. */
async function toWav(blob: Blob): Promise<File> {
  // The context's rate is not a preference — `decodeAudioData` resamples to it,
  // so constructing the context at 16 kHz *is* the resampling step and there is
  // no separate pass to write.
  const context = new AudioContext({ sampleRate: SAMPLE_RATE });
  let audio: AudioBuffer;
  try {
    audio = await context.decodeAudioData(await blob.arrayBuffer());
  } finally {
    void context.close();
  }

  const name = `voice-note-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.wav`;
  return new File([encodeWav(mixToMono(audio))], name, { type: "audio/wav" });
}

/** Every channel averaged into one. A laptop mic is usually mono already, but a
 *  headset or an interface is not, and speech models take one channel. */
function mixToMono(audio: AudioBuffer): Float32Array {
  if (audio.numberOfChannels === 1) return audio.getChannelData(0);

  const mono = new Float32Array(audio.length);
  for (let channel = 0; channel < audio.numberOfChannels; channel++) {
    const samples = audio.getChannelData(channel);
    for (let i = 0; i < samples.length; i++) mono[i] += samples[i];
  }
  for (let i = 0; i < mono.length; i++) mono[i] /= audio.numberOfChannels;
  return mono;
}

/**
 * Float samples to a 16-bit PCM WAV.
 *
 * The 44-byte header is fixed boilerplate — the field offsets are the format,
 * not a choice — so it is written literally rather than through a builder that
 * would be longer than the thing it builds.
 */
function encodeWav(samples: Float32Array): ArrayBuffer {
  const bytes = samples.length * 2;
  const buffer = new ArrayBuffer(44 + bytes);
  const view = new DataView(buffer);
  const ascii = (at: number, text: string) => {
    for (let i = 0; i < text.length; i++) view.setUint8(at + i, text.charCodeAt(i));
  };

  ascii(0, "RIFF");
  view.setUint32(4, 36 + bytes, true); // everything after this field
  ascii(8, "WAVE");
  ascii(12, "fmt ");
  view.setUint32(16, 16, true); // length of this header
  view.setUint16(20, 1, true); // 1 = uncompressed PCM
  view.setUint16(22, 1, true); // channels
  view.setUint32(24, SAMPLE_RATE, true);
  view.setUint32(28, SAMPLE_RATE * 2, true); // bytes per second
  view.setUint16(32, 2, true); // bytes per frame
  view.setUint16(34, 16, true); // bits per sample
  ascii(36, "data");
  view.setUint32(40, bytes, true);

  for (let i = 0; i < samples.length; i++) {
    // Clamped before scaling: a sample slightly over 1.0 (which decoding can
    // produce) would otherwise wrap round to a loud click.
    const sample = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, sample * (sample < 0 ? 0x8000 : 0x7fff), true);
  }
  return buffer;
}
