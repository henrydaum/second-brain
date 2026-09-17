/**
 * The microphone.
 *
 * **This is not dictation.** assistant-ui has a dictation adapter, and it is
 * the wrong shape for this server: it exists to stream a transcript into the
 * composer as you speak, which needs speech recognition that answers
 * continuously — the browser's own, or a service reachable per syllable.
 * Second Brain transcribes on the far side, once, when a file lands in the
 * attachment pipeline. So the honest wiring is the one Second Brain already
 * has: record, attach, send.
 *
 * Which means everything below the button is machinery that already exists.
 * The recording becomes an ordinary `File`, goes to `composer.addAttachment`,
 * and from there follows a dragged PDF's exact path — upload to host scratch,
 * `frontend.submit` with `ingest: true`, indexed like any other incoming file.
 * Nothing here knows it is audio except the encoder.
 *
 * It lands in the composer rather than sending straight away, deliberately: a
 * voice note can be reviewed, captioned, or removed as an attachment before it
 * goes. The live state therefore needs only one action: stop and attach.
 */

import { useCallback, useEffect, useRef, useState, type FC } from "react";
import { MicIcon, SquareIcon } from "lucide-react";
import { useAui } from "@assistant-ui/react";

import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
import { canRecord, record, type Recording } from "@/lib/audio";
import { useSession } from "@/runtime/provider";

/** m:ss. Long enough for a voice note, and a number climbing past 9:59 is its
 *  own warning. */
function elapsed(seconds: number): string {
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}

export const VoiceNoteButton: FC = () => {
  const aui = useAui();
  const { report } = useSession();

  // The live recorder, held in a ref rather than in state: it is not something
  // that gets drawn, and putting it in state would make every tick of the timer
  // a reason to reconsider it.
  const recorder = useRef<Recording | null>(null);
  const [recording, setRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);

  // Ticks only while recording, and clears itself on the way out — including on
  // unmount, which is what stops a closed tab from leaving the microphone on.
  useEffect(() => {
    if (!recording) return;
    const timer = setInterval(() => setSeconds((s) => s + 1), 1000);
    return () => clearInterval(timer);
  }, [recording]);

  useEffect(
    () => () => {
      recorder.current?.cancel();
    },
    [],
  );

  const begin = useCallback(async () => {
    try {
      recorder.current = await record();
      setSeconds(0);
      setRecording(true);
    } catch (error) {
      // Almost always a refused permission. Worth saying: a microphone button
      // that does nothing is indistinguishable from one that is broken.
      report(error);
    }
  }, [report]);

  const finish = useCallback(async () => {
    const active = recorder.current;
    recorder.current = null;
    setRecording(false);
    if (!active) return;
    try {
      const file = await active.stop();
      // From here it is an attachment like any other, failures included — the
      // chip reports its own upload.
      await aui.composer.addAttachment(file);
    } catch (error) {
      report(error);
    }
  }, [aui, report]);

  // Nothing to offer without a microphone API — which is the case on plain
  // http from anywhere but localhost, since `getUserMedia` needs a secure
  // context. Better absent than present and unexplainable.
  if (!canRecord()) return null;

  if (!recording) {
    return (
      <TooltipIconButton
        tooltip="Record a voice note"
        side="bottom"
        type="button"
        variant="ghost"
        size="icon"
        className="hover:bg-muted-foreground/15 dark:hover:bg-muted-foreground/30 size-7 rounded-full"
        aria-label="Record a voice note"
        onClick={() => void begin()}
      >
        <MicIcon className="size-4.5 stroke-[1.5px]" />
      </TooltipIconButton>
    );
  }

  return (
    <div className="flex items-center gap-1" data-slot="voice-note-recording">
      <TooltipIconButton
        tooltip="Stop and attach"
        side="bottom"
        type="button"
        variant="ghost"
        size="icon"
        className="text-destructive size-7 rounded-full"
        aria-label="Stop recording"
        onClick={() => void finish()}
      >
        <SquareIcon className="size-3.5 animate-pulse fill-current" />
      </TooltipIconButton>

      {/* Tabular figures so the seconds do not shuffle the layout as they
          climb. */}
      <span className="text-destructive w-8 text-xs tabular-nums">
        {elapsed(seconds)}
      </span>

    </div>
  );
};
