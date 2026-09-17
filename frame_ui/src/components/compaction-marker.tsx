/**
 * Where the agent's memory of this conversation starts.
 *
 * Compaction folds everything said so far into a summary and writes a marker
 * row. Nothing is deleted — every message above the line is still stored, and
 * this app still shows all of it — but from the agent's side that transcript is
 * gone, replaced by a paragraph of summary. Scrollback that did not say so
 * would be a transcript quietly disagreeing with what the agent can answer
 * about, and the person would find out by being told "I don't recall that"
 * about a message plainly on their screen.
 *
 * **So the line is drawn where the loss happened, not where the reading
 * stops.** A wiggly rule rather than a banner: this is a seam in the
 * conversation, not an event in it, and anything with a headline and a body
 * would read as something the agent said. The sentence lives in the tooltip,
 * where it answers the question the line raises without repeating itself down
 * the length of a long conversation.
 *
 * assistant-ui has no compaction primitive. What it does have is the **system
 * role** — a first-class message that is in the transcript without being of the
 * conversation, whose default renderer draws nothing — and that is exactly this
 * shape, so the marker travels as one rather than through a channel of its own.
 * See `runtime/convert.ts`.
 */

import { useId, type FC } from "react";
import { MessagePrimitive, useAuiState } from "@assistant-ui/react";

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { COMPACTED } from "@/runtime/convert";

/**
 * What the line means, in one sentence.
 *
 * Worded for what is durably true. The session that did the compacting keeps
 * the last couple of messages beside the summary for a while, but that does not
 * survive a reload and is not a promise this can make on the conversation's
 * behalf — and of the two ways to be wrong here, "the agent knows less than you
 * think" is the one that costs nothing.
 */
const EXPLANATION =
  "The agent can only see a summary of everything above this line — not the messages themselves.";

/** The rule's shape, in the pattern tile's own units — which are pixels, since
 *  the tile is placed in user space and the svg is drawn at its natural size. */
const WAVE = {
  /** One full period: crest, then trough. */
  period: 20,
  /** The tile, and therefore the height of the whole rule. */
  height: 10,
  /** How far the crest rises above the midline. Keep `amplitude + thickness`
   *  inside `height / 2` or the curve is clipped by its own tile. */
  amplitude: 4,
  thickness: 1.25,
};

/**
 * One tileable period of the wave.
 *
 * **Drawn a period either side of the tile it will be clipped to**, so the
 * curve carries on across the join instead of ending there — a stroke that
 * stopped at the tile edge would leave a row of rounded caps down the rule.
 *
 * Derived rather than written out because four numbers have to agree: the
 * pattern box, the midline, the half-period each arc spans, and the svg's own
 * height. Written by hand, changing one of them silently misaligns the others —
 * a taller tile with the same path puts the wave off-centre and repeats a
 * sliver of a second one underneath it.
 */
function wavePath({ period, height, amplitude }: typeof WAVE): string {
  const half = period / 2;
  // The first arc is a full quadratic; the rest are `t`, which mirrors the
  // previous control point and so alternates crest and trough on its own.
  return (
    `M ${-period} ${height / 2} q ${half / 2} ${-amplitude} ${half} 0` +
    ` t ${half} 0`.repeat(4)
  );
}

/**
 * The rule itself.
 *
 * Split from the message wiring below it so it can be drawn — and tested —
 * without a thread around it.
 *
 * **It says nothing about when.** The marker's time is stored and the turn
 * carries it, but a compaction is not an appointment: what a reader needs from
 * this line is what the agent can no longer see, and the answer does not change
 * with the hour. A timestamp here only competed with the sentence that does
 * answer it.
 */
export const CompactionRule: FC = () => {
  // Patterns are referenced by id, and a conversation can hold more than one
  // marker. Sharing one id across instances happens to render the same, but it
  // is duplicate ids in the document, which is nobody's intent.
  const wave = useId();

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <div
          data-slot="compaction-marker"
          // `cursor-default` and `select-none` for the reason the message time
          // has them: an I-beam over a decorative rule reads as an invitation
          // to select something that is not text.
          //
          // **Not a flex container, and that is not a style choice.** An `svg`
          // with no `width` attribute and no `viewBox` has no intrinsic width
          // to offer, and asked for one as a flex item WebKit resolves it to
          // zero rather than to the `width: 100%` an svg defaults to — so this
          // marker drew nothing at all on iOS while being perfect everywhere
          // else. A block container asks the question the svg can answer.
          className="group w-full cursor-default py-1 select-none"
        >
          {/* The tooltip is a hover, and a hover is not available to everyone.
              This is the same sentence, said where a screen reader will reach
              it as it passes the line. */}
          <span className="sr-only">Conversation compacted. {EXPLANATION}</span>
          {/* Faded rather than a border colour: `--border` is white at 10% in
              the dark theme, which for a full-width hairline is close to
              invisible — and a marker nobody notices is the failure this whole
              component exists to prevent. One token at two strengths also
              means the hover is the same line getting clearer, not a different
              colour arriving. */}
          <svg
            aria-hidden
            // Height as an attribute rather than a class, because it is the
            // tile's height: one row of wave, exactly. A class here would be a
            // fourth number to keep in step with `WAVE` by hand.
            height={WAVE.height}
            // Width as an attribute *as well as* a class, which is belt and
            // braces rather than duplication: the class is what makes it fill
            // the thread, the attribute is what stops a browser having to guess
            // at the intrinsic size before it gets there.
            width="100%"
            // `block`, or the svg sits on a text baseline and carries a strip
            // of descender space under a rule that is ten pixels tall.
            className="text-muted-foreground/40 group-hover:text-muted-foreground block w-full transition-colors duration-(--sb-motion-enter)"
          >
            <defs>
              <pattern
                id={wave}
                width={WAVE.period}
                height={WAVE.height}
                patternUnits="userSpaceOnUse"
              >
                <path
                  d={wavePath(WAVE)}
                  fill="none"
                  stroke="currentColor"
                  strokeWidth={WAVE.thickness}
                  strokeLinecap="round"
                />
              </pattern>
            </defs>
            <rect width="100%" height="100%" fill={`url(#${wave})`} />
          </svg>
        </div>
      </TooltipTrigger>
      {/* `subtle`, like the message time: this elaborates on something already
          on screen rather than announcing what a control will do. */}
      <TooltipContent side="top" variant="subtle" className="max-w-72">
        <p className="text-foreground font-medium">Conversation compacted</p>
        <p>{EXPLANATION}</p>
      </TooltipContent>
    </Tooltip>
  );
};

export const CompactionMarker: FC = () => {
  // Keyed on the flag rather than on the role. A system message this app did
  // not put there is one it cannot explain, and drawing nothing for it is both
  // assistant-ui's own default and the only honest option.
  const compacted = useAuiState(
    (s) => s.message.metadata?.custom?.[COMPACTED] === true,
  );

  if (!compacted) return null;

  return (
    <MessagePrimitive.Root
      data-role="system"
      className="fade-in animate-in mx-auto w-full max-w-(--thread-max-width) px-2 duration-(--sb-motion-reveal)"
    >
      <CompactionRule />
    </MessagePrimitive.Root>
  );
};
