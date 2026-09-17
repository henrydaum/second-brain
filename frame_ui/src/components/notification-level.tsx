/**
 * How the four levels look, in the one place both surfaces read it from.
 *
 * The banner and the panel draw the same notification at different sizes, and a
 * level that meant "amber triangle" in one and "orange dot" in the other would
 * make the panel look like a different list of events than the one you just
 * watched go by.
 *
 * **Styling only.** Nothing branches on `level` kernel-side and an unrecognised
 * value arrives normalised to `info`, so this is a closed four-way choice rather
 * than a lookup that needs a fallback for correctness — `levelOf` supplies one
 * anyway, because the *stored* column is not covered by that guarantee.
 *
 * The old text carried `✓` and `✕` glyphs inside the message. Those are gone
 * from the wire precisely so a client can style rather than parse, which is what
 * this file is.
 */

import type { FC } from "react";
import {
  CheckCircle2Icon,
  CircleAlertIcon,
  InfoIcon,
  TriangleAlertIcon,
} from "lucide-react";

import type { Level } from "@/lib/notifications";
import { cn } from "@/lib/utils";

const ICONS = {
  info: InfoIcon,
  success: CheckCircle2Icon,
  warning: TriangleAlertIcon,
  error: CircleAlertIcon,
} as const;

/** Foreground only. Neither surface tints its whole background by level: a
 *  stack of three filled panels in three colours is a traffic light, and the
 *  thing worth reading is the title. */
const TONES: Record<Level, string> = {
  info: "text-muted-foreground",
  success: "text-emerald-600 dark:text-emerald-400",
  warning: "text-amber-600 dark:text-amber-400",
  error: "text-destructive",
};

export const LevelIcon: FC<{ level: Level; className?: string }> = ({
  level,
  className,
}) => {
  const Icon = ICONS[level];
  // Decorative: the level is also carried by the copy — a failed registration
  // says it failed — so a screen reader announcing "error icon" before the
  // title would be reading the same fact twice in the wrong order.
  return (
    <Icon aria-hidden className={cn("size-4 shrink-0", TONES[level], className)} />
  );
};
