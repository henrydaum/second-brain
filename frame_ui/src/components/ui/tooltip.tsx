import type * as React from "react";
import { Tooltip as TooltipPrimitive } from "radix-ui";

import { cn } from "@/lib/utils";

/** Long enough that crossing the toolbar does not fire a row of tooltips,
 *  short enough that pausing on a control answers you. Zero — the value this
 *  shipped with — makes every pointer sweep flash labels. */
function TooltipProvider({
  delayDuration = 500,
  ...props
}: React.ComponentProps<typeof TooltipPrimitive.Provider>) {
  return (
    <TooltipPrimitive.Provider
      data-slot="tooltip-provider"
      delayDuration={delayDuration}
      {...props}
    />
  );
}

function Tooltip({
  ...props
}: React.ComponentProps<typeof TooltipPrimitive.Root>) {
  return (
    <TooltipProvider>
      <TooltipPrimitive.Root data-slot="tooltip" {...props} />
    </TooltipProvider>
  );
}

function TooltipTrigger({
  ...props
}: React.ComponentProps<typeof TooltipPrimitive.Trigger>) {
  return <TooltipPrimitive.Trigger data-slot="tooltip-trigger" {...props} />;
}

/**
 * How loud the tooltip is.
 *
 * `default` inverts against the page — the right weight for a control, where
 * the tooltip is telling you what pressing the thing will do. `subtle` is the
 * same shape and the same motion in a card's colours instead, for a label that
 * is only elaborating on text already on screen. Same family, quieter voice.
 */
type TooltipVariant = "default" | "subtle";

function TooltipContent({
  className,
  sideOffset = 0,
  variant = "default",
  children,
  ...props
}: React.ComponentProps<typeof TooltipPrimitive.Content> & {
  variant?: TooltipVariant;
}) {
  const subtle = variant === "subtle";

  return (
    <TooltipPrimitive.Portal>
      <TooltipPrimitive.Content
        data-slot="tooltip-content"
        data-variant={variant}
        sideOffset={sideOffset}
        className={cn(
          "fade-in-0 zoom-in-[0.98] data-[side=bottom]:slide-in-from-top-1 data-[side=left]:slide-in-from-right-1 data-[side=right]:slide-in-from-left-1 data-[side=top]:slide-in-from-bottom-1 data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-[0.98] animate-in data-[state=closed]:animate-out z-50 w-fit origin-(--radix-tooltip-content-transform-origin) px-3 py-1.5 text-xs text-balance",
          subtle
            ? "bg-popover text-muted-foreground border"
            : "bg-foreground text-background",
          className,
        )}
        {...props}
      >
        {children}
        <TooltipPrimitive.Arrow
          className={cn(
            "z-50 size-2.5 translate-y-[calc(-50%_-_2px)] rotate-45 rounded-[2px]",
            // Fill only, no border. The arrow is a rotated square, so which of
            // its edges face outward depends on which side the tooltip opened
            // on — a border that looks right above a trigger runs across the
            // base of one below it. The panel's own border stopping at the
            // arrow is the lesser artifact, and the usual one.
            subtle
              ? "bg-popover fill-popover"
              : "bg-foreground fill-foreground",
          )}
        />
      </TooltipPrimitive.Content>
    </TooltipPrimitive.Portal>
  );
}

export { Tooltip, TooltipTrigger, TooltipContent };
