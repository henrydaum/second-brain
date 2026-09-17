/**
 * An icon button that explains itself.
 *
 * **This was a stub, and the stub was load-bearing.** It accepted `variant`,
 * `size` and `side` and then dropped all three into `...props`, where they
 * landed on the DOM as meaningless attributes — so a caller asking for
 * `variant="default"` got a transparent button, and `variant="outline"` got no
 * border. The Send control was the worst of it: it sat beside a Cancel button
 * built from the real `Button` and the two looked like they came from different
 * applications. Routing through `Button` is what makes a stated variant mean
 * something.
 *
 * The tooltip was the other half. `title=` is the browser's tooltip: it appears
 * after a second or so, in the OS's own styling, and cannot be themed or
 * positioned. Every other floating surface here is Radix, so this is too.
 */

import type { ComponentProps, ReactNode } from "react";

import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

type TooltipIconButtonProps = ComponentProps<typeof Button> & {
  /** Shown on hover and focus, and used as the accessible name. Required —
   *  an icon with no name is a button nobody can identify. */
  tooltip: string;
  side?: "top" | "right" | "bottom" | "left";
  children?: ReactNode;
};

export function TooltipIconButton({
  tooltip,
  side = "bottom",
  variant = "ghost",
  size = "icon",
  className,
  children,
  onClick,
  ...props
}: TooltipIconButtonProps) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          type="button"
          variant={variant}
          size={size}
          // Before the spread, so a caller with a more specific name — the
          // recording button, which says "Stop recording" while its tooltip
          // says "Stop and attach" — can still override it.
          aria-label={tooltip}
          className={cn("shrink-0", className)}
          onClick={(event) => {
            onClick?.(event);
            // A pointer click leaves focus on the trigger. If that click also
            // opens or changes a surface, Radix otherwise presents its focus
            // tooltip as if the person hovered the newly labelled button.
            // Keyboard activation has detail 0 and keeps its useful focus.
            if (event.detail > 0) event.currentTarget.blur();
          }}
          {...props}
        >
          {children}
        </Button>
      </TooltipTrigger>
      {/* Radix will not open a tooltip for a disabled trigger, which is the
          behaviour you want: a control that cannot be used should not be
          advertising what it would have done. */}
      <TooltipContent side={side}>{tooltip}</TooltipContent>
    </Tooltip>
  );
}
