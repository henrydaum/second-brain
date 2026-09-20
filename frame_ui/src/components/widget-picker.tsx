/**
 * Which widget is in the panel.
 *
 * The catalog is **refreshed each time the menu opens**, not kept from load:
 * the agent may have authored a widget since the page started, and a picker
 * that cannot see it looks exactly like an authoring tool that did not work. A
 * refresh that fails keeps the previous list, because a stale list beats no
 * list.
 *
 * It is the panel's title as well as its control — the widget's name *is* the
 * heading, so the header carries one thing instead of a label beside a
 * duplicate of it.
 */

import { ChevronDownIcon } from "lucide-react";
import { useCallback, useEffect, useState, type FC } from "react";

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { listWidgets, type Widget } from "@/lib/widgets";

export const WidgetPicker: FC<{
  widgets: Widget[];
  chosen: string | null;
  onRefresh: (widgets: Widget[]) => void;
  onChoose: (name: string | null) => void;
  disabled?: boolean;
}> = ({ widgets, chosen, onRefresh, onChoose, disabled = false }) => {
  const [failure, setFailure] = useState<string | null>(null);

  const refresh = useCallback(() => {
    listWidgets().then(
      (found) => { onRefresh(found); setFailure(null); },
      (error: unknown) => setFailure(
        error instanceof Error ? error.message : String(error),
      ),
    );
  }, [onRefresh]);

  // Once at mount, so the panel can restore what was open last time without
  // waiting for somebody to touch the menu.
  useEffect(refresh, [refresh]);

  const current = widgets.find((widget) => widget.name === chosen);

  return (
    <DropdownMenu onOpenChange={(open) => open && refresh()}>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          disabled={disabled}
          className="hover:bg-accent/60 -ms-1 flex min-w-0 items-center gap-1 rounded-md px-2 py-1 text-sm font-medium outline-none focus-visible:ring-2 focus-visible:ring-(--ring)"
        >
          <span className="truncate">{current?.name ?? "Widget"}</span>
          <ChevronDownIcon className="size-3.5 shrink-0 opacity-60" />
        </button>
      </DropdownMenuTrigger>

      <DropdownMenuContent align="start" className="max-h-80 w-56 overflow-y-auto">
        {failure && (
          <DropdownMenuItem disabled className="text-xs whitespace-normal">
            {failure}
          </DropdownMenuItem>
        )}
        {!failure && widgets.length === 0 && (
          <DropdownMenuItem disabled className="text-xs whitespace-normal">
            No widgets installed. A widget is one HTML file in the kernel's
            `widgets/` folder.
          </DropdownMenuItem>
        )}

        <DropdownMenuRadioGroup
          value={chosen ?? ""}
          onValueChange={(value) => { if (!disabled) onChoose(value || null); }}
        >
          {widgets.map((widget) => (
            <DropdownMenuRadioItem
              key={widget.path}
              value={widget.name}
              className="flex-col items-start gap-0"
            >
              <span className="truncate">{widget.name}</span>
              {/* Which tree it came from, because two widgets can share a name
                  and the kernel resolves that by precedence — bundled beats
                  installed beats workspace. `shadowed` is how a draft that
                  lost says so. */}
              <span className="text-muted-foreground text-xs">
                {widget.tree}
                {widget.shadowed?.length
                  ? ` · shadows ${widget.shadowed.length}`
                  : ""}
              </span>
            </DropdownMenuRadioItem>
          ))}
        </DropdownMenuRadioGroup>

        {chosen && (
          <>
            <DropdownMenuSeparator />
            <DropdownMenuItem disabled={disabled} onSelect={() => onChoose(null)}>
              Empty
            </DropdownMenuItem>
          </>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  );
};
