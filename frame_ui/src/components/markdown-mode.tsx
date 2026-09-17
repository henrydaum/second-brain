/**
 * Rendered or raw, and the one control that says which.
 *
 * **The state is shared rather than per-view, and that is what lets the button
 * live somewhere else.** The viewer dialog puts this in its footer, beside the
 * download link, because a control with a band of its own costs a strip of
 * vertical space on every note whether anybody touches it or not — and the
 * footer is already there. But the thing it controls is three components away,
 * rendered inside `FileView`, so the two cannot pass state between them
 * without one of them owning the other.
 *
 * A module-level value with `useSyncExternalStore` over it is the smaller
 * answer than a context: there is exactly one of these per tab, nothing renders
 * a second copy under different rules, and a provider that wraps the app to
 * hold one enum is more machinery than the enum.
 *
 * **It is a mood, not a setting.** Somebody who wants source wants it for the
 * next note too, so it carries across files — and it does not touch storage,
 * because it has no business outliving the tab. That is also why it is not in
 * Settings: there is nothing here to configure.
 */

import { useSyncExternalStore, type FC } from "react";
import { CodeXmlIcon, EyeIcon } from "lucide-react";

import { cn } from "@/lib/utils";

export type MarkdownMode = "preview" | "source";

let current: MarkdownMode = "preview";
const listeners = new Set<() => void>();

function subscribe(onChange: () => void) {
  listeners.add(onChange);
  return () => listeners.delete(onChange);
}

export function setMarkdownMode(next: MarkdownMode): void {
  if (next === current) return;
  current = next;
  for (const listener of listeners) listener();
}

export function useMarkdownMode(): MarkdownMode {
  return useSyncExternalStore(
    subscribe,
    () => current,
    // A server snapshot for completeness rather than for use: this app has no
    // server render, and "preview" is the state a first paint should show.
    () => "preview" as MarkdownMode,
  );
}

/**
 * The two states, as one object.
 *
 * `aria-pressed` rather than a radio group: these are two buttons that do a
 * thing, not a field carrying a value. Sized for the footer strip it sits in —
 * see `FileViewerDialog` — which is why it is smaller than a control would
 * ordinarily be.
 */
export const MarkdownModePicker: FC = () => {
  const mode = useMarkdownMode();

  return (
    <div
      role="group"
      aria-label="How to show this file"
      className="sb-file-mode bg-muted/40 inline-flex shrink-0 items-center gap-0.5 rounded-md border p-0.5"
    >
      {(
        [
          ["preview", "Preview", EyeIcon],
          ["source", "Source", CodeXmlIcon],
        ] as const
      ).map(([value, label, Icon]) => (
        <button
          key={value}
          type="button"
          aria-pressed={mode === value}
          onClick={() => setMarkdownMode(value)}
          className={cn(
            "focus-visible:ring-ring inline-flex items-center gap-1 rounded-sm px-1.5 py-0.5 text-[11px] outline-none focus-visible:ring-2",
            mode === value
              ? "bg-background text-foreground shadow-xs"
              : "hover:text-foreground",
          )}
        >
          <Icon className="size-3" aria-hidden />
          {label}
        </button>
      ))}
    </div>
  );
};
