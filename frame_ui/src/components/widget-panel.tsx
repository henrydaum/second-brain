/**
 * The widget panel: where a widget lives, in all three of the places it can be.
 *
 * On a desktop it is where the files drawer used to be — in the flow, so
 * opening it reflows the thread instead of covering the part you were reading.
 * On a phone it is a fixed split above the conversation. In both, the expand
 * control gives it the whole window. `runtime/widget-mode.ts` says why those
 * are one mode rather than three features.
 *
 * **It is one element in one place in the tree, and the mode is only CSS.**
 * That is the load-bearing decision in this file and it is not a
 * simplification: React reparents by remounting, so an `<aside>` that moved
 * between a flex row, a flex column and a fixed layer would tear down and
 * rebuild whatever is inside it on every mode change. A widget is an iframe —
 * a document with a scroll position, a half-typed field and possibly an open
 * stream — and remounting it reloads all of that, silently, presenting as
 * "going fullscreen resets my widget". So the three modes are three class
 * lists on one node, and `inert` rather than unmounting handles being closed.
 *
 * **The width is the person's and is remembered**, because a panel that resets
 * to a default on every reload is one you re-drag every morning. It is in
 * `localStorage` for the reason the sidebar's collapsed state is: it is a fact
 * about this window rather than about the account, and a `config.write` from
 * this page would raise an approval dialog for the crime of dragging a border.
 *
 * **Which widget is showing is not that kind of fact, and it moved out.** It
 * was in `localStorage` beside the width, which made the panel a property of
 * the window you happened to open: the same conversation showed different
 * things on a laptop and a phone, and the agent had no way to know what the
 * person was looking at. The binding is on the conversation row now and the
 * provider owns it — see `useWidget` — because the panel is no longer the only
 * thing that decides what is in it. The width stayed, because a dragged border
 * really is about this window and nothing else.
 */

import {
  useCallback, useEffect, useLayoutEffect, useRef, useState,
  type FC, type KeyboardEvent, type PointerEvent,
} from "react";
import { Minimize2Icon, Maximize2Icon, RefreshCwIcon, XIcon } from "lucide-react";

import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
import { WidgetFrame, type WidgetFrameHandle } from "@/components/widget-frame";
import { WidgetPicker } from "@/components/widget-picker";
import { useResolvedTheme } from "@/lib/theme";
import { cn } from "@/lib/utils";
import { useWidget } from "@/runtime/domains";
import { getWidget, listWidgets, type Binding, type Widget } from "@/lib/widgets";
import type { WidgetMode } from "@/runtime/widget-mode";

const WIDTH_KEY = "second-brain:widget-width";

/**
 * The tolerances, and what each one is protecting.
 *
 * `MIN_WIDTH` is the panel's: narrower than this and a widget is a column of
 * ellipsed text, which is worse than the panel being shut. `MIN_THREAD` is the
 * conversation's — and it is measured against the *composer*, not the message
 * column, because the composer is what breaks first: its attachment row, mode
 * picker and model selector are a fixed amount of furniture that starts
 * overlapping long before a paragraph stops being readable. `MAX_WIDTH` is
 * neither: a panel wider than this on a large display is a second window, and
 * at that point what you want is fullscreen rather than a drawer pretending to
 * be one.
 */
/**
 * How long the panel takes to stop being a layer.
 *
 * It matches `--sb-motion-panel` with a few milliseconds of slack, and it is a
 * number here rather than a read of the custom property because it also paces
 * the two-step close below — a duration that has to be *waited* cannot come
 * from a stylesheet without measuring it.
 */
const EXIT_MS = 260;

const MIN_WIDTH = 280;
const MAX_WIDTH = 720;
const MIN_THREAD = 600;
const DEFAULT_WIDTH = 384;

/**
 * Clamp a width against the window as it is *now*.
 *
 * Both halves matter. Clamping on drag stops you dragging the thread away;
 * clamping on resize stops a width that was fine on a wide display from
 * surviving into a narrow one — the panel would be legal by its own limits and
 * still leave no room to type in.
 */
export function clampWidth(width: number, viewport: number): number {
  const ceiling = Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, viewport - MIN_THREAD));
  return Math.round(Math.min(ceiling, Math.max(MIN_WIDTH, width)));
}

function storedWidth(): number {
  try {
    const raw = Number(localStorage.getItem(WIDTH_KEY));
    return Number.isFinite(raw) && raw > 0 ? raw : DEFAULT_WIDTH;
  } catch {
    // Private windows and blocked site data both throw here. A default width is
    // a fine thing to fall back to; a blank page is not.
    return DEFAULT_WIDTH;
  }
}

export const WidgetPanel: FC<{
  open: boolean;
  mode: WidgetMode;
  onClose: () => void;
  onToggleFull: () => void;
}> = ({ open, mode, onClose, onToggleFull }) => {
  const [width, setWidth] = useState(storedWidth);
  const [dragging, setDragging] = useState(false);
  const [widgets, setWidgets] = useState<Widget[]>([]);
  const scheme = useResolvedTheme();
  /**
   * What this conversation holds, and how to change it.
   *
   * Owned by the provider rather than by this component, because three things
   * now decide it: the menu below, the agent calling `sdk.widget.set`, and a
   * conversation switch. The last two arrive as render frames, which reach the
   * provider and not this file.
   */
  const { widgetBinding, chooseWidget: choose, widgetCatalog } = useWidget();
  const chosen = widgetBinding?.name ?? null;

  /**
   * The widget the panel is holding, or nothing.
   *
   * Resolved against the listing rather than taken from the binding's own
   * `path`, because the picker needs the whole row anyway and the two must not
   * disagree about which file a name means. A name that matches no installed
   * widget is ordinary rather than an error — the store uninstalled it, or the
   * agent renamed a file — and it is said below rather than silently treated
   * as empty, which looks identical to a panel nobody has filled in.
   */
  const widget = widgets.find((entry) => entry.name === chosen) ?? null;
  const [refreshing, setRefreshing] = useState(false);
  const transitionPending = useRef(false);
  const widgetFrame = useRef<WidgetFrameHandle>(null);
  const [refreshError, setRefreshError] = useState<string | null>(null);
  const [refresh, setRefresh] = useState<{ binding: Binding; state: string | null; version: number } | null>(null);
  const bindingRef = useRef(widgetBinding);
  bindingRef.current = widgetBinding;
  const frameKey = `${widgetBinding?.conversationId ?? "new"}:${widget?.path ?? ""}:${refresh?.version ?? 0}`;
  const [changedSource, setChangedSource] = useState<string | null>(null);
  /**
   * Re-read the listing whenever the kernel says a widget file moved.
   *
   * The picker refreshes at mount and when it is opened, which answers "what
   * is installed" for somebody who goes looking. It cannot answer it for
   * somebody who is *not* looking — and that is the case that matters, because
   * the agent writing a widget and then binding it is the ordinary way one
   * appears. Without this the panel says "No widget named …" about a file that
   * is right there on disk, until the person happens to open the menu.
   */
  useEffect(() => {
    if (widgetCatalog.version === 0) return;
    let live = true;
    listWidgets().then(
      (found) => { if (live) setWidgets(found); },
      // The picker reports a failed listing; this one is a background refresh
      // of something already on screen, and it is retried at the next change.
      () => {},
    );
    return () => { live = false; };
  }, [widgetCatalog.version]);
  /**
   * The prompt the mounted frame compares its source against.
   *
   * Only for changes naming the widget on screen, so editing an unrelated file
   * costs the mounted document nothing at all. Zero while nothing has named
   * it, which is also what the frame is seeded with.
   */
  const sourceCheck = widgetCatalog.name && widgetCatalog.name === chosen
    ? widgetCatalog.version : 0;
  const onSourceChange = useCallback((changed: boolean) => {
    setChangedSource(changed ? frameKey : null);
  }, [frameKey]);

  const afterSaving = async (action: () => Promise<void>) => {
    if (transitionPending.current) return;
    const binding = widgetBinding;
    transitionPending.current = true;
    setRefreshing(true);
    setRefreshError(null);
    try {
      await widgetFrame.current?.flush();
      if (bindingRef.current === binding) await action();
    } catch (error) {
      if (bindingRef.current === binding) {
        setRefreshError(error instanceof Error ? error.message : "Could not save widget state.");
      }
    } finally {
      transitionPending.current = false;
      setRefreshing(false);
    }
  };
  const refreshWidget = async () => {
    if (!widget || !widgetBinding) return;
    const binding = widgetBinding;
    await afterSaving(async () => {
      // Binding announcements do not track saves. Read the newly saved state
      // after flushing, before replacing the frame.
      const saved = await getWidget();
      if (bindingRef.current !== binding || saved.name !== binding.name ||
          saved.conversationId !== binding.conversationId) return;
      setRefresh(current => ({ binding, state: saved.state, version: (current?.version ?? 0) + 1 }));
    });
  };
  const chooseAfterSaving = (name: string | null) => {
    if (name !== chosen) void afterSaving(() => choose(name));
  };
  const full = mode === "full";
  const side = mode === "side";

  /**
   * Still drawn as a layer, on the way out of being one.
   *
   * **Leaving fullscreen is an animation, and for its whole duration the panel
   * must stay out of the flow.** Dropped back into it the moment the mode
   * changes, its width animates down *as a flex item* — so the thread grows
   * through the entire 220ms and every line of it re-wraps on every frame,
   * which is the same reflow the spacer exists to prevent, arriving on the way
   * back. Expanding looked right and contracting did not, which is the tell
   * that only one direction had been thought about.
   *
   * It also buys the close. `hidden` is the correct resting state for a closed
   * fullscreen panel and a terrible transition — the window simply loses a
   * layer between one frame and the next. Staying a layer while the width runs
   * to zero slides it off the right edge instead, which is what the panel does
   * from every other mode.
   */
  const [leaving, setLeaving] = useState(false);
  const wasFull = useRef(false);

  useEffect(() => {
    const now = full && open;
    // Mobile's `full` → `pinned` keeps its immediate behaviour: there the chat
    // is *below* the panel rather than beside it, so it grows downward from a
    // fixed top edge and no text moves sideways.
    if (wasFull.current && !now && (!open || side)) setLeaving(true);
    wasFull.current = now;
  }, [full, open, side]);

  useEffect(() => {
    if (!leaving) return;
    // A timer rather than `transitionend`, which does not fire at all when the
    // duration is zero — exactly the case `prefers-reduced-motion` produces,
    // and the one where being stuck as a fixed layer forever would be worst.
    const timer = window.setTimeout(() => setLeaving(false), EXIT_MS);
    return () => window.clearTimeout(timer);
  }, [leaving]);

  /** Drawn over the app rather than in the row: fullscreen, or on its way out
   *  of fullscreen. */
  const layer = (full && open) || leaving;

  // Before the first paint, not after: a width read from storage that is too
  // wide for this window would otherwise be visible for one frame.
  useLayoutEffect(() => {
    setWidth((current) => clampWidth(current, window.innerWidth));
  }, []);

  useEffect(() => {
    const onResize = () => setWidth((current) => clampWidth(current, window.innerWidth));
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  /**
   * Escape leaves fullscreen rather than closing.
   *
   * Fullscreen is the state that hides every other way out — the session bar,
   * the sidebar and the button that opened this are all underneath it — so the
   * key everyone presses at a takeover has to mean something here. It steps
   * back to the shared mode, because that is what Escape does everywhere else:
   * undo the last enlargement, not discard the thing.
   */
  useEffect(() => {
    if (!full || !open) return;
    const onKey = (event: globalThis.KeyboardEvent) => {
      if (event.key === "Escape") onToggleFull();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [full, open, onToggleFull]);

  /**
   * Closing from fullscreen is the two gestures it looks like.
   *
   * **It exits fullscreen first, then hides**, which is exactly what the person
   * would do by pressing the two buttons in turn — and doing it that way means
   * there is no third animation to tune. A single width running from the whole
   * window to nothing covers several times the distance of an ordinary close in
   * the same 220ms, so it reads as a snap even though the duration is right;
   * two ordinary moves in sequence are each the speed everything else moves at.
   */
  const exitTimer = useRef(0);
  useEffect(() => () => window.clearTimeout(exitTimer.current), []);

  const closeFromHere = useCallback(() => {
    if (!full) return onClose();
    onToggleFull();
    exitTimer.current = window.setTimeout(onClose, EXIT_MS);
  }, [full, onClose, onToggleFull]);

  const remember = useCallback((next: number) => {
    try {
      localStorage.setItem(WIDTH_KEY, String(next));
    } catch {
      /* Refused. The panel still works; it just will not come back this wide. */
    }
  }, []);

  /**
   * Dragging the border.
   *
   * Pointer capture rather than window listeners, so the drag survives the
   * pointer leaving the handle — which it does immediately, since the handle is
   * six pixels wide and the gesture is fifty.
   *
   * The panel is anchored to the right edge, so its width is the distance from
   * the pointer to that edge. Reading it from `clientX` each move rather than
   * accumulating deltas is what keeps the handle under the finger: an
   * accumulated width drifts by exactly the amount the clamp threw away, and
   * then the border stops tracking the pointer at the limits.
   */
  const onPointerDown = (event: PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) return;
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    setDragging(true);
  };

  const onPointerMove = (event: PointerEvent<HTMLDivElement>) => {
    if (!dragging) return;
    setWidth(clampWidth(window.innerWidth - event.clientX, window.innerWidth));
  };

  const endDrag = (event: PointerEvent<HTMLDivElement>) => {
    if (!dragging) return;
    event.currentTarget.releasePointerCapture?.(event.pointerId);
    setDragging(false);
    remember(width);
  };

  /** The keyboard's version of the same gesture, since a drag handle that only
   *  answers to a pointer is a size only some people can set. */
  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const step = event.shiftKey ? 64 : 16;
    // Left widens: the panel grows towards the middle of the window.
    const delta = event.key === "ArrowLeft" ? step
      : event.key === "ArrowRight" ? -step
      : 0;
    if (!delta) return;
    event.preventDefault();
    const next = clampWidth(width + delta, window.innerWidth);
    setWidth(next);
    remember(next);
  };

  /**
   * The panel's own measurements, by mode.
   *
   * Written as inline styles rather than classes because two of the three are
   * numbers the person set or the window decided — a Tailwind class cannot
   * carry a dragged width, and `w-0` on a closed panel has to beat whatever the
   * open one was at.
   *
   * **Fullscreen is a width too, and that is what makes it animate correctly.**
   * Left to a class the width would go unset, which has two consequences and
   * both were bugs: the panel grew from the left edge rightwards — backwards,
   * since it lives on the right and should open towards the middle — and
   * closing it changed no measurement at all, leaving a full-screen inert layer
   * exactly where it was. It looked like nothing had happened, swallowed its
   * own contract button and handed the click to the session bar underneath.
   */
  const size = layer
    // `100%` rather than `100vw`: for a fixed element a percentage resolves
    // against the viewport *without* its scrollbar, so the panel does not hang
    // a few pixels off the left edge on a desktop. On the way out it is the
    // width the panel is going *to* — the side width, or nothing at all — and
    // the element stays fixed until it gets there.
    ? { width: full && open ? "100%" : open ? width : 0 }
    : side
      ? { width: open ? width : 0 }
      // Pinned: a fixed split, because there is no room on a phone for a
      // gesture that divides two small things into two smaller ones.
      : { height: open ? "calc(45dvh + env(safe-area-inset-top, 0px))" : 0 };

  return (
    <>
      {/*
        The handle is a sibling of the panel rather than a border on it, because
        a 0.5px divider is not something anybody can hit. It is six pixels of
        hit area drawn as nothing, sitting on the seam. It exists only while the
        panel is open *and* beside the chat — there is nothing to resize in the
        other two modes, and a control for a size that is not adjustable is a
        control that lies.
      */}
      {open && side && (
        <div
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize the widget panel"
          aria-valuenow={width}
          aria-valuemin={MIN_WIDTH}
          aria-valuemax={MAX_WIDTH}
          tabIndex={0}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={endDrag}
          onPointerCancel={endDrag}
          onKeyDown={onKeyDown}
          className={cn(
            "focus-visible:ring-ring relative z-10 hidden w-1.5 shrink-0 cursor-col-resize",
            "touch-none select-none outline-none focus-visible:ring-2 md:block",
            // The line under the pointer, not a second divider: the panel's own
            // `sb-divider-start` is the seam, and this only tints it.
            "after:bg-foreground/15 after:absolute after:inset-y-0 after:left-0 after:w-full",
            "after:opacity-0 after:transition-opacity hover:after:opacity-100",
            dragging && "after:opacity-100",
          )}
        />
      )}

      {/*
        A stand-in holding the width the panel gave up when it left the flow.
        **The chat must not reflow when the widget goes fullscreen** — the
        panel is supposed to grow *over* it, and a thread that jumps wider
        behind a layer that is about to cover it is a rearrangement nobody
        asked to watch. Keeping the gap open also means coming back out of
        fullscreen costs no reflow either.

        Desktop only: on a phone the layer covers the whole window, so whatever
        happens behind it is not visible to anybody.
      */}
      {layer && (
        <div
          data-slot="widget-spacer"
          aria-hidden
          // Zero while the panel is closing, so the gap it was holding shuts at
          // the same rate the panel leaves and the thread widens once, smoothly,
          // instead of snapping when the layer disappears.
          style={{ width: open ? width : 0 }}
          className="hidden shrink-0 transition-[width] md:block"
        />
      )}

      {/* The mobile panel covers the fixed session bar while this spacer
          reserves the split above the chat. The app already reserves the
          safe area, so only the panel itself adds that inset to its height. */}
      {!side && !layer && (
        <div
          aria-hidden
          style={{ height: open ? "45dvh" : 0 }}
          className="order-first w-full shrink-0 transition-[height] md:hidden"
        />
      )}

      <aside
        data-slot="widget-panel"
        data-mode={mode}
        aria-label="Widget"
        // `inert` rather than unmounting, the same call the files drawer makes:
        // a widget is a live document in here, and unmounting it on every close
        // would reload it. A closed panel must still not hold focus or be
        // reachable by tab.
        inert={!open}
        style={size}
        className={cn(
          "sb-panel bg-sidebar flex flex-col overflow-hidden",
          // Animating the size is right for a press of a button and wrong
          // during a drag, where it lands as the border lagging the pointer.
          !dragging && "transition-[width,height]",
          layer
            // Over everything, including the sidebar and the session bar. A
            // layer rather than a rearrangement: nothing else in the window has
            // to know this happened, and nothing else remounts when it does.
            //
            // **Anchored to the right edge, not to both.** `inset-0` pins the
            // left edge as well, so a width animating outwards moves the
            // *right* edge and the panel opens left-to-right — away from the
            // side it actually lives on. Held against the right, the growing
            // width moves the left edge instead, and the panel opens across the
            // chat the way it is already sitting beside it.
            ? "fixed top-0 right-0 bottom-0 z-50 h-dvh pt-[env(safe-area-inset-top)]"
            : side
              // Beside the chat, from `md`. Hidden below it, where `pinned` is
              // the shared mode instead.
              ? "sb-divider-start hidden h-full shrink-0 md:flex"
              // Above the chat, which is the one arrangement that leaves the
              // composer where it has always been. The composer lives inside
              // the thread's scroll viewport and sticks to its bottom, so a
              // widget *under* the chat is a widget under the composer —
              // typing in the middle of the display, with the keyboard opening
              // over the thing you are watching.
              : "sb-divider-bottom fixed inset-x-0 top-0 z-30 w-full md:hidden",
          !open && "border-0",
          // A fullscreen panel has no measurement that shrinks it out of the
          // way — `100%` is the whole window whether or not anybody wants to
          // see it — so being closed has to be said outright. The other two
          // modes animate to nothing and need no such line.
          // Only once it has finished leaving: a fullscreen panel has no
          // measurement that shrinks it out of the way, so being closed has to
          // be said outright — but saying it immediately is what made closing a
          // disappearance rather than an exit.
          !open && full && !leaving && "hidden",
        )}
      >
        {/*
          The header is in every mode, and in fullscreen it is the only way
          back. Its width is pinned while the panel is beside the chat so the
          contents do not reflow as it animates shut — text re-wrapping on every
          frame reads as the panel being squeezed rather than put away.
        */}
        <header
          className={cn(
            "sb-divider-bottom flex h-12 shrink-0 items-center gap-2 px-2",
            !side && !layer && "h-[calc(3rem+env(safe-area-inset-top,0px))] pt-[env(safe-area-inset-top)]",
          )}
          // Pinned to the panel's width only while the panel *is* that width.
          // While it is a layer the panel is anchored to the right edge and its
          // left edge is what moves, so a header pinned to the narrow width
          // would be laid out from that left edge — sending the controls to the
          // far side of the screen at the click and walking them back as the
          // panel shrank. Left to fill the layer, they stay where they were.
          style={side && !layer ? { width } : undefined}
        >
          <div className="min-w-0 flex-1">
            <WidgetPicker
              widgets={widgets}
              chosen={chosen}
              onRefresh={setWidgets}
              onChoose={chooseAfterSaving}
              disabled={refreshing}
            />
          </div>
          {changedSource === frameKey && <TooltipIconButton
            tooltip="Refresh"
            side="bottom"
            className="text-muted-foreground size-8"
            disabled={!widget || refreshing}
            onClick={() => void refreshWidget()}
          >
            <RefreshCwIcon className={cn("size-3.5", refreshing && "motion-safe:animate-spin")} />
          </TooltipIconButton>}
          <TooltipIconButton
            tooltip={full ? "Exit fullscreen" : "Fullscreen"}
            side="bottom"
            className="size-8"
            aria-pressed={full}
            onClick={onToggleFull}
          >
            {full ? <Minimize2Icon className="size-4" /> : <Maximize2Icon className="size-4" />}
          </TooltipIconButton>
          <TooltipIconButton tooltip="Hide widget" side="bottom" className="size-8" onClick={closeFromHere}>
            <XIcon className="size-4" />
          </TooltipIconButton>
        </header>
        {refreshError && <p role="alert" className="text-muted-foreground px-4 py-2 text-xs">{refreshError}</p>}

        {/*
          The widget gets this box entirely: no padding, no border, no scroll
          container of the frame's. `overflow-hidden` rather than `auto`
          because the document inside does its own scrolling — an outer
          scrollbar here would be a second one for the same content.
        */}
        <div
          className="min-h-0 w-full flex-1 overflow-hidden"
          inert={refreshing}
          style={side && !layer ? { width } : undefined}
        >
          {widget ? (
            /*
              Keyed by conversation *and* path, so a new document is built
              whenever either changes rather than swapping the `srcdoc` under a
              running one.

              **The conversation is half the key because state is delivered
              once, at mount.** The relay resolves its `state` promise on the
              first announcement and never again, by contract — so the only way
              to hand a document a different conversation's saved game is to
              build a new document. Keyed on the path alone, switching between
              two conversations that both show 2048 kept the first board and
              claimed it was the second.

              It comes off the *binding* rather than off `conversationId` in
              the provider, and that is not interchangeable: the conversation
              frame and the widget frame arrive separately, so a key from the
              provider can change a render before the new state lands — which
              rebuilds the document around the *old* state and then cannot
              correct it, because by then `delivered` is set. Taking both from
              one object is what makes them change together.

              The cost is one rebuild when a pending widget is adopted by the
              conversation its first message creates. That is survivable by
              design — the state it remounts with is the state it just saved —
              and bounded by the relay's 500ms save debounce.
            */
            <WidgetFrame
              ref={widgetFrame}
              key={frameKey}
              widget={widget}
              scheme={scheme}
              state={refresh?.binding === widgetBinding ? refresh.state : widgetBinding?.state ?? null}
              watchSource={open}
              sourceCheck={sourceCheck}
              onSourceChange={onSourceChange}
            />
          ) : (
            <p className="text-muted-foreground p-4 text-xs">
              {chosen
                ? `No widget named "${chosen}" is installed.`
                : "Nothing here yet. Choose a widget from the menu above."}
            </p>
          )}
        </div>
      </aside>
    </>
  );
};
