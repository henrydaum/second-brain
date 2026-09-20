/**
 * One widget, in one box.
 *
 * A widget is an HTML document written by somebody the frame has no reason to
 * trust — the store, or the agent. So the app does not run it; it *contains*
 * it, and then talks to it. Everything in this file is one of those two jobs,
 * and both of them go through `lib/html-app.ts`, the one relay, of which this
 * is the only caller.
 *
 * **The containment is `sandbox="allow-scripts"`, and the thing that matters is
 * what is missing.** Adding `allow-same-origin` beside it cancels the sandbox
 * out entirely: the widget gets this page's origin back, and with it the
 * same-origin proxy that puts a bearer token on every `/sdk` call. It would
 * then be able to make any Request Second Brain has, silently, without ever
 * holding the credential. Withheld, the document loads into an *opaque origin*
 * — its own, shared with nothing — and can reach this page only by posting a
 * message.
 *
 * **The document is delivered, not `srcdoc`-ed, and that is a deployment fact
 * rather than a preference.** A `srcdoc` frame inherits the embedding page's
 * Content-Security-Policy, and the production gateway serves this app with a
 * hash-only `script-src`. So every inline script in a widget — the bridge and
 * the author's own — is blocked the moment it is deployed, while working
 * perfectly in development, where Vite sends no policy at all. `/html-app-host
 * .html` is a real URL with its own policy (`sandbox allow-scripts;
 * script-src 'unsafe-inline'; connect-src 'none'`), which is both stricter
 * about the network and permissive about the scripts it is meant to run. The
 * host is handed the document by `postMessage` and writes it in.
 */

import { useCallback, useEffect, useImperativeHandle, useRef, useState, type FC, type Ref } from "react";

import { appDocument, attachAppRelay, flushAppState, tellApp } from "@/lib/html-app";
import { widgetStyles } from "@/lib/widget-document";
import { readWidget, type Widget } from "@/lib/widgets";

export type WidgetFrameHandle = { flush: () => Promise<void> };

export const WidgetFrame: FC<{
  ref?: Ref<WidgetFrameHandle>;
  widget: Widget;
  scheme: "light" | "dark";
  /**
   * The widget's own saved state, as the kernel stored it.
   *
   * Delivered at mount and never again. It is the document's to interpret —
   * nothing out here parses it — and a widget that never calls
   * `brain.state.set` never sees anything but null, which is correct.
   */
  state?: string | null;
  watchSource?: boolean;
  /**
   * Bumped when the kernel says this widget's file changed.
   *
   * **A prompt to look, not the answer.** The kernel watches the disk and this
   * page cannot, so the announcement is the only way a mounted document learns
   * its source moved — but the watcher fires on *mtime*, and saving a file
   * unchanged is not an edit. So one fetch and a content comparison follow,
   * which is what keeps reverting a file to what is running from leaving a
   * Refresh button offering to reload the same bytes.
   *
   * This used to be a 3-second poll that re-read the whole document — a
   * multi-megabyte fetch on a timer to learn a boolean, which the watcher
   * already knew.
   */
  sourceCheck?: number;
  onSourceChange?: (changed: boolean) => void;
}> = ({ ref, widget, scheme, state = null, watchSource = false, sourceCheck = 0, onSourceChange }) => {
  const frame = useRef<HTMLIFrameElement>(null);
  const [source, setSource] = useState<string | null>(null);
  const [originalHtml, setOriginalHtml] = useState<string | null>(null);
  const [failure, setFailure] = useState<string | null>(null);
  /**
   * Whether the host page is up.
   *
   * **Both halves arrive on their own schedule, and neither can wait for the
   * other.** The iframe starts loading the moment React renders it; the
   * widget's source is a `fetch` that resolves whenever it resolves. Delivering
   * from the `load` handler alone meant that if the fetch was slower — which
   * depends on disk, on the kernel, on nothing the app controls — the handler
   * found no document, returned, and was never called again. The host sat on
   * "Loading HTML App…" forever, and it looked intermittent because it *was*:
   * the same code raced differently on every open.
   *
   * So neither event delivers. They each record that they happened, and the
   * effect below fires when both have.
   */
  const [loaded, setLoaded] = useState(false);
  /** Whether there is anything worth looking at yet. See the `className`. */
  const [ready, setReady] = useState(false);
  const delivered = useRef(false);

  /**
   * The token binds a document to this mount.
   *
   * It is not the boundary — the relay's `event.source` check is — but it is
   * what stops a widget that has been replaced being answered, or told
   * anything, on behalf of its successor.
   */
  const token = useRef(crypto.randomUUID());
  useImperativeHandle(ref, () => ({
    flush: () => frame.current && delivered.current
      ? flushAppState(frame.current, token.current)
      : Promise.resolve(),
  }), []);

  /**
   * The document, built once per widget.
   *
   * `scheme` is deliberately *not* a dependency. Rebuilding it would reload the
   * widget — losing its scroll position, its half-typed field and anything it
   * was holding — and a theme change is not worth that. The scheme is told
   * instead, and the bridge swaps the stylesheet in place.
   */
  useEffect(() => {
    let live = true;
    setSource(null);
    setOriginalHtml(null);
    setFailure(null);
    delivered.current = false;
    token.current = crypto.randomUUID();

    readWidget(widget).then(
      (html) => {
        if (!live) return;
        setOriginalHtml(html);
        setSource(appDocument(html, token.current, { styles: widgetStyles(scheme) }));
      },
      (error: unknown) => {
        if (live) setFailure(error instanceof Error ? error.message : String(error));
      },
    );
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- see above: the
    // scheme is read once at build time and told from then on.
  }, [widget.path]);

  /**
   * Whether the file this document was built from still says what it said.
   *
   * Asked when the kernel says so and at no other time. The token it watches
   * is seeded at mount, so the announcement that arrives *while* the source is
   * still loading is not lost — the check simply waits for something to
   * compare against — and the document this frame was just built from is never
   * compared with itself.
   */
  const checked = useRef(sourceCheck);
  useEffect(() => {
    if (!watchSource || originalHtml === null || !onSourceChange) return;
    if (sourceCheck === checked.current) return;
    checked.current = sourceCheck;
    let live = true;
    readWidget({ path: widget.path, name: widget.name }).then(
      (html) => { if (live) onSourceChange(html !== originalHtml); },
      // A missing file or a failed connection is not evidence of new source.
      () => {},
    );
    return () => { live = false; };
  }, [watchSource, sourceCheck, originalHtml, widget.path, widget.name, onSourceChange]);

  /** The relay, for the life of this frame. Closing it stops new calls and
   *  drops late results — it cannot undo work the kernel already accepted. */
  const attach = useCallback((element: HTMLIFrameElement | null) => {
    if (element) return attachAppRelay(element, token.current, widget.name);
  }, [widget.name]);

  const announce = useCallback(() => {
    const element = frame.current;
    if (!element) return;
    const { width, height } = element.getBoundingClientRect();
    tellApp(element, token.current, "scheme", scheme, { css: widgetStyles(scheme) });
    tellApp(element, token.current, "size", { width, height });
    // Its saved state, once, with the first size and scheme. A widget frame is
    // an opaque origin with no storage of its own, so this is the only moment
    // anything it kept last time can reach it. Parsed here rather than in the
    // document so a corrupted blob is one console line instead of a widget
    // that throws before it draws.
    let saved: unknown = null;
    if (state) {
      try {
        saved = JSON.parse(state);
      } catch (error) {
        console.error("Discarding unreadable widget state", error);
      }
    }
    tellApp(element, token.current, "state", saved);
  }, [scheme, state]);

  /**
   * Hand the host its document, once both it and the document exist.
   *
   * `document.write` produces a second `load`, and a widget is free to
   * navigate its own frame afterwards. Delivering again would replace whatever
   * it had become with the document it started as, so the guard is a ref rather
   * than a dependency — it must not be something a re-render can reconsider.
   */
  useEffect(() => {
    const element = frame.current;
    if (!element || !source || !loaded || delivered.current) return;
    delivered.current = true;
    element.contentWindow?.postMessage(
      { channel: "second-brain-html-mount-v1", html: source },
      "*",
    );
    // The first size and scheme, on the next tick. The host writes the document
    // synchronously but its scripts run after this returns, so a message posted
    // now lands in a realm with no bridge in it and is lost without a sound.
    // That is how a widget came to read `0 × 0` and keep it.
    const timer = window.setTimeout(() => {
      announce();
      setReady(true);
    }, 0);
    return () => window.clearTimeout(timer);
  }, [source, loaded, announce]);

  /** A theme change, told rather than rebuilt. */
  useEffect(() => {
    const element = frame.current;
    if (!element || !delivered.current) return;
    tellApp(element, token.current, "scheme", scheme, { css: widgetStyles(scheme) });
  }, [scheme]);

  /**
   * The box's size, told on change.
   *
   * A widget is sized *by the app* — the panel's width, the drawer sliding,
   * the window resizing — and is never consulted about it. Being told is what
   * lets one lay itself out at the size it has without measuring a window it
   * cannot see.
   */
  useEffect(() => {
    const element = frame.current;
    if (!element || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(([entry]) => {
      if (!delivered.current) return;
      const { width, height } = entry.contentRect;
      tellApp(element, token.current, "size", { width, height });
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, [source]);

  if (failure) {
    return <p className="text-muted-foreground p-4 text-xs">{failure}</p>;
  }

  return (
    <iframe
      ref={(element) => {
        frame.current = element;
        return attach(element);
      }}
      // Named for the person, not the file: this is what a screen reader
      // announces when it reaches the box.
      title={widget.name}
      // A real URL with its own policy — see the module note. The document
      // arrives by message.
      src="/html-app-host.html"
      onLoad={() => setLoaded(true)}
      // The whole boundary, and the absence is the point.
      sandbox="allow-scripts"
      referrerPolicy="no-referrer"
      allow="camera 'none'; microphone 'none'; geolocation 'none'; clipboard-read 'none'; clipboard-write 'none'"
      /*
        `block` rather than the default `inline`, which leaves a descender's
        worth of background under every iframe, and no border because the
        browser's default is a 2px inset ridge nobody asked for.

        **The colour and the fade are both about the gap before the document
        arrives.** The host page is transparent, so what shows through in the
        meantime is this element — `bg-sidebar`, the panel's own colour, which
        is also what the widget's `--sb-bg` resolves to. So the box never
        changes colour; it only gains contents. The fade is what stops those
        contents appearing all at once at whatever moment the file happened to
        load, which is the part that reads as a flash even once the white is
        gone.
      */
      data-ready={ready}
      className="motion-safe:transition-opacity block h-full w-full border-0 bg-sidebar opacity-0 duration-(--sb-motion-panel) data-[ready=true]:opacity-100"
    />
  );
};
