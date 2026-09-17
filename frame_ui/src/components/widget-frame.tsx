/**
 * One widget, in one box.
 *
 * A widget is an HTML document written by somebody the frame has no reason to
 * trust — the store, or the agent. So the frame does not run it; it *contains*
 * it, and then talks to it. Everything in this file is one of those two jobs.
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
 * Two consequences follow from the opaque origin and neither is optional:
 *
 * - `event.origin` arrives as the string `"null"`, so it identifies nobody. A
 *   widget is recognised by `event.source === frame.contentWindow` instead,
 *   which is an object identity and cannot be spoofed by a message.
 * - A reply must be posted with `"*"` as its target, because an opaque origin
 *   has no name to address. That is safe only *because* the source check above
 *   has already decided who is being answered.
 *
 * **The bridge grants no authority of its own.** It relays Requests through the
 * same `sdk()` the app uses, so the kernel classifies them exactly as it would
 * anything else and an unsafe one still raises a dialog — which the frame is
 * holding the stream to be asked. What it withholds is the `frontend.*` family:
 * identity, attendance and approval answers belong to the frame.
 */

import { useEffect, useRef, useState, type FC } from "react";

import { RequestFailed, sdk } from "@/lib/client";
import { tokenBlock, widgetDocument } from "@/lib/widget-document";
import { readWidget, type Widget } from "@/lib/widgets";

const CHANNEL = "sb-widget-v1";

/** How many Requests one widget may have in flight. A runaway loop is a bug
 *  worth reporting to its author rather than a reason to flood the kernel. */
const MAX_PENDING = 64;

/** Request families a widget may never reach. */
const RESERVED = ["frontend."];

type Message = {
  channel?: string;
  token?: string;
  kind?: string;
  id?: number;
  type?: string;
  args?: Record<string, unknown>;
};

export const WidgetFrame: FC<{ widget: Widget; scheme: "light" | "dark" }> = ({
  widget,
  scheme,
}) => {
  const frame = useRef<HTMLIFrameElement>(null);
  const [source, setSource] = useState<string | null>(null);
  const [failure, setFailure] = useState<string | null>(null);

  /**
   * The document, built once per widget.
   *
   * `scheme` is deliberately *not* a dependency. Rebuilding the `srcdoc` would
   * reload the widget — losing its scroll position, its half-typed field and
   * anything it was holding — and a theme change is not worth that. The scheme
   * crosses as a message instead, and the bridge swaps the token block in
   * place.
   *
   * The token is minted per mount and checked on every message. It is not a
   * security boundary — the source check is — but it is what keeps a widget
   * that was replaced from being answered by its successor's frame.
   */
  const token = useRef(crypto.randomUUID());

  useEffect(() => {
    let live = true;
    setSource(null);
    setFailure(null);
    token.current = crypto.randomUUID();

    readWidget(widget).then(
      (html) => {
        if (!live) return;
        setSource(widgetDocument(html, {
          channel: CHANNEL,
          token: token.current,
          scheme,
        }));
      },
      (error: unknown) => {
        if (live) setFailure(error instanceof Error ? error.message : String(error));
      },
    );
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- see the note on
    // `scheme` above: it is read at build time and pushed as a message after.
  }, [widget.path]);

  /** Relaying Requests. One listener for the life of the mount. */
  useEffect(() => {
    let inFlight = 0;

    const reply = (payload: Record<string, unknown>) => {
      frame.current?.contentWindow?.postMessage(
        { channel: CHANNEL, ...payload },
        "*",
      );
    };

    const onMessage = (event: MessageEvent<Message>) => {
      const box = frame.current?.contentWindow;
      // Identity, not origin: an opaque origin reports itself as "null".
      if (!box || event.source !== box) return;
      const message = event.data;
      if (!message || message.channel !== CHANNEL) return;
      if (message.token !== token.current) return;
      if (message.kind !== "call" || typeof message.type !== "string") return;

      const id = message.id;
      const type = message.type;

      if (RESERVED.some((prefix) => type.startsWith(prefix))) {
        return reply({
          kind: "result",
          id,
          error: `${type} is the frame's, not a widget's`,
          code: "not_permitted",
        });
      }
      if (inFlight >= MAX_PENDING) {
        return reply({
          kind: "result",
          id,
          error: "too many Requests in flight",
          code: "busy",
        });
      }

      inFlight += 1;
      sdk(type, message.args ?? {}).then(
        (data) => reply({ kind: "result", id, data }),
        (error: unknown) => reply({
          kind: "result",
          id,
          error: error instanceof Error ? error.message : String(error),
          code: error instanceof RequestFailed ? error.code : "",
        }),
      ).finally(() => { inFlight -= 1; });
    };

    window.addEventListener("message", onMessage);
    return () => window.removeEventListener("message", onMessage);
  }, []);

  /** The scheme, pushed rather than rebuilt — see the note above. */
  useEffect(() => {
    frame.current?.contentWindow?.postMessage(
      { channel: CHANNEL, kind: "scheme", value: scheme, css: tokenBlock(scheme) },
      "*",
    );
  }, [scheme, source]);

  /**
   * The box's size, pushed on change.
   *
   * A widget is sized *by the frame* — the panel's width, the drawer sliding,
   * the window resizing — and is never consulted about it. Being told is what
   * lets one lay itself out at the size it has without measuring a window it
   * cannot see.
   */
  useEffect(() => {
    const element = frame.current;
    if (!element || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      element.contentWindow?.postMessage(
        { channel: CHANNEL, kind: "size", value: { width, height } },
        "*",
      );
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, [source]);

  if (failure) {
    return (
      <p className="text-muted-foreground p-4 text-xs">
        {failure}
      </p>
    );
  }

  return (
    <iframe
      ref={frame}
      // Named for the person, not the file: this is what a screen reader
      // announces when it reaches the box.
      title={widget.name}
      // The whole boundary, and the absence is the point. See the module note.
      sandbox="allow-scripts"
      srcDoc={source ?? ""}
      // `block` rather than the default `inline`, which leaves a descender's
      // worth of background under every iframe, and a border of zero because
      // the browser's default is a 2px inset ridge nobody asked for.
      className="block h-full w-full border-0"
    />
  );
};
