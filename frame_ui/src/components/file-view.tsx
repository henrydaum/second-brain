/**
 * Showing a host file, whatever it turns out to be.
 *
 * One component, one prop, eight outcomes — because the alternative is every
 * caller deciding for itself what a `.png` is, and a drawer that draws files
 * differently from the transcript that produced them.
 *
 * **The bytes come from `/files`, not from `fs.read_bytes`.** Reading windows
 * over the SDK into a `Blob` and then an object URL works for a small image and
 * cannot work at all for video: seeking needs `Range`, and `Range` needs a URL.
 * It also meant holding every file a long conversation ever showed in memory at
 * once. `fileUrl` hands the browser a real URL with a real `Content-Type` and
 * lets it do what it is good at.
 *
 * **Every kind wears the same frame.** A bordered box, the same skeleton while
 * it loads, the same one-line explanation when it fails. The point is that a
 * CSV and an MP4 look like two views of one system rather than two features,
 * and that a failure reads as a fact about the file rather than as the UI
 * breaking.
 *
 * All of this is a single file rather than a directory: the renderers are ten
 * to thirty lines each and they only make sense next to the frame they share.
 */

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type FC,
  type ReactNode,
  type Ref,
} from "react";
import { DownloadIcon, FileIcon, MusicIcon } from "lucide-react";

import { HighlightedCode } from "@/components/assistant-ui/code-block";
import { DotMatrix } from "@/components/assistant-ui/dot-matrix";
import { useMarkdownMode } from "@/components/markdown-mode";
import { MarkdownPreview } from "@/components/markdown-preview";
import { fileUrl } from "@/lib/client";
import { delimiterFor, parseDelimited } from "@/lib/csv";
import {
  describeStatus,
  fetchWholeBytes,
  FileUnavailable,
  formatBytes,
  kindOf,
  nameOf,
  probeStatus,
  readText,
  suffixOf,
  type FileKind,
} from "@/lib/files";
import { useRememberedScroll } from "@/lib/scroll-memory";
import { cn } from "@/lib/utils";
import { useFileActivityMaybe } from "@/runtime/file-activity-provider";

/** How much of a table anybody reads in a pane. Past this it is a data set
 *  rather than a document, and the download link is the honest offer. */
const ROW_CAP = 200;

/** Uploads currently top out below this. Keeping a bound prevents a corrupt
 *  range response from turning a preview into unbounded browser memory. */
const PDF_CAP = 64 * 1024 * 1024;

type FileViewSize = "inline" | "full";

/* ── The shared frame ───────────────────────────────────────────────── */

const Frame: FC<{
  children: ReactNode;
  className?: string;
  /** For the frames that scroll, so `useRememberedScroll` can reach the box
   *  that actually does it. A plain prop — React 19 stopped needing
   *  `forwardRef` for this. */
  ref?: Ref<HTMLDivElement>;
  /**
   * The accessible name, for the frames that scroll — and what makes them
   * focusable.
   *
   * **A scrollable box that cannot take focus cannot be scrolled from the
   * keyboard**, and only Firefox gives one a tab stop unasked. Without this the
   * arrow keys do nothing in the viewer: the browser scrolls the nearest
   * scrollable *ancestor* of whatever has focus, and the box holding the
   * document is a descendant of the dialog rather than an ancestor of anything
   * focusable. One `tabindex` buys arrows, Page Up/Down, Home and End, all of
   * them native and none of them written here.
   */
  scrolls?: string;
}> = ({ children, className, ref, scrolls }) => (
  <div
    ref={ref}
    data-slot="file-view"
    tabIndex={scrolls === undefined ? undefined : 0}
    aria-label={scrolls}
    className={cn(
      "flex items-center justify-center overflow-hidden rounded-lg border",
      // Documents need a uniform reading surface even inside a glass viewer.
      scrolls !== undefined ? "bg-popover" : "bg-muted/30",
      scrolls !== undefined &&
        "focus-visible:ring-ring outline-none focus-visible:ring-2",
      className,
    )}
  >
    {children}
  </div>
);

/**
 * Nothing yet.
 *
 * **At `full` this fills whatever it is given rather than choosing a size.**
 * The viewer hands it a stage of fixed dimensions, so the box you see while a
 * file loads is exactly the box the file lands in — no reflow, and nothing
 * jumping under the pointer half a second after you clicked. Sizing itself
 * here is what made the old one a thin tall sliver that then sprang open.
 */
const Loading: FC<{ path: string; size: FileViewSize }> = ({ path, size }) => (
  <Frame className={size === "full" ? "size-full" : "h-40 w-full"}>
    <span className="text-muted-foreground inline-flex items-center gap-2 text-xs">
      <DotMatrix state="loading" aria-hidden />
      {nameOf(path)}
    </span>
  </Frame>
);

/**
 * Why there is nothing to show.
 *
 * **A path in a ledger row is a record of what happened, not a promise the file
 * is still there.** So this is an ordinary outcome rather than an error state,
 * and it says which outcome: a file the policy refuses still exists and a file
 * that 404s does not, and those want different things done about them.
 */
const Unavailable: FC<{ path: string; reason: string; size: FileViewSize }> = ({
  path,
  reason,
  size,
}) => (
  <Frame
    className={cn(
      "flex-col gap-1 p-6 text-center",
      size === "full" ? "size-full" : "w-full",
    )}
  >
    <FileIcon className="text-muted-foreground size-5" aria-hidden />
    <span className="text-sm font-medium">{nameOf(path)}</span>
    <span className="text-muted-foreground text-xs">{reason}</span>
  </Frame>
);

/* ── Asynchronous odds and ends ─────────────────────────────────────── */

/** Whichever renderer this path wants, once the kernel has been asked. */
function useKind(path: string): FileKind | null {
  const [kind, setKind] = useState<FileKind | null>(null);

  useEffect(() => {
    let cancelled = false;
    setKind(null);
    void kindOf(path).then((answer) => {
      if (!cancelled) setKind(answer);
    });
    return () => {
      cancelled = true;
    };
  }, [path]);

  return kind;
}

/**
 * The ref every scrolling frame takes: it keeps its place, and it takes focus.
 *
 * The focus half only applies at `full`, where the file *is* the screen it is
 * on — so the arrow keys scroll it the moment it opens, with no click first.
 * Inline in a transcript the same call would be scroll-jacking: a file preview
 * appearing mid-conversation must not steal the caret from the composer.
 *
 * `preventScroll` because the place has just been restored and revealing a
 * freshly focused element is exactly the kind of thing that would undo it.
 */
function useScroller(path: string, variant: string, size: FileViewSize) {
  const remember = useRememberedScroll(path, variant);

  return useCallback(
    (node: HTMLDivElement | null) => {
      const cleanup = remember(node);
      if (node && size === "full") node.focus({ preventScroll: true });
      return cleanup;
    },
    [remember, size],
  );
}

type Loaded = { text: string; truncated: boolean; total: number };

/** A text-shaped file, whole — see `fetchWhole` for why that takes a loop, and
 *  `readText` for why it usually does not have to. */
function useText(path: string): {
  loaded: Loaded | null;
  failure: string | null;
} {
  const [loaded, setLoaded] = useState<Loaded | null>(null);
  const [failure, setFailure] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoaded(null);
    setFailure(null);

    void (async () => {
      try {
        const whole = await readText(path);
        if (!cancelled) setLoaded(whole);
      } catch (error) {
        if (cancelled) return;
        setFailure(
          error instanceof FileUnavailable
            ? error.message
            : "This file could not be read.",
        );
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [path]);

  return { loaded, failure };
}

/**
 * What a media element's silent failure actually was.
 *
 * `<img>` and `<video>` report trouble as a bare `onError` with no status on it
 * anywhere, so a deleted file and a refused one look identical. One `HEAD`
 * afterwards is what turns that into the right sentence — and it only costs a
 * round trip on the path where something already went wrong.
 */
function useMediaFailure(path: string) {
  const [failure, setFailure] = useState<string | null>(null);

  useEffect(() => setFailure(null), [path]);

  const onError = () => {
    void probeStatus(path).then((status) =>
      setFailure(
        status === 0
          ? "This file could not be reached."
          : status < 300
            ? // The file is there and readable, so the element is the one
              // objecting: a codec the browser has no decoder for, or bytes
              // that do not match the extension they were given.
              "This file is there, but the browser could not display it."
            : describeStatus(status),
      ),
    );
  };

  return { failure, onError };
}

/* ── The renderers ──────────────────────────────────────────────────── */

const ImageView: FC<{ path: string; size: FileViewSize }> = ({ path, size }) => {
  const { failure, onError } = useMediaFailure(path);
  if (failure) return <Unavailable path={path} reason={failure} size={size} />;

  return (
    <img
      src={fileUrl(path)}
      alt={nameOf(path)}
      loading="lazy"
      decoding="async"
      onError={onError}
      className={cn(
        "rounded-lg border object-contain",
        // `max-*`, never a fixed size: the stage already has the dimensions,
        // and the picture fits itself inside them.
        size === "full" ? "max-h-full max-w-full" : "max-h-80",
      )}
    />
  );
};

const VideoView: FC<{ path: string; size: FileViewSize }> = ({ path, size }) => {
  const { failure, onError } = useMediaFailure(path);
  if (failure) return <Unavailable path={path} reason={failure} size={size} />;

  return (
    // No `preload="auto"`: the route honours Range, so the browser fetches the
    // header, draws a first frame, and leaves the rest until the scrubber asks.
    <video
      controls
      preload="metadata"
      src={fileUrl(path)}
      onError={onError}
      className={cn(
        "w-full rounded-lg border bg-black",
        size === "full" ? "max-h-full" : "max-h-80",
      )}
    />
  );
};

const AudioView: FC<{ path: string; size: FileViewSize }> = ({ path, size }) => {
  const { failure, onError } = useMediaFailure(path);
  if (failure) return <Unavailable path={path} reason={failure} size={size} />;

  return (
    <Frame className="w-full flex-col items-stretch gap-2 p-3">
      <span className="inline-flex items-center gap-2 text-xs font-medium">
        <MusicIcon className="size-3.5 shrink-0" aria-hidden />
        <span className="truncate">{nameOf(path)}</span>
      </span>
      <audio
        controls
        preload="metadata"
        src={fileUrl(path)}
        onError={onError}
        className="w-full"
      />
    </Frame>
  );
};

const TableView: FC<{ path: string; size: FileViewSize }> = ({ path, size }) => {
  const { loaded, failure } = useText(path);
  const scroller = useScroller(path, size, size);
  if (failure) return <Unavailable path={path} reason={failure} size={size} />;
  if (!loaded) return <Loading path={path} size={size} />;

  const rows = parseDelimited(loaded.text, delimiterFor(suffixOf(path)));
  if (!rows.length) {
    return <Unavailable path={path} reason="This file is empty." size={size} />;
  }

  // The first row is the header, which is what a spreadsheet export always
  // means by it — and being wrong costs one misnamed column rather than a
  // misread table.
  const [header, ...body] = rows;
  const shown = body.slice(0, ROW_CAP);

  return (
    <div className={cn("flex w-full flex-col", size === "full" && "h-full")}>
      {/* The scroll lives here, not on the page: a table thirty columns wide
          must not make the whole conversation scroll sideways. */}
      <Frame
        ref={scroller}
        scrolls={nameOf(path)}
        className="document-scrollbar block w-full min-h-0 flex-1 overflow-auto"
      >
        <table className="w-full border-collapse text-xs">
          <thead className="bg-muted/60 sticky top-0">
            <tr>
              {header.map((cell, i) => (
                <th
                  key={i}
                  className="border-b px-2.5 py-1.5 text-start font-medium whitespace-nowrap"
                >
                  {cell}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {shown.map((row, r) => (
              <tr key={r} className="even:bg-muted/20">
                {header.map((_, c) => (
                  <td
                    key={c}
                    className="max-w-64 truncate border-b px-2.5 py-1 align-top"
                    title={row[c] ?? ""}
                  >
                    {row[c] ?? ""}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </Frame>
      <p className="text-muted-foreground mt-1.5 shrink-0 text-[11px]">
        {body.length} {body.length === 1 ? "row" : "rows"}
        {shown.length < body.length && ` · showing ${shown.length}`}
        {loaded.truncated &&
          ` · read the first ${formatBytes(loaded.text.length)} of ${formatBytes(loaded.total)}`}
      </p>
    </div>
  );
};

/**
 * Markdown, rendered — with the source one click away.
 *
 * **Rendering it by default is a reversal, and a deliberate one.** This used to
 * go through `TextView` on the argument that what the agent *wrote* is the
 * thing being inspected, and that rendering it hides the difference between a
 * file and its output. That argument holds for a file the agent just produced
 * and it is simply wrong for the far commoner case: a note out of the vault,
 * which was written to be read. So the reader picks.
 *
 * **The picker is not here.** It sits in the viewer's footer, next to the
 * download link, because a control of its own costs a band of vertical space
 * above every note whether anybody touches it or not — see
 * `components/markdown-mode.tsx` for how the two halves find each other, and
 * why that is a module rather than a context.
 */
const MarkdownView: FC<{ path: string; size: FileViewSize }> = ({
  path,
  size,
}) => {
  const { loaded, failure } = useText(path);
  const mode = useMarkdownMode();
  // Preview and Source are the same file at wildly different heights, so they
  // remember where they were separately — landing halfway down the source
  // because that is halfway down the rendering would be worse than the top.
  const scroller = useScroller(path, `${size}:${mode}`, size);
  // Null wherever the viewer is mounted outside the file-activity provider,
  // which is what makes following a link between notes optional rather than a
  // crash — see `useFileActivityMaybe`.
  const activity = useFileActivityMaybe();

  /**
   * Following a link, as a callback that never changes identity.
   *
   * Through a ref because the context value does change — a new one every time
   * the agent touches a file — and a fresh callback would defeat the memo on
   * `MarkdownPreview`, re-parsing the whole note each time. Nothing here needs
   * the *current* activity except at the moment of a click, which is exactly
   * what a ref is for.
   */
  const latest = useRef(activity);
  latest.current = activity;
  const openFile = useCallback(
    (target: string) => latest.current?.view([target], 0),
    [],
  );

  if (failure) return <Unavailable path={path} reason={failure} size={size} />;
  if (!loaded) return <Loading path={path} size={size} />;

  return (
    <div className={cn("flex w-full flex-col", size === "full" && "h-full")}>
      <Frame
        ref={scroller}
        scrolls={nameOf(path)}
        className={cn(
          "document-scrollbar block w-full overflow-auto",
          size === "full" ? "min-h-0 flex-1" : "max-h-80",
        )}
      >
        {mode === "preview" ? (
          <MarkdownPreview
            text={loaded.text}
            path={path}
            onOpenFile={activity ? openFile : undefined}
            // A measure, not a full-bleed column: the dialog is as wide as the
            // screen allows and prose set across all of it is unreadable. The
            // inline copy is already narrow, so it only pays the padding.
            className={cn(
              "p-3",
              size === "full" && "mx-auto max-w-[72ch] p-4 sm:p-6",
            )}
          />
        ) : (
          <HighlightedCode
            code={loaded.text}
            language="md"
            // The `Frame` is already the surface; a second background inside it
            // reads as a lighter panel floating in a darker one.
            transparent
            className="p-3 font-mono text-xs leading-relaxed break-words whitespace-pre-wrap"
          />
        )}
      </Frame>
      {loaded.truncated && (
        <p className="text-muted-foreground mt-1.5 shrink-0 text-[11px]">
          The first {formatBytes(loaded.text.length)} of{" "}
          {formatBytes(loaded.total)} — download it to see the rest.
        </p>
      )}
    </div>
  );
};

const TextView: FC<{ path: string; size: FileViewSize }> = ({ path, size }) => {
  const { loaded, failure } = useText(path);
  const scroller = useScroller(path, size, size);
  if (failure) return <Unavailable path={path} reason={failure} size={size} />;
  if (!loaded) return <Loading path={path} size={size} />;

  return (
    <div className={cn("flex w-full flex-col", size === "full" && "h-full")}>
      <Frame
        ref={scroller}
        scrolls={nameOf(path)}
        className={cn(
          "document-scrollbar block w-full overflow-auto p-3",
          size === "full" ? "min-h-0 flex-1" : "max-h-80",
        )}
      >
        {/* Shown as source, which for everything that reaches here is the only
            thing it could be — Markdown, the one text format with a rendering
            of its own, goes to `MarkdownView` above and keeps this as its other
            half. Coloured by extension, through the same highlighter the chat's
            fenced blocks use — a file read here and the same file quoted in a
            reply should not look like two different things. */}
        <HighlightedCode
          code={loaded.text}
          language={suffixOf(path).slice(1)}
          // The `Frame` around this is already the surface; a second background
          // inside it reads as a lighter panel floating in a darker one.
          transparent
          className="font-mono text-xs leading-relaxed break-words whitespace-pre-wrap"
        />
      </Frame>
      {loaded.truncated && (
        <p className="text-muted-foreground mt-1.5 shrink-0 text-[11px]">
          The first {formatBytes(loaded.text.length)} of{" "}
          {formatBytes(loaded.total)} — download it to see the rest.
        </p>
      )}
    </div>
  );
};

/**
 * PDFs need a blob URL rather than `/files` directly. Chrome's native PDF
 * viewer runs in an extension frame, so the gateway's SAMEORIGIN protection
 * can reject the otherwise same-origin response. Fetching the bytes first
 * keeps that protection intact and gives the extension a local blob instead.
 */
const PdfView: FC<{ path: string; size: FileViewSize }> = ({ path, size }) => {
  const [url, setUrl] = useState<string | null>(null);
  const [failure, setFailure] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let objectUrl: string | null = null;
    setUrl(null);
    setFailure(null);

    void (async () => {
      try {
        const whole = await fetchWholeBytes(path, PDF_CAP);
        if (whole.truncated) {
          throw new Error("This PDF is too large to preview; download it instead.");
        }
        if (cancelled) return;
        objectUrl = URL.createObjectURL(
          new Blob([whole.bytes as BlobPart], { type: "application/pdf" }),
        );
        setUrl(objectUrl);
      } catch (error) {
        if (!cancelled) {
          setFailure(
            error instanceof Error
              ? error.message
              : "This PDF could not be read.",
          );
        }
      }
    })();

    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [path]);

  if (failure) return <Unavailable path={path} reason={failure} size={size} />;
  if (!url) return <Loading path={path} size={size} />;
  return (
    <embed
      src={url}
      type="application/pdf"
      title={nameOf(path)}
      className={cn(
        "w-full rounded-lg border",
        size === "full" ? "h-full" : "h-80",
      )}
    />
  );
};

/**
 * SVG can stay on its real URL; unlike Chrome's PDF viewer it does not cross
 * into an extension origin while rendering.
 *
 * **But it must not be given a scripting context, and `<embed>` gave it one.**
 * An SVG is a document, not a picture, whenever a framing element loads it: the
 * browser makes it a page *at this app's origin*, so a `<script>` inside one
 * ran as the app. That is not a hypothetical — the file could be anything the
 * agent generated, downloaded, or that somebody attached — and the consequence
 * is not a defaced pane: the production gateway attaches the backend bearer
 * token to same-origin `/sdk` calls, so such a script could read any file on
 * the host and write anywhere, without ever learning the token.
 *
 * `sandbox=""` — every permission withheld, `allow-scripts` above all — is what
 * takes that away. The document renders into an opaque origin with scripting
 * off, which for a picture costs nothing at all. `<iframe>` rather than
 * `<embed>` because only an iframe honours the attribute; the box, the border
 * and the radius are the same ones this always had.
 *
 * The gateway sends `Content-Security-Policy: sandbox` on `/files` as well, so
 * the same file opened at its own URL is neutered too. Two independent
 * mechanisms, because this one is worth not getting wrong once.
 */
const EmbedView: FC<{ path: string; size: FileViewSize }> = ({ path, size }) =>
  suffixOf(path) === ".pdf" ? (
    <PdfView path={path} size={size} />
  ) : (
    <iframe
      src={fileUrl(path)}
      title={nameOf(path)}
      sandbox=""
      className={cn(
        "w-full rounded-lg border",
        size === "full" ? "h-full" : "h-80",
      )}
    />
  );

/** Everything with no better answer. `/files` serves it as
 *  `application/octet-stream`, which is a download, so this offers one. */
const DownloadView: FC<{ path: string }> = ({ path }) => (
  <a
    href={fileUrl(path)}
    download={nameOf(path)}
    className="hover:bg-accent inline-flex items-center gap-2 rounded-lg border px-3 py-2 text-xs"
  >
    <DownloadIcon className="size-3.5" aria-hidden />
    {nameOf(path)}
  </a>
);

/* ── The one entry point ────────────────────────────────────────────── */

export const FileView: FC<{ path: string; size?: FileViewSize }> = ({
  path,
  size = "inline",
}) => {
  const kind = useKind(path);
  if (kind === null) return <Loading path={path} size={size} />;

  // No `default`: a ninth kind is a compile error here rather than a blank
  // pane somebody notices in a month.
  switch (kind) {
    case "image":
      return <ImageView path={path} size={size} />;
    case "video":
      return <VideoView path={path} size={size} />;
    case "audio":
      return <AudioView path={path} size={size} />;
    case "table":
      return <TableView path={path} size={size} />;
    case "markdown":
      return <MarkdownView path={path} size={size} />;
    case "text":
      return <TextView path={path} size={size} />;
    case "embed":
      return <EmbedView path={path} size={size} />;
    case "download":
      return <DownloadView path={path} />;
  }
};
