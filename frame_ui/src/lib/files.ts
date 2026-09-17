/**
 * Deciding what a host file *is*, and getting its bytes.
 *
 * Two jobs that belong together because they share the same trap: the obvious
 * answer is wrong in both.
 *
 * **Modality does not decide the renderer.** `parse.modality` answers the
 * question *how should the model ingest this*, and the viewer is asking a
 * different one — *how should a person look at this*. For most files the two
 * agree. For a handful they legitimately do not: `.csv` and `.md` answer
 * `"text"` and always will, though one is a table and the other is a document,
 * and `.pdf` and `.svg` answer `"unknown"` while the browser renders both
 * perfectly. `"unknown"` means *no parser is registered*, never *not
 * renderable*. So `kindOf` below checks those extensions first and only then
 * asks.
 *
 * **A `fetch` of a large file gets a fragment.** `/files` honours `Range`, and
 * one response body has to cross the wire in one message — so past a certain
 * size the route answers `206` with the first window *even though nothing sent
 * a `Range` header*, putting the real length in `Content-Range`. A media
 * element notices and follows up on its own; a `fetch` does not, and quietly
 * hands back a truncated file. `fetchWhole` is the loop that closes that gap,
 * and it is why the renderers that need bytes in hand all go through one
 * function.
 */

import { authHeaders, fileUrl, sdk } from "@/lib/client";
import { forgetPlaces } from "@/lib/scroll-memory";

/* ── Naming ─────────────────────────────────────────────────────────── */

/** The last segment of a path. Host paths may be POSIX or Windows — the server
 *  is a Mac and the file could have come from anywhere — so both separators
 *  count. */
export const nameOf = (path: string) => path.split(/[\\/]/).pop() || path;

/** Everything before the last segment, or "" at the root. Shown in muted type
 *  beside the name, because two files called `report.csv` in different
 *  directories are otherwise indistinguishable in a list. */
export function dirOf(path: string): string {
  const cut = path.search(/[\\/][^\\/]*$/);
  return cut <= 0 ? "" : path.slice(0, cut);
}

/**
 * A relative path, made absolute against the directory it was written in.
 *
 * A Markdown note refers to its neighbours the way a person would —
 * `attachments/chart.png`, `../plan.md` — and nothing else in this app has ever
 * needed to resolve one, because every other path arrives from the kernel
 * already whole.
 *
 * **The separator is the base's, not this platform's.** These are host paths
 * and the host is somebody else's machine; `node:path` would answer with
 * Windows separators in a browser bundle that never runs on Windows, and does
 * not exist there anyway.
 *
 * `..` stops at the root rather than walking off the top of it, so a note with
 * one `../` too many resolves to a path that is merely wrong instead of to
 * something that reads as a different drive.
 */
export function resolveAgainst(dir: string, relative: string): string {
  // Already absolute — POSIX, a Windows drive, or a UNC share. Nothing to join.
  if (/^([\\/]|[a-z]:[\\/])/i.test(relative)) return relative;

  const separator = dir.includes("\\") ? "\\" : "/";
  const parts = dir ? dir.split(/[\\/]/) : [];
  // A leading "" (POSIX root) or a bare "C:" is the floor; popping it would
  // turn an absolute path into a relative one halfway through the walk.
  const atRoot = () =>
    parts.length === 0 ||
    (parts.length === 1 && (parts[0] === "" || parts[0].endsWith(":")));

  for (const step of relative.split(/[\\/]/)) {
    if (step === "" || step === ".") continue;
    if (step === "..") {
      if (!atRoot()) parts.pop();
      continue;
    }
    parts.push(step);
  }

  return parts.join(separator);
}

/**
 * The extension, lowercased, **with its leading dot**.
 *
 * Deliberately not `extensionOf` from `lib/upload.ts`, which strips the dot:
 * that one feeds `frontend.submit`, whose `extension` argument is spelled
 * without one, and `parse.modality` is spelled with. Two callers, two
 * spellings, and converting between them at each call site is how one of them
 * eventually gets it wrong.
 */
export function suffixOf(path: string): string {
  const name = nameOf(path);
  const dot = name.lastIndexOf(".");
  return dot > 0 ? name.slice(dot).toLowerCase() : "";
}

/** Bytes, in the units a person reads. */
export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return "";
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit++;
  }
  // One decimal until three figures, where it stops adding anything: "13.7 MB"
  // is worth reading, "413.2 MB" is noise.
  return `${value < 100 ? value.toFixed(1) : Math.round(value)} ${units[unit]}`;
}

/* ── Which renderer ─────────────────────────────────────────────────── */

/**
 * How a file should be *shown*. Not a modality — see the note at the top.
 *
 * No `default` anywhere this is switched on, so adding a kind is a compile
 * error at every site that has to care.
 */
export type FileKind =
  | "image"
  | "video"
  | "audio"
  | "table"
  | "markdown"
  | "text"
  | "embed"
  | "download";

/** The renderer kinds plus the one visual-only distinction worth making.
 * Code is still rendered as text; it simply should not look like prose in a
 * file list or attachment tile. */
export type FileIconKind = FileKind | "code";

/** Extensions a person wants as a table, whatever the model wants. */
const TABLE = new Set([".csv", ".tsv"]);

/**
 * Extensions a person wants *rendered*, whatever the model wants.
 *
 * The same shape of disagreement as `.csv`: modality answers `"text"` and is
 * right — the model should ingest a note as text — while a reader opening one
 * wants the headings, the lists and the tables it was written to have.
 *
 * `.mdx` is deliberately absent. It is a source format with imports and JSX in
 * it, and rendering the JSX away hides most of the file.
 */
const MARKDOWN = new Set([".md", ".markdown", ".mdown", ".mkd", ".mkdn"]);

/** Programming and source/config formats that benefit from a code glyph.
 * This is deliberately icon-only: `kindOf` still asks the kernel how to open
 * them, while lists can identify them immediately without a Request each. */
const CODE = new Set([
  ".astro",
  ".bash",
  ".bat",
  ".c",
  ".cc",
  ".clj",
  ".cljs",
  ".cmd",
  ".cpp",
  ".cs",
  ".css",
  ".cxx",
  ".dart",
  ".ex",
  ".exs",
  ".fish",
  ".fs",
  ".fsx",
  ".go",
  ".h",
  ".hpp",
  ".html",
  ".htm",
  ".ipynb",
  ".java",
  ".js",
  ".json",
  ".jsx",
  ".kt",
  ".kts",
  ".less",
  ".lua",
  ".m",
  ".php",
  ".ps1",
  ".py",
  ".pyw",
  ".r",
  ".rb",
  ".rs",
  ".sass",
  ".scala",
  ".scss",
  ".sh",
  ".sql",
  ".svelte",
  ".swift",
  ".toml",
  ".ts",
  ".tsx",
  ".vb",
  ".vue",
  ".xml",
  ".yaml",
  ".yml",
  ".zsh",
]);

/** Extensions the browser draws natively and the kernel has no parser for.
 *  Modality-blind on purpose: both answer `"unknown"`. */
const EMBED = new Set([".pdf", ".svg"]);

/**
 * The static half of the modality map, as a fallback only.
 *
 * The kernel answers image/audio/video from a static table that needs no parser
 * installed, so this can only disagree with it by being out of date. It exists
 * for the case where the Request itself fails — a dropped connection should
 * still draw a PNG rather than offering to download it.
 */
const STATIC: Record<string, string> = {
  ".png": "image",
  ".jpg": "image",
  ".jpeg": "image",
  ".gif": "image",
  ".webp": "image",
  ".bmp": "image",
  ".heic": "image",
  ".mp4": "video",
  ".webm": "video",
  ".mov": "video",
  ".avi": "video",
  ".mkv": "video",
  ".wav": "audio",
  ".mp3": "audio",
  ".m4a": "audio",
  ".ogg": "audio",
  ".flac": "audio",
};

/**
 * The kernel's own answer for an extension, asked once per extension.
 *
 * Memoised on the *promise* rather than the value, so a burst of files sharing
 * an extension — which is the normal case, a folder of screenshots — makes one
 * Request rather than one each. It answers `"unknown"` and never `null`.
 */
const modalities = new Map<string, Promise<string>>();

function modalityOf(suffix: string): Promise<string> {
  const known = modalities.get(suffix);
  if (known) return known;

  const asked = sdk<string>("parse.modality", { extension: suffix })
    .then((answer) => (typeof answer === "string" ? answer : "unknown"))
    .catch(() => STATIC[suffix] ?? "unknown");

  modalities.set(suffix, asked);
  return asked;
}

/**
 * What to draw for a path.
 *
 * The order is the whole point and it is not modality-first:
 *
 * 1. `.csv`/`.tsv` → a table, because modality says `"text"` and means it.
 * 2. `.md` and friends → Markdown, for the same reason: `"text"` is the right
 *    answer to the question the kernel was asked and the wrong one here.
 * 3. `.pdf`/`.svg` → an embed, because modality says `"unknown"` and the
 *    browser disagrees.
 * 4. then ask, and take image/video/audio/text at their word.
 * 5. anything left is a download, which is what `/files` would serve it as.
 */
export async function kindOf(path: string): Promise<FileKind> {
  const suffix = suffixOf(path);
  if (TABLE.has(suffix)) return "table";
  if (MARKDOWN.has(suffix)) return "markdown";
  if (EMBED.has(suffix)) return "embed";

  switch (await modalityOf(suffix)) {
    case "image":
      return "image";
    case "video":
      return "video";
    case "audio":
      return "audio";
    case "text":
      return "text";
    default:
      return "download";
  }
}

/**
 * A guess at the kind, without asking.
 *
 * **For icons, and nothing else.** A list of forty files cannot await forty
 * Requests before it draws anything, and the icon beside a name is not worth
 * being right about — it is worth being *immediate* about. Where the answer
 * matters, `kindOf` asks.
 *
 * Everything the static map does not recognise comes back `"download"`, which
 * draws a plain file icon: the honest picture of "something, unopened".
 */
export function guessKind(path: string): FileKind {
  const suffix = suffixOf(path);
  if (TABLE.has(suffix)) return "table";
  if (MARKDOWN.has(suffix)) return "markdown";
  if (EMBED.has(suffix)) return "embed";
  switch (STATIC[suffix]) {
    case "image":
      return "image";
    case "video":
      return "video";
    case "audio":
      return "audio";
    default:
      return "download";
  }
}

/** Immediate classification for shared file icons. */
export function guessIconKind(path: string): FileIconKind {
  if (CODE.has(suffixOf(path))) return "code";
  return guessKind(path);
}

/* ── Getting the bytes ──────────────────────────────────────────────── */

/**
 * `/files` answered with something other than the file.
 *
 * Carries the status because that is the only thing that distinguishes the
 * cases people actually hit, and they need different sentences: a file the
 * policy refuses is still there, and a file that 404s is not.
 */
export class FileUnavailable extends Error {
  readonly status: number;

  // Written out rather than as a constructor parameter property: this project
  // compiles with `erasableSyntaxOnly`. See `RequestFailed` in `client.ts`.
  constructor(status: number) {
    super(describeStatus(status));
    this.name = "FileUnavailable";
    this.status = status;
  }
}

/**
 * What a status from `/files` means, in a sentence.
 *
 * A path in a ledger row is a record of what happened, not a promise the file
 * is still there — so `404` is the common case rather than the exotic one, and
 * saying "gone" is the difference between an explanation and an empty pane.
 */
export function describeStatus(status: number): string {
  switch (status) {
    case 400:
      return "This path is a directory, not a file.";
    case 403:
      return "Second Brain would not hand this file over.";
    case 404:
      return "This file is gone — moved or deleted since it was recorded.";
    case 416:
      return "This file is shorter than it was when it was recorded.";
    default:
      return `This file could not be read (${status}).`;
  }
}

/** How much of a text file a viewer will hold. Past this it is not something
 *  anybody is going to read in a pane, and the tail costs a round trip per
 *  window to fetch. */
export const TEXT_CAP = 2 * 1024 * 1024;

export type WholeText = {
  text: string;
  /** The file is longer than `TEXT_CAP` and this is the front of it. */
  truncated: boolean;
  /** The file's real length in bytes, as the server reports it. */
  total: number;
};

export type WholeBytes = {
  bytes: Uint8Array;
  /** The file is longer than the caller's cap and this is only its front. */
  truncated: boolean;
  /** The file's real length in bytes, as the server reports it. */
  total: number;
};

/** `bytes 0-8191/40000` → 40000. Null when the header is absent or unparseable,
 *  which is how a `200` is told from a `206`. */
function totalFrom(header: string | null): number | null {
  const match = /\/(\d+)\s*$/.exec(header ?? "");
  return match ? Number(match[1]) : null;
}

/**
 * Read a whole text file, following the windows the route serves it in.
 *
 * The loop is the point — see the note at the top of this file. It stops on
 * three conditions, and the third is a guard rather than an expectation: a
 * window that returns no bytes would otherwise spin forever against a server
 * that has stopped making progress.
 *
 * Decoding happens once, at the end, over the joined bytes. Decoding each
 * window separately would corrupt any multi-byte character that straddled a
 * boundary — a rare bug that looks like a corrupt file rather than a bad read.
 */
export async function fetchWholeBytes(
  path: string,
  cap: number = TEXT_CAP,
): Promise<WholeBytes> {
  const url = fileUrl(path);
  const windows: Uint8Array[] = [];
  let at = 0;
  let total = Number.POSITIVE_INFINITY;

  while (at < total && at < cap) {
    const response = await fetch(url, {
      headers: {
        ...authHeaders(),
        // Omitted on the first pass so the server picks its own window; after
        // that we are explicitly asking for the rest.
        ...(at > 0 ? { Range: `bytes=${at}-` } : {}),
      },
    });
    if (!response.ok) throw new FileUnavailable(response.status);

    const reported = totalFrom(response.headers.get("Content-Range"));
    if (reported !== null) total = reported;

    const window = new Uint8Array(await response.arrayBuffer());
    // A 200 means the whole body arrived; there is no second window to ask for.
    if (reported === null) total = at + window.length;

    if (window.length === 0) break;
    windows.push(window);
    at += window.length;
  }

  const joined = new Uint8Array(windows.reduce((n, w) => n + w.length, 0));
  let offset = 0;
  for (const window of windows) {
    joined.set(window, offset);
    offset += window.length;
  }

  return {
    bytes: joined,
    truncated: at < total,
    total: Number.isFinite(total) ? total : joined.length,
  };
}

/** Text-flavoured wrapper around the byte-preserving range loop above. */
export async function fetchWhole(
  path: string,
  cap: number = TEXT_CAP,
): Promise<WholeText> {
  const whole = await fetchWholeBytes(path, cap);
  return {
    text: new TextDecoder().decode(whole.bytes),
    truncated: whole.truncated,
    total: whole.total,
  };
}

/**
 * Text already read, so that reopening a file is instant.
 *
 * Images cost nothing to reopen because the browser's own cache holds them —
 * same URL, same bytes. Text and tables go through `fetch`, get no cache
 * headers from this route, and were re-read and re-parsed on every open, which
 * is the half-second pause you see stepping back and forth through a group.
 *
 * **Bounded, and invalidated by the ledger.** A handful of files, because one
 * of them may be two megabytes; and dropped the moment a ledger row says the
 * file changed, which is the one signal that makes a cached copy wrong. That
 * signal already arrives on its own — see `forgetFile`.
 */
const texts = new Map<string, Promise<WholeText>>();
const TEXT_CACHE = 8;

export function readText(path: string): Promise<WholeText> {
  const known = texts.get(path);
  if (known) {
    // Re-insert to mark it as the most recently used; `Map` keeps insertion
    // order, which is the whole of the eviction policy below.
    texts.delete(path);
    texts.set(path, known);
    return known;
  }

  const asked = fetchWhole(path);
  texts.set(path, asked);
  // A failure is never cached. A file that is missing now may be back in a
  // moment, and re-asking costs one request.
  void asked.catch(() => texts.delete(path));

  for (const oldest of texts.keys()) {
    if (texts.size <= TEXT_CACHE) break;
    texts.delete(oldest);
  }
  return asked;
}

/** Drop what was held for a path, because it is not that file any more. Called
 *  with every path a ledger poll turns up — the agent writing a file is exactly
 *  the event that makes a cached copy of it a lie, and it makes a remembered
 *  scroll offset into one too: a place in a document that has since been
 *  rewritten points at whatever happens to be there now. */
export function forgetFile(path: string): void {
  texts.delete(path);
  forgetPlaces(path);
}

/**
 * Ask what the server thinks of a path, without reading it.
 *
 * **This is how a broken `<img>` gets an explanation.** A media element reports
 * failure as a bare `onError` with no status anywhere on it, so a file that was
 * deleted and a file the policy refused look identical. One request afterwards
 * turns that into the right sentence, and it costs a round trip only on the
 * path where something already went wrong.
 *
 * **A one-byte `Range`, not a `HEAD`.** `HEAD` is the obvious choice and it is
 * the wrong one: on its error path the route answers with the `Content-Length`
 * of a body it then does not send, and the dev proxy turns that mismatch into a
 * `502` — so the one request whose entire job is to report the real status
 * reported `502` for every failure there is. Asking for a single byte gets a
 * true status with no body worth mentioning.
 */
export async function probeStatus(path: string): Promise<number> {
  try {
    const response = await fetch(fileUrl(path), {
      headers: { ...authHeaders(), Range: "bytes=0-0" },
    });
    return response.status;
  } catch {
    // The request never landed — offline, or the server went away. Not a
    // status, and not something 404's sentence should be put on.
    return 0;
  }
}
