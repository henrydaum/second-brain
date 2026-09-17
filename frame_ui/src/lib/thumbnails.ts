/**
 * Small pictures for file lists, made once and kept.
 *
 * ## The problem this exists for
 *
 * A row in the files drawer is a 36-pixel square, and what used to fill it was
 * `<img src={fileUrl(path)}>` — the original file. CSS made it 36 pixels; the
 * browser did not. A phone photo is decoded at its real size and held as a
 * bitmap of four bytes per pixel, so one 12-megapixel picture in a thumbnail
 * costs about 48MB of renderer memory. Thirty of them in one conversation is a
 * gigabyte and a half, which is the point where the panel stops being slow and
 * starts being broken.
 *
 * The transfer was as bad as the decode. `/files` sends no validators, so the
 * browser's HTTP cache cannot reuse anything — and below `xl` the drawer lives
 * in a `Sheet`, which Radix unmounts on close. Every single opening therefore
 * re-read every photo in the conversation across the tunnel, through the
 * kernel's policy check, writing one ledger row apiece.
 *
 * ## What replaces it
 *
 * Read the file once, decode it once, and keep a square the size the box
 * actually is. Everything else here is in service of "once":
 *
 * - **Keyed by path and version.** The version is the timestamp of the most
 *   recent thing that happened to the file, so a rewritten file is a different
 *   key and a stale thumbnail can never be served for it.
 * - **Data URLs, not object URLs.** An object URL has to be revoked, and an
 *   LRU that revokes a URL some off-screen `<img>` is still pointing at breaks
 *   that row when it scrolls back into view. A data URL is a string: eviction
 *   is forgetting it, and anything still holding one still works. A 192-pixel
 *   WebP is a few kilobytes, so the strings are cheaper than the bug.
 * - **A `CacheStorage` copy**, so a reload does not repeat the reads. This is
 *   the `Cache-Control` header the route does not send, kept on our side.
 * - **Two at a time.** The decode holds one full-size bitmap while it runs, and
 *   the whole point is not to hold thirty. Newest request first: what you just
 *   scrolled to matters more than what you scrolled past.
 *
 * None of it is required. Where `createImageBitmap` is missing — jsdom, and
 * anything old enough to matter — callers fall back to the original URL and
 * behave exactly as they did before.
 */

import { fetchWholeBytes } from "@/lib/files";

/**
 * The side of a stored thumbnail, in device pixels.
 *
 * The largest box that draws one is the 56px attachment tile, so this covers it
 * on a 3× screen with a little left over. One size for every caller: a second
 * size would double the reads to save a few kilobytes of string.
 */
export const THUMB_PX = 192;

/** How many thumbnails stay in memory. Each is a few kilobytes of base64. */
const MEMORY_CACHE = 400;

/** How much of a file will be read to make a thumbnail. Past this the picture
 *  is not worth what it costs to fetch it; the row keeps its icon. */
const IMAGE_CAP = 32 * 1024 * 1024;

/** Decodes in flight. See the note above — this is the memory ceiling. */
const WIDTH = 2;

/** Bump the version to abandon every stored thumbnail, should `THUMB_PX` or
 *  the encoding change. Older caches are deleted on first use. */
const STORE = "sb-thumbs-v1";

/** Roughly how many thumbnails the persistent store keeps. Trimmed from the
 *  front, which for `CacheStorage` is insertion order — oldest first. */
const STORE_CAP = 800;

/* ── The cache ──────────────────────────────────────────────────────── */

/** Key → the promise of a data URL. `Map` keeps insertion order, which is the
 *  whole of the eviction policy. */
const made = new Map<string, Promise<string>>();

/**
 * Keys that could not be made into a picture.
 *
 * **Remembered, unlike a failed text read.** A path in this app is a record of
 * what happened rather than a promise the file is still there, so a missing
 * file is the common case, not the exception — and the drawer asks again on
 * every mount. Without this, a conversation full of deleted photos re-runs a
 * 404 per row per opening. `forgetThumbnail` clears it when a ledger row says
 * the file changed, which is the one event that could make it readable again.
 */
const failed = new Set<string>();

/** Whether a thumbnail can be made at all here. */
export function thumbnailsWork(): boolean {
  return (
    typeof createImageBitmap === "function" && typeof document !== "undefined"
  );
}

/** Between a path and its version in a cache key. A character no path has,
 *  so no path can spell another path's key. */
const SEPARATOR = String.fromCharCode(0);

function keyFor(path: string, version?: string | number): string {
  return version === undefined ? path : path + SEPARATOR + version;
}

/**
 * A square picture of a file, as a data URL.
 *
 * Rejects when the file cannot be read or is not something the browser decodes
 * — callers draw their icon and leave it at that.
 */
export function thumbnail(
  path: string,
  version?: string | number,
): Promise<string> {
  const key = keyFor(path, version);

  const known = made.get(key);
  if (known) {
    // Re-insert to mark it most recently used.
    made.delete(key);
    made.set(key, known);
    return known;
  }
  if (failed.has(key)) {
    return Promise.reject(new Error(`No thumbnail for ${path}`));
  }

  const asked = build(path, key);
  made.set(key, asked);
  void asked.catch(() => {
    made.delete(key);
    failed.add(key);
  });

  for (const oldest of made.keys()) {
    if (made.size <= MEMORY_CACHE) break;
    made.delete(oldest);
  }
  return asked;
}

/** Drop what was held for a path, because it is not that file any more. Called
 *  alongside `forgetFile`; see its note. A versioned key already misses after a
 *  rewrite, so this is mostly about letting a failure be retried. */
export function forgetThumbnail(path: string): void {
  // Deleting the entry the loop is standing on is defined for both of these,
  // so neither needs copying first.
  const prefix = path + SEPARATOR;
  for (const key of made.keys()) {
    if (key === path || key.startsWith(prefix)) made.delete(key);
  }
  for (const key of failed) {
    if (key === path || key.startsWith(prefix)) failed.delete(key);
  }
}

/** Everything, for tests. */
export function forgetThumbnails(): void {
  made.clear();
  failed.clear();
}

async function build(path: string, key: string): Promise<string> {
  const stored = await fromStore(key);
  if (stored) return stored;

  const encoded = await queued(() => render(path));
  void toStore(key, encoded);
  return await dataUrl(encoded);
}

/* ── Making one ─────────────────────────────────────────────────────── */

/**
 * Read the file, decode it once, and draw the square.
 *
 * `fetchWholeBytes` rather than handing the URL to an `<img>`: the route serves
 * a large file as `206` with only the first window even when nothing asked for
 * a range, and that loop is what follows the rest of it.
 *
 * The bitmap is closed the moment it has been drawn. It is the expensive object
 * in this whole file — the full-size decode — and leaving it to the garbage
 * collector puts the peak this function exists to bound back on the heap.
 */
async function render(path: string): Promise<Blob> {
  const whole = await fetchWholeBytes(path, IMAGE_CAP);
  const source = new Blob([whole.bytes as BlobPart]);

  // `from-image` so a photo taken sideways is a thumbnail the right way up: a
  // phone writes the rotation as EXIF and leaves the pixels where they were.
  const bitmap = await createImageBitmap(source, {
    imageOrientation: "from-image",
  });

  try {
    return await square(bitmap);
  } finally {
    bitmap.close();
  }
}

/**
 * The bitmap, centre-cropped into a `THUMB_PX` square.
 *
 * Cropped here rather than left to `object-fit: cover` in the row: cropping at
 * this size means the stored picture is the picture that gets drawn, so no
 * caller can be handed a 20000-pixel-tall panorama squashed into a 3-pixel
 * band and asked to make a tile out of it.
 */
async function square(bitmap: ImageBitmap): Promise<Blob> {
  const side = Math.min(bitmap.width, bitmap.height);
  if (!side) throw new Error("An image with no pixels in it");
  const left = (bitmap.width - side) / 2;
  const top = (bitmap.height - side) / 2;

  const canvas = surface(THUMB_PX);
  const context = canvas.getContext("2d") as
    | OffscreenCanvasRenderingContext2D
    | CanvasRenderingContext2D
    | null;
  if (!context) throw new Error("No 2d context for a thumbnail");
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(bitmap, left, top, side, side, 0, 0, THUMB_PX, THUMB_PX);

  return await encode(canvas);
}

type Surface = OffscreenCanvas | HTMLCanvasElement;

/** An `OffscreenCanvas` where there is one — its encode does not run on the
 *  main thread — and an ordinary canvas everywhere else. */
function surface(side: number): Surface {
  if (typeof OffscreenCanvas === "function") {
    return new OffscreenCanvas(side, side);
  }
  const canvas = document.createElement("canvas");
  canvas.width = side;
  canvas.height = side;
  return canvas;
}

/** WebP, which is a fraction of PNG at this quality and is understood
 *  everywhere `createImageBitmap` is. A canvas that refuses the type quietly
 *  answers PNG instead rather than failing, which is a fine outcome. */
async function encode(canvas: Surface): Promise<Blob> {
  if ("convertToBlob" in canvas) {
    return await canvas.convertToBlob({ type: "image/webp", quality: 0.82 });
  }
  return await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error("Encode failed"))),
      "image/webp",
      0.82,
    );
  });
}

function dataUrl(blob: Blob): Promise<string> {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error ?? new Error("Read failed"));
    reader.readAsDataURL(blob);
  });
}

/* ── Two at a time, newest first ────────────────────────────────────── */

let running = 0;
const waiting: (() => void)[] = [];

/**
 * Hold work until there is room for it.
 *
 * **The queue is a stack.** Opening a long drawer asks for every visible row at
 * once and then scrolling asks for more; the rows you are looking at now are
 * the ones asked for last. First-in-first-out would fill the top of a list
 * nobody is reading any more before it touched the part on screen.
 */
async function queued<T>(work: () => Promise<T>): Promise<T> {
  if (running >= WIDTH) {
    await new Promise<void>((resume) => waiting.push(resume));
  }
  running += 1;
  try {
    return await work();
  } finally {
    running -= 1;
    waiting.pop()?.();
  }
}

/* ── Keeping them across reloads ────────────────────────────────────── */

/**
 * The persistent half, which is allowed to be missing.
 *
 * `CacheStorage` needs a secure context, which loopback and the tunnel both
 * are, but a browser can still refuse for its own reasons — a private window,
 * storage pressure, a policy. Every path through here swallows its own failure
 * and answers nothing, because the only cost of a miss is the work this was
 * trying to save.
 */
async function store(): Promise<Cache | null> {
  if (typeof caches === "undefined") return null;
  try {
    const cache = await caches.open(STORE);
    void sweep();
    return cache;
  } catch {
    return null;
  }
}

/** A key as something `CacheStorage` will accept. Not a real route — nothing
 *  ever fetches it — but a `Request` must be a URL, so it is a URL. */
function stamped(key: string): Request {
  return new Request(
    new URL(
      `/thumb/${STORE}?key=${encodeURIComponent(key)}`,
      window.location.origin,
    ),
  );
}

async function fromStore(key: string): Promise<string | null> {
  try {
    const cache = await store();
    const hit = await cache?.match(stamped(key));
    if (!hit) return null;
    return await dataUrl(await hit.blob());
  } catch {
    return null;
  }
}

async function toStore(key: string, blob: Blob): Promise<void> {
  try {
    const cache = await store();
    await cache?.put(stamped(key), new Response(blob));
  } catch {
    // Out of quota, most likely. Memory still has it for this session.
  }
}

/** Delete older cache versions, and trim this one back to `STORE_CAP`. Once per
 *  page load: it walks every key, and none of it is urgent. */
let swept = false;
async function sweep(): Promise<void> {
  if (swept || typeof caches === "undefined") return;
  swept = true;
  try {
    for (const name of await caches.keys()) {
      if (name.startsWith("sb-thumbs-") && name !== STORE) {
        await caches.delete(name);
      }
    }
    const cache = await caches.open(STORE);
    const keys = await cache.keys();
    for (const old of keys.slice(0, Math.max(0, keys.length - STORE_CAP))) {
      await cache.delete(old);
    }
  } catch {
    // None of it is load-bearing.
  }
}
