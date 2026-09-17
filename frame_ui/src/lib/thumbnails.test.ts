import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// Same stub as `files.test.ts`, and for the same reason: `fileUrl` reaches for
// `window.location`, and nothing under test here wants a DOM.
vi.mock("@/lib/client", () => ({
  sdk: vi.fn(),
  authHeaders: () => ({ Authorization: "Bearer test" }),
  fileUrl: (path: string) =>
    `http://host/files?path=${encodeURIComponent(path)}`,
}));

const { forgetThumbnail, forgetThumbnails, thumbnail, thumbnailsWork } =
  await import("@/lib/thumbnails");

/* ── A browser, in the shape this module asks for one ────────────────── */

/** Reads that have been served, in order. The point of most of these tests is
 *  how *short* this list is. */
let reads: string[] = [];
/** Decodes that were started but have not finished, so a test can hold one
 *  open and see what the queue does behind it. */
let decoding: Array<() => void> = [];

function serve(bytes: number) {
  return vi.fn(async (input: string) => {
    reads.push(new URL(input).searchParams.get("path") ?? "");
    return {
      ok: true,
      status: 200,
      headers: { get: () => null },
      arrayBuffer: async () => new ArrayBuffer(bytes),
    };
  });
}

/** A decode that resolves when the test says so. `close` is asserted on: the
 *  full-size bitmap not outliving the draw is the whole memory story. */
function bitmaps() {
  return vi.fn(async () => {
    await new Promise<void>((go) => decoding.push(go));
    return { width: 4000, height: 3000, close: vi.fn() };
  });
}

class FakeCanvas {
  drew = 0;
  width: number;
  height: number;
  constructor(width: number, height: number) {
    this.width = width;
    this.height = height;
  }
  getContext() {
    return {
      imageSmoothingEnabled: false,
      imageSmoothingQuality: "low",
      drawImage: (..._args: unknown[]) => {
        this.drew += 1;
      },
    };
  }
  async convertToBlob() {
    return new Blob([new Uint8Array([1, 2, 3])], { type: "image/webp" });
  }
}

class FakeReader {
  result: string | null = null;
  error: unknown = null;
  onload: (() => void) | null = null;
  onerror: (() => void) | null = null;
  readAsDataURL(blob: Blob) {
    void blob;
    this.result = "data:image/webp;base64,AQID";
    queueMicrotask(() => this.onload?.());
  }
}

beforeEach(() => {
  reads = [];
  decoding = [];
  forgetThumbnails();
  vi.stubGlobal("fetch", serve(64));
  vi.stubGlobal("createImageBitmap", bitmaps());
  vi.stubGlobal("OffscreenCanvas", FakeCanvas);
  vi.stubGlobal("FileReader", FakeReader);
  vi.stubGlobal("document", {});
  // Left undefined on purpose: the persistent half is optional, and every path
  // through it has to survive a browser that does not offer it.
  vi.stubGlobal("caches", undefined);
});

afterEach(async () => {
  // A decode left hanging holds a slot in the queue, and the queue outlives
  // the test — one stuck decode would starve every test after it.
  await finish();
  vi.unstubAllGlobals();
});

/** Let everything that is not blocked on a decode settle. */
const settle = () => new Promise<void>((done) => setTimeout(done, 0));

/** Finish every decode that is waiting, and any that start behind them.
 *  Settles first: the work reaches the decode through a fetch, so there is
 *  nothing to release on the turn it was asked for. */
async function finish() {
  for (let pass = 0; pass < 6; pass += 1) {
    await settle();
    while (decoding.length) decoding.shift()?.();
  }
}

describe("thumbnailsWork", () => {
  it("says no without a decoder, which is what makes the fallback safe", () => {
    vi.stubGlobal("createImageBitmap", undefined);
    expect(thumbnailsWork()).toBe(false);
  });

  it("says yes with one", () => {
    expect(thumbnailsWork()).toBe(true);
  });
});

describe("thumbnail", () => {
  it("reads a file once however many times it is asked for", async () => {
    const first = thumbnail("/srv/a.jpg", 100);
    const second = thumbnail("/srv/a.jpg", 100);
    await finish();

    expect(await first).toBe("data:image/webp;base64,AQID");
    expect(await second).toBe(await first);
    expect(reads).toEqual(["/srv/a.jpg"]);

    // The mount after the panel was closed and opened again: still no read.
    expect(await thumbnail("/srv/a.jpg", 100)).toBe(await first);
    expect(reads).toHaveLength(1);
  });

  it("treats a changed file as a different picture", async () => {
    void thumbnail("/srv/a.jpg", 100);
    await finish();
    void thumbnail("/srv/a.jpg", 200);
    await finish();

    expect(reads).toEqual(["/srv/a.jpg", "/srv/a.jpg"]);
  });

  it("closes the full-size bitmap once it has been drawn", async () => {
    const made = vi.mocked(createImageBitmap);
    void thumbnail("/srv/a.jpg", 100);
    await finish();

    const bitmap = await made.mock.results[0].value;
    expect(bitmap.close).toHaveBeenCalled();
  });

  it("decodes two at a time and no more", async () => {
    for (const name of ["a", "b", "c", "d"]) {
      void thumbnail(`/srv/${name}.jpg`, 1);
    }
    await settle();

    expect(decoding).toHaveLength(2);
    expect(reads).toEqual(["/srv/a.jpg", "/srv/b.jpg"]);
    await finish();
    expect(reads).toHaveLength(4);
  });

  it("serves the newest waiting request first", async () => {
    for (const name of ["a", "b", "c", "d"]) {
      void thumbnail(`/srv/${name}.jpg`, 1);
    }
    await settle();

    // `a` and `b` are in flight; `c` and `d` are waiting. Finishing one lets
    // the *last* one asked for through — the row you just scrolled to, not the
    // row you scrolled past.
    decoding.shift()?.();
    await settle();
    expect(reads).toEqual(["/srv/a.jpg", "/srv/b.jpg", "/srv/d.jpg"]);
    await finish();
  });

  it("remembers a file it could not read, so a reopen does not ask again", async () => {
    vi.stubGlobal("fetch", async () => {
      reads.push("gone");
      return { ok: false, status: 404, headers: { get: () => null } };
    });

    await expect(thumbnail("/srv/gone.jpg", 1)).rejects.toThrow();
    await expect(thumbnail("/srv/gone.jpg", 1)).rejects.toThrow();
    expect(reads).toEqual(["gone"]);
  });

  it("lets a failure be retried once the file is written again", async () => {
    vi.stubGlobal("fetch", async () => {
      reads.push("gone");
      return { ok: false, status: 404, headers: { get: () => null } };
    });
    await expect(thumbnail("/srv/gone.jpg", 1)).rejects.toThrow();

    forgetThumbnail("/srv/gone.jpg");
    vi.stubGlobal("fetch", serve(64));
    const again = thumbnail("/srv/gone.jpg", 1);
    await finish();

    expect(await again).toBe("data:image/webp;base64,AQID");
  });

  it("drops every version of a path that has changed", async () => {
    void thumbnail("/srv/a.jpg", 100);
    void thumbnail("/srv/a.jpg", 200);
    void thumbnail("/srv/kept.jpg", 100);
    await finish();
    reads = [];

    forgetThumbnail("/srv/a.jpg");
    void thumbnail("/srv/a.jpg", 100);
    void thumbnail("/srv/kept.jpg", 100);
    await finish();

    expect(reads).toEqual(["/srv/a.jpg"]);
  });
});
