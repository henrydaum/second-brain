import { beforeEach, describe, expect, it, vi } from "vitest";

const sdk = vi.fn();

vi.mock("@/lib/client", () => ({
  sdk: (...args: unknown[]) => sdk(...args),
}));

const { MAX_UPLOAD_BYTES, attachmentSubmitArgs, uploadToHost } = await import(
  "@/lib/upload"
);

beforeEach(() => {
  sdk.mockReset();
  sdk.mockImplementation(async (request: string) =>
    request === "fs.temp" ? "/tmp/upload.png" : true,
  );
});

/**
 * Enough of a `File` to be read the way the uploader reads one.
 *
 * `size` and `slice` rather than `arrayBuffer`, deliberately: the uploader is
 * only allowed to touch a window at a time, so a stand-in that answers the
 * whole file at once would let a regression back to `file.arrayBuffer()` pass
 * this suite unnoticed. `declared` exists for the size check, which has to
 * happen before anything is read at all.
 */
function fakeFile(
  bytes: Uint8Array,
  name = "photo.png",
  declared = bytes.length,
): File {
  return {
    name,
    size: declared,
    slice: (start: number, end: number) => ({
      arrayBuffer: async () =>
        bytes.slice(start, Math.min(end, bytes.length)).buffer,
    }),
  } as unknown as File;
}

describe("browser attachment uploads", () => {
  it("submits several files as one atomic frontend message", () => {
    expect(
      attachmentSubmitArgs(
        [
          { path: "/tmp/chart.png", name: "Chart.png" },
          { path: "/tmp/notes.pdf", name: "Notes.pdf" },
        ],
        "Compare these",
      ),
    ).toEqual({
      input_kind: "attachment",
      files: [
        {
          path: "/tmp/chart.png",
          file_name: "Chart.png",
          extension: "png",
        },
        {
          path: "/tmp/notes.pdf",
          file_name: "Notes.pdf",
          extension: "pdf",
        },
      ],
      caption: "Compare these",
      ingest: true,
    });
  });

  it("keeps base64 write requests below a four-megabyte gateway limit", async () => {
    const bytes = new Uint8Array(2 * 1024 * 1024 + 1);

    const progress: number[] = [];
    const upload = uploadToHost(fakeFile(bytes));
    let step = await upload.next();
    while (!step.done) {
      progress.push(step.value);
      step = await upload.next();
    }

    const writes = sdk.mock.calls.filter(
      ([request]) => request === "fs.write_bytes",
    );
    expect(writes).toHaveLength(2);
    expect(writes[0][1].data.length).toBeLessThan(4 * 1024 * 1024);
    expect(writes.map(([, args]) => args.mode)).toEqual([
      "overwrite",
      "append",
    ]);
    expect(progress.at(-1)).toBe(1);
    expect(step.value).toBe("/tmp/upload.png");
  });

  it("still creates a file with no bytes in it", async () => {
    const upload = uploadToHost(fakeFile(new Uint8Array(0), "empty.txt"));
    let step = await upload.next();
    while (!step.done) step = await upload.next();

    expect(
      sdk.mock.calls.filter(([request]) => request === "fs.write_bytes"),
    ).toHaveLength(1);
  });

  /**
   * The refusal has to arrive *through* the generator rather than before it, so
   * the composer has already drawn the chip that is about to report it. An
   * async generator body does not run until the first `next()`, which is what
   * makes that true — pinned here because moving the check to the caller would
   * quietly turn an explained failure back into a click that did nothing.
   */
  it("refuses a file past the limit, once it is being driven", async () => {
    const upload = uploadToHost(
      fakeFile(new Uint8Array(0), "film.mov", MAX_UPLOAD_BYTES + 1),
    );

    await expect(upload.next()).rejects.toThrow(/100 MB/);
    expect(sdk).not.toHaveBeenCalled();
  });

  it("never reads the whole file into memory", async () => {
    const bytes = new Uint8Array(5 * 1024 * 1024);
    const file = fakeFile(bytes);
    const whole = vi.fn();
    (file as unknown as { arrayBuffer: unknown }).arrayBuffer = whole;

    const upload = uploadToHost(file);
    let step = await upload.next();
    while (!step.done) step = await upload.next();

    expect(whole).not.toHaveBeenCalled();
  });
});
