import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { recallPlace, rememberPlace } from "@/lib/scroll-memory";

const sdk = vi.fn();

// The whole client module is stubbed rather than just `sdk`: `fileUrl` and
// `serverUrl` reach for `window.location`, which this suite deliberately runs
// without — none of what is under test here needs a DOM.
vi.mock("@/lib/client", () => ({
  sdk: (type: string, args: Record<string, unknown>) => sdk(type, args),
  authHeaders: () => ({ Authorization: "Bearer test" }),
  fileUrl: (path: string) => `http://host/files?path=${encodeURIComponent(path)}`,
}));

const {
  describeStatus,
  dirOf,
  fetchWhole,
  fetchWholeBytes,
  FileUnavailable,
  forgetFile,
  formatBytes,
  guessIconKind,
  kindOf,
  nameOf,
  readText,
  resolveAgainst,
  suffixOf,
} = await import("@/lib/files");

describe("guessIconKind", () => {
  it.each([
    ["Demo.mp4", "video"],
    ["results.csv", "table"],
    ["analysis.PY", "code"],
    ["component.tsx", "code"],
    ["notes.md", "markdown"],
  ])("classifies %s as %s", (path, kind) => {
    expect(guessIconKind(path)).toBe(kind);
  });
});

/** One `/files` answer. `total` present means a `206` with a `Content-Range`,
 *  which is what a file too big for one wire message looks like. */
function reply(body: string, status = 200, total?: number) {
  const bytes = new TextEncoder().encode(body);
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: {
      get: (name: string) =>
        name === "Content-Range" && total !== undefined
          ? `bytes 0-${bytes.length - 1}/${total}`
          : null,
    },
    arrayBuffer: async () => bytes.buffer,
  };
}

describe("naming", () => {
  it("takes the last segment, whichever separator was used", () => {
    expect(nameOf("/srv/app/notes.md")).toBe("notes.md");
    expect(nameOf("C:\\Users\\henry\\notes.md")).toBe("notes.md");
    expect(nameOf("notes.md")).toBe("notes.md");
  });

  it("gives the parent directory, or nothing at the root", () => {
    expect(dirOf("/srv/app/notes.md")).toBe("/srv/app");
    expect(dirOf("/notes.md")).toBe("");
    expect(dirOf("notes.md")).toBe("");
  });

  it("keeps the dot on the extension and lowercases it", () => {
    // `.` is where `parse.modality` expects it, unlike `extensionOf` in
    // `lib/upload.ts` — see the note on `suffixOf`.
    expect(suffixOf("/srv/CHART.PNG")).toBe(".png");
    expect(suffixOf("/srv/Makefile")).toBe("");
    expect(suffixOf("/srv/.gitignore")).toBe("");
  });
});

describe("resolveAgainst", () => {
  it("joins a relative path to the directory it was written in", () => {
    expect(resolveAgainst("/srv/vault", "attachments/chart.png")).toBe(
      "/srv/vault/attachments/chart.png",
    );
    expect(resolveAgainst("/srv/vault/daily", "../plan.md")).toBe(
      "/srv/vault/plan.md",
    );
    expect(resolveAgainst("/srv/vault", "./here.md")).toBe("/srv/vault/here.md");
  });

  it("leaves an already-absolute path alone", () => {
    expect(resolveAgainst("/srv/vault", "/etc/hosts")).toBe("/etc/hosts");
    expect(resolveAgainst("/srv/vault", "C:\\notes\\plan.md")).toBe(
      "C:\\notes\\plan.md",
    );
  });

  it("keeps the separator the host used", () => {
    expect(resolveAgainst("C:\\Users\\henry", "notes\\plan.md")).toBe(
      "C:\\Users\\henry\\notes\\plan.md",
    );
  });

  it("stops at the root rather than walking off the top of it", () => {
    // A note with one `../` too many should resolve to somewhere merely wrong,
    // not to something that reads as a different volume.
    expect(resolveAgainst("/srv", "../../../etc/hosts")).toBe("/etc/hosts");
    expect(resolveAgainst("C:\\", "..\\..\\x.md")).toBe("C:\\x.md");
  });
});

describe("formatBytes", () => {
  it("scales into readable units", () => {
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(2048)).toBe("2.0 KB");
    expect(formatBytes(14_400_000)).toBe("13.7 MB");
  });
});

describe("kindOf", () => {
  // A block body, not `() => sdk.mockReset()`: `mockReset` answers with the
  // mock itself, and a function returned from `beforeEach` is taken as a
  // teardown and *called* — which would invoke `sdk` after each test, with
  // whatever implementation it was left with.
  beforeEach(() => {
    sdk.mockReset();
  });

  it("shows a .csv as a table even though modality calls it text", () => {
    // `parse_text` claims .csv and the first registration wins, so this answer
    // is permanent — branching on modality alone would render a spreadsheet as
    // a wall of commas forever.
    sdk.mockResolvedValue("text");
    return expect(kindOf("/srv/report.csv")).resolves.toBe("table");
  });

  it("renders a .md even though modality calls it text", async () => {
    // Same disagreement as .csv, and the same resolution: "text" is the right
    // answer to how the model should ingest a note and the wrong one for how a
    // person should read it.
    sdk.mockResolvedValue("text");
    await expect(kindOf("/srv/vault/plan.md")).resolves.toBe("markdown");
    await expect(kindOf("/srv/vault/README.markdown")).resolves.toBe("markdown");
    // Never even asked: the extension decided it.
    expect(sdk).not.toHaveBeenCalled();

    // `.mdx` is deliberately not in that set — it is source with imports and
    // JSX in it, and rendering the JSX away would hide most of the file.
    await expect(kindOf("/srv/vault/post.mdx")).resolves.toBe("text");
  });

  it("embeds a .pdf even though modality calls it unknown", async () => {
    sdk.mockResolvedValue("unknown");
    await expect(kindOf("/srv/paper.pdf")).resolves.toBe("embed");
    // Never even asked: the extension decided it.
    expect(sdk).not.toHaveBeenCalled();
  });

  it("takes image, video, audio and text at their word", async () => {
    sdk.mockResolvedValueOnce("image");
    await expect(kindOf("/srv/a.png")).resolves.toBe("image");
    sdk.mockResolvedValueOnce("video");
    await expect(kindOf("/srv/a.mp4")).resolves.toBe("video");
    sdk.mockResolvedValueOnce("audio");
    await expect(kindOf("/srv/a.wav")).resolves.toBe("audio");
    sdk.mockResolvedValueOnce("text");
    await expect(kindOf("/srv/a.log")).resolves.toBe("text");
  });

  it("offers anything else as a download", async () => {
    sdk.mockResolvedValue("unknown");
    await expect(kindOf("/srv/model.gguf")).resolves.toBe("download");
  });

  it("asks once per extension, however many files share it", async () => {
    sdk.mockResolvedValue("image");
    await Promise.all([
      kindOf("/srv/one.jxl"),
      kindOf("/srv/two.jxl"),
      kindOf("/srv/three.jxl"),
    ]);
    expect(sdk).toHaveBeenCalledTimes(1);
  });

  it("still draws a known image when the Request fails", async () => {
    // image/audio/video come from a static kernel-side map, so a dropped
    // connection is no reason to demote a PNG to a download link.
    // `mockImplementation` rather than `mockRejectedValue`: the latter builds
    // its rejected promise before anything is there to catch it.
    sdk.mockImplementation(() => Promise.reject(new Error("offline")));
    await expect(kindOf("/srv/photo.heic")).resolves.toBe("image");
    await expect(kindOf("/srv/thing.qqq")).resolves.toBe("download");
  });
});

describe("fetchWhole", () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });
  afterEach(() => vi.unstubAllGlobals());

  it("takes a 200 as the whole file", async () => {
    fetchMock.mockResolvedValueOnce(reply("a,b\n1,2"));
    const whole = await fetchWhole("/srv/small.csv");
    expect(whole).toEqual({ text: "a,b\n1,2", truncated: false, total: 7 });
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("follows the windows of a 206 that nobody asked for", async () => {
    // The trap this function exists for: a large file answers 206 with the
    // first window even with no Range header sent. One fetch would silently
    // return a third of the file.
    fetchMock
      .mockResolvedValueOnce(reply("aaaa", 206, 12))
      .mockResolvedValueOnce(reply("bbbb", 206, 12))
      .mockResolvedValueOnce(reply("cccc", 206, 12));

    const whole = await fetchWhole("/srv/big.csv");
    expect(whole.text).toBe("aaaabbbbcccc");
    expect(whole.truncated).toBe(false);
    expect(whole.total).toBe(12);
    expect(fetchMock).toHaveBeenCalledTimes(3);

    // The first pass sends no Range; the rest ask for the remainder.
    const ranges = fetchMock.mock.calls.map(
      ([, init]) => (init.headers as Record<string, string>).Range,
    );
    expect(ranges).toEqual([undefined, "bytes=4-", "bytes=8-"]);
  });

  it("preserves arbitrary bytes for binary viewers", async () => {
    const bytes = new Uint8Array([0, 255, 37, 80, 68, 70]);
    const response = { ...reply("", 200) };
    response.arrayBuffer = async () => bytes.buffer;
    fetchMock.mockResolvedValueOnce(response);

    await expect(fetchWholeBytes("/srv/paper.pdf")).resolves.toEqual({
      bytes,
      truncated: false,
      total: bytes.length,
    });
  });

  it("stops at the cap and says it was truncated", async () => {
    fetchMock.mockResolvedValue(reply("aaaa", 206, 1000));
    const whole = await fetchWhole("/srv/huge.log", 8);
    expect(whole.text).toBe("aaaaaaaa");
    expect(whole.truncated).toBe(true);
    expect(whole.total).toBe(1000);
  });

  it("gives up rather than spinning when a window comes back empty", async () => {
    fetchMock.mockResolvedValue(reply("", 206, 1000));
    const whole = await fetchWhole("/srv/stuck.log");
    expect(whole.text).toBe("");
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("throws the status, so the pane can say which failure this was", async () => {
    fetchMock.mockResolvedValueOnce(reply("", 404));
    await expect(fetchWhole("/srv/gone.md")).rejects.toMatchObject({
      name: "FileUnavailable",
      status: 404,
    });
    await expect(fetchWhole("/srv/gone.md")).rejects.toBeInstanceOf(Error);
  });

  it("decodes multi-byte characters split across a window boundary", async () => {
    // "é" is two bytes; decoding each window separately would produce U+FFFD.
    const bytes = new TextEncoder().encode("café");
    const head = { ...reply("", 206, bytes.length) };
    head.arrayBuffer = async () => bytes.slice(0, 4).buffer;
    const tail = { ...reply("", 206, bytes.length) };
    tail.arrayBuffer = async () => bytes.slice(4).buffer;
    fetchMock.mockResolvedValueOnce(head).mockResolvedValueOnce(tail);

    await expect(fetchWhole("/srv/x.txt")).resolves.toMatchObject({
      text: "café",
    });
  });
});

describe("readText", () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });
  afterEach(() => vi.unstubAllGlobals());

  it("reads a file once, however often it is reopened", async () => {
    fetchMock.mockResolvedValue(reply("a,b\n1,2"));
    const a = await readText("/srv/cached.csv");
    const b = await readText("/srv/cached.csv");
    expect(b).toBe(a);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("reads it again once the ledger says it changed", async () => {
    fetchMock.mockResolvedValue(reply("before"));
    await readText("/srv/notes.md");
    forgetFile("/srv/notes.md");
    fetchMock.mockResolvedValue(reply("after"));
    await expect(readText("/srv/notes.md")).resolves.toMatchObject({
      text: "after",
    });
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("drops where you were in it too, for the same reason", async () => {
    // The bytes and the scroll offset go stale on the same event. An offset
    // into a document that has since been rewritten points at whatever happens
    // to be there now, which is worse than starting at the top.
    fetchMock.mockResolvedValue(reply("before"));
    await readText("/srv/moving.md");
    rememberPlace("/srv/moving.md", "full:preview", 640);

    forgetFile("/srv/moving.md");

    expect(recallPlace("/srv/moving.md", "full:preview")).toBe(0);
  });

  it("never caches a failure", async () => {
    // A file that is missing now may be back in a moment, and the retry is one
    // request — where a cached rejection would be permanent for the page.
    fetchMock.mockResolvedValueOnce(reply("", 404));
    await expect(readText("/srv/flaky.md")).rejects.toMatchObject({
      status: 404,
    });
    fetchMock.mockResolvedValueOnce(reply("back"));
    await expect(readText("/srv/flaky.md")).resolves.toMatchObject({
      text: "back",
    });
  });

  it("keeps only a handful, oldest out first", async () => {
    fetchMock.mockResolvedValue(reply("x"));
    for (let i = 0; i < 9; i++) await readText(`/srv/f${i}.md`);
    // Nine files through a cache of eight: the first is gone, the last is not.
    fetchMock.mockClear();
    await readText("/srv/f8.md");
    expect(fetchMock).not.toHaveBeenCalled();
    await readText("/srv/f0.md");
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});

describe("describeStatus", () => {
  it("names the two failures that mean different things", () => {
    expect(describeStatus(404)).toMatch(/gone/i);
    expect(describeStatus(403)).toMatch(/would not hand/i);
  });

  it("still says something for a status it does not know", () => {
    expect(describeStatus(500)).toContain("500");
  });

  it("is what a FileUnavailable carries as its message", () => {
    expect(new FileUnavailable(404).message).toBe(describeStatus(404));
  });
});
