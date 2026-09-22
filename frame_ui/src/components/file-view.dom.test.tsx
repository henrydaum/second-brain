/** @vitest-environment jsdom */

/**
 * What the file viewer has to get right that looking at it will not tell you.
 *
 * **The SVG sandbox.** An SVG handed to a framing element is a *document* at
 * this app's origin, and the production gateway credentials same-origin `/sdk`
 * calls — so a script inside one would be able to read and write the host
 * without ever learning the token. `sandbox=""` is what withholds scripting,
 * and `<embed>`, which is what this used to be, cannot express it at all. The
 * pane looks identical either way; only a hostile file tells them apart.
 *
 * **The Markdown renderer, for the same reason twice over.** Raw HTML in a note
 * is the same hole by another route, and a relative link that the browser
 * follows navigates the whole app away from the conversation. Both are quiet
 * failures — the first is invisible until it is exploited, the second until
 * somebody clicks — so both are pinned here.
 */

import "@testing-library/jest-dom/vitest";
import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { FileView } from "@/components/file-view";
import {
  MarkdownModePicker,
  setMarkdownMode,
} from "@/components/markdown-mode";
import { forgetFile } from "@/lib/files";

// Testing Library only registers its own cleanup when the test globals are
// exposed, and this project runs vitest without them — so two renders would
// otherwise both be in the document and the query would answer with the first.
afterEach(cleanup);

describe("an SVG in the file viewer", () => {
  // `.svg` is decided by extension in `kindOf`, before anything is asked of the
  // server, so this needs no stubbing to reach the branch under test.
  it("renders sandboxed, with scripting withheld", async () => {
    render(<FileView path="/tmp/diagram.svg" />);

    const frame = await screen.findByTitle("diagram.svg");
    expect(frame.tagName).toBe("IFRAME");
    // Present and empty. An absent attribute is no sandbox at all, and any
    // value containing `allow-scripts` hands the origin straight back.
    expect(frame).toHaveAttribute("sandbox", "");
  });

  it("keeps the box it always had", async () => {
    render(<FileView path="/tmp/diagram.svg" size="full" />);

    const frame = await screen.findByTitle("diagram.svg");
    expect(frame).toHaveClass("w-full", "rounded-lg", "border", "h-full");
  });
});

describe("a Markdown file in the file viewer", () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });
  afterEach(() => vi.unstubAllGlobals());

  /** One `/files` answer holding a whole file, and a path nothing else in this
   *  suite has read — `readText` caches by path and would otherwise hand the
   *  next test the previous test's note. */
  let served = 0;
  function serve(text: string): string {
    const bytes = new TextEncoder().encode(text);
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      headers: { get: () => null },
      arrayBuffer: async () => bytes.buffer,
    });
    const path = `/srv/vault/note-${served++}.md`;
    forgetFile(path);
    return path;
  }

  it("renders the note rather than printing its source", async () => {
    const path = serve("# Monday\n\nWrite it **down**.\n");
    render(<FileView path={path} />);

    await screen.findByRole("heading", { level: 1, name: "Monday" });
    expect(screen.getByText("down").tagName).toBe("STRONG");
  });

  it("shows front matter as metadata instead of as a giant heading", async () => {
    // Left to CommonMark, the closing fence turns `title: Weekly plan` into a
    // setext `<h2>` under a rule — the top of every vault note reading like a
    // bug. See `splitFrontMatter`.
    const path = serve("---\ntitle: Weekly plan\n---\n# Monday\n");
    render(<FileView path={path} />);

    await screen.findByRole("heading", { level: 1, name: "Monday" });
    expect(screen.getByText("Weekly plan")).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: /title:/ })).toBeNull();
  });

  it("drops raw HTML rather than letting a file into this origin", async () => {
    // The same hole `sandbox=""` closes for SVG, by another route: this app's
    // origin is one the gateway credentials, so markup out of a file must not
    // become markup in the page. `react-markdown` refuses it unless
    // `rehype-raw` is added, and this is what says never add it.
    const path = serve(
      'Hello <img src="x" onerror="alert(1)"> <script>alert(2)</script>\n',
    );
    const { container } = render(<FileView path={path} />);

    await screen.findByText(/Hello/);
    expect(container.querySelector("script")).toBeNull();
    expect(container.querySelector("img")).toBeNull();
  });

  it("fetches a relative image from the host, not from this app", async () => {
    const path = serve("![chart](attachments/chart.png)\n");
    const { container } = render(<FileView path={path} />);

    const image = await waitFor(() => {
      const found = container.querySelector("img");
      expect(found).not.toBeNull();
      return found as HTMLImageElement;
    });
    expect(image.getAttribute("src")).toContain(
      `path=${encodeURIComponent("/srv/vault/attachments/chart.png")}`,
    );
  });

  it("points a link to a neighbouring note at that note", async () => {
    const path = serve(
      "[the plan](notes/plan.md) and [a site](https://x.test)\n",
    );
    render(<FileView path={path} />);

    const neighbour = await screen.findByRole("link", { name: "the plan" });
    // A real `/files` URL, so a ⌘-click still means something — the click
    // handler is what keeps a plain one from navigating the app away.
    expect(neighbour.getAttribute("href")).toContain(
      `path=${encodeURIComponent("/srv/vault/notes/plan.md")}`,
    );
    expect(neighbour).not.toHaveAttribute("target");

    // Off-site goes off-site, and takes nothing with it.
    const offsite = screen.getByRole("link", { name: "a site" });
    expect(offsite).toHaveAttribute("href", "https://x.test");
    expect(offsite).toHaveAttribute("target", "_blank");
    expect(offsite.getAttribute("rel")).toContain("noreferrer");
  });

  it("follows the shared Preview/Source choice, for every file", async () => {
    // The highlighter splits a line into a span per token, so the source is
    // read off the block as a whole rather than looked up as a run of text.
    const sourceIn = (root: HTMLElement) =>
      root.querySelector("pre")?.textContent ?? "";

    const first = serve("# Monday\n");
    const one = render(<FileView path={first} />);
    await screen.findByRole("heading", { level: 1, name: "Monday" });

    // The control itself is three components away, in the viewer's footer —
    // this is the half that has to hear about it. See `markdown-mode.tsx`.
    act(() => setMarkdownMode("source"));
    expect(screen.queryByRole("heading", { name: "Monday" })).toBeNull();
    expect(sourceIn(one.container)).toContain("# Monday");

    // The choice is a mood rather than a per-file setting: it carries to the
    // next note, so wanting source does not mean asking for it all day.
    one.unmount();
    const second = serve("# Tuesday\n");
    const two = render(<FileView path={second} />);

    await waitFor(() => expect(sourceIn(two.container)).toContain("# Tuesday"));
    expect(screen.queryByRole("heading", { name: "Tuesday" })).toBeNull();

    // Put it back, so the default this suite started from is the one the next
    // test sees — the store outlives any single render.
    act(() => setMarkdownMode("preview"));
    await screen.findByRole("heading", { level: 1, name: "Tuesday" });
  });
});

/**
 * The control that drives the view above.
 *
 * It lives in the viewer dialog's footer rather than over the file, so that it
 * costs no vertical space — which is precisely why it cannot be tested through
 * `FileView`, and why it is worth pinning that pressing it moves the shared
 * state the view reads.
 */
describe("the Preview/Source picker", () => {
  afterEach(() => act(() => setMarkdownMode("preview")));

  it("reports which state is on, and switches to the other", async () => {
    const user = userEvent.setup();
    render(<MarkdownModePicker />);

    expect(screen.getByRole("button", { name: "Preview" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );

    await user.click(screen.getByRole("button", { name: "Source" }));
    expect(screen.getByRole("button", { name: "Source" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(screen.getByRole("button", { name: "Preview" })).toHaveAttribute(
      "aria-pressed",
      "false",
    );
  });
});

describe("a file only a parser can read", () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });
  afterEach(() => vi.unstubAllGlobals());

  /** Answers `/sdk/<type>` the way the HTTP frontend does: `{data}` on 200,
   *  `{error, code}` otherwise. */
  function answer(
    parseFile: { status: number; body: unknown },
    route: Record<string, unknown> = {
      modality: "text",
      known: true,
      generic: false,
    },
  ) {
    fetchMock.mockImplementation(async (url: URL | string) => {
      const type = String(url).split("/sdk/")[1]?.split("?")[0];
      const reply =
        type === "parse.modality"
          ? { status: 200, body: { data: route } }
          : parseFile;
      return {
        ok: reply.status < 300,
        status: reply.status,
        json: async () => reply.body,
      };
    });
  }

  it("shows the kernel's extraction, with the original a click away", async () => {
    answer({ status: 200, body: { data: "Quarterly plan\n\nShip it." } });
    render(<FileView path="/srv/docs/plan.gdoc" />);

    expect(await screen.findByText(/Quarterly plan/)).toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: "download the original" }),
    ).toHaveAttribute("download", "plan.gdoc");
    // Never read as bytes: a `.gdoc` on disk is a JSON stub, not the document.
    expect(
      fetchMock.mock.calls.some(([url]) => String(url).includes("/files")),
    ).toBe(false);
  });

  it("says a protected file was refused rather than that parsing broke", async () => {
    answer({
      status: 403,
      body: { error: "not readable", code: "not_permitted" },
    });
    render(<FileView path="/srv/docs/secret.gdoc" />);

    expect(
      await screen.findByText("Second Brain would not hand this file over."),
    ).toBeInTheDocument();
  });
});

describe("a tabular file or an archive, read by the kernel", () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });
  afterEach(() => vi.unstubAllGlobals());

  function answer(route: Record<string, unknown>, data: unknown) {
    fetchMock.mockImplementation(
      async (url: URL | string, init?: RequestInit) => {
        const type = String(url).split("/sdk/")[1]?.split("?")[0];
        const body =
          type === "parse.modality"
            ? { data: route }
            : { data, asked: JSON.parse(String(init?.body ?? "{}")) };
        return { ok: true, status: 200, json: async () => body };
      },
    );
  }

  it("draws each sheet of a workbook as a table, one at a time", async () => {
    answer(
      {
        modality: "tabular",
        known: true,
        generic: false,
        modalities: ["tabular", "text"],
      },
      [
        "=== Budget (2 rows) ===",
        "item,cost",
        "rent,900",
        "food,300",
        "",
        "=== Notes (1 row) ===",
        "note",
        "pay early",
      ].join("\n"),
    );
    render(<FileView path="/srv/money.xlsx" />);

    expect(await screen.findByText("rent")).toBeInTheDocument();
    expect(
      screen.getByRole("columnheader", { name: "cost" }),
    ).toBeInTheDocument();
    expect(screen.queryByText("pay early")).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole("tab", { name: "Notes" }));
    expect(await screen.findByText("pay early")).toBeInTheDocument();
    expect(screen.queryByText("rent")).not.toBeInTheDocument();
  });

  it("lists what an archive holds, asking for the container route", async () => {
    answer(
      {
        modality: "container",
        known: true,
        generic: false,
        modalities: ["container"],
      },
      ["/tmp/x/readme.txt", "/tmp/x/photo.png"],
    );
    render(<FileView path="/srv/bundle.zip" />);

    expect(await screen.findByText("readme.txt")).toBeInTheDocument();
    expect(screen.getByText("photo.png")).toBeInTheDocument();
    const parseCall = fetchMock.mock.calls.find(([url]) =>
      String(url).includes("/sdk/parse.file"),
    );
    expect(JSON.parse(String(parseCall?.[1]?.body))).toMatchObject({
      modality: "container",
    });
  });
});
