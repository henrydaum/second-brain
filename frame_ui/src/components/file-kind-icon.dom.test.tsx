/** @vitest-environment jsdom */

import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const thumbnail = vi.fn();
const thumbnailsWork = vi.fn(() => true);

vi.mock("@/lib/thumbnails", () => ({
  thumbnail: (path: string, version?: string | number) =>
    thumbnail(path, version),
  thumbnailsWork: () => thumbnailsWork(),
}));

vi.mock("@/lib/client", () => ({
  fileUrl: (path: string) =>
    `http://host/files?path=${encodeURIComponent(path)}`,
}));

const { FileThumbnail } = await import("@/components/file-kind-icon");

/** Every tile that has been observed, so a test can decide when one is looked
 *  at. jsdom has no layout and therefore no real `IntersectionObserver`. */
let seen: Array<() => void> = [];

class ObserverStub {
  ping: (entries: { isIntersecting: boolean }[]) => void;
  constructor(ping: (entries: { isIntersecting: boolean }[]) => void) {
    this.ping = ping;
  }
  observe() {
    seen.push(() => this.ping([{ isIntersecting: true }]));
  }
  unobserve() {}
  disconnect() {}
}

beforeEach(() => {
  seen = [];
  thumbnail.mockReset();
  thumbnailsWork.mockReturnValue(true);
  vi.stubGlobal("IntersectionObserver", ObserverStub);
});

// Nothing configures a global teardown in this project, so a rendered tree
// would otherwise still be in the document for the next test to find.
afterEach(cleanup);

describe("FileThumbnail", () => {
  it("asks for nothing until the tile is on screen", async () => {
    thumbnail.mockResolvedValue("data:image/webp;base64,AQID");
    render(<FileThumbnail name="/srv/photo.jpg" path="/srv/photo.jpg" />);

    // The whole point: a drawer of three hundred rows costs three hundred
    // observers and no reads.
    expect(thumbnail).not.toHaveBeenCalled();
    expect(document.querySelector("img")).toBeNull();

    seen.forEach((look) => look());
    await waitFor(() =>
      expect(screen.getByRole("presentation", { hidden: true })).toHaveAttribute(
        "src",
        "data:image/webp;base64,AQID",
      ),
    );
    expect(thumbnail).toHaveBeenCalledTimes(1);
  });

  it("never asks for a file that is gone, whatever it is called", () => {
    render(<FileThumbnail name="/srv/photo.jpg" />);

    seen.forEach((look) => look());
    expect(thumbnail).not.toHaveBeenCalled();
    expect(document.querySelector("img")).toBeNull();
  });

  it("never asks for something that is not a picture", () => {
    render(<FileThumbnail name="/srv/notes.md" path="/srv/notes.md" />);

    seen.forEach((look) => look());
    expect(thumbnail).not.toHaveBeenCalled();
  });

  it("falls back to the original file where nothing can make a thumbnail", async () => {
    thumbnailsWork.mockReturnValue(false);
    render(<FileThumbnail name="/srv/photo.jpg" path="/srv/photo.jpg" />);

    await waitFor(() =>
      expect(screen.getByRole("presentation", { hidden: true })).toHaveAttribute(
        "src",
        "http://host/files?path=%2Fsrv%2Fphoto.jpg",
      ),
    );
    expect(thumbnail).not.toHaveBeenCalled();
  });

  it("keeps its icon and draws nothing when there is no picture to be had", async () => {
    thumbnail.mockRejectedValue(new Error("404"));
    render(<FileThumbnail name="/srv/photo.jpg" path="/srv/photo.jpg" />);

    seen.forEach((look) => look());
    await waitFor(() => expect(thumbnail).toHaveBeenCalled());
    expect(document.querySelector("img")).toBeNull();
  });

  it("drops the old square when the file underneath it changes", async () => {
    thumbnail.mockResolvedValue("data:image/webp;base64,AQID");
    const { rerender } = render(
      <FileThumbnail name="/srv/photo.jpg" path="/srv/photo.jpg" version={1} />,
    );
    seen.forEach((look) => look());
    await waitFor(() => expect(document.querySelector("img")).not.toBeNull());

    // The agent wrote over it. Showing the picture of what used to be there is
    // worse than showing the icon for a moment.
    seen = [];
    thumbnail.mockReturnValue(new Promise(() => {}));
    rerender(
      <FileThumbnail name="/srv/photo.jpg" path="/srv/photo.jpg" version={2} />,
    );
    expect(document.querySelector("img")).toBeNull();

    seen.forEach((look) => look());
    await waitFor(() => expect(thumbnail).toHaveBeenLastCalledWith("/srv/photo.jpg", 2));
  });
});
