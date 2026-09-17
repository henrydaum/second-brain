/**
 * Getting a file from the browser onto the host.
 *
 * **There is no upload endpoint.** `frontend.submit` with
 * `input_kind: "attachment"` takes a *path on the host* — not bytes, not a URL —
 * which is fine for a Python frontend that downloaded the file itself and
 * useless to a browser holding a `File`. So the path is built out of ordinary
 * Requests: ask for a scratch path, write the bytes into it a window at a time,
 * then submit the path.
 *
 * `ingest: true` on the submit is the part worth not losing. It moves the file
 * into the attachment cache — a watched directory — so the extraction and
 * indexing pipeline treats it like any other incoming file. Leaving it in temp
 * skips all of that, and the agent gets a path it can read rather than a
 * document the system knows about.
 */

import { sdk } from "@/lib/client";

/**
 * How much of the file goes in one Request.
 *
 * One answer has to fit in one wire message, and base64 costs a third on top of
 * the raw bytes. Two megabytes stays below gateways with a 4 MB request-body
 * limit after base64 and the JSON envelope are added. Smaller than it could be
 * on a direct connection, deliberately: remote gateways vary, the progress bar
 * moves more often, and a retry costs less.
 */
const CHUNK_BYTES = 2 * 1024 * 1024;

/**
 * The largest file this composer will take.
 *
 * **A limit is not a policy decision here, it is the honest edge of what the
 * route can do.** Every byte crosses as base64 inside a JSON Request, so the
 * transfer is a third larger than the file and takes a round trip per two
 * megabytes; a phone on a tunnelled connection is the client that finds the
 * ceiling first. Without a stated one the failure was a tab that stopped
 * responding and then died, with nothing said about why.
 *
 * A hundred megabytes covers a long voice note, any document, and video worth
 * sending to a model. It is checked before the first byte is read, so an
 * oversized file costs nothing but the sentence explaining it.
 */
export const MAX_UPLOAD_BYTES = 100 * 1024 * 1024;

const megabytes = (bytes: number) => Math.round(bytes / (1024 * 1024));

/** `btoa` takes a string, so the bytes have to become one first. Done in small
 *  slices because `String.fromCharCode(...bytes)` with a multi-megabyte spread
 *  overflows the call stack. */
function base64(bytes: Uint8Array): string {
  let binary = "";
  const STRIDE = 8192;
  for (let i = 0; i < bytes.length; i += STRIDE) {
    binary += String.fromCharCode(...bytes.subarray(i, i + STRIDE));
  }
  return btoa(binary);
}

/** The extension, with its dot, or "" — `fs.temp` wants a suffix so the file it
 *  makes is recognisable to whatever opens it later. */
function suffixOf(name: string): string {
  const dot = name.lastIndexOf(".");
  return dot > 0 ? name.slice(dot) : "";
}

/**
 * The extension without its dot, which is what `frontend.submit` calls
 * `extension`.
 *
 * The kernel would work this out from the file name on its own, but the
 * extension is what decides the file's *modality* — whether it is offered to an
 * image parser, an audio parser, or none — and which extensions the current
 * model will accept at all. Saying it explicitly, as the Telegram frontend
 * does, keeps that decision on something we chose rather than on how a path
 * happened to be spelled.
 */
export function extensionOf(name: string): string {
  return suffixOf(name).replace(/^\./, "");
}

type UploadedAttachment = { path: string; name: string };

/**
 * One composer message, however many files it carries.
 *
 * `send_attachment` hands priority to the agent, so several sequential
 * Requests can never represent one multi-file message: after the first, the
 * rest arrive on the agent's turn. The HTTP frontend's `files` form is the
 * atomic spelling and deliberately works for a one-file list too.
 */
export function attachmentSubmitArgs(
  files: UploadedAttachment[],
  caption: string,
) {
  return {
    input_kind: "attachment" as const,
    files: files.map((file) => ({
      path: file.path,
      file_name: file.name,
      extension: extensionOf(file.name),
    })),
    caption,
    // Into the watched attachment cache, so every file is extracted and
    // indexed rather than left in scratch.
    ingest: true,
  };
}

/**
 * Write a file to host scratch and answer with its path.
 *
 * **A generator, so that progress can be shown.** It yields the fraction
 * written after each window and finally *returns* the path. The caller drives it
 * with `.next()` and turns each yielded fraction into a redraw of the
 * attachment chip — which is the only way to report progress out of a loop
 * without a callback that cannot itself cause a render.
 *
 * **The file is read one window at a time, not all at once.** `file.arrayBuffer()`
 * — which this used to open with — decodes the whole thing into memory before a
 * single byte is sent, and then each window is copied again into a base64
 * string a third larger. A video attached from a phone therefore needed several
 * times its own size in headroom, and the tab died before anything explained
 * why. `slice` is a view rather than a copy, so peak memory is now one window
 * regardless of the file, and the size the caller sees is the size on disk.
 *
 * **Nothing is thrown before the first `next()`.** An async generator's body
 * does not run until it is driven, so the size refusal below reaches the caller
 * after it has already claimed the attachment chip — which is what makes an
 * oversized file show up as a chip that failed, wearing the reason, rather than
 * as a click that did nothing.
 */
export async function* uploadToHost(
  file: File,
): AsyncGenerator<number, string, void> {
  if (file.size > MAX_UPLOAD_BYTES) {
    throw new Error(
      `This file is ${megabytes(file.size)} MB. Second Brain takes attachments up to ${megabytes(MAX_UPLOAD_BYTES)} MB.`,
    );
  }

  // `fs.temp` is always allowed — it is the one filesystem call that never
  // raises a dialog, which is what makes this chain usable at all.
  const answer = await sdk<string | { path?: string }>("fs.temp", {
    suffix: suffixOf(file.name),
  });
  const path = typeof answer === "string" ? answer : (answer?.path ?? "");
  if (!path) throw new Error("fs.temp did not answer with a path");

  // An empty file still needs creating, hence the do/while shape: one write
  // always happens, even at length zero.
  let offset = 0;
  do {
    const window = new Uint8Array(
      await file.slice(offset, offset + CHUNK_BYTES).arrayBuffer(),
    );
    await sdk("fs.write_bytes", {
      path,
      data: base64(window),
      // The first window creates the file; the rest extend it. Getting this
      // backwards silently keeps only the last chunk.
      mode: offset === 0 ? "overwrite" : "append",
    });
    offset += CHUNK_BYTES;
    yield Math.min(1, offset / Math.max(1, file.size));
  } while (offset < file.size);

  return path;
}
