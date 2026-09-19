/**
 * The composer's attachment adapter.
 *
 * Beside `provider.tsx` rather than in it so that file exports nothing but its
 * component — see the note at the top of `domains.ts` for what a mixed export
 * costs under Fast Refresh.
 */

import type {
  AttachmentAdapter,
  PendingAttachment,
} from "@assistant-ui/react";

import { uploadToHost } from "@/lib/upload";
import {
  forgetStagedPath,
  rememberStagedPath,
} from "@/runtime/staged-attachments";

/* ── Attachments ────────────────────────────────────────────────────────
 *
 * The host path an upload produced, kept beside the attachment rather than
 * inside it. `CompleteAttachment.content` is message content — what the model
 * would see — and a scratch path is not that; it is a detail of how the bytes
 * got across. So it lives in the shared staged-attachment registry, keyed by
 * attachment id, and `onNew` reads it back when the message is actually sent.
 */

/** Exported for its own test. Nothing else should reach for it: it is handed
 *  to the runtime below, and the composer is the only thing that drives it. */
export const attachmentAdapter: AttachmentAdapter = {
  // Everything.
  //
  // **A bare star, not the MIME wildcard.** assistant-ui treats this as a
  // literal, not a pattern: the single star is special-cased as "no filter",
  // and anything else goes through `fileMatchesAccept`, which compares MIME
  // types and extensions against the list. The MIME wildcard matches neither
  // of those, so *every* file was rejected — and `AddAttachment` swallows that
  // rejection, which is why picking a file did nothing rather than saying why.
  // The same string is also handed to the file input's `accept`, so the picker
  // itself was filtering everything out before we were even asked.
  accept: "*",

  async *add({ file }) {
    const id = crypto.randomUUID();
    const base = {
      id,
      type: file.type.startsWith("image/") ? ("image" as const) : ("file" as const),
      name: file.name,
      contentType: file.type,
      file,
    };

    // **Yielded before anything is attempted.** assistant-ui shows the chip on
    // the first yield, so work done before it happens behind nothing at all.
    // Claiming the chip up front is what gives a failure somewhere to be drawn.
    yield {
      ...base,
      status: { type: "running", reason: "uploading", progress: 0 },
    } satisfies PendingAttachment;

    // **A failure is yielded, never thrown.** Both of assistant-ui's call sites
    // — the paperclip and the dropzone — await this inside a `try {} catch {}`
    // with an empty body, so an exception from here is discarded and the chip
    // is left frozen at whatever it last showed: 0%, forever, with nothing
    // said. The library's own upload adapter yields an `incomplete` status for
    // the same reason. That status is what `AttachmentProgress` draws as a red
    // tile and `AttachmentLabel` explains in the tooltip; without this, both
    // were unreachable code.
    try {
      // Uploading here rather than in `send` so the progress bar means
      // something: by the time the person hits send, the bytes are already
      // across and the send is one small Request.
      const upload = uploadToHost(file);
      let step = await upload.next();
      while (!step.done) {
        yield {
          ...base,
          status: { type: "running", reason: "uploading", progress: step.value },
        } satisfies PendingAttachment;
        step = await upload.next();
      }
      rememberStagedPath(id, step.value);
    } catch (error) {
      yield {
        ...base,
        status: {
          type: "incomplete",
          reason: "error",
          // The sentence the person reads. `uploadToHost` writes the one about
          // size; a refused or failed write arrives here as its own Request
          // failure, which until now was equally silent.
          message:
            error instanceof Error
              ? error.message
              : "This file could not be attached.",
        },
      } satisfies PendingAttachment;
      return;
    }

    yield {
      ...base,
      status: { type: "requires-action", reason: "composer-send" },
    } satisfies PendingAttachment;
  },

  async send(attachment) {
    // The upload already happened in `add`. All that is left is to promote the
    // chip to complete; the actual `frontend.submit` happens in `onNew`, which
    // is the only place that knows the accompanying text.
    return {
      ...attachment,
      status: { type: "complete" },
      content: [{ type: "text", text: `[attachment: ${attachment.name}]` }],
    };
  },

  async remove(attachment) {
    // The scratch file is left on the host. `fs.delete` is a policy-gated write
    // and would raise a dialog for something the person did not ask about —
    // asking permission to tidy up is worse than the stray temp file.
    forgetStagedPath(attachment.id);
  },
};
