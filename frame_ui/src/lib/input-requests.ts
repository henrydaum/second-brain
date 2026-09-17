/**
 * Questions the kernel blocks a turn on, and how to answer them.
 *
 * **These are not only permission prompts.** One kernel primitive —
 * `runtime.request_input` — backs a sandbox permission gate, `ui.ask`, and a
 * tool asking the person something, and all three arrive as the same `approval`
 * frame. So `type` may be any of `boolean`, `string`, `integer`, `number`,
 * `array` or `object`, with or without an `enum`, and wording the dialog as a
 * permission grant mislabels an ordinary question as one.
 *
 * The wire kind is still spelled `approval` because that is the protocol's name
 * for it; everything above the wire calls it an input request, which is what it
 * is.
 */

import type { ApprovalPayload, FormFieldPayload } from "@/lib/events";

/**
 * One question, as the queue holds it.
 *
 * `id` is nullable for the one case the kernel cannot give us one: a pending
 * approval it knows about but has no recorded order for, where the answer goes
 * to "whatever is next" rather than to a named request.
 */
export type InputRequest = Omit<ApprovalPayload, "id"> & { id: string | null };

/** One selectable answer: `value` goes to the server, `label` to the person. */
type Choice = { value: unknown; label: string };

/**
 * Pair `enum` with `enum_labels` **by index**.
 *
 * One named function rather than an inline `.map` at each call site, because
 * this is the rule that is easy to get subtly, silently wrong. Labels may be
 * absent even when values are not, so each falls back to its own value.
 *
 * Getting it backwards puts internal spellings like `always:api.search.brave.com`
 * on a person's buttons.
 */
export function choicesOf(request: InputRequest): Choice[] {
  if (!Array.isArray(request.enum) || request.enum.length === 0) {
    // No enum and a non-boolean type is free text — the only shape here that
    // cannot be answered by pressing something.
    if ((request.type ?? "boolean") !== "boolean") return [];
    // A bare boolean. **Allow/Deny rather than Yes/No** because the kernel only
    // reaches this shape from `_CallableAction._approval`, which is a gated
    // command asking to run; `ui.ask` with a boolean sends `type: "boolean"`
    // too, but with the question in the title, where Allow/Deny still reads.
    return [
      { value: true, label: "Allow" },
      { value: false, label: "Deny" },
    ];
  }
  const labels = request.enum_labels ?? [];
  return request.enum.map((value, index) => ({
    value,
    label: String(labels[index] ?? value),
  }));
}

/**
 * What `frontend.pending {details: true}` hands back.
 *
 * A render is an event and events are not re-sent, so this is the only route
 * back to a question the page was not connected for. Both kinds are here
 * because they are one thing — a session blocked until a person answers — and
 * restoring one but not the other still strands people.
 */
export type PendingInput =
  | { kind: "approval"; payload: ApprovalPayload }
  | { kind: "form_field"; payload: FormFieldPayload }
  | null;

/** Whether the kernel's answer is a shape we understand. A kernel older than
 *  `details` answers the bare id instead, which is not one. */
export function isPendingInput(value: unknown): value is NonNullable<PendingInput> {
  if (value === null || typeof value !== "object") return false;
  const kind = (value as { kind?: unknown }).kind;
  return kind === "approval" || kind === "form_field";
}
