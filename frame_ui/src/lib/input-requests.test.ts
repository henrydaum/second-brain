/**
 * The enum/label pairing, which the dialog's own comment calls out as the rule
 * that is easy to get subtly, silently wrong — and which nothing checked.
 */

import { describe, expect, it } from "vitest";

import { choicesOf, isPendingInput, type InputRequest } from "@/lib/input-requests";

const request = (fields: Partial<InputRequest>): InputRequest => ({
  id: "approve_1",
  title: "A question",
  ...fields,
});

describe("choicesOf", () => {
  it("pairs values with labels by index", () => {
    // Answer the value, show the label. Backwards, this puts internal
    // spellings like `always:api.search.brave.com` on a person's buttons.
    const choices = choicesOf(
      request({
        type: "string",
        enum: ["allow", "always:api.brave.com", "deny"],
        enum_labels: ["Allow once", "Always allow api.brave.com", "Deny"],
      }),
    );

    expect(choices).toEqual([
      { value: "allow", label: "Allow once" },
      { value: "always:api.brave.com", label: "Always allow api.brave.com" },
      { value: "deny", label: "Deny" },
    ]);
  });

  it("falls back to the value when a label is missing", () => {
    // `enum_labels` may be null even when `enum` is not, and may be shorter.
    const choices = choicesOf(
      request({ type: "string", enum: ["allow", "deny"], enum_labels: ["Allow"] }),
    );

    expect(choices.map((choice) => choice.label)).toEqual(["Allow", "deny"]);
  });

  it("tolerates enum_labels being null outright", () => {
    const choices = choicesOf(
      request({ type: "string", enum: [1, 2], enum_labels: null }),
    );

    expect(choices).toEqual([
      { value: 1, label: "1" },
      { value: 2, label: "2" },
    ]);
  });

  it("offers no buttons for free text", () => {
    // `ui.ask` with a string type and no choices. Without this the dialog
    // would draw Allow/Deny for a question that wants a sentence.
    expect(choicesOf(request({ type: "string" }))).toEqual([]);
    expect(choicesOf(request({ type: "integer", enum: [] }))).toEqual([]);
  });

  it("offers Allow and Deny for a bare boolean", () => {
    expect(choicesOf(request({}))).toEqual([
      { value: true, label: "Allow" },
      { value: false, label: "Deny" },
    ]);
    expect(choicesOf(request({ type: "boolean", enum: null }))).toEqual([
      { value: true, label: "Allow" },
      { value: false, label: "Deny" },
    ]);
  });
});

describe("isPendingInput", () => {
  it("accepts the two kinds the server tags", () => {
    expect(isPendingInput({ kind: "approval", payload: {} })).toBe(true);
    expect(isPendingInput({ kind: "form_field", payload: {} })).toBe(true);
  });

  it("rejects what a kernel older than `details` answers", () => {
    // A bare id, or `true` for "one exists but I cannot name it". Treating
    // either as a payload is how `true` ends up sent back as a request id.
    expect(isPendingInput("approve_abc")).toBe(false);
    expect(isPendingInput(true)).toBe(false);
    expect(isPendingInput(null)).toBe(false);
    expect(isPendingInput(undefined)).toBe(false);
  });
});
