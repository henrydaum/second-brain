/** A conventional form renderer for Second Brain's state-machine commands. */

import { useEffect, useId, useRef, useState, type FC, type FormEvent } from "react";
import {
  CheckCircle2Icon,
  CheckIcon,
  CircleIcon,
  LoaderCircleIcon,
  XCircleIcon,
} from "lucide-react";

import {
  CommandMarkdown,
  CommandOutput,
} from "@/components/command-renderers";
import { Button } from "@/components/ui/button";
import { useSurfaceReveal } from "@/components/ui/use-surface-reveal";
import { FormPathPicker } from "@/components/form-path-picker";
import { cn, titleCase } from "@/lib/utils";
import { useApprovals, useSession } from "@/runtime/provider";

/** A field's own label. "Value" when the step did not name it — a form control
 *  still needs something to be called. */
const humanize = (value?: string) => titleCase(value || "Value");

export const CommandPanel: FC = () => {
  const { state, say, dismissCommand } = useSession();
  const { inputRequests } = useApprovals();
  const { command, form } = state;
  const display = form?.display;
  const fieldId = useId();
  const textArea = useRef<HTMLTextAreaElement>(null);
  const [typed, setTyped] = useState("");
  const [selectedChoice, setSelectedChoice] = useState<number | null>(null);
  const [cancelling, setCancelling] = useState(false);
  const [advancing, setAdvancing] = useState(false);
  const formSurface = useSurfaceReveal<HTMLFormElement>(form);

  useEffect(() => {
    if (!form) return;
    const defaultValue = form?.field?.default;
    setTyped(defaultValue == null ? "" : form.display?.input_mode === "json" && typeof defaultValue !== "string"
      ? JSON.stringify(defaultValue, null, 2) : String(defaultValue));
    const choices = form?.display?.choices ?? [];
    const defaultIndex = choices.findIndex(
      (choice) => String(choice.value) === String(defaultValue),
    );
    setSelectedChoice(defaultIndex >= 0 ? defaultIndex : null);
    setAdvancing(false);
  }, [form]);

  useEffect(() => {
    const element = textArea.current;
    if (!element) return;
    element.style.height = "auto";
    element.style.height = `${Math.min(element.scrollHeight, 320)}px`;
  }, [typed, form]);

  if (!command && !form) return null;

  const choices = display?.choices ?? [];
  const mode = display?.input_mode ?? "text";
  const collected = Object.entries(form?.collected ?? command?.args ?? {});
  const finished = command?.status === "finished";
  const failed = finished && command?.ok === false;
  // The command state machine emits a short acknowledgement for navigation
  // submissions. It is transport feedback, not command output; showing it
  // beneath the newly restored form produces the stray "Back." line that the
  // actual Back button already communicated.
  const visibleOutcome =
    command?.outcome.filter((text) => !/^back\.?$/i.test(text.trim())) ?? [];
  // What the command is doing right now, when it bothered to say. Only while
  // it runs: once it has finished, what it *did* is the outcome below.
  const running = finished ? undefined : command?.narration?.trim() || undefined;

  /**
   * A question is up, so this form must not send anything.
   *
   * **Not cosmetic.** While the session sits in `approving_request` the state
   * machine coerces plain text into the *answer* to that question, so a form
   * step submitted now would be eaten by the dialog instead of filling in the
   * field — and the person would have silently answered a question they were
   * looking at a different form for. The dialog's own backdrop happens to block
   * the pointer today; that is a side effect of where it renders, not a rule,
   * and this is the rule.
   */
  const blocked = inputRequests.length > 0;

  const advance = async (text: string) => {
    if (blocked) return;
    setAdvancing(true);
    const submitted = await say(text);
    if (!submitted) setAdvancing(false);
  };
  const cancel = async () => {
    if (blocked) return;
    setCancelling(true);
    const submitted = await say("/cancel");
    if (submitted) dismissCommand();
    else setCancelling(false);
  };

  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (choices.length > 0 || advancing || blocked) return;
    void advance(typed);
  };
  // One name for "this control cannot act", so a control added later cannot
  // pick up only half the reasons.
  const busy = advancing || blocked;

  return (
    <div
      data-slot="command-panel"
      className="min-w-0 w-full max-w-full overflow-hidden text-sm"
    >
      {collected.length > 0 && (
        <dl className="bg-muted/30 mb-6 grid gap-x-6 gap-y-3 rounded-lg px-4 py-3 sm:grid-cols-2 lg:grid-cols-3">
          {collected.map(([name, value]) => (
            <div key={name} className="min-w-0">
              <dt className="text-muted-foreground text-xs font-medium">
                {humanize(name)}
              </dt>
              <dd className="mt-0.5 truncate font-medium" title={String(value)}>
                {String(value)}
              </dd>
            </div>
          ))}
        </dl>
      )}

      {cancelling && (
        <div className="text-muted-foreground flex items-center gap-2 py-8">
          <LoaderCircleIcon className="size-4 animate-spin" />
          Cancelling…
        </div>
      )}

      {!cancelling && display && form && (
        <form ref={formSurface} onSubmit={submit} className="sb-step-surface space-y-6" data-pending={advancing || undefined}>
          {/* A fieldset's browser default min-width is its min-content width.
              A wide prompt table would therefore widen the choice grid below
              it unless this boundary explicitly permits shrinking. */}
          <fieldset className="min-w-0">
            <legend className="sr-only">{display.prompt}</legend>
            <div className="max-w-3xl">
              <CommandMarkdown
                text={display.prompt}
                className="[&_p:first-child]:mt-0 [&_p:last-child]:mb-0"
              />
              {display.assist && (
                <p className="text-muted-foreground mt-2 text-sm">
                  {display.assist}
                </p>
              )}
            </div>

            {choices.length > 0 ? (
              <div
                className={cn(
                  "mt-5 grid gap-2",
                  form.field?.columns === 1
                    ? "grid-cols-1"
                    : "sm:grid-cols-2",
                )}
              >
                {choices.map((choice, index) => {
                  const selected = selectedChoice === index;
                  return (
                    <label
                      key={`${index}-${String(choice.value)}`}
                      className={cn(
                        "sb-choice-control has-[:focus-visible]:border-ring flex min-h-11 min-w-0 cursor-pointer items-center gap-3 rounded-lg border px-3 py-2.5 text-start has-[:focus-visible]:ring-[3px]",
                        selected
                          ? "border-primary bg-primary/5"
                          : "bg-card/60 hover:bg-muted/70",
                      )}
                    >
                      <input
                        type="radio"
                        name={`${fieldId}-choice`}
                        value={index}
                        checked={selected}
                        disabled={busy}
                        /**
                         * **Sending is on the click, not on the change.**
                         *
                         * A step whose default is one of its choices arrives
                         * with that choice already checked, and re-picking a
                         * checked radio changes nothing, so it fires no change
                         * event. With the submission hanging off `onChange`
                         * that step was a dead end: no Continue button (picking
                         * is what continues), and no way to deselect and pick
                         * again — which is exactly what a one-choice step like
                         * "how should Second Brain connect to this model" looks
                         * like. Clicking, including a keyboard Space or the
                         * arrow keys that move the selection, always fires
                         * click. `onChange` is left to record selection only,
                         * so a choice never advances the form twice.
                         */
                        onChange={() => setSelectedChoice(index)}
                        onClick={() => {
                          setSelectedChoice(index);
                          void advance(String(choice.value));
                        }}
                        /* Space fires click on its own; Enter does not, and a
                           choice step has no submit button for the browser to
                           press implicitly, so it would otherwise do nothing. */
                        onKeyDown={(event) => {
                          if (event.key !== "Enter") return;
                          event.preventDefault();
                          setSelectedChoice(index);
                          void advance(String(choice.value));
                        }}
                        className="sr-only"
                      />
                      <span
                        className={cn(
                          "flex size-5 shrink-0 items-center justify-center rounded-full",
                          selected
                            ? "bg-primary text-primary-foreground"
                            : "text-muted-foreground",
                        )}
                      >
                        {selected ? (
                          <CheckIcon className="size-3.5" />
                        ) : (
                          <CircleIcon className="size-5" />
                        )}
                      </span>
                      <span className="min-w-0 font-medium whitespace-normal [overflow-wrap:anywhere]">
                        {choice.label ?? String(choice.value)}
                      </span>
                    </label>
                  );
                })}
              </div>
            ) : (
              <div className="mt-5 max-w-3xl">
                <label htmlFor={fieldId} className="mb-2 block font-medium">
                  {humanize(form.field?.name)}
                  {form.field?.required !== false && (
                    <span className="text-destructive ms-1" aria-hidden>
                      *
                    </span>
                  )}
                </label>
                {mode !== "number" ? (
                  <textarea
                    ref={textArea}
                    id={fieldId}
                    autoFocus
                    disabled={busy}
                    required={form.field?.required !== false}
                    rows={mode === "json" ? 6 : 1}
                    value={typed}
                    onChange={(event) => setTyped(event.target.value)}
                    className={cn(
                      "border-input bg-background focus-visible:border-ring focus-visible:ring-ring/30 min-h-10 max-h-80 w-full resize-none overflow-y-auto rounded-lg border px-3 py-2 text-base leading-6 outline-none focus-visible:ring-[3px]",
                      mode === "json" && "font-mono",
                    )}
                  />
                ) : (
                  <input
                    id={fieldId}
                    autoFocus
                    disabled={busy}
                    required={form.field?.required !== false}
                    type="number"
                    step={
                      form.field?.type === "integer" ||
                      form.field?.type === "int"
                        ? "1"
                        : undefined
                    }
                    value={typed}
                    onChange={(event) => setTyped(event.target.value)}
                    /**
                     * **16px, because iOS reads anything smaller as an
                     * invitation to zoom.**
                     *
                     * Safari magnifies the page when a field below 16px takes
                     * focus and never undoes it, which left Settings enlarged
                     * with its own close button panned off-screen — the dialog
                     * is `position: fixed`, so it does not travel back, and
                     * Radix has the page behind it locked. Typing a context
                     * size became a trap you escaped by pinching.
                     *
                     * Stated here rather than inherited: the base stylesheet
                     * gives form controls `font: inherit`, so this was 14px by
                     * way of the panel's `text-sm` with nothing on the element
                     * to say so. The other fields Settings can raise carry the
                     * same note.
                     */
                    className="border-input bg-background focus-visible:border-ring focus-visible:ring-ring/30 h-10 w-full rounded-lg border px-3 text-base outline-none focus-visible:ring-[3px]"
                  />
                )}
                {mode !== "number" && <FormPathPicker inputId={fieldId} value={typed} mode={mode}
                  disabled={busy} identity={form} onChange={setTyped} />}
              </div>
            )}
          </fieldset>

          <div className="flex flex-wrap items-center justify-between gap-3 border-t pt-4">
            <div className="flex items-center gap-1">
              {display.allow_cancel !== false && (
                <Button
                  type="button"
                  variant="ghost"
                  disabled={cancelling || blocked}
                  onClick={() => void cancel()}
                >
                  Cancel
                </Button>
              )}
              {display.allow_back && (
                <Button
                  type="button"
                  variant="ghost"
                  disabled={busy}
                  onClick={() => void advance("/back")}
                >
                  Back
                </Button>
              )}
            </div>
            <div className="flex items-center gap-2">
              {display.allow_skip && (
                <Button
                  type="button"
                  variant="outline"
                  disabled={busy}
                  onClick={() => void advance("/skip")}
                >
                  Skip
                </Button>
              )}
              {choices.length === 0 && (
                <Button type="submit" disabled={busy}>
                  Continue
                </Button>
              )}
            </div>
          </div>
        </form>
      )}

      {!cancelling && command && visibleOutcome.length > 0 && (
        <CommandOutput output={visibleOutcome} />
      )}

      {!cancelling &&
        command &&
        !display &&
        visibleOutcome.length === 0 &&
        !finished && (
        <div className="text-muted-foreground flex items-center gap-2 py-8">
          <LoaderCircleIcon className="size-4 animate-spin" />
          {/* A command long enough to narrate itself says what it is doing
              here, in the panel that started it. Before this it had nowhere
              to say it but the chat — a package install announcing "Copying
              package files" into the transcript, from a settings screen. The
              generic line still covers every command that says nothing. */}
          {advancing
            ? "Loading next step…"
            : (running ?? "Running command…")}
        </div>
      )}

      {!cancelling && command && !display && finished && (
        <div className="mt-6 flex items-center justify-between gap-4 border-t pt-4">
          <span
            className={cn(
              "inline-flex items-center gap-2 text-sm",
              failed ? "text-destructive" : "text-muted-foreground",
            )}
          >
            {failed ? (
              <XCircleIcon className="size-4" />
            ) : (
              <CheckCircle2Icon className="size-4 text-emerald-600" />
            )}
            {failed ? (command.error ?? "Command failed") : "Complete"}
          </span>
          <Button type="button" variant="outline" onClick={dismissCommand}>
            Close
          </Button>
        </div>
      )}
    </div>
  );
};
