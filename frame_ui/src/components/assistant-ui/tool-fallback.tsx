import type { ToolCallMessagePartComponent } from "@assistant-ui/react";
import {
  CheckIcon,
  ChevronRightIcon,
  CircleAlertIcon,
  LoaderCircleIcon,
  WrenchIcon,
} from "lucide-react";

import { ToolInput } from "@/components/tool-input";

import { cn, titleCase } from "@/lib/utils";

function printable(value: unknown) {
  if (typeof value === "string") return value;
  return JSON.stringify(value, null, 2);
}

export const ToolFallback: ToolCallMessagePartComponent = ({
  toolName,
  args,
  argsText,
  result,
  status,
  isError,
}) => {
  const running = status.type === "running";
  const failed = status.type === "incomplete" || isError === true;

  /**
   * What the agent said it was doing, when it said anything.
   *
   * `narration` is a reserved argument name: the model fills it in to explain
   * *why* it is reaching for a tool, and the kernel strips it before the tool
   * runs. It is the one part of a call written for a person to read, and until
   * now the only way to see it was to expand the row and find it in the JSON —
   * so a wall of identically-titled "Read File" rows stayed a wall.
   *
   * Beside the name rather than instead of it. Which tool ran is the fact that
   * does not vary in reliability; the narration is the model's own account of
   * itself, and reads as a gloss on the name rather than a replacement for it.
   */
  const narration =
    typeof args?.narration === "string" ? args.narration.trim() : "";

  // Emptiness is decided here, not by `result !== undefined`. A tool is free to
  // report nothing at all, and a heading over a blank box reads as a value that
  // went missing rather than one that was never sent.
  const outcome = result === undefined ? "" : printable(result).trim();
  const hasDetails = Boolean(argsText) || outcome !== "";

  const icon = running ? (
    <LoaderCircleIcon className="size-3.5 animate-spin" />
  ) : failed ? (
    <CircleAlertIcon className="size-3.5" />
  ) : (
    <CheckIcon className="size-3.5" />
  );

  return (
    <details
      className={cn(
        "group/tool rounded-lg border bg-background/70 text-sm",
        failed && "border-destructive/40",
      )}
    >
      <summary
        className={cn(
          "flex min-h-10 list-none items-center gap-2.5 px-3 py-2",
          hasDetails && "cursor-pointer",
        )}
      >
        <WrenchIcon className="text-muted-foreground size-3.5 shrink-0" />
        {/* The name holds its width and the narration gives up whatever is
            left, so a long blurb ellipsizes rather than squeezing out the one
            word that says which tool this was. */}
        <span className="flex min-w-0 flex-1 items-baseline gap-1.5">
          <span className="shrink-0 font-medium">{titleCase(toolName)}</span>
          {narration !== "" && (
            <span className="text-muted-foreground min-w-0 truncate text-xs">
              <span aria-hidden>— </span>
              {narration}
            </span>
          )}
        </span>
        <span
          className={cn(
            "text-muted-foreground flex items-center gap-1.5 text-xs",
            failed && "text-destructive",
          )}
        >
          {icon}
          {running ? "Running" : failed ? "Failed" : "Done"}
        </span>
        {hasDetails && (
          <ChevronRightIcon className="text-muted-foreground size-3.5 transition-transform group-open/tool:rotate-90" />
        )}
      </summary>
      {hasDetails && (
        <div className="space-y-3 border-t px-3 py-3">
          {argsText && <ToolInput args={args} argsText={argsText} />}
          {outcome !== "" && (
            <div>
              {/* One or the other, never both: the kernel leaves the summary
                  empty on a failure so that the error is the whole answer. */}
              <p
                className={cn(
                  "mb-1.5 text-xs font-medium",
                  failed ? "text-destructive" : "text-muted-foreground",
                )}
              >
                {failed ? "Error" : "Result"}
              </p>
              <pre className="bg-muted/60 max-h-56 overflow-auto rounded-md p-2.5 text-xs whitespace-pre-wrap">
                {outcome}
              </pre>
            </div>
          )}
        </div>
      )}
    </details>
  );
};
