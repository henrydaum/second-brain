/**
 * A background agent reporting back to the turn that was waiting on it.
 *
 * **Collapsed by default, and the report is one click away.** It is written for
 * the model, which reads it and answers in its own words a moment later, so
 * showing it open would put the same findings on screen twice. But it is also
 * the only first-hand account of what the agent found, and the agent's summary
 * of it is a summary — so it opens in place rather than sending you to the
 * child's conversation, which cannot be switched to while this turn is still
 * running anyway.
 *
 * Dressed like the tool-group trigger beside it — muted, one line, a chevron,
 * brightening on hover — because it is the same kind of thing: the agent's
 * machinery, surfacing briefly in the middle of its reply. A failure is not
 * red: nothing here is the person's to fix, and the agent says what it means
 * next.
 */

import { useState, type FC } from "react";
import { ChevronDownIcon, CornerDownRightIcon } from "lucide-react";

import { CommandMarkdown } from "@/components/command-renderers";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { cn } from "@/lib/utils";

export type AgentReturnedData = {
  title: string;
  state: "done" | "failed" | "cancelled";
  text?: string;
  conversationId: number;
};

const VERB: Record<AgentReturnedData["state"], string> = {
  done: "returned",
  failed: "failed",
  cancelled: "timed out",
};

export const AgentReturned: FC<AgentReturnedData> = ({ title, state, text }) => {
  const [open, setOpen] = useState(false);
  const verb = VERB[state] ?? "returned";
  const report = (text ?? "").trim();

  const label = (
    <span className="truncate">
      <span className="text-foreground/80 font-medium">{title}</span> {verb}
    </span>
  );

  // A timeout has nothing to show, so it is a line rather than a control that
  // opens onto nothing.
  if (!report) {
    return (
      <div data-slot="agent-returned" data-state={state}
        className="text-muted-foreground fade-in animate-in flex max-w-full items-center gap-2 py-1 text-sm duration-(--sb-motion-reveal)">
        <CornerDownRightIcon aria-hidden className="size-4 shrink-0" />
        {label}
      </div>
    );
  }

  return (
    <Collapsible open={open} onOpenChange={setOpen}
      data-slot="agent-returned" data-state={state}
      className="fade-in animate-in w-full duration-(--sb-motion-reveal)">
      <CollapsibleTrigger
        className="group/trigger text-muted-foreground hover:text-foreground flex max-w-full items-center gap-2 py-1 text-sm transition-colors">
        <CornerDownRightIcon aria-hidden className="size-4 shrink-0" />
        {label}
        <ChevronDownIcon aria-hidden className={cn(
          "size-4 shrink-0 transition-transform duration-(--animation-duration) ease-out",
          "group-data-[state=closed]/trigger:-rotate-90",
        )} />
      </CollapsibleTrigger>
      <CollapsibleContent>
        {/* Indented to the label and ruled on the left, so it reads as the
            agent's words quoted into the reply rather than the reply itself. */}
        <div className="sb-divider-start text-muted-foreground mt-1 mb-2 ml-2 max-h-96 overflow-y-auto pl-4">
          <CommandMarkdown text={report} />
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
};
