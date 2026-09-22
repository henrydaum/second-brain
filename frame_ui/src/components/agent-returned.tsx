/**
 * A background agent reporting back to the turn that was waiting on it.
 *
 * **The report is not shown, and that is the point of the row.** It is written
 * for the model, which reads it and answers in its own words a moment later;
 * printing it here would put the same findings on screen twice, the first time
 * unedited. What the person is owed is that it arrived — and, since the report
 * is a preview, where the whole transcript lives.
 *
 * Dressed like the tool-group trigger beside it — muted, one line, brightening
 * on hover — because it is the same kind of thing: the agent's machinery,
 * surfacing briefly in the middle of its reply. A failure is not red: nothing
 * here is the person's to fix, and the agent says what it means next.
 */

import type { FC } from "react";
import { CornerDownRightIcon } from "lucide-react";

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useConversations } from "@/runtime/domains";

export type AgentReturnedData = {
  title: string;
  state: "done" | "failed" | "cancelled";
  conversationId: number;
};

const VERB: Record<AgentReturnedData["state"], string> = {
  done: "returned",
  failed: "failed",
  cancelled: "timed out",
};

export const AgentReturned: FC<AgentReturnedData> = ({ title, state, conversationId }) => {
  const { openConversation } = useConversations();
  const verb = VERB[state] ?? "returned";

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          data-slot="agent-returned"
          data-state={state}
          onClick={() => void openConversation(conversationId)}
          className="text-muted-foreground hover:text-foreground fade-in animate-in flex max-w-full items-center gap-2 py-1 text-sm transition-colors duration-(--sb-motion-reveal)"
        >
          <CornerDownRightIcon aria-hidden className="size-4 shrink-0" />
          <span className="truncate">
            <span className="text-foreground/80 font-medium">{title}</span>{" "}
            {verb}
          </span>
        </button>
      </TooltipTrigger>
      <TooltipContent side="top" variant="subtle">
        Open its conversation
      </TooltipContent>
    </Tooltip>
  );
};
