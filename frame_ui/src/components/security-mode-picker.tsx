import { useState, type FC } from "react";
import { ChevronDownIcon, ShieldCheckIcon } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  useApprovals,
  useSecurity,
  useSession,
} from "@/runtime/provider";

const MODES = [
  {
    id: "lockdown" as const,
    label: "Lockdown",
    description: "Refuse actions that would require approval.",
  },
  {
    id: "ask" as const,
    label: "Ask",
    description: "Ask before actions that require approval.",
  },
  {
    id: "yolo" as const,
    label: "YOLO",
    description: "Approve actions automatically.",
  },
];

export const SecurityModePicker: FC = () => {
  const { securityMode, setSecurityMode } = useSecurity();
  const { state } = useSession();
  const { inputRequests } = useApprovals();
  const [open, setOpen] = useState(false);
  const [changing, setChanging] = useState(false);
  const selected = MODES.find((mode) => mode.id === securityMode) ?? MODES[1];
  const disabled =
    changing ||
    state.typing ||
    state.form !== null ||
    inputRequests.length > 0;

  const choose = async (mode: (typeof MODES)[number]["id"]) => {
    setOpen(false);
    if (mode === securityMode) return;
    setChanging(true);
    try {
      await setSecurityMode(mode);
    } finally {
      setChanging(false);
    }
  };

  return (
    <DropdownMenu open={open} onOpenChange={setOpen}>
      <DropdownMenuTrigger asChild>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          disabled={disabled}
          onMouseDown={(event) => event.stopPropagation()}
          className="text-muted-foreground hover:text-foreground h-7 gap-1.5 rounded-full px-2 text-xs font-normal"
          aria-label={`Security mode: ${selected.label}`}
        >
          <ShieldCheckIcon className="size-3.5" />
          <span className="hidden sm:inline">
            {changing ? "Changing…" : selected.label}
          </span>
          <ChevronDownIcon className="hidden size-3 sm:block" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent
        side="top"
        align="start"
        sideOffset={8}
        onMouseDown={(event) => event.stopPropagation()}
        className="w-72"
      >
        <DropdownMenuLabel>
          <span className="block">Security mode</span>
          <p className="text-muted-foreground mt-0.5 text-xs">
            Choose how this conversation handles approval requests.
          </p>
        </DropdownMenuLabel>
        <DropdownMenuRadioGroup value={securityMode} onValueChange={(value) => void choose(value as (typeof MODES)[number]["id"])}>
          {MODES.map((mode) => (
            <DropdownMenuRadioItem
              key={mode.id}
              value={mode.id}
              className="items-start py-2"
            >
              <span>
                <span className="block text-sm font-medium">{mode.label}</span>
                <span className="text-muted-foreground mt-0.5 block text-xs leading-relaxed">
                  {mode.description}
                </span>
              </span>
            </DropdownMenuRadioItem>
          ))}
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  );
};
