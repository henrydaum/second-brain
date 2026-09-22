import { Suspense, useRef, useState, type FC } from "react";
import { LoaderCircleIcon, Settings2Icon } from "lucide-react";

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import { lazyWithPreload } from "@/lib/lazy";
import { useSession } from "@/runtime/domains";

const [LazySettingsContent, preloadSettings] = lazyWithPreload(
  () => import("@/components/settings-dialog"),
  (module) => module.SettingsDialogContent,
);

export { preloadSettings };

/**
 * The dialog shell is eager and persistent. Only its body crosses the lazy
 * boundary, so resolving the Settings chunk cannot remount an already-open
 * Radix dialog and replay its entrance animation.
 */
export const SettingsDialog: FC<{
  open: boolean;
  onOpenChange: (open: boolean) => void;
}> = ({ open, onOpenChange }) => {
  const { say, state, dismissCommand } = useSession();
  const [commandActionPending, setCommandActionPending] = useState(false);
  const commandActionPendingRef = useRef(false);
  const activeName = state.form?.name ?? state.command?.name;
  const commandActive = Boolean(activeName);
  const commandRunning =
    commandActive && state.command?.status !== "finished";
  /**
   * **The command's body is running, and nothing here can stop it.**
   *
   * `/cancel` only pops a *frame* — a form step or a pending approval. Once
   * the body runs there is no frame, so the cancel this dialog used to send on
   * close was accepted and did nothing: `/update` went on pulling, the panel
   * forgot it, and reopening Settings let a second run start alongside the
   * first — whose approval then answered "That request is no longer active".
   * So while the body runs, Settings stays put until the command finishes.
   */
  const commandExecuting = commandRunning && !state.form;
  const locked = commandActionPending || commandExecuting;

  const afterCurrentCommand = async (
    action: () => void | Promise<void>,
  ) => {
    if (!commandActive) {
      await action();
      return true;
    }
    if (commandExecuting || commandActionPendingRef.current) return false;

    commandActionPendingRef.current = true;
    setCommandActionPending(true);
    try {
      if (commandRunning) {
        const submitted = await say("/cancel");
        if (!submitted) return false;
      }
      dismissCommand();
      await action();
      return true;
    } finally {
      commandActionPendingRef.current = false;
      setCommandActionPending(false);
    }
  };

  const handleOpenChange = (next: boolean) => {
    // Closing while a command is active is also a cancellation. Wait for the
    // server to accept it before hiding Settings so a failed submission does
    // not leave the session waiting on an invisible question.
    if (!next && commandExecuting) return;
    if (!next && commandActive) {
      void afterCurrentCommand(() => onOpenChange(false));
      return;
    }
    onOpenChange(next);
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent
        className="sb-glass sb-glass-sheet flex h-[min(94dvh,54rem)] w-[min(calc(100vw-1rem),70rem)] max-w-none flex-col gap-0 overflow-hidden p-0 sm:max-w-none"
        overlayClassName="bg-black/45 backdrop-blur-[2px]"
        closeButtonDisabled={locked}
        onOpenAutoFocus={(event) => {
          event.preventDefault();
          (event.currentTarget as HTMLElement | null)?.focus();
        }}
      >
        <header className="flex h-14 min-w-0 shrink-0 items-center gap-3 border-b ps-4 pe-14 sm:h-16 sm:px-6 sm:pe-16">
          <span className="bg-primary text-primary-foreground flex size-8 items-center justify-center rounded-lg">
            <Settings2Icon className="size-4" />
          </span>
          <div className="min-w-0">
            <DialogTitle className="truncate text-base">Second Brain settings</DialogTitle>
            <DialogDescription className="truncate text-xs">
              Kernel, agents, security, plugins, and packages
            </DialogDescription>
          </div>
        </header>

        <Suspense fallback={<SettingsFallback />}>
          <LazySettingsContent
            commandActionPending={locked}
            afterCurrentCommand={afterCurrentCommand}
          />
        </Suspense>
      </DialogContent>
    </Dialog>
  );
};

const SettingsFallback: FC = () => (
  <div
    className="bg-popover text-muted-foreground flex flex-1 items-center justify-center gap-2 text-sm"
    role="status"
  >
    <LoaderCircleIcon className="size-4 animate-spin" />
    Loading settings…
  </div>
);
