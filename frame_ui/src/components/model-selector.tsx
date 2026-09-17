import {
  ChevronDownIcon,
  LoaderCircleIcon,
  SlidersHorizontalIcon,
} from "lucide-react";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { spellModelWord } from "@/lib/model-names";
import { useModels, useSession, useSettings } from "@/runtime/provider";

/**
 * One argument of a command line, quoted for the kernel's `shlex` parser.
 *
 * Model names are the user's own text and routinely carry `/`; nothing stops
 * one carrying a quote. Single quotes are literal to `shlex`, so the only
 * character needing care is the quote itself, closed and re-opened around an
 * escaped one.
 */
function quoteArg(value: string): string {
  return `'${value.replaceAll("'", `'\\''`)}'`;
}

/** A compact, provider-free label for the composer trigger. The exact profile
 * key remains untouched everywhere that identity matters. */
export function compactModelName(modelName: string | null | undefined): string {
  const exact = (modelName ?? "").trim();
  if (!exact) return "Select model";
  const parts = exact.split("/").filter(Boolean);
  const suffix = parts.at(-1) || exact;
  const words = suffix.replace(/[-_]+/g, " ").trim().split(/\s+/);
  if (words.length === 0 || !words[0]) return exact;
  return words.map(spellModelWord).join(" ");
}

export function ModelSelector() {
  const [open, setOpen] = useState(false);
  const {
    models,
    modelName,
    agentProfile,
    modelsLoading,
    modelsFailure,
    switchingModel,
    setModel,
  } = useModels();
  const { openSettings } = useSettings();
  const { say, state } = useSession();

  /**
   * Open this model's parameters, not merely the section that lists models.
   *
   * `/llm <model> edit` fills the two steps before the one worth arriving at,
   * so the form lands on the field list — every setting this profile has, each
   * configured parameter with its value, and Add a parameter. Opening the
   * section alone would be one click from the composer and one more from
   * there, which is what the panel already offered.
   *
   * **The dialog goes up before the command, and the order is the point** —
   * the same order `notification-panel` uses for the same reason. `openSettings`
   * puts it there now; the command is a round trip away, and a link that did
   * nothing visible until the server answered would read as broken. It also
   * decides where a *failed* submit lands: a settings page, rather than
   * nowhere.
   */
  const configureModel = () => {
    // Checked here as well as on the item. Radix disables an item with
    // `pointer-events: none`, which is presentation — it is the right thing
    // for the pointer and it is not a guarantee, and this one is about not
    // submitting a command onto a running turn.
    if (!modelName || state.typing) return;
    setOpen(false);
    openSettings("plugins");
    void say(`/llm ${quoteArg(modelName)} edit`);
  };
  const label = modelsLoading && !modelName ? "Loading…" : compactModelName(modelName);

  return (
    <DropdownMenu open={open} onOpenChange={setOpen}>
      <DropdownMenuTrigger asChild>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          aria-label={modelName ? `Model: ${modelName}` : "Select model"}
          className="text-muted-foreground hover:text-foreground h-7 min-w-0 max-w-36 flex-1 gap-1 px-2 text-xs sm:w-auto sm:max-w-52 sm:flex-none sm:justify-end"
        >
          {switchingModel && <LoaderCircleIcon className="size-3 animate-spin" />}
          <span className="min-w-0 truncate sm:text-right">{label}</span>
          <ChevronDownIcon className="size-3.5 shrink-0 opacity-60" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent
        align="end"
        side="top"
        className="w-[min(20rem,calc(100vw-2rem))]"
      >
        <DropdownMenuLabel className="text-muted-foreground text-xs font-normal">
          Default language model
        </DropdownMenuLabel>
        {models.length > 0 ? (
          <DropdownMenuRadioGroup
            value={modelName ?? ""}
            onValueChange={(value) => {
              setOpen(false);
              void setModel(value);
            }}
            className="max-h-64 overflow-y-auto"
          >
            {models.map((model) => (
              <DropdownMenuRadioItem
                key={model.model_name}
                value={model.model_name}
                disabled={switchingModel}
                className="font-mono text-xs"
              >
                <span className="min-w-0 truncate">{model.model_name}</span>
              </DropdownMenuRadioItem>
            ))}
          </DropdownMenuRadioGroup>
        ) : (
          <p className="text-muted-foreground px-2 py-3 text-sm">
            {modelsLoading
              ? "Loading configured models…"
              : modelsFailure
                ? "Models are temporarily unavailable."
                : "No LLM profiles are configured."}
          </p>
        )}
        <DropdownMenuSeparator />
        {/*
          * A link rather than a control.
          *
          * This was four dots writing `off/low/medium/high` into
          * `reasoning_effort`, which was a guess in three directions at once:
          * that the parameter is called that, that this model takes it, and
          * that those are its values. None is true generally — the name is
          * `effort` or `thinking` elsewhere, most models take a different set
          * or none, and `off` is a level at no provider at all.
          *
          * Building it from what the backend reports was tried and works, but
          * it can only ever show a control for a model something has vouched
          * for, which is a minority of them — so the panel would offer the
          * setting sometimes and silently not others, with no way to tell
          * which case you were in. Settings shows every parameter a profile
          * has and what it accepts, with the caveats attached. One
          * destination that is always right beats a control that is
          * occasionally present.
          */}
        <DropdownMenuItem
          onClick={configureModel}
          // While a turn is running, every way back *out* of Settings — closing
          // it, or moving to another section — submits `/cancel`, which lands
          // on that turn. Ending a turn is a decision, and it should be made at
          // the composer's Stop button rather than arrived at by following a
          // link about a parameter. Same guard the notification links use.
          disabled={!modelName || state.typing}
          className="text-xs"
        >
          <SlidersHorizontalIcon className="size-3.5 opacity-60" />
          Configure this model
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <div className="text-muted-foreground flex min-h-9 items-center px-2 text-xs">
          {/* Wrapped so the label and value stay one inline run — as direct
              children of a flex box the space between them is discarded. */}
          <span className="min-w-0 truncate">
            Agent profile: <span className="text-foreground">{agentProfile}</span>
          </span>
        </div>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
