/**
 * The light/dark control.
 *
 * A three-way menu rather than a two-state toggle, because "System" is a real
 * answer and the one most people are already on — see `lib/theme.ts`. The shape
 * is deliberately the same as the other Settings controls: a quiet row opening
 * a radio menu, with the current choice visible before the menu is opened.
 */

import type { FC } from "react";
import { MonitorIcon, MoonIcon, SunIcon } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useTheme, type Theme } from "@/lib/theme";

const OPTIONS: { id: Theme; label: string; icon: typeof SunIcon }[] = [
  { id: "system", label: "System", icon: MonitorIcon },
  { id: "light", label: "Light", icon: SunIcon },
  { id: "dark", label: "Dark", icon: MoonIcon },
];

export const ThemePicker: FC = () => {
  const { theme, setTheme, resolved } = useTheme();

  // The trigger shows what you are *looking at*, not what you *chose* — on
  // "System" a sun icon at night would be simply wrong.
  const TriggerIcon = resolved === "dark" ? MoonIcon : SunIcon;
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        {/* A plain `Button`, matching `SecurityModePicker`. A tooltip on a
            control that opens a labelled menu is one hover surface too many,
            and it keeps this trigger to a single `asChild` hand-off. */}
        <Button
          type="button"
          variant="ghost"
          size="sm"
          // Verb-first, like Update, Restart and Shut down beside it — this is
          // a control that does something, and the bare noun read as a heading
          // over them rather than as one of them.
          aria-label="Change appearance"
          className="text-muted-foreground hover:text-foreground h-8 w-full justify-start gap-2 px-3 font-normal"
        >
          <TriggerIcon className="size-4" />
          <span>Change appearance</span>
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" sideOffset={8} className="w-44">
        <DropdownMenuLabel>Appearance</DropdownMenuLabel>
        <DropdownMenuRadioGroup
          value={theme}
          onValueChange={(value) => setTheme(value as Theme)}
        >
          {OPTIONS.map((option) => {
            const Icon = option.icon;
            return (
              <DropdownMenuRadioItem
                key={option.id}
                value={option.id}
              >
                <Icon className="text-muted-foreground size-4 shrink-0" />
                <span>{option.label}</span>
              </DropdownMenuRadioItem>
            );
          })}
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  );
};
