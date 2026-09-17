import type { FC } from "react";
import {
  BotIcon,
  BoxIcon,
  BrainCircuitIcon,
  BracesIcon,
  BugIcon,
  CalendarClockIcon,
  CheckCircle2Icon,
  CommandIcon,
  CompassIcon,
  CpuIcon,
  FileCodeIcon,
  FolderTreeIcon,
  ListTreeIcon,
  NetworkIcon,
  PackageIcon,
  PowerIcon,
  RefreshCwIcon,
  RotateCcwIcon,
  Settings2Icon,
  ShieldCheckIcon,
  SquareTerminalIcon,
  WrenchIcon,
} from "lucide-react";

import type { Command } from "@/lib/commands";
import { titleCase } from "@/lib/utils";

/**
 * The three sections, and the axis they are sorted on.
 *
 * **Provenance, not topic.** Where a command came from, rather than what it is
 * about. Topic was tried and it produced three pages holding one or two
 * commands each and a Miscellaneous page holding everything else — and
 * Miscellaneous is not a category, it is the residue that proves the axis was
 * wrong. Topic also cannot survive the list growing: commands arrive from
 * `/packages` or get written into the workspace by the agent, so both the
 * commands and their topics are unbounded, while their origins are not.
 */
export type SettingsPageId = "kernel" | "plugins" | "additional";

type SettingsPage = {
  id: SettingsPageId;
  label: string;
  description: string;
  icon: FC<{ className?: string }>;
};

export const SETTINGS_PAGES: SettingsPage[] = [
  {
    id: "kernel",
    label: "Kernel",
    description: "The commands built into Second Brain itself.",
    icon: CpuIcon,
  },
  {
    id: "plugins",
    label: "Plugins",
    description:
      "Manage the capabilities installed around the kernel, and the store they come from.",
    icon: BoxIcon,
  },
  {
    id: "additional",
    label: "Additional",
    description:
      "Commands added by installed packages or written into the workspace.",
    icon: CommandIcon,
  },
];

/**
 * Which commands a page claims, in the order it shows them.
 *
 * **One declaration for both**, because they answer to the same thing: a page
 * is a short hand-picked list, and a hand-picked list has an order. Sorted by
 * name instead, `config` — the one people actually come here for — landed
 * second, between `agent` and `debug`.
 *
 * `additional` names nothing on purpose. It is the fallthrough, and listing
 * its members is exactly what it cannot do: they arrive from the store or from
 * the agent's own hand long after this file was written.
 */
const PAGE_COMMANDS: Record<SettingsPageId, readonly string[]> = {
  kernel: [
    "config",
    "agent",
    "permissions",
    "schedule",
    "locations",
    "debug",
    "setup",
  ],
  plugins: [
    "packages",
    "scripts",
    "llm",
    "commands",
    "tools",
    "tasks",
    "services",
    "frontends",
  ],
  additional: [],
};

/** Primary commands receive a subtle fill within their settings group. */
export const FEATURED_COMMANDS: ReadonlySet<string> = new Set(["config"]);

export const SYSTEM_ACTIONS = [
  {
    name: "update",
    label: "Update",
    description: "Pull the latest changes from the repository.",
    icon: RefreshCwIcon,
    tone: "ghost" as const,
    danger: false,
  },
  {
    name: "restart",
    label: "Restart",
    description: "Restart the running application.",
    icon: RotateCcwIcon,
    tone: "ghost" as const,
    danger: false,
  },
  {
    name: "quit",
    label: "Shut down",
    description: "Stop the running application.",
    icon: PowerIcon,
    tone: "ghost" as const,
    danger: true,
  },
] as const;

export const SYSTEM_ACTION_NAMES: ReadonlySet<string> = new Set(
  SYSTEM_ACTIONS.map((action) => action.name),
);

/**
 * Commands that remain valid in chat but already have first-class controls
 * elsewhere in the UI. Repeating them here makes Settings look like an action
 * menu and gives two ways to perform the same immediate action.
 *
 * Each one has somewhere it went, and each is a better home than a card in a
 * dialog:
 *
 * - `cancel` and `new` — the composer and the header, where you already are.
 * - `clear` and `compact` — the conversation menu. Both act on the conversation
 *   on screen, and neither is a setting; nothing about them persists past the
 *   thread they are pointed at.
 * - `conversations` — the sidebar, which covers opening, renaming, filing and
 *   deleting. See `conversation-menu.tsx` for the one capability that goes with
 *   it and why that was worth trading.
 */
const DEDICATED_UI_COMMAND_NAMES: ReadonlySet<string> = new Set([
  "cancel",
  "new",
  "mode",
  "clear",
  "compact",
  "conversations",
]);

/**
 * Which section a notification is about, when it is about one at all.
 *
 * **Keyed on `source`, which is the only field here that can be trusted.** It is
 * stamped by the kernel off the live provenance chain rather than stated by
 * whoever raised the notification, so a plugin cannot route itself somewhere it
 * does not belong by claiming to be the config announcer.
 *
 * Most notifications map to nothing and that is the honest answer — a scheduled
 * agent finishing has a *conversation* to offer, not a settings page. Returning
 * null is what keeps the link off those rows rather than sending them somewhere
 * arbitrary.
 *
 * **The section, not the setting.** Settings has no per-setting address to link
 * to: a section is a page, and each individual setting is a slash command run
 * inside it. So `config` is as precise as this can honestly be, and the
 * notification's own body already names which settings changed.
 */
const NOTIFICATION_PAGES: Record<string, SettingsPageId> = {
  config: "kernel",
};

export function pageForNotification(source: string): SettingsPageId | null {
  return NOTIFICATION_PAGES[source] ?? null;
}

/**
 * The section that owns a setting `/config` will not open.
 *
 * **These have a home; it is just not `/config`.** A `hidden` declaration means
 * the setting is managed somewhere better than a generic key/value form —
 * `default_llm_profile` and `llm_profiles` through `/llm`, the agent profiles
 * through `/agent`, `frontend_profiles` through `/frontends` — and `/config`
 * leaves every one of them out of its catalogue.
 *
 * **Still worth answering even where the page is the same.** The agent
 * profiles name `kernel`, which is where `/config` lives too, so the
 * destination alone no longer distinguishes them. Being *listed here at all* is what does the
 * work: `openSetting` runs `settingCommand` only for settings this map does not
 * claim, so a hit is what keeps it from opening a `/config` form for a setting
 * `/config` has never heard of.
 *
 * Only the settings with somewhere better to be. A hidden setting managed by
 * nothing the UI shows — `scheduled_jobs`, which belongs to the timekeeper
 * service — is absent on purpose and falls back to the section, where the
 * notification body has already named it.
 */
const SETTING_PAGES: Record<string, SettingsPageId> = {
  llm_profiles: "plugins",
  default_llm_profile: "plugins",
  agent_profiles: "kernel",
  active_agent_profile: "kernel",
  frontend_profiles: "plugins",
};

export function pageForSetting(setting: string): SettingsPageId | null {
  return SETTING_PAGES[setting] ?? null;
}

/**
 * The command that opens one setting.
 *
 * `/config` takes its arguments positionally — category, then setting name — and
 * naming the setting is what makes the form skip the category and plugin steps
 * and go straight to that setting's own page. `all` is the category that does
 * not have to be right: it is the one that matches wherever the setting
 * actually lives, so this never has to work out whether something is a kernel,
 * plugin or user setting to link to it.
 *
 * Safe to interpolate only because `settingNamesOf` has already thrown away
 * anything that is not identifier-shaped. Nothing here quotes, and nothing here
 * needs to.
 */
export function settingCommand(setting: string): string {
  return `/config all ${setting}`;
}


/**
 * Which page shows a command.
 *
 * **Anything unclaimed is Additional, and that is the rule rather than a
 * default.** The kernel does not say where a command came from — `command.list`
 * answers name, description and category, and nothing about the file it was
 * loaded from — so "not one of the ones we named" is the only provenance signal
 * available. It is the right one for the question: the commands this file
 * cannot enumerate are exactly the ones a package or the agent installed.
 *
 * The cost is that a *new kernel* command lands on Additional until it is added
 * to `PAGE_COMMANDS` above. That is the same upkeep `COMMAND_PRESENTATION`
 * already asks for, and it fails by showing the command in the wrong section
 * rather than by not showing it.
 */
export function pageForCommand(name?: string | null): SettingsPageId {
  if (!name || SYSTEM_ACTION_NAMES.has(name)) return "additional";
  for (const page of SETTINGS_PAGES) {
    if (PAGE_COMMANDS[page.id].includes(name)) return page.id;
  }
  return "additional";
}

export function commandsForPage(
  commands: Command[],
  page: SettingsPageId,
): Command[] {
  const order = PAGE_COMMANDS[page];
  return commands
    .filter((command) => {
      if (SYSTEM_ACTION_NAMES.has(command.name)) return false;
      if (DEDICATED_UI_COMMAND_NAMES.has(command.name.toLowerCase()))
        return false;
      return pageForCommand(command.name) === page;
    })
    // **Additional keeps the order it arrived in, and relies on this to.**
    // Its list is empty, so every `indexOf` is -1, every comparison is 0, and a
    // stable sort leaves `listCommands`' own sort by name standing — which is
    // the only sensible order for a page whose contents nobody here has seen.
    .sort((a, b) => order.indexOf(a.name) - order.indexOf(b.name));
}

const COMMAND_PRESENTATION: Record<
  string,
  { title: string; detail: string; icon: FC<{ className?: string }> }
> = {
  llm: {
    title: "Language models",
    detail: "Choose, edit, load, or remove model profiles.",
    icon: BrainCircuitIcon,
  },
  agent: {
    title: "Agent profiles",
    detail: "Select the tools, model, and behavior used for a session.",
    icon: BotIcon,
  },
  permissions: {
    title: "Standing permissions",
    detail: "Review and withdraw permissions granted previously.",
    icon: ShieldCheckIcon,
  },
  commands: {
    title: "Commands",
    detail: "Review commands provided by the kernel and installed packages.",
    icon: BracesIcon,
  },
  tools: {
    title: "Tools",
    detail: "Inspect available tools and run one directly.",
    icon: WrenchIcon,
  },
  tasks: {
    title: "Tasks",
    detail: "Pause, resume, retry, reset, or trigger tasks.",
    icon: CheckCircle2Icon,
  },
  services: {
    title: "Services",
    detail: "Load and manage persistent capabilities.",
    icon: NetworkIcon,
  },
  frontends: {
    title: "Frontends",
    detail: "Enable or disable connected interaction surfaces.",
    icon: ListTreeIcon,
  },
  config: {
    title: "Open configuration",
    detail: "Browse kernel, plugin, and user settings.",
    icon: Settings2Icon,
  },
  packages: {
    title: "Manage packages",
    detail: "Browse, install, or uninstall store packages by category.",
    icon: PackageIcon,
  },
  scripts: {
    title: "Scripts",
    detail: "Browse and run scripts available in the workspace.",
    icon: FileCodeIcon,
  },
  schedule: {
    title: "Scheduled jobs",
    detail: "Manage background agents and recurring pipeline tasks.",
    icon: CalendarClockIcon,
  },
  locations: {
    title: "Locations",
    detail: "Show the project and plugin directories on disk.",
    icon: FolderTreeIcon,
  },
  debug: {
    title: "Debug",
    detail: "Inspect the live session, model, and recent log errors.",
    icon: BugIcon,
  },
  setup: {
    title: "Setup",
    detail: "Install a starter bundle, then configure an LLM and frontend.",
    icon: CompassIcon,
  },
};

export function commandPresentation(command: Command) {
  return (
    COMMAND_PRESENTATION[command.name] ?? {
      title: titleCase(command.name),
      detail: command.description || "Run this Second Brain command.",
      icon: SquareTerminalIcon,
    }
  );
}
