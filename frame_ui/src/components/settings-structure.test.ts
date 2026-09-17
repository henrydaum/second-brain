import { describe, expect, it } from "vitest";
import { FileCodeIcon } from "lucide-react";

import {
  commandPresentation,
  commandsForPage,
  pageForCommand,
} from "@/components/settings-structure";
import type { Command } from "@/lib/commands";

const named = (...names: string[]): Command[] =>
  names.map((name) => ({ name }));

describe("commandsForPage", () => {
  it("omits commands that already have dedicated conversation controls", () => {
    const commands = named(
      "cancel",
      "new",
      "mode",
      "clear",
      "compact",
      "conversations",
      "reveal",
    );

    expect(
      commandsForPage(commands, "additional").map((command) => command.name),
    ).toEqual(["reveal"]);
  });

  it("orders a page by its own list rather than by name", () => {
    // The catalogue arrives sorted by name, which puts `config` — the command
    // the page exists for — second, between `agent` and `debug`.
    const commands = named("agent", "config", "debug", "setup");

    expect(
      commandsForPage(commands, "kernel").map((command) => command.name),
    ).toEqual(["config", "agent", "debug", "setup"]);
  });

  it("leaves Additional in the order the catalogue supplied", () => {
    // Nothing here is named in `PAGE_COMMANDS`, so the sort has no opinion and
    // must not invent one — `listCommands` already sorted these by name.
    const commands = named("alpha", "beta", "gamma");

    expect(
      commandsForPage(commands, "additional").map((command) => command.name),
    ).toEqual(["alpha", "beta", "gamma"]);
  });
});

describe("pageForCommand", () => {
  it("sends a command no page claims to Additional", () => {
    // The whole third section is this rule: the kernel does not report where a
    // command came from, so "unclaimed" is how an installed or agent-written
    // one is recognised.
    expect(pageForCommand("something-a-package-installed")).toBe("additional");
  });

  it("keeps the kernel and plugin commands off it", () => {
    expect(pageForCommand("config")).toBe("kernel");
    expect(pageForCommand("setup")).toBe("kernel");
    expect(pageForCommand("packages")).toBe("plugins");
    expect(pageForCommand("scripts")).toBe("plugins");
    expect(pageForCommand("frontends")).toBe("plugins");
    // A capability like the rest of that page, and the kernel agrees — `/llm`
    // declares `category = "Capabilities"` alongside tools and services.
    expect(pageForCommand("llm")).toBe("plugins");
  });
});

describe("commandPresentation", () => {
  it("gives Scripts a dedicated code-file icon", () => {
    expect(commandPresentation({ name: "scripts" })).toMatchObject({
      title: "Scripts",
      icon: FileCodeIcon,
    });
  });
});
