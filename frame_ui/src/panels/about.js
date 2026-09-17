/**
 * The proving panel.
 *
 * It exists to show that a slot can hold something that is not an iframe, and
 * to be the shortest possible example of the contract: one default export
 * taking `(host, element)`, returning something with `destroy`.
 *
 * Note what it does *not* do. It does not import the frame, the config, the
 * layouts or another panel. It does not reach for `document.querySelector`
 * outside its own element. It asks the kernel through `host.call`, exactly as a
 * widget asks through `brain.call`, and it is told its size and scheme rather
 * than measuring the window or asking the operating system. Every one of those
 * is a rule a widget could not break if it tried; a panel can, which is why
 * they are written out here and checked by `scripts/check-panels.mjs`.
 *
 * It is styled with the frame's tokens and no stylesheet of its own. A panel
 * shares the frame's document, so it must not introduce global rules — the
 * whole point of the token table is that it can style itself without them.
 */

export default function about(host, element) {
  element.replaceChildren();

  const root = document.createElement("div");
  root.style.cssText = `
    display: flex; flex-direction: column; gap: 0.75rem;
    height: 100%; padding: var(--sb-space); overflow: auto;
    font-size: var(--sb-text);`;

  const title = document.createElement("h2");
  title.textContent = "Second Brain";
  title.style.cssText = "margin: 0; font-size: 1.0625rem; font-weight: 600;";

  const facts = document.createElement("dl");
  facts.style.cssText = `
    display: grid; grid-template-columns: auto 1fr; gap: 0.25rem 1rem;
    margin: 0; font-size: var(--sb-text-sm);`;

  const rows = new Map();
  for (const label of ["Kind", "Box", "Scheme", "Workspace"]) {
    const term = document.createElement("dt");
    term.textContent = label;
    term.style.color = "var(--sb-muted)";
    const value = document.createElement("dd");
    value.style.margin = "0";
    value.textContent = "…";
    facts.append(term, value);
    rows.set(label, value);
  }
  rows.get("Kind").textContent = "Panel (built in, not sandboxed)";

  root.append(title, facts);
  element.append(root);

  const draw = () => {
    rows.get("Box").textContent = `${host.size.width} × ${host.size.height}`;
    rows.get("Scheme").textContent = host.scheme;
  };
  draw();

  const offSize = host.on("size", draw);
  const offScheme = host.on("scheme", draw);

  // One real Request, to prove the host reaches the kernel by the same route a
  // widget's `brain.call` does.
  let live = true;
  host.call("paths.get", { name: "workspace" }).then(
    (path) => { if (live) rows.get("Workspace").textContent = String(path); },
    // A refusal is an ordinary answer here, not a reason for the panel to be
    // broken. It says so and carries on.
    (error) => { if (live) rows.get("Workspace").textContent = error.message; },
  );

  return {
    destroy() {
      live = false;
      offSize();
      offScheme();
    },
  };
}
