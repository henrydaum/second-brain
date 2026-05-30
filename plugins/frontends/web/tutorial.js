// Tutorial carousel. Mounted into the empty-state hero on first canvas, and
// re-built inside the Help modal when the user clicks the header `?` button.
// Same data drives both, but each mount gets its own DOM + state so they
// don't fight over a single nodeset.

(function () {
  const STEPS = [
    {
      title: "Ask for an image.",
      body: "Type a request into the chat and Second Brain will render an image.",
      note: "This can take a little while. Press the '+ New canvas' button to retry."
    },
    {
      title: "Build on it.",
      body: "Keep chatting to add more layers. Ask to change something if you don't like it.",
      note: "Second Brain can make mistakes, but it can also fix them."
    },
    {
      title: "Fine-tune the layers.",
      body: "Press the gear icon ⚙ to open the control panel. To adjust a setting, make the change and press 'Regenerate' to apply it.",
      note: "Every layer has its own settings. Try changing them to get different effects."
    },
    {
      title: "Manual controls.",
      body: "Type into the text box within the control panel to search techniques. Press \"+\" to add a layer, \"-\" to delete a layer, and the arrows ↕ to move a layer in the stack.",
      note: "Useful if you prefer manual controls instead of chatting."
    },
    {
      title: "Save, download, or share.",
      body: "Use the buttons above the canvas to keep the image, export a PNG, or send it to a friend.",
      note: "Anything you share can be opened and remixed by anyone who sees it — they get the layers and settings, but your personal canvas remains unaffected."
    },
    {
      title: "Make an account.",
      body: "Sign in to keep your saved canvases and unlock Second Brain's code-writing abilities. This unlocks unparallel customizability.",
      note: "Second Brain uses code to generate images. Writing new code typically takes longer because Second Brain tests it and checks for bugs."
    },
    {
      title: "Add more credits.",
      body: "Need more usage? Top up with extra credits from your account page.",
      note: "Usage limits keep things fair for everyone."
    }
  ];

  const SUGGESTION_COUNT = 3;
  function pickSuggestions() {
    const pool = (typeof PROMPT_SUGGESTIONS !== "undefined" && Array.isArray(PROMPT_SUGGESTIONS)) ? PROMPT_SUGGESTIONS.slice() : [];
    for (let i = pool.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [pool[i], pool[j]] = [pool[j], pool[i]];
    }
    return pool.slice(0, SUGGESTION_COUNT);
  }

  function el(tag, attrs, ...children) {
    const node = document.createElement(tag);
    if (attrs) {
      for (const k in attrs) {
        if (k === "class") node.className = attrs[k];
        else if (k === "text") node.textContent = attrs[k];
        else if (k.startsWith("on") && typeof attrs[k] === "function") node.addEventListener(k.slice(2), attrs[k]);
        else node.setAttribute(k, attrs[k]);
      }
    }
    for (const c of children) {
      if (c == null) continue;
      node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    }
    return node;
  }

  // Steps 1–CORE_STEPS are the main tutorial; anything after is a "bonus"
  // section the visitor doesn't need to read to get going. The carousel
  // still flows continuously, but the label + counter + a visual gap in
  // the dot row make the split obvious.
  const CORE_STEPS = 5;

  function buildTutorial(host, opts) {
    opts = opts || {};
    host.innerHTML = "";
    host.classList.add("tutorial-host");
    const container = el("div", { class: "tutorial" });
    host.appendChild(container);

    let idx = 0;
    const chipChoices = pickSuggestions();

    const slideWrap = el("div", { class: "tutorial-slide" });
    const dotsWrap = el("div", { class: "tutorial-dots" });
    const prevBtn = el("button", { type: "button", class: "tutorial-nav-btn", "aria-label": "Previous step" }, "‹");
    const nextBtn = el("button", { type: "button", class: "tutorial-nav-btn", "aria-label": "Next step" }, "›");
    const counter = el("span", { class: "tutorial-counter" });

    function render() {
      const s = STEPS[idx];
      const isBonus = idx >= CORE_STEPS;
      const bonusCount = STEPS.length - CORE_STEPS;
      const stepLabel = isBonus
        ? "Bonus " + (idx - CORE_STEPS + 1) + " of " + bonusCount
        : "Step " + (idx + 1) + " of " + CORE_STEPS;
      slideWrap.innerHTML = "";
      slideWrap.appendChild(el("div", { class: "tutorial-step" + (isBonus ? " bonus" : "") }, stepLabel));
      slideWrap.appendChild(el("h2", { class: "tutorial-title" }, s.title));
      slideWrap.appendChild(el("p", { class: "tutorial-body" }, s.body));
      if (s.note) slideWrap.appendChild(el("p", { class: "tutorial-note" }, "Note: " + s.note));
      if (idx === 0 && chipChoices.length) {
        const chips = el("div", { class: "tutorial-chips" });
        for (const c of chipChoices) {
          chips.appendChild(el("button", {
            type: "button",
            class: "tutorial-chip",
            "data-prompt": c.prompt,
            onclick: () => opts.onTryIt && opts.onTryIt(c.prompt)
          }, c.label));
        }
        slideWrap.appendChild(chips);
      }
      // Step 4 (Manual controls): one demo chip that opens the controls
      // drawer and pre-fills the search input so the visitor can see what
      // technique search looks like without having to figure out where to type.
      if (idx === 3 && opts.onSearchDemo) {
        const chips = el("div", { class: "tutorial-chips" });
        chips.appendChild(el("button", {
          type: "button",
          class: "tutorial-chip",
          onclick: () => opts.onSearchDemo("Koch")
        }, "Try: search \"Koch\""));
        slideWrap.appendChild(chips);
      }

      dotsWrap.innerHTML = "";
      for (let i = 0; i < STEPS.length; i++) {
        const classes = ["tutorial-dot"];
        if (i === idx) classes.push("active");
        if (i >= CORE_STEPS) classes.push("bonus");
        // First bonus dot gets a wider gap so the split reads visually.
        if (i === CORE_STEPS) classes.push("bonus-start");
        const dot = el("button", {
          type: "button",
          class: classes.join(" "),
          "aria-label": (i >= CORE_STEPS ? "Go to bonus " + (i - CORE_STEPS + 1) : "Go to step " + (i + 1)),
          onclick: () => { idx = i; render(); }
        });
        dotsWrap.appendChild(dot);
      }
      counter.textContent = isBonus
        ? "Bonus " + (idx - CORE_STEPS + 1) + " / " + bonusCount
        : (idx + 1) + " / " + CORE_STEPS;
      prevBtn.disabled = idx === 0;
      nextBtn.disabled = idx === STEPS.length - 1;
    }

    prevBtn.addEventListener("click", () => { if (idx > 0) { idx--; render(); } });
    nextBtn.addEventListener("click", () => { if (idx < STEPS.length - 1) { idx++; render(); } });

    const controls = el("div", { class: "tutorial-controls" }, prevBtn, dotsWrap, nextBtn);
    const footer = el("div", { class: "tutorial-footer" }, counter);

    container.appendChild(slideWrap);
    container.appendChild(controls);
    container.appendChild(footer);

    function onKey(e) {
      if (!container.isConnected) { document.removeEventListener("keydown", onKey); return; }
      if (e.key === "ArrowLeft" && idx > 0) { idx--; render(); }
      else if (e.key === "ArrowRight" && idx < STEPS.length - 1) { idx++; render(); }
    }
    document.addEventListener("keydown", onKey);

    render();
    return { destroy() { document.removeEventListener("keydown", onKey); host.innerHTML = ""; host.classList.remove("tutorial-host"); } };
  }

  window.SBTutorial = { build: buildTutorial, steps: STEPS };
})();
