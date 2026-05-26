// Tutorial carousel. Mounted into the empty-state hero on first canvas, and
// re-built inside the Help modal when the user clicks the header `?` button.
// Same data drives both, but each mount gets its own DOM + state so they
// don't fight over a single nodeset.

(function () {
  const STEPS = [
    {
      title: "Ask for an image.",
      body: "Type a request into the chat and Second Brain will render an image.",
      note: "This can take a little while. Don't get discouraged."
    },
    {
      title: "Build on it.",
      body: "Keep chatting to add more layers. Ask to change something, if it isn't right.",
      note: "Second Brain can make mistakes, but it can also fix them."
    },
    {
      title: "Fine-tune the layers.",
      body: "Press the settings (gear) icon to open the layer panel. Adjust a control, then press Regenerate to apply it.",
      note: "Every layer has its own settings. Try changing them to get different effects."
    },
    {
      title: "Manual controls.",
      body: "Type into the text box with the settings panel open to search skills. Press \"+\" to add a layer, use the \"-\" and three-dots buttons to delete or move a layer.",
      note: "Useful if you prefer manual controls instead of chatting."
    },
    {
      title: "Save, download, or share.",
      body: "Use the buttons above the canvas to keep the image, export a PNG, or send it to someone.",
      note: "Anything you share can be opened and remixed by anyone who sees it — they get the layers and settings, but your personal canvas remains unaffected."
    },
    {
      title: "Make an account.",
      body: "Sign in to keep your saved canvases and unlock Second Brain's code-writing abilities.",
      note: "Second Brain uses code to generate images. Writing new code takes longer because Second Brain tests it and checks for bugs."
    },
    {
      title: "Add more credits.",
      body: "Need more usage? Top up with extra credits — $2.99 per pack.",
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
      slideWrap.innerHTML = "";
      slideWrap.appendChild(el("div", { class: "tutorial-step" }, "Step " + (idx + 1) + " of " + STEPS.length));
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

      dotsWrap.innerHTML = "";
      for (let i = 0; i < STEPS.length; i++) {
        const dot = el("button", {
          type: "button",
          class: "tutorial-dot" + (i === idx ? " active" : ""),
          "aria-label": "Go to step " + (i + 1),
          onclick: () => { idx = i; render(); }
        });
        dotsWrap.appendChild(dot);
      }
      counter.textContent = (idx + 1) + " / " + STEPS.length;
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
