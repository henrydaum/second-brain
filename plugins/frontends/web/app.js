const sid = localStorage.sbDemoSession || (localStorage.sbDemoSession = crypto.randomUUID());
// Mirror sid into a cookie so /files img requests carry the identity
// (image <img src=...> requests can't easily set query params, but they
// send cookies automatically). Tomorrow's real auth replaces this.
document.cookie = `sb_sid=${encodeURIComponent(sid)}; path=/; SameSite=Strict; max-age=31536000`;
const messages = document.querySelector("#messages");
const form = document.querySelector("#chatForm");
const input = document.querySelector("#chatInput");
const showcase = document.querySelector(".showcase");
const heroImage = document.querySelector("#heroImage");
const downloadImage = document.querySelector("#downloadImage");
const regenerateBtn = document.querySelector("#regenerateImage");
const shareBtn = document.querySelector("#shareImage");
const sharePanel = document.querySelector("#sharePanel");
const shareTitle = document.querySelector("#shareTitle");
const shareArtist = document.querySelector("#shareArtist");
const gallery = document.querySelector("#gallery");
const paginator = document.querySelector("#paginator");
const controlsPanel = document.querySelector("#controlsPanel");
const controlsDrawer = document.querySelector("#controlsDrawer");
const controlsToggle = document.querySelector("#controlsToggle");
const controlsCountEl = document.querySelector("#controlsCount");
const layersStack = document.querySelector("#layersStack");
const emptyState = document.querySelector("#emptyState");
const NEAR_BOTTOM_PX = 80;
const GALLERY_PAGE = 18;
let palettesCache = [];
let currentControlsPanels = [];
let galleryPage = 1;
const controlDebounce = new Map();
const toolStatusEls = new Map();
const collapsedSkills = new Set(JSON.parse(localStorage.sbCollapsedSkills || "[]"));
const atBottom = () => messages.scrollHeight - messages.scrollTop - messages.clientHeight < NEAR_BOTTOM_PX;
const bottom = (force = false) => {
  const stick = force || atBottom();
  if (stick) requestAnimationFrame(() => messages.scrollTop = messages.scrollHeight);
};
const add = (role, text, useMd = false) => {
  const el = document.createElement("article");
  el.className = role;
  if (useMd) el.innerHTML = mdToHtml(text);
  else el.textContent = text;
  messages.appendChild(el);
  bottom();
  return el;
};
function mdToHtml(src) {
  let s = String(src ?? "").replace(/[&<>]/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[m]));
  s = s.replace(/```([\s\S]*?)```/g, (_, c) => `<pre><code>${c.replace(/^\n/, "")}</code></pre>`);
  s = s.replace(/`([^`\n]+)`/g, '<code>$1</code>');
  s = s.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  s = s.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
  s = s.replace(/__([^_\n]+)__/g, '<strong>$1</strong>');
  s = s.replace(/(^|[\s(])\*([^*\n]+)\*(?=[\s).,!?;:]|$)/g, '$1<em>$2</em>');
  s = s.replace(/(^|[\s(])_([^_\n]+)_(?=[\s).,!?;:]|$)/g, '$1<em>$2</em>');
  s = s.replace(/^(#{1,3})\s+(.+)$/gm, (_, h, t) => `<h${h.length + 2}>${t}</h${h.length + 2}>`);
  s = s.replace(/(?:^|\n)((?:- .+(?:\n|$))+)/g, m => {
    const items = m.trim().split(/\n/).map(l => `<li>${l.replace(/^- /, "")}</li>`).join("");
    return `\n<ul>${items}</ul>`;
  });
  return s;
}
async function post(url, body = {}) {
  const res = await fetch(url, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({session_id:sid, ...body})});
  return res.json();
}
async function poll() { try { render((await fetch(`/api/events?session_id=${encodeURIComponent(sid)}`).then(r => r.json())).events); } catch {} }
async function get(url) { return fetch(`${url}${url.includes("?") ? "&" : "?"}session_id=${encodeURIComponent(sid)}`).then(r => r.json()); }
function approval(ev) {
  const el = document.createElement("article");
  el.className = "assistant";
  el.textContent = `${ev.title}\n${ev.body}`;
  const actions = document.createElement("div");
  actions.className = "approval-actions";
  for (const [label, value, cls] of [["Approve", true, "approve"], ["Deny", false, "deny"]]) {
    const btn = document.createElement("button");
    btn.type = "button"; btn.className = cls; btn.textContent = label;
    btn.onclick = async () => { actions.querySelectorAll("button").forEach(b => b.disabled = true); render((await post("/api/approval", {value})).events); };
    actions.appendChild(btn);
  }
  el.appendChild(actions);
  messages.appendChild(el);
  bottom();
}
function renderToolStatus(ev) {
  const id = ev.call_id || `${ev.name}:${Date.now()}`;
  // Loader tracking — render-class tool statuses keep the loader alive.
  if (ev.status === "started") loaderToolStart(id);
  else if (ev.status === "finished") loaderToolEnd(id);
  if (ev.progress && typeof ev.progress.total === "number") {
    setProgress(ev.progress.done | 0, ev.progress.total | 0);
  }
  // "progressed" events without a chat counterpart shouldn't render a chat line —
  // they're for the canvas arc only.
  if (ev.status === "progressed" && !toolStatusEls.has(id)) { bottom(); return; }
  let el = toolStatusEls.get(id);
  if (!el) {
    el = document.createElement("article");
    el.className = "status tool-status running";
    messages.appendChild(el);
    toolStatusEls.set(id, el);
  }
  const finished = ev.status === "finished";
  const failed = finished && (ev.ok === false || !!ev.error);
  const glyph = failed ? "✕" : finished ? "✓" : "";
  el.classList.toggle("running", !finished);
  el.classList.toggle("done", finished && !failed);
  el.classList.toggle("failed", failed);
  el.textContent = `${glyph} ${ev.name}${failed && ev.error ? ` — ${ev.error}` : ""}`;
  if (finished) toolStatusEls.delete(id);
  bottom();
}
function render(events) {
  for (const ev of events || []) {
    if (ev.type === "message") add("assistant", ev.content, true);
    else if (ev.type === "status") add("status", ev.content);
    else if (ev.type === "tool_status") renderToolStatus(ev);
    else if (ev.type === "error") { add("error", ev.content); loaderForceStop(); }
    else if (ev.type === "form") add("assistant", `${ev.form?.display?.prompt || "Input required"}\n${(ev.form?.display?.choices || []).map(c => c.label || c.value).join(" / ")}`);
    else if (ev.type === "approval") approval(ev);
    else if (ev.type === "paywall") openPaywall(ev);
    else if (ev.type === "account") setAccount(ev.account);
    else if (ev.type === "hero_image") {
      setCanvas(ev.canvas || {url: ev.url, name: ev.name});
      loaderForceStop();
    }
    else if (ev.type === "canvas_reset") { setCanvas(null); loaderForceStop(); }
    else if (ev.type === "shared") { sharePanel.hidden = true; loadGallery(1); }
    else if (ev.type === "attachment") add("assistant", `Attachment: ${ev.name}`);
  }
  bottom();
}

// ----- account + paywall + auth -----
const accountChip = document.querySelector("#accountChip");
const signInBtn = document.querySelector("#signInBtn");
const accountLink = document.querySelector("#accountLink");
const logoutLink = document.querySelector("#logoutLink");
const paywallModal = document.querySelector("#paywallModal");
const signinModal = document.querySelector("#signinModal");
const promoModal = document.querySelector("#promoModal");
const buyBtn = document.querySelector("#buyBtn");
const signinForm = document.querySelector("#signinForm");
const signinEmail = document.querySelector("#signinEmail");
const signinStatus = document.querySelector("#signinStatus");
const promoForm = document.querySelector("#promoForm");
const promoCode = document.querySelector("#promoCode");
const promoStatus = document.querySelector("#promoStatus");

function openModal(el) { el.hidden = false; }
function closeModal(el) { el.hidden = true; }
document.querySelectorAll("[data-close]").forEach(b => b.addEventListener("click", () => closeModal(document.getElementById(b.dataset.close))));
[paywallModal, signinModal, promoModal].forEach(m => m.addEventListener("click", e => { if (e.target === m) closeModal(m); }));

function setAccount(acc) {
  if (acc?.signed_in) {
    const credits = acc.tier === "unlimited" ? "∞ messages" : `${(acc.credits || 0).toLocaleString()} left`;
    accountChip.textContent = `${acc.email} · ${credits}`;
    accountChip.hidden = false;
    signInBtn.hidden = true;
    accountLink.hidden = false;
    logoutLink.hidden = false;
  } else {
    accountChip.hidden = true;
    signInBtn.hidden = false;
    accountLink.hidden = true;
    logoutLink.hidden = true;
  }
}
async function refreshAccount() {
  try { const r = await get("/api/account"); setAccount(r.account); } catch {}
}
function openPaywall(ev) {
  const card = paywallModal.querySelector(".modal-card");
  const lead = card.querySelector(".modal-lead");
  if (ev?.price_cents && ev?.credits) {
    lead.innerHTML = `Keep going for <strong>$${(ev.price_cents/100).toFixed(2)}</strong> — get <strong>${ev.credits.toLocaleString()} messages</strong>.`;
  }
  openModal(paywallModal);
}
buyBtn.addEventListener("click", async () => {
  buyBtn.disabled = true;
  buyBtn.textContent = "Opening Stripe…";
  try {
    const r = await post("/api/checkout");
    if (r.ok && r.url) { window.location = r.url; return; }
    buyBtn.textContent = "Buy $2.99";
    buyBtn.disabled = false;
    add("error", r.error || "Could not start checkout.");
  } catch (e) {
    buyBtn.textContent = "Buy $2.99";
    buyBtn.disabled = false;
    add("error", e.message);
  }
});
signInBtn.addEventListener("click", () => openModal(signinModal));
document.querySelector("#signInFromPaywall").addEventListener("click", () => { closeModal(paywallModal); openModal(signinModal); });
document.querySelector("#promoLink").addEventListener("click", () => { closeModal(paywallModal); openModal(promoModal); });
signinForm.addEventListener("submit", async e => {
  e.preventDefault();
  signinStatus.hidden = true;
  const btn = signinForm.querySelector("button");
  btn.disabled = true;
  try {
    const r = await post("/api/auth/request", {email: signinEmail.value});
    signinStatus.hidden = false;
    signinStatus.className = "modal-status " + (r.ok ? "ok" : "err");
    signinStatus.textContent = r.ok
      ? (r.delivered ? "Check your inbox for the sign-in link." : "Link generated — see server logs (email not configured).")
      : (r.error || "Could not send link.");
  } catch (err) {
    signinStatus.hidden = false; signinStatus.className = "modal-status err"; signinStatus.textContent = err.message;
  } finally { btn.disabled = false; }
});
promoForm.addEventListener("submit", async e => {
  e.preventDefault();
  promoStatus.hidden = true;
  const btn = promoForm.querySelector("button");
  btn.disabled = true;
  try {
    const r = await post("/api/promo/redeem", {code: promoCode.value});
    promoStatus.hidden = false;
    promoStatus.className = "modal-status " + (r.ok ? "ok" : "err");
    promoStatus.textContent = r.ok
      ? `Redeemed — granted ${r.granted}.`
      : (r.error || "Could not redeem code.");
    if (r.ok) { refreshAccount(); setTimeout(() => closeModal(promoModal), 1200); }
    if (!r.ok && r.need_auth) { closeModal(promoModal); openModal(signinModal); }
  } catch (err) {
    promoStatus.hidden = false; promoStatus.className = "modal-status err"; promoStatus.textContent = err.message;
  } finally { btn.disabled = false; }
});

// Surface checkout return-state from query string.
(() => {
  const params = new URLSearchParams(window.location.search);
  if (params.get("checkout") === "success") {
    add("status", "Payment confirmed. Welcome back.");
    history.replaceState({}, "", "/");
  } else if (params.get("checkout") === "cancel") {
    add("status", "Checkout canceled.");
    history.replaceState({}, "", "/");
  } else if (params.get("checkout") === "pending") {
    add("status", "Payment is processing — your credits will appear shortly.");
    history.replaceState({}, "", "/");
  }
})();

// ----- Canvas + theming -----
function setCanvas(c) {
  renderControlsPanel(c?.controls_panels || []);
  renderLayersStack(c?.layers || []);
  if (!c?.url) {
    showcase.classList.remove("has-image");
    heroImage.classList.remove("fading");
    heroImage.removeAttribute("src");
    downloadImage.href = "#";
    resetAccents();
    return;
  }
  const newUrl = c.url, newName = c.name || "canvas.png";
  const apply = () => {
    heroImage.src = newUrl; heroImage.alt = newName;
    downloadImage.href = newUrl; downloadImage.download = newName;
    showcase.classList.add("has-image");
  };
  if (!showcase.classList.contains("has-image")) { apply(); heroImage.addEventListener("load", () => applyAccents(heroImage), {once: true}); loadGallery(1); return; }
  // Preload the new image, then crossfade out → swap → crossfade in.
  const pre = new Image();
  pre.crossOrigin = "anonymous";
  pre.onload = () => {
    heroImage.classList.add("fading");
    setTimeout(() => {
      apply();
      heroImage.addEventListener("load", () => applyAccents(heroImage), {once: true});
      requestAnimationFrame(() => requestAnimationFrame(() => heroImage.classList.remove("fading")));
      loadGallery(galleryPage);
    }, 280);
  };
  pre.onerror = () => { apply(); loadGallery(galleryPage); };
  pre.src = newUrl;
}

// ----- Dynamic accent extraction -----
const DEFAULT_ACCENT = "#3df2ff";
const DEFAULT_ACCENT_2 = "#ff4d8d";
function applyAccents(imgEl) {
  try {
    const c = document.createElement("canvas");
    c.width = 32; c.height = 32;
    const ctx = c.getContext("2d");
    ctx.drawImage(imgEl, 0, 0, 32, 32);
    const data = ctx.getImageData(0, 0, 32, 32).data;
    const bins = new Map(); // hue bucket (12°) → {weight, h, s, l}
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i], g = data[i+1], b = data[i+2];
      const [h, s, l] = rgbToHsl(r, g, b);
      if (s < 0.28 || l < 0.25 || l > 0.78) continue;
      const bucket = Math.floor(h / 12);
      const w = s * (1 - Math.abs(l - 0.55) * 1.4);
      const cur = bins.get(bucket) || {weight: 0, h: 0, s: 0, l: 0, count: 0};
      cur.weight += w; cur.h += h * w; cur.s += s * w; cur.l += l * w; cur.count += 1;
      bins.set(bucket, cur);
    }
    if (!bins.size) { resetAccents(); return; }
    const sorted = [...bins.values()].sort((a, b) => b.weight - a.weight);
    const a1 = avgBin(sorted[0]);
    // Find a second bin > 60° away on the hue wheel
    const a2 = avgBin(sorted.find(b => Math.min(Math.abs(b.h/b.weight - a1.h), 360 - Math.abs(b.h/b.weight - a1.h)) > 60) || sorted[1] || sorted[0]);
    const accent = hslToHex(a1.h, Math.min(1, a1.s * 1.15), Math.min(0.7, Math.max(0.5, a1.l)));
    const accent2 = hslToHex(a2.h, Math.min(1, a2.s * 1.15), Math.min(0.7, Math.max(0.5, a2.l)));
    document.documentElement.style.setProperty("--accent", accent);
    document.documentElement.style.setProperty("--accent-2", accent2);
    document.documentElement.style.setProperty("--accent-glow", hexWithAlpha(accent, 0.22));
    document.documentElement.style.setProperty("--accent-2-glow", hexWithAlpha(accent2, 0.32));
  } catch (e) {
    // Likely a canvas taint (cross-origin) — silently fall back to defaults.
    resetAccents();
  }
}
function avgBin(b) { return {h: b.h / b.weight, s: b.s / b.weight, l: b.l / b.weight}; }
function resetAccents() {
  document.documentElement.style.removeProperty("--accent");
  document.documentElement.style.removeProperty("--accent-2");
  document.documentElement.style.removeProperty("--accent-glow");
  document.documentElement.style.removeProperty("--accent-2-glow");
}
function rgbToHsl(r, g, b) {
  r/=255; g/=255; b/=255;
  const max = Math.max(r,g,b), min = Math.min(r,g,b);
  let h = 0, s = 0; const l = (max+min)/2;
  if (max !== min) {
    const d = max - min;
    s = l > 0.5 ? d/(2-max-min) : d/(max+min);
    switch (max) { case r: h = (g-b)/d + (g<b?6:0); break; case g: h = (b-r)/d + 2; break; case b: h = (r-g)/d + 4; break; }
    h *= 60;
  }
  return [h, s, l];
}
function hslToHex(h, s, l) {
  const c = (1 - Math.abs(2*l - 1)) * s;
  const x = c * (1 - Math.abs(((h/60) % 2) - 1));
  const m = l - c/2;
  let r=0,g=0,b=0;
  if (h < 60) [r,g,b] = [c,x,0];
  else if (h < 120) [r,g,b] = [x,c,0];
  else if (h < 180) [r,g,b] = [0,c,x];
  else if (h < 240) [r,g,b] = [0,x,c];
  else if (h < 300) [r,g,b] = [x,0,c];
  else [r,g,b] = [c,0,x];
  const to = v => Math.round((v+m)*255).toString(16).padStart(2,"0");
  return `#${to(r)}${to(g)}${to(b)}`;
}
function hexWithAlpha(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n>>16)&255},${(n>>8)&255},${n&255},${a})`;
}

async function loadCanvas() { const r = await get("/api/canvas"); setCanvas(r.canvas); }

// ----- Gallery + pagination -----
async function loadGallery(page = 1) {
  galleryPage = Math.max(1, page);
  const offset = (galleryPage - 1) * GALLERY_PAGE;
  const r = await get(`/api/gallery?limit=${GALLERY_PAGE}&offset=${offset}`);
  const items = Array.isArray(r.items) ? r.items : (Array.isArray(r) ? r : []);
  const total = typeof r.total === "number" ? r.total : items.length + offset;
  gallery.innerHTML = items.map(x => `<article class="gallery-card"><img src="${x.url}" alt=""><div><strong>${esc(x.title)}</strong><small>${esc(x.artist)}${x.score ? ` · ${(x.score*100).toFixed(0)}% similar` : ""}</small><button data-path="${esc(x.path)}">Remix</button></div></article>`).join("") || "<article class='assistant'>No shared canvases yet.</article>";
  renderPaginator(total);
}
function renderPaginator(total) {
  const pages = Math.max(1, Math.ceil(total / GALLERY_PAGE));
  if (pages <= 1) { paginator.hidden = true; paginator.innerHTML = ""; return; }
  paginator.hidden = false;
  const cur = galleryPage;
  const btns = [];
  btns.push(`<button type="button" data-page="${cur-1}" ${cur===1?"disabled":""}>‹</button>`);
  // Window of pages around current; always show 1 and last.
  const set = new Set([1, pages, cur-1, cur, cur+1]);
  for (let p = 1; p <= pages; p++) {
    if (set.has(p)) btns.push(`<button type="button" data-page="${p}" class="${p===cur?"active":""}">${p}</button>`);
    else if (p === 2 || p === pages-1) btns.push(`<button type="button" disabled>…</button>`);
  }
  btns.push(`<button type="button" data-page="${cur+1}" ${cur===pages?"disabled":""}>›</button>`);
  paginator.innerHTML = btns.join("");
}
paginator.addEventListener("click", e => {
  const b = e.target.closest("button[data-page]");
  if (!b || b.disabled) return;
  const p = +b.dataset.page;
  if (!Number.isFinite(p)) return;
  loadGallery(p);
  document.querySelector(".gallery-section").scrollIntoView({behavior: "smooth", block: "start"});
});

async function loadPalettes() {
  const r = await get("/api/palettes");
  palettesCache = r.palettes || [];
  renderControlsPanel(currentControlsPanels);
}
function paletteSwatchHtml(p, activeId) {
  const cls = p.id === activeId ? "swatch active" : "swatch";
  return `<button type="button" class="${cls}" title="${esc(p.name)}" data-palette="${esc(p.id)}" style="--swatch:conic-gradient(${Object.values(p.colors).join(",")})"></button>`;
}

// ----- Controls drawer -----
function renderControlsPanel(panels) {
  currentControlsPanels = panels || [];
  const n = currentControlsPanels.length;
  controlsCountEl.textContent = String(n);
  if (!n) {
    controlsToggle.hidden = true;
    controlsDrawer.hidden = true;
    controlsDrawer.classList.remove("open");
    controlsToggle.classList.remove("open");
    controlsPanel.innerHTML = "";
    return;
  }
  controlsToggle.hidden = false;
  controlsDrawer.hidden = false;
  controlsPanel.innerHTML = currentControlsPanels.map(renderPanel).join("");
}
function renderPanel(panel) {
  const collapsed = collapsedSkills.has(panel.skill_name) ? " collapsed" : "";
  const widgets = (panel.schema || []).map(spec => renderWidget(panel, spec)).join("");
  return `<section class="ctl-panel${collapsed}" data-chain="${panel.chain_index}" data-skill="${esc(panel.skill_name)}"><header class="ctl-head" data-toggle-skill="${esc(panel.skill_name)}"><span>${esc(panel.skill_name)}</span><span class="ctl-chev">▼</span></header><div class="ctl-body">${widgets}</div></section>`;
}
controlsToggle.addEventListener("click", () => {
  const open = !controlsDrawer.classList.contains("open");
  controlsDrawer.classList.toggle("open", open);
  controlsToggle.classList.toggle("open", open);
  localStorage.sbDrawerOpen = open ? "1" : "0";
});
controlsPanel.addEventListener("click", e => {
  const head = e.target.closest("[data-toggle-skill]");
  if (head) {
    const skill = head.dataset.toggleSkill;
    const section = head.closest(".ctl-panel");
    const willCollapse = !section.classList.contains("collapsed");
    section.classList.toggle("collapsed", willCollapse);
    if (willCollapse) collapsedSkills.add(skill); else collapsedSkills.delete(skill);
    localStorage.sbCollapsedSkills = JSON.stringify([...collapsedSkills]);
    return;
  }
  const target = e.target;
  const sw = target.closest("button[data-palette]");
  if (sw) {
    const row = sw.closest(".ctl-palette");
    const chain = +row.dataset.chain;
    postControl({chain_index: chain, name: "palette", value: sw.dataset.palette});
    return;
  }
  if (target.matches(".ctl-seg")) {
    postControl({chain_index: +target.dataset.chain, name: target.dataset.name, value: JSON.parse(target.dataset.value)});
    return;
  }
  const pan = target.closest(".ctl-pan");
  const arrow = target.closest("button[data-dir]");
  if (pan && arrow) {
    const step = +pan.dataset.step || 0.1;
    let x = +pan.dataset.x || 0, y = +pan.dataset.y || 0;
    if (arrow.dataset.dir === "left") x -= step;
    else if (arrow.dataset.dir === "right") x += step;
    else if (arrow.dataset.dir === "up") y -= step;
    else if (arrow.dataset.dir === "down") y += step;
    pan.dataset.x = x; pan.dataset.y = y;
    const cc = pan.querySelector(".ctl-pan-c"); if (cc) cc.textContent = `${fmtNum(x)}, ${fmtNum(y)}`;
    postControl({chain_index: +pan.dataset.chain, name: pan.dataset.name, value: {x, y}});
    return;
  }
  if (target.matches(".ctl-btn")) {
    postControl({chain_index: +target.dataset.chain, name: target.dataset.name, value: null});
  }
});
controlsPanel.addEventListener("input", e => {
  const el = e.target;
  const chain = +el.dataset.chain;
  if (Number.isNaN(chain)) return;
  if (el.dataset.kind === "slider") {
    const valEl = el.parentElement.querySelector(".ctl-val");
    if (valEl) valEl.textContent = fmtNum(+el.value);
    debounceControl(`${chain}.${el.dataset.name}`, () => postControl({chain_index: chain, name: el.dataset.name, value: +el.value}));
  }
});
controlsPanel.addEventListener("change", e => {
  const el = e.target;
  const chain = +el.dataset.chain;
  if (Number.isNaN(chain)) return;
  if (el.dataset.kind === "bool") {
    postControl({chain_index: chain, name: el.dataset.name, value: el.checked});
  }
});

function renderLayersStack(layers) {
  if (!layers.length) { layersStack.hidden = true; layersStack.innerHTML = ""; return; }
  layersStack.hidden = false;
  layersStack.innerHTML = layers.map(l => `<div class="layer-chip" title="${esc(l.skill_name)}"><span class="layer-face">${esc(l.skill_name)}</span><button type="button" class="layer-x" data-chain="${l.chain_index}" aria-label="Delete ${esc(l.skill_name)}">×</button></div>`).join("");
}
function renderWidget(panel, spec) {
  const v = panel.values || {};
  const id = `c${panel.chain_index}-${spec.name}`;
  if (spec.type === "slider") {
    const cur = v[spec.name] ?? spec.default;
    return `<label class="ctl-row" for="${id}"><span>${esc(spec.label)}</span><input id="${id}" type="range" min="${spec.min}" max="${spec.max}" step="${spec.step}" value="${cur}" data-chain="${panel.chain_index}" data-name="${esc(spec.name)}" data-kind="slider"><span class="ctl-val">${fmtNum(cur)}</span></label>`;
  }
  if (spec.type === "bool") {
    const on = !!(v[spec.name] ?? spec.default);
    return `<label class="ctl-row"><span>${esc(spec.label)}</span><input type="checkbox" ${on?"checked":""} data-chain="${panel.chain_index}" data-name="${esc(spec.name)}" data-kind="bool"></label>`;
  }
  if (spec.type === "enum") {
    const cur = v[spec.name] ?? spec.default;
    const opts = (spec.options || []).map(o =>
      `<button type="button" class="${JSON.stringify(o.value)===JSON.stringify(cur)?"ctl-seg active":"ctl-seg"}" data-chain="${panel.chain_index}" data-name="${esc(spec.name)}" data-kind="enum" data-value='${esc(JSON.stringify(o.value))}'>${esc(o.label)}</button>`
    ).join("");
    return `<div class="ctl-row"><span>${esc(spec.label)}</span><div class="ctl-segs">${opts}</div></div>`;
  }
  if (spec.type === "pan") {
    const xp = spec.x_param, yp = spec.y_param;
    const xv = v[xp] ?? spec.x_default ?? 0;
    const yv = v[yp] ?? spec.y_default ?? 0;
    return `<div class="ctl-row"><span>${esc(spec.label)}</span><div class="ctl-pan" data-chain="${panel.chain_index}" data-name="${esc(spec.name)}" data-step="${spec.step}" data-x="${xv}" data-y="${yv}"><button type="button" class="ctl-pan-up" data-dir="up">↑</button><button type="button" class="ctl-pan-left" data-dir="left">←</button><span class="ctl-pan-c">${fmtNum(xv)}, ${fmtNum(yv)}</span><button type="button" class="ctl-pan-right" data-dir="right">→</button><button type="button" class="ctl-pan-down" data-dir="down">↓</button></div></div>`;
  }
  if (spec.type === "button") {
    return `<div class="ctl-row"><span>${esc(spec.label)}</span><button type="button" class="ctl-btn" data-chain="${panel.chain_index}" data-name="${esc(spec.name)}" data-kind="button">${esc(spec.label)}</button></div>`;
  }
  if (spec.type === "palette") {
    const cur = v.palette || "";
    const swatches = palettesCache.map(p => paletteSwatchHtml(p, cur)).join("");
    return `<div class="ctl-row ctl-palette" data-chain="${panel.chain_index}" data-name="palette"><span>${esc(spec.label || "Palette")}</span><div class="ctl-palette-strip">${swatches}</div></div>`;
  }
  return "";
}
function fmtNum(v) {
  if (typeof v !== "number") return String(v);
  const abs = Math.abs(v);
  return abs >= 100 ? v.toFixed(0) : abs >= 10 ? v.toFixed(1) : v.toFixed(2);
}
function debounceControl(key, fn, ms = 160) {
  clearTimeout(controlDebounce.get(key));
  controlDebounce.set(key, setTimeout(fn, ms));
}
async function postControl(body) {
  if (controlsPanel.classList.contains("loading")) return;
  controlsPanel.classList.add("loading");
  loaderTicketStart();
  try { render((await post("/api/skill_control", body)).events); }
  catch (err) { add("error", err.message); }
  finally { controlsPanel.classList.remove("loading"); loaderTicketEnd(); }
}
function esc(x) { return String(x ?? "").replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m])); }

// ----- Canvas loader: particle field + chain-progress arc -----
const loaderCanvasEl = document.querySelector("#loadCanvas");
const progressArc = document.querySelector("#progressArc");
const progressFill = progressArc.querySelector(".fill");
let userTickets = 0;
const activeRenderTools = new Set();
let loaderRaf = 0;
let loaderSafetyTimer = 0;
let loaderStopGrace = 0;
let particleCtx = null;
let particles = [];
let pW = 0, pH = 0, pT = 0;
let accentRgb = [165, 175, 195];
let accent2Rgb = [195, 175, 195];

function loaderShouldShow() { return userTickets > 0 || activeRenderTools.size > 0; }
function loaderSync() {
  if (loaderShouldShow()) {
    if (loaderStopGrace) { clearTimeout(loaderStopGrace); loaderStopGrace = 0; }
    if (!showcase.classList.contains("loading")) loaderStartNow();
  } else if (showcase.classList.contains("loading") && !loaderStopGrace) {
    // Short grace period — the next polling tick might bring a tool_status.
    loaderStopGrace = setTimeout(loaderForceStop, 1500);
  }
}
function loaderStartNow() {
  showcase.classList.add("loading");
  particleInit();
  loaderRaf = requestAnimationFrame(particleStep);
  clearTimeout(loaderSafetyTimer);
  loaderSafetyTimer = setTimeout(loaderForceStop, 45000);
}
function loaderForceStop() {
  userTickets = 0; activeRenderTools.clear();
  showcase.classList.remove("loading");
  if (loaderRaf) { cancelAnimationFrame(loaderRaf); loaderRaf = 0; }
  clearTimeout(loaderSafetyTimer); loaderSafetyTimer = 0;
  if (loaderStopGrace) { clearTimeout(loaderStopGrace); loaderStopGrace = 0; }
  setProgress(0, 0);
}
function loaderTicketStart() { userTickets++; loaderSync(); }
function loaderTicketEnd() { if (userTickets > 0) userTickets--; loaderSync(); }
function loaderToolStart(id) { activeRenderTools.add(id); loaderSync(); }
function loaderToolEnd(id) { activeRenderTools.delete(id); loaderSync(); }
async function withLoader(fn) {
  loaderTicketStart();
  try { return await fn(); }
  finally { loaderTicketEnd(); }
}
function setProgress(done, total) {
  if (total <= 1 || done <= 0) {
    progressArc.classList.remove("visible");
    progressFill.style.strokeDashoffset = "1";
    return;
  }
  progressArc.classList.add("visible");
  progressFill.style.strokeDashoffset = String(Math.max(0, 1 - (done / total)));
}

// Read current accent CSS vars, then desaturate + soften so the field stays low-key.
function refreshAccentRgb() {
  const root = document.documentElement;
  const a = (getComputedStyle(root).getPropertyValue("--accent").trim()) || "#3df2ff";
  const a2 = (getComputedStyle(root).getPropertyValue("--accent-2").trim()) || "#ff4d8d";
  accentRgb = softenRgb(parseColor(a));
  accent2Rgb = softenRgb(parseColor(a2));
}
function parseColor(s) {
  s = s.trim();
  if (s.startsWith("#")) {
    let h = s.slice(1);
    if (h.length === 3) h = h.split("").map(c => c+c).join("");
    const n = parseInt(h, 16);
    return [(n>>16)&255, (n>>8)&255, n&255];
  }
  const m = s.match(/rgba?\(([^)]+)\)/);
  if (m) {
    const p = m[1].split(",").map(x => parseFloat(x));
    return [p[0]|0, p[1]|0, p[2]|0];
  }
  return [180, 180, 180];
}
// Pull toward gray (desaturate) and toward a soft mid-tone (lower contrast).
function softenRgb([r,g,b]) {
  const gray = 0.299*r + 0.587*g + 0.114*b;
  const SAT = 0.38;    // 0 = full gray, 1 = full color
  const LIFT = 0.55;   // 0 = original, 1 = pure 180-gray
  const dr = gray + (r - gray) * SAT;
  const dg = gray + (g - gray) * SAT;
  const db = gray + (b - gray) * SAT;
  return [
    Math.round(dr * (1 - LIFT) + 180 * LIFT),
    Math.round(dg * (1 - LIFT) + 180 * LIFT),
    Math.round(db * (1 - LIFT) + 180 * LIFT),
  ];
}

// Small value-noise grid for vector-field flow.
const NSIZE = 64;
let nGrid = null;
function noiseInit() {
  nGrid = new Float32Array(NSIZE * NSIZE);
  for (let i = 0; i < nGrid.length; i++) nGrid[i] = Math.random();
}
function noise2(x, y) {
  const xi = Math.floor(x), yi = Math.floor(y);
  const xf = x - xi, yf = y - yi;
  const wx0 = ((xi % NSIZE) + NSIZE) % NSIZE;
  const wx1 = (wx0 + 1) % NSIZE;
  const wy0 = ((yi % NSIZE) + NSIZE) % NSIZE;
  const wy1 = (wy0 + 1) % NSIZE;
  const u = xf*xf*(3-2*xf), v = yf*yf*(3-2*yf);
  const n00 = nGrid[wy0*NSIZE + wx0], n10 = nGrid[wy0*NSIZE + wx1];
  const n01 = nGrid[wy1*NSIZE + wx0], n11 = nGrid[wy1*NSIZE + wx1];
  return (n00*(1-u) + n10*u) * (1-v) + (n01*(1-u) + n11*u) * v;
}

const PARTICLE_COUNT = 360;
function particleInit() {
  refreshAccentRgb();
  if (!nGrid) noiseInit();
  const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
  pW = loaderCanvasEl.width = Math.max(2, Math.floor(loaderCanvasEl.clientWidth * dpr));
  pH = loaderCanvasEl.height = Math.max(2, Math.floor(loaderCanvasEl.clientHeight * dpr));
  particleCtx = loaderCanvasEl.getContext("2d");
  particleCtx.fillStyle = "rgba(8,9,13,1)";
  particleCtx.fillRect(0, 0, pW, pH);
  particles = [];
  for (let i = 0; i < PARTICLE_COUNT; i++) {
    particles.push({ x: Math.random()*pW, y: Math.random()*pH, mix: Math.random() });
  }
  pT = 0;
}
let accentRefreshTick = 0;
function particleStep() {
  if (!particleCtx) return;
  pT += 0.0028;
  accentRefreshTick = (accentRefreshTick + 1) % 36;
  if (accentRefreshTick === 0) refreshAccentRgb();
  // Soft trail fade — low alpha keeps the field low-key.
  particleCtx.fillStyle = "rgba(8,9,13,.075)";
  particleCtx.fillRect(0, 0, pW, pH);
  for (const p of particles) {
    const n = noise2(p.x/120, p.y/120 + pT*3.5);
    const angle = n * Math.PI * 4;
    const speed = 0.4 + n * 0.55;
    p.x += Math.cos(angle) * speed;
    p.y += Math.sin(angle) * speed;
    if (p.x < 0) p.x += pW; else if (p.x >= pW) p.x -= pW;
    if (p.y < 0) p.y += pH; else if (p.y >= pH) p.y -= pH;
    const r = accentRgb[0]*(1-p.mix) + accent2Rgb[0]*p.mix;
    const g = accentRgb[1]*(1-p.mix) + accent2Rgb[1]*p.mix;
    const b = accentRgb[2]*(1-p.mix) + accent2Rgb[2]*p.mix;
    particleCtx.fillStyle = `rgba(${r|0},${g|0},${b|0},.42)`;
    particleCtx.fillRect(p.x, p.y, 1.3, 1.3);
  }
  loaderRaf = requestAnimationFrame(particleStep);
}

// ----- Chat + actions -----
form.addEventListener("submit", async e => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  add("user", text);
  const thinking = document.createElement("article");
  thinking.className = "status";
  thinking.textContent = "Thinking...";
  messages.appendChild(thinking);
  bottom(true);
  loaderTicketStart();
  try { const result = await post("/api/chat", {message:text}); thinking.remove(); render(result.events); }
  catch (err) { add("error", err.message); }
  finally { loaderTicketEnd(); }
});
document.querySelector("#newChat").addEventListener("click", async () => {
  messages.innerHTML = "";
  toolStatusEls.clear();
  render((await post("/api/new")).events);
  loadGallery(1);
});

// Empty-state example chips: pre-fill composer, focus, but let user send.
emptyState.addEventListener("click", e => {
  const chip = e.target.closest(".es-chip");
  if (!chip) return;
  input.value = chip.dataset.prompt || chip.textContent;
  input.focus();
});

downloadImage.addEventListener("click", () => {
  if (downloadImage.getAttribute("href") && downloadImage.getAttribute("href") !== "#") {
    post("/api/download").catch(() => {});
  }
});
regenerateBtn.addEventListener("click", async () => {
  if (regenerateBtn.disabled) return;
  regenerateBtn.disabled = true;
  regenerateBtn.classList.add("loading");
  loaderTicketStart();
  try { render((await post("/api/regenerate")).events); }
  catch (err) { add("error", err.message); }
  finally { regenerateBtn.disabled = false; regenerateBtn.classList.remove("loading"); loaderTicketEnd(); }
});
shareBtn.addEventListener("click", () => { sharePanel.hidden = !sharePanel.hidden; shareTitle.value ||= "untitled"; shareArtist.value ||= "anonymous"; });
document.querySelector("#shareConfirm").addEventListener("click", async () => render((await post("/api/share", {title:shareTitle.value, artist:shareArtist.value})).events));
gallery.addEventListener("click", async e => {
  if (!e.target.matches("button[data-path]")) return;
  scrollTo({top:0, behavior:"smooth"});
  loaderTicketStart();
  try { render((await post("/api/remix", {path:e.target.dataset.path})).events); }
  catch (err) { add("error", err.message); }
  finally { loaderTicketEnd(); }
});
layersStack.addEventListener("click", async e => {
  const x = e.target.closest(".layer-x");
  if (!x) return;
  const chain = +x.dataset.chain;
  if (Number.isNaN(chain)) return;
  try { render((await post("/api/layer_delete", {chain_index: chain})).events); }
  catch (err) { add("error", err.message); }
});
setInterval(poll, 1200);
loadPalettes(); loadCanvas(); loadGallery(1); refreshAccount();
