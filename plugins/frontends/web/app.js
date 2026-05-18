// TIPS is defined in tips.js, loaded before this script in index.html.
function pickTip() {
  const el = document.querySelector("#tipText");
  if (el && Array.isArray(TIPS) && TIPS.length) {
    el.textContent = TIPS[Math.floor(Math.random() * TIPS.length)];
  }
}
pickTip();

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
const saveBtn = document.querySelector("#saveImage");
const shareBtn = document.querySelector("#shareImage");
const sharePanel = document.querySelector("#sharePanel");
const shareTitle = document.querySelector("#shareTitle");
const shareArtist = document.querySelector("#shareArtist");
const gallery = document.querySelector("#gallery");
const paginator = document.querySelector("#paginator");
const archive = document.querySelector("#archive");
const archivePaginator = document.querySelector("#archivePaginator");
const galleryTabs = document.querySelector("#galleryTabs");
const sharedPanel = document.querySelector("#sharedPanel");
const archivePanel = document.querySelector("#archivePanel");
const controlsPanel = document.querySelector("#controlsPanel");
const controlsDrawer = document.querySelector("#controlsDrawer");
const controlsToggle = document.querySelector("#controlsToggle");
const emptyState = document.querySelector("#emptyState");
const NEAR_BOTTOM_PX = 80;
const GALLERY_PAGE = 18;
let palettesCache = [];
let currentControlsPanels = [];
const galleryPages = {shared: 1, archive: 1};
let activeTab = "shared";
const controlDebounce = new Map();
let typingEl = null;
let agentBusy = false;
const sendBtn = form.querySelector("button");
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
  // Tool-status chips are intentionally not shown in chat — the loader and the
  // agent's own step-by-step messages cover what the user needs to see.
  // We still drive the loader so "Thinking…" stays alive across tool calls.
  const id = ev.call_id || `${ev.name}:${Date.now()}`;
  if (ev.status === "started") loaderToolStart(id);
  else if (ev.status === "finished") loaderToolEnd(id);
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
    }
    else if (ev.type === "canvas_reset") { setCanvas(null); }
    else if (ev.type === "shared") { sharePanel.hidden = true; loadGallery(1); }
    else if (ev.type === "attachment") add("assistant", `Attachment: ${ev.name}`);
    else if (ev.type === "typing") setTyping(!!ev.on);
  }
  bottom();
}

function setBusy(on) {
  agentBusy = !!on;
  if (agentBusy) {
    sendBtn.type = "button";
    sendBtn.textContent = "Cancel";
    sendBtn.classList.add("cancel");
    sendBtn.disabled = false;
  } else {
    sendBtn.type = "submit";
    sendBtn.textContent = "Send";
    sendBtn.classList.remove("cancel");
    sendBtn.disabled = false;
  }
}
sendBtn.addEventListener("click", async e => {
  if (!agentBusy) return; // let submit handler run
  e.preventDefault();
  // Stay disabled until the in-flight chat() returns and setBusy(false) re-enables.
  sendBtn.disabled = true;
  try { render((await post("/api/cancel")).events); }
  catch (err) { add("error", err.message); }
});

function setTyping(on) {
  if (on) {
    if (typingEl && typingEl.isConnected) { bottom(); return; }
    typingEl = document.createElement("article");
    typingEl.className = "status typing";
    typingEl.textContent = "Thinking…";
    messages.appendChild(typingEl);
    bottom();
  } else if (typingEl) {
    typingEl.remove();
    typingEl = null;
  }
}

// ----- account + paywall + auth -----
const accountAvatar = document.querySelector("#accountAvatar");
const avatarInner = document.querySelector("#avatarInner");
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
  const remaining = acc?.messages_remaining;
  const max = acc?.messages_max;
  const unlimited = max == null;
  const fill = unlimited ? 1 : Math.max(0, Math.min(1, (remaining ?? 0) / max));
  const low = !unlimited && typeof remaining === "number" && remaining <= 5;
  const fillColor = low ? "var(--hot)" : "var(--accent)";
  const pct = (fill * 100).toFixed(2);
  accountAvatar.style.background = `conic-gradient(${fillColor} ${pct}%, var(--line) 0)`;
  accountAvatar.style.boxShadow = unlimited ? "0 0 8px var(--accent-glow)" : "none";
  avatarInner.textContent = (acc?.signed_in && acc.email) ? acc.email[0].toUpperCase() : "·";
  const tier = acc?.tier || "free";
  const tierLabel = acc?.signed_in ? (tier === "unlimited" ? "Unlimited" : tier === "paid" ? "Paid" : "Free") : "Free demo";
  const tail = unlimited ? "Unlimited" : `${(remaining ?? 0).toLocaleString()} messages left`;
  accountAvatar.title = `${tierLabel} — ${tail}`;
  accountAvatar.hidden = false;
}
async function refreshAccount() {
  try { const r = await get("/api/account"); setAccount(r.account); }
  catch (err) { console.warn("[account] refresh failed:", err); }
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
  if (!c?.url) {
    showcase.classList.remove("has-image");
    renderControlsPanel([]);
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
    renderControlsPanel(c?.controls_panels || []);
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
      loadGallery(galleryPages.shared);
    }, 280);
  };
  pre.onerror = () => { apply(); loadGallery(galleryPages.shared); };
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
    const accent = hslToHex(a1.h, Math.min(0.55, a1.s), Math.min(0.66, Math.max(0.55, a1.l * 0.5 + 0.31)));
    const accent2 = hslToHex(a2.h, Math.min(0.55, a2.s), Math.min(0.66, Math.max(0.55, a2.l * 0.5 + 0.31)));
    document.documentElement.style.setProperty("--accent", accent);
    document.documentElement.style.setProperty("--accent-2", accent2);
    const glow = hexWithAlpha(accent, 0.10);
    const glow2 = hexWithAlpha(accent2, 0.14);
    document.documentElement.style.setProperty("--accent-glow", glow);
    document.documentElement.style.setProperty("--accent-2-glow", glow2);
    try { localStorage.sbAccents = JSON.stringify({accent, accent2, glow, glow2}); } catch {}
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
const GALLERY_TABS = {
  shared: {url: "/api/gallery", grid: gallery, paginator: paginator, empty: "No shared canvases yet.", action: "Remix", endpoint: "/api/remix"},
  archive: {url: "/api/archive", grid: archive, paginator: archivePaginator, empty: "Your archive is empty. Save a canvas to start.", action: "Remix", endpoint: "/api/archive_remix"},
};

async function loadGalleryFor(kind, page = 1) {
  const cfg = GALLERY_TABS[kind];
  if (!cfg) return;
  galleryPages[kind] = Math.max(1, page);
  const offset = (galleryPages[kind] - 1) * GALLERY_PAGE;
  const r = await get(`${cfg.url}?limit=${GALLERY_PAGE}&offset=${offset}`);
  const items = Array.isArray(r.items) ? r.items : (Array.isArray(r) ? r : []);
  const total = typeof r.total === "number" ? r.total : items.length + offset;
  cfg.grid.innerHTML = items.map(x => `<article class="gallery-card"><img src="${x.url}" alt=""><div><strong>${esc(x.title)}</strong><small>${esc(x.artist)}${x.score ? ` · ${(x.score*100).toFixed(0)}% similar` : ""}</small><button data-kind="${kind}" data-path="${esc(x.path)}">${cfg.action}</button></div></article>`).join("") || `<article class='assistant'>${cfg.empty}</article>`;
  renderPaginatorFor(kind, total);
}
function loadGallery(page = 1) { return loadGalleryFor("shared", page); }

function renderPaginatorFor(kind, total) {
  const cfg = GALLERY_TABS[kind];
  const el = cfg.paginator;
  const pages = Math.max(1, Math.ceil(total / GALLERY_PAGE));
  if (pages <= 1) { el.hidden = true; el.innerHTML = ""; return; }
  el.hidden = false;
  const cur = galleryPages[kind];
  const btns = [];
  btns.push(`<button type="button" data-page="${cur-1}" ${cur===1?"disabled":""}>‹</button>`);
  const set = new Set([1, pages, cur-1, cur, cur+1]);
  for (let p = 1; p <= pages; p++) {
    if (set.has(p)) btns.push(`<button type="button" data-page="${p}" class="${p===cur?"active":""}">${p}</button>`);
    else if (p === 2 || p === pages-1) btns.push(`<button type="button" disabled>…</button>`);
  }
  btns.push(`<button type="button" data-page="${cur+1}" ${cur===pages?"disabled":""}>›</button>`);
  el.innerHTML = btns.join("");
}

for (const [kind, cfg] of Object.entries(GALLERY_TABS)) {
  cfg.paginator.addEventListener("click", e => {
    const b = e.target.closest("button[data-page]");
    if (!b || b.disabled) return;
    const p = +b.dataset.page;
    if (!Number.isFinite(p)) return;
    loadGalleryFor(kind, p);
    document.querySelector(".gallery-section").scrollIntoView({behavior: "smooth", block: "start"});
  });
}

galleryTabs.addEventListener("click", e => {
  const b = e.target.closest("button[data-tab]");
  if (!b) return;
  const kind = b.dataset.tab;
  activeTab = kind;
  galleryTabs.querySelectorAll(".tab").forEach(t => t.classList.toggle("active", t === b));
  sharedPanel.hidden = kind !== "shared";
  archivePanel.hidden = kind !== "archive";
  loadGalleryFor(kind, galleryPages[kind]);
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
  const hasImage = showcase.classList.contains("has-image");
  if (!hasImage) {
    controlsToggle.hidden = true;
    controlsDrawer.hidden = true;
    controlsDrawer.classList.remove("open");
    controlsToggle.classList.remove("open");
    controlsPanel.innerHTML = "";
    return;
  }
  controlsToggle.hidden = false;
  controlsDrawer.hidden = false;
  const globalPanel = `<section class="ctl-panel"><header class="ctl-head">Global</header><div class="ctl-body"><div class="ctl-row"><span>Regenerate</span><button type="button" class="ctl-global" id="globalRegenerate" title="Re-render the current chain with new seeds">Regenerate</button></div></div></section>`;
  controlsPanel.innerHTML = globalPanel + currentControlsPanels.map(renderPanel).join("");
}
function renderPanel(panel) {
  const widgets = (panel.schema || []).map(spec => renderWidget(panel, spec)).join("");
  return `<section class="ctl-panel" data-chain="${panel.chain_index}" data-skill="${esc(panel.skill_name)}"><header class="ctl-head">${esc(panel.skill_name)}</header><div class="ctl-body">${widgets}</div></section>`;
}
controlsToggle.addEventListener("click", () => {
  const open = !controlsDrawer.classList.contains("open");
  controlsDrawer.classList.toggle("open", open);
  controlsToggle.classList.toggle("open", open);
  localStorage.sbDrawerOpen = open ? "1" : "0";
});
controlsPanel.addEventListener("click", async e => {
  if (e.target.id === "globalRegenerate") {
    const btn = e.target;
    if (btn.disabled) return;
    btn.disabled = true;
    loaderTicketStart();
    try { render((await post("/api/regenerate")).events); }
    catch (err) { add("error", err.message); }
    finally { btn.disabled = false; loaderTicketEnd(); }
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

// ----- Canvas loader: blur the hero while work is in flight -----
let userTickets = 0;
const activeRenderTools = new Set();
let loaderSafetyTimer = 0;
let loaderStopGrace = 0;

function loaderShouldShow() { return userTickets > 0 || activeRenderTools.size > 0; }
function loaderSync() {
  if (loaderShouldShow()) {
    if (loaderStopGrace) { clearTimeout(loaderStopGrace); loaderStopGrace = 0; }
    if (!showcase.classList.contains("loading")) loaderStartNow();
  } else if (showcase.classList.contains("loading") && !loaderStopGrace) {
    loaderStopGrace = setTimeout(loaderForceStop, 200);
  }
}
function loaderStartNow() {
  showcase.classList.add("loading");
  clearTimeout(loaderSafetyTimer);
  loaderSafetyTimer = setTimeout(loaderForceStop, 45000);
}
function loaderForceStop() {
  userTickets = 0; activeRenderTools.clear();
  showcase.classList.remove("loading");
  clearTimeout(loaderSafetyTimer); loaderSafetyTimer = 0;
  if (loaderStopGrace) { clearTimeout(loaderStopGrace); loaderStopGrace = 0; }
}
function loaderTicketStart() { userTickets++; loaderSync(); }
function loaderTicketEnd() { if (userTickets > 0) userTickets--; loaderSync(); }
function loaderToolStart(id) { activeRenderTools.add(id); loaderSync(); }
function loaderToolEnd(id) { activeRenderTools.delete(id); loaderSync(); }

// ----- Chat + actions -----
form.addEventListener("submit", async e => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  add("user", text);
  bottom(true);
  loaderTicketStart();
  setBusy(true);
  try { render((await post("/api/chat", {message:text})).events); }
  catch (err) { add("error", err.message); }
  finally { loaderTicketEnd(); setTyping(false); setBusy(false); }
});
document.querySelector("#newChat").addEventListener("click", async () => {
  messages.innerHTML = `<div class="ephemeral-note" id="tipNote"><strong>Tip</strong><span id="tipText"></span></div>`;
  pickTip();
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
saveBtn.addEventListener("click", async () => {
  if (saveBtn.disabled) return;
  saveBtn.disabled = true;
  saveBtn.classList.add("loading");
  try {
    const r = await post("/api/save");
    render(r.events);
    loadGalleryFor("archive", 1);
  } catch (err) { add("error", err.message); }
  finally { saveBtn.disabled = false; saveBtn.classList.remove("loading"); }
});
shareBtn.addEventListener("click", () => { sharePanel.hidden = !sharePanel.hidden; shareTitle.value ||= "untitled"; shareArtist.value ||= "anonymous"; });
document.querySelector("#shareConfirm").addEventListener("click", async () => render((await post("/api/share", {title:shareTitle.value, artist:shareArtist.value})).events));
async function handleGalleryRemix(e) {
  const btn = e.target.closest("button[data-path]");
  if (!btn) return;
  const kind = btn.dataset.kind || "shared";
  const endpoint = GALLERY_TABS[kind]?.endpoint || "/api/remix";
  scrollTo({top:0, behavior:"smooth"});
  loaderTicketStart();
  try { render((await post(endpoint, {path: btn.dataset.path})).events); }
  catch (err) { add("error", err.message); }
  finally { loaderTicketEnd(); }
}
gallery.addEventListener("click", handleGalleryRemix);
archive.addEventListener("click", handleGalleryRemix);
(function rehydrateAccents() {
  try {
    const a = JSON.parse(localStorage.sbAccents || "null");
    if (!a) return;
    const root = document.documentElement.style;
    if (a.accent) root.setProperty("--accent", a.accent);
    if (a.accent2) root.setProperty("--accent-2", a.accent2);
    if (a.glow) root.setProperty("--accent-glow", a.glow);
    if (a.glow2) root.setProperty("--accent-2-glow", a.glow2);
  } catch {}
})();
async function loadHistory() {
  try {
    const r = await get("/api/history");
    const msgs = Array.isArray(r.history) ? r.history : [];
    if (!msgs.length) return;
    // Replace boilerplate greeting only when real history exists.
    messages.innerHTML = "";
    for (const m of msgs) add(m.role === "user" ? "user" : "assistant", m.content, m.role === "assistant");
  } catch {}
}
setInterval(poll, 1200);
loadHistory(); loadPalettes(); loadCanvas(); loadGallery(1); loadGalleryFor("archive", 1); refreshAccount();
