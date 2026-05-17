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
const controlsPanel = document.querySelector("#controlsPanel");
const layersStack = document.querySelector("#layersStack");
const NEAR_BOTTOM_PX = 80;
let palettesCache = [];
let currentControlsPanels = [];
const controlDebounce = new Map();
const toolStatusEls = new Map();
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
    else if (ev.type === "error") add("error", ev.content);
    else if (ev.type === "form") add("assistant", `${ev.form?.display?.prompt || "Input required"}\n${(ev.form?.display?.choices || []).map(c => c.label || c.value).join(" / ")}`);
    else if (ev.type === "approval") approval(ev);
    else if (ev.type === "paywall") openPaywall(ev);
    else if (ev.type === "account") setAccount(ev.account);
    else if (ev.type === "hero_image") {
      setCanvas(ev.canvas || {url: ev.url, name: ev.name});
    }
    else if (ev.type === "canvas_reset") setCanvas(null);
    else if (ev.type === "shared") { sharePanel.hidden = true; loadGallery(); }
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
function setCanvas(c) {
  renderControlsPanel(c?.controls_panels || []);
  renderLayersStack(c?.layers || []);
  if (!c?.url) {
    showcase.classList.remove("has-image"); heroImage.classList.remove("fading"); heroImage.removeAttribute("src"); downloadImage.href = "#";
    return;
  }
  const newUrl = c.url, newName = c.name || "canvas.png";
  const apply = () => { heroImage.src = newUrl; heroImage.alt = newName; downloadImage.href = newUrl; downloadImage.download = newName; showcase.classList.add("has-image"); };
  if (!showcase.classList.contains("has-image")) { apply(); loadGallery(); return; }
  // Preload the new image, then crossfade out → swap → crossfade in.
  const pre = new Image();
  pre.onload = () => {
    heroImage.classList.add("fading");
    setTimeout(() => {
      apply();
      requestAnimationFrame(() => requestAnimationFrame(() => heroImage.classList.remove("fading")));
      loadGallery();
    }, 280);
  };
  pre.onerror = () => { apply(); loadGallery(); };
  pre.src = newUrl;
}
async function loadCanvas() { const r = await get("/api/canvas"); setCanvas(r.canvas); }
async function loadGallery() {
  const r = await get("/api/gallery");
  gallery.innerHTML = (r.items || []).map(x => `<article class="gallery-card"><img src="${x.url}" alt=""><div><strong>${esc(x.title)}</strong><small>${esc(x.artist)}${x.score ? ` · ${(x.score*100).toFixed(0)}% similar` : ""}</small><button data-path="${esc(x.path)}">Remix</button></div></article>`).join("") || "<article class='assistant'>No shared canvases yet.</article>";
}
async function loadPalettes() {
  const r = await get("/api/palettes");
  palettesCache = r.palettes || [];
  renderControlsPanel(currentControlsPanels);
}
function paletteSwatchHtml(p, activeId) {
  const cls = p.id === activeId ? "swatch active" : "swatch";
  return `<button type="button" class="${cls}" title="${esc(p.name)}" data-palette="${esc(p.id)}" style="--swatch:conic-gradient(${Object.values(p.colors).join(",")})"></button>`;
}
function renderControlsPanel(panels) {
  currentControlsPanels = panels || [];
  if (!currentControlsPanels.length) {
    controlsPanel.hidden = true;
    controlsPanel.innerHTML = "";
    return;
  }
  controlsPanel.hidden = false;
  controlsPanel.innerHTML = currentControlsPanels.map(renderPanel).join("");
}
function renderLayersStack(layers) {
  if (!layers.length) { layersStack.hidden = true; layersStack.innerHTML = ""; return; }
  layersStack.hidden = false;
  layersStack.innerHTML = layers.map(l => `<div class="layer-chip" title="${esc(l.skill_name)}"><span class="layer-face">${esc(l.skill_name)}</span><button type="button" class="layer-x" data-chain="${l.chain_index}" aria-label="Delete ${esc(l.skill_name)}">×</button></div>`).join("");
}
function renderPanel(panel) {
  const widgets = (panel.schema || []).map(spec => renderWidget(panel, spec)).join("");
  return `<section class="ctl-panel" data-chain="${panel.chain_index}"><header class="ctl-head">${esc(panel.skill_name)}</header>${widgets}</section>`;
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
  try { render((await post("/api/skill_control", body)).events); }
  catch (err) { add("error", err.message); }
  finally { controlsPanel.classList.remove("loading"); }
}
function esc(x) { return String(x ?? "").replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m])); }
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
  try { const result = await post("/api/chat", {message:text}); thinking.remove(); render(result.events); }
  catch (err) { add("error", err.message); }
});
document.querySelector("#newChat").addEventListener("click", async () => {
  messages.innerHTML = "";
  toolStatusEls.clear();
  render((await post("/api/new")).events);
  loadGallery();
});
downloadImage.addEventListener("click", () => {
  // Fire-and-forget telemetry — the actual download is the plain <a download>.
  if (downloadImage.getAttribute("href") && downloadImage.getAttribute("href") !== "#") {
    post("/api/download").catch(() => {});
  }
});
regenerateBtn.addEventListener("click", async () => {
  if (regenerateBtn.disabled) return;
  regenerateBtn.disabled = true;
  regenerateBtn.classList.add("loading");
  try { render((await post("/api/regenerate")).events); }
  catch (err) { add("error", err.message); }
  finally { regenerateBtn.disabled = false; regenerateBtn.classList.remove("loading"); }
});
shareBtn.addEventListener("click", () => { sharePanel.hidden = !sharePanel.hidden; shareTitle.value ||= "untitled"; shareArtist.value ||= "anonymous"; });
document.querySelector("#shareConfirm").addEventListener("click", async () => render((await post("/api/share", {title:shareTitle.value, artist:shareArtist.value})).events));
gallery.addEventListener("click", async e => { if (e.target.matches("button[data-path]")) { render((await post("/api/remix", {path:e.target.dataset.path})).events); scrollTo({top:0, behavior:"smooth"}); } });
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
controlsPanel.addEventListener("click", e => {
  const target = e.target;
  // Palette swatch.
  const sw = target.closest("button[data-palette]");
  if (sw) {
    const row = sw.closest(".ctl-palette");
    const chain = +row.dataset.chain;
    postControl({chain_index: chain, name: "palette", value: sw.dataset.palette});
    return;
  }
  // Enum segment.
  if (target.matches(".ctl-seg")) {
    postControl({chain_index: +target.dataset.chain, name: target.dataset.name, value: JSON.parse(target.dataset.value)});
    return;
  }
  // Pan arrow.
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
    const c = pan.querySelector(".ctl-pan-c"); if (c) c.textContent = `${fmtNum(x)}, ${fmtNum(y)}`;
    postControl({chain_index: +pan.dataset.chain, name: pan.dataset.name, value: {x, y}});
    return;
  }
  // Button (action) widget.
  if (target.matches(".ctl-btn")) {
    postControl({chain_index: +target.dataset.chain, name: target.dataset.name, value: null});
  }
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
loadPalettes(); loadCanvas(); loadGallery(); refreshAccount();

const canvas = document.querySelector("#fractal");
const ctx = canvas.getContext("2d");
let t = 0;
function draw() {
  const w = canvas.width = Math.max(2, Math.floor(canvas.clientWidth * .55 / 2) * 2);
  const h = canvas.height = Math.max(2, Math.floor(canvas.clientHeight * .55 / 2) * 2);
  const img = ctx.createImageData(w, h);
  const zoom = 1.25 + Math.sin(t) * .35;
  const cx = -0.745 + Math.cos(t * .37) * .018;
  const cy = 0.112 + Math.sin(t * .31) * .018;
  for (let y = 0; y < h; y += 2) for (let x = 0; x < w; x += 2) {
    let zx = (x / w - .5) * 3.2 / zoom + cx, zy = (y / h - .5) * 2.35 / zoom + cy, i = 0;
    const ox = zx, oy = zy;
    for (; i < 72 && zx*zx + zy*zy < 4; i++) { const nx = zx*zx - zy*zy + ox; zy = 2*zx*zy + oy; zx = nx; }
    const p = (y*w + x) * 4, a = i / 72, r = 20 + 235*a, g = 30 + 120*Math.sin(a*5+t), b = 80 + 175*(1-a);
    for (const d of [0,4,w*4,w*4+4]) { img.data[p+d]=r; img.data[p+d+1]=g; img.data[p+d+2]=b; img.data[p+d+3]=255; }
  }
  ctx.putImageData(img, 0, 0);
  t += .018;
  setTimeout(draw, 90);
}
draw();
