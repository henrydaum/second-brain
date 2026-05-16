const sid = localStorage.sbDemoSession || (localStorage.sbDemoSession = crypto.randomUUID());
const messages = document.querySelector("#messages");
const form = document.querySelector("#chatForm");
const input = document.querySelector("#chatInput");
const showcase = document.querySelector(".showcase");
const heroImage = document.querySelector("#heroImage");
const downloadImage = document.querySelector("#downloadImage");
const shareBtn = document.querySelector("#shareImage");
const sharePanel = document.querySelector("#sharePanel");
const shareTitle = document.querySelector("#shareTitle");
const shareArtist = document.querySelector("#shareArtist");
const gallery = document.querySelector("#gallery");
const stats = document.querySelector("#canvasStats");
const history = document.querySelector("#history");
const bottom = () => requestAnimationFrame(() => messages.scrollTop = messages.scrollHeight);
const add = (role, text) => {
  const el = document.createElement("article");
  el.className = role;
  el.textContent = text;
  messages.appendChild(el);
  bottom();
};
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
function render(events) {
  for (const ev of events || []) {
    if (ev.type === "message") add("assistant", ev.content);
    else if (ev.type === "status") add("status", ev.content);
    else if (ev.type === "error") add("error", ev.content);
    else if (ev.type === "form") add("assistant", `${ev.form?.display?.prompt || "Input required"}\n${(ev.form?.display?.choices || []).map(c => c.label || c.value).join(" / ")}`);
    else if (ev.type === "approval") approval(ev);
    else if (ev.type === "hero_image") {
      setCanvas(ev.canvas || {url: ev.url, name: ev.name});
      add("status", `Showcase updated: ${ev.name}`);
    }
    else if (ev.type === "canvas_reset") setCanvas(null);
    else if (ev.type === "shared") { sharePanel.hidden = true; loadGallery(); }
    else if (ev.type === "attachment") add("assistant", `Attachment: ${ev.name}`);
  }
  bottom();
}
function setCanvas(c) {
  if (!c?.url) {
    showcase.classList.remove("has-image"); heroImage.removeAttribute("src"); downloadImage.href = "#";
    stats.textContent = "Type anything. The system will turn it into a procedural base, color harmony, DeepDream texture, and final polish.";
    history.innerHTML = "<span>Fresh canvas</span>"; return;
  }
  heroImage.src = c.url; heroImage.alt = c.name || "Generated canvas"; downloadImage.href = c.url; downloadImage.download = c.name || "canvas.png"; showcase.classList.add("has-image");
  const s = c.stats || {}, g = (s.guidance || []).join(", ");
  stats.textContent = s.brightness == null ? "Canvas updated." : `Beauty ${s.beauty_score ?? "?"} · brightness ${s.brightness} · contrast ${s.contrast} · detail ${s.detail}${g ? ` · ${g}` : ""}`;
  history.innerHTML = (c.history || []).slice(-5).map(x => `<span>${x.op || "image"}</span>`).join("") || "<span>Canvas</span>";
  loadGallery();
}
async function loadCanvas() { const r = await get("/api/canvas"); setCanvas(r.canvas); }
async function loadGallery() {
  const r = await get("/api/gallery");
  gallery.innerHTML = (r.items || []).map(x => `<article class="gallery-card"><img src="${x.url}" alt=""><div><strong>${esc(x.title)}</strong><small>${esc(x.artist)}${x.score ? ` · ${(x.score*100).toFixed(0)}% similar` : ""}</small><button data-path="${esc(x.path)}">Remix</button></div></article>`).join("") || "<article class='assistant'>No shared canvases yet.</article>";
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
  bottom();
  try { const result = await post("/api/chat", {message:text}); thinking.remove(); render(result.events); }
  catch (err) { add("error", err.message); }
});
document.querySelector("#newChat").addEventListener("click", async () => {
  messages.innerHTML = "";
  render((await post("/api/new")).events);
  loadGallery();
});
shareBtn.addEventListener("click", () => { sharePanel.hidden = !sharePanel.hidden; shareTitle.value ||= "untitled"; shareArtist.value ||= "anonymous"; });
document.querySelector("#shareConfirm").addEventListener("click", async () => render((await post("/api/share", {title:shareTitle.value, artist:shareArtist.value})).events));
gallery.addEventListener("click", async e => { if (e.target.matches("button[data-path]")) { render((await post("/api/remix", {path:e.target.dataset.path})).events); scrollTo({top:0, behavior:"smooth"}); } });
setInterval(poll, 1200);
loadCanvas(); loadGallery();

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
