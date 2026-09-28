"use strict";

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
};
const svgIcon = (paths, cls = "icon") =>
  `<svg viewBox="0 0 24 24" class="${cls}">${paths}</svg>`;
const ICONS = {
  file: '<path d="M14 3H7a2 2 0 00-2 2v14a2 2 0 002 2h10a2 2 0 002-2V8z"/><path d="M14 3v5h5M9 13h6M9 17h6"/>',
  check: '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
  x: '<path d="M6 6l12 12M18 6L6 18"/>',
  code: '<path d="M9 8l-5 4 5 4M15 8l5 4-5 4"/>',
  download: '<path d="M12 4v11M7 10l5 5 5-5M5 20h14"/>',
  bars: '<path d="M6 17V11M10 17V7M14 17v-4M18 17V9"/>',
};

const state = {
  sid: null,
  source: null,
  busy: false,
  turn: null,        // current assistant turn { body, last: {type, el}, cells: Map }
  datasets: [],
  hasUserMessage: false,
  config: null,
};
let markReady;
state.ready = new Promise((resolve) => (markReady = resolve));

const thread = $("#thread");
const scroller = $("#scroller");
const input = $("#input");
const sendBtn = $("#sendBtn");

/* ---------------- Utilities ---------------- */

function sanitize(html) {
  return window.DOMPurify ? DOMPurify.sanitize(html) : escapeHtml(html);
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function renderMarkdown(md) {
  if (window.marked) return sanitize(marked.parse(md, { gfm: true, breaks: false }));
  return `<p>${escapeHtml(md).replace(/\n/g, "<br>")}</p>`;
}
function highlight(code) {
  if (window.hljs) {
    try { return hljs.highlight(code, { language: "python", ignoreIllegals: true }).value; } catch (_) { /* fall through */ }
  }
  return escapeHtml(code);
}
function fmtNum(n) {
  return typeof n === "number" ? n.toLocaleString() : n;
}
function fmtSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}
function fmtStat(v) {
  if (v == null) return "–";
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e6 || a < 1e-3)) return v.toExponential(2);
  return Number(v.toFixed(a >= 100 ? 1 : 3)).toLocaleString();
}

let toastTimer;
function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), 4000);
}

/* Keep the view pinned to the bottom while streaming, unless the user scrolled up. */
let pinned = true;
scroller.addEventListener("scroll", () => {
  pinned = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 120;
});
let scrollQueued = false;
function autoScroll(force = false) {
  if (!(pinned || force) || scrollQueued) return;
  scrollQueued = true;
  requestAnimationFrame(() => {
    scroller.scrollTop = scroller.scrollHeight;
    scrollQueued = false;
  });
}

/* ---------------- Rendering ---------------- */

function append(node) {
  $("#empty").hidden = true;
  thread.appendChild(node);
  autoScroll();
  return node;
}

function resetThread() {
  thread.querySelectorAll(":scope > :not(#empty)").forEach((n) => n.remove());
  $("#empty").hidden = false;
  state.turn = null;
  state.datasets = [];
  state.hasUserMessage = false;
  renderDatasets();
}

function startTurn() {
  const wrap = el("div", "turn");
  const avatar = el("div", "avatar");
  avatar.innerHTML = svgIcon(ICONS.bars, "");
  const body = el("div", "turn-body");
  wrap.append(avatar, body);
  append(wrap);
  state.turn = { wrap, body, last: null, cells: new Map(), working: null };
  return state.turn;
}

function ensureTurn() {
  return state.turn || startTurn();
}

function setWorking(on, label = "Analyzing…") {
  const turn = state.turn;
  if (!turn) return;
  if (turn.working) { turn.working.remove(); turn.working = null; }
  if (on) {
    turn.working = el("div", "working");
    turn.working.innerHTML = `<span class="pulse"><i></i><i></i><i></i></span><span>${escapeHtml(label)}</span>`;
    turn.body.appendChild(turn.working);
    autoScroll();
  }
}

function addBlock(node, type) {
  const turn = ensureTurn();
  if (turn.working) turn.body.insertBefore(node, turn.working);
  else turn.body.appendChild(node);
  turn.last = { type, el: node };
  autoScroll();
  return node;
}

function appendText(text) {
  const turn = ensureTurn();
  if (!turn.last || turn.last.type !== "text") {
    const node = el("div", "md");
    node._md = "";
    addBlock(node, "text");
  }
  const node = turn.last.el;
  node._md += text;
  if (!node._queued) {
    node._queued = true;
    requestAnimationFrame(() => {
      node._queued = false;
      node.innerHTML = renderMarkdown(node._md);
      autoScroll();
    });
  }
}

function appendThinking(text) {
  const turn = ensureTurn();
  if (!turn.last || turn.last.type !== "thinking") {
    const d = el("details", "thinking");
    d.innerHTML = `<summary>Reasoning</summary><div class="thinking-text"></div>`;
    addBlock(d, "thinking");
  }
  turn.last.el.querySelector(".thinking-text").textContent += text;
  autoScroll();
}

function getCell(id) {
  const turn = ensureTurn();
  let cell = turn.cells.get(id);
  if (cell) return cell;
  const node = el("div", "cell collapsed cell-streaming");
  node.innerHTML = `
    <div class="cell-head">
      <span class="cell-status"><span class="spinner"></span></span>
      <span class="cell-title">Writing code…</span>
      <span class="cell-meta"></span>
      <button class="cell-toggle" type="button">${svgIcon(ICONS.code)}<span>Code</span></button>
    </div>
    <pre class="cell-code"><code></code></pre>
    <div class="cell-out"></div>`;
  node.querySelector(".cell-toggle").addEventListener("click", () => node.classList.toggle("collapsed"));
  addBlock(node, "cell");
  cell = {
    node,
    title: node.querySelector(".cell-title"),
    code: node.querySelector(".cell-code code"),
    status: node.querySelector(".cell-status"),
    meta: node.querySelector(".cell-meta"),
    out: node.querySelector(".cell-out"),
  };
  turn.cells.set(id, cell);
  return cell;
}

function onCodeDelta(evt) {
  const cell = getCell(evt.id);
  if (evt.title) cell.title.textContent = evt.title;
  cell.code.textContent = evt.code;
  const pre = cell.code.parentElement;
  pre.scrollTop = pre.scrollHeight;
}

function onCode(evt) {
  const cell = getCell(evt.id);
  cell.node.classList.remove("cell-streaming");
  cell.title.textContent = evt.title;
  cell.code.innerHTML = highlight(evt.code);
  cell.meta.textContent = "Running…";
  setWorking(true, "Running code…");
}

function onResult(evt) {
  const cell = getCell(evt.id);
  cell.node.classList.remove("cell-streaming");
  if (!evt.code && cell.title.textContent === "Writing code…") cell.title.textContent = "Invalid code cell (retrying)";
  const failed = Boolean(evt.error);
  cell.status.innerHTML = failed
    ? `<span class="err">${svgIcon(ICONS.x)}</span>`
    : `<span class="ok">${svgIcon(ICONS.check)}</span>`;
  const bits = [];
  if (evt.duration != null) bits.push(`${evt.duration}s`);
  if (evt.timed_out) bits.push("timed out");
  if (evt.interrupted) bits.push("stopped");
  cell.meta.textContent = bits.join(" · ");

  const out = cell.out;
  if (evt.stdout) out.appendChild(el("pre", "out out-stream", evt.stdout.replace(/\s+$/, "")));
  (evt.text || []).forEach((t) => out.appendChild(el("pre", "out out-stream", t)));
  (evt.html || []).forEach((h) => {
    const d = el("div", "out out-html");
    d.innerHTML = sanitize(h);
    out.appendChild(d);
  });
  (evt.images || []).forEach((src) => {
    const d = el("div", "out out-image");
    const img = el("img");
    img.src = src;
    img.alt = cell.title.textContent;
    img.loading = "lazy";
    img.addEventListener("load", () => autoScroll());
    img.addEventListener("click", () => openLightbox(src));
    d.appendChild(img);
    out.appendChild(d);
  });
  if (evt.stderr && !failed) out.appendChild(el("pre", "out out-stream out-stderr", evt.stderr.replace(/\s+$/, "")));
  if (failed) {
    const d = el("div", "out out-error");
    d.appendChild(el("pre", null, evt.error));
    out.appendChild(d);
  }
  if (evt.files && evt.files.length) {
    const d = el("div", "out out-files");
    evt.files.forEach((f) => {
      const a = el("a", "file-chip");
      a.href = f.url;
      a.download = f.name.split("/").pop();
      a.innerHTML = `${svgIcon(ICONS.download)}<span>${escapeHtml(f.name)}</span><small>${fmtSize(f.size)}</small>`;
      d.appendChild(a);
    });
    out.appendChild(d);
  }
  setWorking(state.busy, "Analyzing…");
  autoScroll();
}

function onDataset(evt) {
  state.turn = null;
  state.datasets.push({ name: evt.name, var: evt.var, rows: evt.rows, cols: evt.cols });
  renderDatasets();
  renderSuggestions();

  const p = evt.profile;
  const card = el("div", "dataset-card");
  const missingCells = p.columns.reduce((s, c) => s + c.missing, 0);
  const totalCells = Math.max(1, p.rows * p.columns.length);
  card.innerHTML = `
    <div class="dataset-card-head">
      <div class="file-badge">${svgIcon(ICONS.file)}</div>
      <div class="grow">
        <div class="title">${escapeHtml(evt.name)}</div>
        <div class="sub">Loaded as <span class="var-chip">${escapeHtml(evt.var)}</span> — ready for questions</div>
      </div>
    </div>
    <div class="stat-row">
      <div class="stat"><span class="v">${fmtNum(p.rows)}</span><span class="k">rows</span></div>
      <div class="stat"><span class="v">${fmtNum(p.cols)}</span><span class="k">columns</span></div>
      <div class="stat"><span class="v">${((100 * missingCells) / totalCells).toFixed(1)}%</span><span class="k">missing cells</span></div>
      <div class="stat"><span class="v">${fmtNum(p.duplicate_rows)}</span><span class="k">duplicate rows</span></div>
      <div class="stat"><span class="v">${p.memory_mb} MB</span><span class="k">in memory</span></div>
    </div>
    <details><summary>Preview first rows</summary><div class="preview"></div></details>
    <details><summary>Column summary</summary><div class="preview cols"></div></details>`;
  card.querySelector(".preview").innerHTML = sanitize(p.head_html);

  const table = el("table", "col-table");
  table.innerHTML = `<thead><tr><th>Column</th><th>Type</th><th>Missing</th><th>Unique</th><th>Summary</th></tr></thead>`;
  const tbody = el("tbody");
  p.columns.forEach((c) => {
    const tr = el("tr");
    const pct = p.rows ? (100 * c.missing) / p.rows : 0;
    const summary = "mean" in c
      ? `min ${fmtStat(c.min)} · median ${fmtStat(c["50%"])} · max ${fmtStat(c.max)}`
      : Object.entries(c.top || {}).slice(0, 3).map(([k, v]) => `${k.length > 18 ? k.slice(0, 17) + "…" : k} (${v})`).join(", ");
    tr.innerHTML = `<td class="mono"></td><td class="mono"></td>
      <td class="num"><span class="miss-bar"><i style="width:${Math.min(100, pct).toFixed(0)}%"></i></span>${pct.toFixed(1)}%</td>
      <td class="num">${fmtNum(c.unique)}</td><td></td>`;
    tr.children[0].textContent = c.name;
    tr.children[1].textContent = c.dtype;
    tr.children[4].textContent = summary;
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  card.querySelector(".cols").appendChild(table);
  append(card);
}

function renderDatasets() {
  const list = $("#datasetList");
  list.innerHTML = "";
  state.datasets.forEach((d) => {
    const li = el("li", "dataset-item");
    li.innerHTML = `<span class="file-icon">${svgIcon(ICONS.file)}</span>
      <div class="meta"><div class="name"></div>
      <div class="dims">${fmtNum(d.rows)} × ${fmtNum(d.cols)} · <span class="var-chip"></span></div></div>`;
    li.querySelector(".name").textContent = d.name;
    li.querySelector(".name").title = d.name;
    li.querySelector(".var-chip").textContent = d.var;
    list.appendChild(li);
  });
}

function renderSuggestions() {
  const box = $("#suggestions");
  box.innerHTML = "";
  if (!state.datasets.length || state.hasUserMessage || state.busy) return;
  const d = state.datasets[state.datasets.length - 1];
  const items = [
    `Give me an overview of ${d.name}: data quality, key distributions and notable patterns`,
    "Which variables are most strongly related? Visualize the relationships",
    "What are the most interesting insights in this data?",
    "Build a predictive model for the most important outcome in this data and explain what drives it",
  ];
  items.forEach((text) => {
    const b = el("button", "suggestion", text);
    b.type = "button";
    b.addEventListener("click", () => sendMessage(text));
    box.appendChild(b);
  });
}

function setBusy(busy) {
  state.busy = busy;
  sendBtn.classList.toggle("stop", busy);
  sendBtn.setAttribute("aria-label", busy ? "Stop" : "Send");
  sendBtn.title = busy ? "Stop the analysis" : "Send";
  renderSuggestions();
  updateSendEnabled();
}

function updateSendEnabled() {
  sendBtn.disabled = !state.busy && !input.value.trim();
}

function openLightbox(src) {
  const lb = $("#lightbox");
  lb.querySelector("img").src = src;
  lb.hidden = false;
}
$("#lightbox").addEventListener("click", () => ($("#lightbox").hidden = true));
document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("#lightbox").hidden = true; });

/* ---------------- Event handling ---------------- */

function handle(evt) {
  switch (evt.type) {
    case "reset":
      resetThread();
      evt.events.forEach(handle);
      setBusy(evt.busy);
      if (evt.busy) { ensureTurn(); setWorking(true); }
      autoScroll(true);
      break;
    case "user": {
      state.turn = null;
      state.hasUserMessage = true;
      append(el("div", "msg-user", evt.text));
      $("#topbarTitle").textContent = evt.text.length > 80 ? evt.text.slice(0, 79) + "…" : evt.text;
      renderSuggestions();
      break;
    }
    case "dataset": onDataset(evt); break;
    case "turn_start":
      startTurn();
      setBusy(true);
      setWorking(true, "Thinking…");
      break;
    case "thinking_delta": appendThinking(evt.text); break;
    case "text_delta": appendText(evt.text); break;
    case "code_delta": onCodeDelta(evt); break;
    case "code": onCode(evt); break;
    case "result": onResult(evt); break;
    case "notice": addBlock(el("div", "notice", evt.text), "notice"); break;
    case "error":
      if (state.turn) addBlock(el("div", "error-box", evt.message), "error");
      else append(el("div", "error-box", evt.message));
      break;
    case "turn_end":
      if (state.turn) {
        setWorking(false);
        // Any cell still marked running was abandoned (stopped / failed).
        state.turn.cells.forEach((c) => {
          if (c.status.querySelector(".spinner")) { c.status.innerHTML = ""; c.meta.textContent = "not run"; }
        });
        if (evt.status === "stopped") addBlock(el("div", "stopped", "Analysis stopped."), "notice");
      }
      state.turn = null;
      setBusy(false);
      break;
  }
}

/* ---------------- Session & network ---------------- */

function connect() {
  if (state.source) state.source.close();
  const conn = $("#conn");
  const src = new EventSource(`/api/sessions/${state.sid}/stream`);
  state.source = src;
  src.onopen = () => { conn.className = "conn ok"; conn.querySelector(".conn-label").textContent = "Connected"; };
  src.onerror = async () => {
    conn.className = "conn bad";
    conn.querySelector(".conn-label").textContent = "Reconnecting";
    // If the server restarted, the session is gone: start a fresh one.
    try {
      const r = await fetch(`/api/sessions/${state.sid}`);
      if (r.status === 404) { src.close(); await newSession(false); }
    } catch (_) { /* server down; EventSource keeps retrying */ }
  };
  src.onmessage = (m) => handle(JSON.parse(m.data));
}

async function newSession(closePrevious = true) {
  const prev = closePrevious && state.sid ? `?previous=${encodeURIComponent(state.sid)}` : "";
  const r = await fetch(`/api/sessions${prev}`, { method: "POST" });
  if (!r.ok) { toast("Could not start a new session"); return; }
  const s = await r.json();
  state.sid = s.id;
  markReady();
  try { localStorage.setItem("analytics.sid", s.id); } catch (_) { /* storage unavailable */ }
  $("#topbarTitle").textContent = "New analysis";
  resetThread();
  connect();
}

async function init() {
  try {
    const cfg = await (await fetch("/api/config")).json();
    state.config = cfg;
    $("#modelInfo").innerHTML = `Model <code>${escapeHtml(cfg.model)}</code> · effort ${escapeHtml(cfg.effort)}<br>Runs locally in your Python environment`;
    if (!cfg.has_credentials) {
      const b = $("#banner");
      b.innerHTML = "No Claude API key found. Add <code>ANTHROPIC_API_KEY=...</code> to the <code>.env</code> file in the project folder and restart the server. You can still upload and preview data.";
      b.hidden = false;
    }
  } catch (_) { /* non-fatal */ }

  let sid = null;
  try { sid = localStorage.getItem("analytics.sid"); } catch (_) { /* storage unavailable */ }
  if (sid) {
    const r = await fetch(`/api/sessions/${sid}`);
    if (r.ok) { state.sid = sid; markReady(); connect(); return; }
  }
  await newSession(false);
}

async function sendMessage(text) {
  text = (text ?? input.value).trim();
  if (!text || state.busy) return;
  await state.ready;
  input.value = "";
  autosize();
  pinned = true;
  const r = await fetch(`/api/sessions/${state.sid}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: text }),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    toast(err.detail || "Could not send the message");
    input.value = text;
    autosize();
  }
}

async function stopAnalysis() {
  await fetch(`/api/sessions/${state.sid}/stop`, { method: "POST" });
}

async function uploadFiles(fileList) {
  const files = Array.from(fileList || []);
  if (!files.length) return;
  if (state.busy) { toast("Wait for the current analysis to finish before uploading"); return; }
  await state.ready;
  const status = $("#uploadStatus");
  const names = files.map((f) => f.name).join(", ");
  status.textContent = `Uploading and profiling ${names}…`;
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  try {
    const r = await fetch(`/api/sessions/${state.sid}/upload`, { method: "POST", body: form });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      toast(err.detail || "Upload failed");
    }
  } catch (_) {
    toast("Upload failed — is the server running?");
  } finally {
    status.textContent = "Enter to send · Shift+Enter for a new line";
    $("#fileInput").value = "";
    input.focus();
  }
}

/* ---------------- Wiring ---------------- */

function autosize() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 200) + "px";
  updateSendEnabled();
}
input.addEventListener("input", autosize);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    sendMessage();
  }
});
$("#composer").addEventListener("submit", (e) => {
  e.preventDefault();
  if (state.busy) stopAnalysis();
  else sendMessage();
});

const pickFile = () => $("#fileInput").click();
$("#attachBtn").addEventListener("click", pickFile);
$("#sideUpload").addEventListener("click", pickFile);
$("#emptyUpload").addEventListener("click", pickFile);
$("#fileInput").addEventListener("change", (e) => uploadFiles(e.target.files));
$("#newSession").addEventListener("click", () => {
  document.body.classList.remove("sidebar-open");
  newSession(true);
});
$("#menuBtn").addEventListener("click", () => document.body.classList.toggle("sidebar-open"));
$("#scroller").addEventListener("click", () => document.body.classList.remove("sidebar-open"));

let dragDepth = 0;
window.addEventListener("dragenter", (e) => {
  if (!e.dataTransfer || !Array.from(e.dataTransfer.types).includes("Files")) return;
  e.preventDefault();
  dragDepth++;
  document.body.classList.add("dragging");
});
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("dragleave", () => {
  dragDepth = Math.max(0, dragDepth - 1);
  if (!dragDepth) document.body.classList.remove("dragging");
});
window.addEventListener("drop", (e) => {
  e.preventDefault();
  dragDepth = 0;
  document.body.classList.remove("dragging");
  if (e.dataTransfer && e.dataTransfer.files.length) uploadFiles(e.dataTransfer.files);
});

autosize();
init();
