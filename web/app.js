"use strict";
// Q-Meter Circuit Simulator front end. The model (qmeter.py, deuteron_lineshape.py) and the signal analysis
// (analysis.py) run in a Pyodide worker.

// Mirror of qmeter.DEFAULTS so the page can render before Python loads; replaced by the real values on ready.
// Keys starting with a_ are analysis settings (analysis.ANALYSIS_DEFAULTS), not model parameters.
const DEF = {
  trim_sup: 1e-9, trim_sig: 1e-9, trim_ref: 1e-9, trim_coil: 1e-9, capacitance: 1.0, phase: 0.0, sweep_range: 0.4,
  sig: true, species: "p", pol: 0.0027, b: 5.0, d_A: 0.05, d_wq: 0.027, d_eta: 0.0,
  u: 1.0, r_cc: 681.0, c_stray: 0.001e-12, l_0: 0.175e-6, r_coil: 0.3, r_amp: 50.0, r_source: 50.0, r: 2.0,
  v_f: 0.78, alph: 0.0343, l_cab: 2.54e-7, c_per_m: 95.2e-12, eta: 0.0013, n_steps: 500,
  a_te: 1.0, a_w0: 0.01, a_w1: 0.25, a_w2: 0.75, a_w3: 0.99, a_s0: 0.25, a_s1: 0.75, a_order: 3, a_drift: 0,
};
const GYRO = { p: 42.576, d: 6.53566 };

const TRIMS = [
  { p: "trim_sup", name: "Supply" },
  { p: "trim_sig", name: "Signal" },
  { p: "trim_ref", name: "Reference" },
  { p: "trim_coil", name: "Coil (λ/2)" },
];
const CONSTS = [
  { p: "r_cc", name: "R<sub>cc</sub> constant current (Ω)", step: 1, min: 1e-6 },
  { p: "r", name: "Damping R (Ω)", step: 0.5, min: 0 },
  { p: "l_0", name: "Coil L<sub>0</sub> (µH)", scale: 1e-6, step: 0.005, min: 1e-9 },
  { p: "r_coil", name: "Coil R (Ω)", step: 0.05, min: 0 },
  { p: "c_stray", name: "Stray C (pF)", scale: 1e-12, step: 0.001, min: 1e-9 },
  { p: "r_source", name: "Source Z (Ω)", step: 1, min: 0 },
  { p: "r_amp", name: "Detector Z (Ω)", step: 1, min: 0 },
  { p: "u", name: "RF voltage (V)", step: 0.1, min: 0 },
  { p: "v_f", name: "Cable velocity factor", step: 0.01, min: 0.01, max: 1 },
  { p: "alph", name: "Cable attenuation α", step: 0.001, min: 0 },
  { p: "l_cab", name: "Cable L (µH)", scale: 1e-6, step: 0.01, min: 0 },
  { p: "c_per_m", name: "Cable C (pF/m)", scale: 1e-12, step: 0.1, min: 1e-6 },
  { p: "eta", name: "Filling factor η", step: 0.0001, min: 0 },
  { p: "n_steps", name: "Sweep points", step: 50, min: 10, max: 20000, int: true },
  { p: "d_A", name: "Deuteron broadening A", step: 0.005, min: 0.001, max: 1 },
  { p: "d_wq", name: "Deuteron ω<sub>Q</sub> (kHz)", scale: 1e-3, step: 0.5, min: 1e-4, max: 1 },
  { p: "d_eta", name: "Deuteron EFG asymmetry η", step: 0.01, min: 0, max: 1 },
];

const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const clean = (v) => Number.parseFloat(Number(v).toPrecision(12));

const P = { ...DEF };
let result = null;       // latest {sim, analysis} output, plus the params it was computed from
let pinned = null;       // reference sweep
let hoverF = null;       // shared crosshair frequency (MHz)
let pyReady = false;
let anView = "en";       // analysis plots: enhanced ("en") or TE ("te") sweep

function modelParams() {
  const out = {};
  for (const k of Object.keys(P)) if (!k.startsWith("a_")) out[k] = P[k];
  return out;
}
function analysisSettings() {
  return { te_temp: P.a_te, wings: [P.a_w0, P.a_w1, P.a_w2, P.a_w3], sum_range: [P.a_s0, P.a_s1],
           order: P.a_order, drift_ppm: P.a_drift };
}

// ---------- URL state ----------
function readUrl() {
  const q = new URLSearchParams(location.search);
  for (const [k, v] of q) {
    if (!(k in DEF)) continue;
    if (k === "species") P.species = v === "d" ? "d" : "p";
    else if (k === "sig") P.sig = v !== "0" && v !== "false";
    else if (Number.isFinite(Number(v))) P[k] = Number(v);
  }
}
function writeUrl() {
  const q = new URLSearchParams();
  for (const k of Object.keys(DEF)) {
    if (P[k] === DEF[k]) continue;
    q.set(k, k === "sig" ? (P.sig ? "1" : "0") : String(P[k]));
  }
  const s = q.toString();
  history.replaceState(null, "", s ? "?" + s : location.pathname);
}

// ---------- controls ----------
// Inputs carry data-p (the key in P) and optionally data-scale (P value = shown value × scale),
// data-min / data-max (limits on the P value) and data-int.
function bindInput(el) {
  const key = el.dataset.p, scale = Number(el.dataset.scale || 1);
  const min = el.dataset.min !== undefined ? Number(el.dataset.min) : -Infinity;
  const max = el.dataset.max !== undefined ? Number(el.dataset.max) : Infinity;
  el.addEventListener("input", () => {
    let v = Number.parseFloat(el.value) * scale;
    const ok = el.value.trim() !== "" && Number.isFinite(v) && v >= min && v <= max;
    el.classList.toggle("bad", !ok);
    if (!ok) return;
    P[key] = el.dataset.int !== undefined ? Math.round(v) : clean(v);
    onChange(key);
  });
}

function buildTrims() {
  $("trims").innerHTML = TRIMS.map((t) => `
    <div class="ctl trim">
      <label class="eyebrow" for="p_${t.p}">${t.name}</label>
      <div class="row">
        <input type="number" id="p_${t.p}" data-p="${t.p}" data-min="0" step="0.1" min="0">
        <input type="range" id="r_${t.p}" min="0" max="4" step="0.005" aria-label="${t.name} length slider">
      </div>
      <span class="len mono" id="len_${t.p}"></span>
    </div>`).join("");
  for (const t of TRIMS) {
    $("r_" + t.p).addEventListener("input", (e) => {
      P[t.p] = clean(Number(e.target.value)) || DEF[t.p];
      $("p_" + t.p).value = fmtInput(P[t.p]);
      $("p_" + t.p).classList.remove("bad");
      onChange(t.p);
    });
  }
}

function buildConsts() {
  $("consts").innerHTML = CONSTS.map((c) => `
    <div class="ctl"><label class="eyebrow" for="p_${c.p}">${c.name}</label>
      <input type="number" id="p_${c.p}" data-p="${c.p}" step="${c.step}"
        ${c.scale ? `data-scale="${c.scale}"` : ""} ${c.min !== undefined ? `data-min="${c.min}"` : ""}
        ${c.max !== undefined ? `data-max="${c.max}"` : ""} ${c.int ? "data-int" : ""}></div>`).join("");
}

function fmtInput(v, scale = 1) {
  const x = v / scale;
  if (x !== 0 && Math.abs(x) < 1e-6 && scale === 1) return "0";     // the 1e-9 trim floor reads as zero
  return String(clean(x));
}

function syncControls() {
  for (const el of document.querySelectorAll("input[data-p]")) {
    if (el.classList.contains("bad") || document.activeElement === el) continue;
    el.value = fmtInput(P[el.dataset.p], Number(el.dataset.scale || 1));
  }
  for (const t of TRIMS) $("r_" + t.p).value = Math.min(4, P[t.p]);
  setSeg("speciesSeg", P.species);
  setSeg("sigSeg", P.sig ? "1" : "0");
  setSeg("anViewSeg", anView);
  updateLengths();
}

function setSeg(id, v) {
  for (const b of $(id).querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.v === v));
}

function wireSeg(id, apply, recompute = true) {
  $(id).addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    apply(b.dataset.v);
    syncControls();
    if (recompute) onChange(id); else render();
  });
}

function larmor() { return P.b * GYRO[P.species]; }
function lambda2() { return (P.v_f * 2.99e8) / (2 * larmor() * 1e6); }

function updateLengths() {
  const lb2 = lambda2();
  $("lb2Note").textContent = `λ/2 = ${lb2.toFixed(3)} m at ${larmor().toFixed(3)} MHz, v = ${P.v_f}c`;
  for (const t of TRIMS) $("len_" + t.p).textContent = `${(Math.max(P[t.p], 1e-9) * lb2).toFixed(3)} m`;
}

function onChange(src) {
  if (src !== "tune" && !$("tuneMsg").hidden && !$("tuneMsg").querySelector(".dot.busy")) setTuneMsg("", "");
  updateLengths();
  writeUrl();
  requestCompute();
}

// ---------- worker ----------
const worker = new Worker("worker.js?v=__V__");
let reqId = 0, inflight = false, dirty = false;

function setStatus(kind, text) {
  $("pyDot").className = "dot " + kind;
  $("pyText").textContent = text;
}

let sentParams = null;
function requestCompute() {
  if (!pyReady) return;
  if (inflight) { dirty = true; return; }
  inflight = true; dirty = false;
  sentParams = { ...P };
  worker.postMessage({ type: "compute", id: ++reqId, params: modelParams(), settings: analysisSettings() });
}

const TUNE_BTNS = ["btnTuneDiode", "btnTunePhase"];
function requestTune(mode) {
  if (!pyReady) return;
  for (const id of TUNE_BTNS) $(id).disabled = true;
  setTuneMsg("busy", `Tuning for the ${mode} signal…`);
  worker.postMessage({ type: "tune", id: ++reqId, mode, params: modelParams() });
}
function setTuneMsg(kind, text) {
  const el = $("tuneMsg");
  el.hidden = !text;
  el.querySelector(".dot").className = "dot " + kind;
  el.querySelector("span").textContent = text || "";
}

let readyText = "";
worker.onmessage = (ev) => {
  const m = ev.data;
  if (m.type === "progress") setStatus("busy", m.text);
  else if (m.type === "fatal") setStatus("err", "Python failed to load: " + m.text);
  else if (m.type === "ready") {
    const py = JSON.parse(m.defaults);
    const ad = JSON.parse(m.analysisDefaults);
    Object.assign(py, {
      a_te: ad.te_temp, a_w0: ad.wings[0], a_w1: ad.wings[1], a_w2: ad.wings[2], a_w3: ad.wings[3],
      a_s0: ad.sum_range[0], a_s1: ad.sum_range[1], a_order: ad.order, a_drift: ad.drift_ppm,
    });
    for (const k of Object.keys(py)) {
      if (P[k] === DEF[k]) P[k] = py[k];   // keep anything set from the URL or edited while loading
      DEF[k] = py[k];
    }
    pyReady = true;
    readyText = `Python ready (Pyodide ${m.version}, NumPy ${m.numpy})`;
    setStatus("ok", readyText);
    for (const id of TUNE_BTNS) $(id).disabled = false;
    syncControls();
    requestCompute();
  } else if (m.type === "result" || m.type === "error") {
    inflight = false;
    if (m.type === "result") {
      result = JSON.parse(m.json);
      result.params = sentParams;
      setStatus("ok", readyText);
      $("btnCsv").disabled = false;
      render();
    } else {
      setStatus("err", "Model error: " + m.text);
    }
    if (dirty) requestCompute();
  } else if (m.type === "tuned" || m.type === "tuneError") {
    for (const id of TUNE_BTNS) $(id).disabled = false;
    if (m.type === "tuneError") { setTuneMsg("err", "Can't tune: " + m.text); return; }
    const r = JSON.parse(m.json);
    P.capacitance = Number(r.capacitance.toPrecision(6));   // ~100 Hz of dip position at 213 MHz
    P.phase = Math.round(r.phase * 1000) / 1000;
    for (const id of ["p_capacitance", "p_phase"]) $(id).classList.remove("bad");
    syncControls();
    onChange("tune");
    setTuneMsg("ok", m.mode === "diode"
      ? `Diode tune: capacitance ${P.capacitance} × tune (phase unchanged; it does not affect the diode)`
      : `Phase tune: capacitance ${P.capacitance} × tune, phase ${P.phase}°`);
  }
};
worker.onerror = (e) => setStatus("err", "Worker error: " + (e.message || e));

// ---------- plotting ----------
function niceStep(span, n) {
  const raw = span / Math.max(1, n);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const r = raw / mag;
  return (r < 1.5 ? 1 : r < 3 ? 2 : r < 7 ? 5 : 10) * mag;
}
// Returns HTML (numbers only, so safe for innerHTML).
function fmtSci(v, d = 4) {
  if (!Number.isFinite(v)) return "–";
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e-2 && a < 1e4) return v.toPrecision(d);
  const e = Math.floor(Math.log10(a));
  return `${(v / Math.pow(10, e)).toFixed(d - 1)}×10<sup>${e}</sup>`;
}
const pct = (v, d = 2) => (Number.isFinite(v) ? `${(100 * v).toFixed(d)}%` : "–");

const plots = {};

function setupCanvas(cv) {
  const dpr = window.devicePixelRatio || 1;
  const W = cv.clientWidth, H = cv.clientHeight;
  if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) {
    cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
  }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  return { ctx, W, H };
}

function messageCanvas(id, text) {
  const { ctx, W, H } = setupCanvas($(id));
  ctx.fillStyle = css("--muted"); ctx.font = `13px ${css("--f-body")}`;
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillText(text, W / 2, H / 2);
  delete plots[id];
}

// series: [{y, x?, color, width, dash, label}]; bands: [{x0, x1, color}]; points: [{x, y, color}]
function drawPlot(id, x, series, { marker = null, readEl = null, unit = "V", bands = [], points = [] } = {}) {
  const cv = $(id);
  const { ctx, W, H } = setupCanvas(cv);
  const m = { l: 62, r: 14, t: 22, b: 36 };
  const pw = W - m.l - m.r, ph = H - m.t - m.b;
  if (pw < 20 || ph < 20 || !x || !x.length) return;

  let ymin = Infinity, ymax = -Infinity;
  for (const s of series) for (const v of s.y) if (Number.isFinite(v)) { if (v < ymin) ymin = v; if (v > ymax) ymax = v; }
  if (!Number.isFinite(ymin)) { ymin = -1; ymax = 1; }
  if (ymax - ymin < 1e-300) { const d = Math.abs(ymax) * 0.05 || 1e-12; ymin -= d; ymax += d; }
  const pad = (ymax - ymin) * 0.06; ymin -= pad; ymax += pad;
  const xmin = x[0], xmax = x[x.length - 1];
  const X = (v) => m.l + ((v - xmin) / (xmax - xmin)) * pw;
  const Y = (v) => m.t + (1 - (v - ymin) / (ymax - ymin)) * ph;

  for (const b of bands) {
    ctx.fillStyle = b.color;
    ctx.fillRect(X(b.x0), m.t, X(b.x1) - X(b.x0), ph);
  }

  // common exponent for y labels
  const amax = Math.max(Math.abs(ymin), Math.abs(ymax));
  let ex = Math.floor(Math.log10(amax));
  if (ex >= -2 && ex <= 3) ex = 0;
  const ys = Math.pow(10, ex);

  ctx.font = `11px ${css("--f-mono")}`;
  ctx.lineWidth = 1;
  ctx.strokeStyle = css("--line");
  ctx.fillStyle = css("--muted");

  const ystep = niceStep(ymax - ymin, Math.max(3, Math.floor(ph / 45)));
  const ydec = Math.max(0, -Math.floor(Math.log10(ystep / ys) + 1e-9));
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (let t = Math.ceil(ymin / ystep) * ystep; t <= ymax + ystep * 1e-6; t += ystep) {
    const yy = Math.round(Y(t)) + 0.5;
    ctx.beginPath(); ctx.moveTo(m.l, yy); ctx.lineTo(m.l + pw, yy); ctx.stroke();
    const lab = (Math.abs(t) < ystep * 1e-6 ? 0 : t / ys).toFixed(ydec);
    ctx.fillText(lab, m.l - 6, yy);
  }
  const xstep = niceStep(xmax - xmin, Math.max(3, Math.floor(pw / 80)));
  const xdec = Math.max(0, -Math.floor(Math.log10(xstep) + 1e-9));
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (let t = Math.ceil(xmin / xstep) * xstep; t <= xmax + xstep * 1e-6; t += xstep) {
    const xx = Math.round(X(t)) + 0.5;
    ctx.beginPath(); ctx.moveTo(xx, m.t); ctx.lineTo(xx, m.t + ph); ctx.stroke();
    ctx.fillText(t.toFixed(xdec), xx, m.t + ph + 6);
  }
  ctx.font = `12px ${css("--f-body")}`;
  ctx.fillText("frequency (MHz) →", m.l + pw / 2, m.t + ph + 20);
  ctx.textAlign = "left"; ctx.textBaseline = "bottom";
  const ulab = ex ? `${unit}  ×10` : unit;
  ctx.fillText(ulab, 6, m.t - 6);
  if (ex) {
    const w = ctx.measureText(ulab).width;
    ctx.font = `10px ${css("--f-body")}`;
    ctx.fillText(String(ex).replace("-", "−"), 6 + w + 1, m.t - 11);
  }

  // zero line, Larmor marker
  if (ymin < 0 && ymax > 0) {
    ctx.strokeStyle = css("--muted"); ctx.globalAlpha = 0.6;
    ctx.beginPath(); ctx.moveTo(m.l, Y(0)); ctx.lineTo(m.l + pw, Y(0)); ctx.stroke();
    ctx.globalAlpha = 1;
  }
  if (marker != null && marker >= xmin && marker <= xmax) {
    ctx.strokeStyle = css("--muted"); ctx.setLineDash([3, 4]);
    ctx.beginPath(); ctx.moveTo(X(marker), m.t); ctx.lineTo(X(marker), m.t + ph); ctx.stroke();
    ctx.setLineDash([]);
  }

  ctx.save();
  ctx.beginPath(); ctx.rect(m.l, m.t, pw, ph); ctx.clip();
  ctx.lineJoin = "round";
  for (const s of series) {
    const sx = s.x || x;
    ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 2; ctx.setLineDash(s.dash || []);
    ctx.beginPath();
    let pen = false;
    for (let i = 0; i < sx.length; i++) {
      const v = s.y[i];
      if (!Number.isFinite(v)) { pen = false; continue; }
      if (pen) ctx.lineTo(X(sx[i]), Y(v)); else { ctx.moveTo(X(sx[i]), Y(v)); pen = true; }
    }
    ctx.stroke();
  }
  ctx.setLineDash([]);
  for (const p of points) {
    ctx.fillStyle = p.color; ctx.strokeStyle = css("--canvas-bg"); ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(X(p.x), Y(p.y), 4.5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
  }

  // crosshair
  let read = "";
  if (hoverF != null && hoverF >= xmin && hoverF <= xmax) {
    const xx = Math.round(X(hoverF)) + 0.5;
    ctx.strokeStyle = css("--ink"); ctx.globalAlpha = 0.4; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(xx, m.t); ctx.lineTo(xx, m.t + ph); ctx.stroke();
    ctx.globalAlpha = 1;
    const parts = [`f = ${hoverF.toFixed(4)} MHz`];
    for (const s of series) {
      if (!s.label) continue;
      const sx = s.x || x;
      const i = nearest(sx, hoverF);
      if (i < 0) continue;
      const v = s.y[i];
      if (!Number.isFinite(v)) continue;
      ctx.fillStyle = s.color;
      ctx.beginPath(); ctx.arc(X(sx[i]), Y(v), 3.2, 0, 2 * Math.PI); ctx.fill();
      parts.push(`${s.label} ${fmtSci(v, 5)} ${unit}`);
    }
    read = parts.join(" · ");
  }
  ctx.restore();
  if (readEl) $(readEl).innerHTML = read || "Hover or drag across a plot to read values.";
  plots[id] = { m, pw, xmin, xmax };
}

function nearest(arr, v) {
  if (!arr.length) return -1;
  const step = (arr[arr.length - 1] - arr[0]) / (arr.length - 1);
  return Math.max(0, Math.min(arr.length - 1, Math.round((v - arr[0]) / step)));
}

function render() {
  if (!result) return;
  const r = result.sim, pn = pinned && pinned.sim;
  const cur = (k) => ({ y: r[k], color: css("--copper"), width: 2.2, label: "now" });
  const pin = (k) => pn ? [{ x: pn.f_mhz, y: pn[k], color: css("--shell"), width: 1.6, dash: [6, 4], label: "pinned" }] : [];
  drawPlot("cvDiode", r.f_mhz, [...pin("diode"), cur("diode")], { marker: r.f0_mhz, readEl: "rdDiode" });
  drawPlot("cvPhase", r.f_mhz, [...pin("phase"), cur("phase")], { marker: r.f0_mhz, readEl: "rdPhase" });
  renderAnalysis();
}

// ---------- analysis ----------
function renderAnalysis() {
  const a = result.analysis;
  if (!a) {
    const msg = "Analysis: " + (result.analysis_error || "unavailable");
    messageCanvas("cvBase", msg); messageCanvas("cvSub", msg);
    $("anResults").innerHTML = `<span class="k" style="grid-column:1/-1">${msg.replace(/</g, "&lt;")}</span>`;
    return;
  }
  const f = a.f_mhz, lo = f[0], span = f[f.length - 1] - f[0];
  const fx = (frac) => lo + frac * span;      // sweep fraction -> frequency (index bounds in analysis.py)
  const s = a.settings;
  const wingBand = bandColor("--shell", 0.12), sumBand = bandColor("--copper", 0.1);
  const wings = [{ x0: fx(s.wings[0]), x1: fx(s.wings[1]), color: wingBand }, { x0: fx(s.wings[2]), x1: fx(s.wings[3]), color: wingBand }];
  const sum = [{ x0: fx(s.sum_range[0]), x1: fx(s.sum_range[1]), color: sumBand }];
  const sw = a[anView];
  const label = anView === "te" ? "TE" : "signal";

  drawPlot("cvBase", f, [
    { y: sw.basesub, color: css("--copper"), width: 2, label: `${label} − baseline` },
    { y: sw.fit, color: css("--shell"), width: 1.8, dash: [6, 4], label: "polynomial" },
  ], { marker: a.f0_mhz, readEl: "rdBase", bands: wings });

  const series = [{ y: sw.sub, color: css("--copper"), width: 2, label: "fit-subtracted" }];
  const points = [];
  if (a.species === "d" && anView === "en") {
    if (a.fit) series.push({ y: a.fit.curve, color: css("--fitc"), width: 1.6, dash: [6, 4], label: "absorption fit" });
    if (a.fit_complex) series.push({ y: a.fit_complex.curve, color: css("--shell"), width: 1.6, dash: [2, 3], label: "complex fit" });
    if (a.ratio) {
      points.push({ x: a.ratio.f_hi, y: a.ratio.h_hi, color: css("--ink") });
      points.push({ x: a.ratio.f_lo, y: a.ratio.h_lo, color: css("--ink") });
    }
  }
  drawPlot("cvSub", f, series, { marker: a.f0_mhz, readEl: "rdSub", bands: sum, points });
  $("subLegend").hidden = !(a.species === "d" && anView === "en");
  renderResults(a);
}

function bandColor(token, alpha) {
  // canvas fillStyle doesn't understand color-mix() everywhere, so build rgba from the token
  const c = css(token);
  const m = /^#([0-9a-f]{6})$/i.exec(c);
  if (!m) return `rgba(128,128,128,${alpha})`;
  const n = parseInt(m[1], 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${alpha})`;
}

function chip(err) {
  const e = Math.abs(err);
  const cls = e < 0.01 ? "ok" : e < 0.05 ? "warn" : "bad";
  return `<span class="chip ${cls}">${err >= 0 ? "+" : "−"}${(100 * e).toFixed(1)}%</span>`;
}

function renderResults(a) {
  const relErr = (pm) => (a.pol_true !== 0 ? pm / a.pol_true - 1 : NaN);
  const rows = [
    ["hdr", "Thermal equilibrium calibration"],
    [`TE polarization at ${a.settings.te_temp} K`, pct(a.pol_te, 4)],
    ["TE area A<sub>TE</sub>", fmtSci(a.area_te)],
    ["Calibration constant CC = P<sub>TE</sub>/A<sub>TE</sub>", fmtSci(a.cc)],
    ["hr"],
    ["hdr", "Polarization from the signal"],
    ["True polarization (input)", pct(a.pol_true)],
    ["Signal area A", fmtSci(a.area)],
    ["Area method, P = CC · A", pct(a.pol_area), chip(relErr(a.pol_area))],
  ];
  if (a.species === "d") {
    if (a.ratio) rows.push([`Peak-height ratio, r = ${a.ratio.r.toFixed(4)}`, pct(a.ratio.pol), chip(relErr(a.ratio.pol))]);
    if (a.fit) rows.push([`Absorption lineshape fit (A = ${a.fit.A.toFixed(3)}, ω<sub>Q</sub> = ${(a.fit.wq * 1e3).toFixed(2)} kHz, ξ = ${a.fit.xi.toFixed(3)})`, pct(a.fit.pol), chip(relErr(a.fit.pol))]);
    if (a.fit_complex) rows.push([`Complex lineshape fit (phase ${a.fit_complex.phase_deg.toFixed(2)}°)`, pct(a.fit_complex.pol), chip(relErr(a.fit_complex.pol))]);
  }
  $("anResults").innerHTML = rows.map((row) => {
    if (row[0] === "hr") return "<hr>";
    if (row[0] === "hdr") return `<span class="hdr" style="grid-column:1/-1">${row[1]}</span>`;
    return `<span class="k">${row[0]}</span><span class="v">${row[1]}</span><span class="e">${row[2] || ""}</span>`;
  }).join("");
}

// crosshair interaction, shared by all plots
for (const id of ["cvDiode", "cvPhase", "cvBase", "cvSub"]) {
  const cv = $(id);
  const move = (e) => {
    const g = plots[id];
    if (!g) return;
    const rect = cv.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const t = (px - g.m.l) / g.pw;
    hoverF = t < 0 || t > 1 ? null : g.xmin + t * (g.xmax - g.xmin);
    render();
  };
  cv.addEventListener("pointermove", move);
  cv.addEventListener("pointerdown", move);
  cv.addEventListener("pointerleave", (e) => { if (e.pointerType === "mouse") { hoverF = null; render(); } });
}

// ---------- actions ----------
$("btnPin").addEventListener("click", () => {
  if (!result) return;
  pinned = result;
  $("btnUnpin").hidden = false; $("pinLegend").hidden = false;
  render();
});
$("btnUnpin").addEventListener("click", () => {
  pinned = null;
  $("btnUnpin").hidden = true; $("pinLegend").hidden = true;
  render();
});
$("btnReset").addEventListener("click", () => {
  Object.assign(P, DEF);
  for (const el of document.querySelectorAll("input.bad")) el.classList.remove("bad");
  syncControls();
  onChange();
});
$("btnLink").addEventListener("click", async () => {
  const b = $("btnLink");
  try { await navigator.clipboard.writeText(location.href); b.textContent = "Link copied"; }
  catch { b.textContent = "Copy failed: use the address bar"; }
  setTimeout(() => { b.textContent = "Copy link to settings"; }, 1800);
});
$("btnCsv").addEventListener("click", () => {
  if (!result) return;
  const r = result.sim, a = result.analysis;
  const cols = [["f_mhz", r.f_mhz], ["diode", r.diode], ["phase", r.phase]];
  if (a) {
    cols.push(["baseline_subtracted", a.en.basesub], ["polynomial_fit", a.en.fit], ["fit_subtracted", a.en.sub]);
    cols.push(["te_baseline_subtracted", a.te.basesub], ["te_fit_subtracted", a.te.sub]);
  }
  const head = "# Q-Meter Circuit Simulator, https://github.com/jdmax/QMeterSim\n# " +
    Object.entries(result.params).map(([k, v]) => `${k}=${v}`).join(" ") + "\n" + cols.map((c) => c[0]).join(",") + "\n";
  const body = r.f_mhz.map((_, i) => cols.map((c) => c[1][i]).join(",")).join("\n");
  const url = URL.createObjectURL(new Blob([head + body + "\n"], { type: "text/csv" }));
  const el = Object.assign(document.createElement("a"), { href: url, download: "qmeter_sim.csv" });
  document.body.appendChild(el); el.click(); el.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

$("btnTuneDiode").addEventListener("click", () => requestTune("diode"));
$("btnTunePhase").addEventListener("click", () => requestTune("phase"));

for (const b of document.querySelectorAll("[data-example]")) {
  b.addEventListener("click", () => {
    const [species, pol] = b.dataset.example.split(":");
    P.species = species; P.pol = Number(pol);
    syncControls();
    onChange("example");
  });
}

// ---------- equations ----------
function renderEquations() {
  const els = document.querySelectorAll("[data-tex]");
  if (!window.katex) {         // CDN unavailable: show the TeX source as plain text
    for (const el of els) el.textContent = el.dataset.tex;
    return;
  }
  for (const el of els) {
    katex.render(el.dataset.tex, el, { displayMode: el.tagName === "DIV", throwOnError: false });
  }
}

// ---------- init ----------
renderEquations();
readUrl();
buildTrims();
buildConsts();
for (const el of document.querySelectorAll("input[data-p]")) bindInput(el);
wireSeg("speciesSeg", (v) => { P.species = v; });
wireSeg("sigSeg", (v) => { P.sig = v === "1"; });
wireSeg("anViewSeg", (v) => { anView = v; }, false);
{
  const q = new URLSearchParams(location.search);
  $("constPanel").open = CONSTS.some((c) => q.has(c.p));
}
syncControls();

new ResizeObserver(() => render()).observe(document.querySelector(".wrap"));
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
new MutationObserver(render).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
// collapsed sections have zero-size canvases; redraw when one opens
for (const d of document.querySelectorAll("details")) d.addEventListener("toggle", render);
