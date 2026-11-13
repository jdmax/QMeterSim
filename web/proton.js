"use strict";
// The proton line at high polarization: measured lines (proton_data.json, made by tools/make_proton_data.py)
// and the simulator's model g(x; P) = A0(x - sP) + P A1(x - sP).

const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const isDark = () => {
  const t = document.documentElement.dataset.theme;
  return t === "dark" || (t !== "light" && matchMedia("(prefers-color-scheme: dark)").matches);
};

// one-hue sequential ramp for polarization: light -> dark on a light surface, dark -> light on a dark one
const RAMP_LIGHT = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"];
const RAMP_DARK = ["#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5", "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#cde2fb"];
function hexRgb(h) { const n = parseInt(h.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
function rampColor(t) {
  const ramp = isDark() ? RAMP_DARK : RAMP_LIGHT;
  const u = Math.max(0, Math.min(1, t)) * (ramp.length - 1);
  const i = Math.min(ramp.length - 2, Math.floor(u)), f = u - i;
  const a = hexRgb(ramp[i]), b = hexRgb(ramp[i + 1]);
  return `rgb(${a.map((v, k) => Math.round(v + f * (b[k] - v))).join(",")})`;
}

const pct = (p, d = 1) => `${(100 * p).toFixed(d)}%`;
const sgn = (v, d = 1) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(d)}`;

// ---------- plotting ----------
function niceStep(range, n) {
  const raw = range / Math.max(1, n), p = Math.pow(10, Math.floor(Math.log10(raw))), m = raw / p;
  return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
}
function setupCanvas(cv) {
  const r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  cv.width = Math.max(1, Math.round(r.width * dpr)); cv.height = Math.max(1, Math.round(r.height * dpr));
  const ctx = cv.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = css("--canvas-bg"); ctx.fillRect(0, 0, r.width, r.height);
  return { ctx, W: r.width, H: r.height };
}
const hover = {};       // canvas id -> {px, py} in CSS pixels
const geom = {};        // canvas id -> mapping for hover

// spec: {series:[{x,y,color,width,dash,alpha}], points:[{x,y,color,r}], xLabel, yLabel, xFmt, yFmt,
//        xmin,xmax,ymin,ymax, zero, onHover(xv, yv) -> html}
function plot(id, spec) {
  const cv = $(id);
  const { ctx, W, H } = setupCanvas(cv);
  const m = { l: 54, r: 12, t: 14, b: 38 };
  const pw = W - m.l - m.r, ph = H - m.t - m.b;
  if (pw < 20 || ph < 20) return;
  let xmin = spec.xmin, xmax = spec.xmax, ymin = spec.ymin, ymax = spec.ymax;
  const all = [...spec.series.map((s) => ({ x: s.x, y: s.y })), ...(spec.points || []).map((p) => ({ x: [p.x], y: [p.y] }))];
  if (xmin == null || xmax == null) {
    let a = Infinity, b = -Infinity;
    for (const s of all) for (const v of s.x) { if (v < a) a = v; if (v > b) b = v; }
    if (xmin == null) xmin = a; if (xmax == null) xmax = b;
  }
  if (ymin == null || ymax == null) {
    let a = Infinity, b = -Infinity;
    for (const s of all) for (let i = 0; i < s.y.length; i++) {
      const xv = s.x[i], v = s.y[i];
      if (xv < xmin || xv > xmax || !Number.isFinite(v)) continue;
      if (v < a) a = v; if (v > b) b = v;
    }
    if (spec.zero) { a = Math.min(a, 0); b = Math.max(b, 0); }
    if (b - a < 1e-12) { a -= 1; b += 1; }
    const pad = (b - a) * 0.07;
    if (ymin == null) ymin = a - (spec.zero && a === 0 ? 0 : pad);
    if (ymax == null) ymax = b + pad;
  }
  const X = (v) => m.l + ((v - xmin) / (xmax - xmin)) * pw;
  const Y = (v) => m.t + (1 - (v - ymin) / (ymax - ymin)) * ph;
  geom[id] = { m, pw, ph, xmin, xmax, ymin, ymax };

  ctx.font = `11px ${css("--f-mono")}`;
  ctx.lineWidth = 1; ctx.strokeStyle = css("--line"); ctx.fillStyle = css("--muted");
  const ys = niceStep(ymax - ymin, Math.max(3, Math.floor(ph / 42)));
  const ydec = Math.max(0, -Math.floor(Math.log10(ys) + 1e-9));
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (let t = Math.ceil(ymin / ys) * ys; t <= ymax + ys * 1e-6; t += ys) {
    const yy = Math.round(Y(t)) + 0.5;
    ctx.beginPath(); ctx.moveTo(m.l, yy); ctx.lineTo(m.l + pw, yy); ctx.stroke();
    ctx.fillText(spec.yFmt ? spec.yFmt(t) : (Math.abs(t) < ys * 1e-6 ? 0 : t).toFixed(ydec).replace("-", "−"), m.l - 6, yy);
  }
  const xs = spec.xStep || niceStep(xmax - xmin, Math.max(3, Math.floor(pw / 75)));
  const xdec = Math.max(0, -Math.floor(Math.log10(xs) + 1e-9));
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (let t = Math.ceil(xmin / xs) * xs; t <= xmax + xs * 1e-6; t += xs) {
    const xx = Math.round(X(t)) + 0.5;
    ctx.beginPath(); ctx.moveTo(xx, m.t); ctx.lineTo(xx, m.t + ph); ctx.stroke();
    ctx.fillText(spec.xFmt ? spec.xFmt(t) : (Math.abs(t) < xs * 1e-6 ? 0 : t).toFixed(xdec).replace("-", "−"), xx, m.t + ph + 6);
  }
  ctx.font = `12px ${css("--f-body")}`;
  ctx.fillText(spec.xLabel || "", m.l + pw / 2, m.t + ph + 21);
  if (spec.yLabel) {
    ctx.save(); ctx.translate(12, m.t + ph / 2); ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(spec.yLabel, 0, 0); ctx.restore();
  }
  if (ymin < 0 && ymax > 0) {
    ctx.strokeStyle = css("--muted"); ctx.globalAlpha = 0.6;
    ctx.beginPath(); ctx.moveTo(m.l, Y(0)); ctx.lineTo(m.l + pw, Y(0)); ctx.stroke(); ctx.globalAlpha = 1;
  }

  ctx.save();
  ctx.beginPath(); ctx.rect(m.l, m.t, pw, ph); ctx.clip();
  ctx.lineJoin = "round"; ctx.lineCap = "round";
  for (const s of spec.series) {
    ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 2; ctx.setLineDash(s.dash || []);
    ctx.globalAlpha = s.alpha == null ? 1 : s.alpha;
    ctx.beginPath();
    let pen = false;
    for (let i = 0; i < s.x.length; i++) {
      const v = s.y[i];
      if (!Number.isFinite(v)) { pen = false; continue; }
      if (pen) ctx.lineTo(X(s.x[i]), Y(v)); else { ctx.moveTo(X(s.x[i]), Y(v)); pen = true; }
    }
    ctx.stroke();
  }
  ctx.globalAlpha = 1; ctx.setLineDash([]);
  for (const p of spec.points || []) {
    ctx.fillStyle = p.color; ctx.strokeStyle = css("--canvas-bg"); ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(X(p.x), Y(p.y), p.r || 4.5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
  }
  // crosshair and readout
  const hv = hover[id];
  let html = "";
  if (hv && spec.onHover) {
    const xv = xmin + ((hv.px - m.l) / pw) * (xmax - xmin), yv = ymin + (1 - (hv.py - m.t) / ph) * (ymax - ymin);
    if (xv >= xmin && xv <= xmax) {
      const xx = Math.round(X(xv)) + 0.5;
      ctx.strokeStyle = css("--muted"); ctx.setLineDash([2, 3]);
      ctx.beginPath(); ctx.moveTo(xx, m.t); ctx.lineTo(xx, m.t + ph); ctx.stroke(); ctx.setLineDash([]);
      html = spec.onHover(xv, yv, ctx, X, Y) || "";
    }
  }
  ctx.restore();
  if (spec.readEl) $(spec.readEl).innerHTML = html || spec.idle || "Hover or drag across the plot to read values.";
}

function interp(xs, ys, x) {
  if (x <= xs[0]) return ys[0];
  if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
  let lo = 0, hi = xs.length - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (xs[mid] <= x) lo = mid; else hi = mid; }
  const f = (x - xs[lo]) / (xs[hi] - xs[lo]);
  return ys[lo] + f * (ys[hi] - ys[lo]);
}
function nearestIndex(xs, x) {
  let best = 0;
  for (let i = 1; i < xs.length; i++) if (Math.abs(xs[i] - x) < Math.abs(xs[best] - x)) best = i;
  return best;
}

// ---------- line measurements ----------
const REGION = [-290, 150];   // kHz from the P = 0 center: the line; outside it the sweep is noise
function measure(x, y) {
  const h = x[1] - x[0];
  let area = 0, mom = 0, i0 = -1;
  for (let i = 0; i < x.length; i++) {
    if (x[i] < REGION[0] || x[i] > REGION[1]) continue;
    area += y[i] * h; mom += y[i] * x[i] * h;
    if (i0 < 0 || y[i] > y[i0]) i0 = i;
  }
  // parabolic peak through the top three samples
  const a = y[i0 - 1], b = y[i0], c = y[i0 + 1];
  const d = (a - c) / (2 * (a - 2 * b + c)) || 0;
  const xp = x[i0] + d * h, yp = b - 0.25 * (a - c) * d;
  const cross = (dir) => {
    let i = i0;
    while (i + dir >= 0 && i + dir < y.length && y[i + dir] > yp / 2) i += dir;
    const j = i + dir;
    if (j < 0 || j >= y.length) return x[i];
    return x[i] + (x[j] - x[i]) * (y[i] - yp / 2) / (y[i] - y[j]);
  };
  return { area, centroid: mom / area, xp, yp, lo: xp - cross(-1), hi: cross(+1) - xp };
}

// ---------- state ----------
let D = null;            // proton_data.json
let lines = [];          // measured lines, sorted by polarization
let model = null;
let norm = "area";
let modelP = 0.5;
let pRange = [0, 1];
let hlLine = -1;

function setupData(d) {
  D = d;
  const x = d.x_khz;
  lines = d.lines.map((ln) => ({ ...ln, ...measure(x, ln.y) })).sort((a, b) => a.pol - b.pol);
  for (const ln of lines) ln.yn = ln.y.map((v) => 1000 * v / ln.area);       // area-normalized, per MHz
  pRange = [lines[0].pol, lines[lines.length - 1].pol];
  const md = d.model;
  const knots = md.A0.map((_, j) => (j - md.half) * md.step_khz);
  const area = md.step_khz * md.A0.reduce((s, v) => s + v, 0);
  model = { ...md, knots, area };
  // straight-line fit of the centroid against P
  const n = lines.length, sx = lines.reduce((s, l) => s + l.pol, 0), sy = lines.reduce((s, l) => s + l.centroid, 0);
  const sxx = lines.reduce((s, l) => s + l.pol * l.pol, 0), sxy = lines.reduce((s, l) => s + l.pol * l.centroid, 0);
  const slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
  model.fit = { slope, icpt: (sy - slope * sx) / n };
}
const colorOf = (p) => rampColor((p - pRange[0]) / (pRange[1] - pRange[0]));

function modelLine(P, xs) {
  const s = model.shift_khz_per_pol * P;
  return xs.map((x) => {
    const u = x - s;
    if (u <= model.knots[0] || u >= model.knots[model.knots.length - 1]) return 0;
    return 1000 * (interp(model.knots, model.A0, u) + P * interp(model.knots, model.A1, u)) / model.area;
  });
}

// ---------- charts ----------
const START = () => new Date(D.timeline.start.replace(" UTC", "Z").replace(" ", "T"));
const H0 = () => { const t = START(); return t.getUTCHours() + t.getUTCMinutes() / 60 + t.getUTCSeconds() / 3600; };
function clock(hourOfDay) {           // hours since 00:00 UTC -> "HH:MM"
  const m = Math.round(hourOfDay * 60);
  return `${String(Math.floor(m / 60) % 24).padStart(2, "0")}:${String(m % 60).padStart(2, "0")}`;
}

function drawTime() {
  const tl = D.timeline, h0 = H0();
  const hx = tl.hours.map((h) => h + h0);
  const pts = lines.map((ln) => {
    const k = D.lines.findIndex((l) => l.time === ln.time && l.pol === ln.pol);
    return { x: tl.line_hours[k] + h0, y: 100 * ln.pol, color: colorOf(ln.pol), r: 5 };
  });
  plot("cvTime", {
    series: [{ x: hx, y: tl.pol.map((p) => 100 * p), color: css("--ink-2"), width: 1.6 }],
    points: pts, zero: true, ymax: 85, xLabel: "time on 9 October 2022 (UTC)", yLabel: "polarization (%)",
    xFmt: clock, readEl: "rdTime",
    onHover: (xv) => {
      const i = nearestIndex(hx, xv);
      return `${clock(hx[i])} UTC · P = ${pct(tl.pol[i])}`;
    },
  });
  $("timeLegend").innerHTML =
    `<span style="color:var(--ink-2)"><i class="sw" style="background:currentColor;height:2px"></i><span style="color:var(--muted)">each event</span></span>` +
    `<span><i class="sw dot-m" style="color:${colorOf(pRange[1])}"></i><span>example lines, colored by polarization as below</span></span>`;
}

function drawLines() {
  const x = D.x_khz;
  const key = norm === "area" ? "yn" : "y";
  const scale = norm === "area" ? 1 : 1000;          // as measured: mV
  const series = lines.map((ln, i) => ({
    x, y: ln[key].map((v) => v * scale), color: colorOf(ln.pol),
    width: i === hlLine ? 3 : 1.8, alpha: hlLine < 0 || i === hlLine ? 1 : 0.35,
  }));
  if (hlLine >= 0) series.push(series.splice(hlLine, 1)[0]);       // highlighted line on top
  plot("cvLines", {
    series, xmin: -220, xmax: 160, zero: true,
    xLabel: `offset from the P = 0 line center, ${D.center_mhz.toFixed(3)} MHz (kHz)`,
    yLabel: norm === "area" ? "signal, area = 1 (per MHz)" : "signal (mV)", readEl: "rdLines",
    onHover: (xv, yv) => {
      let best = -1, bd = Infinity;
      lines.forEach((ln, i) => {
        const d = Math.abs(interp(x, ln[key], xv) * scale - yv);
        if (d < bd) { bd = d; best = i; }
      });
      if (best !== hlLine) { hlLine = best; requestAnimationFrame(drawLines); }
      const ln = lines[best];
      const what = ln.n > 1 ? `average of ${ln.n} events around ${ln.time.slice(11)}` : `event at ${ln.time.slice(11)}`;
      const v = interp(x, ln[key], xv) * scale;
      return `P = ${pct(ln.pol)} · ${what} · ${sgn(xv)} kHz: ${v.toFixed(norm === "area" ? 2 : 3)}${norm === "area" ? "" : " mV"}`;
    },
  });
  $("rampBar").style.background = `linear-gradient(90deg, ${(isDark() ? RAMP_DARK : RAMP_LIGHT).join(",")})`;
  $("rampLabels").textContent = `${pct(pRange[0], 0)} → ${pct(pRange[1], 0)}`;
}

function scatterSpec(id, readEl, sets, yLabel, fmtRead, extra = {}) {
  const series = [], points = [];
  for (const s of sets) {
    if (s.line) series.push(s.line);
    for (const ln of lines) points.push({ x: 100 * ln.pol, y: s.val(ln), color: s.color, r: 4.5 });
  }
  plot(id, {
    series, points, xmin: 20, xmax: 82, xLabel: "polarization (%)", yLabel, readEl, ...extra,
    onHover: (xv) => {
      const ln = lines[nearestIndex(lines.map((l) => 100 * l.pol), xv)];
      return fmtRead(ln);
    },
  });
}

function drawChanges() {
  const fit = model.fit;
  const fx = [20, 82];
  scatterSpec("cvShift", "rdShift", [
    { val: (l) => l.centroid, color: css("--copper"),
      line: { x: fx, y: fx.map((p) => fit.icpt + fit.slope * p / 100), color: css("--copper"), width: 1.6, dash: [6, 4] } },
    { val: (l) => l.xp, color: css("--shell") },
  ], "offset (kHz)", (l) => `P = ${pct(l.pol)}: center ${sgn(l.centroid)} kHz, peak ${sgn(l.xp)} kHz`, { zero: true });
  scatterSpec("cvWidth", "rdWidth", [
    { val: (l) => l.lo, color: css("--copper") },
    { val: (l) => l.hi, color: css("--shell") },
  ], "half width (kHz)", (l) => `P = ${pct(l.pol)}: ${l.lo.toFixed(1)} kHz below, ${l.hi.toFixed(1)} kHz above the peak`, { zero: true });
  const ref = lines[0].yp / lines[0].pol;
  scatterSpec("cvPeak", "rdPeak", [
    { val: (l) => l.yp / l.pol / ref, color: css("--copper") },
  ], "peak ÷ P, relative", (l) => `P = ${pct(l.pol)}: peak per unit P ×${(l.yp / l.pol / ref).toFixed(3)}`,
  { ymin: 0.95, ymax: 1.3, yFmt: (v) => v.toFixed(2) });

  const lo = lines[0], hi = lines[lines.length - 1];
  $("findings").innerHTML = [
    `The center of area moves up <b>${fit.slope.toFixed(1)} kHz per unit polarization</b>, from ${sgn(lo.centroid)} kHz at ${pct(lo.pol, 0)} to ${sgn(hi.centroid)} kHz at ${pct(hi.pol, 0)}. The peak moves further, from ${sgn(lo.xp)} to ${sgn(hi.xp)} kHz, because the line also changes shape.`,
    `Above the peak the line narrows from ${lo.hi.toFixed(1)} to ${hi.hi.toFixed(1)} kHz at half height; below it, from ${lo.lo.toFixed(1)} to ${hi.lo.toFixed(1)} kHz. The sharp edge is on the high-frequency side.`,
    `For the same polarization the line gets taller: peak height per unit P is ${((hi.yp / hi.pol / ref - 1) * 100).toFixed(0)}% higher at ${pct(hi.pol, 0)} than at ${pct(lo.pol, 0)}. The area per unit P is constant here by construction, since the logged polarization is the calibrated area.`,
  ].map((s) => `<li>${s}</li>`).join("");
}

function drawModel() {
  const xs = [];
  for (let v = -250; v <= 200; v += 0.5) xs.push(v);
  const gP = modelLine(modelP, xs), g0 = modelLine(0, xs);
  const series = [{ x: xs, y: g0, color: css("--muted"), width: 1.6, dash: [2, 3] }];
  // nearest measured line, if one is close to this polarization
  let near = null;
  for (const ln of lines) if (Math.abs(ln.pol - modelP) <= 0.015 && (!near || Math.abs(ln.pol - modelP) < Math.abs(near.pol - modelP))) near = ln;
  if (near) series.push({ x: D.x_khz, y: near.yn, color: css("--shell"), width: 1.8, dash: [6, 4] });
  series.push({ x: xs, y: gP, color: css("--copper"), width: 2.2 });
  $("measLegend").hidden = !near;
  if (near) $("measLegendText").textContent = `measured, P = ${pct(near.pol)}`;
  const mm = measure(xs, gP);
  const inside = modelP >= pRange[0] - 0.005 && modelP <= pRange[1] + 0.005;
  plot("cvModel", {
    series, xmin: -220, xmax: 160, zero: true, xLabel: "offset from the P = 0 line center (kHz)",
    yLabel: "signal, area = 1 (per MHz)", readEl: "rdModel",
    idle: `P = ${pct(modelP, 0)}: center ${sgn(mm.centroid)} kHz, peak ${sgn(mm.xp)} kHz, half widths ${mm.lo.toFixed(1)} / ${mm.hi.toFixed(1)} kHz` +
      (inside ? " · within the measured range" : " · extrapolated beyond the measured 24–79%"),
    onHover: (xv) => `${sgn(xv)} kHz: model ${interp(xs, gP, xv).toFixed(2)}` + (near ? `, measured ${interp(D.x_khz, near.yn, xv).toFixed(2)}` : ""),
  });
  const xk = model.knots;
  plot("cvParts", {
    series: [
      { x: xk, y: model.A1, color: css("--shell"), width: 2 },
      { x: xk, y: model.A0, color: css("--copper"), width: 2.2 },
    ], xmin: -220, xmax: 160, zero: true, xLabel: "offset from the P = 0 line center (kHz)",
    yLabel: "relative to the A₀ peak", readEl: "rdParts",
    onHover: (xv) => `${sgn(xv)} kHz: A₀ ${interp(xk, model.A0, xv).toFixed(3)}, A₁ ${interp(xk, model.A1, xv).toFixed(3)}`,
  });
  $("maxErr").textContent = `${(100 * model.max_err).toFixed(1)}%`;
}

function drawAll() {
  if (!D) return;
  drawTime(); drawLines(); drawChanges(); drawModel();
}
const DRAW = { cvTime: drawTime, cvLines: drawLines, cvShift: drawChanges, cvWidth: drawChanges, cvPeak: drawChanges, cvModel: drawModel, cvParts: drawModel };

// ---------- interaction ----------
for (const id of Object.keys(DRAW)) {
  const cv = $(id);
  const move = (e) => {
    const r = cv.getBoundingClientRect();
    hover[id] = { px: e.clientX - r.left, py: e.clientY - r.top };
    if (D) DRAW[id]();
  };
  cv.addEventListener("pointermove", move);
  cv.addEventListener("pointerdown", move);
  cv.addEventListener("pointerleave", (e) => {
    if (e.pointerType !== "mouse") return;
    delete hover[id];
    if (id === "cvLines") hlLine = -1;
    if (D) DRAW[id]();
  });
}
for (const b of $("normSeg").querySelectorAll("button")) {
  b.addEventListener("click", () => {
    norm = b.dataset.v;
    for (const o of $("normSeg").querySelectorAll("button")) o.setAttribute("aria-pressed", String(o === b));
    drawLines();
  });
}
$("pSlider").addEventListener("input", (e) => {
  modelP = Number(e.target.value) / 100;
  $("pValue").textContent = `${e.target.value}%`;
  drawModel();
});

function renderEquations() {
  const els = document.querySelectorAll("[data-tex]");
  if (!window.katex) { for (const el of els) el.textContent = el.dataset.tex; return; }
  for (const el of els) katex.render(el.dataset.tex, el, { displayMode: el.tagName === "DIV", throwOnError: false });
}

renderEquations();
new ResizeObserver(() => drawAll()).observe(document.querySelector(".wrap"));
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", drawAll);
new MutationObserver(drawAll).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
fetch("proton_data.json?v=__V__").then((r) => r.json()).then((d) => { setupData(d); drawAll(); })
  .catch((err) => { $("rdLines").textContent = "Could not load proton_data.json: " + err; });
