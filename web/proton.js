"use strict";
// The proton line while the polarization moves: episodes of measured lines (proton_data.json, made by
// tools/make_proton_data.py) and the simulator's model g(x; P) = A0(u) + P A±(u), u = x - s± P.

const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const isDark = () => {
  const t = document.documentElement.dataset.theme;
  return t === "dark" || (t !== "light" && matchMedia("(prefers-color-scheme: dark)").matches);
};

// polarization colors: blue for P > 0, orange for P < 0, each light -> dark with |P| on a light surface
// (dark -> light on a dark one)
const RAMPS = {
  pos: {
    light: ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"],
    dark: ["#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5", "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#cde2fb"],
  },
  neg: {
    light: ["#f0b48c", "#eca274", "#e6905c", "#de7e45", "#d16d33", "#c25f27", "#b4531b", "#9c4616", "#843a12", "#6c2f0e"],
    dark: ["#8a3c10", "#a2481a", "#b8582a", "#cc6a38", "#dc7a45", "#e5834a", "#eb9566", "#f0a882", "#f4bb9e", "#f8d3c0"],
  },
};
const P_SCALE = [0, 0.8];            // |P| mapped onto the ramps
function hexRgb(h) { const n = parseInt(h.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
const ramp = (p) => RAMPS[p < 0 ? "neg" : "pos"][isDark() ? "dark" : "light"];
function colorOf(p) {
  const r = ramp(p);
  const t = Math.max(0, Math.min(1, (Math.abs(p) - P_SCALE[0]) / (P_SCALE[1] - P_SCALE[0])));
  const u = t * (r.length - 1), i = Math.min(r.length - 2, Math.floor(u)), f = u - i;
  const a = hexRgb(r[i]), b = hexRgb(r[i + 1]);
  return `rgb(${a.map((v, k) => Math.round(v + f * (b[k] - v))).join(",")})`;
}
const signColor = (p) => colorOf(p < 0 ? -0.6 : 0.6);
// episode kinds, one hue each
const KIND = { rise: { v: "--k-rise", name: "spin-up" }, reversal: { v: "--k-rev", name: "reversal" } };
const kindColor = (k) => css(KIND[k].v);

const pct = (p, d = 1) => `${p < 0 ? "−" : ""}${Math.abs(100 * p).toFixed(d)}%`;
const spct = (p, d = 0) => (Math.abs(100 * p) < 0.5 * Math.pow(10, -d) ? "0%" : `${p > 0 ? "+" : ""}${pct(p, d)}`);
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
const geom = {};        // canvas id -> axis ranges and plot size, for nearest-point searches

// spec: {series:[{x,y,color,width,dash,alpha}], points:[{x,y,color,r,hollow}], bands:[{x0,x1,color,label}],
//        xLabel, yLabel, xFmt, yFmt, xStep, xmin,xmax,ymin,ymax, zero, readEl, idle, onHover(xv, yv) -> html}
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
      if (xv < xmin || xv > xmax || v == null || !Number.isFinite(v)) continue;
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
  geom[id] = { xmin, xmax, ymin, ymax, pw, ph };

  for (const b of spec.bands || []) {
    const x0 = Math.max(m.l, X(b.x0)), x1 = Math.min(m.l + pw, Math.max(X(b.x1), X(b.x0) + 3));
    if (x1 <= x0) continue;
    ctx.fillStyle = b.color || css("--muted"); ctx.globalAlpha = b.alpha == null ? 0.28 : b.alpha;
    ctx.fillRect(x0, m.t, x1 - x0, ph); ctx.globalAlpha = 1;
  }
  ctx.font = `11px ${css("--f-mono")}`;
  ctx.lineWidth = 1; ctx.strokeStyle = css("--line"); ctx.fillStyle = css("--muted");
  const ys = niceStep(ymax - ymin, Math.max(3, Math.floor(ph / 42)));
  const ydec = Math.max(0, -Math.floor(Math.log10(ys) + 1e-9));
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (let t = Math.ceil(ymin / ys - 1e-9) * ys; t <= ymax + ys * 1e-6; t += ys) {
    const yy = Math.round(Y(t)) + 0.5;
    ctx.beginPath(); ctx.moveTo(m.l, yy); ctx.lineTo(m.l + pw, yy); ctx.stroke();
    ctx.fillText(spec.yFmt ? spec.yFmt(t) : (Math.abs(t) < ys * 1e-6 ? 0 : t).toFixed(ydec).replace("-", "−"), m.l - 6, yy);
  }
  const xs = spec.xStep || niceStep(xmax - xmin, Math.max(3, Math.floor(pw / 75)));
  const xdec = Math.max(0, -Math.floor(Math.log10(xs) + 1e-9));
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (let t = Math.ceil(xmin / xs - 1e-9) * xs; t <= xmax + xs * 1e-6; t += xs) {
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
      if (v == null || !Number.isFinite(v)) { pen = false; continue; }
      if (pen) ctx.lineTo(X(s.x[i]), Y(v)); else { ctx.moveTo(X(s.x[i]), Y(v)); pen = true; }
    }
    ctx.stroke();
  }
  ctx.globalAlpha = 1; ctx.setLineDash([]);
  for (const p of spec.points || []) {
    ctx.beginPath(); ctx.arc(X(p.x), Y(p.y), p.r || 4.5, 0, 2 * Math.PI);
    if (p.hollow) {
      ctx.fillStyle = css("--canvas-bg"); ctx.fill();
      ctx.strokeStyle = p.color; ctx.lineWidth = 2; ctx.stroke();
    } else {
      ctx.fillStyle = p.color; ctx.fill();
      ctx.strokeStyle = css("--canvas-bg"); ctx.lineWidth = 2; ctx.stroke();
    }
  }
  // band labels, kept clear of each other
  ctx.font = `600 11px ${css("--f-mono")}`; ctx.textAlign = "center"; ctx.textBaseline = "top";
  const groups = [];                    // bands too close to label one by one share a label, "3–6"
  for (const b of spec.bands || []) {
    if (!b.label) continue;
    const xc = (X(b.x0) + X(b.x1)) / 2, g = groups[groups.length - 1];
    if (g && xc - g.x1 < 16) { g.x1 = xc; g.last = b.label; } else groups.push({ x0: xc, x1: xc, first: b.label, last: b.label });
  }
  for (const g of groups) {
    const text = g.first === g.last ? g.first : `${g.first}–${g.last}`, w = ctx.measureText(text).width;
    const xc = Math.min(m.l + pw - w / 2 - 2, Math.max(m.l + w / 2 + 2, (g.x0 + g.x1) / 2));
    ctx.fillStyle = css("--ink"); ctx.fillText(text, xc, m.t + 3);
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
      html = spec.onHover(xv, yv) || "";
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

// ---------- line measurements (as in tools/make_proton_data.py) ----------
function measure(x, y) {      // y with absorption positive
  const h = x[1] - x[0];
  let i0 = -1;
  for (let i = 1; i < x.length - 1; i++) if (Math.abs(x[i]) < 150 && (i0 < 0 || y[i] > y[i0])) i0 = i;
  let area = 0, mom = 0;
  for (let i = 0; i < x.length; i++) {
    if (Math.abs(x[i] - x[i0]) >= 130) continue;
    area += y[i] * h; mom += y[i] * x[i] * h;
  }
  // parabolic peak through the top three samples
  const a = y[i0 - 1], b = y[i0], c = y[i0 + 1];
  const d = (a - c) / (2 * (a - 2 * b + c)) || 0;
  const xp = x[i0] + d * h, yp = b - 0.25 * (a - c) * d;
  const cross = (dir) => {
    let i = i0;
    while (i + dir > 0 && i + dir < y.length - 1 && y[i + dir] > yp / 2) i += dir;
    const j = i + dir;
    return x[i] + (x[j] - x[i]) * (y[i] - yp / 2) / (y[i] - y[j]);
  };
  return { area, centroid: mom / area, xp, yp, lo: xp - cross(-1), hi: cross(+1) - xp };
}

// ---------- state ----------
let D = null;            // proton_data.json
let runs = {};           // id -> run
let eps = [];            // episodes, in order
let model = null;
let sel = "sep-spinup";  // chosen episode
let norm = "area";
let modelP = 0.5;
let hlLine = -1;
const epOf = (id) => eps.find((e) => e.id === id);

function setupData(d) {
  D = d;
  runs = Object.fromEntries(d.runs.map((r) => [r.id, r]));
  const md = d.model;
  eps = d.episodes.map((e) => {
    const run = runs[e.run], x = run.x_khz;
    const lines = d.lines.slice(e.lines[0], e.lines[1]).map((ln) => {
      const s = Math.sign(ln.pol) || 1;
      const usable = ln.one_sign && Math.abs(ln.pol) >= md.min_pol;
      const ya = ln.y_mv.map((v) => s * v);
      const mm = usable ? measure(x, ya) : null;
      return { ...ln, x, yn: mm ? ya.map((v) => 1000 * v / mm.area) : null };      // area-normalized, per MHz
    });
    // time taken by the middle 80% of the change in polarization
    const t = e.traj, dp = e.p_to - e.p_from;
    const at = (frac) => t.minutes[t.pol.findIndex((p) => (p - e.p_from) / dp >= frac)];
    return { ...e, lines, mid80: at(0.9) - at(0.1), runName: run.name };
  });
  const knots = md.A0.map((_, j) => (j - md.half) * md.step_khz);
  const area = md.step_khz * md.A0.reduce((s, v) => s + v, 0);
  model = { ...md, knots, area };
  buildTable(); buildEpisodeButtons(); fillText();
}

const shiftOf = (P) => (P >= 0 ? model.shift_pos : model.shift_neg) * P;
function modelLine(P, xs) {           // area 1, per MHz
  const s = shiftOf(P), pc = Math.max(model.pol_min, Math.min(model.pol_max, P));
  const part = pc >= 0 ? model.APOS : model.ANEG;
  return xs.map((x) => {
    const u = x - s;
    if (u <= model.knots[0] || u >= model.knots[model.knots.length - 1]) return 0;
    return 1000 * (interp(model.knots, model.A0, u) + pc * interp(model.knots, part, u)) / model.area;
  });
}

// ---------- time ----------
const startOf = (iso) => new Date(iso.replace(" UTC", "Z").replace(" ", "T"));
const hourOfDay = (iso) => { const t = startOf(iso); return t.getUTCHours() + t.getUTCMinutes() / 60 + t.getUTCSeconds() / 3600; };
function clock(h) {                    // hours since 00:00 UTC -> "HH:MM"
  const m = Math.round(h * 60);
  return `${String(Math.floor(m / 60) % 24).padStart(2, "0")}:${String(m % 60).padStart(2, "0")}`;
}
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
function dayOf(iso, h) {               // hours since 00:00 UTC of the start day -> "4 Jul"
  const t = startOf(iso), d = new Date(Date.UTC(t.getUTCFullYear(), t.getUTCMonth(), t.getUTCDate()) + Math.floor(h / 24 + 1e-9) * 864e5);
  return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]}`;
}
const dayClock = (iso, h) => `${dayOf(iso, h)} ${clock(h)}`;
const mins = (m) => (m < 1.5 ? `${Math.round(60 * m)} s` : m < 90 ? `${m < 10 ? m.toFixed(1) : Math.round(m)} min` : `${(m / 60).toFixed(1)} h`);

// ---------- the data section ----------
function drawTimeRun(run) {
  const id = `cvTime_${run.id}`, tl = run.timeline, h0 = hourOfDay(run.start);
  const hx = tl.hours.map((h) => h + h0);
  const mine = eps.filter((e) => e.run === run.id);
  const bands = mine.map((e) => ({
    x0: e.start_hours + h0, x1: e.end_hours + h0, color: kindColor(e.kind), label: String(e.n),
    alpha: e.id === sel ? 0.55 : 0.28,
  }));
  const span = hx[hx.length - 1] - hx[0], w = $(id).getBoundingClientRect().width;
  const long = span > 30;
  plot(id, {
    series: [{ x: hx, y: tl.pol.map((p) => 100 * p), color: css("--ink-2"), width: 1.6 }],
    bands, zero: true, ymin: long ? -85 : 0, ymax: 90, yLabel: "polarization (%)",
    xLabel: long ? `${dayOf(run.start, hx[0])} to ${dayOf(run.start, hx[hx.length - 1])} 2022 (UTC)` : `time on ${dayOf(run.start, hx[0])} 2022 (UTC)`,
    xFmt: long ? (h) => (Math.abs(h % 24) < 1e-6 ? dayOf(run.start, h) : clock(h)) : clock,
    xStep: long ? (w > 420 ? 12 : 24) : (w > 420 ? 2 : 4), readEl: `rdTime_${run.id}`,
    idle: mine.map((e) => `${e.n} ${e.title.toLowerCase()}`).join(" · "),
    onHover: (xv) => {
      const i = nearestIndex(hx, xv), e = mine.find((q) => hx[i] >= q.start_hours + h0 && hx[i] < q.end_hours + h0);
      return `${long ? dayClock(run.start, hx[i]) : clock(hx[i])} UTC · P = ${pct(tl.pol[i])}` + (e ? ` · episode ${e.n}, ${e.title.toLowerCase()}` : "");
    },
  });
}
function drawTime() {
  for (const run of D.runs) drawTimeRun(run);
  $("kindLegend").innerHTML =
    `<span style="color:var(--ink-2)"><i class="sw" style="background:currentColor;height:2px"></i><span style="color:var(--muted)">each event</span></span>` +
    Object.values(KIND).map((k) => `<span><i class="sw band" style="background:var(${k.v});opacity:0.5"></i><span>${k.name}</span></span>`).join("") +
    `<span>numbers are the episodes in the table</span>`;
}

function buildTable() {
  const body = $("epTable").querySelector("tbody");
  body.innerHTML = eps.map((e) => {
    const h = hourOfDay(e.start);
    return `<tr data-ep="${e.id}"><td class="num"><button type="button" class="epbtn" aria-label="Show episode ${e.n}">${e.n}</button></td>` +
      `<td class="when">${dayOf(e.start, h)} ${clock(h)}</td>` +
      `<td><i class="kind" style="background:var(${KIND[e.kind].v})"></i>${e.title}<span class="what">${e.what}</span></td>` +
      `<td class="num">${spct(e.p_from)} → ${spct(e.p_to)}</td><td class="num">${mins(e.mid80)}</td></tr>`;
  }).join("");
  for (const tr of body.querySelectorAll("tr")) tr.addEventListener("click", () => choose(tr.dataset.ep));
}
function buildEpisodeButtons() {
  $("epSeg").innerHTML = eps.map((e) => `<button type="button" data-ep="${e.id}" title="${e.runName}: ${e.title}">${e.n}</button>`).join("");
  for (const b of $("epSeg").querySelectorAll("button")) b.addEventListener("click", () => choose(b.dataset.ep));
}
function choose(id) {
  sel = id; hlLine = -1;
  for (const tr of $("epTable").querySelectorAll("tbody tr")) tr.setAttribute("aria-selected", String(tr.dataset.ep === id));
  for (const b of $("epSeg").querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.ep === id));
  const e = epOf(id), h = hourOfDay(e.start);
  $("epCaption").innerHTML = `<b>Episode ${e.n}: ${e.title}</b>, ${dayOf(e.start, h)} 2022, ${clock(h)} UTC. ${e.what} ` +
    `The polarization goes from ${spct(e.p_from)} to ${spct(e.p_to)}; the middle 80% of the change takes ${mins(e.mid80)}.`;
  drawAll();
}
function setNorm(v) {
  norm = v;
  for (const o of $("normSeg").querySelectorAll("button")) o.setAttribute("aria-pressed", String(o.dataset.v === v));
}

// ---------- the lines of an episode ----------
function rampLegend(el, extra = "") {
  const bar = (key) => `<span class="ramp" style="background:linear-gradient(90deg, ${RAMPS[key][isDark() ? "dark" : "light"].join(",")})"></span>`;
  $(el).innerHTML =
    `<span class="group"><span>P &gt; 0</span>${bar("pos")}<span class="ramp-label">0 → +${100 * P_SCALE[1]}%</span></span>` +
    `<span class="group"><span>P &lt; 0</span>${bar("neg")}<span class="ramp-label">0 → −${100 * P_SCALE[1]}%</span></span>` + extra;
}
const lineWhen = (ln) => `${mins(ln.minutes)} in`;

function drawLines() {
  const e = epOf(sel);
  const key = norm === "area" ? "yn" : "y_mv";
  const shown = e.lines.filter((ln) => ln[key]);
  const series = shown.map((ln, i) => ({
    x: ln.x, y: ln[key], color: colorOf(ln.pol),
    width: i === hlLine ? 3 : 1.8, alpha: hlLine < 0 || i === hlLine ? 1 : 0.3,
  }));
  if (hlLine >= 0 && hlLine < series.length) series.push(series.splice(hlLine, 1)[0]);       // highlighted line on top
  const left = e.lines.length - shown.length;
  plot("cvLines", {
    series, xmin: -200, xmax: 160, zero: true,
    xLabel: `offset from the ${e.runName} P = 0 line center, ${runs[e.run].center_mhz.toFixed(3)} MHz (kHz)`,
    yLabel: norm === "area" ? "signal, area = 1 (per MHz)" : "signal (mV)", readEl: "rdLines",
    idle: `${shown.length} lines` + (left ? ` · ${left} left out of this view (too small to scale, or both signs); choose “As measured” to see them` : "") + " · hover to read values",
    onHover: (xv, yv) => {
      let best = -1, bd = Infinity;
      shown.forEach((ln, i) => {
        const d = Math.abs(interp(ln.x, ln[key], xv) - yv);
        if (d < bd) { bd = d; best = i; }
      });
      if (best !== hlLine) { hlLine = best; requestAnimationFrame(drawLines); }
      const ln = shown[best];
      if (!ln) return "";
      const v = interp(ln.x, ln[key], xv);
      return `P = ${pct(ln.pol)} · ${lineWhen(ln)} · ${sgn(xv)} kHz: ${v.toFixed(norm === "area" ? 2 : 3)}${norm === "area" ? "" : " mV"}`;
    },
  });
  rampLegend("rampLegend");
}

// ---------- trajectories ----------
const TRAJ = [
  { cv: "cvC", rd: "rdC", key: "c", label: "offset (kHz)", fmt: (v) => `center ${sgn(v)} kHz` },
  { cv: "cvPk", rd: "rdPk", key: "pk", label: "peak ÷ area, relative to the model at P = 0", fmt: (v) => `peak for the area ×${v.toFixed(3)}`, rel: true },
  { cv: "cvLo", rd: "rdLo", key: "lo", label: "half width (kHz)", fmt: (v) => `${v.toFixed(1)} kHz below the peak` },
  { cv: "cvHi", rd: "rdHi", key: "hi", label: "half width (kHz)", fmt: (v) => `${v.toFixed(1)} kHz above the peak` },
];
function drawTraj() {
  for (const T of TRAJ) {
    const scale = T.rel ? 1 / model.peak0 : 1;
    const val = (v) => (v == null ? null : v * scale);
    const cv = model.curves;
    const series = [];
    const order = [...eps.filter((e) => e.id !== sel), epOf(sel)];
    for (const e of order) {
      series.push({
        x: e.traj.pol.map((p) => 100 * p), y: e.traj[T.key].map(val), color: kindColor(e.kind),
        width: e.id === sel ? 3 : 1.6, alpha: e.id === sel ? 1 : 0.55,
      });
    }
    series.push({ x: cv.pol.map((p) => 100 * p), y: cv[T.key].map(val), color: css("--ink"), width: 1.6, dash: [5, 4] });
    const e0 = epOf(sel);
    plot(T.cv, {
      series, xmin: -80, xmax: 82, xLabel: "polarization (%)", yLabel: T.label, readEl: T.rd,
      zero: T.key === "c", ymin: T.key === "c" ? undefined : T.rel ? 0.82 : 10, ymax: T.rel ? 1.22 : T.key === "c" ? undefined : 38,
      yFmt: T.rel ? (v) => v.toFixed(2) : undefined,
      idle: `Heavy line: episode ${e0.n}, ${e0.title.toLowerCase()}. Hover to read any path.`,
      onHover: (xv, yv) => {
        const g = geom[T.cv];
        let best = null, bd = Infinity;
        for (const e of eps) {
          const t = e.traj;
          for (let i = 0; i < t.pol.length; i++) {
            const v = val(t[T.key][i]);
            if (v == null) continue;
            const dx = (100 * t.pol[i] - xv) / (g.xmax - g.xmin) * g.pw, dy = (v - yv) / (g.ymax - g.ymin) * g.ph;
            const d2 = dx * dx + dy * dy;
            if (d2 < bd) { bd = d2; best = { e, i, v }; }
          }
        }
        if (!best) return "";
        return `Episode ${best.e.n}, ${best.e.title.toLowerCase()} · ${mins(best.e.traj.minutes[best.i])} in · P = ${pct(best.e.traj.pol[best.i])}: ${T.fmt(best.v)}`;
      },
    });
  }
  $("trajLegend").innerHTML =
    Object.values(KIND).map((k) => `<span style="color:var(${k.v})"><i class="sw line"></i><span style="color:var(--muted)">${k.name}</span></span>`).join("") +
    `<span style="color:var(--ink)"><i class="sw dash"></i><span style="color:var(--muted)">model</span></span>` +
    `<span>gaps: lines too small to measure, or with both signs</span>`;
}

// value of a measure where an episode's path first crosses polarization P
function valueAt(e, key, P) {
  const t = e.traj;
  for (let i = 1; i < t.pol.length; i++) {
    const a = t.pol[i - 1], b = t.pol[i];
    if ((a - P) * (b - P) > 0 || a === b || t[key][i - 1] == null || t[key][i] == null) continue;
    return t[key][i - 1] + (t[key][i] - t[key][i - 1]) * (P - a) / (b - a);
  }
  return null;
}
const curveAt = (key, P) => interp(model.curves.pol, model.curves[key], P);
function median(a) { const s = [...a].sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; }

function fillText() {
  const m = model, at = 0.4;
  const w = (id) => { const e = epOf(id), v = e && valueAt(e, "lo", at); return v == null ? null : `${v.toFixed(1)} kHz (${e.n})`; };
  const pk = (id) => { const e = epOf(id), v = e && valueAt(e, "pk", at); return v == null ? null : `×${(v / m.peak0).toFixed(2)} (${e.n})`; };
  const P1 = m.pol_max, P2 = m.pol_min;
  $("findings").innerHTML = [
    `While the polarization grows, the center of area moves about <b>${m.shift_pos.toFixed(0)} kHz per unit P</b> upward for positive polarization and <b>${m.shift_neg.toFixed(0)} kHz</b> downward for negative: ${sgn(curveAt("c", P1))} kHz at ${pct(P1, 0)} and ${sgn(curveAt("c", P2))} kHz at ${pct(P2, 0)}. The peak moves further, to ${sgn(curveAt("xp", P1))} and ${sgn(curveAt("xp", P2))} kHz, because the shape changes too.`,
    `Near P = 0 the line is nearly symmetric and ${(curveAt("lo", 0) + curveAt("hi", 0)).toFixed(0)} kHz wide at half height. At ${pct(P1, 0)} it is ${(curveAt("lo", P1) + curveAt("hi", P1)).toFixed(0)} kHz wide with its sharp edge above the peak (${curveAt("hi", P1).toFixed(0)} kHz above, ${curveAt("lo", P1).toFixed(0)} below). At ${pct(P2, 0)} the sharp edge is below the peak (${curveAt("lo", P2).toFixed(0)} kHz below, ${curveAt("hi", P2).toFixed(0)} above): nearly the mirror image, with a longer tail.`,
    `<b>The path matters.</b> At ${pct(at, 0)} the half width below the peak is ${w("sep-spinup")} in the two-hour spin-up and ${w("oct-rise")} in the nine-minute rise (episode numbers in brackets), and the peak for the area is ${pk("sep-spinup")} and ${pk("oct-rise")}: the faster rise fills in the side of the line toward P = 0.`,
    `<b>The two reversals go opposite ways on the way down.</b> At ${pct(at, 0)} the September line has broadened, to ${w("sep-reversal")} below the peak and a peak for the area of ${pk("sep-reversal")}. The July line has narrowed, to ${w("jul-reversal")} and ${pk("jul-reversal")}, because its low-frequency side is already turning negative.`,
    `The area per unit P is constant within each run by construction, since the logged polarization is the calibrated area.`,
  ].map((s) => `<li>${s}</li>`).join("");

  // the mix test
  const by = (kinds) => D.mix.filter((r) => kinds.includes(epOf(r.ep).kind));
  const rng = (rows, k) => `${Math.round(100 * Math.min(...rows.map((r) => r[k])))}–${Math.round(100 * Math.max(...rows.map((r) => r[k])))}%`;
  const down = by(["reversal"]);
  $("mixText").innerHTML = `Both reversals fit this picture. Taking the two-hour spin-up (episode ${epOf("sep-spinup").n}) as the line of an evenly polarized sample, ` +
    `the September reversal's lines on the way down differ from the slow line at the same polarization by ${rng(down, "single")} of the peak. ` +
    `A sum of just two slow lines at different polarizations, typically one near ${pct(median(down.map((r) => r.p1)), 0)} and one near ${pct(median(down.map((r) => r.p2)), 0)}, brings that to ${rng(down, "mix")}: ` +
    `part of the sample still highly polarized, part already near zero. In July the parts are further apart, some still positive and some already negative, and the line has both signs at once. ` +
    `This is evidence for uneven polarization, not proof: nothing here measures where in the sample the polarization is.`;

  $("nFit").textContent = String(m.n_fit);
  $("sPos").textContent = m.shift_pos.toFixed(1);
  $("sNeg").textContent = m.shift_neg.toFixed(1);
  $("settleText").textContent = pct(m.settle_pol, 0);
  $("rangeText").textContent = `${pct(m.pol_min, 0)} to ${pct(m.pol_max, 0)}`;
  const worst = epOf(m.worst.ep);
  $("errText").textContent = `The model reproduces half of them within ${(100 * m.err_median).toFixed(1)}% of the peak, nine in ten within ${(100 * m.err_p90).toFixed(1)}%, ` +
    `and all within ${(100 * m.err_max).toFixed(0)}% (the worst is at ${spct(m.worst.pol)} in episode ${worst.n}).`;
}

// ---------- reversals ----------
function drawRev() {
  for (const e of eps.filter((q) => q.kind === "reversal")) {
    const id = `cvRev_${e.run}`;
    plot(id, {
      series: e.lines.map((ln) => ({ x: ln.x, y: ln.y_mv, color: colorOf(ln.pol), width: 1.8 })),
      xmin: -200, xmax: 160, zero: true, xLabel: `offset from the ${e.runName} P = 0 line center (kHz)`,
      yLabel: "signal (mV)", readEl: `rdRev_${e.run}`,
      idle: `Episode ${e.n}: ${e.lines.length} single events from ${spct(e.p_from)} to ${spct(e.p_to)}. Hover to read values.`,
      onHover: (xv, yv) => {
        let best = null, bd = Infinity;
        for (const ln of e.lines) {
          const d = Math.abs(interp(ln.x, ln.y_mv, xv) - yv);
          if (d < bd) { bd = d; best = ln; }
        }
        return `P = ${pct(best.pol)} · ${lineWhen(best)} · ${sgn(xv)} kHz: ${interp(best.x, best.y_mv, xv).toFixed(3)} mV`;
      },
    });
  }
  rampLegend("revLegend", `<span>each line is colored by the logged polarization of its event</span>`);
}

// ---------- the model ----------
function drawModel() {
  const xs = [];
  for (let v = -250; v <= 200; v += 0.5) xs.push(v);
  const gP = modelLine(modelP, xs), g0 = modelLine(0, xs);
  const series = [{ x: xs, y: g0, color: css("--muted"), width: 1.6, dash: [2, 3] }];
  // nearest fitted example line, if one is close to this polarization
  let near = null;
  for (const e of eps) for (const ln of e.lines) {
    if (!ln.fit || !ln.yn || Math.abs(ln.pol - modelP) > 0.015) continue;
    if (!near || Math.abs(ln.pol - modelP) < Math.abs(near.pol - modelP)) near = { ...ln, n: e.n };
  }
  if (near) series.push({ x: near.x, y: near.yn, color: css("--ink-2"), width: 1.8, dash: [6, 4] });
  const col = modelP === 0 ? css("--muted") : signColor(modelP);
  series.push({ x: xs, y: gP, color: col, width: 2.2 });
  $("modelLegend").style.color = col;
  $("measLegend").hidden = !near;
  if (near) $("measLegendText").textContent = `measured, P = ${pct(near.pol)} (episode ${near.n})`;
  const mm = measure(xs, gP);
  const where = modelP < model.pol_min - 0.005 || modelP > model.pol_max + 0.005 ? "shape held beyond the measured range" : "within the measured range";
  plot("cvModel", {
    series, xmin: -200, xmax: 160, zero: true, xLabel: "offset from the P = 0 line center (kHz)",
    yLabel: "signal, area = 1 (per MHz)", readEl: "rdModel",
    idle: `P = ${pct(modelP, 0)}: center ${sgn(mm.centroid)} kHz, peak ${sgn(mm.xp)} kHz, half widths ${mm.lo.toFixed(1)} / ${mm.hi.toFixed(1)} kHz · ${where}`,
    onHover: (xv) => `${sgn(xv)} kHz: model ${interp(xs, gP, xv).toFixed(2)}` + (near ? `, measured ${interp(near.x, near.yn, xv).toFixed(2)}` : ""),
  });
  const xk = model.knots;
  plot("cvParts", {
    series: [
      { x: xk, y: model.ANEG, color: css("--copper"), width: 2 },
      { x: xk, y: model.APOS, color: css("--shell"), width: 2 },
      { x: xk, y: model.A0, color: css("--ink-2"), width: 2.2 },
    ], xmin: -200, xmax: 160, zero: true, xLabel: "offset from the P = 0 line center (kHz)",
    yLabel: "relative to the A₀ peak", readEl: "rdParts",
    onHover: (xv) => `${sgn(xv)} kHz: A₀ ${interp(xk, model.A0, xv).toFixed(3)}, A+ ${interp(xk, model.APOS, xv).toFixed(3)}, A− ${interp(xk, model.ANEG, xv).toFixed(3)}`,
  });
}

function drawAll() {
  if (!D) return;
  drawTime(); drawLines(); drawTraj(); drawRev(); drawModel();
}
const DRAW = {
  cvTime_jul: drawTime, cvTime_sep: drawTime, cvTime_oct: drawTime, cvLines: drawLines, cvC: drawTraj, cvPk: drawTraj,
  cvLo: drawTraj, cvHi: drawTraj, cvRev_jul: drawRev, cvRev_sep: drawRev, cvModel: drawModel, cvParts: drawModel,
};

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
  b.addEventListener("click", () => { setNorm(b.dataset.v); hlLine = -1; if (D) drawLines(); });
}
$("pSlider").addEventListener("input", (e) => {
  modelP = Number(e.target.value) / 100;
  $("pValue").textContent = `${modelP < 0 ? "−" : ""}${Math.abs(Number(e.target.value))}%`;
  if (D) drawModel();
});

function renderEquations() {
  const els = document.querySelectorAll("[data-tex]");
  if (!window.katex) { for (const el of els) el.textContent = el.dataset.tex; return; }
  for (const el of els) katex.render(el.dataset.tex, el, { displayMode: el.tagName === "DIV", throwOnError: false });
}

const query = new URLSearchParams(location.search);
const p0 = Number(query.get("p"));
if (Number.isFinite(p0) && p0 !== 0 && Math.abs(p0) <= 100) {
  $("pSlider").value = String(Math.round(p0));
  $("pSlider").dispatchEvent(new Event("input"));
}
renderEquations();
new ResizeObserver(() => drawAll()).observe(document.querySelector(".wrap"));
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", drawAll);
new MutationObserver(drawAll).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
fetch("proton_data.json?v=__V__").then((r) => r.json()).then((d) => {
  setupData(d);
  const want = eps.find((e) => String(e.n) === query.get("ep"));
  choose(want ? want.id : sel);
}).catch((err) => { $("rdLines").textContent = "Could not load proton_data.json: " + err; });
