"use strict";
// The proton line at high polarization: measured lines (proton_data.json, made by tools/make_proton_data.py)
// and the simulator's model g(x; P) = A0(u) + P A±(u), u = x - s± P.

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
const P_SCALE = [0.1, 0.8];          // |P| mapped onto the ramps
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

const pct = (p, d = 1) => `${p < 0 ? "−" : ""}${Math.abs(100 * p).toFixed(d)}%`;
const spct = (p, d = 0) => `${p > 0 ? "+" : ""}${pct(p, d)}`;
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

// spec: {series:[{x,y,color,width,dash,alpha}], points:[{x,y,color,r,hollow}], bands:[{x0,x1}], xLabel, yLabel,
//        xFmt, yFmt, xStep, xmin,xmax,ymin,ymax, zero, readEl, idle, onHover(xv, yv) -> html}
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

  for (const b of spec.bands || []) {
    const x0 = Math.max(m.l, X(b.x0)), x1 = Math.min(m.l + pw, X(b.x1));
    if (x1 <= x0) continue;
    ctx.fillStyle = css("--muted"); ctx.globalAlpha = 0.16;
    ctx.fillRect(x0, m.t, x1 - x0, ph); ctx.globalAlpha = 1;
  }
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
    ctx.beginPath(); ctx.arc(X(p.x), Y(p.y), p.r || 4.5, 0, 2 * Math.PI);
    if (p.hollow) {
      ctx.fillStyle = css("--canvas-bg"); ctx.fill();
      ctx.strokeStyle = p.color; ctx.lineWidth = 2; ctx.stroke();
    } else {
      ctx.fillStyle = p.color; ctx.fill();
      ctx.strokeStyle = css("--canvas-bg"); ctx.lineWidth = 2; ctx.stroke();
    }
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

// ---------- line measurements ----------
const REGION = [-290, 150];   // kHz from the P = 0 center: the line; outside it the sweep is noise
function measure(x, y) {      // y with absorption positive
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
let runs = {};           // id -> run
let lines = [];          // settled lines (the model's fit), sorted by polarization
let revLines = [];       // single events from the reversal, in time order
let model = null;
let norm = "area";
let modelP = 0.5;
let hlLine = -1;
let curves = null;       // model measurements against P

function setupData(d) {
  D = d;
  runs = Object.fromEntries(d.runs.map((r) => [r.id, r]));
  const all = d.lines.map((ln) => {
    const x = runs[ln.run].x_khz, s = Math.sign(ln.pol) || 1;
    const ya = ln.y.map((v) => s * v);
    const mm = measure(x, ya);
    return { ...ln, x, ...mm, yn: ya.map((v) => 1000 * v / mm.area) };   // area-normalized, per MHz
  });
  lines = all.filter((l) => l.fit).sort((a, b) => a.pol - b.pol);
  revLines = all.filter((l) => !l.fit).sort((a, b) => a.hours - b.hours);
  const md = d.model;
  const knots = md.A0.map((_, j) => (j - md.half) * md.step_khz);
  const area = md.step_khz * md.A0.reduce((s, v) => s + v, 0);
  model = { ...md, knots, area };
  const neg = lines.filter((l) => l.pol < 0), pos = lines.filter((l) => l.pol > 0);
  model.gap = [neg[neg.length - 1].pol, pos[0].pol];
  // model measurements across the polarization range, for the "what changes" curves
  const xs = []; for (let v = -260; v <= 200; v += 0.5) xs.push(v);
  curves = [];
  for (let p = -0.66; p <= 0.83 + 1e-9; p += 0.01) {
    if (Math.abs(p) < 1e-9) p = 0;
    curves.push({ pol: p, ...measure(xs, modelLine(p, xs)) });
  }
  for (const k of ["centroid", "xp", "lo", "hi", "yp"]) {
    const v = curves.map((c) => c[k]);
    curves.forEach((c, i) => {
      const a = Math.max(0, i - 3), b = Math.min(v.length, i + 4);
      c[k] = v.slice(a, b).reduce((t, w) => t + w, 0) / (b - a);
    });
  }
  const m0 = measure(xs, modelLine(0, xs));
  model.peak0 = m0.yp / m0.area * 1000;
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

// ---------- charts ----------
const startOf = (run) => new Date(run.start.replace(" UTC", "Z").replace(" ", "T"));
const hourOfDay = (run) => { const t = startOf(run); return t.getUTCHours() + t.getUTCMinutes() / 60 + t.getUTCSeconds() / 3600; };
function clock(h) {                    // hours since 00:00 UTC -> "HH:MM"
  const m = Math.round(h * 60);
  return `${String(Math.floor(m / 60) % 24).padStart(2, "0")}:${String(m % 60).padStart(2, "0")}`;
}
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
function dayClock(run, h) {            // hours since 00:00 UTC of the start day -> "4 Jul 12:00"
  const t = startOf(run), d = new Date(Date.UTC(t.getUTCFullYear(), t.getUTCMonth(), t.getUTCDate()) + Math.floor(h / 24 + 1e-9) * 864e5);
  return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]} ${clock(h)}`;
}
const whenOf = (ln) => {
  const run = runs[ln.run];
  return run.id === "jul" ? dayClock(run, ln.hours + hourOfDay(run)) : clock(ln.hours + hourOfDay(run));
};
const kindText = { rise: "October rise", decay: "October decay", plateau: "July plateau", "build-up": "July build-up", reversal: "July reversal" };
const describe = (ln) => `${kindText[ln.kind]}, ${ln.n > 1 ? `average of ${ln.n} events around ${whenOf(ln)}` : `event at ${whenOf(ln)}`}`;

function drawTimeRun(id, run, readEl, opt) {
  const tl = run.timeline, h0 = hourOfDay(run);
  const hx = tl.hours.map((h) => h + h0);
  const pts = lines.filter((l) => l.run === run.id).map((ln) => ({ x: ln.hours + h0, y: 100 * ln.pol, color: colorOf(ln.pol), r: 5 }));
  const bands = run.reversal_hours != null ? [{ x0: run.reversal_hours + h0, x1: run.settled_hours + h0 }] : [];
  plot(id, {
    series: [{ x: hx, y: tl.pol.map((p) => 100 * p), color: css("--ink-2"), width: 1.6 }],
    points: pts, bands, zero: true, ymin: opt.ymin, ymax: 85, xLabel: opt.xLabel, yLabel: "polarization (%)",
    xFmt: opt.tick || opt.fmt, xStep: opt.xStep, readEl,
    onHover: (xv) => {
      const i = nearestIndex(hx, xv);
      return `${opt.fmt(hx[i])} UTC · P = ${pct(tl.pol[i])}`;
    },
  });
}
function drawTime() {
  const o = runs.oct, j = runs.jul;
  drawTimeRun("cvTimeO", o, "rdTimeO", { fmt: clock, ymin: 0, xLabel: "time on 9 October 2022 (UTC)" });
  const w = $("cvTimeJ").getBoundingClientRect().width;
  drawTimeRun("cvTimeJ", j, "rdTimeJ", {
    fmt: (h) => dayClock(j, h), tick: (h) => (Math.abs(h % 24) < 1e-6 ? dayClock(j, h).slice(0, -6) : clock(h)),
    xStep: w > 420 ? 12 : 24, ymin: -70, xLabel: "3 to 6 July 2022 (UTC)",
  });
  $("timeLegend").innerHTML =
    `<span style="color:var(--ink-2)"><i class="sw" style="background:currentColor;height:2px"></i><span style="color:var(--muted)">each event</span></span>` +
    `<span><i class="sw dot-m" style="color:${colorOf(0.6)}"></i><i class="sw dot-m" style="color:${colorOf(-0.6)}"></i><span>example lines, colored by polarization as below</span></span>` +
    `<span><i class="sw band"></i><span>reversal and settling, left out of the model</span></span>`;
}

function rampLegend(el) {
  const bar = (key) => `<span class="ramp" style="background:linear-gradient(90deg, ${RAMPS[key][isDark() ? "dark" : "light"].join(",")})"></span>`;
  $(el).innerHTML =
    `<span class="group"><span>P &gt; 0</span>${bar("pos")}<span class="ramp-label">+${100 * P_SCALE[0]}% → +${100 * P_SCALE[1]}%</span></span>` +
    `<span class="group"><span>P &lt; 0</span>${bar("neg")}<span class="ramp-label">−${100 * P_SCALE[0]}% → −${100 * P_SCALE[1]}%</span></span>`;
}

function drawLines() {
  const key = norm === "area" ? "yn" : "y";
  const scale = norm === "area" ? 1 : 1000;          // as measured: mV
  const series = lines.map((ln, i) => ({
    x: ln.x, y: ln[key].map((v) => v * scale), color: colorOf(ln.pol),
    width: i === hlLine ? 3 : 1.8, alpha: hlLine < 0 || i === hlLine ? 1 : 0.3,
  }));
  if (hlLine >= 0) series.push(series.splice(hlLine, 1)[0]);       // highlighted line on top
  plot("cvLines", {
    series, xmin: -220, xmax: 160, zero: true,
    xLabel: "offset from the run's P = 0 line center (kHz)",
    yLabel: norm === "area" ? "signal, area = 1 (per MHz)" : "signal (mV)", readEl: "rdLines",
    onHover: (xv, yv) => {
      let best = -1, bd = Infinity;
      lines.forEach((ln, i) => {
        const d = Math.abs(interp(ln.x, ln[key], xv) * scale - yv);
        if (d < bd) { bd = d; best = i; }
      });
      if (best !== hlLine) { hlLine = best; requestAnimationFrame(drawLines); }
      const ln = lines[best];
      const v = interp(ln.x, ln[key], xv) * scale;
      return `P = ${pct(ln.pol)} · ${describe(ln)} · ${sgn(xv)} kHz: ${v.toFixed(norm === "area" ? 2 : 3)}${norm === "area" ? "" : " mV"}`;
    },
  });
  rampLegend("rampLegend");
}

function drawRev() {
  const plat = lines.find((l) => l.kind === "plateau");
  const end = lines.filter((l) => l.kind === "build-up").reduce((a, b) => (b.n > a.n ? b : a));
  const refs = [plat, end];
  const series = refs.map((ln) => ({ x: ln.x, y: ln.y.map((v) => 1000 * v), color: colorOf(ln.pol), width: 1.6, dash: [6, 4] }));
  for (const ln of revLines) series.push({ x: ln.x, y: ln.y.map((v) => 1000 * v), color: colorOf(ln.pol), width: 2 });
  const t0 = runs.jul.reversal_hours;
  plot("cvRev", {
    series, xmin: -220, xmax: 160, zero: true, xLabel: "offset from the July P = 0 line center (kHz)",
    yLabel: "signal (mV)", readEl: "rdRev",
    onHover: (xv, yv) => {
      let best = null, bd = Infinity;
      for (const ln of [...revLines, ...refs]) {
        const d = Math.abs(1000 * interp(ln.x, ln.y, xv) - yv);
        if (d < bd) { bd = d; best = ln; }
      }
      const when = best.kind === "reversal" ? `${Math.round(60 * (best.hours - t0))} min after the switch` : describe(best);
      return `P = ${pct(best.pol)} · ${when} · ${sgn(xv)} kHz: ${(1000 * interp(best.x, best.y, xv)).toFixed(3)} mV`;
    },
  });
  $("revLegend").innerHTML =
    revLines.map((ln) => `<span><i class="sw" style="background:${colorOf(ln.pol)}"></i><span>${spct(ln.pol)}, ${Math.round(60 * (ln.hours - t0))} min</span></span>`).join("") +
    refs.map((ln) => `<span style="color:${colorOf(ln.pol)}"><i class="sw dash"></i><span style="color:var(--muted)">${spct(ln.pol)} ${ln.kind === "plateau" ? "plateau" : "settled"}</span></span>`).join("");
}

function scatterSpec(id, readEl, sets, yLabel, fmtRead, extra = {}) {
  const series = [], points = [];
  for (const s of sets) {
    series.push({ x: curves.map((c) => 100 * c.pol), y: curves.map(s.val), color: css("--muted"), width: 1.4, dash: [5, 4] });
    for (const ln of lines) points.push({ x: 100 * ln.pol, y: s.val(ln), color: colorOf(ln.pol), r: 4.5, hollow: s.hollow });
  }
  plot(id, {
    series, points, xmin: -66, xmax: 83, xLabel: "polarization (%)", yLabel, readEl, ...extra,
    onHover: (xv) => fmtRead(lines[nearestIndex(lines.map((l) => 100 * l.pol), xv)]),
  });
}

function drawChanges() {
  scatterSpec("cvShift", "rdShift", [
    { val: (l) => l.centroid },
    { val: (l) => l.xp, hollow: true },
  ], "offset (kHz)", (l) => `P = ${pct(l.pol)}: center ${sgn(l.centroid)} kHz, peak ${sgn(l.xp)} kHz`, { zero: true });
  scatterSpec("cvWidth", "rdWidth", [
    { val: (l) => l.lo },
    { val: (l) => l.hi, hollow: true },
  ], "half width (kHz)", (l) => `P = ${pct(l.pol)}: ${l.lo.toFixed(1)} kHz below, ${l.hi.toFixed(1)} kHz above the peak`, { ymin: 0 });
  const rel = (l) => l.yp / l.area * 1000 / model.peak0;
  scatterSpec("cvPeak", "rdPeak", [{ val: rel }],
    "peak ÷ area, relative", (l) => `P = ${pct(l.pol)}: peak for the area ×${rel(l).toFixed(3)}`,
    { ymin: 0.9, ymax: 1.3, yFmt: (v) => v.toFixed(2) });

  const pos = lines.filter((l) => l.pol > 0), neg = lines.filter((l) => l.pol < 0);
  const hiP = pos[pos.length - 1], loP = pos[0], hiN = neg[0];
  const mirror = pos.reduce((a, b) => (Math.abs(b.pol + hiN.pol) < Math.abs(a.pol + hiN.pol) ? b : a));
  // the two runs' +69% lines, compared on the October line's grid
  const plat = lines.find((l) => l.kind === "plateau");
  const twin = pos.filter((l) => l.run === "oct").reduce((a, b) => (Math.abs(b.pol - plat.pol) < Math.abs(a.pol - plat.pol) ? b : a));
  let dmax = 0, pk = 0;
  twin.x.forEach((x, i) => { if (x > -200 && x < 150) { dmax = Math.max(dmax, Math.abs(interp(plat.x, plat.yn, x) - twin.yn[i])); pk = Math.max(pk, twin.yn[i]); } });
  $("findings").innerHTML = [
    `For positive polarization the center of area moves up <b>${model.shift_pos.toFixed(1)} kHz per unit P</b>, to ${sgn(hiP.centroid)} kHz at ${pct(hiP.pol, 0)}. For negative polarization it moves down only <b>${model.shift_neg.toFixed(1)} kHz per unit P</b>, to ${sgn(hiN.centroid)} kHz at ${pct(hiN.pol, 0)}. The peak moves further, to ${sgn(hiP.xp)} and ${sgn(hiN.xp)} kHz, because the shape changes too.`,
    `At ${pct(hiP.pol, 0)} the line is ${(hiP.lo + hiP.hi).toFixed(1)} kHz wide at half height, with its sharp edge above the peak (${hiP.hi.toFixed(1)} kHz above, ${hiP.lo.toFixed(1)} below; at ${pct(loP.pol, 0)}, ${loP.hi.toFixed(1)} and ${loP.lo.toFixed(1)}). At ${pct(hiN.pol, 0)} the sharp edge is below the peak: ${hiN.lo.toFixed(1)} kHz below, ${hiN.hi.toFixed(1)} above.`,
    `It is not an exact mirror image. At ${pct(hiN.pol, 0)} the line is ${(hiN.lo + hiN.hi).toFixed(1)} kHz wide and its peak for the area is ×${rel(hiN).toFixed(2)} that of the model at P = 0; at ${pct(mirror.pol, 0)} the line is ${(mirror.lo + mirror.hi).toFixed(1)} kHz wide and ×${rel(mirror).toFixed(2)}.`,
    `The +${Math.round(100 * plat.pol)}% lines of the two runs, July's plateau and October's event at ${pct(twin.pol, 1)}, agree within ${(100 * dmax / pk).toFixed(0)}% of the peak, so the difference between the signs is not a difference between the runs. The area per unit P is constant within each run by construction, since the logged polarization is the calibrated area.`,
  ].map((s) => `<li>${s}</li>`).join("");
}

function drawModel() {
  const xs = [];
  for (let v = -250; v <= 200; v += 0.5) xs.push(v);
  const gP = modelLine(modelP, xs), g0 = modelLine(0, xs);
  const series = [{ x: xs, y: g0, color: css("--muted"), width: 1.6, dash: [2, 3] }];
  // nearest settled line, if one is close to this polarization
  let near = null;
  for (const ln of lines) if (Math.abs(ln.pol - modelP) <= 0.015 && (!near || Math.abs(ln.pol - modelP) < Math.abs(near.pol - modelP))) near = ln;
  if (near) series.push({ x: near.x, y: near.yn, color: css("--ink-2"), width: 1.8, dash: [6, 4] });
  const col = modelP === 0 ? css("--muted") : signColor(modelP);
  series.push({ x: xs, y: gP, color: col, width: 2.2 });
  $("modelLegend").style.color = col;
  $("measLegend").hidden = !near;
  if (near) $("measLegendText").textContent = `measured, P = ${pct(near.pol)}`;
  const mm = measure(xs, gP);
  const [g0p, g1p] = model.gap;
  const where = modelP < model.pol_min || modelP > model.pol_max ? "shape held beyond the measured range"
    : modelP > g0p && modelP < g1p ? `interpolated between ${pct(g0p, 0)} and ${pct(g1p, 0)}` : "within the measured range";
  plot("cvModel", {
    series, xmin: -220, xmax: 160, zero: true, xLabel: "offset from the P = 0 line center (kHz)",
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
    ], xmin: -220, xmax: 160, zero: true, xLabel: "offset from the P = 0 line center (kHz)",
    yLabel: "relative to the A₀ peak", readEl: "rdParts",
    onHover: (xv) => `${sgn(xv)} kHz: A₀ ${interp(xk, model.A0, xv).toFixed(3)}, A+ ${interp(xk, model.APOS, xv).toFixed(3)}, A− ${interp(xk, model.ANEG, xv).toFixed(3)}`,
  });
  $("maxErr").textContent = `${(100 * model.max_err).toFixed(1)}%`;
  $("nFit").textContent = String(lines.length);
  $("sPos").textContent = model.shift_pos.toFixed(1);
  $("sNeg").textContent = model.shift_neg.toFixed(1);
  $("rangeText").textContent = `${pct(model.pol_min, 0)} to ${pct(model.pol_max, 0)}`;
}

function drawAll() {
  if (!D) return;
  drawTime(); drawLines(); drawRev(); drawChanges(); drawModel();
}
const DRAW = {
  cvTimeO: drawTime, cvTimeJ: drawTime, cvLines: drawLines, cvRev: drawRev, cvShift: drawChanges, cvWidth: drawChanges,
  cvPeak: drawChanges, cvModel: drawModel, cvParts: drawModel,
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
  b.addEventListener("click", () => {
    norm = b.dataset.v;
    for (const o of $("normSeg").querySelectorAll("button")) o.setAttribute("aria-pressed", String(o === b));
    drawLines();
  });
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

const p0 = Number(new URLSearchParams(location.search).get("p"));
if (Number.isFinite(p0) && p0 !== 0 && Math.abs(p0) <= 100) {
  $("pSlider").value = String(Math.round(p0));
  $("pSlider").dispatchEvent(new Event("input"));
}
renderEquations();
new ResizeObserver(() => drawAll()).observe(document.querySelector(".wrap"));
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", drawAll);
new MutationObserver(drawAll).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
fetch("proton_data.json?v=__V__").then((r) => r.json()).then((d) => { setupData(d); drawAll(); })
  .catch((err) => { $("rdLines").textContent = "Could not load proton_data.json: " + err; });
