"""Build the proton lineshape model and the proton page's data from NMR event logs.

    python web/tools/make_proton_data.py [log directory, default temp/]

The logs (one JSON event per line, as written by the JLab NMR DAQ) are not in the repository; temp/ is
git-ignored. They hold three runs of a 5 T polarized target in 2022 (July, September, October), split here
wherever the logs have a gap of more than a day. Most of each run is a plateau. This script works on the
EPISODES below, the stretches where the polarization moves quickly: spin-ups, knock-downs and reversals.

From them it

  1. measures every event of each episode (center of area, peak, half widths, peak height for the area) and
     picks about a dozen single events per episode as example lines,
  2. fits the polarization-dependent lineshape
         g(x; P) = A0(x - s(P)) + P+ Apos(x - s(P)) + P- Aneg(x - s(P)),
     with P+ = max(P, 0), P- = min(P, 0) and s(P) = s_pos P+ + s_neg P-, to the lines taken while the size of
     the polarization was growing: the rises, the build-up after each reversal, and the plateau before each
     reversal,
  3. tests whether lines taken during fast changes are mixes of two slow spin-up lines, and
  4. writes web/proton_lineshape.py (the simulator's model) and web/proton_data.json (the proton page's data).

Lines with both signs at once (during a reversal) and lines from knock-downs are measured and shown but not
fitted: their shape depends on how the polarization got there, not only on its value.
"""

import glob
import json
import os
import sys
import textwrap
from datetime import datetime, timezone

import numpy as np

WEB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(WEB)

RUN_GAP_S = 86400                              # a gap this long in the logs starts a new run
SKIP_LABELS = ('Junk', 'Baseline')             # events the operators marked as not a signal
RUN_NAMES = {7: ('jul', 'July'), 9: ('sep', 'September'), 10: ('oct', 'October')}

# The stretches where the polarization moves, by UTC time. kind: rise, fall or reversal. settle marks a build-up
# that starts from a knocked-down or reversed sample: its first lines still carry the shape left by what came
# before, so they are fitted only once |P| has grown past SETTLE_POL.
EPISODES = (
    dict(id='jul-reversal', kind='reversal', settle=True, start='2022-07-03 14:07:00', end='2022-07-03 21:00:00',
         title='Reversal', what='Microwaves moved from 140.12 to 140.46 GHz, the other side of the electron line.'),
    dict(id='sep-spinup', kind='rise', start='2022-09-15 00:36:00', end='2022-09-15 02:50:00',
         title='Spin-up from thermal equilibrium',
         what='Microwaves switched on at 140.2 GHz and stepped down to 140.1 GHz as the polarization grew.'),
    dict(id='sep-knock1', kind='fall', start='2022-09-16 16:57:30', end='2022-09-16 17:15:00',
         title='Knock-down', what='Microwaves at 140.47 GHz for one minute, then off.'),
    dict(id='sep-repol1', kind='rise', start='2022-09-16 18:36:30', end='2022-09-16 19:15:40',
         title='Re-polarization from zero', what='Microwaves on at 140.21 GHz after a new baseline, stepped down to 140.16 GHz.'),
    dict(id='sep-knock2', kind='fall', start='2022-09-16 19:15:40', end='2022-09-16 19:37:30',
         title='Knock-down', what='Microwaves at 140.31 GHz, between the two polarizing frequencies, for six minutes, then off.'),
    dict(id='sep-repol2', kind='rise', settle=True, start='2022-09-16 19:37:30', end='2022-09-16 20:05:00',
         title='Re-polarization from 16%', what='Microwaves back on at 140.17 GHz.'),
    dict(id='sep-reversal', kind='reversal', settle=True, start='2022-09-17 01:26:40', end='2022-09-17 03:30:00',
         title='Reversal', what='Microwaves moved from 140.16 to 140.47 GHz.'),
    dict(id='oct-rise', kind='rise', start='2022-10-09 05:19:00', end='2022-10-09 05:50:00',
         title='Fast rise', what='The polarization appears at 24% and reaches 74% in nine minutes.'),
)
N_LINES = {'rise': 12, 'fall': 10, 'reversal': 14}   # example lines per episode
ANCHOR_S = 900                                 # reversals: the plateau is the 15 minutes before the episode
SETTLE_POL = 0.30
MIN_POL = 0.05                                 # smaller lines are too noisy to measure one event at a time
ONE_SIGN = -0.03                               # a line dipping below this fraction of its peak has both signs
FIT_BIN = 0.04                                 # fitted lines: each episode's events averaged in bins of P
LINE_REGION_KHZ = (-260, 180)                  # about each run's P = 0 center; outside it the sweep is noise
CENTROID_HALF_KHZ = 130                        # measurement window, about the peak
GRID_HALF = 190                                # model grid: +-190 samples about the P = 0 center
PAGE_REGION_KHZ = (-205, 165)                  # range of the example lines written for the page
TIMELINE_EVERY = 5                             # page timeline: every 5th event outside the episodes


def stamp(s):
    return datetime.strptime(s, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp()


def iso(s):
    return datetime.fromtimestamp(s, timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')


def load_events(directory):
    t, pol, y, lab, uw = [], [], [], [], []
    f = None
    for fn in sorted(glob.glob(os.path.join(directory, '*.txt'))):
        with open(fn, encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                fs = e.get('fitsub')
                if not fs or e.get('pol') is None or e.get('label') in SKIP_LABELS:
                    continue
                if f is None:
                    f = np.asarray(e['freq_list'], float)
                if len(fs) != len(f) or abs(e['freq_list'][0] - f[0]) > 1e-9:
                    continue
                t.append(e.get('stop_stamp', 0.0))
                pol.append(e['pol'])
                y.append(fs)
                lab.append(e.get('label', ''))
                uw.append(e.get('uwave_freq') or 0.0)
    if f is None:
        sys.exit(f"no events found in {directory}")
    o = np.argsort(t)
    return dict(t=np.asarray(t)[o], pol=np.asarray(pol)[o], y=-np.asarray(y)[o],   # sign flipped: absorption > 0
                uw=np.asarray(uw)[o]), f


def split_runs(ev):
    t = ev['t']
    cut = np.nonzero(np.diff(t) > RUN_GAP_S)[0] + 1
    runs = []
    for a, b in zip(np.r_[0, cut], np.r_[cut, len(t)]):
        run = {k: v[a:b] for k, v in ev.items()}
        month = datetime.fromtimestamp(run['t'][0], timezone.utc).month
        run['id'], run['name'] = RUN_NAMES[month]
        runs.append(run)
    return runs


def measure(x, y):
    """Center of area, parabolic peak, half widths at half height and peak height for the area, of a line y
    (absorption positive) at offsets x in kHz. neg is the deepest dip below zero as a fraction of the peak."""
    h = x[1] - x[0]
    i0 = int(np.argmax(np.where(abs(x) < 150, y, -np.inf)))
    w = abs(x - x[i0]) < CENTROID_HALF_KHZ
    area = float(np.sum(y[w]) * h)
    a, b, c = y[i0 - 1], y[i0], y[i0 + 1]
    d = (a - c) / (2 * (a - 2 * b + c)) if a - 2 * b + c else 0.0
    xp, yp = x[i0] + d * h, b - 0.25 * (a - c) * d

    def cross(step):
        i = i0
        while 0 < i + step < len(y) - 1 and y[i + step] > yp / 2:
            i += step
        j = i + step
        return x[i] + (x[j] - x[i]) * (y[i] - yp / 2) / (y[i] - y[j]) if y[i] != y[j] else x[i]

    return dict(c=float(np.sum((y * x)[w]) * h / area), xp=float(xp), lo=float(xp - cross(-1)),
                hi=float(cross(+1) - xp), pk=float(1000 * yp / area), neg=float(y[w].min() / yp), area=area)


def find_episodes(runs):
    """Attach the events of each episode to it, and to its run the indices that are in any episode."""
    eps = []
    for n, spec in enumerate(EPISODES, 1):
        t0, t1 = stamp(spec['start']), stamp(spec['end'])
        run = next((r for r in runs if r['t'][0] <= t0 <= r['t'][-1]), None)
        if run is None:
            sys.exit(f"no run covers episode {spec['id']}")
        idx = np.nonzero((run['t'] >= t0) & (run['t'] < t1))[0]
        eps.append(dict(spec, n=n, run=run, idx=idx, t0=t0, t1=t1))
    return eps


def centers_and_shifts(runs, lines):
    """Each run's P = 0 line center and the two shifts, from the fitted lines' centers of area. The September
    run has lines near P = 0 and at both signs, so it fixes its own center and both shifts; the other runs'
    centers follow from those shifts."""
    f_khz = runs[0]['f_khz']
    mid = f_khz.mean()
    for ln in lines:
        ln['c'] = measure(f_khz - mid, ln['y'] * np.sign(ln['pol']))['c'] + mid
    sep = [ln for ln in lines if ln['ep']['run']['id'] == 'sep']
    P = np.array([ln['pol'] for ln in sep])
    X = np.stack([np.ones_like(P), np.maximum(P, 0), np.minimum(P, 0)], 1)
    (c0, s_pos, s_neg), *_ = np.linalg.lstsq(X, np.array([ln['c'] for ln in sep]), rcond=None)
    for run in runs:
        mine = [ln for ln in lines if ln['ep']['run'] is run]
        run['c0'] = float(c0) if run['id'] == 'sep' else float(np.mean(
            [ln['c'] - s_pos * max(ln['pol'], 0) - s_neg * min(ln['pol'], 0) for ln in mine]))
    return float(s_pos), float(s_neg)


def polarizing(ep):
    """Indices (into the run) of the episode's events taken while |P| was growing and large enough to measure."""
    run, idx = ep['run'], ep['idx']
    p = run['pol'][idx]
    least = SETTLE_POL if ep.get('settle') else MIN_POL
    if ep['kind'] == 'rise':
        keep = p >= least
    elif ep['kind'] == 'reversal':
        keep = p <= -least
    else:
        keep = np.zeros(len(idx), bool)
    return idx[keep]


def trajectory(ep):
    """Measure every event of the episode about its run's P = 0 center."""
    run = ep['run']
    x = run['f_khz'] - run['c0']
    out = dict(hours=[], pol=[], c=[], xp=[], lo=[], hi=[], pk=[], ok=[])
    for i in ep['idx']:
        p = run['pol'][i]
        m = measure(x, run['y'][i] * (np.sign(p) or 1.0))
        ok = abs(p) >= MIN_POL and m['neg'] > ONE_SIGN
        out['hours'].append((run['t'][i] - ep['t0']) / 3600)
        out['pol'].append(p)
        out['ok'].append(int(ok))
        for k in ('c', 'xp', 'lo', 'hi', 'pk'):
            out[k].append(m[k] if ok else None)
    ep['traj'] = out
    ep['one_sign'] = {int(i): bool(o) for i, o in zip(ep['idx'], out['ok'])}


def example_events(ep):
    """About N single events spread evenly along the path the polarization takes."""
    run, idx = ep['run'], ep['idx']
    path = np.r_[0, np.cumsum(abs(np.diff(run['pol'][idx])))]
    n = N_LINES[ep['kind']]
    picks = sorted({int(np.argmin(abs(path - v))) for v in np.linspace(0, path[-1], n)})
    return [int(idx[k]) for k in picks]


def fit_lines(eps):
    """The lines the model is fitted to: each episode's polarizing, one-signed events averaged in bins of P,
    and the plateau before each reversal."""
    lines = []
    for ep in eps:
        run = ep['run']
        sel = np.array([i for i in polarizing(ep) if ep['one_sign'][int(i)]], int)
        if len(sel):
            p = run['pol'][sel]
            for k in np.unique(np.floor(p / FIT_BIN)):
                m = sel[np.floor(p / FIT_BIN) == k]
                lines.append(dict(ep=ep, pol=float(run['pol'][m].mean()), y=run['y'][m].mean(0), n=len(m),
                                  kind='build-up' if ep['kind'] == 'reversal' else 'rise'))
        if ep['kind'] == 'reversal':
            m = np.nonzero((run['t'] >= ep['t0'] - ANCHOR_S) & (run['t'] < ep['t0']))[0]
            lines.append(dict(ep=ep, pol=float(run['pol'][m].mean()), y=run['y'][m].mean(0), n=len(m),
                              kind='plateau'))
    return lines


def fit_model(runs, lines, s_pos, s_neg):
    f_khz = runs[0]['f_khz']
    h = f_khz[1] - f_khz[0]
    xg = h * np.arange(-GRID_HALF, GRID_HALF + 1)
    shift = lambda p: s_pos * max(p, 0) + s_neg * min(p, 0)
    P = np.array([ln['pol'] for ln in lines])
    X = np.stack([np.ones_like(P), np.maximum(P, 0), np.minimum(P, 0)], 1)

    def aligned(ln, dc=0.0):
        x = f_khz - ln['ep']['run']['c0'] - dc
        s = np.where((x < LINE_REGION_KHZ[0]) | (x > LINE_REGION_KHZ[1]), 0, ln['y'] / ln['pol'])
        g = np.interp(xg + shift(ln['pol']), x, s, left=0, right=0)
        return g / (np.sum(g) * h)

    # The September run has lines near P = 0 and at both signs, so its center is well fixed. The other runs'
    # centers are refined by sliding their lines for the best match to the fitted shapes.
    for _ in range(3):
        G = np.array([aligned(ln) for ln in lines])
        A, *_ = np.linalg.lstsq(X, G, rcond=None)
        for run in runs:
            mine = [k for k, ln in enumerate(lines) if ln['ep']['run'] is run]
            if run['id'] == 'sep' or not mine:
                continue
            trial = np.arange(-8, 8.01, 0.1)
            cost = [sum(np.sum((aligned(lines[k], dc) - X[k] @ A) ** 2) for k in mine) for dc in trial]
            run['c0'] += float(trial[int(np.argmin(cost))])
    G = np.array([aligned(ln) for ln in lines])
    (A0, Ap, An), *_ = np.linalg.lstsq(X, G, rcond=None)
    pk = A0.max()
    sm = lambda v: np.convolve(np.pad(v, 2, mode='edge'), np.ones(5) / 5, mode='valid')
    tail = A0 < 0.02 * pk                      # faint tails: smooth every part
    for a in (A0, Ap, An):
        a[tail] = sm(a)[tail]
    n_t = 12                                   # taper the ends to zero
    tp = np.ones_like(xg)
    tp[:n_t] = np.linspace(0, 1, n_t)
    tp[-n_t:] = np.linspace(1, 0, n_t)
    A0, Ap, An = A0 * tp, Ap * tp, An * tp
    Ap -= A0 * (Ap.sum() / A0.sum())           # area independent of P
    An -= A0 * (An.sum() / A0.sum())
    A0, Ap, An = A0 / pk, Ap / pk, An / pk
    area = h * A0.sum()

    def model(p, x):                           # area 1 per kHz
        u = x - shift(p)
        return (np.interp(u, xg, A0, left=0, right=0) + max(p, 0) * np.interp(u, xg, Ap, left=0, right=0)
                + min(p, 0) * np.interp(u, xg, An, left=0, right=0)) / area

    for ln, g in zip(lines, G):
        ln['err'] = float(np.max(abs(model(ln['pol'], xg + shift(ln['pol'])) - g)) / g.max())
    errs = np.array([ln['err'] for ln in lines])
    worst = lines[int(np.argmax(errs))]
    xs = h * np.arange(-165, 128)             # at the logs' own spacing, so the measures match the data's
    curves = dict(pol=[], c=[], xp=[], lo=[], hi=[], pk=[])
    for p in np.arange(np.floor(100 * P.min()) / 100 - 0.03, P.max() + 0.035, 0.01):
        p = 0.0 if abs(p) < 1e-9 else float(p)
        m = measure(xs, model(p, xs))
        curves['pol'].append(round(p, 2))
        for k in ('c', 'xp', 'lo', 'hi', 'pk'):
            curves[k].append(m[k])
    return dict(s_pos=s_pos, s_neg=s_neg, step=float(h), A0=A0, Ap=Ap, An=An, pol_min=float(P.min()),
                pol_max=float(P.max()), n_fit=len(lines), err_median=float(np.median(errs)),
                err_p90=float(np.percentile(errs, 90)), err_max=float(errs.max()),
                worst=dict(ep=worst['ep']['id'], pol=worst['pol']), curves=curves,
                peak0=measure(xs, model(0.0, xs))['pk'])


def mix_tests(eps, lines_by_ep):
    """Is a line taken during a fast change a mix of polarizations? Compare each September example line with
    P > 0.12 (a) with the slow spin-up line at the same polarization and (b) with the best mix of two slow
    spin-up lines, by the largest difference as a fraction of the peak."""
    slow = next(ep for ep in eps if ep['id'] == 'sep-spinup')
    run = slow['run']
    x = run['f_khz'] - run['c0']
    w = abs(x) < 150
    base = np.array([i for i in polarizing(slow) if slow['one_sign'][int(i)]], int)
    base = base[np.argsort(run['pol'][base])]
    Pb, Yb = run['pol'][base], run['y'][base]
    sub = np.arange(0, len(base), 3)
    out = []
    for ep in eps:
        if ep['run'] is not run or ep is slow:
            continue
        for i in lines_by_ep[ep['id']]:
            p = run['pol'][i]
            if not (0.12 < p < Pb.max()) or not ep['one_sign'][i]:
                continue
            tgt = run['y'][i]
            j = int(np.clip(np.searchsorted(Pb, p), 1, len(Pb) - 1))
            fr = (p - Pb[j - 1]) / (Pb[j] - Pb[j - 1])
            same = p * ((1 - fr) * Yb[j - 1] / Pb[j - 1] + fr * Yb[j] / Pb[j])
            single = np.max(abs(same - tgt)[w]) / tgt[w].max()
            best = (single, p, p)
            for a in sub:
                for b in sub:
                    if b <= a:
                        continue
                    B = np.stack([Yb[a], Yb[b]], 1)[w]
                    co, *_ = np.linalg.lstsq(B, tgt[w], rcond=None)
                    if co.min() < 0:
                        continue
                    r = np.max(abs(B @ co - tgt[w])) / tgt[w].max()
                    if r < best[0]:
                        best = (r, Pb[a], Pb[b])
            out.append(dict(ep=ep['id'], pol=round(float(p), 4), single=round(float(single), 4),
                            mix=round(float(best[0]), 4), p1=round(float(best[1]), 3), p2=round(float(best[2]), 3)))
    return out


MODULE = '''"""Proton NMR absorption lineshape, measured and polarization dependent, with its exact Kramers-Kronig dispersion.

Generated by web/tools/make_proton_data.py; edit that script, not this file.

The shape comes from proton event logs of a 5 T polarized target (212.9 MHz, July, September and October 2022):
{n_fit} fit-subtracted lines taken while the size of the polarization was growing, during spin-ups from P = {pmin_pos:.2f}
to {pmax:.2f} and in the build-up after two reversals from {nmax:.2f} to {pmin:.2f}. As |P| grows the line moves away
from its P = 0 center and sharpens on that side: up and on its high-frequency side for P > 0, down and on its
low-frequency side for P < 0. The negative line is close to the mirror image of the positive one but keeps a
longer tail, so each sign has its own change of shape (the two shifts come out nearly equal):

    g(x; P) = A0(x - s(P)) + P+ APOS(x - s(P)) + P- ANEG(x - s(P)),
    P+ = max(P, 0),  P- = min(P, 0),  s(P) = {s_pos:.2f} kHz * P+ + {s_neg:.2f} kHz * P-,

with x the offset from the Larmor frequency f0 in kHz. A0, APOS and ANEG come from a least squares fit to the
lines. Half of them are reproduced within {err_median:.0f}% of their peak, nine in ten within {err_p90:.0f}%, and all within
{err_max:.0f}%. The shape of a line also depends on how fast the polarization is changing and from where, which a
function of P alone cannot follow: lines from knock-downs, from the passage through zero and from the start of
the build-up that follows either (|P| below {settle:.2f}) are not fitted. A0, the line at P = 0, is shared by both signs. Beyond the
measured range, P < {pmin:.2f} or P > {pmax:.2f}, the shape is held at its value at the end of the range and only the
shift and the size follow P. The integrals of APOS and ANEG are zero, so the area does not depend on P. Faint
tails are smoothed and both ends tapered to zero; samples every {step:.4f} kHz.

Between samples the shape is a straight line, a sum of triangular "hat" functions. The Hilbert transform of a
hat of half-width h centered at 0 is
    H[hat](x) = [F(x+h) - 2 F(x) + F(x-h)] / (pi h),   F(x) = x ln|x|,
so the dispersion chi' = -H[chi''] is exact, with the correct 1/x tails far from the line.
"""

import numpy as np

STEP_KHZ = {step:.6f}            # sample spacing
SHIFT_POS_KHZ = {s_pos:.4f}      # shift of the line per unit polarization, P > 0
SHIFT_NEG_KHZ = {s_neg:.4f}      # and P < 0
POL_MIN, POL_MAX = {pmin:.4f}, {pmax:.4f}   # measured range; the shape is held beyond it
KNOTS_KHZ = STEP_KHZ * np.arange(-{half}, {half_p1})   # offsets from the P = 0 line center
A0 = np.array([
{A0}
])
APOS = np.array([
{Ap}
])
ANEG = np.array([
{An}
])
AREA_KHZ = STEP_KHZ * float(A0.sum())     # integral of the absorption over offset in kHz (A0 peak = 1)


def shift_khz(pol):
    """Offset of the line from its P = 0 position, in kHz."""
    return SHIFT_POS_KHZ * max(pol, 0.0) + SHIFT_NEG_KHZ * min(pol, 0.0)


def _parts(pol):
    """(table, weight) pairs that make up the shape at polarization pol."""
    p = min(max(float(pol), POL_MIN), POL_MAX)
    return [(A0, 1.0), (APOS if p > 0 else ANEG, p)]


def _xlogx(v):
    a = np.abs(v)
    return np.where(a > 0, v * np.log(np.where(a > 0, a, 1.0)), 0.0)


def _hilbert_table(x, table):
    """Exact Hilbert transform of the piecewise-linear function through (KNOTS_KHZ, table)."""
    out = np.zeros_like(x)
    h = STEP_KHZ
    nz = np.nonzero(table)[0]
    for lo in range(0, len(nz), 64):                        # chunks keep memory small
        k = nz[lo:lo + 64]
        u = x[:, None] - KNOTS_KHZ[k][None, :]
        out += ((_xlogx(u + h) - 2 * _xlogx(u) + _xlogx(u - h)) / (np.pi * h)) @ table[k]
    return out


def absorption(x_khz, pol):
    """Absorption shape (A0 peak = 1) at offsets x_khz from f0, for polarization pol."""
    u = np.asarray(x_khz, float) - shift_khz(float(pol))
    return sum(w * np.interp(u, KNOTS_KHZ, a, left=0.0, right=0.0) for a, w in _parts(pol) if w)


def hilbert(x_khz, pol):
    """Hilbert transform of absorption(), exactly."""
    u = np.atleast_1d(np.asarray(x_khz, float)) - shift_khz(float(pol))
    return sum(w * _hilbert_table(u, a) for a, w in _parts(pol) if w)


def susceptibility(f_mhz, pol, f0_mhz, chi0_per_pol):
    """Complex susceptibility chi = chi' - i chi'' at frequencies f_mhz, normalized like the deuteron and the
    notebook's Hill & Hill line: the integral of chi'' over omega is (pi/2) omega0 chi0, chi0 = chi0_per_pol * pol."""
    pol = float(pol)
    x = (np.asarray(f_mhz, float) - f0_mhz) * 1e3
    k = (np.pi / 2) * f0_mhz * 1e3 * chi0_per_pol * pol / AREA_KHZ
    return -k * hilbert(x, pol) - 1j * k * absorption(x, pol)
'''


def write_module(lines, model, path):
    fmt = lambda a: textwrap.fill(', '.join(f'{v:.5f}' for v in a), 112, initial_indent='    ',
                                  subsequent_indent='    ')
    P = np.array([ln['pol'] for ln in lines])
    src = MODULE.format(n_fit=model['n_fit'], pmin_pos=P[P > 0].min(), nmax=P[P < 0].max(),
                        s_pos=model['s_pos'], s_neg=model['s_neg'], err_median=np.ceil(100 * model['err_median']),
                        err_p90=np.ceil(100 * model['err_p90']), err_max=np.ceil(100 * model['err_max']),
                        pmin=model['pol_min'], pmax=model['pol_max'], step=model['step'], half=GRID_HALF,
                        half_p1=GRID_HALF + 1, settle=SETTLE_POL, A0=fmt(model['A0']), Ap=fmt(model['Ap']), An=fmt(model['An']))
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(src)


def rnd(v, d):
    return [None if a is None else round(float(a), d) for a in v]


def write_page_data(runs, eps, lines_by_ep, fitted, model, mixes, path):
    f_khz = runs[0]['f_khz']
    out_runs = []
    for run in runs:
        x = f_khz - run['c0']
        run['mask'] = (x >= PAGE_REGION_KHZ[0]) & (x <= PAGE_REGION_KHZ[1])
        t, pol = run['t'], run['pol']
        mine = [ep for ep in eps if ep['run'] is run]
        t_start = t[0] if run['id'] != 'oct' else mine[0]['t0'] - 3600
        keep = np.zeros(len(t), bool)
        keep[::TIMELINE_EVERY] = True
        for ep in mine:
            keep |= (t >= ep['t0'] - 600) & (t < ep['t1'])
        keep &= t >= t_start
        run['t_start'] = t_start
        out_runs.append(dict(id=run['id'], name=run['name'], start=iso(t_start), center_mhz=round(run['c0'] / 1e3, 5),
                             x_khz=rnd(x[run['mask']], 3),
                             timeline=dict(hours=rnd((t[keep] - t_start) / 3600, 4), pol=rnd(pol[keep], 4))))
    out_eps, out_lines = [], []
    for ep in eps:
        run, tr = ep['run'], ep['traj']
        p = np.array(tr['pol'])
        first = len(out_lines)
        for i in lines_by_ep[ep['id']]:
            out_lines.append(dict(ep=ep['id'], time=iso(run['t'][i]), pol=round(float(run['pol'][i]), 4),
                                  minutes=round((run['t'][i] - ep['t0']) / 60, 2),
                                  one_sign=ep['one_sign'][i],
                                  fit=bool(i in set(polarizing(ep).tolist()) and ep['one_sign'][i]),
                                  y_mv=rnd(1000 * run["y"][i][run["mask"]], 3)))
        out_eps.append(dict(
            id=ep['id'], n=ep['n'], run=run['id'], kind=ep['kind'], title=ep['title'], what=ep['what'],
            start=iso(ep['t0']), start_hours=round((ep['t0'] - run['t_start']) / 3600, 4),
            end_hours=round((ep['t1'] - run['t_start']) / 3600, 4),
            p_from=round(float(p[0]), 4), p_to=round(float(p[-1]), 4), p_min=round(float(p.min()), 4),
            p_max=round(float(p.max()), 4), events=len(p), lines=[first, len(out_lines)],
            traj=dict(minutes=rnd(60 * np.array(tr['hours']), 2), pol=rnd(p, 4), ok=tr['ok'],
                      c=rnd(tr['c'], 2), xp=rnd(tr['xp'], 2), lo=rnd(tr['lo'], 2), hi=rnd(tr['hi'], 2),
                      pk=rnd(tr['pk'], 3))))
    cv = model['curves']
    data = dict(
        source='Proton event logs of a 5 T polarized target, July, September and October 2022 (512-point sweeps '
               'of 212.482-213.282 MHz)',
        step_khz=round(model['step'], 5), runs=out_runs, episodes=out_eps, lines=out_lines,
        fitted=[dict(ep=ln['ep']['id'], kind=ln['kind'], pol=round(ln['pol'], 4), n=ln['n'], err=round(ln['err'], 4))
                for ln in fitted],
        mix=mixes,
        model=dict(shift_pos=round(model['s_pos'], 4), shift_neg=round(model['s_neg'], 4),
                   pol_min=round(model['pol_min'], 4), pol_max=round(model['pol_max'], 4),
                   step_khz=round(model['step'], 6), half=GRID_HALF, A0=rnd(model['A0'], 5),
                   APOS=rnd(model['Ap'], 5), ANEG=rnd(model['An'], 5), n_fit=model['n_fit'],
                   err_median=round(model['err_median'], 4), err_p90=round(model['err_p90'], 4),
                   err_max=round(model['err_max'], 4), worst=model['worst'], peak0=round(model['peak0'], 4),
                   settle_pol=SETTLE_POL, min_pol=MIN_POL,
                   curves=dict(pol=cv['pol'], c=rnd(cv['c'], 2), xp=rnd(cv['xp'], 2), lo=rnd(cv['lo'], 2),
                               hi=rnd(cv['hi'], 2), pk=rnd(cv['pk'], 3))),
    )
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(data, fh, separators=(',', ':'))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    directory = args[0] if args else os.path.join(REPO, 'temp')
    ev, f = load_events(directory)
    runs = split_runs(ev)
    for run in runs:
        run['f_khz'] = 1e3 * f
    eps = find_episodes(runs)
    for run in runs:
        run['c0'] = float(run['f_khz'].mean())  # until the centers are fitted
    for ep in eps:
        trajectory(ep)                          # which events have one sign
    fitted = fit_lines(eps)
    s_pos, s_neg = centers_and_shifts(runs, fitted)
    model = fit_model(runs, fitted, s_pos, s_neg)
    for ep in eps:                              # again, about the fitted centers
        trajectory(ep)
    lines_by_ep = {ep['id']: example_events(ep) for ep in eps}
    mixes = mix_tests(eps, lines_by_ep)
    write_module(fitted, model, os.path.join(WEB, 'proton_lineshape.py'))
    write_page_data(runs, eps, lines_by_ep, fitted, model, mixes, os.path.join(WEB, 'proton_data.json'))

    print(f"{len(ev['t'])} events in {len(runs)} runs, {len(eps)} episodes; {model['n_fit']} fitted lines; shift "
          f"{model['s_pos']:.2f} / {model['s_neg']:.2f} kHz per unit P")
    print("P = 0 centers: " + ", ".join(f"{r['name']} {r['c0'] / 1e3:.5f} MHz" for r in runs))
    print(f"fit error: median {100 * model['err_median']:.1f}%, 90% within {100 * model['err_p90']:.1f}%, worst "
          f"{100 * model['err_max']:.1f}% ({model['worst']['ep']} at P = {model['worst']['pol']:+.3f})")
    for ep in eps:
        p = ep['traj']['pol']
        errs = [ln['err'] for ln in fitted if ln['ep'] is ep]
        print(f"  {ep['n']} {ep['id']:13s} {ep['kind']:8s} {len(p):3d} events, P {p[0]:+.3f} -> {p[-1]:+.3f}"
              + (f", {len(errs)} fitted, worst {100 * max(errs):.1f}%" if errs else ""))
        if '-v' in sys.argv:
            for ln in fitted:
                if ln['ep'] is ep:
                    print(f"      {ln['kind']:8s} P = {ln['pol']:+.3f}  n = {ln['n']:3d}  error {100 * ln['err']:.1f}%")
    for kind in ('rise', 'fall', 'reversal'):
        m = [r for r in mixes if next(e for e in eps if e['id'] == r['ep'])['kind'] == kind]
        if m:
            print(f"  mix test, {kind}: one P {100 * min(r['single'] for r in m):.0f}-{100 * max(r['single'] for r in m):.0f}%"
                  f", two-P mix {100 * min(r['mix'] for r in m):.0f}-{100 * max(r['mix'] for r in m):.0f}% of peak")


if __name__ == '__main__':
    main()
