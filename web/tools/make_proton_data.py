"""Build the proton lineshape model and the proton page's data from NMR event logs.

    python web/tools/make_proton_data.py [log directory, default temp/]

The logs (one JSON event per line, as written by the JLab NMR DAQ) are not in the repository; temp/ is
git-ignored. They hold two runs of a 5 T polarized target, split here wherever the logs have a gap of more than
a day:

  * October 2022, positive polarization: a fast rise from near zero to 74%, then a slow climb to 79% and decay.
  * July 2022, a reversal: a plateau at +69%, the microwave frequency switched, the polarization through zero in
    under 20 minutes, and a slow build-up to -61%.

From them this script

  1. picks the example lines: every single event of the October rise, averages of the October decay binned by
     polarization, the average of the July plateau, averages of the July build-up binned by polarization, and
     a few single events from the July reversal,
  2. fits the polarization-dependent lineshape
         g(x; P) = A0(x - s(P)) + P+ Apos(x - s(P)) + P- Aneg(x - s(P)),
     with P+ = max(P, 0), P- = min(P, 0) and s(P) = s_pos P+ + s_neg P-, to every example line except those of the
     reversal, and
  3. writes web/proton_lineshape.py (the simulator's model) and web/proton_data.json (the proton page's data).

Lines from the reversal and the first SETTLE_H hours after it are left out of the fit. While the polarization
reverses, the line has both signs at once, which no single spin temperature can give, and for a while after it
the line keeps changing shape at the same polarization.
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
N_SPINUP = 8                                   # October: single events of the fast rise
DECAY_BINS = ((0.785, 0.80), (0.77, 0.785), (0.755, 0.77), (0.74, 0.755))
PLATEAU_MIN = 0.68                             # July: the positive plateau before the reversal
SETTLE_H = 1.4                                 # July: hours after the zero crossing left out of the fit
NEG_BINS = ((-0.40, -0.35), (-0.45, -0.40), (-0.50, -0.45), (-0.55, -0.50), (-0.58, -0.55), (-0.60, -0.58),
            (-0.62, -0.60))
REVERSAL_P = (0.52, 0.24, 0.12, -0.05, -0.12, -0.19)   # July: single events shown from the reversal
LINE_REGION_KHZ = (-260, 180)                  # about each run's P = 0 center; outside it the sweep is noise
CENTROID_HALF_KHZ = 130                        # centroid window, about the peak
GRID_HALF = 190                                # model grid: +-190 samples about the P = 0 center
PAGE_REGION_KHZ = (-300, 220)                  # range of the example lines written for the page
TIMELINE_EVERY = 5                             # page timeline: every 5th event, except near the reversal


def load_events(directory):
    t, pol, y = [], [], []
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
                if not fs or e.get('pol') is None:
                    continue
                if f is None:
                    f = np.asarray(e['freq_list'], float)
                if len(fs) != len(f) or abs(e['freq_list'][0] - f[0]) > 1e-9:
                    continue
                t.append(e.get('stop_stamp', 0.0))
                pol.append(e['pol'])
                y.append(fs)
    if f is None:
        sys.exit(f"no events found in {directory}")
    o = np.argsort(t)
    return np.asarray(t)[o], np.asarray(pol)[o], -np.asarray(y)[o], f   # sign flipped: absorption > 0


def split_runs(t, pol, y):
    cut = np.nonzero(np.diff(t) > RUN_GAP_S)[0] + 1
    return [dict(t=t[a:b], pol=pol[a:b], y=y[a:b]) for a, b in zip(np.r_[0, cut], np.r_[cut, len(t)])]


def _line(run, idx, kind, fit=True):
    idx = np.atleast_1d(idx)
    return dict(t=run['t'][idx].mean(), pol=run['pol'][idx].mean(), n=len(idx), kind=kind, fit=fit,
                y=run['y'][idx].mean(0))


def october_lines(run):
    t, pol = run['t'], run['pol']
    sel = np.nonzero(pol > 0.05)[0]
    lines = [_line(run, i, 'rise') for i in sel[:N_SPINUP]]
    rest = sel[N_SPINUP:]
    for a, b in DECAY_BINS:
        m = rest[(pol[rest] >= a) & (pol[rest] < b)]
        if len(m):
            lines.append(_line(run, m, 'decay'))
    run.update(id='oct', t_start=t[sel[0]] - 3600, t_rev=None)
    return lines


def july_lines(run):
    t, pol = run['t'], run['pol']
    i_rev = np.nonzero(pol < PLATEAU_MIN)[0][0]
    i_zero = i_rev + np.nonzero(pol[i_rev:] < 0)[0][0]
    t_settled = t[i_zero] + 3600 * SETTLE_H
    lines = [_line(run, np.nonzero((np.arange(len(t)) < i_rev) & (pol >= PLATEAU_MIN))[0], 'plateau')]
    rev = np.nonzero((np.arange(len(t)) >= i_rev) & (t < t_settled))[0]
    for p in REVERSAL_P:
        lines.append(_line(run, rev[np.argmin(abs(pol[rev] - p))], 'reversal', fit=False))
    late = t >= t_settled
    for a, b in NEG_BINS:
        m = np.nonzero(late & (pol >= a) & (pol < b))[0]
        if len(m):
            lines.append(_line(run, m, 'build-up'))
    run.update(id='jul', t_start=t[0], t_rev=t[i_rev - 1], t_zero=t[i_zero], t_settled=t_settled)
    return lines


def centroid(f, y, pol):
    yy = y * np.sign(pol)
    x = 1e3 * f
    i0 = np.argmax(np.where((x > x[0] + 250) & (x < x[-1] - 250), yy, -np.inf))
    w = abs(x - x[i0]) < CENTROID_HALF_KHZ
    return np.sum((yy * x)[w]) / np.sum(yy[w])


def fit_model(oct_lines, jul_lines, f):
    """Shift and shape. October fixes the positive shift; the July plateau, whose shape matches October's at
    the same polarization, places July's P = 0 center; July's build-up then fixes the negative shift."""
    h = 1e3 * (f[1] - f[0])
    for ln in oct_lines + jul_lines:
        ln['c'] = centroid(f, ln['y'], ln['pol'])
    Po = np.array([ln['pol'] for ln in oct_lines if ln['fit']])
    s_pos, c0_oct = np.polyfit(Po, [ln['c'] for ln in oct_lines if ln['fit']], 1)
    plat = next(ln for ln in jul_lines if ln['kind'] == 'plateau')
    c0_jul = plat['c'] - s_pos * plat['pol']
    neg = [ln for ln in jul_lines if ln['fit'] and ln['pol'] < 0]
    Pn = np.array([ln['pol'] for ln in neg])
    s_neg = np.sum((np.array([ln['c'] for ln in neg]) - c0_jul) * Pn) / np.sum(Pn ** 2)
    shift = lambda p: s_pos * max(p, 0) + s_neg * min(p, 0)
    for ln in oct_lines:
        ln['c0'] = c0_oct
    for ln in jul_lines:
        ln['c0'] = c0_jul

    fit = [ln for ln in oct_lines + jul_lines if ln['fit']]
    P = np.array([ln['pol'] for ln in fit])
    xg = h * np.arange(-GRID_HALF, GRID_HALF + 1)
    G = []
    for ln in fit:
        x = 1e3 * f - ln['c0']
        s = ln['y'] / ln['pol']
        s = np.where((x < LINE_REGION_KHZ[0]) | (x > LINE_REGION_KHZ[1]), 0, s)
        g = np.interp(xg + shift(ln['pol']), x, s, left=0, right=0)
        G.append(g / (np.sum(g) * h))
    G = np.array(G)
    X = np.stack([np.ones_like(P), np.maximum(P, 0), np.minimum(P, 0)], 1)
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

    def model(p, x):
        u = x - shift(p)
        return (np.interp(u, xg, A0, left=0, right=0) + max(p, 0) * np.interp(u, xg, Ap, left=0, right=0)
                + min(p, 0) * np.interp(u, xg, An, left=0, right=0))

    for ln, g in zip(fit, G):
        m = model(ln['pol'], xg + shift(ln['pol'])) * pk
        ln['err'] = float(np.max(abs(m - g)) / g.max())
    return dict(c0_oct=float(c0_oct), c0_jul=float(c0_jul), s_pos=float(s_pos), s_neg=float(s_neg),
                step=float(h), A0=A0, Ap=Ap, An=An, pol_min=float(P.min()), pol_max=float(P.max()),
                max_err=max(ln['err'] for ln in fit), n_fit=len(fit), n_oct=sum(ln['fit'] for ln in oct_lines),
                n_jul=sum(ln['fit'] for ln in jul_lines))


MODULE = '''"""Proton NMR absorption lineshape, measured and polarization dependent, with its exact Kramers-Kronig dispersion.

Generated by web/tools/make_proton_data.py; edit that script, not this file.

The shape comes from proton event logs of a 5 T polarized target (212.9 MHz, 2022): {n_fit} fit-subtracted lines,
{n_oct} from an October run at P = {omin:.2f} to {omax:.2f} and {n_jul} from a July run that reversed the
polarization, at +{jplat:.2f} and from {nmax:.2f} to {nmin:.2f}. As |P| grows the line moves away from its P = 0
center and sharpens on that side: up and on its high-frequency side for P > 0, down and on its low-frequency side
for P < 0. The negative line is not quite the mirror image of the positive one. It moves about half as far and
stays broader, so each sign has its own shift and its own change of shape:

    g(x; P) = A0(x - s(P)) + P+ APOS(x - s(P)) + P- ANEG(x - s(P)),
    P+ = max(P, 0),  P- = min(P, 0),  s(P) = {s_pos:.2f} kHz * P+ + {s_neg:.2f} kHz * P-,

with x the offset from the Larmor frequency f0 in kHz. A0, APOS and ANEG come from a least squares fit to the
lines, which reproduces each within {max_err:.0f}% of its peak. A0, the line at P = 0, is shared by both signs.
There is no settled line between {nmax:.2f} and +{omin:.2f} (the reversal passes through too quickly), so there the
shape is interpolated. Beyond the measured range, P < {pmin:.2f} or P > {pmax:.2f}, the shape is held at its
value at the end of the range and only the shift and the size follow P. The integrals of APOS and ANEG are
zero, so the area does not depend on P. Faint tails are smoothed and both ends tapered to zero; samples every
{step:.4f} kHz.

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


def write_module(oct_lines, jul_lines, model, path):
    fmt = lambda a: textwrap.fill(', '.join(f'{v:.5f}' for v in a), 112, initial_indent='    ',
                                  subsequent_indent='    ')
    Po = [ln['pol'] for ln in oct_lines if ln['fit']]
    Pn = [ln['pol'] for ln in jul_lines if ln['fit'] and ln['pol'] < 0]
    jplat = next(ln['pol'] for ln in jul_lines if ln['kind'] == 'plateau')
    src = MODULE.format(n_fit=model['n_fit'], n_oct=model['n_oct'], n_jul=model['n_jul'], omin=min(Po),
                        omax=max(Po), jplat=jplat, nmin=min(Pn), nmax=max(Pn), s_pos=model['s_pos'],
                        s_neg=model['s_neg'], max_err=np.ceil(100 * model['max_err']), pmin=model['pol_min'],
                        pmax=model['pol_max'], step=model['step'], half=GRID_HALF, half_p1=GRID_HALF + 1,
                        A0=fmt(model['A0']), Ap=fmt(model['Ap']), An=fmt(model['An']))
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(src)


def iso(s):
    return datetime.fromtimestamp(s, timezone.utc).strftime('%Y-%m-%d %H:%M UTC')


def write_page_data(runs, lines, model, f, path):
    out_runs = []
    for run in runs:
        c0 = model['c0_' + run['id']]
        x = 1e3 * f - c0
        m = (x >= PAGE_REGION_KHZ[0]) & (x <= PAGE_REGION_KHZ[1])
        run['mask'] = m
        t, pol = run['t'], run['pol']
        keep = t >= run['t_start']
        thin = np.zeros_like(keep)
        thin[::TIMELINE_EVERY] = True
        if run.get('t_rev'):
            thin |= (t > run['t_rev'] - 1800) & (t < run['t_settled'])
        keep &= thin
        r = dict(id=run['id'], start=iso(run['t_start']), center_mhz=round(c0 / 1e3, 5),
                 x_khz=[round(v, 3) for v in x[m]],
                 timeline=dict(hours=[round((s - run['t_start']) / 3600, 4) for s in t[keep]],
                               pol=[round(float(p), 4) for p in pol[keep]]))
        if run.get('t_rev'):
            r.update(reversal_hours=round((run['t_rev'] - run['t_start']) / 3600, 4),
                     zero_hours=round((run['t_zero'] - run['t_start']) / 3600, 4),
                     settled_hours=round((run['t_settled'] - run['t_start']) / 3600, 4))
        out_runs.append(r)
    by_id = {run['id']: run for run in runs}
    data = dict(
        source='Proton event logs of a 5 T polarized target, July and October 2022 (512-point sweeps of '
               '212.482-213.282 MHz)',
        step_khz=round(model['step'], 5),
        runs=out_runs,
        lines=[dict(run=ln['run'], time=iso(ln['t']), pol=round(float(ln['pol']), 4), n=int(ln['n']),
                    kind=ln['kind'], fit=ln['fit'],
                    hours=round((ln['t'] - by_id[ln['run']]['t_start']) / 3600, 4),
                    err=round(ln['err'], 4) if 'err' in ln else None,
                    y=[round(float(v), 6) for v in ln['y'][by_id[ln['run']]['mask']]]) for ln in lines],
        model=dict(shift_pos=round(model['s_pos'], 4), shift_neg=round(model['s_neg'], 4),
                   pol_min=round(model['pol_min'], 4), pol_max=round(model['pol_max'], 4),
                   step_khz=round(model['step'], 6), half=GRID_HALF,
                   A0=[round(float(v), 5) for v in model['A0']], APOS=[round(float(v), 5) for v in model['Ap']],
                   ANEG=[round(float(v), 5) for v in model['An']], max_err=round(model['max_err'], 4),
                   settle_h=SETTLE_H),
    )
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(data, fh, separators=(',', ':'))


def main():
    directory = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, 'temp')
    t, pol, y, f = load_events(directory)
    runs = split_runs(t, pol, y)
    oct_lines, jul_lines = [], []
    for run in runs:
        if run['pol'].min() < -0.2:
            jul_lines = july_lines(run)
        else:
            oct_lines = october_lines(run)
    if not oct_lines or not jul_lines:
        sys.exit("expected one run with positive polarization only and one that reverses it")
    runs = [r for r in runs if 'id' in r]
    runs.sort(key=lambda r: r['id'] != 'oct')
    for ln in oct_lines:
        ln['run'] = 'oct'
    for ln in jul_lines:
        ln['run'] = 'jul'
    model = fit_model(oct_lines, jul_lines, f)
    write_module(oct_lines, jul_lines, model, os.path.join(WEB, 'proton_lineshape.py'))
    lines = oct_lines + jul_lines
    write_page_data(runs, lines, model, f, os.path.join(WEB, 'proton_data.json'))
    print(f"{len(t)} events in {len(runs)} runs, {len(lines)} example lines ({model['n_fit']} fitted); shift "
          f"{model['s_pos']:.2f} / {model['s_neg']:.2f} kHz per unit P; P = 0 centers {model['c0_oct'] / 1e3:.5f} "
          f"(Oct), {model['c0_jul'] / 1e3:.5f} MHz (Jul); worst fit {100 * model['max_err']:.1f}% of peak")
    for ln in lines:
        if ln['fit']:
            print(f"  {ln['run']} {ln['kind']:9s} P = {ln['pol']:+.3f}  n = {ln['n']:4d}  error {100 * ln['err']:.1f}%")


if __name__ == '__main__':
    main()
