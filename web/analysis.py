"""Analysis of simulated Q-meter sweeps, following jdmax/NMR_Analysis and jdmax/rgc_offline.

For each sweep of the phase-sensitive output:
    1. subtract a baseline sweep taken with no NMR signal in the window,
    2. fit a polynomial to the "wings" (regions away from the line) and subtract it,
    3. sum the points within the integration range to get the area.
A thermal-equilibrium (TE) sweep, whose polarization is known from the temperature and field, calibrates
the area: CC = P_TE / A_TE, and an enhanced signal's polarization is P = CC * A.
For deuterons the polarization also follows from the lineshape itself; here, from the ratio of the two
peak heights of the Pake doublet, r = h(+) / h(-), and P = (r^2 - 1)/(r^2 + r + 1).
"""

import json
import numpy as np

import qmeter
import deuteron_lineshape

PLANCK = 6.62607015e-34   # J s
BOLTZMANN = 1.380649e-23  # J/K

ANALYSIS_DEFAULTS = dict(
    te_temp=1.0,                     # TE temperature, K
    wings=[0.01, 0.25, 0.75, 0.99],  # polynomial fit regions, fractions of the sweep (as in rgc_offline)
    sum_range=[0.25, 0.75],          # integration range, fractions of the sweep
    order=3,                         # polynomial order (NMR_Analysis and rgc_offline use a cubic)
    drift_ppm=0.0,                   # change in tuning capacitance between baseline and signal sweeps, ppm
)


def te_polarization(species, f0_mhz, temp_k):
    """Thermal-equilibrium polarization: tanh(x) for spin 1/2, 4 tanh(x)/(3 + tanh^2 x) for spin 1,
    with x = h f0 / (2 k T)."""
    x = PLANCK * f0_mhz * 1e6 / (2 * BOLTZMANN * temp_k)
    t = np.tanh(x)
    return float(t if species == 'p' else 4 * t / (3 + t * t))


def analyze_sweep(f, raw, baseline, wings, sum_range, order):
    """Baseline subtraction, polynomial fit over the wings, fit subtraction and area, as in rgc_offline."""
    n = len(f)
    idx = np.arange(n)
    basesub = raw - baseline
    b = [x * n for x in wings]
    fit_mask = ((idx > b[0]) & (idx < b[1])) | ((idx > b[2]) & (idx < b[3]))
    if fit_mask.sum() <= order:
        raise ValueError("not enough points in the wings for this polynomial order")
    x = (f - f.mean()) / (f.max() - f.min())       # centered and scaled for a well-conditioned fit
    coef = np.polyfit(x[fit_mask], basesub[fit_mask], int(order))
    fit = np.polyval(coef, x)
    sub = basesub - fit
    s = [x * n for x in sum_range]
    sum_mask = (idx > s[0]) & (idx < s[1])
    return dict(basesub=basesub, fit=fit, sub=sub, area=float(sub[sum_mask].sum()))


def peak_ratio(f, sub, area, f0):
    """Deuteron polarization from the two peak heights of the fit-subtracted doublet.

    The m=+1 transition's peak is above the Larmor frequency and the m=-1 peak below; their heights are
    in the ratio of the populations, r = I+/I-, to first order. Sign is taken from the area so the
    estimate does not depend on the Q-meter's output polarity.
    """
    s = sub * np.sign(area or 1.0)
    win = 0.5  # MHz either side
    hi = (f > f0) & (f < f0 + win)
    lo = (f < f0) & (f > f0 - win)
    i_hi = np.nonzero(hi)[0][np.argmax(s[hi])]
    i_lo = np.nonzero(lo)[0][np.argmax(s[lo])]
    h_hi, h_lo = float(s[i_hi]), float(s[i_lo])
    if h_lo <= 0 or h_hi <= 0:
        return None
    r = h_hi / h_lo
    return dict(r=r, pol=float(deuteron_lineshape.pol_from_r(r)), f_hi=float(f[i_hi]), f_lo=float(f[i_lo]),
                h_hi=h_hi * np.sign(area or 1.0), h_lo=h_lo * np.sign(area or 1.0))


def _nelder_mead(fun, x0, step, max_iter=600, tol=1e-10):
    """Minimal Nelder-Mead minimizer (numpy only, so it runs in Pyodide without scipy)."""
    n = len(x0)
    pts = [np.asarray(x0, float)]
    for i in range(n):
        x = np.array(x0, float)
        x[i] += step[i]
        pts.append(x)
    vals = [fun(x) for x in pts]
    for _ in range(max_iter):
        order = np.argsort(vals)
        pts = [pts[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) <= tol * (abs(vals[0]) + 1e-30):
            break
        c = np.mean(pts[:-1], axis=0)
        xr = c + (c - pts[-1])
        fr = fun(xr)
        if fr < vals[0]:
            xe = c + 2 * (c - pts[-1])
            fe = fun(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * (pts[-1] - c)
            fc = fun(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0]] + [pts[0] + 0.5 * (x - pts[0]) for x in pts[1:]]
                vals = [vals[0]] + [fun(x) for x in pts[1:]]
    i = int(np.argmin(vals))
    return pts[i], vals[i]


def lineshape_fit(f, sub, f0, pol_start=0.0, A_start=0.05, wq_start=0.027, complex_fit=False):
    """Fit the Dulya lineshape (as in DeuteronPeakFit) to the fit-subtracted signal.

    Free parameters: polarization (through r), broadening A, quadrupole frequency wQ, false asymmetry xi,
    and the scale G (solved by linear least squares at each step). wL is fixed at the Larmor frequency and
    eta at 0. With complex_fit, a phase angle mixes in the dispersion (Kramers-Kronig partner of the
    absorption), like DeuteronPeakFit's fit_complex.
    """
    y = np.asarray(sub, float)
    R_of = lambda wq: (f - f0) / (3 * wq)

    def shape(v):
        pol = np.tanh(v[0])
        A, wq, xi = np.exp(v[1]), np.exp(v[2]), v[3]
        absorption, hilbert, _ = deuteron_lineshape.components(f, deuteron_lineshape.r_from_pol(pol), wq, f0, A, 0.0)
        line = absorption
        if complex_fit:
            line = np.cos(v[4]) * absorption + np.sin(v[4]) * hilbert
        return line * (1 + 0.5 * xi * (1 + R_of(wq)))

    def cost(v):
        s = shape(v)
        ss = np.dot(s, s)
        if not np.isfinite(ss) or ss == 0:
            return np.inf
        g = np.dot(s, y) / ss
        return float(np.sum((y - g * s) ** 2))

    pol0 = float(np.clip(pol_start, -0.95, 0.95))
    v0 = [np.arctanh(pol0), np.log(A_start), np.log(wq_start), 0.0]
    step = [0.2, 0.3, 0.1, 0.002]
    if complex_fit:
        v0.append(0.0)
        step.append(0.1)
    v, _ = _nelder_mead(cost, v0, step=step, max_iter=1200 if complex_fit else 600)
    s = shape(v)
    g = np.dot(s, y) / np.dot(s, s)
    curve = g * s
    pol = float(np.tanh(v[0]))
    return dict(pol=pol, r=float(deuteron_lineshape.r_from_pol(pol)), A=float(np.exp(v[1])),
                wq=float(np.exp(v[2])), xi=float(v[3]), G=float(g), curve=curve,
                phase_deg=float(np.degrees(v[4])) if complex_fit else None,
                rms=float(np.sqrt(np.mean((y - curve) ** 2))))


def analyze(params, settings=None):
    s = {**ANALYSIS_DEFAULTS, **(settings or {})}
    if not (0 <= int(s['order']) <= 8):
        raise ValueError("polynomial order must be between 0 and 8")
    w, sr = s['wings'], s['sum_range']
    if len(w) != 4 or not all(0 <= a < b <= 1 for a, b in zip(w, w[1:])):
        raise ValueError("wing bounds must increase from 0 to 1")
    if len(sr) != 2 or not 0 <= sr[0] < sr[1] <= 1:
        raise ValueError("integration range must increase from 0 to 1")
    if float(s['te_temp']) <= 0:
        raise ValueError("TE temperature must be positive")
    p = qmeter._params(params)
    f0 = qmeter.larmor_mhz(p)
    pol = float(p['pol'])
    p_te = te_polarization(p['species'], f0, float(s['te_temp']))

    # The baseline is a sweep with no signal; the signal sweeps may be taken with a slightly drifted tune.
    sig_p = {**params, 'sig': True, 'capacitance': p['capacitance'] * (1 + float(s['drift_ppm']) * 1e-6)}
    base = qmeter.simulate(**{**params, 'sig': False})
    f = base['f_mhz']
    raw_te = qmeter.simulate(**{**sig_p, 'pol': p_te})['phase']
    raw = qmeter.simulate(**{**sig_p, 'pol': pol})['phase']

    args = (s['wings'], s['sum_range'], s['order'])
    te = analyze_sweep(f, raw_te, base['phase'], *args)
    en = analyze_sweep(f, raw, base['phase'], *args)
    cc = p_te / te['area'] if te['area'] else float('nan')
    out = dict(
        f_mhz=f, f0_mhz=f0, species=p['species'],
        pol_true=pol, pol_te=p_te, area_te=te['area'], area=en['area'], cc=cc, pol_area=cc * en['area'],
        te={k: te[k] for k in ('basesub', 'fit', 'sub')},
        en={k: en[k] for k in ('basesub', 'fit', 'sub')},
        settings=s,
    )
    if p['species'] == 'd':
        ratio = peak_ratio(f, en['sub'], en['area'], f0)
        out['ratio'] = ratio
        start = ratio['pol'] if ratio else 0.0
        out['fit'] = lineshape_fit(f, en['sub'], f0, pol_start=start)
        out['fit_complex'] = lineshape_fit(f, en['sub'], f0, pol_start=out['fit']['pol'], complex_fit=True)
    return out


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def run_json(params_json, settings_json):
    """Web worker entry point: the sweep for the plots plus the analysis, in one call."""
    params = json.loads(params_json)
    sim = qmeter.simulate(**params)
    out = dict(sim=_jsonable(sim))
    try:
        out['analysis'] = _jsonable(analyze(params, json.loads(settings_json)))
    except (ValueError, np.linalg.LinAlgError) as e:
        out['analysis_error'] = str(e)
    return json.dumps(out)
