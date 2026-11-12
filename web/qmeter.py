"""QMeter circuit behavior simulation (vectorized core).

J. Maxwell

Series-tank Q-meter model, based initially on MathCAD code by M. Houlden, G. Court, expanded to include
4 transmission lines (supply, signal, reference, coil "lambda/2") and a test polarization signal.

    G.R. Court, M.A. Houlden, et. al. "High precision measurement of the polarization in solid state polarized
    targets using NMR," NIM A 527 (2004) 253-263. https://doi.org/10.1016/j.nima.2004.02.041

This is the same circuit as `update()` in qmeter_sim.ipynb, rewritten with numpy arrays instead of per-point
lambdas so that it is fast enough to run live in the browser (Pyodide). The web app imports this file directly.
"""

import json
import numpy as np

import deuteron_lineshape

GYROMAGNETIC = {'p': 42.576, 'd': 6.53566}  # MHz/T

DEFAULTS = dict(
    # interactive controls (notebook widgets)
    trim_sup=1e-9,      # supply cable length, units of lambda/2
    trim_sig=1e-9,      # signal cable length, units of lambda/2
    trim_ref=1e-9,      # reference cable length, units of lambda/2
    trim_coil=1e-9,     # coil (lambda/2) cable length, units of lambda/2
    capacitance=1.0,    # tuning capacitance, units of the bare-tank tune capacitance
    phase=0.0,          # phase between signal and reference, degrees
    sweep_range=0.4,    # half-width of frequency sweep, MHz
    # notebook "Controls" constants
    sig=True,           # include simulated polarization signal
    species='p',        # 'p' protons, 'd' deuterons
    pol=0.0027,         # polarization (fraction), physical sign: > 0 is absorptive. The notebook's `pol`
                        # is the negative of this (its default -0.0027 is +0.27%, about TE at 5 T, 1.9 K).
    b=5.00,             # holding field, T
    # deuteron lineshape (Dulya et al.; see deuteron_lineshape.py)
    d_A=0.05,           # dipolar broadening width, units of the quadrupole frequency
    d_wq=0.027,         # quadrupole frequency wQ, MHz (doublet horns at about +-3 wQ)
    d_eta=0.0,          # electric field gradient asymmetry
    # circuit constants
    u=1.0,              # input RF voltage
    r_cc=681.0,         # constant current resistor, ohm
    c_stray=0.001e-12,  # stray capacitance, F
    l_0=0.175e-6,       # coil inductance, H
    r_coil=0.3,         # coil resistance, ohm
    r_amp=50.0,         # detector impedance, ohm
    r_source=50.0,      # source output impedance, ohm
    r=2.0,              # damping resistor, ohm
    v_f=0.78,           # cable velocity factor
    alph=0.0343,        # cable attenuation constant
    l_cab=2.54e-7,      # cable inductance
    c_per_m=95.2e-12,   # cable capacitance per meter, F/m
    eta=0.0013,         # filling factor
    n_steps=500,        # points in sweep
)

MIN_TRIM = 1e-9  # zero-length cables are singular in this model (z0 -> inf); clamp like the notebook defaults


def _params(kwargs):
    p = dict(DEFAULTS)
    unknown = set(kwargs) - set(p)
    if unknown:
        raise TypeError(f"unknown parameters: {sorted(unknown)}")
    p.update(kwargs)
    p['species'] = str(p['species']).lower()[:1]
    if p['species'] not in GYROMAGNETIC:
        raise ValueError("species must be 'p' or 'd'")
    return p


def larmor_mhz(p):
    return p['b'] * GYROMAGNETIC[p['species']]


CHI0_PER_POL_P = 3.1e-4   # proton static susceptibility per unit polarization (Seely 1998)
# Deuterons at the same molecular density: magnetization per spin is gamma*hbar*<Iz> = gamma_d*hbar*P (I=1)
# versus gamma_p*hbar*P/2 (I=1/2), so chi0 scales by 2 gamma_d / gamma_p.
CHI0_PER_POL_D = CHI0_PER_POL_P * 2 * GYROMAGNETIC['d'] / GYROMAGNETIC['p']


def susceptibility(p, w):
    """Complex susceptibility chi = chi' - i chi'' of the sample at angular frequencies w."""
    w0 = 2 * np.pi * larmor_mhz(p) * 1e6
    if p['species'] == 'p':
        # Hill & Hill lineshape with Seely's chi0, exactly as in the notebook (whose pol is -p['pol']).
        # For w < w0 the power of a negative number is the principal complex value, as in the notebook.
        chi_0 = CHI0_PER_POL_P * -p['pol']
        del_w = 6e-2 * 2 * np.pi * 1e6    # width between derivative extrema
        x = ((w - w0) / del_w).astype(complex)
        chi_pp = 1.035 * chi_0 * w0 / del_w * (1 / (1 + 4 * np.power(x, 2.35)))
        return 1j * chi_pp
    return deuteron_lineshape.susceptibility(w / (2 * np.pi * 1e6), p['pol'], w0 / (2 * np.pi * 1e6),
                                             CHI0_PER_POL_D, wQ=p['d_wq'], A=p['d_A'], eta=p['d_eta'])


def _circuit(p, w):
    """Evaluate the Q-meter at angular frequencies w (array). Returns complex voltages and lengths."""
    u, r_cc, r_amp, r_source, r = p['u'], p['r_cc'], p['r_amp'], p['r_source'], p['r']
    l_0, r_coil, c_stray = p['l_0'], p['r_coil'], p['c_stray']

    f0 = larmor_mhz(p)                    # Larmor freq in MHz
    w0 = 2 * np.pi * f0 * 1e6             # angular freq
    c = p['capacitance'] / (w0 ** 2 * l_0)  # tank capacitance

    vel = p['v_f'] * 2.99e8               # velocity in cable
    lb2 = vel / (2 * f0 * 1e6)            # lambda/2 cable length in m
    trims = {k: max(float(p[k]), MIN_TRIM) for k in ('trim_sup', 'trim_sig', 'trim_ref', 'trim_coil')}
    len_sup = trims['trim_sup'] * lb2
    len_sig = trims['trim_sig'] * lb2
    len_ref = trims['trim_ref'] * lb2
    len_coil = trims['trim_coil'] * lb2

    alph = p['alph']
    gam = alph + 1j * w / vel             # loss term as signal travels down cable
    r_cab = 2 * 50 * alph                 # cable resistance
    l_cab = p['l_cab']
    c_sup = p['c_per_m'] * len_sup
    c_sig = p['c_per_m'] * len_sig
    c_ref = p['c_per_m'] * len_ref
    c_coil = p['c_per_m'] * len_coil

    # simulated signal: L = L0 (1 + 4 pi eta chi), chi = chi' - i chi''
    if p['sig']:
        chi = susceptibility(p, w)
        l_coil = l_0 * (1 + 4 * np.pi * p['eta'] * chi)
    else:
        l_coil = l_0

    def z0_calc(c_cab):
        """Characteristic impedance of transmission line."""
        return np.sqrt((r_cab + 1j * w * l_cab) / (1j * w * c_cab))

    def trans(z0, zl, length):
        """Transmission line input impedance for characteristic impedance z0 and load zl."""
        t = np.tanh(gam * length)
        return z0 * ((zl + z0 * t) / (z0 + zl * t))

    z0_sup, z0_sig, z0_ref, z0_coil = z0_calc(c_sup), z0_calc(c_sig), z0_calc(c_ref), z0_calc(c_coil)

    zc = 1 / (1j * w * c)                                 # tuning capacitor
    zc_stray = 1 / (1j * w * c_stray)                     # stray capacitance
    zl_pure = r_coil + 1j * w * l_coil                    # coil only
    z_l = zl_pure * zc_stray / (zl_pure + zc_stray)       # coil and stray capacitance
    z_coil = trans(z0_coil, z_l, len_coil)                # coil through its transmission line
    z_tank = r + zc + z_coil                              # tank leg
    z_sup = trans(z0_sup, r_source, len_sup)              # supply line and source
    z_cc = r_cc + 1 / (1 / 50 + 1 / z_sup)                # supply, 50 ohm to ground, constant current resistor
    z_eq = 1 / (1 / z_tank + 1 / z_cc)                    # parallel legs
    z_tot = trans(z0_sig, z_eq, len_sig)                  # parallel legs through signal line

    z_ref = 1 / (1 / trans(z0_ref, r_source, len_ref) + 1 / 50)
    v_ref = u * (r_amp / (r_amp + z_ref))                 # voltage into mixer reference
    v_eq = u * (z_tank / (z_tank + z_cc))                 # voltage into signal line
    v_rf = v_eq * (r_amp / (r_amp + z_tot))               # voltage into mixer rf

    rot = np.exp(1j * p['phase'] * np.pi / 180)
    return dict(
        v_out=v_rf * rot,                                 # voltage into diode
        v_mixer=v_rf * v_ref * rot,                       # output of ideal mixer
        f0_mhz=f0, w0=w0, c_pf=c * 1e12, lambda2_m=lb2,
        len_sup_m=len_sup, len_sig_m=len_sig, len_ref_m=len_ref, len_coil_m=len_coil,
    )


def simulate(**kwargs):
    """Run the Q-meter model over the frequency sweep. Returns a dict of numpy arrays and derived scalars."""
    p = _params(kwargs)
    n_steps = int(p['n_steps'])
    f0 = larmor_mhz(p)
    wlo = 2 * np.pi * (f0 - p['sweep_range']) * 1e6
    whi = 2 * np.pi * (f0 + p['sweep_range']) * 1e6
    dw = (whi - wlo) / n_steps
    w = wlo + np.arange(n_steps) * dw     # sweep frequency (n_steps points, upper end excluded, as in notebook)

    c = _circuit(p, w)
    return dict(
        f_mhz=w / (2 * np.pi) / 1e6,
        diode=np.abs(c['v_out']),
        phase=-np.real(c['v_mixer']),
        **{k: c[k] for k in ('f0_mhz', 'c_pf', 'lambda2_m', 'len_sup_m', 'len_sig_m', 'len_ref_m', 'len_coil_m')},
    )


# ---------------------------------------------------------------------------------------------------------
# Tuning
#
# Diode tune: choose the capacitance so the baseline diode dip is centered on the Larmor frequency,
#     d|V_rf|/dw = 0 at w0, with |V_rf| a minimum there.
# Phase tune: diode-tune the capacitance, then choose the phase so the baseline phase output is symmetric
#     (an extremum) at w0. With Z = V_rf V_ref, the output is -Re(Z e^{i phi}), so its slope -Re(Z' e^{i phi})
#     vanishes for phi = +-pi/2 - arg Z'(w0). Of the two roots, take the one whose curvature matches the
#     ideal zero-length circuit at phase 0, so a tuned curve always has the same orientation.
# Tuning is done on the baseline (polarization signal off) and leaves cable lengths untouched.
# ---------------------------------------------------------------------------------------------------------

C_MIN, C_MAX = 0.02, 50.0   # capacitance search range, units of the bare-tank tune capacitance
_REL_H = 2e-5               # finite-difference step, relative to w0


def _diode_slope(p, w0, h):
    """d|V_rf|/dw at w0 for each capacitance in p['capacitance'] (an array), plus |V_rf(w0)|."""
    caps = np.atleast_1d(p['capacitance'])
    w = np.array([w0 - h, w0, w0 + h])
    v = np.abs(np.stack([_circuit({**p, 'capacitance': cc}, w)['v_out'] for cc in caps]))
    return (v[:, 2] - v[:, 0]) / (2 * h), v[:, 1]


def _tune_capacitance(p):
    p = {**p, 'sig': False, 'phase': 0.0}
    w0 = 2 * np.pi * larmor_mhz(p) * 1e6
    h = _REL_H * w0

    # Coarse log scan. As C increases the dip moves down in frequency, so the slope at w0 goes from
    # - (dip above w0) to + (dip below w0) at the centered capacitance. Pick the crossing with the deepest |V(w0)|.
    grid = np.geomspace(C_MIN, C_MAX, 800)
    slope, mag = _diode_slope({**p, 'capacitance': grid}, w0, h)
    idx = np.nonzero((slope[:-1] < 0) & (slope[1:] >= 0))[0]
    if idx.size == 0:
        raise ValueError(f"no capacitance from {C_MIN:g} to {C_MAX:g} × tune puts the diode dip on the Larmor "
                         "frequency. At this coil cable length the coil probably looks capacitive "
                         "(e.g. near λ/4 = 0.5 λ/2), so the series tank cannot resonate.")
    i = idx[np.argmin(np.minimum(mag[idx], mag[idx + 1]))]

    # bisection on the bracketing pair
    lo, hi = grid[i], grid[i + 1]
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        s, _ = _diode_slope({**p, 'capacitance': mid}, w0, h)
        if s[0] < 0:
            lo = mid
        else:
            hi = mid
        if hi / lo - 1 < 1e-12:
            break
    return float(np.sqrt(lo * hi))


def _mixer_derivs(p, w0, h):
    w = np.array([w0 - h, w0, w0 + h])
    z = _circuit({**p, 'phase': 0.0, 'sig': False}, w)['v_mixer']
    return (z[2] - z[0]) / (2 * h), (z[2] - 2 * z[1] + z[0]) / h ** 2


def _ideal_curvature_sign(p):
    """Sign of d2(phase output)/dw2 at w0 for the zero-length circuit at capacitance 1, phase 0."""
    ideal = {**p, 'capacitance': 1.0, 'phase': 0.0, 'sig': False,
             'trim_sup': 0.0, 'trim_sig': 0.0, 'trim_ref': 0.0, 'trim_coil': 0.0}
    w0 = 2 * np.pi * larmor_mhz(p) * 1e6
    _, d2 = _mixer_derivs(ideal, w0, _REL_H * w0)
    return np.sign(-np.real(d2)) or 1.0


def tune(mode='phase', **kwargs):
    """Return {'capacitance', 'phase'} that center the baseline signal on the Larmor frequency.

    mode 'diode' sets the capacitance only (phase does not affect the diode output);
    mode 'phase' sets the capacitance and then the phase.
    """
    p = _params(kwargs)
    cap = _tune_capacitance(p)
    out = dict(capacitance=cap, phase=float(p['phase']))
    if mode == 'diode':
        return out
    if mode != 'phase':
        raise ValueError("mode must be 'diode' or 'phase'")

    p = {**p, 'capacitance': cap}
    w0 = 2 * np.pi * larmor_mhz(p) * 1e6
    d1, d2 = _mixer_derivs(p, w0, _REL_H * w0)
    want = _ideal_curvature_sign(p)
    best = None
    for phi in (np.pi / 2 - np.angle(d1), -np.pi / 2 - np.angle(d1)):
        curv = -np.real(d2 * np.exp(1j * phi))
        if np.sign(curv) == want:
            best = phi
    if best is None:  # degenerate curvature; fall back to the first root
        best = np.pi / 2 - np.angle(d1)
    deg = (np.degrees(best) + 180.0) % 360.0 - 180.0
    out['phase'] = float(deg)
    return out


def simulate_json(params_json):
    """JSON params in, JSON curves out. (The web worker uses analysis.run_json, which includes this.)"""
    out = simulate(**json.loads(params_json))
    return json.dumps({k: (v.tolist() if isinstance(v, np.ndarray) else float(v)) for k, v in out.items()})


def tune_json(mode, params_json):
    """Entry point for the web worker's tune buttons."""
    return json.dumps(tune(mode, **json.loads(params_json)))
