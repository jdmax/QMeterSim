"""QMeter circuit behavior simulation (vectorized core).

J. Maxwell

Series-tank Q-meter model, based initially on MathCAD code by M. Houlden, G. Court, expanded to include
4 transmission lines (supply, signal, reference, coil "lambda/2") and a test polarization signal.

    G.R. Court, M.A. Houlden, et. al. "High precision measurement of the polarization in solid state polarized
    targets using NMR," NIM A 527 (2004) 253-263. https://doi.org/10.1016/j.nima.2004.02.041

The circuit is that of `update()` in qmeter_sim.ipynb, vectorized with numpy so it runs live in the browser
(Pyodide), with the cables corrected: each line's characteristic impedance comes from the per-metre cable
constants, so it doesn't depend on its length (the notebook divides by the line's total capacitance), and
the source voltage is carried through the supply, signal and reference lines. With zero-length cables the
two agree up to an overall scale. The web app imports this file directly.
"""

import json
import numpy as np

import deuteron_lineshape

GYROMAGNETIC = {'p': 42.576, 'd': 6.53566}  # MHz/T

DEFAULTS = dict(
    # interactive controls (notebook widgets)
    trim_sup=0.0,       # supply cable length, units of lambda/2
    trim_sig=0.0,       # signal cable length, units of lambda/2
    trim_ref=0.0,       # reference cable length, units of lambda/2
    trim_coil=0.0,      # coil (lambda/2) cable length, units of lambda/2
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
    alph=0.0343,        # cable attenuation constant, 1/m
    l_cab=2.54e-7,      # cable inductance per meter, H/m
    c_per_m=95.2e-12,   # cable capacitance per meter, F/m
    eta=0.0013,         # filling factor
    n_steps=500,        # points in sweep
)

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
    """Evaluate the Q-meter at angular frequencies w (array). Returns complex voltages and lengths.

    Each cable is a uniform lossy line with per-metre resistance, inductance and capacitance, so its
    characteristic impedance Z0 (about 51.7 ohm) and propagation constant gamma do not depend on its length.
    Each stage is reduced to a Thevenin source (open-circuit voltage, source impedance) and carried through
    the next line: a line of length l with input source (V, Zs) gives, at its far end,
        open-circuit voltage  V / (cosh(gamma l) + (Zs/Z0) sinh(gamma l)),
        source impedance      T[Zs] = Z0 (Zs + Z0 tanh(gamma l)) / (Z0 + Zs tanh(gamma l)).
    A line terminated in Z0 is matched: it delays and attenuates the voltage but doesn't change the curve's
    shape, which is why the supply and signal lengths aren't critical (Court et al. 2004, sec. 5.6).
    """
    u, r_cc, r_amp, r_source, r = p['u'], p['r_cc'], p['r_amp'], p['r_source'], p['r']
    l_0, r_coil, c_stray = p['l_0'], p['r_coil'], p['c_stray']

    f0 = larmor_mhz(p)                    # Larmor freq in MHz
    w0 = 2 * np.pi * f0 * 1e6             # angular freq
    c = p['capacitance'] / (w0 ** 2 * l_0)  # tank capacitance

    vel = p['v_f'] * 2.99e8               # velocity in cable
    lb2 = vel / (2 * f0 * 1e6)            # lambda/2 cable length in m
    trims = {k: max(float(p[k]), 0.0) for k in ('trim_sup', 'trim_sig', 'trim_ref', 'trim_coil')}
    len_sup = trims['trim_sup'] * lb2
    len_sig = trims['trim_sig'] * lb2
    len_ref = trims['trim_ref'] * lb2
    len_coil = trims['trim_coil'] * lb2

    alph = p['alph']
    gam = alph + 1j * w / vel             # propagation constant, per metre
    r_cab = 2 * 50 * alph                 # cable resistance per metre (alpha = R'/2Z0)
    z0 = np.sqrt((r_cab + 1j * w * p['l_cab']) / (1j * w * p['c_per_m']))   # characteristic impedance

    def trans(zs, length):
        """Impedance at one end of a line whose other end is terminated by zs."""
        t = np.tanh(gam * length)
        return z0 * ((zs + z0 * t) / (z0 + zs * t))

    def carry(v, zs, length):
        """Open-circuit voltage at the far end of a line driven by a source v with impedance zs."""
        return v / (np.cosh(gam * length) + (zs / z0) * np.sinh(gam * length))

    # simulated signal: L = L0 (1 + 4 pi eta chi), chi = chi' - i chi''
    if p['sig']:
        chi = susceptibility(p, w)
        l_coil = l_0 * (1 + 4 * np.pi * p['eta'] * chi)
    else:
        l_coil = l_0

    zc = 1 / (1j * w * c)                                 # tuning capacitor
    zc_stray = 1 / (1j * w * c_stray)                     # stray capacitance
    zl_pure = r_coil + 1j * w * l_coil                    # coil only
    z_l = zl_pure * zc_stray / (zl_pure + zc_stray)       # coil and stray capacitance
    z_coil = trans(z_l, len_coil)                         # coil through its transmission line
    z_tank = r + zc + z_coil                              # tank leg

    # supply: source (u, r_source) -> supply line -> 50 ohm to ground -> Rcc
    z_sup = trans(r_source, len_sup)                      # looking back up the supply line
    v_sup = carry(u, r_source, len_sup) * 50 / (50 + z_sup)   # Thevenin voltage across the 50 ohm
    z_cc = r_cc + 50 * z_sup / (50 + z_sup)               # Thevenin impedance through Rcc
    # tank node
    v_eq = v_sup * z_tank / (z_tank + z_cc)               # open-circuit voltage at the tank
    z_eq = z_tank * z_cc / (z_tank + z_cc)                # parallel legs
    # signal line -> mixer RF input (r_amp)
    z_tot = trans(z_eq, len_sig)                          # source impedance seen by the mixer
    v_rf = carry(v_eq, z_eq, len_sig) * r_amp / (r_amp + z_tot)   # voltage into mixer rf
    # reference line -> mixer LO input (r_amp)
    z_ref = trans(r_source, len_ref)
    v_ref = carry(u, r_source, len_ref) * r_amp / (r_amp + z_ref)  # voltage into mixer reference

    rot = np.exp(1j * p['phase'] * np.pi / 180)
    return dict(
        v_out=v_rf * rot,                                 # voltage into diode
        v_mixer=v_rf * np.conj(v_ref) * rot,              # ideal mixer: responds to the phase difference
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
#     vanishes for phi = +-pi/2 - arg Z'(w0). Of the two roots, take the one for which an absorptive signal
#     (extra coil resistance) has the same sign as in the ideal zero-length circuit at phase 0, so the NMR
#     signal always points the same way. (The baseline's curvature is not a safe guide: when the reference
#     cable's delay doesn't match the supply and signal cables' delay, the root that keeps the baseline a peak
#     inverts the signal.)
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


def _absorption_response(p, w0):
    """Change in the mixer output (phase 0) at w0 for a small absorptive signal: extra coil resistance, which
    is what chi'' adds, since i w L0 (1 - i 4 pi eta chi'') = i w L0 + w L0 4 pi eta chi''."""
    q = {**p, 'phase': 0.0, 'sig': False}
    dr = 1e-4 * (abs(q['r_coil']) + abs(q['r'])) + 1e-9
    z0 = _circuit(q, np.array([w0]))['v_mixer'][0]
    z1 = _circuit({**q, 'r_coil': q['r_coil'] + dr}, np.array([w0]))['v_mixer'][0]
    return z1 - z0


def _ideal_signal_sign(p):
    """Sign of the phase output's response to absorption for the zero-length circuit at capacitance 1, phase 0."""
    ideal = {**p, 'capacitance': 1.0, 'trim_sup': 0.0, 'trim_sig': 0.0, 'trim_ref': 0.0, 'trim_coil': 0.0}
    return np.sign(-np.real(_absorption_response(ideal, 2 * np.pi * larmor_mhz(p) * 1e6))) or 1.0


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
    d1, _ = _mixer_derivs(p, w0, _REL_H * w0)
    want = _ideal_signal_sign(p)
    resp = _absorption_response(p, w0)
    best = None
    for phi in (np.pi / 2 - np.angle(d1), -np.pi / 2 - np.angle(d1)):
        if np.sign(-np.real(resp * np.exp(1j * phi))) == want:
            best = phi
    if best is None:  # degenerate; fall back to the first root
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
