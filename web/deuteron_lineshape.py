"""Deuteron (spin-1) NMR lineshape: absorption and dispersion of a Pake doublet.

Ported from jdmax/DeuteronPeakFit (deuteron_fit.py), MIT license, Copyright (c) 2024 James Maxwell.
That code is a Python translation by J. Maxwell (2021) of C code by C. Dulya:

    C. Dulya et al., "A line-shape analysis for spin-1 NMR signals," NIM A 398 (1997) 109-125.
    Yu.F. Kisselev, C.M. Dulya, T.O. Niinikoski, "Measurement of complex RF susceptibility using a series
        Q-meter," NIM A 354 (1995) 249-261.
    M. McClellan, "Complex deuteron NMR signals," Eur. Phys. J. A 61, 176 (2025).

The private math (_iplus, _iminus, _f, _integrals) is unchanged apart from not modifying arrays in place.
The fitting wrappers (lmfit) are not needed here; instead `susceptibility` turns the lineshape into a
normalized complex susceptibility for the Q-meter model.
"""

import numpy as np


def r_from_pol(p):
    """Invert P = (r^2 - 1)/(r^2 + r + 1) for the m=+1/m=-1 population ratio r (|P| < 1)."""
    p = float(np.clip(p, -0.999, 0.999))
    return (p + np.sqrt(4 - 3 * p * p)) / (2 * (1 - p))


def pol_from_r(r):
    """Vector polarization from the population ratio r."""
    return (r * r - 1) / (r * r + r + 1)


def components(w, r, wQ, wL, A, eta):
    """Absorption of the doublet (G=1, xi=0) and its Hilbert transform.

    w, wQ, wL in the same units. Returns (absorption, hilbert, transition parts dict).

    Note on the dispersion: DeuteronPeakFit's _f(..., disp=True) for transition eps equals
    eps * (2/pi) * H[_f(..., disp=False)], where H is the Hilbert transform (checked numerically to
    machine precision in test_qmeter.py). Its _fit_func_complex sums the two transitions' dispersions
    without the factor eps, which is not the Kramers-Kronig partner of the summed absorption. Here each
    transition's dispersion is multiplied by eps * pi/2, so `hilbert` is H[absorption] (up to the slow
    R dependence of the population factors).
    """
    R = (w - wL) / (3 * wQ)
    Ip = _iplus(r, wQ / wL, R)
    Im = _iminus(r, wQ / wL, R)
    Fm_abs = _f(R, A, -1, eta) / wQ
    Fp_abs = _f(R, A, 1, eta) / wQ
    Hm = -(np.pi / 2) * _f(R, A, -1, eta, disp=True) / wQ
    Hp = +(np.pi / 2) * _f(R, A, 1, eta, disp=True) / wQ
    absorption = Im * Fm_abs + Ip * Fp_abs
    hilbert = Im * Hm + Ip * Hp
    return absorption, hilbert, dict(Fp=Fp_abs, Fm=Fm_abs, Ip=Ip, Im=Im)


_NORM_CACHE = {}


def _single_transition_area(wQ, A, eta):
    """Area (in the units of w) of one transition's absorption F/wQ, on a wide grid; cached."""
    key = (round(wQ, 12), round(A, 12), round(eta, 12))
    if key not in _NORM_CACHE:
        R = np.linspace(-60, 60, 120001)
        dR = R[1] - R[0]
        fp = _f(R, A, 1, eta) / wQ
        fm = _f(R, A, -1, eta) / wQ
        # integral over w of F/wQ = integral over R of F/wQ * 3 wQ
        _NORM_CACHE[key] = 0.5 * (fp.sum() + fm.sum()) * dR * 3 * wQ
    return _NORM_CACHE[key]


def susceptibility(f_mhz, pol, f0_mhz, chi0_per_pol, wQ=0.027, A=0.05, eta=0.0):
    """Complex susceptibility chi = chi' - i chi'' of a deuteron sample with vector polarization `pol`.

    chi'' is the Dulya absorption, scaled so that  integral chi'' dw = (pi/2) w0 chi0  with
    chi0 = chi0_per_pol * pol  (the same normalization as the proton Hill & Hill lineshape).
    chi' = -H[chi''] is its Kramers-Kronig partner for chi = chi' - i chi'' (e^{i w t} convention), so an
    absorptive line has chi' > 0 below resonance and < 0 above. pol > 0 is absorptive (chi'' > 0).
    f_mhz, f0_mhz, wQ in MHz.
    """
    r = r_from_pol(pol)
    absorption, hilbert, _ = components(np.asarray(f_mhz, float), r, wQ, f0_mhz, A, eta)
    # each transition's absorption has area `area1`; Ip + Im = P to first order in wQ/wL, so area = P * area1
    area1 = _single_transition_area(wQ, A, eta)
    scale = (np.pi / 2) * f0_mhz * chi0_per_pol / area1
    return -scale * hilbert - 1j * scale * absorption


# --- Private math primitives (from DeuteronPeakFit) ---------------------------------------------

def _iplus(r, Q, R):
    """Boltzmann population factor for the +1 quadrupole transition."""
    r3QR = r ** (-3 * Q * R)
    NN = r * (r + r3QR) + 1
    return r * (r - r3QR) / NN


def _iminus(r, Q, R):
    """Boltzmann population factor for the -1 quadrupole transition."""
    r3QR = r ** (3 * Q * R)
    NN = r * (r + r3QR) + 1
    return (r * r3QR - 1) / NN


def _f(R, A, eps, eta, disp=False):
    """Powder-averaged lineshape via Romberg quadrature over phi (see DeuteronPeakFit)."""
    if eta < 0.001:
        FF = _integrals(R, A, eps, 3, 0, disp)
    else:
        FF = 0
        dphi = 1

        for i in (0, 1):
            ec2p = eta * np.cos(np.pi * dphi * i)
            Y2 = 3 - ec2p
            Y = np.sqrt(Y2)
            FF += 0.5 * np.sqrt(3) / Y * _integrals(R, A, eps, Y2, 0, disp)

        order = 5
        for N in [2 ** n for n in range(2, order + 1)]:
            dphi = 1 / N

            for i in range(N - 1, 0, -2):
                ec2p = eta * np.cos(np.pi * dphi * i)
                Y2 = 3 - ec2p
                Y = np.sqrt(Y2)
                FF += np.sqrt(3) / Y * _integrals(R, A, eps, Y2, ec2p, disp)

        FF *= dphi

    return FF


def _integrals(R, A, eps, Y2, etac2p, disp=False):
    """Single-orientation Dulya integrand for absorption (disp=False) or dispersion (disp=True)."""
    Y = np.sqrt(Y2)
    Yx2 = 2 * Y
    z2 = 1 - eps * R - etac2p
    q4 = z2 * z2 + A * A
    q2 = np.sqrt(q4)
    qq = np.sqrt(q2)

    cosa = z2 / q2
    cosa_2 = 1 / np.sqrt(2) * np.sqrt(1 + cosa)
    sina_2 = 1 / np.sqrt(2) * np.sqrt(1 - cosa)

    fTmp = Y2 + q2
    fVal = Yx2 * qq * cosa_2

    if disp:
        T = sina_2 * (np.pi / 2 + np.arctan((Y2 - q2) / (Yx2 * qq * sina_2)))
        L = -0.5 * cosa_2 * np.log((fTmp + fVal) / (fTmp - fVal))
        return (T + L) / (np.pi * qq)
    else:
        T = cosa_2 * (np.pi / 2 + np.arctan((Y2 - q2) / (Yx2 * qq * sina_2)))
        L = 0.5 * sina_2 * np.log((fTmp + fVal) / (fTmp - fVal))
        return (T + L) / (2 * qq)
