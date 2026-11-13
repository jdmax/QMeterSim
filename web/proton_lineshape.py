"""Proton NMR absorption lineshape, measured, with its exact Kramers-Kronig dispersion.

The shape is the fit-subtracted proton signal of one event from jdmax/NMR_Analysis (10_9_22_proton.json:
5 T, 212.9 MHz, polarization 79%, 2022-10-09), sampled every 1.5655 kHz. The faint tails (below 2% of the
peak) are lightly smoothed, noise below zero is clipped, and both ends are tapered to zero; the core of the line
is as measured. It is asymmetric: a sharp high-frequency edge and a long low-frequency tail, about 34 kHz wide
at half height. Offsets are from the line's centroid, which is placed at the Larmor frequency (the peak is
12.5 kHz above it). The shape is the same at every polarization; only its size scales.

The absorption between samples is a straight line, so it is a sum of triangular "hat" functions. The Hilbert
transform of a hat of half-width h centered at 0 is
    H[hat](x) = [F(x+h) - 2 F(x) + F(x-h)] / (pi h),   F(x) = x ln|x|,
so the dispersion chi' = -H[chi''] is exact for this absorption, with the correct 1/x tails far from the line.
"""

import numpy as np

STEP_KHZ = 1.565534        # sample spacing
START_KHZ = -277.116026      # offset of the first sample from the centroid
SHAPE = np.array([
    0.00000, 0.00004, 0.00005, 0.00009, 0.00023, 0.00027, 0.00030, 0.00050, 0.00051, 0.00051, 0.00064, 0.00065,
    0.00048, 0.00051, 0.00045, 0.00033, 0.00042, 0.00060, 0.00053, 0.00057, 0.00070, 0.00067, 0.00068, 0.00084,
    0.00079, 0.00068, 0.00077, 0.00071, 0.00065, 0.00076, 0.00082, 0.00068, 0.00076, 0.00076, 0.00076, 0.00090,
    0.00104, 0.00097, 0.00110, 0.00116, 0.00108, 0.00116, 0.00136, 0.00130, 0.00128, 0.00141, 0.00139, 0.00125,
    0.00138, 0.00151, 0.00152, 0.00171, 0.00191, 0.00184, 0.00187, 0.00195, 0.00185, 0.00182, 0.00203, 0.00213,
    0.00217, 0.00229, 0.00231, 0.00226, 0.00230, 0.00232, 0.00248, 0.00274, 0.00283, 0.00285, 0.00289, 0.00289,
    0.00286, 0.00308, 0.00326, 0.00338, 0.00342, 0.00359, 0.00359, 0.00370, 0.00389, 0.00397, 0.00399, 0.00411,
    0.00410, 0.00413, 0.00435, 0.00445, 0.00449, 0.00463, 0.00475, 0.00477, 0.00500, 0.00522, 0.00531, 0.00543,
    0.00560, 0.00563, 0.00571, 0.00596, 0.00606, 0.00612, 0.00634, 0.00651, 0.00665, 0.00698, 0.00728, 0.00739,
    0.00764, 0.00789, 0.00800, 0.00829, 0.00867, 0.00893, 0.00920, 0.00963, 0.00995, 0.01030, 0.01079, 0.01124,
    0.01161, 0.01211, 0.01267, 0.01319, 0.01385, 0.01458, 0.01527, 0.01601, 0.01687, 0.01775, 0.01878, 0.01993,
    0.02116, 0.02239, 0.02357, 0.02534, 0.02750, 0.02966, 0.03205, 0.03508, 0.03826, 0.04187, 0.04641, 0.05151,
    0.05770, 0.06452, 0.07007, 0.07863, 0.08865, 0.09957, 0.11200, 0.12539, 0.14017, 0.15587, 0.17318, 0.19164,
    0.21100, 0.23131, 0.25147, 0.27185, 0.29237, 0.31287, 0.33280, 0.35158, 0.36988, 0.38808, 0.40720, 0.42763,
    0.44841, 0.47070, 0.49388, 0.51880, 0.54675, 0.57880, 0.61464, 0.65339, 0.69367, 0.73508, 0.77678, 0.81978,
    0.86298, 0.90567, 0.94447, 0.97569, 0.99497, 1.00000, 0.98948, 0.96251, 0.92068, 0.86449, 0.79516, 0.71553,
    0.63074, 0.54218, 0.45651, 0.37630, 0.30561, 0.24483, 0.19511, 0.15612, 0.12652, 0.10288, 0.08596, 0.07165,
    0.06088, 0.05235, 0.04448, 0.03858, 0.03422, 0.02999, 0.02593, 0.02279, 0.02070, 0.01895, 0.01725, 0.01569,
    0.01433, 0.01308, 0.01190, 0.01093, 0.01010, 0.00925, 0.00850, 0.00779, 0.00706, 0.00642, 0.00585, 0.00528,
    0.00482, 0.00446, 0.00405, 0.00376, 0.00348, 0.00327, 0.00289, 0.00266, 0.00246, 0.00235, 0.00219, 0.00207,
    0.00188, 0.00171, 0.00150, 0.00137, 0.00124, 0.00113, 0.00101, 0.00087, 0.00071, 0.00066, 0.00059, 0.00052,
    0.00050, 0.00036, 0.00024, 0.00017, 0.00012, 0.00007, 0.00005, 0.00002, 0.00000, 0.00000
])
KNOTS_KHZ = START_KHZ + STEP_KHZ * np.arange(len(SHAPE))
AREA_KHZ = STEP_KHZ * float(SHAPE.sum())     # integral of the absorption (peak 1) over offset in kHz


def _xlogx(v):
    a = np.abs(v)
    return np.where(a > 0, v * np.log(np.where(a > 0, a, 1.0)), 0.0)


def absorption(x_khz):
    """Normalized absorption (peak 1) at offsets x_khz from the centroid."""
    return np.interp(x_khz, KNOTS_KHZ, SHAPE, left=0.0, right=0.0)


def hilbert(x_khz):
    """Hilbert transform of absorption(), exactly (sum of the hat functions' transforms)."""
    x = np.atleast_1d(np.asarray(x_khz, float))
    out = np.zeros_like(x)
    h = STEP_KHZ
    nz = np.nonzero(SHAPE)[0]
    for lo in range(0, len(nz), 64):                        # chunks keep memory small
        k = nz[lo:lo + 64]
        u = x[:, None] - KNOTS_KHZ[k][None, :]
        out += ((_xlogx(u + h) - 2 * _xlogx(u) + _xlogx(u - h)) / (np.pi * h)) @ SHAPE[k]
    return out


def susceptibility(f_mhz, pol, f0_mhz, chi0_per_pol):
    """Complex susceptibility chi = chi' - i chi'' at frequencies f_mhz, normalized like the deuteron and the
    notebook's Hill & Hill line: the integral of chi'' over omega is (pi/2) omega0 chi0, chi0 = chi0_per_pol * pol."""
    x = (np.asarray(f_mhz, float) - f0_mhz) * 1e3
    k = (np.pi / 2) * f0_mhz * 1e3 * chi0_per_pol * pol / AREA_KHZ
    return -k * hilbert(x) - 1j * k * absorption(x)
