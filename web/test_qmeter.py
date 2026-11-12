"""Check that qmeter.py reproduces the notebook's update() exactly.

`notebook_update` below is the notebook code (qmeter_sim.ipynb) with plotting/printing removed and the
notebook's global controls (sig, species, pol) passed as arguments. Run: python web/test_qmeter.py
"""

import itertools
import numpy as np

import analysis
import deuteron_lineshape as dl
import qmeter


def notebook_update(trim_sup, trim_sig, trim_ref, trim_coil, capacitance, phase, sweep_range,
                    sig=True, species='p', pol=-0.0027):
    u = 1
    r_cc = 681
    c_stray = 0.001e-12
    l_0 = 0.175E-6
    r_coil = 0.3
    r_amp = 50
    r_source = 50
    r = 2
    if 'p' in species:
        gr = 42.576
    elif 'd' in species:
        gr = 6.53566
    b = 5.00
    f0 = b * gr
    w0 = 2 * np.pi * f0 * 1e6
    c = capacitance / (w0 ** 2 * l_0)
    flo = f0 - sweep_range
    fhi = f0 + sweep_range
    wlo = 2 * np.pi * flo * 1e6
    whi = 2 * np.pi * fhi * 1e6
    n_steps = 500
    dw = (whi - wlo) / n_steps
    w = lambda k: wlo + k * dw
    f = lambda k: w(k) / (2 * np.pi)
    v_f = 0.78
    vel = v_f * 2.99e8
    lb2 = vel / (2 * f0 * 1e6)
    len_sup = trim_sup * lb2
    len_sig = trim_sig * lb2
    len_ref = trim_ref * lb2
    len_coil = trim_coil * lb2
    alph = 0.0343
    gam = lambda k: complex(alph, w(k) / vel)
    r_cab = 2 * 50 * alph
    l_cab = 2.54e-7
    c_sup = 95.2e-12 * len_sup
    c_sig = 95.2e-12 * len_sig
    c_ref = 95.2e-12 * len_ref
    c_coil = 95.2e-12 * len_coil
    chi_0 = 3.1e-4 * pol
    del_w = 6e-2 * 2 * np.pi * 1e6
    x_of_w = lambda k: (w(k) - w0) / del_w
    chi_pp = lambda k: 1.035 * chi_0 * w0 / del_w * (1 / (1 + 4 * np.power(x_of_w(k), 2.35 + 0j)))
    chi = lambda k: complex(0, chi_pp(k))
    eta = 0.0013
    if sig:
        l_coil = lambda k: l_0 * (1 + 4 * np.pi * eta * chi(k))
    else:
        l_coil = lambda k: l_0

    def z0_calc(r_cab, l_cab, c_cab, k):
        return np.sqrt((complex(r_cab, w(k) * l_cab)) / (complex(0, w(k) * c_cab)))

    def trans(z0, zl, gamma, length):
        return z0 * ((zl + z0 * np.tanh(gamma * length)) / (z0 + zl * np.tanh(gamma * length)))

    z0_sup = lambda k: z0_calc(r_cab, l_cab, c_sup, k)
    z0_sig = lambda k: z0_calc(r_cab, l_cab, c_sig, k)
    z0_ref = lambda k: z0_calc(r_cab, l_cab, c_ref, k)
    z0_coil = lambda k: z0_calc(r_cab, l_cab, c_coil, k)
    zc = lambda k: 1 / complex(0, w(k) * c)
    zc_stray = lambda k: 1 / complex(0, w(k) * c_stray)
    zl_pure = lambda k: complex(r_coil, w(k) * l_coil(k))
    z_l = lambda k: zl_pure(k) * zc_stray(k) / (zl_pure(k) + zc_stray(k))
    z_coil = lambda k: trans(z0_coil(k), z_l(k), gam(k), len_coil)
    z_tank = lambda k: r + zc(k) + z_coil(k)
    z_sup = lambda k: trans(z0_sup(k), r_source, gam(k), len_sup)
    z_cc = lambda k: r_cc + 1 / (1 / 50 + 1 / z_sup(k))
    z_eq = lambda k: 1 / (1 / z_tank(k) + 1 / z_cc(k))
    z_tot = lambda k: trans(z0_sig(k), z_eq(k), gam(k), len_sig)
    z_ref = lambda k: 1 / (1 / trans(z0_ref(k), r_source, gam(k), len_ref) + 1 / 50)
    v_ref = lambda k: u * (r_amp / (r_amp + z_ref(k)))
    v_eq = lambda k: u * (z_tank(k) / (z_tank(k) + z_cc(k)))
    v_rf = lambda k: v_eq(k) * (r_amp / (r_amp + z_tot(k)))
    phi = phase * np.pi / 180
    v_out = lambda k: v_rf(k) * np.exp(complex(0, phi))
    v_mixer = lambda k: v_rf(k) * v_ref(k) * np.exp(complex(0, phi))
    points = np.arange(0., n_steps, 1)
    v_phase = -np.real(np.vectorize(v_mixer)(points))
    v_diode = np.absolute(np.vectorize(v_out)(points))
    return f(points) / 1e6, v_phase, v_diode, c * 1e12, lb2


def test_matches_notebook():
    cases = [
        dict(trim_sup=1e-9, trim_sig=1e-9, trim_ref=1e-9, trim_coil=1e-9, capacitance=1, phase=0, sweep_range=0.4),
        dict(trim_sup=10., trim_sig=10., trim_ref=.5, trim_coil=1e-5, capacitance=1.0, phase=0, sweep_range=0.4),
        dict(trim_sup=0.3, trim_sig=1.7, trim_ref=2.2, trim_coil=0.5, capacitance=0.97, phase=35, sweep_range=1.2),
    ]
    n = 0
    for case, sig, species, pol_nb in itertools.product(cases, (True, False), ('p', 'd'), (-0.0027, 0.4)):
        if species == 'd' and sig:
            continue  # deuterons now use the Dulya lineshape, not the notebook's proton lineshape
        f, ph, di, c_pf, lb2 = notebook_update(**case, sig=sig, species=species, pol=pol_nb)
        # qmeter's pol has the physical sign: it is the negative of the notebook's
        res = qmeter.simulate(**case, sig=sig, species=species, pol=-pol_nb)
        assert np.allclose(res['f_mhz'], f, rtol=1e-12, atol=0)
        assert np.allclose(res['phase'], ph, rtol=1e-9, atol=1e-15), (case, sig, species)
        assert np.allclose(res['diode'], di, rtol=1e-9, atol=1e-15), (case, sig, species)
        assert np.isclose(res['c_pf'], c_pf) and np.isclose(res['lambda2_m'], lb2)
        n += 1
    print(f"test_matches_notebook: {n} cases OK")


def test_json_entry():
    import json
    for species in ('p', 'd'):
        out = json.loads(analysis.run_json(json.dumps(dict(trim_coil=0.5, species=species, pol=0.3)), "{}"))
        sim, an = out['sim'], out['analysis']
        assert len(sim['f_mhz']) == 500 and all(np.isfinite(sim['diode'])) and all(np.isfinite(sim['phase']))
        assert len(an['en']['sub']) == 500 and np.isfinite(an['cc'])
        assert ('fit' in an) == (species == 'd')
    out = json.loads(analysis.run_json(json.dumps({}), json.dumps(dict(order=40))))
    assert 'analysis_error' in out and 'sim' in out
    print("test_json_entry: OK")


def _hilbert(y):
    n = len(y)
    return np.real(np.fft.ifft(np.fft.fft(y) * (-1j * np.sign(np.fft.fftfreq(n)))))


def test_deuteron_lineshape():
    # population ratio <-> polarization
    for p in (-0.9, -0.3, 0.0, 0.42, 0.95):
        assert np.isclose(dl.pol_from_r(dl.r_from_pol(p)), p)
    # DeuteronPeakFit's dispersion for transition eps is eps * (2/pi) * Hilbert[absorption]
    R = np.linspace(-300, 300, 2 ** 20, endpoint=False)
    m = abs(R) < 4
    for eps in (1, -1):
        h = _hilbert(dl._f(R, 0.05, eps, 0.0))
        d = dl._f(R, 0.05, eps, 0.0, disp=True)
        k = np.dot(d[m], h[m]) / np.dot(h[m], h[m])
        assert np.isclose(k, eps * 2 / np.pi, rtol=1e-3) and np.std(d[m] - k * h[m]) < 1e-3 * np.std(d[m])
    # the susceptibility is causal (chi' = -H[chi'']), absorptive for P > 0, and normalized by its area
    f0 = 32.678
    f = np.linspace(f0 - 60, f0 + 60, 2 ** 20, endpoint=False)
    for p in (0.3, -0.5):
        chi = dl.susceptibility(f, p, f0, 1.0)
        chi_pp = -chi.imag
        area = chi_pp.sum() * (f[1] - f[0])
        assert np.isclose(area, (np.pi / 2) * f0 * p, rtol=0.01), (area, p)
        m = abs(f - f0) < 0.5
        assert np.corrcoef(chi.real[m], -_hilbert(chi_pp)[m])[0, 1] > 0.999
    print("test_deuteron_lineshape: OK")


def test_analysis():
    # TE polarization at 5 T, 1 K
    assert np.isclose(analysis.te_polarization('p', 212.88, 1.0), 0.00511, rtol=2e-3)
    assert np.isclose(analysis.te_polarization('d', 32.678, 1.0), 0.00105, rtol=5e-3)
    # small deuteron signals are linear: TE-calibrated area and the lineshape fit both recover P
    for p in (0.3, -0.3):
        a = analysis.analyze(dict(species='d', pol=p))
        assert abs(a['pol_area'] - p) < 0.002 and abs(a['fit']['pol'] - p) < 0.005, (p, a['pol_area'], a['fit']['pol'])
    # a large proton signal is not: the Q-meter response saturates, so the area under-reads
    a = analysis.analyze(dict(species='p', pol=0.9))
    assert 0.8 < a['pol_area'] < 0.9
    print("test_analysis: OK")


def _extremum_offset_khz(f, y, f0):
    """Offset from f0 of the slope zero-crossing nearest f0."""
    d = np.gradient(y, f)
    k = np.nonzero(np.sign(d[:-1]) != np.sign(d[1:]))[0]
    assert k.size, "no extremum in sweep"
    return 1e3 * (f[k[np.argmin(abs(f[k] - f0))]] - f0)


def test_tune():
    # zero-length cables: tune is the bare tank, which the notebook calls capacitance 1, phase 0
    r = qmeter.tune('phase')
    assert abs(r['capacitance'] - 1) < 0.01 and abs(r['phase']) < 0.5, r

    cases = [dict(trim_coil=1), dict(trim_sup=1, trim_sig=1, trim_ref=0.3, trim_coil=1),
             dict(trim_sup=10., trim_sig=10., trim_ref=.5, trim_coil=1e-5),
             dict(species='d', trim_coil=1, trim_sig=2.3)]
    for case in cases:
        rd = qmeter.tune('diode', **case, phase=17.0)
        assert rd['phase'] == 17.0, "diode tune must leave phase alone"
        r = qmeter.tune('phase', **case)
        assert np.isclose(r['capacitance'], rd['capacitance'])
        s = qmeter.simulate(**case, **r, sig=False, n_steps=8001)
        assert abs(_extremum_offset_khz(s['f_mhz'], s['diode'], s['f0_mhz'])) < 0.5, case
        assert abs(_extremum_offset_khz(s['f_mhz'], s['phase'], s['f0_mhz'])) < 0.5, case
        # tuned phase output has the same orientation as the ideal zero-length circuit at phase 0 (a peak)
        i0 = np.argmin(abs(s['f_mhz'] - s['f0_mhz']))
        assert s['phase'][i0] > s['phase'][0] and s['phase'][i0] > s['phase'][-1], case

    # quarter-wave coil cable: the coil looks capacitive and the series tank cannot be tuned
    try:
        qmeter.tune('diode', trim_coil=0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("expected no tune for a quarter-wave coil cable")
    print(f"test_tune: {len(cases) + 2} cases OK")


if __name__ == '__main__':
    test_matches_notebook()
    test_json_entry()
    test_tune()
    test_deuteron_lineshape()
    test_analysis()
