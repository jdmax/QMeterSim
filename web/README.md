# Q-Meter Circuit Simulator (web app)

The model from [`qmeter_sim.ipynb`](../qmeter_sim.ipynb), running in the browser at
**https://jdmax.github.io/QMeterSim/** with no installation. Python ([Pyodide](https://pyodide.org)) runs this
folder's own `qmeter.py`, `deuteron_lineshape.py` and `analysis.py` in a web worker, so the curves update live as you
change cable lengths, tuning, phase, species, field and the circuit constants.

- **Tune** buttons on the diode and phase plots set the capacitance (and, for the phase signal, the phase) that center
  the curve on the Larmor frequency for the current cable lengths.
- **Proton signal:** fitted to measured 5 T proton lines taken while the polarization was growing: spin-ups from
  7% to 78% and the build-up after two reversals from −31% to −73%. Near P = 0 the line is nearly symmetric and
  about 46 kHz wide. For positive P it shifts up (19.7 kHz per unit P) and sharpens on its high-frequency side,
  to 37 kHz wide with a long low-frequency tail at 78%. For negative P it shifts down (15.7 kHz per unit P) and
  sharpens on its low-frequency side, nearly the mirror image with a longer tail: 44 kHz wide at −73%. Beyond
  the measured range the shape is held, and only the shift and size follow P. A real line also depends on how
  fast the polarization is changing, which the model leaves out. Its dispersion is
  computed exactly from it, and its size follows Seely's susceptibility with the notebook's Hill & Hill
  normalization. This replaces the notebook's analytic lineshape, whose complex power below resonance made the
  line lopsided the wrong way and its dispersion unphysical.
- **Deuteron signal:** the Pake doublet of Dulya et al. [[1]](#1), ported from
  [DeuteronPeakFit](https://github.com/jdmax/DeuteronPeakFit), with its Kramers–Kronig dispersion. It is normalized
  to the same susceptibility per spin as the proton signal.
- **Signal analysis:** the simulated sweep is treated the way real data is treated in
  [NMR_Analysis](https://github.com/jdmax/NMR_Analysis) and [rgc_offline](https://github.com/jdmax/rgc_offline).
  The steps are baseline subtraction, a polynomial fit to the wings, integration, and calibration against a simulated
  thermal-equilibrium signal. For deuterons, the polarization also comes from the lineshape, by peak-height ratio and
  by absorption and complex lineshape fits. Since the true polarization is known, the page shows how well each method
  recovers it.
- You can pin a curve as a reference, download the sweep as CSV, and share a link that keeps your settings. The page
  also writes out the model's equations.

Polarization here has its physical sign: positive is absorptive, as at thermal equilibrium. The notebook's `pol` is
the negative of this, so its default of −0.0027 is +0.27%.

### Differences from the notebook's cables

The web app's transmission lines differ from the notebook's in three ways. With zero-length cables, the two
models agree up to an overall scale.

1. Each line's characteristic impedance uses the cable's capacitance per meter, so it is about 51.7 Ω whatever
   the length. The notebook divides by the line's total capacitance, so its impedance changes with length (about
   99 Ω at λ/4 for protons, 35 Ω at 2λ), and zero-length lines are singular.
2. The source voltage is carried through the supply, signal and reference lines, each stage as a Thevenin
   source. The notebook used the source voltage directly at the tank and at the mixer.
3. The mixer output is Re(V<sub>rf</sub> V<sub>ref</sub><sup>*</sup>), the phase difference between its
   inputs, rather than Re(V<sub>rf</sub> V<sub>ref</sub>). This way the reference cable's delay offsets the
   supply and signal cables' delay.

As a result, the supply and signal lines are matched (terminated in 50 Ω), and their lengths don't change the
shape of the Q-curve, as described by Court *et al.* (NIM A 527 (2004), sec. 5.6). They only delay the signal,
which the reference cable or the phase setting corrects.

## Files

* `qmeter.py`: the notebook's circuit, vectorized with numpy, with corrected cables (see below), plus tuning
* `deuteron_lineshape.py`: the deuteron lineshape and susceptibility
* `proton_lineshape.py`: the polarization-dependent proton lineshape, tabulated from measured lines, and its
  exact Kramers–Kronig dispersion
* `analysis.py`: baseline and polynomial subtraction, integration, TE calibration, deuteron lineshape fits
* `test_qmeter.py`: checks `qmeter.py` against a verbatim copy of the notebook's `update()` code with
  zero-length cables, checks that matched cables don't change the Q-curve, and checks the tuning, the deuteron
  lineshape and the analysis
* `index.html`, `app.js`, `worker.js`, `style.css`: the page
* `proton.html`, `proton.js`, `proton_data.json`: a second page on how the proton line changes while the
  polarization moves, with measured lines from two spin-ups and two reversals
* `tools/make_proton_data.py`: builds `proton_lineshape.py` and `proton_data.json` from NMR event logs (the logs
  themselves are not in the repository; by default it reads the git-ignored `temp/`)
* `serve_web.py`: assembles the site in `_site/`

## Running locally

From the repository root, with numpy installed:

```
python web/test_qmeter.py      # validation tests
python web/serve_web.py        # build web/_site/ and serve it at http://localhost:8000
```

On every push to `master`, `../.github/workflows/pages.yml` runs the tests, builds the site and deploys it to GitHub
Pages. This needs Settings → Pages → Build and deployment → Source set to **GitHub Actions**. The workflow has to stay
in `.github/` at the repository root, because that's the only place GitHub looks for it.

## References

<a id="1">[1]</a>
C. Dulya, et. al. "A line-shape analysis for spin-1 NMR signals," NIM A 398 (1997) 109–125.
(https://doi.org/10.1016/S0168-9002(97)00317-3)

<a id="2">[2]</a>
J.D. Maxwell, J. Brock, C. Cuevas, H. Dong, C.D. Keith, J.J. Pierce, "A new cw-NMR Q-meter for dynamically polarized
targets for particle physics," NIM A 1087 (2026) 171417. (https://doi.org/10.1016/j.nima.2026.171417)

See the page itself for the full reference list, and the [main README](../README.md) for the notebook.
