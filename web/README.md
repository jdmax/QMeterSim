# Q-Meter Circuit Simulator (web app)

The model from [`qmeter_sim.ipynb`](../qmeter_sim.ipynb), running in the browser at
**https://jdmax.github.io/QMeterSim/** with no installation. Python ([Pyodide](https://pyodide.org)) runs this
folder's own `qmeter.py`, `deuteron_lineshape.py` and `analysis.py` in a web worker, so the curves update live as you
change cable lengths, tuning, phase, species, field and the circuit constants.

- **Tune** buttons on the diode and phase plots set the capacitance (and, for the phase signal, the phase) that center
  the curve on the Larmor frequency for the current cable lengths.
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

## Files

* `qmeter.py`: the notebook's circuit, vectorized with numpy (the notebook itself is unchanged), plus tuning
* `deuteron_lineshape.py`: the deuteron lineshape and susceptibility
* `analysis.py`: baseline and polynomial subtraction, integration, TE calibration, deuteron lineshape fits
* `test_qmeter.py`: checks `qmeter.py` against a verbatim copy of the notebook's `update()` code, and checks the
  tuning, the deuteron lineshape and the analysis
* `index.html`, `app.js`, `worker.js`, `style.css`: the page
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
