// Runs the Q-meter model (qmeter.py) and the signal analysis (analysis.py) in Pyodide, off the main thread.
const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.27.2/full/";
const PY_FILES = ["qmeter.py", "deuteron_lineshape.py", "proton_lineshape.py", "analysis.py"];

async function boot() {
  postMessage({ type: "progress", text: "Loading Pyodide…" });
  importScripts(PYODIDE + "pyodide.js");
  const py = await loadPyodide({ indexURL: PYODIDE });
  postMessage({ type: "progress", text: "Loading NumPy…" });
  await py.loadPackage("numpy");
  postMessage({ type: "progress", text: "Loading Q-meter model…" });
  for (const f of PY_FILES) {
    const r = await fetch("py/" + f + self.location.search, { cache: "no-store" });
    if (!r.ok) throw new Error(`could not fetch py/${f} (${r.status})`);
    py.FS.writeFile(f, await r.text());
  }
  py.runPython("import sys; sys.path.insert(0, '.')\nimport numpy, qmeter, analysis");
  const defaults = py.runPython("import json; json.dumps(qmeter.DEFAULTS)");
  const analysisDefaults = py.runPython("json.dumps(analysis.ANALYSIS_DEFAULTS)");
  postMessage({ type: "ready", version: py.version, numpy: py.runPython("numpy.__version__"), defaults, analysisDefaults });
  return py;
}

const ready = boot().catch((e) => {
  postMessage({ type: "fatal", text: String(e && e.message ? e.message : e) });
  throw e;
});

onmessage = async (ev) => {
  const msg = ev.data;
  if (msg.type !== "compute" && msg.type !== "tune") return;
  let py;
  try { py = await ready; } catch { return; }
  const qm = py.globals.get("qmeter"), an = py.globals.get("analysis");
  try {
    if (msg.type === "compute") {
      postMessage({ type: "result", id: msg.id, json: an.run_json(JSON.stringify(msg.params), JSON.stringify(msg.settings)) });
    } else {
      postMessage({ type: "tuned", id: msg.id, mode: msg.mode, json: qm.tune_json(msg.mode, JSON.stringify(msg.params)) });
    }
  } catch (e) {
    // last line of a Python traceback is "ValueError: message"; drop the exception class name
    const last = String(e && e.message ? e.message : e).trim().split("\n").filter(Boolean).pop();
    const text = last.replace(/^\w+(Error|Exception): /, "");
    postMessage({ type: msg.type === "compute" ? "error" : "tuneError", id: msg.id, mode: msg.mode, text });
  }
};
