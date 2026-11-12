"""Assemble the static web app into web/_site/ and (optionally) serve it locally.

    python web/serve_web.py           build web/_site/ and serve http://localhost:8000
    python web/serve_web.py --build   build only (used by the GitHub Pages workflow)

The page files and the Python modules the browser runs (in Pyodide) both live in web/. The modules are copied
into _site/py/, which is where worker.js fetches them from.
"""

import argparse
import functools
import hashlib
import http.server
import shutil
from pathlib import Path

WEB = Path(__file__).resolve().parent
REPO = WEB.parent
SITE = WEB / "_site"
PAGE_FILES = ["index.html", "app.js", "worker.js", "style.css"]
PY_FILES = ["qmeter.py", "deuteron_lineshape.py", "analysis.py"]
REPO_ASSETS = ["schem.PNG"]   # shared with the notebook's README, so kept at the repo root


def build():
    if SITE.exists():
        shutil.rmtree(SITE)
    (SITE / "py").mkdir(parents=True)
    for f in PAGE_FILES:
        shutil.copy2(WEB / f, SITE / f)
    for f in PY_FILES:
        shutil.copy2(WEB / f, SITE / "py" / f)
    for f in REPO_ASSETS:
        shutil.copy2(REPO / f, SITE / f)
    (SITE / ".nojekyll").touch()

    # Cache-bust: replace the __V__ placeholder in asset URLs with a hash of the site's code, so a browser
    # never pairs this page with a stale (or another localhost app's) style.css / app.js / worker.js.
    stamped = PAGE_FILES + [f"py/{f}" for f in PY_FILES]
    h = hashlib.sha256()
    for f in stamped:
        h.update((SITE / f).read_bytes())
    version = h.hexdigest()[:10]
    for f in ("index.html", "app.js"):
        p = SITE / f
        p.write_text(p.read_text(encoding="utf-8").replace("__V__", version), encoding="utf-8")
    print(f"built {SITE} (v={version})")


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    """Disable browser caching. Other local apps served on the same localhost port use the same file
    names (worker.js, style.css), and a heuristically cached copy would otherwise be loaded instead."""

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def serve(port):
    handler = functools.partial(NoCacheHandler, directory=str(SITE))
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as httpd:
        print(f"serving http://localhost:{port}  (Ctrl+C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true", help="build _site/ only, do not serve")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    build()
    if not args.build:
        serve(args.port)
