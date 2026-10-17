"""Generates index.html (single page, ready for GitHub Pages) from
engine.js (engine) and template.html (visuals).

    python web/build.py              # writes ../index.html
    python web/build.py output.html  # also writes the version without <!doctype>
"""

import sys
from pathlib import Path

here = Path(__file__).resolve().parent
engine = (here / "engine.js").read_text(encoding="utf-8")
# the engine goes inline, without the export block used only in Node
engine = engine.replace('if (typeof module !== "undefined") module.exports = Engine;', "")
assert "</script>" not in engine

fragment = (here / "template.html").read_text(encoding="utf-8").replace("/*ENGINE*/", engine)

header = (
    '<!doctype html>\n<html lang="en">\n<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
    '<meta name="description" content="Compares A* and potential fields on the same random scenario, '
    'with a PID-driven robot following each path.">\n'
)
target = here.parent / "index.html"
target.write_text(header + fragment, encoding="utf-8")
print(f"wrote {target} ({target.stat().st_size} bytes)")

if len(sys.argv) > 1:
    Path(sys.argv[1]).write_text(fragment, encoding="utf-8")
    print(f"wrote {sys.argv[1]}")
