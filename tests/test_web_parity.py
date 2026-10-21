"""The JavaScript port behind the phone web app (docs/engine.js) must agree with this package.

tools/web_parity/check.js replays the Python answers stored in
tools/web_parity/cases.json (regenerate them with tools/web_parity/export.py)
and exits with status 1 on any mismatch.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CHECK = ROOT / "tools" / "web_parity" / "check.js"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH")
def test_javascript_engine_matches_python():
    proc = subprocess.run(["node", str(CHECK)], cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout[-6000:] + proc.stderr[-3000:]
