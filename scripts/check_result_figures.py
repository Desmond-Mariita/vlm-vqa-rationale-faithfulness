#!/usr/bin/env python3
"""Render numeric plots in a temporary directory and compare approved hashes."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1]
policy=json.loads((ROOT/"provenance/image_policy.json").read_text())
expected={Path(r["path"]).name:r["sha256"] for r in policy["visual_files"] if r["classification"]=="AUTHOR_CREATED_NON_VCR"}
with tempfile.TemporaryDirectory(prefix="thesis-numeric-figures-") as td:
    base=Path(td);output=base/"figures"
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",MPLCONFIGDIR=str(base/"mpl"),XDG_CACHE_HOME=str(base/"cache"))
    subprocess.run([sys.executable,"scripts/build_manuscript_result_figures.py","--output-dir",str(output)],cwd=ROOT,env=env,check=True)
    actual={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}
    print(json.dumps({"status":"PASS" if actual==expected else "FAIL","hashes":actual,"expected":expected},indent=2))
    if actual!=expected:raise SystemExit(1)
