#!/usr/bin/env python3
"""Run the public CPU checks without model/data/API access."""
from pathlib import Path
import os
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
for command in [
    [sys.executable, "scripts/build_public_tables.py", "--check"],
    [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
]:
    subprocess.run(command, cwd=ROOT, env=env, check=True)
print("PASS: numeric Level B and CPU release gates; exact PDF build remains on hold")
