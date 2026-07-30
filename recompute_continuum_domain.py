#!/usr/bin/env python3
"""Regenerate the smooth-domain reconstruction audit.

This stage produces the kernel-robustness range and the total-divergence closure
quoted in the Methods section. It is an independent reconstruction check on the
sign, chirality and parameter dependence of the contour drift.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

if __name__ == "__main__":
    runpy.run_path(str(ROOT / "src" / "continuum_domain_analysis.py"), run_name="__main__")
