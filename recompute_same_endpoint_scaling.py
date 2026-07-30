#!/usr/bin/env python3
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "src"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
from same_endpoint_scaling import run_analysis

summary = run_analysis(ROOT / "data" / "same_endpoint_scaling_results")
report = ROOT / "outputs" / "same_endpoint_scaling_report.txt"
report.parent.mkdir(parents=True, exist_ok=True)
report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
if not summary.get("pass", False):
    raise SystemExit(1)
