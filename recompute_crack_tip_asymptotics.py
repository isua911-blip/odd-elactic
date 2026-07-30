#!/usr/bin/env python3
"""Entry point for crack-tip asymptotics."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from crack_tip_asymptotics import run_analysis


def main() -> None:
    output = ROOT / "data" / "crack_tip_asymptotics_results"
    summary = run_analysis(output)
    print(json.dumps(summary, indent=2))
    if not summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
