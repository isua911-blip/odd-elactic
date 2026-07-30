#!/usr/bin/env python3
"""Entry point for continuum material-force verification."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from continuum_material_force import run_manufactured_verification


def main() -> None:
    out = ROOT / "data" / "continuum_theory_results" / "continuum_balance_verification.json"
    result = run_manufactured_verification(out)
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
