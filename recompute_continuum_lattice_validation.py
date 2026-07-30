#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from continuum_lattice_validation import run_all

if __name__ == "__main__":
    (ROOT / "outputs").mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="recompute all continuum-lattice validation calculations, including long-time mobility tails")
    args = parser.parse_args()
    result = run_all(ROOT / "data" / "continuum_lattice_validation_results", recompute_all=args.full)
    (ROOT / "outputs" / "continuum_lattice_validation_report.txt").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["pass"] else 1)
