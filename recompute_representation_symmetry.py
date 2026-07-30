#!/usr/bin/env python3
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from representation_symmetry_validation import run_all

if __name__ == "__main__":
    (ROOT / "outputs").mkdir(parents=True, exist_ok=True)
    result = run_all(ROOT / "data" / "representation_symmetry_results")
    (ROOT / "outputs" / "representation_symmetry_report.txt").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["pass"] else 1)
