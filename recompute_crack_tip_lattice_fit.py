#!/usr/bin/env python3
"""Run the Lattice-to-continuum crack-tip field fit."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "src"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from crack_tip_lattice_fit import run_analysis


def main() -> None:
    summary = run_analysis(ROOT / "data" / "crack_tip_lattice_fit_results")
    report = [
        "Lattice crack-tip field fit",
        f"pass={summary['pass']}",
        f"passive residual={summary['passive_nominal_relative_residual']:.6e}",
        f"k_o=0.15 matched residual={summary['ko_0p15_matched_relative_residual']:.6e}",
        f"k_o=0.15 passive-basis residual={summary['ko_0p15_passive_basis_relative_residual']:.6e}",
        f"mirror coefficient error={summary['mirror_max_abs_coefficient_difference']:.6e}",
    ]
    (ROOT / "outputs" / "crack_tip_lattice_fit_report.txt").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if not summary["pass"]:
        raise SystemExit("Lattice crack-tip fit verification failed")


if __name__ == "__main__":
    (ROOT / "outputs").mkdir(parents=True, exist_ok=True)
    main()
