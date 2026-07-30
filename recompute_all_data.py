#!/usr/bin/env python3
"""Dependency-aware regeneration of all archived numerical datasets.

The script creates output directories itself and runs producers in the order
required by downstream audit modules. Existing files are overwritten by their
producer; worker caches under ``data_recomputed`` are retained unless --clean
is supplied.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

STAGES: list[tuple[str, list[str]]] = [
    ("continuum_balance", [PYTHON, "recompute_continuum_balance.py"]),
    ("crack_tip_asymptotics", [PYTHON, "recompute_crack_tip_asymptotics.py"]),
    ("baselines", [PYTHON, "recompute_baselines.py"]),
    ("apparent_j", [PYTHON, "recompute_apparent_j.py"]),
    ("discrete_configurational", [PYTHON, "recompute_discrete_configurational.py"]),
    ("localization_gauge", [PYTHON, "recompute_localization_gauge.py"]),
    ("crack_tip_lattice_fit", [PYTHON, "recompute_crack_tip_lattice_fit.py"]),
    ("same_endpoint_scaling", [PYTHON, "recompute_same_endpoint_scaling.py"]),
    ("same_endpoint_size_scaling", [PYTHON, "recompute_same_endpoint_size_scaling.py"]),
    ("gauge_convergence", [PYTHON, "recompute_gauge_convergence.py"]),
    ("advance_resistance", [PYTHON, "recompute_advance_resistance.py"]),
    ("protocol_threshold", [PYTHON, "recompute_protocol_threshold.py"]),
    ("protocol_mobility", [PYTHON, "recompute_protocol_mobility.py"]),
    ("protocol_family", [PYTHON, "recompute_protocol_family.py"]),
    ("directional_driving", [PYTHON, "recompute_directional_driving.py"]),
    ("configurational_work_bridge", [PYTHON, "recompute_configurational_work_bridge.py"]),
    ("finite_modulus_validation", [PYTHON, "recompute_finite_modulus_validation.py"]),
    ("refinement_validation", [PYTHON, "recompute_refinement_validation.py"]),
    ("representation_symmetry", [PYTHON, "recompute_representation_symmetry.py"]),
    ("continuum_domain", [PYTHON, "recompute_continuum_domain.py"]),
    ("continuum_lattice_validation", [PYTHON, "recompute_continuum_lattice_validation.py", "--full"]),
]

DATA_DIRS = [
    "advance_resistance_results", "apparent_j_results", "baseline_results",
    "configurational_work_bridge_results", "continuum_lattice_validation_results",
    "continuum_domain_results", "continuum_theory_results", "crack_tip_asymptotics_results",
    "crack_tip_lattice_fit_results", "directional_driving_results",
    "discrete_configurational_results", "finite_modulus_validation_results",
    "gauge_convergence_results", "localization_gauge_results",
    "protocol_family_results", "protocol_threshold_results",
    "refinement_validation_results", "representation_symmetry_results",
    "same_endpoint_scaling_results", "same_endpoint_size_scaling_results",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="list stages and exit")
    parser.add_argument("--from-stage", choices=[name for name, _ in STAGES])
    parser.add_argument("--to-stage", choices=[name for name, _ in STAGES])
    parser.add_argument("--only", choices=[name for name, _ in STAGES])
    parser.add_argument("--clean", action="store_true", help="remove generated data directories and worker caches first")
    parser.add_argument("--verify", action="store_true", help="run verification and figure generation after numerical stages")
    args = parser.parse_args()

    if args.list:
        for i, (name, command) in enumerate(STAGES, start=1):
            print(f"{i:02d} {name}: {' '.join(command)}")
        return

    if args.clean:
        for name in DATA_DIRS:
            shutil.rmtree(ROOT / "data" / name, ignore_errors=True)
        shutil.rmtree(ROOT / "data_recomputed", ignore_errors=True)

    (ROOT / "outputs").mkdir(parents=True, exist_ok=True)
    for name in DATA_DIRS:
        (ROOT / "data" / name).mkdir(parents=True, exist_ok=True)

    selected = STAGES
    if args.only:
        selected = [stage for stage in STAGES if stage[0] == args.only]
    else:
        names = [name for name, _ in STAGES]
        start = names.index(args.from_stage) if args.from_stage else 0
        stop = names.index(args.to_stage) + 1 if args.to_stage else len(STAGES)
        if start >= stop:
            raise SystemExit("--from-stage must not occur after --to-stage")
        selected = STAGES[start:stop]

    env = os.environ.copy()
    env.update({
        "MPLBACKEND": "Agg",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    })
    for name, command in selected:
        print(f"\n=== {name} ===", flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)

    if args.verify:
        subprocess.run([PYTHON, "verify_results.py"], cwd=ROOT, env=env, check=True)
        subprocess.run([PYTHON, "generate_figures.py", "--out", "manuscript"], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
