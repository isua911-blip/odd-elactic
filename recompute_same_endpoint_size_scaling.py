#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "src"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from same_endpoint_scaling import ScalingConfig, first_order_map, full_odd_work
from same_endpoint_size_scaling import summarize

SIZES = (32, 48, 64)
RATIOS = (0.5, 1.0, 2.0, 4.0)
ANGLES = (0.0, math.pi / 8.0, math.pi / 4.0, 3.0 * math.pi / 8.0)
KO_VALUES = (-0.02, -0.01, 0.01, 0.02)


def config_for(nx: int) -> ScalingConfig:
    return ScalingConfig(
        nx=nx,
        ny=3 * nx // 4,
        crack_half_length=nx / 8.0,
        t_end=40000.0 * (nx / 32.0) ** 2,
        rtol=2.0e-8,
        relative_atol=1.0e-10,
    )


def worker_first(args) -> None:
    cfg = config_for(args.nx)
    rows, delta_even = first_order_map(cfg, [args.ratio], [args.theta])
    row = rows[0]
    row.update({
        "nx": cfg.nx,
        "ny": cfg.ny,
        "crack_half_length": cfg.crack_half_length,
        "passive_endpoint_even_energy_change": delta_even,
        "passive_resistance": -delta_even,
        "extra_symmetry_point": bool(args.extra),
    })
    args.out.write_text(json.dumps(row, indent=2), encoding="utf-8")


def worker_full(args) -> None:
    cfg = config_for(args.nx)
    result = full_odd_work(cfg, args.ko, args.ratio, args.theta)
    result.update({
        "nx": cfg.nx,
        "ny": cfg.ny,
        "crack_half_length": cfg.crack_half_length,
        "extreme": args.extreme,
    })
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")


def run_subprocess(arguments: list[str], env: dict[str, str]) -> None:
    print("worker", " ".join(arguments), flush=True)
    subprocess.run([sys.executable, str(Path(__file__).resolve()), *arguments], check=True, env=env, timeout=600, start_new_session=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("first", "full"))
    parser.add_argument("--nx", type=int)
    parser.add_argument("--ratio", type=float)
    parser.add_argument("--theta", type=float)
    parser.add_argument("--ko", type=float)
    parser.add_argument("--extreme")
    parser.add_argument("--extra", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if args.worker == "first":
        worker_first(args)
        return
    if args.worker == "full":
        worker_full(args)
        return

    work = ROOT / "data_recomputed" / "same_endpoint_size_workers"
    work.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})

    first_rows = []
    extrema = {}
    for nx in SIZES:
        size_dir = work / f"nx_{nx}"
        size_dir.mkdir(exist_ok=True)
        for ratio in RATIOS:
            for ai, theta in enumerate(ANGLES):
                path = size_dir / f"first_r{ratio:g}_a{ai}.json"
                if not path.exists():
                    run_subprocess(["--worker", "first", "--nx", str(nx), "--ratio", str(ratio), "--theta", repr(theta), "--out", str(path)], env)
                first_rows.append(json.loads(path.read_text()))
        extra_path = size_dir / "first_reciprocal_extra.json"
        if not extra_path.exists():
            run_subprocess(["--worker", "first", "--nx", str(nx), "--ratio", "0.5", "--theta", repr(math.pi / 2.0), "--extra", "--out", str(extra_path)], env)
        first_rows.append(json.loads(extra_path.read_text()))

        base = [r for r in first_rows if int(r["nx"]) == nx and not r["extra_symmetry_point"]]
        extrema[nx] = {
            "min_phi": min(base, key=lambda r: float(r["phi_first_order"])),
            "max_phi": max(base, key=lambda r: float(r["phi_first_order"])),
        }

    full_rows = []
    for nx in SIZES:
        size_dir = work / f"nx_{nx}"
        for ko in KO_VALUES:
            for extreme, row in extrema[nx].items():
                path = size_dir / f"full_{extreme}_ko{ko:+.3f}.json"
                if not path.exists():
                    run_subprocess([
                        "--worker", "full", "--nx", str(nx), "--ko", str(ko),
                        "--ratio", str(row["mobility_ratio"]), "--theta", str(row["theta"]),
                        "--extreme", extreme, "--out", str(path),
                    ], env)
                full_rows.append(json.loads(path.read_text()))

    summary = summarize(ROOT / "data" / "same_endpoint_size_scaling_results", first_rows, full_rows)
    report = ROOT / "outputs" / "same_endpoint_size_scaling_report.txt"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary.get("pass", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
