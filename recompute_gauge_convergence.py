#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "src"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from gauge_convergence import (
    GaugeConvergenceConfig,
    compute_size_rows,
    summarize_rows,
)


def read_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-nx", type=int)
    parser.add_argument("--worker-out", type=Path)
    args = parser.parse_args()
    config = GaugeConvergenceConfig()

    if args.worker_nx is not None:
        if args.worker_out is None:
            raise SystemExit("--worker-out is required with --worker-nx")
        rr, fr = compute_size_rows(args.worker_nx, config)
        # Reuse the single-size summarizer only to write deterministic CSV files.
        single = GaugeConvergenceConfig(nx_values=(args.worker_nx,))
        summarize_rows(args.worker_out, single, rr, fr)
        return

    out = ROOT / "data" / "gauge_convergence_results"
    work = ROOT / "data_recomputed" / "gauge_convergence_workers"
    work.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    raw_rows: list[dict[str, object]] = []
    force_rows: list[dict[str, object]] = []
    for nx in config.nx_values:
        worker_out = work / f"nx_{nx}"
        if not (worker_out / "gauge_convergence_raw.csv").exists():
            print(f"gauge worker Nx={nx}", flush=True)
            subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--worker-nx", str(nx), "--worker-out", str(worker_out)],
                check=True,
                env=env,
            )
        raw_rows.extend(read_rows(worker_out / "gauge_convergence_raw.csv"))
        force_rows.extend(read_rows(worker_out / "gauge_force_reproduction.csv"))

    summary = summarize_rows(out, config, raw_rows, force_rows)
    report = ROOT / "outputs" / "gauge_convergence_report.txt"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary.get("pass", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
