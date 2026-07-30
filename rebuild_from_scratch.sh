#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export MPLBACKEND=Agg
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
rm -rf outputs data_recomputed logs
mkdir -p outputs logs
find manuscript -maxdepth 1 -type f \( -name 'fig*.pdf' -o -name 'manuscript.pdf' -o -name '*.aux' -o -name '*.bbl' -o -name '*.blg' -o -name '*.log' -o -name '*.out' -o -name '*.toc' -o -name '*.fls' -o -name '*.fdb_latexmk' \) -delete
python recompute_all_data.py --clean 2>&1 | tee logs/full_recompute.log
python verify_results.py 2>&1 | tee logs/verification.log
python generate_figures.py --out manuscript 2>&1 | tee logs/figure_generation.log
bash build_manuscript.sh 2>&1 | tee logs/manuscript_build.log
python package_finalize.py
