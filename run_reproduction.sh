#!/usr/bin/env bash
# Verify the packaged results and rebuild figures and PDF.
#
# If data/ is absent or incomplete this script performs the full numerical
# rebuild first, so a clean unpack of the archive always reaches a verified
# state without manual intervention. The rebuild is expensive; see
# rebuild_from_scratch.sh for the same path with logging.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export MPLBACKEND=Agg
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python verify_results.py --self-test

REQUIRED=(
  "data/continuum_theory_results/continuum_balance_verification.json"
  "data/baseline_results/summary.json"
  "data/advance_resistance_results/advance_unit_steps.csv"
  "data/configurational_work_bridge_results/summary.json"
)
MISSING=0
for f in "${REQUIRED[@]}"; do
  [ -f "$f" ] || { echo "missing dataset: $f"; MISSING=1; }
done

if [ "$MISSING" -ne 0 ]; then
  echo
  echo "One or more packaged datasets are absent. Running the full numerical"
  echo "rebuild before verification. This may take several hours."
  echo
  mkdir -p logs
  python recompute_all_data.py 2>&1 | tee logs/full_recompute.log
fi

python verify_results.py
python generate_figures.py --out manuscript
bash build_manuscript.sh
echo
echo "Reproduction complete: manuscript/manuscript.pdf"
