#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT/manuscript"
rm -f manuscript.aux manuscript.bbl manuscript.blg manuscript.log manuscript.out manuscript.fls manuscript.fdb_latexmk manuscript.toc
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
if command -v bibtex.original >/dev/null 2>&1; then
  bibtex.original manuscript
elif command -v bibtex >/dev/null 2>&1; then
  bibtex manuscript
else
  echo "BibTeX executable not found" >&2
  exit 1
fi
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
rm -f manuscript.aux manuscript.bbl manuscript.blg manuscript.log manuscript.out manuscript.fls manuscript.fdb_latexmk manuscript.toc
