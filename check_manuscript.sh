#!/usr/bin/env bash
# Compile the manuscript and report everything worth failing on.
#
# pdflatex in nonstopmode exits 0 and still produces a PDF after a hard error,
# so a successful-looking build can hide a broken table or a missing brace.
# This script greps the log for errors explicitly, which is the check that
# matters most and the one easiest to forget.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT

cp "$ROOT/manuscript.tex" "$ROOT/references.bib" "$BUILD/" || exit 1
cp "$ROOT"/figures/*.pdf "$BUILD/" 2>/dev/null

cd "$BUILD" || exit 1
pdflatex -interaction=nonstopmode manuscript.tex >/dev/null 2>&1
bibtex manuscript >/dev/null 2>&1
pdflatex -interaction=nonstopmode manuscript.tex >/dev/null 2>&1
pdflatex -interaction=nonstopmode manuscript.tex >build.log 2>&1

status=0
report () { printf '  %-24s %s\n' "$1" "$2"; }

if [ ! -f manuscript.pdf ]; then
    echo "FAIL: no PDF produced"; exit 1
fi

errors=$(grep -c '^! ' build.log)
report "LaTeX errors" "$errors"
if [ "$errors" -ne 0 ]; then
    grep -A2 '^! ' build.log | sed 's/^/      /'
    status=1
fi

undef=$(grep -c 'undefined' build.log)
report "undefined references" "$undef"
[ "$undef" -ne 0 ] && status=1

missing=$(grep -ci 'file .* not found' build.log)
report "missing figures" "$missing"
[ "$missing" -ne 0 ] && status=1

overfull=$(grep 'Overfull \\hbox' build.log | sed 's/.*(\([0-9.]*\)pt.*/\1/' | awk '$1>20' | wc -l)
report "overfull hbox >20pt" "$overfull"

pages=$(python3 -c "from pypdf import PdfReader; print(len(PdfReader('manuscript.pdf').pages))" 2>/dev/null || echo "?")
report "pages" "$pages"

# figures are included at \textwidth; the generators use a 10.5 pt base font
python3 - <<'PY'
from pathlib import Path
try:
    from pypdf import PdfReader
except ImportError:
    raise SystemExit
TW = 469.75502 / 72.27
small = []
for f in sorted(Path("../figures").glob("*.pdf")) or sorted(Path(".").glob("fig*.pdf")):
    w = float(PdfReader(f).pages[0].mediabox.width) / 72
    pt = 10.5 * TW / w
    if pt < 5.6:
        small.append((f.name, pt))
print(f'  {"figures under 5.6 pt":<24} {len(small)}')
for n, pt in small:
    print(f"      {n}  {pt:.2f} pt")
PY

echo
if [ "$status" -eq 0 ]; then
    echo "PASS"
else
    echo "FAIL"
fi
exit "$status"
