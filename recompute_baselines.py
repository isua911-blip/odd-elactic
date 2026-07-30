#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from lattice_baselines import main
if __name__=='__main__':
    sys.argv += ['--out', str(ROOT/'data'/'baseline_results')] if '--out' not in sys.argv else []
    main()
