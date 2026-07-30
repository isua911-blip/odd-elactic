#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from apparent_j_analysis import main
if __name__=='__main__':
    sys.argv += ['--out', str(ROOT/'data'/'apparent_j_results')] if '--out' not in sys.argv else []
    main()
