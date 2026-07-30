#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
CODE = ROOT / "src"
sys.path.insert(0, str(CODE))
from directional_driving import main
if __name__ == "__main__":
    main()
