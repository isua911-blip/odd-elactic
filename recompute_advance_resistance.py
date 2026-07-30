#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from advance_resistance import main

if __name__ == "__main__":
    (ROOT / "outputs").mkdir(parents=True, exist_ok=True)
    main()
