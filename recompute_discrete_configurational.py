#!/usr/bin/env python3
from __future__ import annotations
import subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data'/'discrete_configurational_results'
OUT.mkdir(parents=True,exist_ok=True)
env=None
subprocess.run([sys.executable,str(ROOT/'src'/'discrete_configurational_analysis.py'),'--out',str(OUT)],check=True,env=env)
subprocess.run([sys.executable,str(ROOT/'src'/'discrete_configurational_robustness.py')],check=True,env=env)
