#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def version(module_name: str) -> str:
    try:
        module = __import__(module_name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception as exc:
        return f"unavailable ({exc})"


def command_version(command: list[str]) -> str:
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        text = (result.stdout or result.stderr).strip().splitlines()
        return text[0] if text else "unknown"
    except Exception as exc:
        return f"unavailable ({exc})"


def main() -> None:
    shutil.rmtree(ROOT / "data_recomputed", ignore_errors=True)
    for path in ROOT.rglob("__pycache__"):
        shutil.rmtree(path, ignore_errors=True)
    for path in ROOT.rglob("*.pyc"):
        path.unlink(missing_ok=True)

    env_lines = [
        f"Platform: {platform.platform()}",
        f"Python: {sys.version.replace(chr(10), ' ')}",
        f"NumPy: {version('numpy')}",
        f"SciPy: {version('scipy')}",
        f"pandas: {version('pandas')}",
        f"Matplotlib: {version('matplotlib')}",
        f"Pillow: {version('PIL')}",
        f"pdfTeX: {command_version(['pdflatex', '--version'])}",
        f"BibTeX: {command_version(['bibtex', '--version'])}",
    ]
    (ROOT / "ENVIRONMENT.txt").write_text("\n".join(env_lines) + "\n", encoding="utf-8")

    manifest = ROOT / "MANIFEST.sha256"
    manifest.unlink(missing_ok=True)
    entries: list[str] = []
    excluded_parts = {".git", "data_recomputed", "__pycache__"}
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path == manifest:
            continue
        rel = path.relative_to(ROOT)
        if any(part in excluded_parts for part in rel.parts):
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append(f"{digest}  {rel.as_posix()}")
    manifest.write_text("\n".join(entries) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
