from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SD_SCRIPTS_ROOT = PROJECT_ROOT / "vendor" / "sd-scripts"


def ensure_local_sd_scripts() -> Path:
    if not (SD_SCRIPTS_ROOT / "anima_minimal_inference.py").exists():
        raise FileNotFoundError(f"vendored sd-scripts Anima code not found: {SD_SCRIPTS_ROOT}")
    value = str(SD_SCRIPTS_ROOT)
    if value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)
    return SD_SCRIPTS_ROOT
