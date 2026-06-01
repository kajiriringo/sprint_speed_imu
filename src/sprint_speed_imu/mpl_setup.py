"""Matplotlib cache setup for sandboxed macOS environments."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def configure_matplotlib_cache() -> None:
    cache_dir = Path(tempfile.gettempdir()) / "sprint_speed_imu_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
