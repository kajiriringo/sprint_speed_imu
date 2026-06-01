"""Summary and artifact writing."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from .config import RunConfig
from .plotting import plot_speed_curve


ASSUMPTIONS = [
    "Straight sprint.",
    "Sensor fixed to trunk, waist, sacrum, or lower back.",
    "Known distance is correct when distance_source=manual.",
    "Estimated speed is not official timing or photogate-equivalent speed.",
]

def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and not path.is_dir():
        raise FileExistsError(f"Output path exists and is not a directory: {path}.")
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {path}. Use --overwrite.")
    path.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for artifact in path.iterdir():
            if artifact.is_dir():
                shutil.rmtree(artifact)
            else:
                artifact.unlink()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_artifacts(
    config: RunConfig,
    curve: pd.DataFrame,
    debug_df: pd.DataFrame | None,
    summary: dict[str, Any],
    warnings: list[str],
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    curve.to_csv(config.output_dir / "speed_curve.csv", index=False)
    write_json(config.output_dir / "summary.json", summary)
    write_json(config.output_dir / "run_config.json", config.to_jsonable())
    (config.output_dir / "warnings.txt").write_text("\n".join(warnings) + ("\n" if warnings else ""), encoding="utf-8")
    if debug_df is not None:
        debug_df.to_csv(config.output_dir / "debug_intermediate.csv", index=False)
    plot_speed_curve(curve, summary, config.output_dir / "speed_curve.png")
