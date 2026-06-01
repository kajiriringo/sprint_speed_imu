"""CSV loading, column mapping, and unit normalization."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import RunConfig
from .errors import InputFormatError
from .schema import ACC_COLUMNS, GYRO_COLUMNS, MAG_COLUMNS, STANDARD_COLUMNS

COMMON_ALIASES: dict[str, tuple[str, ...]] = {
    "time_s": ("time", "time(s)", "times", "timestamp", "seconds", "sec"),
    "ax": ("accx", "acc_x", "accelerationx", "ax(g)", "xacc", "accelx"),
    "ay": ("accy", "acc_y", "accelerationy", "ay(g)", "yacc", "accely"),
    "az": ("accz", "acc_z", "accelerationz", "az(g)", "zacc", "accelz"),
    "gx": ("gyrox", "gyro_x", "angularvelocityx", "wx", "gx(dps)", "asx"),
    "gy": ("gyroy", "gyro_y", "angularvelocityy", "wy", "gy(dps)", "asy"),
    "gz": ("gyroz", "gyro_z", "angularvelocityz", "wz", "gz(dps)", "asz"),
    "roll": ("anglex", "angle_x", "roll(deg)", "eulerx"),
    "pitch": ("angley", "angle_y", "pitch(deg)", "eulery"),
    "yaw": ("anglez", "angle_z", "yaw(deg)", "eulerz"),
    "qw": ("quatw", "q0", "quaternionw"),
    "qx": ("quatx", "q1", "quaternionx"),
    "qy": ("quaty", "q2", "quaterniony"),
    "qz": ("quatz", "q3", "quaternionz"),
    "hx": ("magx", "mag_x", "magneticx", "mx", "h_x"),
    "hy": ("magy", "mag_y", "magneticy", "my", "h_y"),
    "hz": ("magz", "mag_z", "magneticz", "mz", "h_z"),
}


def _canonical(value: str) -> str:
    value = re.sub(r"\([^)]*\)", "", value.strip().lower())
    return re.sub(r"[^a-z0-9]+", "", value)


def _read_column_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except OSError as exc:
        raise InputFormatError(f"Failed to read column map: {path}") from exc
    if not isinstance(data, dict):
        raise InputFormatError("--column-map must point to a JSON object.")
    return {str(k): str(v) for k, v in data.items()}


def _build_rename_map(columns: list[str], config: RunConfig) -> dict[str, str]:
    rename: dict[str, str] = {}
    by_canonical = {_canonical(column): column for column in columns}

    if config.time_column != "time_s" and config.time_column in columns:
        rename[config.time_column] = "time_s"

    for standard, source in _read_column_map(config.column_map).items():
        if standard not in STANDARD_COLUMNS:
            continue
        if source in columns:
            rename[source] = standard
        elif _canonical(source) in by_canonical:
            rename[by_canonical[_canonical(source)]] = standard

    existing_targets = set(rename.values())
    for standard in STANDARD_COLUMNS:
        if standard in existing_targets or standard in columns:
            continue
        candidates = (standard, *COMMON_ALIASES.get(standard, ()))
        for candidate in candidates:
            source = by_canonical.get(_canonical(candidate))
            if source is not None:
                rename[source] = standard
                break
    return rename


def read_imu_csv(config: RunConfig) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    warnings: list[str] = []
    try:
        raw = pd.read_csv(config.input_path, sep=None, engine="python")
    except OSError as exc:
        raise InputFormatError(f"Failed to read input CSV: {config.input_path}") from exc
    except pd.errors.ParserError as exc:
        raise InputFormatError(f"Failed to parse input CSV: {config.input_path}") from exc

    if raw.empty:
        raise InputFormatError("Input CSV has no rows.")

    input_meta = {
        "path": str(config.input_path),
        "rows": int(len(raw)),
        "columns": [str(column) for column in raw.columns],
    }

    df = raw.rename(columns=_build_rename_map(list(raw.columns), config)).copy()

    if "time_s" not in df.columns:
        if config.sample_rate_hz is None:
            raise InputFormatError(
                "time_s column is missing. Provide --sample-rate-hz or --column-map."
            )
        if config.sample_rate_hz <= 0:
            raise InputFormatError("--sample-rate-hz must be positive.")
        df["time_s"] = np.arange(len(df), dtype=float) / config.sample_rate_hz
        warnings.append("time_s was generated from --sample-rate-hz.")

    df["time_s"], time_warnings = _normalize_time_column(df["time_s"])
    warnings.extend(time_warnings)

    for column in [*ACC_COLUMNS, *GYRO_COLUMNS, *MAG_COLUMNS, "roll", "pitch", "yaw", "qw", "qx", "qy", "qz"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    missing_ratio = float(df[[c for c in STANDARD_COLUMNS if c in df.columns]].isna().mean().max())
    if missing_ratio > 0.05:
        warnings.append(f"Input contains missing values above 5% ({missing_ratio:.1%}).")

    if config.acc_unit == "g":
        for column in ACC_COLUMNS:
            if column in df.columns:
                df[column] = df[column] * config.gravity
    elif config.acc_unit != "mps2":
        raise InputFormatError("--acc-unit must be g or mps2.")

    if config.gyro_unit == "dps":
        for column in GYRO_COLUMNS:
            if column in df.columns:
                df[column] = np.deg2rad(df[column])
    elif config.gyro_unit != "radps":
        raise InputFormatError("--gyro-unit must be dps or radps.")

    df.attrs["missing_ratio"] = missing_ratio
    return df, input_meta, warnings


def _normalize_time_column(series: pd.Series) -> tuple[pd.Series, list[str]]:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.astype(float), []

    timestamps = pd.to_datetime(series, errors="coerce")
    if timestamps.notna().all():
        start = timestamps.iloc[0]
        seconds = (timestamps - start).dt.total_seconds().astype(float)
        return seconds, ["time_s was generated from datetime timestamps in the time column."]

    return numeric.astype(float), []
