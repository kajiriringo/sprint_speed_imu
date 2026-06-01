"""Column schema helpers."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
import numpy as np

from .errors import InputFormatError, MissingColumnError

STANDARD_COLUMNS = [
    "time_s",
    "ax",
    "ay",
    "az",
    "gx",
    "gy",
    "gz",
    "roll",
    "pitch",
    "yaw",
    "qw",
    "qx",
    "qy",
    "qz",
]

ACC_COLUMNS = ["ax", "ay", "az"]
GYRO_COLUMNS = ["gx", "gy", "gz"]
EULER_COLUMNS = ["roll", "pitch", "yaw"]
QUATERNION_COLUMNS = ["qw", "qx", "qy", "qz"]


def missing_columns(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [column for column in columns if column not in df.columns]


def finite_invalid_columns(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    invalid: list[str] = []
    for column in columns:
        if column not in df.columns:
            invalid.append(column)
            continue
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        if len(values) == 0 or not np.all(np.isfinite(values)):
            invalid.append(column)
    return invalid


def valid_quaternion_columns(df: pd.DataFrame) -> bool:
    if missing_columns(df, QUATERNION_COLUMNS):
        return False
    if finite_invalid_columns(df, QUATERNION_COLUMNS):
        return False
    quat = df[QUATERNION_COLUMNS].to_numpy(dtype=float)
    return bool(np.all(np.linalg.norm(quat, axis=1) > 1e-9))


def valid_euler_columns(df: pd.DataFrame) -> bool:
    return not missing_columns(df, EULER_COLUMNS) and not finite_invalid_columns(df, EULER_COLUMNS)


def validate_required_columns(df: pd.DataFrame, method: str) -> None:
    base_missing = missing_columns(df, ["time_s", *ACC_COLUMNS])
    if base_missing:
        raise MissingColumnError(
            "Missing required columns: " + ", ".join(base_missing)
        )
    base_invalid = finite_invalid_columns(df, ["time_s", *ACC_COLUMNS])
    if base_invalid:
        raise InputFormatError(
            "Required columns contain missing or non-numeric values after interpolation: "
            + ", ".join(base_invalid)
        )

    if method == "pca":
        return

    if method == "attitude":
        has_euler_columns = not missing_columns(df, EULER_COLUMNS)
        has_quaternion_columns = not missing_columns(df, QUATERNION_COLUMNS)
        if not (has_euler_columns or has_quaternion_columns):
            raise MissingColumnError(
                "method=attitude requires roll,pitch,yaw or quaternion columns."
            )
        if not (valid_euler_columns(df) or valid_quaternion_columns(df)):
            raise InputFormatError(
                "method=attitude requires finite roll,pitch,yaw or valid non-zero quaternion values."
            )
        return

    raise MissingColumnError(f"Unknown method: {method}")
